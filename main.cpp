#include <iostream>
#include "kristforge.h"
#include <cmath>
#include <future>
#include <chrono>
#include <string>
#include <tclap/CmdLine.h>
#include <random>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/cURLpp.hpp>
#include <json/json.h>
#include <uWS/uWS.h>

class KristAddressConstraint : public TCLAP::Constraint<std::string> {
public:
	std::string description() const override { return "krist address"; }

	std::string shortID() const override { return "krist address"; }

	bool check(const std::string &value) const override { return value.size() == 10; }
};

void printDeviceList() {
	const char *fmtString = "%-30.30s | %-15.15s | %-7.7s\n";
	printf(fmtString, "Device", "ID", "Score");
	std::vector devs = kristforge::getAllDevices();

	for (const cl::Device &d : devs) {
		auto devName = d.getInfo<CL_DEVICE_NAME>();
		auto id = kristforge::getDeviceID(d);
		auto score = kristforge::scoreDevice(d);

		printf(fmtString, devName.data(), id.value_or("(n/a)").data(), std::to_string(score).data());
	}
}

std::array<char, 2> generatePrefix() {
	static const std::string prefixChars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	static std::random_device rd;
	static std::mt19937 rng(rd());
	static std::uniform_int_distribution<unsigned long> dist(0, prefixChars.size());

	std::array<char, 2> prefix = {prefixChars[dist(rng)], prefixChars[dist(rng)]};
	return prefix;
};

enum ErrorCodes {
	OK = 0,
	INVALID_ARGS = 1,
	OPENCL_ERROR = 2,
	INTERNAL_ERROR = 3,
	NETWORK_ERROR = 4
};

std::string negotiateWebsocketConnection(std::string nodeURL, bool verbose = false) {
	curlpp::Cleanup cleaner;
	curlpp::Easy req;

	req.setOpt(new curlpp::options::Url(nodeURL));
	req.setOpt(new curlpp::options::Verbose(verbose));
	req.setOpt(new curlpp::options::Post(true));

	std::stringstream stream;
	stream << req;

	Json::Value root(stream.str());
	stream >> root;

	if (root["ok"].asBool()) {
		return root["url"].asString();
	} else {
		throw kristforge::Error("Websocket negotiation failed: " + root["error"].asString());
	}
}

int main(int argc, const char **argv) {
	try {
		TCLAP::CmdLine cmd("Mine krist using compatible OpenCL devices");

		// @formatter:off
		TCLAP::UnlabeledValueArg<std::string> addressArg("address", "The krist address to mine for", false, "k5ztameslf", new KristAddressConstraint, cmd);
		TCLAP::SwitchArg allDevicesArg("a", "all-devices", "Specifies that all compatible devices should be used to mine", cmd);
		TCLAP::MultiArg<std::string> devicesArg("d", "device", "Specifies that the given device should be used to mine", false, "device id", cmd);
		TCLAP::SwitchArg listDevicesArg("l", "list-devices", "Display a list of compatible devices and their IDs", cmd);
		TCLAP::ValueArg<std::string> nodeArg("n", "node", "Specifies which krist node to connect to", false, "https://krist.ceriat.net/ws/start", "krist node url", cmd);
		TCLAP::SwitchArg verboseArg("v", "verbose", "Enables extra logging", cmd);
		// @formatter:on

		cmd.parse(argc, argv);

		if (listDevicesArg.isSet()) {
			printDeviceList();
			return ErrorCodes::OK;
		}

		std::vector<kristforge::Miner> miners;
		if (allDevicesArg.isSet()) {
			if (devicesArg.isSet()) {
				std::cerr << "-a cannot be used with -d" << std::endl;
				return ErrorCodes::INVALID_ARGS;
			}

			for (const cl::Device &dev : kristforge::getAllDevices()) {
				miners.emplace_back(dev, generatePrefix(), pow(2, 24));
			}
		} else {
			for (const std::string &id : devicesArg) {
				auto dev = kristforge::getDeviceByID(id);

				if (!dev) {
					std::cerr << "Unknown device ID: " << id << std::endl;
					return ErrorCodes::INVALID_ARGS;
				}

				miners.emplace_back(*dev, generatePrefix());
			}
		}

		if (miners.empty()) {
			std::cerr << "No devices specified" << std::endl;
			return ErrorCodes::INVALID_ARGS;
		}

		std::cout << "Using miners:" << std::endl;
		for (const kristforge::Miner &m : miners) {
			std::cout << m << std::endl;
		}

		// used for websocket conn later
		uWS::Hub hub;

		// init shared mining state
		std::shared_ptr<kristforge::MiningState> state(new kristforge::MiningState(
				kristforge::mkAddress(addressArg.getValue()),
				[&hub](const std::string &solution, const kristforge::Miner &miner) {
					std::cout << "Solution " << solution << " found by " << miner << std::endl;

					Json::Value root;
					root["type"] = "submit_block";
					root["address"] = solution.substr(0, 10);
					root["nonce"] = solution.substr(22, 12);

					static long id = 1;
					root["id"] = id++;

					static std::unique_ptr<Json::StreamWriter> writer(Json::StreamWriterBuilder().newStreamWriter());
					std::ostringstream ss;
					writer->write(root, &ss);

					std::cout << "Sending " << ss.str() << std::endl;
					((uWS::Group<false>)hub).broadcast(ss.str().data(), ss.str().size(), uWS::OpCode::TEXT);

					return true;
				}));

		// setup websocket callbacks
		hub.onConnection([](uWS::WebSocket<false> *ws, uWS::HttpRequest req) {
			std::cout << "Connected!" << std::endl;
		});

		hub.onDisconnection([&](uWS::WebSocket<false> *ws, int code, char *msg, size_t length) {
			std::cout << "Disconnected - attempting to reconnect" << std::endl;
			state->removeBlock();
			hub.connect(negotiateWebsocketConnection(nodeArg.getValue(), verboseArg.getValue()));
		});

		//192.99.175.37: {"type":"event","event":"block","block":{"height":471070,"address":"kmqc25jc9z","hash":"000000006ef7e42bd78a362936d105ac91d2c776c11b3daa92fc337bc7856a30","short_hash":"000000006ef7","value":1,"time":"2018-04-07T02:11:23.750Z","difficulty":50911},"new_work":49866}
		hub.onMessage([&](uWS::WebSocket<false> *ws, char *msg, size_t length, uWS::OpCode opCode) {
			if (verboseArg.isSet()) {
				std::cout << ws->getAddress().address << ": " << std::string(msg, length) << std::endl;
			}

			Json::Value root;
			std::stringstream(std::string(msg, length)) >> root;

			std::string type = root["type"].asString();

			if (type == "event") {
				std::string evt = root["event"].asString();

				if (evt == "block") {
					state->setBlock(root["new_work"].asInt64(), root["block"]["short_hash"].asString());
					std::cout << "Latest block: " << state->getBlock() << std::endl;
				}
			} else if (type == "hello") {
				state->setBlock(root["work"].asInt64(), root["last_block"]["short_hash"].asString());
				std::cout << "Latest block: " << state->getBlock() << std::endl;
			}
		});

		std::vector<std::thread> threads;

		for (const kristforge::Miner &m : miners) {
			threads.emplace_back([&]{ m.mine(state); });
		}

		hub.connect(negotiateWebsocketConnection(nodeArg.getValue(), verboseArg.getValue()));
		hub.getLoop()->run();

		for (std::thread &t : threads) {
			t.join();
		}

	} catch (TCLAP::ArgException &e) {
		std::cerr << "Error " << e.error() << " for arg " << e.argId() << std::endl;
		return ErrorCodes::INVALID_ARGS;
	} catch (cl::Error &e) {
		std::cerr << "OpenCL error: " << e.what() << " (code " << e.err() << ")" << std::endl;
		return ErrorCodes::OPENCL_ERROR;
	} catch (kristforge::Error &e) {
		std::cerr << "Internal error: " << e.what() << std::endl;
		return ErrorCodes::INTERNAL_ERROR;
	} catch (curlpp::LogicError &e) {
		std::cerr << "Network error: " << e.what() << std::endl;
		return ErrorCodes::NETWORK_ERROR;
	} catch (curlpp::RuntimeError &e) {
		std::cerr << "Network error: " << e.what() << std::endl;
		return ErrorCodes::NETWORK_ERROR;
	} catch (Json::LogicError &e) {
		std::cerr << "JSON error: " << e.what() << std::endl;
		return ErrorCodes::INTERNAL_ERROR;
	}

	return ErrorCodes::OK;
}