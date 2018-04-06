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
				miners.emplace_back(dev, generatePrefix());
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

		// start network stuff
		std::string wsURL;
		{
			curlpp::Cleanup cleaner;
			curlpp::Easy req;

			req.setOpt(new curlpp::options::Url(nodeArg.getValue()));
			req.setOpt(new curlpp::options::Verbose(verboseArg.isSet()));
			req.setOpt(new curlpp::options::Post(true));

			std::stringstream stream;
			stream << req;

			Json::Value root(stream.str());
			stream >> root;

			if (root["ok"].asBool()) {
				wsURL = root["url"].asString();
			} else {
				std::cerr << "Websocket negotiation refused: " << root["error"].asString() << std::endl;
				return ErrorCodes::NETWORK_ERROR;
			}
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