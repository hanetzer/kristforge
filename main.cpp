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
#include "krist.h"
#include <cmath>
#include <iomanip>

class KristAddressConstraint : public TCLAP::Constraint<std::string> {
public:
	std::string description() const override { return "krist address"; }

	std::string shortID() const override { return "krist address"; }

	bool check(const std::string &value) const override { return value.size() == 10; }
};

class VectorSizeConstraint : public TCLAP::Constraint<int> {
	std::string description() const override { return "1 | 2 | 4"; }

	std::string shortID() const override { return "1 | 2 | 4"; }

	bool check(const int &i) const override { return i == 1 || i == 2 || i == 4; }
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

std::string formatHashrate(long hashesPerSecond) {
	static const char *suffixes[] = {"h/s", "kh/s", "Mh/s", "Gh/s", "Th/s", "Ph/s", "Eh/s"};

	auto scale = std::max(0, static_cast<int>(0, log(hashesPerSecond) / log(1000)));
	double value = hashesPerSecond / pow(1000, scale);

	std::stringstream out;
	out << std::fixed << std::setprecision(2) << value << " " << suffixes[scale];
	return out.str();
}

int main(int argc, const char **argv) {
	try {
		TCLAP::CmdLine cmd("Mine krist using compatible OpenCL devices");

		// @formatter:off
		TCLAP::UnlabeledValueArg<std::string> addressArg("address", "The krist address to mine for", false, "k5ztameslf", new KristAddressConstraint, cmd);
		TCLAP::SwitchArg allDevicesArg("a", "all-devices", "Specifies that all compatible devices should be used to mine", cmd);
		TCLAP::SwitchArg bestDeviceArg("b", "best-device", "Specifices that the best compatible device should be used to mine", cmd);
		TCLAP::MultiArg<std::string> devicesArg("d", "device", "Specifies that the given device should be used to mine", false, "device id", cmd);
		TCLAP::SwitchArg listDevicesArg("l", "list-devices", "Display a list of compatible devices and their IDs", cmd);
		TCLAP::ValueArg<std::string> nodeArg("n", "node", "Specifies which krist node to connect to", false, "https://krist.ceriat.net/ws/start", "krist node url", cmd);
		TCLAP::SwitchArg verboseArg("v", "verbose", "Enables extra logging", cmd);
		TCLAP::ValueArg<int> vectorSizeArg("V", "vector-size", "Sets the vector size", false, 1, new VectorSizeConstraint, cmd);
		TCLAP::SwitchArg testArg("t", "tests-only", "Don't mine, just run tests", cmd);
		TCLAP::MultiArg<int> deviceNumsArg("N", "device-num", "Use a given device by its position in the device list - generally a bad idea!", false, "number", cmd);
		TCLAP::ValueArg<long> worksizeArg("w", "worksize", "Sets the work group size", false, -1, "number", cmd);
		// @formatter:on

		cmd.parse(argc, argv);

		if (listDevicesArg.isSet()) {
			printDeviceList();
			return ErrorCodes::OK;
		}

		std::vector<cl::Device> allDevices = kristforge::getAllDevices();
		std::vector<cl::Device> selectedDevices;

		if (allDevicesArg.isSet()) {
			for (const cl::Device &dev : allDevices) {
				selectedDevices.push_back(dev);
			}
		}

		if (bestDeviceArg.isSet()) {
			auto best = kristforge::getBestDevice(allDevices);

			if (!best) {
				std::cerr << "No available devices" << std::endl;
				return ErrorCodes::INTERNAL_ERROR;
			}

			selectedDevices.push_back(*best);
		}

		for (const std::string &id : devicesArg) {
			auto dev = kristforge::getDeviceByID(id, allDevices);

			if (!dev) {
				std::cerr << "Unknown device ID: " << id << std::endl;
				return ErrorCodes::INVALID_ARGS;
			}

			selectedDevices.push_back(*dev);
		}

		for (const int i : deviceNumsArg) {
			if (i >= allDevices.size()) {
				std::cerr << "Invalid device number: " << i << std::endl;
				return ErrorCodes::INVALID_ARGS;
			}

			selectedDevices.push_back(allDevices[i]);
		}

		if (selectedDevices.empty()) {
			std::cerr << "No devices specified" << std::endl;
			return ErrorCodes::INVALID_ARGS;
		}

		std::vector<kristforge::Miner> miners;

		std::optional<long> globalWorksize = worksizeArg.isSet() ? worksizeArg.getValue() : std::optional<long>();

		for (const cl::Device &dev : selectedDevices) {
			miners.emplace_back(dev, generatePrefix(), vectorSizeArg.getValue(),
			                    worksizeArg.isSet() ? worksizeArg.getValue() : globalWorksize);
		}

		std::cout << "Running tests:" << std::endl;
		for (const kristforge::Miner &m : miners) {
			std::cout << m << std::endl;
			m.runTests();
		}
		std::cout << "All miners tested successfully" << std::endl;

		if (testArg.isSet()) {
			return ErrorCodes::OK;
		}

		krist::MiningComms *comms;

		// init shared mining state
		std::shared_ptr<kristforge::MiningState> state(new kristforge::MiningState(
				kristforge::mkAddress(addressArg.getValue()),
				[&](const std::string &solution, const kristforge::Miner &miner) {
					std::cout << "Solution " << solution << " found by " << miner << std::endl;
					return comms->submitSolution(solution)->get_future().get();
				}));

		comms = new krist::MiningComms(nodeArg.getValue(), state, verboseArg.getValue());

		std::vector<std::thread> threads;

		for (const kristforge::Miner &m : miners) {
			threads.emplace_back([&] { m.mine(state); });
		}

		std::thread statusThread([state] {
			for (long hashes = 0; true; hashes = state->getTotalHashes()) {
				std::this_thread::sleep_for(std::chrono::seconds(3));
				long diff = state->getTotalHashes() - hashes;

				std::cout << "Speed: " << formatHashrate(diff / 3) << " Solved: " << state->getTotalSolved()
				          << std::endl;
			}
		});

		comms->run();

		for (std::thread &t : threads) {
			t.join();
		}

		statusThread.detach();

		delete comms;

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