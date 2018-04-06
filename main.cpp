#include <iostream>
#include "kristforge.h"
#include <cmath>
#include <future>
#include <chrono>
#include <string>
#include <tclap/CmdLine.h>

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

int main(int argc, const char **argv) {
	try {
		TCLAP::CmdLine cmd("Mine krist using compatible OpenCL devices");

		// @formatter:off
		TCLAP::UnlabeledValueArg<std::string> addressArg("address", "The krist address to mine for", false, "k5ztameslf", new KristAddressConstraint, cmd);
		TCLAP::MultiArg<std::string> devicesArg("d", "device", "Specifies that the given device should be used to mine", false, "device id", cmd);
		TCLAP::SwitchArg listDevicesArg("l", "list-devices", "Display a list of compatible devices and their IDs", cmd);
		// @formatter:on

		cmd.parse(argc, argv);

		if (listDevicesArg.isSet()) {
			printDeviceList();
			return 0;
		}

	} catch (TCLAP::ArgException &e) {
		std::cerr << "Error " << e.error() << " for arg " << e.argId() << std::endl;
	}
}