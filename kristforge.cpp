#include "kristforge.h"
#include "kristforge_opencl.cpp"
#include "cl_amd.h"
#include <vector>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <iostream>

long kristforge::scoreDevice(const cl::Device &dev) {
	return dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() * dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
}

bool kristforge::isCompatible(const cl::Device &dev) {
	return dev.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU;
}

std::vector<cl::Device> kristforge::getAllDevices() {
	// all devices from default platform
	std::vector<cl::Device> devs;
	cl::Platform::getDefault().getDevices(CL_DEVICE_TYPE_ALL, &devs);

	// remove incompatible devices
	std::remove_if(devs.begin(), devs.end(), [](const cl::Device &d) { return !isCompatible(d); });

	return devs;
}

std::optional<cl::Device> kristforge::getBestDevice(const std::vector<cl::Device> &devs) {
	cl::Device best;

	for (const cl::Device &d : devs) {
		if (!isCompatible(d)) {
			continue;
		}

		if (!best() || scoreDevice(best) < scoreDevice(d)) {
			best = d;
		}
	}

	if (!best()) {
		return std::optional<cl::Device>();
	}

	return std::optional(best);
}

cl::Program compileMiner(const cl::Context &ctx, const cl::Device &dev, const std::string &compileOpts = "") {
	cl::Program program(ctx, openclSource, false);

	std::vector<cl::Device> devs;
	devs.push_back(dev);

	try {
		program.build(devs, compileOpts.data());
	} catch (const cl::Error &e) {
		if (e.err() == CL_BUILD_PROGRAM_FAILURE &&
		    program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev) == CL_BUILD_ERROR) {

			// compilation error - get log and throw error
			std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			throw kristforge::Error("OpenCL Compilation Error:\n" + log);
		} else {
			throw e;
		}
	}

	// compilation successful
	return program;
}

std::optional<std::string> kristforge::getDeviceID(const cl::Device &dev) {
	auto exts = dev.getInfo<CL_DEVICE_EXTENSIONS>();

	if (exts.find("cl_amd_device_attribute_query")) {
		cl_device_topology_amd topo;

		cl_int status = clGetDeviceInfo(dev(), CL_DEVICE_TOPOLOGY_AMD, sizeof(topo), &topo, nullptr);

		if (status == CL_SUCCESS) {
			if (topo.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD) {
				std::stringstream out;
				out << "PCIE:" << std::to_string(topo.pcie.bus)
				    << ":" << std::to_string(topo.pcie.device)
				    << ":" << std::to_string(topo.pcie.function);

				return out.str();
			}
		}
	}

	// TODO: support nvidia

	return std::optional<std::string>();
}

std::ostream &kristforge::operator<<(std::ostream &os, const kristforge::Miner &m) {
	auto dev = m.getDevice();
	auto devName = dev.getInfo<CL_DEVICE_NAME>();
	auto platformName = cl::Platform(dev.getInfo<CL_DEVICE_PLATFORM>()).getInfo<CL_PLATFORM_NAME>();
	auto id = getDeviceID(m.getDevice());

	return os << "Miner ("
	          << devName.data()
	          << (id ? " [" + *id + "]" : "")
	          << " on "
	          << platformName.data()
	          << ")";
}

kristforge::Miner::Miner(const cl::Device &dev, const char prefix[2], const std::optional<long> worksize) :
		dev(dev),
		ctx(cl::Context(dev)),
		cmd(cl::CommandQueue(ctx, dev)),
		program(compileMiner(ctx, dev)),
		worksize(worksize ? *worksize : 1), // todo: actually calculate a work size here
		prefix(prefix) {}

const char hex[] = "0123456789abcdef";

std::string toHex(const unsigned char *data, size_t dataSize) {
	std::string output;

	for (int i = 0; i < dataSize; i++) {
		output += hex[data[i] >> 4];
		output += hex[data[i] & 0x0f];
	}

	return output;
}

void kristforge::Miner::runTests() const noexcept(false) {
	cl::Kernel testDigest55(program, "testDigest55");
	unsigned char inData[64] = "abc", outData[32];

	cl::Buffer input(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_WRITE_ONLY, sizeof(inData));
	cl::Buffer output(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(outData));

	testDigest55.setArg(0, input); // input data
	testDigest55.setArg(1, 3); // input length
	testDigest55.setArg(2, output); // output

	cmd.enqueueWriteBuffer(input, CL_FALSE, 0, sizeof(inData), inData);
	cmd.enqueueTask(testDigest55);
	cmd.enqueueReadBuffer(output, CL_FALSE, 0, sizeof(outData), outData);
	cmd.finish();

	std::string got = toHex(outData, 32);
	if (got != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
		throw kristforge::Error("testDigest55 failed: got " + got);
	}
}

void kristforge::Miner::mine(const char *address,
                             const char *block,
                             long work,
                             std::shared_ptr<MiningState> state) {
	cl::Kernel mine(program, "krist_miner");

	char solution[34] = {0};

	// init buffers
	cl::Buffer addressBuf(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 10);
	cl::Buffer blockBuf(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 12);
	cl::Buffer prefixBuf(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 2);
	cl::Buffer solutionBuf(ctx, CL_MEM_WRITE_ONLY, 34);

	// copy data to buffers
	cmd.enqueueWriteBuffer(addressBuf, CL_FALSE, 0, 10, address);
	cmd.enqueueWriteBuffer(blockBuf, CL_FALSE, 0, 12, block);
	cmd.enqueueWriteBuffer(prefixBuf, CL_FALSE, 0, 2, prefix.data());
	cmd.enqueueWriteBuffer(solutionBuf, CL_FALSE, 0, 34, solution);
	cmd.finish();

	// set constant args
	mine.setArg(0, addressBuf);
	mine.setArg(1, blockBuf);
	mine.setArg(2, prefixBuf);
	mine.setArg(4, work);
	mine.setArg(5, solutionBuf);

	// main mining loop
	long offset;
	for (offset = 0; !(state->stopFlag); offset += worksize, state->totalHashes += worksize) {
		// update offset
		mine.setArg(3, offset);

		// invoke kernel, read result
		cmd.enqueueNDRangeKernel(mine, 0, worksize);
		cmd.enqueueReadBuffer(solutionBuf, CL_FALSE, 0, 34, solution);
		cmd.finish();

		// solved
		if (solution[0] != 0) {
			state->solved = true;
			state->solution = std::string(solution, 34);

			state->stop();
		}
	}
}

void kristforge::MinerPool::run() {

}
