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
	std::vector<cl::Device> devs;
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (const auto& p: platforms) {
		std::vector<cl::Device> tmpDevs;
		p.getDevices(CL_DEVICE_TYPE_ALL, &tmpDevs);

		for (const auto &d : tmpDevs) {
			devs.push_back(d);
		}
	}

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
		return {};
	}

	return std::optional(best);
}

std::optional<cl::Device> kristforge::getDeviceByID(const std::string &id, const std::vector<cl::Device> &devs) {
	cl::Device matching;

	for (const cl::Device &d : devs) {
		if (getDeviceID(d) == id) {
			matching = d;
		}
	}

	if (!matching()) {
		return {};
	}

	return std::optional(matching);
}

std::string getCompileOpts(const cl::Device &dev) {
	std::string exts = dev.getInfo<CL_DEVICE_EXTENSIONS>();

	std::stringstream opts;

	if (exts.find("cl_amd_media_ops") != std::string::npos) {
		opts << "-D BITALIGN ";
	}

	return opts.str();
}

cl::Program compileMiner(const cl::Context &ctx, const cl::Device &dev, std::optional<std::string> compileOpts = {}) {
	if (!compileOpts) {
		compileOpts = getCompileOpts(dev);
	}

	cl::Program program(ctx, openclSource, false);

	std::vector<cl::Device> devs;
	devs.push_back(dev);

	try {
		program.build(devs, compileOpts->data());
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

template<int N>
std::array<char, N> toArray(const std::string &from) {
	if (from.size() != N) {
		throw std::range_error("Length must equal " + std::to_string(N));
	}

	std::array<char, N> data = {};
	std::copy(from.begin(), from.end(), data.begin());
	return data;
};

kristforge::kristAddress kristforge::mkAddress(const std::string &from) { return toArray<10>(from); }

kristforge::blockShorthash kristforge::mkBlockShorthash(const std::string &from) { return toArray<12>(from); }

void kristforge::MiningState::stop() {
	std::lock_guard<std::mutex> guard(mtx);
	stopped = true;
	blockValid = false;
	cv.notify_all();
}

void kristforge::MiningState::removeBlock() {
	std::lock_guard<std::mutex> guard(mtx);
	blockValid = false;
	cv.notify_all();
}

void kristforge::MiningState::setBlock(long work, blockShorthash prevBlock) {
	std::lock_guard<std::mutex> guard(mtx);
	this->work = work;
	this->prevBlock = prevBlock;
	blockIndex++;
	blockValid = true;
	cv.notify_all();
}

void kristforge::MiningState::solved(const std::string &solution, const Miner &miner) {
	{
		std::lock_guard<std::mutex> guard(mtx);
		blockValid = false;
		cv.notify_all();
	}

	if (!solveCB(solution, miner)) {
		// failed
		std::lock_guard<std::mutex> guard(mtx);
		blockValid = true;
		cv.notify_all();
	} else {
		// success
		numSolved++;
	}
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

long optimalWorksize(const cl::Device &dev) {
	long value = 1;
	std::vector<size_t> sizes = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

	for (const size_t s : sizes) {
		value *= s;
	}

	return value;
}

kristforge::Miner::Miner(const cl::Device &dev, std::array<char, 2> prefix, std::optional<long> worksize) :
		dev(dev),
		ctx(cl::Context(dev)),
		cmd(cl::CommandQueue(ctx, dev)),
		program(compileMiner(ctx, dev)),
		worksize(worksize ? *worksize : optimalWorksize(dev)),
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

void kristforge::Miner::mine(std::shared_ptr<MiningState> state) const {
	cl::Kernel kernel(program, "krist_miner");

	// init buffers
	cl::Buffer addressBuf(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 10);
	cl::Buffer blockBuf(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 12);
	cl::Buffer prefixBuf(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, 2);
	cl::Buffer solutionBuf(ctx, CL_MEM_READ_ONLY, 34);

	// copy constant data to buffers
	cmd.enqueueWriteBuffer(addressBuf, CL_FALSE, 0, 10, state->address.data());
	cmd.enqueueWriteBuffer(prefixBuf, CL_FALSE, 0, 2, prefix.data());

	// set buffer args
	kernel.setArg(0, addressBuf);
	kernel.setArg(1, blockBuf);
	kernel.setArg(2, prefixBuf);
	kernel.setArg(5, solutionBuf);

	while (!(state->stopped)) {
		if (!state->blockValid) {
			std::unique_lock<std::mutex> lock(state->mtx);
			state->cv.wait(lock, [&state] { return state->blockValid.load(); });
		}

//		if (!state->blockValid) continue;

		// we have a valid block, start mining
		const long index = state->blockIndex;

		char solutionOut[34] = {0};

		// set inputs
		cmd.enqueueWriteBuffer(blockBuf, CL_FALSE, 0, 12, state->prevBlock.data());
		cmd.enqueueWriteBuffer(solutionBuf, CL_FALSE, 0, 34, solutionOut);
		kernel.setArg(4, state->work.load());
		cmd.finish();

		for (long offset = 0; state->blockValid && state->blockIndex == index; offset += worksize, state->totalHashes += worksize) {
			// set offset
			kernel.setArg(3, offset);

			// invoke kernel
			cmd.enqueueNDRangeKernel(kernel, 0, worksize);
			cmd.enqueueReadBuffer(solutionBuf, CL_FALSE, 0, 34, solutionOut);
			cmd.finish();

			// check for a valid solution
			if (solutionOut[0] != 0) {
				state->solved(std::string(solutionOut, 34), *this);
				break;
			}
		}
	}
}
