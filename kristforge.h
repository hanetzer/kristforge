#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <stdexcept>
#include <optional>
#include <memory>
#include <thread>
#include <utility>
#include <functional>
#include <condition_variable>

namespace kristforge {
	/**
	 * Computes a score for the given device - higher is better
	 */
	long scoreDevice(const cl::Device &dev);

	/**
	 * Checks whether the given OpenCL device can be used for mining krist
	 */
	bool isCompatible(const cl::Device &dev);

	/**
	 * Gets all compatible devices
	 */
	std::vector<cl::Device> getAllDevices();

	/**
	 * Gets the best compatible device
	 */
	std::optional<cl::Device> getBestDevice(const std::vector<cl::Device> &devs = getAllDevices());

	/**
	 * Gets a unique ID for this device, if supported
	 */
	std::optional<std::string> getDeviceID(const cl::Device &dev);

	/**
	 * Represents a kristforge error
	 */
	class Error : public std::exception {
	public:
		explicit Error(std::string msg) : message(std::move(msg)) {}

		const char *what() const noexcept override { return message.data(); }

	private:
		const std::string message;
	};

	typedef std::array<char, 10> kristAddress;
	typedef std::array<char, 12> blockShorthash;

	class Miner;

	/**
	 * Represents a shared mining state, used to synchronize mining and network activity
	 */
	class MiningState {
	public:
		explicit MiningState(kristAddress address, std::function<void(std::string)> solveCB) :
				solveCB(std::move(solveCB)),
				address(address) {};

		MiningState(const MiningState &) = delete;

		MiningState &operator=(const MiningState &) = delete;

		/**
		 * Signals miners to stop and terminate
		 */
		void stop();

		/**
		 * Removes the block, signalling miners to stop and wait for a new one
		 */
		void removeBlock();

		/**
		 * Sets the block, signalling miners to restart for this block
		 * @param work The current work value
		 * @param prevHeight The previous block height
		 * @param prevBlock The previous block short hash
		 */
		void setBlock(long work, blockShorthash prevBlock);

	private:
		/**
		 * The address to mine for
		 */
		const kristAddress address;

		/**
		 * The callback for solved blocks
		 */
		const std::function<void(std::string)> solveCB;

		/**
		 * If set, mining should be completely stopped
		 */
		std::atomic<bool> stopped = false;

		/**
		 * Whether the current block data is valid for mining
		 */
		std::atomic<bool> blockValid = false;

		/**
		 * A value that's incremented every time a new block is set, used for validation
		 */
		std::atomic<long> blockIndex = 0;

		/**
		 * The work value
		 */
		std::atomic<long> work = 0;

		/**
		 * The short hash of the previous block
		 */
		blockShorthash prevBlock = blockShorthash();

		std::mutex mtx;
		std::condition_variable cv;

		friend Miner;
	};

	/**
	 * An OpenCL accelerated krist miner
	 */
	class Miner {
	public:
		/**
		 * Creates a miner using the given OpenCL device
		 */
		explicit Miner(const cl::Device &dev, const char prefix[2], std::optional<long> worksize = {});

		/**
		 * Runs tests to ensure the OpenCL is working properly
		 */
		void runTests() const noexcept(false);

		/**
		 * Gets the OpenCL device used by this miner
		 */
		inline const cl::Device getDevice() const { return dev; }

		/**
		 * Mines using this miner
		 * @param state The shared mining state
		 */
		void mine(std::shared_ptr<MiningState> state);

//		void mine(const char *kristAddress, const char *block, long work, std::shared_ptr<MiningState> stateptr);

	private:
		const cl::Device dev;
		const cl::Context ctx;
		const cl::CommandQueue cmd;
		const cl::Program program;

		const long worksize;
		const std::string prefix;
	};

	/**
	 * Writes a brief description of the miner, including the OpenCL device and platform name
	 */
	std::ostream &operator<<(std::ostream &os, const Miner &m);
}