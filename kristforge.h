#pragma once

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <stdexcept>
#include <optional>
#include <memory>
#include <thread>
#include <utility>
#include <functional>

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

	class Miner;

	/**
	 * Represents a mining state. Can (and should) be used by multiple miners at once.
	 */
	class MiningState {
	public:
		MiningState() = default;

		MiningState(const MiningState &) = delete;

		MiningState &operator=(const MiningState &) = delete;

		/**
		 * Gets the total number of hashes that have been checked
		 */
		inline long getTotalHashes() const { return totalHashes; }

		/**
		 * Whether mining is stopped
		 */
		inline bool isStopped() const { return stopFlag; }

		/**
		 * Signals that mining should be stopped
		 */
		inline bool stop() { stopFlag = true; }

		/**
		 * Whether a solution has been found
		 */
		inline bool solutionFound() const { return solved; }

		/**
		 * Gets the solution
		 */
		inline std::optional<std::string> getSolution() const { return solution; }

		/**
		 * Resets this state
		 */
		void reset() {
			totalHashes = 0;
			stopFlag = false;
			solved = false;
			solution = {};
		}

	private:
		std::atomic<long> totalHashes = 0;
		std::atomic<bool> stopFlag = false;

		std::atomic<bool> solved = false;
		std::optional<std::string> solution = {};

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
		 * Starts mining using this miner
		 * @param address The address to mine for
		 * @param block The latest block
		 * @param work The work value
		 * @param stateptr A mining state to synchronize mining events
		 */
		void mine(const char *address, const char *block, long work, std::shared_ptr<MiningState> stateptr);

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

	class MinerPool {
	public:
		MinerPool(std::vector<Miner> miners,
		          std::function<void(const std::string&)> solveCallback,
		          std::string block,
		          long work) : miners(std::move(miners)),
                               solveCallback(solveCallback),
		                       state(new MiningState()),
		                       block(std::move(block)),
		                       work(work) {};

		/**
		 * Signals miners to restart for the new block
		 */
		void blockChanged(const std::string &block, long work) {
			this->block = block;
			this->work = work;
		}

		/**
		 * Runs all miners in this pool
		 */
		void run();

	private:
		const std::vector<Miner> miners;
		const std::function<void(const std::string&)> solveCallback;
		const std::shared_ptr<MiningState> state;

		std::string block;
		long work;
	};
}