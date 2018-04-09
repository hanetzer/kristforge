#pragma once

#include "kristforge.h"
#include <memory>
#include <uWS/uWS.h>
#include <future>

namespace krist {
	class MiningComms {
	public:
		explicit MiningComms(std::string node, std::shared_ptr<kristforge::MiningState> state, bool verbose = false);

		void run();

		/**
		 * Submits a solution for the current block
		 * @param solution The complete solution
		 * @return A promise resolving to whether the solution caused a block change (i.e. whether the solution was accepted)
		 */
		std::shared_ptr<std::promise<bool>> submitSolution(const std::string &solution);

	private:
		const bool verbose;
		const std::string node;
		const std::shared_ptr<kristforge::MiningState> state;
		std::map<long, std::shared_ptr<std::promise<bool>>> waitingSubmissions;

		uWS::Hub hub;

		void connect();
	};
}