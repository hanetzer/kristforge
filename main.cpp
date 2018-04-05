#include <iostream>
#include "kristforge.h"
#include <cmath>
#include <future>
#include <chrono>

int main() {
	std::optional<cl::Device> best = kristforge::getBestDevice();

	if (!best) {
		std::cerr << "No compatible OpenCL devices available." << std::endl;
		return 1;
	}

	kristforge::Miner miner(*best, "aa", (int) pow(2, 24));
	std::cout << "Using: " << miner << std::endl;
	miner.runTests();
	std::cout << "Tests passed" << std::endl;
//
//	std::shared_ptr<kristforge::MiningState> state(new kristforge::MiningState());
//	auto f = std::async(std::launch::async, [&]{ miner.mine("k5ztameslf", "000000000cad", 77120, state); });
//
//	f.wait_for(std::chrono::seconds(30));
//	state->stop();
//
//	std::cout << "Solution: " << (state->solutionFound() ? *state->getSolution() : "Unsolved") << std::endl;
}