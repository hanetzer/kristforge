#include <iostream>
#include "kristforge.h"
#include <cmath>

int main() {
	std::optional<cl::Device> best = kristforge::getBestDevice();

	if (!best) {
		std::cerr << "No compatible OpenCL devices available." << std::endl;
		return 1;
	}

	kristforge::Miner miner(*best, "aa", (int)pow(2, 24));
	std::cout << "Using: " << miner << std::endl;
	miner.runTests();
	std::cout << "Tests passed" << std::endl;

	std::shared_ptr<kristforge::MiningState> state(new kristforge::MiningState());
	miner.startMining("k5ztameslf", "000000000cad", 7712, state);

	std::cout << "solution: " << *(state->getSolution()) << std::endl;
}