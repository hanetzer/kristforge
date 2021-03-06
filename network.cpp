#include "network.h"

#include <sstream>
#include <future>
#include <chrono>
#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <json/json.h>
#include <uWS/uWS.h>

std::string requestWebsocketURI(const std::string &url, bool verbose) {
	curlpp::Cleanup cleanup;
	curlpp::Easy req;

	req.setOpt(new curlpp::options::Url(url));
	req.setOpt(new curlpp::options::Post(true));
	req.setOpt(new curlpp::options::Verbose(verbose));

	std::stringstream stream;
	stream << req;

	Json::Value root;
	stream >> root;

	if (root["ok"].asBool()) {
		return root["url"].asString();
	} else {
		throw std::runtime_error(root["error"].isString() ? root["error"].asString() : "unknown error");
	}
}

class SubmitState {
public:
	SubmitState() = default;

	SubmitState(const SubmitState &) = delete;

	SubmitState &operator=(const SubmitState &) = delete;

	/** Set the solution, blocking until the previous one has been processed */
	void setSolution(kristforge::Solution s) {
		std::unique_lock lock(mtx);
		if (solution) cv.wait(lock, [&] { return !solution; });
		solution = s;
	};

	/** Gets the current solution */
	std::optional<kristforge::Solution> getSolutionImmediately() {
		std::lock_guard lock(mtx);
		return solution;
	}

	/** Removes the solution and increments ID, allowing a new solution to be set */
	void removeSolution() {
		std::lock_guard lock(mtx);
		solution.reset();
		id++;
		cv.notify_all();
	}

	/** ID of current submission */
	long getID() {
		std::lock_guard lock(mtx);
		return id;
	}

private:
	std::mutex mtx;
	std::condition_variable cv;
	std::optional<kristforge::Solution> solution;
	long id = 1;
};

void kristforge::network::run(const std::string &node, const std::shared_ptr<kristforge::State> &state, Options opts) {
	using namespace uWS;

	Hub hub;
	auto *const hubClient = dynamic_cast<Group<false> *>(&hub);

	// used to synchronize submission state
	SubmitState submit;

	hub.onConnection([&](WebSocket<false> *ws, const HttpRequest &req) {
		if (opts.onConnect) (*opts.onConnect)();
	});

	hub.onDisconnection([&](WebSocket<false> *ws, int code, char *msg, size_t length) {
		state->unsetTarget();
		submit.removeSolution();
		if (opts.onDisconnect) (*opts.onDisconnect)(opts.autoReconnect);
		if (opts.autoReconnect) hub.connect(requestWebsocketURI(node, opts.verbose));
	});

	hub.onMessage([&](WebSocket<false> *ws, char *msg, size_t length, OpCode op) {
		if (opts.verbose) std::cout << std::string(msg, length) << std::endl;

		Json::Value root;
		std::istringstream(std::string(msg, length)) >> root;

		if (root["id"].isNumeric() && root["id"].asInt64() == submit.getID()) {
			// block submission reply - contains mining info
			if (root["ok"].asBool()) {
				if (opts.onSolved) (*opts.onSolved)(*submit.getSolutionImmediately(), root["block"]["height"].asInt64());
				state->setTarget(kristforge::Target(root["block"]["short_hash"].asString(), root["work"].asInt64()));
			} else {
				if (opts.onRejected) (*opts.onRejected)(*submit.getSolutionImmediately(), root["error"].asString());
			}

			submit.removeSolution();
		} else if (root["type"] == "hello") {
			// hello packet - sent on first connect, contains mining info
			state->setTarget(kristforge::Target(root["last_block"]["short_hash"].asString(), root["work"].asInt64()));
		} else if (root["type"] == "event" && root["event"] == "block") {
			// block event - sent when any block is mined, contains mining info
			state->setTarget(kristforge::Target(root["block"]["short_hash"].asString(), root["new_work"].asInt64()));
		}
	});

	// register solution callback using an Async so that it's called on this thread
	std::function<void(uS::Async *)> onSolution = [&](uS::Async *a) {
		std::optional<Solution> solution = submit.getSolutionImmediately();

		if (!solution) return;

		static Json::StreamWriter *writer = Json::StreamWriterBuilder().newStreamWriter();

		Json::Value root;
		root["type"] = "submit_block";
		root["id"] = submit.getID();
		root["address"] = solution->address;
		root["nonce"] = solution->nonce;

		std::ostringstream ss;
		writer->write(root, &ss);

		hubClient->broadcast(ss.str().data(), ss.str().size(), TEXT);

		if (opts.onSubmitted) (*opts.onSubmitted)(*solution);
	};

	uS::Async solutionAsync(hub.getLoop());
	solutionAsync.setData(&onSolution);
	solutionAsync.start([](uS::Async *a) { (*reinterpret_cast<std::function<void(uS::Async *)> *>(a->getData()))(a); });

	// start a new thread that triggers the Async
	std::thread solutionChecker([&] {
		while (!state->isStopped()) {
			submit.setSolution(state->popSolution());
			solutionAsync.send();
		}
	});

	hub.connect(requestWebsocketURI(node, opts.verbose));
	hub.run();
	solutionChecker.join();
}
