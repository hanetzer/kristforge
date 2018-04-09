#include "krist.h"
#include <iostream>
#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <sstream>
#include <json/json.h>

krist::MiningComms::MiningComms(std::string node, std::shared_ptr<kristforge::MiningState> state, bool verbose) :
		node(std::move(node)),
		state(std::move(state)),
		verbose(verbose) {}

void krist::MiningComms::run() noexcept(false) {
	using namespace uWS;

	hub.onConnection([&](WebSocket<false> *ws, HttpRequest req) {
		std::cout << "Connected!" << std::endl;
	});

	hub.onDisconnection([&](WebSocket<false> *ws, int code, char *msg, size_t length) {
		// forces miners to wait for a new block, after reconnecting
		state->removeBlock();
		for (auto const& e : waitingSubmissions) {
			e.second->set_value(true);
		}
		waitingSubmissions.clear();

		std::cout << "Disconnected, attempting to reconnect" << std::endl;
		connect();
	});

	hub.onMessage([&](WebSocket<false> *ws, char *msg, size_t length, OpCode op) {
		if (verbose) {
			std::cout << ws->getAddress().address << ": " << std::string(msg, length) << std::endl;
		}

		Json::Value root;
		std::stringstream(std::string(msg, length)) >> root;

		std::string type = root["type"].asString();
		std::optional<std::pair<long, std::string>> nextBlock = {};

		if (type == "event") {
			std::string evt = root["event"].asString();

			if (evt == "block") {
				nextBlock = std::pair(root["new_work"].asInt64(), root["block"]["short_hash"].asString());
			}
		} else if (type == "hello") {
			nextBlock = std::pair(root["work"].asInt64(), root["last_block"]["short_hash"].asString());
		} else {
			if (root["id"].isNumeric()) {
				long id = root["id"].asInt();
				bool success = root["success"].asBool();

				if (waitingSubmissions.count(id) > 0) {
					waitingSubmissions[id]->set_value(success);
					waitingSubmissions.erase(id);
				}

				// is this necessary?
				if (success) {
					nextBlock = std::pair(root["work"].asInt64(), root["block"]["short_hash"].asString());
				}
			}
		}

		if (nextBlock) {
			state->setBlock(nextBlock->first, nextBlock->second);
		}
	});

	connect();
	hub.getLoop()->run();
}

void krist::MiningComms::connect() {
	curlpp::Cleanup cleaner;
	curlpp::Easy req;

	req.setOpt(new curlpp::options::Url(node));
	req.setOpt(new curlpp::options::Post(true));
	req.setOpt(new curlpp::options::Verbose(verbose));

	std::stringstream stream;
	stream << req;

	Json::Value root;
	stream >> root;

	hub.connect(root["url"].asString());
}

inline long nextID() {
	static long id = 0;
	return ++id;
}

std::shared_ptr<std::promise<bool>> krist::MiningComms::submitSolution(const std::string &solution) {
	long id = nextID();

	std::shared_ptr<std::promise<bool>> promise(new std::promise<bool>);
	waitingSubmissions[id] = promise;

	Json::Value root;
	root["type"] = "submit_block";
	root["id"] = id;
	root["address"] = std::string(state->getAddress().data(), 10);
	root["nonce"] = solution;

	static Json::StreamWriter *writer = Json::StreamWriterBuilder().newStreamWriter();

	std::ostringstream ss;
	writer->write(root, &ss);

	if (verbose) {
		std::cout << "Sending " << ss.str() << std::endl;
	}

	static_cast<uWS::Group<false>>(hub).broadcast(ss.str().data(), ss.str().size(), uWS::OpCode::TEXT);

	return promise;
}