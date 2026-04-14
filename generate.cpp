#include "nis_engine.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace rra::nis_engine;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: generate <model.bin> <prompt> [length]\n";
        return 1;
    }

    NISEngine engine;
    if (!engine.load_checkpoint(argv[1])) {
        std::cerr << "Failed to load model: " << argv[1] << "\n";
        return 1;
    }

    std::string prompt = argv[2];
    int length = (argc > 3) ? std::stoi(argv[3]) : 100;

    std::cout << "--- GENERATING ---\n";
    std::cout << prompt;

    // Convert prompt to input events
    for (char c : prompt) {
        InputEvent ev;
        ev.x = static_cast<uint32_t>(static_cast<uint8_t>(c));
        ev.current = 1.0f;
        std::vector<InputEvent> evs = {ev};
        engine.forward_input(evs);
        engine.execute_tick(TickMode::Standard);
    }

    for (int i = 0; i < length; ++i) {
        engine.execute_tick(TickMode::Cognitive);
        EngineOutput out = engine.read_output();
        char c = static_cast<char>(out.byte);
        std::cout << (c >= 32 && c <= 126 ? c : '.');
        
        InputEvent next_ev;
        next_ev.x = static_cast<uint32_t>(out.byte);
        next_ev.current = 1.0f;
        std::vector<InputEvent> next = {next_ev};
        engine.forward_input(next);
        engine.execute_tick(TickMode::Standard);
    }

    std::cout << "\n--- DONE ---\n";
    return 0;
}
