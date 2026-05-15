#include "../nn/brain_tree/brain_tree.hpp"
#include <iostream>
#include <chrono>

using namespace rra::nn::topology;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "EVENT-DRIVEN BRAIN TREE PERFORMANCE TEST" << std::endl;
    std::cout << "========================================" << std::endl;

    CorticalTissue cortex;

    // Build a small cluster of 1000 biological neurons
    const int num_neurons = 1000;
    for (int i = 0; i < num_neurons; ++i) {
        cortex.add_neuron(i);
    }

    // Connect them sequentially
    for (int i = 0; i < num_neurons - 1; ++i) {
        cortex.connect_neurons(i, i + 1);
    }

    // We want to simulate 1.0 second (1000 ms) of biological thought.
    float target_biological_time_ms = 1000.0f; 

    std::cout << "Simulating " << num_neurons << " neurons for " << target_biological_time_ms << " ms of biological time." << std::endl;
    std::cout << "Integration Model: Asynchronous Event-Driven (No dt)" << std::endl;
    std::cout << "Running..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Inject spikes into the first neuron periodically to start thought waves
    for (int t = 0; t < 1000; t += 10) {
        cortex.force_spike(0, static_cast<float>(t)); 
    }
    
    // Calculate exact analytical time warps up to target time
    cortex.run_until(target_biological_time_ms);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "========================================" << std::endl;
    std::cout << "RESULT:" << std::endl;
    std::cout << "Biological Time Simulated: 1.0 seconds" << std::endl;
    std::cout << "Real-World CPU Time Taken: " << diff.count() << " seconds" << std::endl;
    std::cout << "Speed Ratio: " << (1.0 / diff.count()) << "x real-time" << std::endl;

    return 0;
}
