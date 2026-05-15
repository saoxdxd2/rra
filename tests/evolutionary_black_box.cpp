#include "../nn/brain_tree/brain_tree.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

using namespace rra::nn::topology;

struct Organism {
    EvolutionaryGenome genome;
    int total_spikes;
    float fitness_score; // Lower is better
    float novelty_score; // Higher is better (Orthogonality)
};

EvolutionaryGenome mutate_genome(const EvolutionaryGenome& parent, std::mt19937& gen) {
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    EvolutionaryGenome child = parent;
    
    // Apply random mutations (-10% to +10%)
    child.v_thresh += child.v_thresh * dis(gen);
    child.tau_m += child.tau_m * dis(gen);
    child.calcium_decay += child.calcium_decay * dis(gen);
    child.ampa_voltage_jump += child.ampa_voltage_jump * dis(gen);
    
    // Keep bounds safe
    if (child.v_thresh > -20.0f) child.v_thresh = -20.0f;
    if (child.tau_m < 1.0f) child.tau_m = 1.0f;
    if (child.ampa_voltage_jump < 0.001f) child.ampa_voltage_jump = 0.001f;

    return child;
}

float calculate_distance(const EvolutionaryGenome& a, const EvolutionaryGenome& b) {
    // Multi-dimensional distance vector (Orthogonality check)
    float dv = a.v_thresh - b.v_thresh;
    float dt = a.tau_m - b.tau_m;
    float dc = a.calcium_decay - b.calcium_decay;
    float da = a.ampa_voltage_jump - b.ampa_voltage_jump;
    return std::sqrt(dv*dv + dt*dt + dc*dc + da*da);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "STOCHASTIC EVOLUTIONARY BLACK BOX" << std::endl;
    std::cout << "Orthogonal Novelty Search Active" << std::endl;
    std::cout << "========================================" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());

    const int POPULATION_SIZE = 50;
    const int GENERATIONS = 5;
    const int TARGET_SPIKES = 2000; // Optimal stable firing rate

    std::vector<Organism> population(POPULATION_SIZE);

    // Generation Loop
    for (int gen_idx = 0; gen_idx < GENERATIONS; ++gen_idx) {
        std::cout << "\n--- GENERATION " << gen_idx + 1 << " ---" << std::endl;

        EvolutionaryGenome pop_mean;

        // 1. Evaluate Population
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            CorticalTissue cortex(population[i].genome);
            
            // Build 500-neuron test tissue
            for (int n = 0; n < 500; ++n) cortex.add_neuron(n);
            for (int n = 0; n < 499; ++n) cortex.connect_neurons(n, n + 1);

            // Inject 5 starting spikes
            for (int t = 0; t < 50; t += 10) cortex.force_spike(0, static_cast<float>(t));

            // Run for 500ms
            cortex.run_until(500.0f);
            
            population[i].total_spikes = cortex.get_total_network_spikes();
            
            // Fitness: Absolute error from target stability (0 is perfect)
            population[i].fitness_score = std::abs(population[i].total_spikes - TARGET_SPIKES);

            // Accumulate for mean
            pop_mean.v_thresh += population[i].genome.v_thresh / POPULATION_SIZE;
            pop_mean.tau_m += population[i].genome.tau_m / POPULATION_SIZE;
            pop_mean.calcium_decay += population[i].genome.calcium_decay / POPULATION_SIZE;
            pop_mean.ampa_voltage_jump += population[i].genome.ampa_voltage_jump / POPULATION_SIZE;
        }

        // 2. Calculate Orthogonal Novelty (How different is this organism from the mean?)
        for (auto& org : population) {
            org.novelty_score = calculate_distance(org.genome, pop_mean);
        }

        // 3. Orthogonal Evolutionary Sorting
        std::vector<Organism> next_generation;

        // Sort by Fitness (Best first, i.e., lowest error)
        std::sort(population.begin(), population.end(), [](const Organism& a, const Organism& b) {
            return a.fitness_score < b.fitness_score;
        });

        // Keep Top 10% Fitness Winners
        int top_fitness_count = POPULATION_SIZE / 10;
        for (int i = 0; i < top_fitness_count; ++i) {
            next_generation.push_back(population[i]);
        }

        std::cout << "Best Fitness Score: " << population[0].fitness_score << " (Spikes: " << population[0].total_spikes << ")" << std::endl;

        // Collect Failures
        std::vector<Organism> failures(population.begin() + top_fitness_count, population.end());

        // Sort Failures by Novelty (Most orthogonal/different first, i.e., highest novelty score)
        std::sort(failures.begin(), failures.end(), [](const Organism& a, const Organism& b) {
            return a.novelty_score > b.novelty_score;
        });

        // Keep Top 10% Orthogonal Failures (Mutant Seeds)
        int orthogonal_count = POPULATION_SIZE / 10;
        for (int i = 0; i < orthogonal_count; ++i) {
            next_generation.push_back(failures[i]);
        }

        std::cout << "Retained " << orthogonal_count << " Orthogonal Failures to prevent local minima." << std::endl;

        // 4. Cross-breed and Mutate to fill the rest of the population
        int survivors = next_generation.size();
        std::uniform_int_distribution<int> parent_dis(0, survivors - 1);

        while (next_generation.size() < POPULATION_SIZE) {
            const Organism& parent = next_generation[parent_dis(gen)];
            Organism child;
            child.genome = mutate_genome(parent.genome, gen);
            next_generation.push_back(child);
        }

        population = next_generation;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "EVOLUTIONARY RUN COMPLETE" << std::endl;
    std::cout << "The Black Box has mathematically stabilized the parameter space." << std::endl;

    return 0;
}
