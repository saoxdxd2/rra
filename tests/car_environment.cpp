// ==============================================================================
// car_environment.cpp
// A 2D Sensorimotor Car Environment for the rra Brain Engine.
//
// The car has 5 distance sensor rays. Their readings drive spike rates into
// 5 input neurons. Two output neurons control left/right steering.
// A delayed dopamine reward is injected based on forward progress.
// The brain_tree engine learns purely through biological plasticity!
// ==============================================================================

#include "../nn/brain_tree/brain_tree.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>

using namespace rra::nn::topology;

// ============================================================
// CONSTANTS
// ============================================================
constexpr float PI           = 3.14159265358979f;
constexpr float DEG2RAD      = PI / 180.0f;
constexpr float DT_MS        = 5.0f;   // 5ms physics tick
constexpr float SIM_TICK_MS  = 20.0f;  // 20ms of brain simulation per tick
constexpr float CAR_SPEED    = 0.6f;   // Fixed forward speed (units/tick)
constexpr float STEER_RATE   = 3.5f;   // Degrees per tick per output spike

// Neuron IDs
//   0–4 : Sensor inputs (Left Far, Left, Center, Right, Right Far)
//   5   : Output - Steer Left
//   6   : Output - Steer Right
//   7-14: Hidden interneurons
constexpr int N_NEURONS   = 15;
constexpr int SENSOR_L2   = 0;
constexpr int SENSOR_L1   = 1;
constexpr int SENSOR_C    = 2;
constexpr int SENSOR_R1   = 3;
constexpr int SENSOR_R2   = 4;
constexpr int MOTOR_LEFT  = 5;
constexpr int MOTOR_RIGHT = 6;
constexpr int HIDDEN_BASE = 7;

// ============================================================
// TRACK: A closed rectangular race loop (list of wall segments)
// Each segment = { x1, y1, x2, y2 }
// ============================================================
struct Seg { float x1, y1, x2, y2; };

const std::vector<Seg> OUTER_WALLS = {
    {0, 0, 60, 0},   {60, 0, 60, 30},  {60, 30, 0, 30},  {0, 30, 0, 0}
};
const std::vector<Seg> INNER_WALLS = {
    {12, 8, 48, 8},  {48, 8, 48, 22},  {48, 22, 12, 22}, {12, 22, 12, 8}
};

// ============================================================
// MATH UTILITIES
// ============================================================
float deg2rad(float d) { return d * DEG2RAD; }

// Ray-segment intersection: returns distance t along ray, or INF if no hit
float ray_segment_intersect(float rx, float ry, float rdx, float rdy,
                             float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float denom = rdx * dy - rdy * dx;
    if (std::abs(denom) < 1e-6f) return 1e9f;
    float t = ((x1 - rx) * dy - (y1 - ry) * dx) / denom;
    float u = ((x1 - rx) * rdy - (y1 - ry) * rdx) / denom;
    if (t >= 0.0f && u >= 0.0f && u <= 1.0f) return t;
    return 1e9f;
}

float cast_ray(float rx, float ry, float angle_deg, float max_dist = 20.0f) {
    float rdx = std::cos(deg2rad(angle_deg));
    float rdy = std::sin(deg2rad(angle_deg));
    float closest = max_dist;
    for (const auto& s : OUTER_WALLS) {
        float t = ray_segment_intersect(rx, ry, rdx, rdy, s.x1, s.y1, s.x2, s.y2);
        if (t < closest) closest = t;
    }
    for (const auto& s : INNER_WALLS) {
        float t = ray_segment_intersect(rx, ry, rdx, rdy, s.x1, s.y1, s.x2, s.y2);
        if (t < closest) closest = t;
    }
    return closest;
}

bool point_in_track(float x, float y) {
    // Must be inside outer box and outside inner box
    bool in_outer = (x > 0.5f && x < 59.5f && y > 0.5f && y < 29.5f);
    bool in_inner = (x > 12.5f && x < 47.5f && y > 8.5f && y < 21.5f);
    return in_outer && !in_inner;
}

// ============================================================
// TERMINAL RENDERER
// ============================================================
void render(float cx, float cy, float angle,
            const float sensors[5], int spikes_left, int spikes_right,
            int generation, int tick, float total_reward) {
    // Map track to 62x32 terminal grid
    const int W = 62, H = 32;
    std::vector<std::string> grid(H, std::string(W, ' '));

    // Draw outer walls
    for (int x = 0; x <= 60; x++) { grid[0][x+1] = '#'; grid[31][x+1] = '#'; }
    for (int y = 0; y <= 30; y++) { grid[y+1][0]  = '#'; grid[y+1][61] = '#'; }

    // Draw inner walls
    for (int x = 12; x <= 48; x++) { grid[9][x+1] = '-'; grid[23][x+1] = '-'; }
    for (int y = 8; y <= 22; y++)  { grid[y+1][13] = '|'; grid[y+1][49] = '|'; }

    // Draw car
    int gx = (int)std::clamp(cx, 0.0f, 59.0f) + 1;
    int gy = (int)std::clamp(cy, 0.0f, 29.0f) + 1;
    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
        // Direction character
        char dc = '>';
        float a = std::fmod(angle, 360.0f);
        if (a < 0) a += 360.0f;
        if      (a < 45 || a >= 315) dc = '>';
        else if (a < 135)            dc = 'v';
        else if (a < 225)            dc = '<';
        else                         dc = '^';
        grid[gy][gx] = dc;
    }

    // Draw sensor rays
    for (int i = 0; i < 5; i++) {
        float ray_a = angle + (i - 2) * 30.0f;
        float rdx = std::cos(deg2rad(ray_a));
        float rdy = std::sin(deg2rad(ray_a));
        float d = sensors[i];
        int ex = gx + (int)(rdx * std::min(d, 10.0f) * 0.5f);
        int ey = gy + (int)(rdy * std::min(d, 10.0f) * 0.5f);
        ex = std::clamp(ex, 0, W - 1);
        ey = std::clamp(ey, 0, H - 1);
        if (ey != gy || ex != gx) grid[ey][ex] = (d < 4.0f ? '!' : '.');
    }

    // Print
    std::cout << "\033[H"; // ANSI: move cursor home
    std::cout << "+-- rra Neural Car  Gen:" << std::setw(4) << generation
              << "  Tick:" << std::setw(5) << tick
              << "  Reward:" << std::fixed << std::setprecision(1) << total_reward << " --+\n";
    for (const auto& row : grid) std::cout << row << "\n";

    std::cout << "\n  Sensors  [L2:" << std::fixed << std::setprecision(1) << sensors[0]
              << " L1:" << sensors[1]
              << " C:" << sensors[2]
              << " R1:" << sensors[3]
              << " R2:" << sensors[4] << "]\n";
    std::cout << "  Motor Spikes  [LEFT:" << spikes_left << "  RIGHT:" << spikes_right << "]\n";
}

// ============================================================
// BRAIN ASSEMBLY
// ============================================================
CorticalTissue build_brain() {
    CorticalTissue tissue;

    // Add all neurons
    for (int i = 0; i < N_NEURONS; i++) {
        tissue.add_neuron((uint64_t)i);
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> delay_dist(1.0f, 8.0f);

    // Connect each sensor to every hidden neuron (feedforward)
    for (int s = 0; s < 7; s++) {
        for (int h = HIDDEN_BASE; h < N_NEURONS; h++) {
            tissue.connect_neurons((uint64_t)s, (uint64_t)h, delay_dist(rng));
        }
    }

    // Connect each hidden neuron to both motor outputs
    for (int h = HIDDEN_BASE; h < N_NEURONS; h++) {
        tissue.connect_neurons((uint64_t)h, MOTOR_LEFT,  delay_dist(rng));
        tissue.connect_neurons((uint64_t)h, MOTOR_RIGHT, delay_dist(rng));
    }

    // Lateral inhibition: motor neurons inhibit each other (via a sign-flipped weight in biology)
    tissue.connect_neurons(MOTOR_LEFT,  MOTOR_RIGHT, 1.0f);
    tissue.connect_neurons(MOTOR_RIGHT, MOTOR_LEFT,  1.0f);

    return tissue;
}

// ============================================================
// SENSORIMOTOR LOOP
// ============================================================
int main() {
    std::cout << "\033[?25l"; // Hide cursor
    std::cout << "\033[2J";   // Clear screen

    // Car starting position on the bottom corridor
    float car_x = 6.0f, car_y = 15.0f, car_angle = 0.0f;
    float total_reward = 0.0f;
    int total_ticks = 0;
    int crashes = 0;

    const int MAX_GENERATIONS = 200;
    const int MAX_TICKS_PER_GEN = 400;

    for (int gen = 0; gen < MAX_GENERATIONS; gen++) {
        // Reset car position
        car_x = 6.0f; car_y = 15.0f; car_angle = 0.0f;
        
        auto tissue = build_brain();
        float gen_time = 0.0f;
        float gen_reward = 0.0f;

        for (int tick = 0; tick < MAX_TICKS_PER_GEN; tick++, total_ticks++) {
            // -------------------------------------------------
            // 1. SENSE
            // -------------------------------------------------
            float sensors[5];
            for (int i = 0; i < 5; i++) {
                float ray_angle = car_angle + (i - 2) * 30.0f;
                sensors[i] = cast_ray(car_x, car_y, ray_angle);
            }

            // -------------------------------------------------
            // 2. ENCODE Sensor → Spike Rate
            //    Close obstacles = HIGH spike rate
            // -------------------------------------------------
            float max_sensor_dist = 18.0f;
            for (int s = 0; s < 5; s++) {
                float proximity = 1.0f - (sensors[s] / max_sensor_dist);
                proximity = std::clamp(proximity, 0.0f, 1.0f);
                // Fire multiple spikes proportional to proximity, spread across the 20ms window
                int num_spikes = (int)(proximity * 4.0f);
                for (int k = 0; k < num_spikes; k++) {
                    tissue.force_spike((uint64_t)s, gen_time + k * (SIM_TICK_MS / (num_spikes + 1)));
                }
            }

            // -------------------------------------------------
            // 3. THINK (20ms of asynchronous spike propagation)
            // -------------------------------------------------
            int spikes_before_l = tissue.get_neuron_spikes(MOTOR_LEFT);
            int spikes_before_r = tissue.get_neuron_spikes(MOTOR_RIGHT);

            tissue.run_until(gen_time + SIM_TICK_MS);
            gen_time += SIM_TICK_MS;

            int spikes_left  = tissue.get_neuron_spikes(MOTOR_LEFT)  - spikes_before_l;
            int spikes_right = tissue.get_neuron_spikes(MOTOR_RIGHT) - spikes_before_r;

            // -------------------------------------------------
            // 4. ACT (Neural Spikes → Steering)
            // -------------------------------------------------
            float steer = (float)(spikes_right - spikes_left) * STEER_RATE;
            car_angle += steer;

            float move_x = std::cos(deg2rad(car_angle)) * CAR_SPEED;
            float move_y = std::sin(deg2rad(car_angle)) * CAR_SPEED;
            car_x += move_x;
            car_y += move_y;

            // -------------------------------------------------
            // 5. REWARD (Delayed Dopamine)
            // -------------------------------------------------
            float reward = 0.0f;
            if (!point_in_track(car_x, car_y)) {
                // Crash
                reward = -2.0f;
                tissue.inject_dopamine(reward, gen_time);
                gen_reward += reward;
                crashes++;

                // Reset to track start
                car_x = 6.0f; car_y = 15.0f; car_angle = 0.0f;
            } else {
                // Forward progress reward (center sensor clearance is good)
                reward = 0.05f + (sensors[2] / max_sensor_dist) * 0.1f;
                if (tick % 5 == 0) {
                    // Only inject dopamine every 5 ticks (100ms delayed reward)
                    tissue.inject_dopamine(reward, gen_time);
                }
                gen_reward += reward;
            }
            total_reward += reward;

            // -------------------------------------------------
            // 6. RENDER
            // -------------------------------------------------
            if (tick % 3 == 0) {
                render(car_x, car_y, car_angle, sensors, spikes_left, spikes_right,
                       gen, total_ticks, total_reward);
                std::this_thread::sleep_for(std::chrono::milliseconds(40));
            }
        }

        std::cout << "\n  Gen " << gen
                  << "  reward=" << std::fixed << std::setprecision(2) << gen_reward
                  << "  crashes=" << crashes << "\n";
    }

    std::cout << "\033[?25h"; // Restore cursor
    std::cout << "\n\n  Final Network Spikes: " << "\n";
    return 0;
}
