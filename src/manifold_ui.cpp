#include <iostream>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "manifold_ipc.hpp"

using namespace rra::gnf::ipc;

int main() {
    std::cout << "[UI] Starting Zero-Copy 4D Manifold Visualizer...\n";
    
    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_READ, FALSE, IPC_MAP_NAME);
    if (!hMapFile) {
        std::cerr << "[ERROR] Could not open IPC shared memory. Is the engine running?\n";
        return 1;
    }

    void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, sizeof(ManifoldIPCData));
    if (!pBuf) {
        std::cerr << "[ERROR] Could not map view of file.\n";
        CloseHandle(hMapFile);
        return 1;
    }

    auto* ipc_data = static_cast<ManifoldIPCData*>(pBuf);
    
    cv::namedWindow("Geometric Neural Field (4D Manifold Projection)", cv::WINDOW_AUTOSIZE);
    
    const int WIDTH = 800;
    const int HEIGHT = 800;
    cv::Mat frame(HEIGHT, WIDTH, CV_8UC3);

    while (true) {
        frame = cv::Scalar(15, 15, 20); // Dark background
        
        uint64_t counter = ipc_data->frame_counter.load(std::memory_order_relaxed);
        
        // Project Neurons (Draw soft circles)
        for (int i = 0; i < 512; ++i) {
            float nx = ipc_data->neuron_points[i].x;
            float ny = ipc_data->neuron_points[i].y;
            
            int px = static_cast<int>(nx * (WIDTH - 100)) + 50;
            int py = static_cast<int>(ny * (HEIGHT - 100)) + 50;
            
            if (ipc_data->neuron_points[i].is_anchor) {
                // Top-8 Variance Output Anchors are Red
                cv::circle(frame, cv::Point(px, py), 12, cv::Scalar(0, 0, 255), -1);
                cv::circle(frame, cv::Point(px, py), 14, cv::Scalar(0, 50, 255), 2);
            } else {
                // Dim depending on variance score
                int intensity = std::min(255, static_cast<int>(ipc_data->neuron_points[i].variance_score * 5000.0f + 50));
                cv::circle(frame, cv::Point(px, py), 4, cv::Scalar(intensity, intensity, intensity), -1);
            }
        }

        // Project Bytes (Blue/Cyan squares)
        for (int b = 0; b < 256; ++b) {
            float bx = ipc_data->byte_coords[b].x;
            float by = ipc_data->byte_coords[b].y;
            
            int px = static_cast<int>(bx * (WIDTH - 100)) + 50;
            int py = static_cast<int>(by * (HEIGHT - 100)) + 50;
            
            cv::rectangle(frame, cv::Rect(px - 3, py - 3, 6, 6), cv::Scalar(255, 200, 0), cv::FILLED);
            
            // Draw velocity vectors
            float vx = ipc_data->byte_coords[b].velocity[0] * 5000.0f;
            float vy = ipc_data->byte_coords[b].velocity[1] * 5000.0f;
            cv::line(frame, cv::Point(px, py), cv::Point(px + static_cast<int>(vx), py + static_cast<int>(vy)), cv::Scalar(100, 100, 50), 1);
        }
        
        // Render HUD stats
        char hud_str[256];
        snprintf(hud_str, sizeof(hud_str), "Frame: %llu | CE Loss: %.4f | Swarm Entropy: %.4f",
                 counter,
                 ipc_data->current_ce_loss.load(std::memory_order_relaxed),
                 ipc_data->swarm_entropy.load(std::memory_order_relaxed));
                 
        cv::putText(frame, hud_str, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        
        cv::imshow("Geometric Neural Field (4D Manifold Projection)", frame);
        int key = cv::waitKey(33);
        if (key == 27) break; // ESC
    }

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}
