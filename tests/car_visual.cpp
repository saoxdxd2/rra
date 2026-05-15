// car_visual.cpp â€” rra Neural Car | Population + Checkpoints + Plasticity
// cl /std:c++17 /O2 /EHsc /Fe:tests\car_visual.exe tests\car_visual.cpp nn\brain_tree\brain_tree.cpp /I. /link gdiplus.lib user32.lib gdi32.lib kernel32.lib ole32.lib
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <objbase.h>
#include <objidl.h>
#include <gdiplus.h>
#pragma comment(lib,"gdiplus.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
#include "../nn/brain_tree/brain_tree.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <numeric>
using namespace Gdiplus;
using namespace rra::nn::topology;

// â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
constexpr int   WIN_W   = 1280, WIN_H = 720;
constexpr int   PANEL_W = 300,  VIEW_W = WIN_W - PANEL_W;
constexpr float SCALE   = 14.f, OX = 20.f, OY = 60.f;
constexpr float PI      = 3.14159265f, D2R = PI/180.f;
constexpr float CAR_SPD_BASE=0.4f, STEER=3.0f, SIM_MS=20.f;
constexpr float SPD_MIN=0.1f,  SPD_MAX_BASE=1.2f, SPD_CAP=2.5f;
constexpr int   N_POP=16;
constexpr int   RACE_TICKS=900;   // total race timeout per generation
constexpr int   CP_TIMEOUT=200;   // ticks to reach next checkpoint before penalty
// Neuron map: 0-4 sensors | 5 steer-L | 6 steer-R | 7 accel | 8 brake | 9-16 hidden
constexpr int MOTOR_L=5,MOTOR_R=6,MOTOR_ACCEL=7,MOTOR_BRAKE=8,HIDDEN_BASE=9,N_NEURONS=30;

// â”€â”€ Track geometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
struct Seg{float x1,y1,x2,y2;};
const std::vector<Seg> OUTER={{0,0,60,0},{60,0,60,30},{60,30,0,30},{0,30,0,0}};
const std::vector<Seg> INNER={{12,8,48,8},{48,8,48,22},{48,22,12,22},{12,22,12,8}};

// Checkpoints: (cx, cy, radius) â€” clockwise loop starting top-centre
struct CP{float x,y,r;};
const std::vector<CP> CPS={
    {50,4,4},{55,15,4},{50,26,4},
    {30,26,4},{10,26,4},{5,15,4},
    {10,4,4},{30,4,4}
};

// â”€â”€ Math â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
REAL wx(float x){return OX+x*SCALE;}
REAL wy(float y){return OY+y*SCALE;}

float rseg(float rx,float ry,float dx,float dy,float x1,float y1,float x2,float y2){
    float ex=x2-x1,ey=y2-y1,d=dx*ey-dy*ex;
    if(fabsf(d)<1e-6f)return 1e9f;
    float t=((x1-rx)*ey-(y1-ry)*ex)/d;
    float u=((x1-rx)*dy-(y1-ry)*dx)/d;
    return(t>=0&&u>=0&&u<=1)?t:1e9f;
}
float ray(float rx,float ry,float a,float mx=18.f){
    float dx=cosf(a),dy=sinf(a),c=mx;
    for(auto&s:OUTER){float t=rseg(rx,ry,dx,dy,s.x1,s.y1,s.x2,s.y2);if(t<c)c=t;}
    for(auto&s:INNER){float t=rseg(rx,ry,dx,dy,s.x1,s.y1,s.x2,s.y2);if(t<c)c=t;}
    return c;
}
bool onTrack(float x,float y){
    return x>0.5f&&x<59.5f&&y>0.5f&&y<29.5f&&!(x>12.5f&&x<47.5f&&y>8.5f&&y<21.5f);
}

// â”€â”€ Brain factory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
std::unique_ptr<CorticalTissue> mkBrain(uint32_t seed,const std::vector<float>&inherit={}){
    auto t=std::make_unique<CorticalTissue>();
    for(int i=0;i<26;i++) t->add_neuron(i, NeuronType::EXCITATORY);
    for(int i=26;i<N_NEURONS;i++) t->add_neuron(i, NeuronType::INHIBITORY);
    
    if(!inherit.empty()) {
        t->set_weights(inherit);
    } else {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dd(1.f,8.f);
        // sensors â†’ hidden
        for(int s=0;s<5;s++) for(int h=HIDDEN_BASE;h<N_NEURONS;h++) t->connect_neurons(s,h,dd(rng));
        // hidden â†’ all 4 motor outputs
        for(int h=HIDDEN_BASE;h<N_NEURONS;h++){
            t->connect_neurons(h,MOTOR_L,dd(rng));
            t->connect_neurons(h,MOTOR_R,dd(rng));
            t->connect_neurons(h,MOTOR_ACCEL,dd(rng));
            t->connect_neurons(h,MOTOR_BRAKE,dd(rng));
        }
        // lateral inhibition between steering outputs and between throttle outputs
        t->connect_neurons(MOTOR_L,MOTOR_R,1.f); t->connect_neurons(MOTOR_R,MOTOR_L,1.f);
        t->connect_neurons(MOTOR_ACCEL,MOTOR_BRAKE,1.f); t->connect_neurons(MOTOR_BRAKE,MOTOR_ACCEL,1.f);
    }
    return t;
}

// â”€â”€ Shared state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
struct Agent{
    float x=30,y=4,angle=0;
    float sensors[5]={};
    float ema_sl=0,ema_sr=0,ema_sa=0,ema_sb=0; // Smoothed EMA steering & acceleration
    float speed=CAR_SPD_BASE;
    int   cp=0,crashes=0,cp_timer=0,race_ticks=0;
    float fit=0;
    bool  alive=true;
    ARGB  col;
};
const ARGB COLORS[N_POP]={
    Color::MakeARGB(220,0,200,255),   // cyan
    Color::MakeARGB(220,255,80,50),   // orange
    Color::MakeARGB(220,80,255,100),  // green
    Color::MakeARGB(220,255,50,200),  // pink
    Color::MakeARGB(220,255,230,0),   // yellow
    Color::MakeARGB(220,160,80,255),  // purple
    Color::MakeARGB(220,255,100,100), // red
    Color::MakeARGB(220,100,255,100), // lime
    Color::MakeARGB(220,100,100,255), // blue
    Color::MakeARGB(220,255,150,0),   // amber
    Color::MakeARGB(220,0,255,200),   // teal
    Color::MakeARGB(220,255,0,255),   // magenta
    Color::MakeARGB(220,200,200,200), // silver
    Color::MakeARGB(220,100,200,100), // moss
    Color::MakeARGB(220,200,100,200), // lavender
    Color::MakeARGB(220,255,255,255), // white
};

struct SharedState{
    Agent agents[N_POP];
    float best_fit=0;
    int   gen=0,ticks=0;
    float neuron_pulse[N_NEURONS]={};
    float cur_speed=CAR_SPD_BASE;
};
SharedState g_state={};
std::vector<float> g_best_weights;
bool g_running=true;

// â”€â”€ Logger â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void writeLog(int gen, float speed,
              const Agent agents[], const std::vector<float>& weights){
    // Compute population statistics
    float fits[N_POP], fmean=0, fstd=0;
    int   cps[N_POP];
    int   total_crashes=0;
    for(int i=0;i<N_POP;i++){
        fits[i]=agents[i].fit; fmean+=fits[i];
        cps[i]=agents[i].cp;
        total_crashes+=agents[i].crashes;
    }
    fmean/=N_POP;
    for(int i=0;i<N_POP;i++) fstd+=(fits[i]-fmean)*(fits[i]-fmean);
    fstd=sqrtf(fstd/N_POP);
    float best_fit=*std::max_element(fits,fits+N_POP);
    float worst_fit=*std::min_element(fits,fits+N_POP);
    float best_cp =*std::max_element(cps,cps+N_POP);
    float mean_cp =0; for(int i=0;i<N_POP;i++) mean_cp+=cps[i]; mean_cp/=N_POP;

    // Scalar Energy: The magnitude of the weight vector (L2 norm)
    float scalar_energy = 0;
    for(float w : weights) scalar_energy += w*w;
    scalar_energy = sqrtf(scalar_energy);

    // Synapse weight statistics
    float ampa_sum=0,ampa_sq=0,delay_sum=0,delay_sq=0;
    int n_syn=0;
    for(int i=0;i<(int)weights.size()-3;i+=4){
        float a=weights[i+2], d=weights[i+3];
        ampa_sum+=a; ampa_sq+=a*a;
        delay_sum+=d; delay_sq+=d*d;
        n_syn++;
    }
    float ampa_mean=n_syn?ampa_sum/n_syn:0;
    float ampa_std =n_syn?sqrtf(ampa_sq/n_syn-ampa_mean*ampa_mean):0;
    float delay_mean=n_syn?delay_sum/n_syn:0;
    float delay_std =n_syn?sqrtf(delay_sq/n_syn-delay_mean*delay_mean):0;

    // CSV row (Extended with Scalar and Diversity)
    bool is_new = std::ifstream("logs/training_log.csv").peek() == std::ifstream::traits_type::eof();
    std::ofstream csv("logs/training_log.csv", std::ios::app);
    if(is_new || gen == 0) csv<<"gen,speed,best_fit,worst_fit,mean_fit,std_fit,best_cp,mean_cp,crashes,"
                                "n_syn,ampa_mean,ampa_std,delay_mean_ms,delay_std_ms,scalar_energy\n";
    csv<<gen<<","
       <<std::fixed<<std::setprecision(4)
       <<speed<<","<<best_fit<<","<<worst_fit<<","<<fmean<<","<<fstd<<","
       <<best_cp<<","<<mean_cp<<","<<total_crashes<<","
       <<n_syn<<","<<ampa_mean<<","<<ampa_std<<","
       <<delay_mean<<","<<delay_std<<","<<scalar_energy<<"\n";

    // Human-readable math summary (append)
    std::ofstream sum("logs/training_summary.txt", std::ios::app);
    sum<<"\nâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•\n";
    sum<<"Generation "<<gen<<"\n";
    sum<<"  Speed       v = "<<std::fixed<<std::setprecision(3)<<speed
       <<" u/tick  (v0=0.400 Ã— 1.08^"<<gen<<" â‰ˆ "<<speed<<")\n";
    sum<<"  Fitness     best="<<std::setprecision(2)<<best_fit
       <<"  Î¼="<<fmean<<"  Ïƒ="<<fstd<<"\n";
    sum<<"  Checkpoints best="<<best_cp<<"  Î¼="<<std::setprecision(1)<<mean_cp<<"\n";
    sum<<"  Crashes     total="<<total_crashes<<"\n";
    sum<<"  Synapses    N="<<n_syn<<"\n";
    sum<<"  AMPA        Î¼="<<std::setprecision(4)<<ampa_mean
       <<"  Ïƒ="<<ampa_std<<"  (Î”W governed by UniversalPlasticity MLP, 369 params)\n";
    sum<<"  Axonal Delay Î¼="<<std::setprecision(2)<<delay_mean
       <<"ms  Ïƒ="<<delay_std<<"ms\n";
    sum<<"  Learning     Î”W(t) = MLP(trace_v_preÂ·e^(-t/Ï„_pre), trace_v_postÂ·e^(-t/Ï„_post),"
         " trace_CaÂ·e^(-t/Ï„_Ca), R)  Ï„=11ms\n";
}

// â”€â”€ .safetensors I/O (HuggingFace spec) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Format: [8 bytes LE uint64 = header_len] [header_len bytes UTF-8 JSON] [flat F32 data]
// Sorting note: get_weights() uses std::sort (introsort, O(N log N)) on neuron IDs
// to guarantee a DETERMINISTIC weight vector -- critical for cross-generation inheritance.
void saveSafetensors(const std::string& path, const std::vector<float>& w,
                     int gen, float best_fit){
    uint64_t n = w.size();
    std::ostringstream jss;
    jss << "{\"__metadata__\":{\"gen\":\"" << gen
        << "\",\"best_fit\":\"" << best_fit
        << "\",\"n_neurons\":\"" << N_NEURONS
        << "\",\"layout\":\"[pre_id, post_id, ampa, delay] flat array\"},"
        << "\"synapse_weights\":{\"dtype\":\"F32\",\"shape\":[" << n
        << "],\"data_offsets\":[0," << (n*4) << "]}}";
    std::string meta = jss.str();
    while(meta.size()%8!=0) meta+=' ';
    uint64_t hlen = (uint64_t)meta.size();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(meta.data(), (std::streamsize)hlen);
    f.write(reinterpret_cast<const char*>(w.data()), (std::streamsize)(n*4));
}

// Returns empty vector if file not found
std::vector<float> loadSafetensors(const std::string& path){
    std::ifstream f(path, std::ios::binary);
    if(!f.is_open()) return {};
    uint64_t hlen=0;
    f.read(reinterpret_cast<char*>(&hlen), 8);
    if(hlen==0||hlen>1024*1024) return {}; // sanity
    std::string header(hlen,' ');
    f.read(header.data(), hlen);
    // Read remaining bytes as F32 tensor
    auto start=f.tellg();
    f.seekg(0, std::ios::end);
    auto end=f.tellg();
    uint64_t data_bytes = end-start;
    if(data_bytes%4!=0) return {};
    std::vector<float> w(data_bytes/4);
    f.seekg(start);
    f.read(reinterpret_cast<char*>(w.data()), data_bytes);
    return w;
}

// â”€â”€ Simulation thread â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void simThread(){
    std::mt19937 grng(1337);
    std::normal_distribution<float> noise(0.f,0.05f);

    // â”€â”€ Resume from checkpoint if available â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    CreateDirectoryA("checkpoints",nullptr);
    CreateDirectoryA("logs",nullptr);
    auto resume = loadSafetensors("checkpoints/latest.safetensors");
    if(!resume.empty()){
        g_best_weights = resume;
        // Try to read gen from logs
        std::ifstream gi("checkpoints/latest_gen.txt");
        if(gi.is_open()){ int g=0; gi>>g; g_state.gen=g; }
    }
    while(g_running){
        // Build population â€” agent 0 gets best weights, rest get mutated copies
        std::unique_ptr<CorticalTissue> brains[N_POP];
        for(int i=0;i<N_POP;i++){
            std::vector<float> init=g_best_weights;
            if(!init.empty() && i>0)
                for(auto&v:init) v+=noise(grng);
            brains[i]=mkBrain(grng()+i, init);
        }

        Agent agents[N_POP];
        for(int i=0;i<N_POP;i++){
            agents[i].x=30; agents[i].y=4; agents[i].angle=0;
            agents[i].speed=CAR_SPD_BASE; agents[i].fit=0;
            agents[i].cp=0; agents[i].crashes=0;
            agents[i].cp_timer=0; agents[i].race_ticks=0;
            agents[i].alive=true; agents[i].col=COLORS[i];
        }

        float gtime=0;
        int alive_count=N_POP;

        for(int tick=0;tick<RACE_TICKS&&g_running&&alive_count>0;tick++){
            for(int i=0;i<N_POP;i++){
                if(!agents[i].alive) continue;
                auto& ag=agents[i];
                auto& br=*brains[i];

                // Sense
                for(int r=0;r<5;r++)
                    ag.sensors[r]=ray(ag.x,ag.y,(ag.angle+(r-2)*30.f)*D2R);
                // Encode proximity -> Analog continuous current mapped to Quanta
                for(int s=0;s<5;s++){
                    float prox=std::max(0.f,1.f-ag.sensors[s]/18.f);
                    br.force_spike(s, gtime, prox * SENSORY_GAIN);
                }
                // Read motor neurons before/after brain tick
                int bL=br.get_neuron_spikes(MOTOR_L),    bR=br.get_neuron_spikes(MOTOR_R);
                int bA=br.get_neuron_spikes(MOTOR_ACCEL), bB=br.get_neuron_spikes(MOTOR_BRAKE);
                br.run_until(gtime+SIM_MS);
                
                // EMA Motor Decoding
                float alpha = 1.0f - expf(-SIM_MS / MOTOR_DECAY_TAU);
                ag.ema_sl = ag.ema_sl * (1.0f - alpha) + (br.get_neuron_spikes(MOTOR_L)-bL) * alpha;
                ag.ema_sr = ag.ema_sr * (1.0f - alpha) + (br.get_neuron_spikes(MOTOR_R)-bR) * alpha;
                ag.ema_sa = ag.ema_sa * (1.0f - alpha) + (br.get_neuron_spikes(MOTOR_ACCEL)-bA) * alpha;
                ag.ema_sb = ag.ema_sb * (1.0f - alpha) + (br.get_neuron_spikes(MOTOR_BRAKE)-bB) * alpha;

                // Neural throttle: model decides acceleration
                float dv=(ag.ema_sa-ag.ema_sb)*0.04f;
                float spd_max=std::min(SPD_CAP, SPD_MAX_BASE+g_state.gen*0.06f);
                ag.speed=std::clamp(ag.speed+dv, SPD_MIN, spd_max);

                // Act
                ag.angle+=(ag.ema_sr-ag.ema_sl)*STEER;
                ag.x+=cosf(ag.angle*D2R)*ag.speed;
                ag.y+=sinf(ag.angle*D2R)*ag.speed;
                ag.race_ticks++; ag.cp_timer++;

                // â”€â”€ Reward matrix (all scenarios) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                float reward=0;
                bool crashed=!onTrack(ag.x,ag.y);

                if(crashed){
                    // R1: Wall collision
                    reward=-3.f; ag.crashes++;
                    br.inject_dopamine(reward,gtime+SIM_MS);
                    ag.fit+=reward;
                    auto&last=CPS[ag.cp%CPS.size()];
                    ag.x=last.x; ag.y=last.y; ag.angle=0; ag.speed=CAR_SPD_BASE;
                    if(ag.crashes>10){ag.alive=false;alive_count--;continue;}
                } else {
                    // R2: Checkpoint timeout â€” stuck too long (GLITCH FIX)
                    // If they don't reach the checkpoint, they die. This prevents circling for passive rewards.
                    if(ag.cp_timer>CP_TIMEOUT){
                        reward=-10.f;
                        br.inject_dopamine(reward,gtime+SIM_MS);
                        ag.fit+=reward; 
                        ag.alive=false; alive_count--; continue;
                    }
                    // R3: Race timeout
                    if(ag.race_ticks>=RACE_TICKS){
                        reward=-5.f;
                        br.inject_dopamine(reward,gtime+SIM_MS);
                        ag.fit+=reward; ag.alive=false; alive_count--; continue;
                    }
                    // R4: Dangerous proximity (any sensor < 2.5)
                    float min_s=*std::min_element(ag.sensors,ag.sensors+5);
                    if(min_s<2.5f){ reward-=0.3f; }
                    // R5: Too slow (cowardice penalty)
                    if(ag.speed<0.15f){ reward-=0.05f; }
                    // R6: Speed+clearance bonus (fast AND safe = good)
                    float clearance=ag.sensors[2]/18.f;
                    reward+=clearance*ag.speed*0.08f;
                    // R7: Lap progress micro-reward
                    reward+=0.02f;
                    if(tick%5==0) br.inject_dopamine(reward,gtime+SIM_MS);
                    ag.fit+=reward;

                    // R8: Checkpoint reached
                    auto&next=CPS[ag.cp%CPS.size()];
                    float dx=ag.x-next.x,dy=ag.y-next.y;
                    if(sqrtf(dx*dx+dy*dy)<next.r){
                        float cp_reward=(ag.cp_timer<100)?8.f:5.f; // speed bonus for fast CP
                        ag.fit+=cp_reward;
                        br.inject_dopamine(cp_reward,gtime+SIM_MS);
                        ag.cp++; ag.cp_timer=0;
                        // R9: Full lap!
                        if(ag.cp%CPS.size()==0 && ag.cp>0){
                            float lap_reward=20.f;
                            ag.fit+=lap_reward;
                            br.inject_dopamine(lap_reward,gtime+SIM_MS);
                        }
                    }
                }
            }
            gtime+=SIM_MS;

            // Update shared state
            SharedState ss;
            ss.gen=g_state.gen; ss.ticks=g_state.ticks+tick;
            ss.cur_speed=agents[0].alive?agents[0].speed:0.f;
            float best=0;
            for(int i=0;i<N_POP;i++){
                ss.agents[i]=agents[i];
                if(agents[i].fit>best) best=agents[i].fit;
            }
            ss.best_fit=best;
            // pulse from agent 0
            for(int n=0;n<N_NEURONS;n++){
                int sp=brains[0]->get_neuron_spikes(n);
                ss.neuron_pulse[n]=std::min(1.f,sp*0.04f);
            }
            g_state=ss;
            std::this_thread::sleep_for(std::chrono::milliseconds(22));
        }

        // Evolution: Contrary Scalar Selection
        // Instead of just taking the max, we evaluate a "Potential" that balances
        // high fitness with the "contrary" magnitude (Scalar Energy).
        struct ScoredAgent { int idx; float score; };
        std::vector<ScoredAgent> ranking;
        for(int i=0; i<N_POP; i++){
            // Potential P = Fitness + λ * ||W_i||
            // This favors high-energy solutions that explore the landscape.
            float weight_mag = 0;
            auto w_i = brains[i]->get_weights();
            for(float val : w_i) weight_mag += val*val;
            weight_mag = sqrtf(weight_mag);
            
            float score = agents[i].fit + 0.1f * weight_mag;
            ranking.push_back({i, score});
        }
        
        std::ranges::sort(ranking, [](const auto& a, const auto& b){ return a.score > b.score; });
        
        int bi = ranking[0].idx;
        g_best_weights = brains[bi]->get_weights();
        float best_fit_gen = agents[bi].fit;

        // Save .safetensors checkpoint
        std::string cp_path = "checkpoints/gen_" + std::to_string(g_state.gen) + ".safetensors";
        saveSafetensors(cp_path, g_best_weights, g_state.gen, best_fit_gen);
        saveSafetensors("checkpoints/latest.safetensors", g_best_weights, g_state.gen, best_fit_gen);
        // Write generation index for resume
        { std::ofstream gi("checkpoints/latest_gen.txt"); gi<<g_state.gen+1; }

        float avg_spd=0;
        for(int i=0;i<N_POP;i++) avg_spd+=agents[i].speed;
        writeLog(g_state.gen,avg_spd/N_POP,agents,g_best_weights);
        g_state.gen++;
    }
}

// â”€â”€ GDI+ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
static StringFormat g_sf;
void DT(Graphics&g,const wchar_t*s,const Font*f,float x,float y,const Brush*b){
    RectF rc(x,y,1800.f,50.f);
    g.DrawString(s,-1,f,rc,&g_sf,b);
}
void DT(Graphics&g,const std::wstring&s,const Font*f,float x,float y,const Brush*b){
    DT(g,s.c_str(),f,x,y,b);
}

// â”€â”€ Render â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
void renderFrame(Graphics&g,const SharedState&s){
    // Background
    SolidBrush bg(Color(255,12,14,22));
    g.FillRectangle(&bg,0,0,WIN_W,WIN_H);

    // Track
    SolidBrush trackB(Color(255,25,40,25));
    g.FillRectangle(&trackB,wx(0),wy(0),60*SCALE,30*SCALE);
    SolidBrush islandB(Color(255,12,14,22));
    g.FillRectangle(&islandB,wx(12),wy(8),36*SCALE,14*SCALE);
    Pen outerP(Color(255,60,200,60),2.5f);
    Pen innerP(Color(255,50,160,50),2.f);
    g.DrawRectangle(&outerP,wx(0),wy(0),60*SCALE,30*SCALE);
    g.DrawRectangle(&innerP,wx(12),wy(8),36*SCALE,14*SCALE);

    // Dashed center lines
    Pen dash(Color(80,60,90,60),1.f);
    REAL dp[]={6,6}; dash.SetDashPattern(dp,2);
    g.DrawLine(&dash,wx(0),wy(4),wx(60),wy(4));
    g.DrawLine(&dash,wx(0),wy(26),wx(60),wy(26));
    g.DrawLine(&dash,wx(6),wy(0),wx(6),wy(30));
    g.DrawLine(&dash,wx(54),wy(0),wx(54),wy(30));

    // Checkpoints
    for(size_t c=0;c<CPS.size();c++){
        Pen cp(Color(60,0,180,255),1.f);
        float r=CPS[c].r*SCALE;
        g.DrawEllipse(&cp,wx(CPS[c].x)-r,wy(CPS[c].y)-r,r*2,r*2);
    }

    // Cars: sensors then body
    for(int i=N_POP-1;i>=0;i--){
        const auto&a=s.agents[i];
        if(!a.alive)continue;
        float cx=wx(a.x),cy=wy(a.y);
        Color col(a.col);

        // Sensor rays
        for(int r=0;r<5;r++){
            float ra=(a.angle+(r-2)*30.f)*D2R;
            float d=a.sensors[r];
            float ex=cx+cosf(ra)*d*SCALE, ey=cy+sinf(ra)*d*SCALE;
            float prox=1.f-d/18.f;
            BYTE ri=(BYTE)(255*prox);
            Pen rp(Color(120,ri,80,0),1.f);
            g.DrawLine(&rp,(REAL)cx,(REAL)cy,(REAL)ex,(REAL)ey);
        }

        // Glow
        for(int gw=3;gw>0;gw--){
            float r2=6.f+(float)(gw*3);
            SolidBrush gb(Color((BYTE)(15*gw),col.GetR(),col.GetG(),col.GetB()));
            g.FillEllipse(&gb,cx-r2,cy-r2,r2*2,r2*2);
        }
        // Body
        SolidBrush body(col);
        g.FillEllipse(&body,cx-6.f,cy-6.f,12.f,12.f);
        // Direction
        float ax=cx+cosf(a.angle*D2R)*10, ay=cy+sinf(a.angle*D2R)*10;
        Pen dp2(Color(200,255,255,255),1.5f);
        g.DrawLine(&dp2,(REAL)cx,(REAL)cy,(REAL)ax,(REAL)ay);

        // CP label
        FontFamily ff(L"Consolas"); Font sf(&ff,9);
        SolidBrush wb(Color(255,220,220,220));
        DT(g,std::to_wstring(a.cp)+L"ck",&sf,cx+8,cy-8,&wb);
    }

    // Header bar
    SolidBrush hdr(Color(200,10,14,30));
    g.FillRectangle(&hdr,0.f,0.f,(REAL)VIEW_W,52.f);
    FontFamily ff(L"Consolas");
    Font hf(&ff,13,FontStyleBold);
    SolidBrush cyan(Color(255,0,190,255));
    DT(g,L"rra | Biological Neural Car  | Polychronization + Eligibility Traces + Universal Plasticity MLP",&hf,10,8,&cyan);

    // â”€â”€ Side panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SolidBrush panBg(Color(255,10,12,22));
    g.FillRectangle(&panBg,(REAL)VIEW_W,0.f,(REAL)PANEL_W,(REAL)WIN_H);
    Pen panEdge(Color(255,0,100,160),1.f);
    g.DrawLine(&panEdge,(REAL)VIEW_W,0.f,(REAL)VIEW_W,(REAL)WIN_H);

    Font tf(&ff,16,FontStyleBold);
    Font lf(&ff,10,FontStyleRegular);
    Font bf(&ff,12,FontStyleBold);
    SolidBrush white(Color(255,210,230,255));
    SolidBrush dim(Color(255,70,90,120));
    SolidBrush grn(Color(255,0,210,90));
    SolidBrush red(Color(255,240,60,40));

    float px=(float)VIEW_W+12, py=16;
    DT(g,L"rra Neural Car",&tf,px,py,&cyan); py+=28;

    auto stat=[&](const wchar_t*lab,const std::wstring&val,const Brush*col){
        DT(g,lab,&lf,px,py,&dim);
        DT(g,val,&bf,px+110,py,col);
        py+=20;
    };
    auto fmt=[](float f,int d=1){std::wostringstream o;o<<std::fixed<<std::setprecision(d)<<f;return o.str();};

    stat(L"Generation:", std::to_wstring(s.gen),  &cyan);
    stat(L"Ticks:",      std::to_wstring(s.ticks), &white);
    stat(L"Best Fit:",   fmt(s.best_fit),           &grn);
    // v(g) = v0 * 1.08^g
    stat(L"Speed:",      fmt(s.cur_speed,3)+L" u/t",&white);
    py+=8;

    // Population table
    DT(g,L"Population",&lf,px,py,&dim); py+=16;
    for(int i=0;i<N_POP;i++){
        const auto&a=s.agents[i];
        Color col(a.col);
        SolidBrush cb(col);
        g.FillEllipse(&cb,(REAL)px,(REAL)(py+2),10.f,10.f);
        std::wostringstream row;
        row<<L"#"<<i<<(a.alive?L" ":L"âœ•")<<L" CP:"<<a.cp
           <<L" Fit:"<<std::fixed<<std::setprecision(1)<<a.fit
           <<L" Cr:"<<a.crashes;
        SolidBrush rb(a.alive?Color(255,200,220,255):Color(255,80,80,80));
        DT(g,row.str(),&lf,px+16,py,&rb);
        py+=16;
    }

    py+=8;
    // Sensor bars
    DT(g,L"Lead Car Sensors",&lf,px,py,&dim); py+=14;
    const auto&lead=s.agents[0];
    const wchar_t*sn[5]={L"LL",L"L ",L"C ",L"R ",L"RR"};
    for(int i=0;i<5;i++){
        float fill=1.f-lead.sensors[i]/18.f;
        if(fill<0)fill=0;
        BYTE ri=(BYTE)(255*fill),gi=(BYTE)(100*(1-fill));
        Color sc(200,ri,gi,0);
        SolidBrush sb(sc); Pen sp(Color(60,ri,gi,0),1.f);
        DT(g,sn[i],&lf,px,py,&dim);
        g.FillRectangle(&sb,(REAL)(px+24),(REAL)py,(REAL)(fill*180.f),11.f);
        g.DrawRectangle(&sp,(REAL)(px+24),(REAL)py,180.f,11.f);
        std::wostringstream dv; dv<<std::fixed<<std::setprecision(1)<<lead.sensors[i];
        DT(g,dv.str(),&lf,px+212,py,&white);
        py+=14;
    }

    py+=8;
    // Neuron pulses
    DT(g,L"Neuron Activity",&lf,px,py,&dim); py+=16;
    for(int n=0;n<15;n++){
        float pulse=s.neuron_pulse[n];
        BYTE br=(BYTE)(255*pulse);
        Color nc;
        if(n<5)      nc=Color(220,0,(BYTE)(120+br/2),br);
        else if(n<7) nc=Color(220,br,(BYTE)(br/2),0);
        else         nc=Color(220,0,(BYTE)(br/2),br);
        SolidBrush nb(nc);
        float nx2=px+(float)((n%8)*34), ny2=py+(float)((n/8)*24);
        g.FillEllipse(&nb,(REAL)nx2,(REAL)ny2,16.f,16.f);
        Pen np2(Color(60,80,80,120),1.f);
        g.DrawEllipse(&np2,(REAL)nx2,(REAL)ny2,16.f,16.f);
    }
}

// â”€â”€ Win32 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HWND g_hwnd; ULONG_PTR g_gdip;

LRESULT CALLBACK WndProc(HWND hwnd,UINT msg,WPARAM wp,LPARAM lp){
    if(msg==WM_DESTROY){g_running=false;PostQuitMessage(0);return 0;}
    if(msg==WM_KEYDOWN&&wp==VK_ESCAPE){g_running=false;DestroyWindow(hwnd);return 0;}
    if(msg==WM_PAINT){
        PAINTSTRUCT ps; HDC hdc=BeginPaint(hwnd,&ps);
        HDC mdc=CreateCompatibleDC(hdc);
        HBITMAP bmp=CreateCompatibleBitmap(hdc,WIN_W,WIN_H);
        SelectObject(mdc,bmp);
        Graphics gfx(mdc);
        gfx.SetSmoothingMode(SmoothingModeAntiAlias);
        gfx.SetTextRenderingHint(TextRenderingHintAntiAlias);
        SharedState snap=g_state;
        renderFrame(gfx,snap);
        BitBlt(hdc,0,0,WIN_W,WIN_H,mdc,0,0,SRCCOPY);
        DeleteObject(bmp); DeleteDC(mdc); EndPaint(hwnd,&ps); return 0;
    }
    return DefWindowProc(hwnd,msg,wp,lp);
}

int WINAPI WinMain(HINSTANCE hInst,HINSTANCE,LPSTR,int){
    GdiplusStartupInput gi; GdiplusStartup(&g_gdip,&gi,nullptr);
    WNDCLASSEXW wc={}; wc.cbSize=sizeof(wc); wc.style=CS_HREDRAW|CS_VREDRAW;
    wc.lpfnWndProc=WndProc; wc.hInstance=hInst;
    wc.hCursor=LoadCursor(nullptr,IDC_ARROW);
    wc.hbrBackground=(HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName=L"rra2"; RegisterClassExW(&wc);
    RECT r={0,0,WIN_W,WIN_H}; AdjustWindowRect(&r,WS_OVERLAPPEDWINDOW,FALSE);
    g_hwnd=CreateWindowExW(0,L"rra2",L"rra Neural Car â€” Population Training",
        WS_OVERLAPPEDWINDOW,80,40,r.right-r.left,r.bottom-r.top,
        nullptr,nullptr,hInst,nullptr);
    ShowWindow(g_hwnd,SW_SHOW); UpdateWindow(g_hwnd);
    std::thread sim(simThread);
    SetTimer(g_hwnd,1,33,nullptr);
    MSG msg;
    while(GetMessage(&msg,nullptr,0,0)){
        if(msg.message==WM_TIMER) InvalidateRect(g_hwnd,nullptr,FALSE);
        TranslateMessage(&msg); DispatchMessage(&msg);
    }
    g_running=false; sim.join();
    GdiplusShutdown(g_gdip); return 0;
}
