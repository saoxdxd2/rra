#include "brain_tree.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

namespace rra::nn::topology {

// =============================================================================
// UNIVERSAL PLASTICITY EQUATION (DISCOVERED BY TPU)
// =============================================================================

const float UniversalPlasticity::w1[4][16] = {
    {-0.3629184365272522, 0.22264432907104492, -0.48989230394363403, 0.2930605411529541, -0.14317190647125244, -0.21407383680343628, -0.3447994589805603, 0.2492944449186325, -0.27117830514907837, -0.2981727123260498, 0.032600998878479004, 0.09715521335601807, 0.31673604249954224, 0.20882076025009155, 0.37666988372802734, 0.4512891173362732},
    {0.43994444608688354, -0.376049280166626, -0.27517014741897583, 0.1885174959897995, -0.38384395837783813, 0.41974425315856934, 0.0767030119895935, -0.35801684856414795, 0.5002720355987549, -0.16556048393249512, -0.20847421884536743, 0.4140074551105499, -0.4758610725402832, 0.4287518858909607, -0.09284812211990356, 0.39090389013290405},
    {0.38936150074005127, 0.31405889987945557, -0.22750204801559448, 0.3411502242088318, -0.04046601057052612, -0.1572086215019226, 0.26763033866882324, 0.4761238396167755, 0.6014629602432251, -0.02528280019760132, 0.31671053171157837, -0.4113791584968567, -0.03021601215004921, -0.11428970098495483, 0.42147213220596313, -0.021322712302207947},
    {-0.4248473048210144, 0.21413779258728027, -0.43471384048461914, -0.47503727674484253, -0.43394142389297485, 0.09027713537216187, -0.06775176525115967, 0.2785853147506714, 0.11079269647598267, -0.06622314453125, 0.16444909572601318, 0.2909180819988251, 0.4895424246788025, 0.31771403551101685, 0.42175447940826416, -0.17003470659255981},
};

const float UniversalPlasticity::b1[16] = {-0.09627240896224976, 0.04738050699234009, 0.2727903127670288, 0.3142290711402893, -0.30766481161117554, -0.007753193378448486, 0.2064623236656189, 0.30815744400024414, 0.07231497764587402, 0.12693405151367188, -0.4448511600494385, -0.3838866949081421, -0.03394967317581177, -0.48612648248672485, 0.4606032371520996, 0.3292856216430664};

const float UniversalPlasticity::w2[16][16] = {
    {0.016851156949996948, -0.20117110013961792, 0.2220742404460907, 0.08110874891281128, -0.07120338082313538, -0.0785493552684784, -0.06612232327461243, -0.06664207577705383, -0.018391907215118408, 0.034171104431152344, -0.11693534255027771, 0.15636342763900757, 0.21107381582260132, -0.006239414215087891, 0.05003628134727478, 0.14844459295272827},
    {-0.1660679578781128, 0.15006393194198608, -0.17211678624153137, 0.12630245089530945, 0.06988435983657837, -0.19868645071983337, -0.03935819864273071, 0.07275435328483582, 0.20460575819015503, 0.09369516372680664, -0.16351762413978577, -0.1997237503528595, 0.02051982283592224, -0.07432585954666138, 0.05736386775970459, 0.15636664628982544},
    {-0.06310248374938965, 0.0754496157169342, 0.21243026852607727, -0.17305302619934082, 0.0192490816116333, -0.008954286575317383, 0.05520334839820862, -0.1662866175174713, -0.20136603713035583, -0.24590370059013367, 0.1116454005241394, 0.07804405689239502, 0.07805266976356506, 0.06759360432624817, 0.22033092379570007, 0.24312996864318848},
    {0.18871024250984192, 0.2184232771396637, 0.03321760892868042, -0.04426947236061096, -0.15791523456573486, -0.14747223258018494, -0.16166061162948608, -0.19939151406288147, -0.15120407938957214, 0.028048396110534668, 0.236479252576828, -0.2323136329650879, -0.0588720440864563, -0.01672804355621338, -0.19710565, 0.04166400},
    {-0.21292251348495483, 0.048294633626937866, 0.19306683540344238, -0.09732988476753235, -0.05703023076057434, -0.10157737135887146, -0.05590987205505371, -0.14865753054618835, 0.1378576159477234, -0.1669645607471466, -0.061529457569122314, -0.15888500213623047, -0.16003450751304626, -0.16818368434906006, 0.23743200302124023, 0.109333336353302},
    {0.1606968641281128, -0.19429177045822144, -0.18271726369857788, 0.14925134181976318, -0.18480876088142395, 0.21739447116851807, 0.0315973162651062, -0.2260562777519226, 0.027584224939346313, -0.1403484046459198, 0.052745521068573, 0.15663009881973267, 0.2166212499141693, 0.11584949493408203, 0.039405614137649536, 0.18628475069999695},
    {-0.19808214902877808, 0.10931345820426941, -0.06318044662475586, 0.056831300258636475, -0.16605675220489502, -0.0025443434715270996, 0.1136624813079834, -0.23893597722053528, 0.054541438817977905, 0.009485095739364624, -0.2423439919948578, -0.04880410432815552, -0.1624646782875061, 0.189541757106781, 0.23111209273338318, -0.17189133167266846},
    {0.17224901914596558, 0.15997859835624695, -0.05753001570701599, -0.07259681820869446, -0.007814556360244751, -0.1459670066833496, 0.21992331743240356, -0.22633317112922668, 0.2205524444580078, 0.14055639505386353, 0.01989307999610901, 0.17470616102218628, 0.21780818700790405, 0.19181063771247864, -0.12052130699157715, 0.1090465784072876},
    {-0.1234150230884552, -0.010517030954360962, 0.1860606074333191, 0.1425911784172058, -0.03814297914505005, 0.07463780045509338, -0.23796269297599792, -0.0856080949306488, 0.07186251878738403, -0.03374972939491272, 0.03867155313491821, -0.022067546844482422, -0.13963881134986877, 0.1560477912425995, -0.003215879201889038, 0.0005006790161132812},
    {0.1655750274658203, -0.16900011897087097, -0.20790520310401917, -0.06616410613059998, -0.033538818359375, -0.04819950461387634, -0.18783950805664062, -0.026461541652679443, -0.22085434198379517, -0.10623559355735779, 0.1341717541217804, 0.16560781002044678, -0.23045507073402405, -0.1281125545501709, 0.0875093936920166, 0.09103992581367493},
    {-0.22010508179664612, 0.05749049782752991, 0.06368154287338257, -0.08529141545295715, -0.10067582130432129, -0.1952027678489685, 0.004292309284210205, 0.05564063787460327, -0.2403673529624939, -0.06836804747581482, -0.09800082445144653, 0.0008492767810821533, 0.06476700305938721, -0.2329983115196228, -0.20295602083206177, 0.13621202111244202},
    {-0.03736528754234314, -0.1644621193408966, 0.07793664932250977, 0.11603790521621704, -0.13297992944717407, -0.06711447238922119, -0.08856478333473206, -0.05564197897911072, -0.019558221101760864, -0.23532939957214355, 0.16950058937072754, 0.19836395978927612, -0.09526246786117554, 0.025431782007217407, 0.20483750104904175, -0.03970185},
    {-0.013565361499786377, -0.13973090052604675, 0.044809699058532715, -0.0960911214351654, 0.06767791509628296, -0.10694694519042969, -0.04395204782485962, 0.22023668885231018, 0.06468731164932251, 0.008682847023010254, -0.07683444023132324, -0.0314556360244751, -0.008542180061340332, 0.12204700708389282, -0.15614357590675354, 0.19027161598205566},
    {-0.009619057178497314, 0.004542529582977295, 0.15848305821418762, 0.1398983895778656, -0.1727069616317749, 0.009662538766860962, 0.14508605003356934, -0.06355825066566467, 0.2187616527080536, -0.06205904483795166, -0.17467916011810303, -0.15159446001052856, -0.11026895046234131, 0.0014290213584899902, -0.0222318172454834, -0.0901665985584259},
    {-0.19467535614967346, 0.13933107256889343, 0.01475110650062561, 0.06523475050926208, -0.010251522064208984, 0.19934791326522827, -0.15263327956199646, -0.0020832419395446777, 0.2457883358001709, -0.013822227716445923, 0.021061748266220093, -0.1160159707069397, -0.13275521993637085, 0.046625494956970215, 0.05674457550048828, 0.19800987839698792},
    {-0.15592962503433228, -0.2279529869556427, 0.16887113451957703, 0.23640495538711548, -0.21086034178733826, -0.009970247745513916, -0.02515140175819397, -0.07641297578811646, -0.24899587035179138, 0.037380099296569824, 0.02085617184638977, 0.1378481090068817, -0.022343188524246216, -0.21709102392196655, 0.19878742098808289, -0.08331739902496338},
};

const float UniversalPlasticity::b2[16] = {-0.16188499331474304, 0.02284577488899231, -0.1940387487411499, -0.15360009670257568, -0.030543535947799683, 0.21870774030685425, -0.22587144374847412, -0.01386520266532898, 0.08850681781768799, -0.0766032338142395, 0.0816902220249176, 0.0402965247631073, 0.011145144701004028, -0.16100746393203735, -0.08803930878639221, -0.01812576875090599};

const float UniversalPlasticity::w3[16][1] = {
    {-0.1496814489364624},
    {0.13770869374275208},
    {-0.04906770586967468},
    {0.1996093988418579},
    {0.06822094321250916},
    {0.05445930361747742},
    {0.07109692692756653},
    {0.18540233373641968},
    {0.08508104085922241},
    {0.09746679663658142},
    {-0.23635584115982056},
    {-0.1930118203163147},
    {-0.14879906177520752},
    {0.21384695172309875},
    {0.19163542985916138},
    {0.20750340819358826},
};

const float UniversalPlasticity::b3[1] = {0.031951963901519775};

torch::Tensor UniversalPlasticity::w1_t;
torch::Tensor UniversalPlasticity::b1_t;
torch::Tensor UniversalPlasticity::w2_t;
torch::Tensor UniversalPlasticity::b2_t;
torch::Tensor UniversalPlasticity::w3_t;
torch::Tensor UniversalPlasticity::b3_t;
bool UniversalPlasticity::tensors_initialized = false;

void UniversalPlasticity::init_tensors() {
    if (tensors_initialized) return;
    
    w1_t = torch::from_blob((void*)w1, {4, 16}, torch::kFloat32).clone();
    b1_t = torch::from_blob((void*)b1, {16}, torch::kFloat32).clone();
    w2_t = torch::from_blob((void*)w2, {16, 16}, torch::kFloat32).clone();
    b2_t = torch::from_blob((void*)b2, {16}, torch::kFloat32).clone();
    w3_t = torch::from_blob((void*)w3, {16, 1}, torch::kFloat32).clone();
    b3_t = torch::from_blob((void*)b3, {1}, torch::kFloat32).clone();
    
    tensors_initialized = true;
}

torch::Tensor UniversalPlasticity::evaluate_delta_w_batch(const torch::Tensor& inputs) {
    if (!tensors_initialized) {
        init_tensors();
    }
    
    torch::Tensor h1 = torch::relu(torch::matmul(inputs, w1_t) + b1_t);
    torch::Tensor h2 = torch::relu(torch::matmul(h1, w2_t) + b2_t);
    torch::Tensor out = torch::matmul(h2, w3_t) + b3_t;
    return out.squeeze(-1); // returns (M,)
}

// ---------------------------------------------------------
// Event-Driven Chemical Synapse
// ---------------------------------------------------------
EventDrivenSynapse::EventDrivenSynapse(uint64_t post_id, EventDrivenNeuron* parent, float delay) 
    : post_synaptic_id(post_id), 
      parent_neuron(parent),
      last_update_time(0.0f),
      pre_calcium_concentration(0.0f),
      vesicle_pool(200.0f),
      ampa_receptor_count(1.0f),
      axonal_delay(delay),
      trace_v_pre(0.0f),
      trace_ca(0.0f)
{}

float EventDrivenSynapse::process_presynaptic_spike(float current_time, CorticalTissue* tissue) {
    float dt = current_time - last_update_time;
    if (dt < 0.0f) dt = 0.0f; 

    // 1. Analytical Physics Decay
    pre_calcium_concentration *= std::exp(-dt / CALCIUM_DECAY);
    vesicle_pool = 200.0f - (200.0f - vesicle_pool) * std::exp(-VESICLE_REFILL_RATE * dt);

    // 2. Eligibility Trace Decay
    trace_v_pre *= std::exp(-dt / TAU_TRACE_PRE);
    trace_ca *= std::exp(-dt / TAU_TRACE_CA);

    // 3. Instant Spike Influx
    pre_calcium_concentration += CALCIUM_INFLUX;
    trace_v_pre += 1.0f;
    trace_ca += CALCIUM_INFLUX;

    // 4. SNARE Complex & Exact Fused Quanta
    float calcium_binding = std::pow(pre_calcium_concentration, 4.0f);
    float fusion_prob = 1.0f - std::exp(-FUSION_RATE * calcium_binding);
    float vesicles_fused = vesicle_pool * fusion_prob;
    vesicle_pool -= vesicles_fused;

    last_update_time = current_time;
    return vesicles_fused * ampa_receptor_count;
}

void EventDrivenSynapse::apply_dopamine_wave(float current_time, float global_reward, CorticalTissue* tissue) {
    float dt = current_time - last_update_time;
    if (dt < 0.0f) dt = 0.0f;

    float current_trace_v_pre = trace_v_pre * std::exp(-dt / TAU_TRACE_PRE);
    float current_trace_ca = trace_ca * std::exp(-dt / TAU_TRACE_CA);

    float current_trace_v_post = 0.0f;
    auto post_neuron = tissue->get_neuron(post_synaptic_id);
    if (post_neuron) {
        current_trace_v_post = post_neuron->trace_v_post;
    }

    torch::Tensor inp = torch::tensor({{current_trace_v_pre, current_trace_v_post, current_trace_ca, global_reward}}, torch::kFloat32);
    float delta_w = UniversalPlasticity::evaluate_delta_w_batch(inp).item<float>();
    
    ampa_receptor_count += delta_w * 0.01f; 
    ampa_receptor_count -= ampa_receptor_count * 0.001f;
    axonal_delay += delta_w * 0.1f;
    
    if (ampa_receptor_count < 0.0f) ampa_receptor_count = 0.0f;
    if (ampa_receptor_count > 5.0f) ampa_receptor_count = 5.0f;
    if (axonal_delay < 1.0f) axonal_delay = 1.0f;
    if (axonal_delay > 15.0f) axonal_delay = 15.0f;
}

// ---------------------------------------------------------
// Event-Driven Biological Neuron
// ---------------------------------------------------------
EventDrivenNeuron::EventDrivenNeuron(uint64_t id, NeuronType t) 
    : morton_code(id), 
      type(t),
      V_m(V_REST), 
      last_update_time(0.0f),
      total_spikes(0),
      trace_v_post(0.0f),
      last_spike_time(-1000.0f)
{
    uint64_t x = id + 0x9E3779B97F4A7C15ULL;
    for(int i=0; i<8; ++i) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        morton_seed[i] = x * 0x2545F4914F6CDD1DULL;
    }
}

void EventDrivenNeuron::update_to_time(float current_time) {
    float dt = current_time - last_update_time;
    if (dt <= 0.0f) return;

    V_m = V_REST + (V_m - V_REST) * std::exp(-dt / TAU_M);
    trace_v_post *= std::exp(-dt / TAU_TRACE_POST);

    last_update_time = current_time;
}

float EventDrivenNeuron::process_incoming_neurotransmitter(float current_time, float quanta, NeuronType source_type) {
    if (current_time - last_spike_time < 2.0f) {
        return -1.0f;
    }

    update_to_time(current_time);

    float voltage_jump = 0.0f;
    if (source_type == NeuronType::EXCITATORY) {
        voltage_jump = quanta * EventDrivenSynapse::AMPA_VOLTAGE_JUMP * EXCITATORY_SCALE; 
        V_m += voltage_jump - 0.05f * quanta;
    } else {
        voltage_jump = quanta * GABA_VOLTAGE_DROP * INHIBITORY_SCALE;
        V_m += voltage_jump;
    }
    
    trace_v_post += voltage_jump;

    if (V_m >= V_THRESH) {
        V_m = V_REST - 5.0f;
        total_spikes++;
        last_spike_time = current_time;
        return current_time;
    }
    return -1.0f; 
}

// ---------------------------------------------------------
// The Asynchronous Event-Driven Engine
// ---------------------------------------------------------
CorticalTissue::CorticalTissue() : global_time_(0.0f) {}

void CorticalTissue::add_neuron(uint64_t morton_code, NeuronType type) {
    if (neurons_.find(morton_code) == neurons_.end()) {
        neurons_[morton_code] = std::make_shared<EventDrivenNeuron>(morton_code, type);
    }
}

void CorticalTissue::connect_neurons(uint64_t pre_id, uint64_t post_id, float delay) {
    if (neurons_.find(pre_id) != neurons_.end() && neurons_.find(post_id) != neurons_.end()) {
        auto syn = std::make_shared<EventDrivenSynapse>(post_id, neurons_[pre_id].get(), delay);
        neurons_[pre_id]->outgoing_synapses.push_back(syn);
    }
}

void CorticalTissue::force_spike(uint64_t id, float time, float quanta) {
    SpikeEvent e = {time, id, quanta, NeuronType::EXCITATORY}; 
    event_queue_.push(e);
}

EventDrivenNeuron* CorticalTissue::get_neuron(uint64_t id) {
    auto it = neurons_.find(id);
    if (it != neurons_.end()) return it->second.get();
    return nullptr;
}

float CorticalTissue::get_neuron_voltage(uint64_t id) const {
    auto it = neurons_.find(id);
    if (it != neurons_.end()) {
        return it->second->V_m;
    }
    return V_REST;
}

int CorticalTissue::get_neuron_spikes(uint64_t id) const {
    auto it = neurons_.find(id);
    if (it != neurons_.end()) {
        return it->second->total_spikes;
    }
    return 0;
}

void CorticalTissue::inject_dopamine(float reward, float time) {
    // 1. Sync all neurons first so post-synaptic traces are exact
    for (auto& [id, neuron] : neurons_) {
        neuron->update_to_time(time);
    }
    
    // 2. Gather all synapse traces for batch inference
    std::vector<std::shared_ptr<EventDrivenSynapse>> synapses;
    for (auto& [id, neuron] : neurons_) {
        for (auto& syn : neuron->outgoing_synapses) {
            synapses.push_back(syn);
        }
    }
    
    if (synapses.empty()) return;
    
    int64_t M = synapses.size();
    torch::Tensor inputs = torch::zeros({M, 4}, torch::kFloat32);
    auto inputs_a = inputs.accessor<float, 2>();
    
    for (int64_t i = 0; i < M; ++i) {
        auto& syn = synapses[i];
        float dt = time - syn->last_update_time;
        if (dt < 0.0f) dt = 0.0f;
        
        float current_trace_v_pre = syn->trace_v_pre * std::exp(-dt / TAU_TRACE_PRE);
        float current_trace_ca = syn->trace_ca * std::exp(-dt / TAU_TRACE_CA);
        
        float current_trace_v_post = 0.0f;
        auto post_neuron = get_neuron(syn->post_synaptic_id);
        if (post_neuron) {
            current_trace_v_post = post_neuron->trace_v_post;
        }
        
        inputs_a[i][0] = current_trace_v_pre;
        inputs_a[i][1] = current_trace_v_post;
        inputs_a[i][2] = current_trace_ca;
        inputs_a[i][3] = reward;
    }
    
    // 3. Vectorized batch inference on CPU (highly parallel, bypasses individual loop overhead)
    torch::Tensor delta_w_t = UniversalPlasticity::evaluate_delta_w_batch(inputs);
    auto delta_w_a = delta_w_t.accessor<float, 1>();
    
    for (int64_t i = 0; i < M; ++i) {
        auto& syn = synapses[i];
        float delta_w = delta_w_a[i];
        
        syn->ampa_receptor_count += delta_w * 0.01f;
        syn->ampa_receptor_count -= syn->ampa_receptor_count * 0.001f;
        syn->axonal_delay += delta_w * 0.1f;
        
        if (syn->ampa_receptor_count < 0.0f) syn->ampa_receptor_count = 0.0f;
        if (syn->ampa_receptor_count > 5.0f) syn->ampa_receptor_count = 5.0f;
        if (syn->axonal_delay < 1.0f) syn->axonal_delay = 1.0f;
        if (syn->axonal_delay > 15.0f) syn->axonal_delay = 15.0f;
    }
    
    // 4. Pruning pass
    for (auto& [id, neuron] : neurons_) {
        auto it = neuron->outgoing_synapses.begin();
        while (it != neuron->outgoing_synapses.end()) {
            if ((*it)->ampa_receptor_count < 0.01f) {
                it = neuron->outgoing_synapses.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void CorticalTissue::run_until(float target_time_ms) {
    while (!event_queue_.empty() && event_queue_.top().arrival_time <= target_time_ms) {
        SpikeEvent current_event = event_queue_.top();
        event_queue_.pop();

        global_time_ = current_event.arrival_time;

        uint64_t current_neuron_id = current_event.target_neuron_id;
        if (neurons_.find(current_neuron_id) == neurons_.end()) continue;

        auto neuron = neurons_[current_neuron_id];
        float spike_time = neuron->process_incoming_neurotransmitter(global_time_, current_event.neurotransmitter_quanta, current_event.source_type);

        if (spike_time >= 0.0f) {
            for (auto& syn : neuron->outgoing_synapses) {
                float quanta_released = syn->process_presynaptic_spike(global_time_, this);
                SpikeEvent next_event = {
                    global_time_ + syn->axonal_delay, 
                    syn->post_synaptic_id, 
                    quanta_released,
                    neuron->type
                };
                event_queue_.push(next_event);
            }
        }
    }

    global_time_ = target_time_ms;
    for (auto& [id, neuron] : neurons_) {
        neuron->update_to_time(global_time_);
    }
}

int CorticalTissue::get_total_network_spikes() const {
    int total = 0;
    for (const auto& [id, neuron] : neurons_) {
        total += neuron->total_spikes;
    }
    return total;
}

std::vector<float> CorticalTissue::get_weights() const {
    std::vector<float> w;
    std::vector<uint64_t> ids;
    for (const auto& [id, _] : neurons_) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    for (uint64_t id : ids) {
        auto& neuron = neurons_.at(id);
        for (auto& syn : neuron->outgoing_synapses) {
            w.push_back(static_cast<float>(id));
            w.push_back(static_cast<float>(syn->post_synaptic_id));
            w.push_back(syn->ampa_receptor_count);
            w.push_back(syn->axonal_delay);
        }
    }
    return w;
}

void CorticalTissue::set_weights(const std::vector<float>& w) {
    for (auto& [id, neuron] : neurons_) {
        neuron->outgoing_synapses.clear();
    }
    
    for (size_t i = 0; i + 3 < w.size(); i += 4) {
        uint64_t pre_id = static_cast<uint64_t>(w[i]);
        uint64_t post_id = static_cast<uint64_t>(w[i+1]);
        float ampa = w[i+2];
        float delay = w[i+3];
        
        if (neurons_.find(pre_id) != neurons_.end() && neurons_.find(post_id) != neurons_.end()) {
            auto syn = std::make_shared<EventDrivenSynapse>(post_id, neurons_[pre_id].get(), delay);
            syn->ampa_receptor_count = ampa;
            neurons_[pre_id]->outgoing_synapses.push_back(syn);
        }
    }
}

// =============================================================================
// LIBTORCH RESERVOIR ENGINE IMPLEMENTATION
// =============================================================================

ReservoirEngine::ReservoirEngine(const ReservoirGenome& genome, uint32_t seed) 
    : genome_(genome) {
    
    // Tensors Initialization
    grid = torch::zeros({256, 2}, torch::kFloat32);
    W_frozen = torch::zeros({N, N}, torch::kFloat32);
    sign = torch::zeros({N}, torch::kFloat32);
    
    R = register_parameter("R", torch::zeros({N * 2, 2}, torch::kFloat32));
    R_bias = register_parameter("R_bias", torch::tensor({0.5f, 0.5f}, torch::kFloat32));
    
    init_grid();
    init_topology(seed);
    reset_state();
}

void ReservoirEngine::init_grid() {
    auto grid_a = grid.accessor<float, 2>();
    for (int i = 0; i < 256; ++i) {
        grid_a[i][0] = (i % 16) / 15.0f;
        grid_a[i][1] = (i / 16) / 15.0f;
    }
}

void ReservoirEngine::init_topology(uint32_t seed) {
    torch::manual_seed(seed);
    
    // Random Bernoulli topology
    torch::Tensor topo = (torch::rand({N, N}) < 0.15).to(torch::kFloat32);
    
    // Dense sensory-to-motor shortcut
    topo.slice(0, 0, N_S).slice(1, N - N_M, N) = 1.0f;
    
    W_frozen = topo * genome_.weight_scale;
    
    // Dale's law sign masks
    sign = torch::ones({N}, torch::kFloat32) * genome_.ampa_jump;
    sign.slice(0, N_S + N_EXC, N_S + N_EXC + N_INH) = -std::abs(genome_.gaba_drop);
}

void ReservoirEngine::reset_state() {
    v = torch::full({N}, -70.0f, torch::kFloat32);
    bd = torch::tensor({0.5f, 0.5f}, torch::kFloat32);
}

torch::Tensor ReservoirEngine::step(const torch::Tensor& target_co) {
    bd = (1.0f - genome_.binder_alpha) * bd + genome_.binder_alpha * target_co;
    
    // Preferred centers for sensory representation
    torch::Tensor centers = torch::linspace(0.0f, 1.0f, N_S / 2);
    
    // Gaussian population tuning
    torch::Tensor dx = (centers - bd[0]) / 0.08f;
    torch::Tensor iext_x = torch::exp(-0.5f * dx * dx);
    
    torch::Tensor dy = (centers - bd[1]) / 0.08f;
    torch::Tensor iext_y = torch::exp(-0.5f * dy * dy);
    
    torch::Tensor iext = torch::cat({iext_x, iext_y, torch::zeros({N - N_S}, torch::kFloat32)}) * genome_.sensory_gain;
    
    torch::Tensor tsp = torch::zeros({N}, torch::kFloat32);
    torch::Tensor all_sum = torch::zeros({N}, torch::kFloat32);
    
    float decay = std::exp(-1.0f / genome_.tau_m);
    float thresh = -56.5f + genome_.thresh_offset;
    
    // Fused topological SNN update on CPU (vectorized under-the-hood by LibTorch backend)
    for (int t = 0; t < TICKS; ++t) {
        v = -70.0f + (v + 70.0f) * decay;
        torch::Tensor syn_input = torch::matmul(tsp, W_frozen * sign.unsqueeze(1));
        v += iext + syn_input;
        
        torch::Tensor spike_mask = (v >= thresh);
        tsp = spike_mask.to(torch::kFloat32);
        v = torch::where(spike_mask, torch::tensor(-75.0f), v);
        all_sum += tsp;
    }
    
    torch::Tensor f = all_sum / static_cast<float>(TICKS);
    torch::Tensor feats = torch::cat({f, f * f}, 0); // 256 vector
    
    return feats;
}

void ReservoirEngine::train_sequence(const std::string& text, float learning_rate) {
    reset_state();
    
    // Disable LibTorch autograd graphs during SNN simulation + online delta updating
    torch::NoGradGuard no_grad; 
    
    for (char token : text) {
        torch::Tensor target_co = grid[static_cast<uint8_t>(token)];
        torch::Tensor feats = step(target_co);
        
        torch::Tensor output = torch::matmul(feats, R) + R_bias;
        torch::Tensor error = target_co - output;
        
        // Manual online delta rule (faster & lighter than PyTorch graph backprop for linear readouts)
        R += learning_rate * torch::outer(feats, error);
        R_bias += learning_rate * error;
    }
}

char ReservoirEngine::process_token(char token) {
    torch::NoGradGuard no_grad;
    
    torch::Tensor target_co = grid[static_cast<uint8_t>(token)];
    torch::Tensor feats = step(target_co);
    
    torch::Tensor output = torch::matmul(feats, R) + R_bias;
    torch::Tensor output_clipped = torch::clamp(output, 0.0f, 1.0f);
    
    // Distance query to all characters in grid
    torch::Tensor dists = torch::sum(torch::square(grid - output_clipped.unsqueeze(0)), 1);
    int64_t pred = dists.argmin().item<int64_t>();
    
    return static_cast<char>(pred);
}

std::string ReservoirEngine::predict_sequence(const std::string& text) {
    reset_state();
    torch::NoGradGuard no_grad;
    
    std::string out_str = "";
    for (char c : text) {
        out_str += process_token(c);
    }
    return out_str;
}

void ReservoirEngine::save_checkpoint(const std::string& filepath) {
    torch::serialize::OutputArchive archive;
    archive.write("W_frozen", W_frozen);
    archive.write("sign", sign);
    archive.write("R", R);
    archive.write("R_bias", R_bias);
    archive.save_to(filepath);
}

void ReservoirEngine::load_checkpoint(const std::string& filepath) {
    torch::serialize::InputArchive archive;
    archive.load_from(filepath);
    archive.read("W_frozen", W_frozen);
    archive.read("sign", sign);
    archive.read("R", R);
    archive.read("R_bias", R_bias);
}

} // namespace rra::nn::topology
