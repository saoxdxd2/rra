import os
import time

try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit
except ImportError:
    print("Please install jax: !pip install jax jaxlib")
    exit(1)

print(f"JAX running on: {jax.devices()}")
print("==================================================================")
print("TPU v5e-1 BLACK BOX: Universal Plasticity NLP Discovery")
print("==================================================================")

# ==============================================================================
# 1. UNIVERSAL PLASTICITY MLP (369 Params from weights.txt)
# ==============================================================================
W1 = jnp.array([
    [-0.3629184365272522, 0.22264432907104492, -0.48989230394363403, 0.2930605411529541, -0.14317190647125244, -0.21407383680343628, -0.3447994589805603, 0.2492944449186325, -0.27117830514907837, -0.2981727123260498, 0.032600998878479004, 0.09715521335601807, 0.31673604249954224, 0.20882076025009155, 0.37666988372802734, 0.4512891173362732],
    [0.43994444608688354, -0.376049280166626, -0.27517014741897583, 0.1885174959897995, -0.38384395837783813, 0.41974425315856934, 0.0767030119895935, -0.35801684856414795, 0.5002720355987549, -0.16556048393249512, -0.20847421884536743, 0.4140074551105499, -0.4758610725402832, 0.4287518858909607, -0.09284812211990356, 0.39090389013290405],
    [0.38936150074005127, 0.31405889987945557, -0.22750204801559448, 0.3411502242088318, -0.04046601057052612, -0.1572086215019226, 0.26763033866882324, 0.4761238396167755, 0.6014629602432251, -0.02528280019760132, 0.31671053171157837, -0.4113791584968567, -0.03021601215004921, -0.11428970098495483, 0.42147213220596313, -0.021322712302207947],
    [-0.4248473048210144, 0.21413779258728027, -0.43471384048461914, -0.47503727674484253, -0.43394142389297485, 0.09027713537216187, -0.06775176525115967, 0.2785853147506714, 0.11079269647598267, -0.06622314453125, 0.16444909572601318, 0.2909180819988251, 0.4895424246788025, 0.31771403551101685, 0.42175447940826416, -0.17003470659255981],
], dtype=jnp.float32)
B1 = jnp.array([-0.09627240896224976, 0.04738050699234009, 0.2727903127670288, 0.3142290711402893, -0.30766481161117554, -0.007753193378448486, 0.2064623236656189, 0.30815744400024414, 0.07231497764587402, 0.12693405151367188, -0.4448511600494385, -0.3838866949081421, -0.03394967317581177, -0.48612648248672485, 0.4606032371520996, 0.3292856216430664], dtype=jnp.float32)
W2 = jnp.array([[0.016851156949996948, -0.20117110013961792, 0.2220742404460907, 0.08110874891281128, -0.07120338082313538, -0.0785493552684784, -0.06612232327461243, -0.06664207577705383, -0.018391907215118408, 0.034171104431152344, -0.11693534255027771, 0.15636342763900757, 0.21107381582260132, -0.006239414215087891, 0.05003628134727478, 0.14844459295272827],[ -0.1660679578781128, 0.15006393194198608, -0.17211678624153137, 0.12630245089530945, 0.06988435983657837, -0.19868645071983337, -0.03935819864273071, 0.07275435328483582, 0.20460575819015503, 0.09369516372680664, -0.16351762413978577, -0.1997237503528595, 0.02051982283592224, -0.07432585954666138, 0.05736386775970459, 0.15636664628982544],[ -0.06310248374938965, 0.0754496157169342, 0.21243026852607727, -0.17305302619934082, 0.0192490816116333, -0.008954286575317383, 0.05520334839820862, -0.1662866175174713, -0.20136603713035583, -0.24590370059013367, 0.1116454005241394, 0.07804405689239502, 0.07805266976356506, 0.06759360432624817, 0.22033092379570007, 0.24312996864318848],[ 0.18871024250984192, 0.2184232771396637, 0.03321760892868042, -0.04426947236061096, -0.15791523456573486, -0.14747223258018494, -0.16166061162948608, -0.19939151406288147, -0.15120407938957214, 0.028048396110534668, 0.236479252576828, -0.2323136329650879, -0.0588720440864563, -0.01672804355621338, -0.19710564613342285, 0.0416640043258667],[ -0.21292251348495483, 0.048294633626937866, 0.19306683540344238, -0.09732988476753235, -0.05703023076057434, -0.10157737135887146, -0.05590987205505371, -0.14865753054618835, 0.1378576159477234, -0.1669645607471466, -0.061529457569122314, -0.15888500213623047, -0.16003450751304626, -0.16818368434906006, 0.23743200302124023, 0.109333336353302],[ 0.1606968641281128, -0.19429177045822144, -0.18271726369857788, 0.14925134181976318, -0.18480876088142395, 0.21739447116851807, 0.0315973162651062, -0.2260562777519226, 0.027584224939346313, -0.1403484046459198, 0.052745521068573, 0.15663009881973267, 0.2166212499141693, 0.11584949493408203, 0.039405614137649536, 0.18628475069999695],[ -0.19808214902877808, 0.10931345820426941, -0.06318044662475586, 0.056831300258636475, -0.16605675220489502, -0.0025443434715270996, 0.1136624813079834, -0.23893597722053528, 0.054541438817977905, 0.009485095739364624, -0.2423439919948578, -0.04880410432815552, -0.1624646782875061, 0.189541757106781, 0.23111209273338318, -0.17189133167266846],[ 0.17224901914596558, 0.15997859835624695, -0.05753001570701599, -0.07259681820869446, -0.007814556360244751, -0.1459670066833496, 0.21992331743240356, -0.22633317112922668, 0.2205524444580078, 0.14055639505386353, 0.01989307999610901, 0.17470616102218628, 0.21780818700790405, 0.19181063771247864, -0.12052130699157715, 0.1090465784072876],[ -0.1234150230884552, -0.010517030954360962, 0.1860606074333191, 0.1425911784172058, -0.03814297914505005, 0.07463780045509338, -0.23796269297599792, -0.0856080949306488, 0.07186251878738403, -0.03374972939491272, 0.03867155313491821, -0.022067546844482422, -0.13963881134986877, 0.1560477912425995, -0.003215879201889038, 0.0005006790161132812],[ 0.1655750274658203, -0.16900011897087097, -0.20790520310401917, -0.06616410613059998, -0.033538818359375, -0.04819950461387634, -0.18783950805664062, -0.026461541652679443, -0.22085434198379517, -0.10623559355735779, 0.1341717541217804, 0.16560781002044678, -0.23045507073402405, -0.1281125545501709, 0.0875093936920166, 0.09103992581367493],[ -0.22010508179664612, 0.05749049782752991, 0.06368154287338257, -0.08529141545295715, -0.10067582130432129, -0.1952027678489685, 0.004292309284210205, 0.05564063787460327, -0.2403673529624939, -0.06836804747581482, -0.09800082445144653, 0.0008492767810821533, 0.06476700305938721, -0.2329983115196228, -0.20295602083206177, 0.13621202111244202],[ -0.03736528754234314, -0.1644621193408966, 0.07793664932250977, 0.11603790521621704, -0.13297992944717407, -0.06711447238922119, -0.08856478333473206, -0.05564197897911072, -0.019558221101760864, -0.23532938957214355, 0.16950058937072754, 0.19836395978927612, -0.09526246786117554, 0.025431782007217407, 0.20483750104904175, -0.03970184922218323],[ -0.013565361499786377, -0.13973090052604675, 0.044809699058532715, -0.0960911214351654, 0.06767791509628296, -0.10694694519042969, -0.04395204782485962, 0.22023668885231018, 0.06468731164932251, 0.008682847023010254, -0.07683444023132324, -0.0314556360244751, -0.008542180061340332, 0.12204700708389282, -0.15614357590675354, 0.19027161598205566],[ -0.009619057178497314, 0.004542529582977295, 0.15848305821418762, 0.1398983895778656, -0.1727069616317749, 0.009662538766860962, 0.14508605003356934, -0.06355825066566467, 0.2187616527080536, -0.06205904483795166, -0.17467916011810303, -0.15159446001052856, -0.11026895046234131, 0.0014290213584899902, -0.0222318172454834, -0.0901665985584259],[ -0.19467535614967346, 0.13933107256889343, 0.01475110650062561, 0.06523475050926208, -0.010251522064208984, 0.19934791326522827, -0.15263327956199646, -0.0020832419395446777, 0.2457883358001709, -0.013822227716445923, 0.021061748266220093, -0.1160159707069397, -0.13275521993637085, 0.046625494956970215, 0.05674457550048828, 0.19800987839698792],[ -0.15592962503433228, -0.2279529869556427, 0.16887113451957703, 0.23640495538711548, -0.21086034178733826, -0.009970247745513916, -0.02515140175819397, -0.07641297578811646, -0.24899587035179138, 0.037380099296569824, 0.02085617184638977, 0.1378481090068817, -0.022343188524246216, -0.21709102392196655, 0.19878742098808289, -0.08331739902496338],], dtype=jnp.float32)
B2 = jnp.array([-0.16188499331474304, 0.02284577488899231, -0.1940387487411499, -0.15360009670257568, -0.030543535947799683, 0.21870774030685425, -0.22587144374847412, -0.01386520266532898, 0.08850681781768799, -0.0766032338142395, 0.0816902220249176, 0.0402965247631073, 0.011145144701004028, -0.16100746393203735, -0.08803930878639221, -0.01812576875090599], dtype=jnp.float32)
W3 = jnp.array([[-0.1496814489364624],[0.13770869374275208],[-0.04906770586967468],[0.1996093988418579],[0.06822094321250916],[0.05445930361747742],[0.07109692692756653],[0.18540233373641968],[0.08508104085922241],[0.09746679663658142],[-0.23635584115982056],[-0.1930118203163147],[-0.14879906177520752],[0.21384695172309875],[0.19163542985916138],[0.20750340819358826]], dtype=jnp.float32)
B3 = jnp.array([0.031951963901519775], dtype=jnp.float32)

@jit
def universal_plasticity_scalar(trace_pre, trace_post, trace_ca, reward):
    """
    Compute a SCALAR plasticity modulator from population-level trace statistics.
    Feeds mean(trace_pre), mean(trace_post), mean(trace_ca), reward into the
    369-param MLP to get a single nonlinear gating signal.
    This avoids the (N,N,16) tensor that OOM'd the TPU.
    """
    x = jnp.array([jnp.mean(trace_pre), jnp.mean(trace_post), jnp.mean(trace_ca), reward])
    h1 = jnp.maximum(0.0, x @ W1 + B1)   # (16,)
    h2 = jnp.maximum(0.0, h1 @ W2 + B2)  # (16,)
    return (h2 @ W3 + B3)[0]             # scalar

# ==============================================================================
# 2. HYPERPARAMETERS & TOPOLOGY
# ==============================================================================
V_REST    = -70.0
V_THRESH  = -56.5
TAU_TRACE = 11.0
N_SENSORY = 32
N_EXC_HIDDEN = 64
N_INH_HIDDEN = 16
N_MOTOR   = 32
NUM_NEURONS = 144

TICKS_PER_TOKEN = 20

# Vocabulary Setup
grid_coords = []
for i in range(256):
    grid_coords.append([(i % 16) / 15.0, (i // 16) / 15.0])
vocab_coords_raw = jnp.array(grid_coords, dtype=jnp.float32)

# Global random structures
key = random.PRNGKey(42)
key, k1, k2 = random.split(key, 3)
base_topology = random.bernoulli(k1, 0.15, (NUM_NEURONS, NUM_NEURONS)).astype(jnp.float32)
EXC_MASK = jnp.concatenate([jnp.ones(N_SENSORY+N_EXC_HIDDEN), jnp.zeros(N_INH_HIDDEN), jnp.ones(N_MOTOR)])
INH_MASK = jnp.concatenate([jnp.zeros(N_SENSORY+N_EXC_HIDDEN), jnp.ones(N_INH_HIDDEN), jnp.zeros(N_MOTOR)])
motor_seeds = random.normal(k2, (N_MOTOR, 2))

# Text
TEXT = "HELLO WORLD THIS IS RRA"
tokens = jnp.array([ord(c) for c in TEXT], dtype=jnp.int32)
n_tokens = len(TEXT)

@jit
def encode_token(token_idx, binder, alpha):
    coord = vocab_coords_raw[token_idx]
    new_binder = (1.0 - alpha) * binder + alpha * coord
    ramp = jnp.linspace(0.0, 1.0, N_SENSORY)
    mix_x = jnp.where(ramp < new_binder[0], 1.0, 0.0)
    mix_y = jnp.where(ramp < new_binder[1], 1.0, 0.0)
    sensory_bits = jnp.concatenate([mix_x[:N_SENSORY//2], mix_y[:N_SENSORY//2]])
    return new_binder, sensory_bits

@jit
def decode_motor(motor_sum):
    motor_contrib = jnp.dot(motor_sum, motor_seeds)
    motor_norm = motor_contrib / (jnp.linalg.norm(motor_contrib) + 1e-6)
    return motor_norm  # Return the 2D output vector, not a pred_idx

# ==============================================================================
# 3. BLACK BOX SIMULATION
# ==============================================================================
@jit
def simulate_brain(genome):
    sensory_gain   = genome[0]
    binder_alpha   = genome[1]
    weight_scale   = genome[2]
    ampa_jump      = genome[3]
    gaba_drop      = -jnp.abs(genome[4])
    lr_scale       = genome[5]
    thresh_offset  = genome[6]
    tau_m          = genome[7]

    v_thresh_eff = V_THRESH + thresh_offset
    W = base_topology * EXC_MASK[:, None] * weight_scale

    def process_token(state, token_idx):
        v, spikes, W_cur, binder, trace_pre, trace_post, trace_ca, accum_loss, accum_spikes = state

        # 1. Encode
        new_binder, sensory_bits = encode_token(token_idx, binder, binder_alpha)
        i_ext = jnp.concatenate([sensory_bits * sensory_gain, jnp.zeros(NUM_NEURONS - N_SENSORY)])

        # 2. 20 ticks
        def tick(tick_state, _):
            tv, tsp, motor_acc = tick_state
            tv = V_REST + (tv - V_REST) * jnp.exp(-1.0 / tau_m)
            exc_in = jnp.dot(tsp, W_cur * EXC_MASK[:, None]) * ampa_jump
            inh_in = jnp.dot(tsp, W_cur * INH_MASK[:, None]) * gaba_drop
            tv = tv + i_ext + exc_in + inh_in
            ns = jnp.where(tv >= v_thresh_eff, 1.0, 0.0)
            tv = jnp.where(ns > 0.5, V_REST - 5.0, tv)
            return (tv, ns, motor_acc + ns[-N_MOTOR:]), ns

        (v, spikes, motor_sum), all_spikes = jax.lax.scan(
            tick, (v, spikes, jnp.zeros(N_MOTOR)), jnp.arange(TICKS_PER_TOKEN)
        )

        # 3. Decode: get motor output as 2D vector
        motor_output = decode_motor(motor_sum)
        
        # 4. Loss = distance to TARGET token's coordinate (not closest token!)
        target_coord = vocab_coords_raw[token_idx]
        target_dist = jnp.sum((motor_output - target_coord)**2)
        
        # Check if the closest vocab entry matches the target
        all_dists = jnp.sum((vocab_coords_raw - motor_output[None, :])**2, axis=1)
        pred_idx = jnp.argmin(all_dists)
        correct = (pred_idx == token_idx).astype(jnp.float32)
        reward = 2.0 * correct - 1.0

        # 5. Eligibility Traces
        total_spikes_step = jnp.sum(all_spikes, axis=0)
        new_trace_pre  = trace_pre  * jnp.exp(-TICKS_PER_TOKEN / TAU_TRACE) + total_spikes_step
        new_trace_post = trace_post * jnp.exp(-TICKS_PER_TOKEN / TAU_TRACE) + total_spikes_step
        new_trace_ca   = trace_ca   * jnp.exp(-TICKS_PER_TOKEN / 5.3) + total_spikes_step * 10.0

        # 6. UNIVERSAL PLASTICITY MLP UPDATE
        plasticity_mod = universal_plasticity_scalar(new_trace_pre, new_trace_post, new_trace_ca, reward)
        post_proxy = jnp.concatenate([jnp.zeros(NUM_NEURONS - N_MOTOR),
                                      motor_sum / (TICKS_PER_TOKEN + 1e-6)])
        dw = plasticity_mod * jnp.outer(new_trace_pre, post_proxy) * lr_scale
        W_new = jnp.clip(W_cur + dw * base_topology * EXC_MASK[:, None], 0.0, 5.0)

        total_sp = jnp.sum(all_spikes)
        return (v, spikes, W_new, new_binder, new_trace_pre, new_trace_post, new_trace_ca,
                accum_loss + target_dist, accum_spikes + total_sp), None

    init = (
        jnp.full((NUM_NEURONS,), V_REST),
        jnp.zeros((NUM_NEURONS,)),
        W,
        jnp.array([0.5, 0.5]),
        jnp.zeros((NUM_NEURONS,)),
        jnp.zeros((NUM_NEURONS,)),
        jnp.zeros((NUM_NEURONS,)),
        0.0, 0.0
    )

    final, _ = jax.lax.scan(process_token, init, tokens)
    loss, spikes_total = final[-2], final[-1]
    
    # Homeostasis: HARD penalty for silent or epileptic brains
    # Target: 5% firing rate across all ticks
    ideal_spikes = NUM_NEURONS * TICKS_PER_TOKEN * n_tokens * 0.05  # ~3312
    spike_ratio = spikes_total / (ideal_spikes + 1.0)
    # Quadratic penalty: heavily punish < 50% or > 200% of ideal
    homeostasis = jnp.where(spike_ratio < 0.5, 10.0, 0.0)  # Dead brain = massive penalty
    homeostasis = homeostasis + jnp.where(spike_ratio > 2.0, 5.0, 0.0)  # Epileptic = big penalty
    homeostasis = homeostasis + (spike_ratio - 1.0)**2 * 0.5  # Gentle pull toward ideal
    
    return loss / n_tokens + homeostasis, spikes_total

simulate_population = jit(vmap(simulate_brain))

# ==============================================================================
# 4. EVOLUTION ENGINE
# ==============================================================================
POPULATION_SIZE = 65536
GENERATIONS = 30 # converges fast

@jit
def mutate(genomes, mut_key):
    noise = random.normal(mut_key, genomes.shape) * 0.1
    g = genomes + genomes * noise
    g = g.at[:, 0].set(jnp.clip(g[:, 0],  1.0,  200.0))
    g = g.at[:, 1].set(jnp.clip(g[:, 1],  0.01,  0.99))
    g = g.at[:, 2].set(jnp.clip(g[:, 2],  0.01,  5.0))
    g = g.at[:, 3].set(jnp.clip(g[:, 3],  0.1,   50.0))
    g = g.at[:, 4].set(jnp.clip(g[:, 4],  0.1,   50.0))
    g = g.at[:, 5].set(jnp.clip(g[:, 5],  0.001,  2.0))
    g = g.at[:, 6].set(jnp.clip(g[:, 6], -20.0,  20.0))
    g = g.at[:, 7].set(jnp.clip(g[:, 7],  5.0,   100.0))
    return g

def run_evolution():
    rng = random.PRNGKey(999)
    print(f"Starting NLP Blackbox with Universal Plasticity MLP...")
    
    base_genome = jnp.array([50.0, 0.3, 0.5, 10.0, 10.0, 0.1, 0.0, 22.0])
    rng, sk = random.split(rng)
    genomes = mutate(jnp.tile(base_genome, (POPULATION_SIZE, 1)), sk)

    for gen in range(GENERATIONS):
        t0 = time.time()
        fitness, spikes = simulate_population(genomes)
        ranking = jnp.argsort(fitness)
        
        top_n = POPULATION_SIZE // 10
        parents = jnp.concatenate([genomes[ranking[:top_n]], genomes[ranking[-top_n:]]], axis=0) # diversity
        next_gen = jnp.tile(parents, (POPULATION_SIZE // parents.shape[0], 1))
        
        rng, sk = random.split(rng)
        genomes = mutate(next_gen, sk)
        
        best_i = ranking[0]
        print(f"Gen {gen+1:03d} | Loss: {float(fitness[best_i]):.4f} | Spikes: {float(spikes[best_i]):.0f} | Time: {time.time()-t0:.2f}s")
        if fitness[best_i] < 1e-5: break

    g = genomes[ranking[0]]
    print(f"\nFinal Genome: {g}")

if __name__ == "__main__":
    run_evolution()
