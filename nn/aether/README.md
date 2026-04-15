# AETHER Field Operator

AETHER replaces token-to-token attention with an implicit field update over a Morton-ordered recursive manifold.

The implemented update is a CPU-practical approximation of:

```text
dPsi/dt = -sum_l omega_l Delta_l Psi + gamma R(Psi) + eta grad_g Psi
```

The implementation avoids an attention matrix. It uses:

- Morton-ordered recursive groups as coarse cells.
- Low-rank spectral heat propagation inside each cell.
- Scale-weighted Laplacian eigenvalue damping.
- Nonlinear resonance as a bounded field reaction.
- Local transport as a nearest-neighbor geometric derivative.
- Sparse geometric probes as virtual microphone arrays.
- Beamforming to reinforce coherent predictive modes.
- Active cancellation to subtract residual/noise modes.
- Adaptive renormalization levels that collapse state while preserving predictive invariants.

For sequence length `n`, model width `d`, number of scales `s`, and active modes `k`, the propagation path is:

```text
O(s * k * n * d)
```

No `QK^T`, no pairwise similarity matrix, and no quadratic attention allocation are used.

The probe path is deliberately small:

```text
O(probe_count * probe_radius * probe_taps)
```

It behaves like cognitive signal processing: sense coherent structures, cache stable probe responses, cancel unstable residuals, and only write back a sparse correction.
