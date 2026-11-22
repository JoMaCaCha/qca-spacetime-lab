# QCA Spacetime Lab — English Research Summary (Phases 17–20)

> **Language note:** This document provides a self-contained English technical summary of the three flagship computational physics research tracks in **QCA Spacetime Lab** (Phases 17–20, including all referee-grade convergence and multi-seed audits). The full historical Spanish technical log across all 20 development phases is archived in [`ANALISIS.md`](ANALISIS.md), and the Phase 17 pilot report is in [`estudio_preguntas_abiertas.md`](estudio_preguntas_abiertas.md).

- **Repository:** https://github.com/JoMaCaCha/qca-spacetime-lab
- **Archival DOI (Zenodo v1.0.0):** https://doi.org/10.5281/zenodo.24450504
- **Interactive WebAssembly SIMD Lab:** https://jomacacha.github.io/qca-spacetime-lab/
- **Author:** Johan Manuel Cabrera Chavarro ([ORCID 0000-0002-7696-4741](https://orcid.org/0000-0002-7696-4741))

---

## Independent Verification Commands

Every numerical result summarized below is reproducible from clean sources via Rust (`cargo test` / `qsu_cli`) and independent Python/NumPy/SciPy scripts:

```bash
# Independent Python/NumPy/SciPy cross-verification (~5 s)
python scripts/verify_via_a_mlodinow_brun.py      # Pathway A: 6x6 spectrum, SO(3) chi^2, Yee-4 S4
python scripts/verify_via_c_schwinger_exact.py    # Pathway C: sparse Lanczos & Schrieffer-Wolff (diff <= 1.2e-7)
python scripts/generate_publication_assets.py     # Regenerates figuras_publicacion/fig_via_{a,b,c}.{pdf,svg,png} + tablas_latex.tex

# Native Rust unit test suites for Phases 17-20
cargo test --release --lib tests::phase17
cargo test --release --lib tests::phase18
cargo test --release --lib tests::phase19
cargo test --release --lib tests::phase20

# Native multithreaded CLI sweeps & referee audits
cargo run --release --bin qsu_cli -- phase20      # Regenerates Phase 20 CSV/JSON datasets
cargo run --release --bin qsu_cli -- pub-checks   # Pathway C chi-convergence + Pathway B 5-seed WL & large-N_f FSS
```

---

## 1. Pathway A — 3+1D Quantum Cellular Automata, Vacuum Birefringence, and Fourth-Order Gauge-Invariant QED

**Core modules & assets:** `src/via_a_qca4.rs`, `src/noncubic_pheno.rs`, `src/studies.rs`, [`scripts/verify_via_a_mlodinow_brun.py`](scripts/verify_via_a_mlodinow_brun.py), `datos_investigacion/via_a_*`, [`figuras_publicacion/fig_via_a.pdf`](figuras_publicacion/fig_via_a.pdf), Web Lab Experiments `129`, `131`, `134`, `137`.

### 1.1 Exact Vacuum Birefringence of Fixed-Order 3+1D Photon QCAs
In Mlodinow & Brun (*Phys. Rev. D* **112**, 074513, 2025; arXiv:2506.20136), the single-step momentum-space evolution operator acting on the six-component photon state is

$$U_{\mathbf{k}} = e^{-i k_x \Delta x\,\gamma_X}\, e^{-i k_y \Delta x\,\gamma_Y}\, e^{-i k_z \Delta x\,\gamma_Z} = \mathrm{diag}\!\big(R(\mathbf{k}),\, R(-\mathbf{k})\big),$$

with $\gamma_j = \sigma_Z \otimes J_j$ and $R(\mathbf{k}) \in SO(3)$. Because

$$\mathrm{Tr}\!\big(R(\pm\mathbf{k})\big) = c_x c_y + c_x c_z + c_y c_z \pm s_x s_y s_z,$$

the rotation angles $\phi(\mathbf{k})$ and $\phi(-\mathbf{k})$ are **not degenerate** whenever $k_x k_y k_z \neq 0$, lifting the opposite-helicity quasi-energies at leading order:

$$\Delta\phi(\mathbf{k}) \equiv \phi(\mathbf{k}) - \phi(-\mathbf{k}) = -n_x n_y n_z\,(k\Delta x)^2 + \mathcal{O}\!\big((k\Delta x)^4\big).$$

This splitting is enforced by field reality $(\mathbf{E}, \mathbf{B} \in \mathbb{R}^3 \implies U(-\mathbf{k}) = U(\mathbf{k})^*)$ and agrees with dimension-5 Standard-Model Extension (SME) effective field theory (Myers & Pospelov, *PRL* **90**, 211601, 2003; Kostelecký & Mewes, *PRD* **80**, 015020, 2009). Direct real-field $(\mathbf{E}, \mathbf{B})$ evolution confirms linear polarization rotation at rate $\Delta\theta/\Delta t = [\phi(\mathbf{k}) - \phi(-\mathbf{k})]/2$ (measuring $-3.725\,01 \times 10^{-3}\ \mathrm{rad/step}$ at $\mathbf{k}\Delta x = (0.21, 0.13, 0.07)$ vs. $-3.724\,94 \times 10^{-3}$ analytical).

Mapping the pure $j = 3$ pattern $\sum_{jm} Y_{jm}(\hat{\mathbf{n}}) k_{(V)jm}^{(5)} = -\frac{1}{2} n_x n_y n_z \Delta x$ onto all seven $j = 3$ optical polarimetry limits of Kislat & Krawczynski (*PRD* **95**, 083013, 2017, Table IV) via a joint profile-$\chi^2$ minimized over $1500$ Haar-random $SO(3)$ orientations yields orientation-independent exclusion of a Planck-scale lattice by $> 5$ orders of magnitude:

| Criterion / Observable | Bound on $\Delta x / \ell_P$ |
|---|---:|
| Time of flight (Mlodinow & Brun, Eq. (51)) | $< 0.36$ |
| Time of flight (corrected sphere rms $1/\sqrt{105}$) | $< 1.25$ |
| Optical polarimetry, individual $95\%$ C.L. (worst $SO(3)$) | $< 2.4 \times 10^{-6}$ |
| Optical polarimetry, joint profile-$\chi^2$ $95\%$ C.L. (worst $SO(3)$) | **$< 1.36 \times 10^{-6}$** |
| Optical polarimetry, joint profile-$\chi^2$ $95\%$ C.L. (median $SO(3)$) | **$< 9.34 \times 10^{-7}$** |
| Optical polarimetry, joint profile-$\chi^2$ $5\sigma$ (worst $SO(3)$) | **$< 3.47 \times 10^{-6}$** |

The same odd-order birefringence obstruction applies to fixed-order BCC photons (Bisio, D'Ariano & Perinotti, *Ann. Phys.* **368**, 177, 2016) and fixed-order split-step QED walks (Eon et al., *Quantum* **7**, 1179, 2023).

### 1.2 Real-Space Fourth-Order Gauge-Invariant QED Pair (`Iso-SC+FCC S₄` + `Yee-4 S₄`)
- **Triple classification of 8 local 3+1D QCA families (Phase 17):** No 2-component local walk survives simultaneous CPT-evenness, doubler freedom, and $\mathcal{O}(k^2)$ isotropy. A palindromic 4-component isotropic SC+FCC ($\frac{1}{3}:\frac{2}{3}$) Wilson–Dirac walk at $r_* = 2/\sqrt{3}$ achieves exact $n = 4$ Lorentz invariance ($\omega(\mathbf{k}) = |\mathbf{k}| + \mathcal{O}(k^5)$) with zero doublers (`gap = 0.3500` at $\Delta t = 0.25$).
- **Real-space $6^3$ coupled circuit (`Iso-SC+FCC S₄` + `Yee-4 S₄`, Phases 18 & 20):** Using strictly local 2-site gates with symmetric L-path Peierls phases on cubic $U(1)$ links and the fourth-order staggered Maxwell Hamiltonian $H_{\text{EM}} = \frac{1}{2}\sum_a \langle E_a, M_a E_a\rangle + \frac{1}{2}\sum_{a<b}\|\mathcal{K}_a\mathcal{K}_b F_{ab}^{(2)}\|^2$ integrated via Yoshida–Suzuki $S_4$:
  - Unitarity norm drift: `4.76e-14`; exact standard Gauss's law: **`8.18e-16`**; local $U(1)$ gauge invariance: `1.25e-16` (density) / `1.62e-14` ($\mathbf{E}$).
  - Real-space $16^3$ polarized wave propagation: `Yee-4 S₄` cuts axial-vs-diagonal anisotropy by **`4,341×`** over Yee-2 (`3.24e-3 -> 7.46e-7`), has zero BZ-corner doublers ($\omega_{\min} = 7/3 = 2.333$), and suppresses oblique $45^\circ$ physical birefringence ($\mathbf{k} = \frac{2\pi}{16}(2,1,1), T=10$) by **`16,571×`** over Mlodinow–Brun (`3.84e-5` vs. `0.6367 rad`; scaling $\mathcal{O}(k^7)$ vs. $\mathcal{O}(k^2)$).
- **4+1D chiral boundary extension (Phase 19):** Embedding `Iso-SC+FCC` / `Symanzik-4` into an open Kaplan–Sen slab ($L_s = 14$) cuts boundary Weyl $\mathcal{O}(k^2)$ anisotropy by **`4,170×`** while preserving Golterman–Shamir opposite-boundary chirality cancellation (`-3.11e-15`).
- **Explicit limitation:** Tree-level $n = 4$ isotropy does not eliminate radiative dimension-4 speed shifts $\delta c^2 = \mathcal{O}(g^2)$ once interactions are turned on (Collins et al., *PRL* **93**, 191301, 2004).

---

## 2. Pathway B — 2D Causal-Set Quantum Gravity Coupled to Sorkin–Johnston Quantum Matter

**Core modules & assets:** `src/via_b_causal_pt.rs`, `src/qgravity.rs`, `src/studies.rs`, `datos_investigacion/via_b_*`, `datos_investigacion/checkpoints/` (20 JSON checkpoints), [`figuras_publicacion/fig_via_b.pdf`](figuras_publicacion/fig_via_b.pdf), Web Lab Experiments `130`, `132`, `135`, `138`.

- **Model:** Dynamic path integral over 2D orders $\Omega_{2\text{D}}$ with the Benincasa–Dowker gravitational action $S_{\text{BD}} = N - 2N_1 + 4N_2 - 2N_3$ coupled to $N_f$ quantum scalar fields via Johnston's exact discrete retarded propagator $G_R = \frac{1}{2}C(I + \frac{1}{2}m^2 C)^{-1}$ (verified against $\frac{1}{2}J_0(m\tau)$ with RMS error `0.0105` on $N = 140$ sprinklings) and the 1-loop Sorkin–Johnston spectral action $S_{\text{mat}} = \frac{1}{2}\ln\det(I + \Delta^T\Delta)$ (exact Cholesky), benchmarked against a classical link-Ising control ($J = 0.45$, cf. Glaser, *CQG* **35**, 084001, 2018).
- **Adaptive 20-replica Parallel Tempering FSS ($N \in \{20, 28, 36, 48, 64\}$, Phase 19):**
  - Pure gravity ($d_\infty = 1.9968 + 0.38/N$), SJ $N_f = 1$ ($1.9974 + 0.88/N$), and SJ $N_f = 2$ ($1.9938 + 1.33/N$) all converge asymptotically to the flat 2D manifold **$d_{\text{MM}}(\infty) = 2.00$** at $\beta = 0$ with short autocorrelation ($\tau_{\text{int}} = 0.5\text{–}0.7$).
  - By contrast, classical link-Ising matter ($J = 0.45$) collapses to **$d_{\text{MM}}(\infty) = 1.6115$** with strong slowing down ($\tau_{\text{int}} = 52\text{–}104$).
- **Multicanonical Wang–Landau across the first-order Surya barrier + 5-seed audit (Phase 20 + `via_b_wl_multiseed.csv`):**
  - Flatness-controlled Wang–Landau + frozen-$g(S)$ multicanonical production directly resolves the bimodal free-energy barrier ($\Delta F > 0$, latent heat $\Delta s \approx 7.5\text{–}11.6$).
  - Across 5 independent seeds: Pure gravity gives $\beta_c = 0.1292 \pm 0.0056$ ($N=20$) $\to 0.1092 \pm 0.0023$ ($N=28$) $\to 0.0936 \pm 0.0050$ ($N=36$), with $\Delta F = 1.30\text{–}1.44$; SJ $N_f = 1$ gives $\beta_c = 0.1296 \pm 0.0022 \to 0.1044 \pm 0.0034 \to 0.0884 \pm 0.0028$, with $\Delta F = 1.52\text{–}2.18$ and $d_{\text{MM}}(0) = 1.957\text{–}2.041$.
- **Near-massless stability & large-$N_f$ finite-size crossover (`via_b_matter_mass_nf.csv` & `via_b_large_nf_fss.csv`):**
  - Down to $m = 0.05$ ($N = 36, \beta = 0$), no IR instability appears ($d_{\text{MM}} = 2.026 \pm 0.008, \tau_{\text{int}} = 0.97$).
  - Scaling $N_f = 8$ across $N = 24 \to 36 \to 48$ reveals that the large-$N_f$ jump is a **finite-size crossover governed by $N_f/N$ competition** between superlinear 2D-order entropy $\ln(N!) \sim N\ln N$ and 1-loop SJ matter $N_f S_{\text{mat}}$: $d_{\text{MM}}(N_f = 8)$ relaxes smoothly toward 2 as $N$ increases (**`3.803 ± 0.210 -> 2.568 ± 0.059 -> 2.196 ± 0.034`**, with $\tau_{\text{int}}: 61.6 \to 27.7 \to 6.55$), in sharp contrast to Ising's persistent asymptotic collapse.

---

## 3. Pathway C — Non-Perturbative IR Lorentz Restoration & BKT Néel Phase in the Two-Flavor Schwinger Model

**Core modules & assets:** `src/via_c_schwinger2f.rs`, `src/dmrg.rs`, `src/mps.rs`, `src/studies.rs`, [`scripts/verify_via_c_schwinger_exact.py`](scripts/verify_via_c_schwinger_exact.py), `datos_investigacion/via_c_*`, [`figuras_publicacion/fig_via_c.pdf`](figuras_publicacion/fig_via_c.pdf), Web Lab Experiments `128`, `133`, `136`, `139`.

- **Non-perturbative trilemma without gauge fields (Phase 17):** For two interacting 1+1D species with unequal bare velocities ($v_A = 1.00, v_B = 0.45$), density-density coupling preserving $U(1)_A \times U(1)_B$ fails to equalize IR velocities (`mismatch = 0.607`), isotropic Heisenberg exchange opens a mass gap ($\Delta_\infty = 0.7441$, exact area law), and gapless flavor mixing equalizes velocities only by breaking $U(1)_A \times U(1)_B$.
- **Breaking the trilemma via 1+1D $U(1)$ gauge confinement (Phases 18–20):** Coupling both oppositely charged species ($Q_r = n_{A,r} - n_{B,r}$, exact $U(1)_A \times U(1)_B$ conservation, zero flavor mixing) to the same quantum electric field $J\sum_r E_r^2$ confines the charged mode into a massive Schwinger boson while locking the gapless neutral meson to a **single IR speed** approaching the bosonization prediction $\sqrt{v_A v_B} = 0.6708$:
  - Exact Lanczos ($N_r \le 10$, 63,504 states; independently verified against SciPy `eigsh` to `max diff = 1.20e-7`): `mismatch = 0.7586` at $J = 0 \to 0.00000$ along the constant-$M_S L$ continuum trajectory ($J = 8/N_r^2$: $v_{\text{IR}}: 0.5326 \to 0.5596 \to 0.5699$, $v_\infty = 0.6281$).
  - Real-time **2-site $U(1)$-TDVP** on the exact un-truncated $D = 7$ MPO ($L = 24\text{–}32$, Hilbert space $2^{32}$, energy drift `2.2–2.7e-5`): local $n_{\alpha,c}|\mathrm{GS}\rangle$ wavepackets synchronize their ballistic light cones by **`7.9×`** at $L = 24$ (`0.728 -> 0.092`) and **`5.1×`** at $L = 32$ (`0.769 -> 0.150`).
  - Contrast with 3+1D perturbative Coulomb QED RG (Chadha & Nielsen, 1983; Anber & Donoghue, 2011): over $M_P \to m_e$, weak 3+1D QED retains `69.4%` of the initial velocity mismatch (`0.7586 -> 0.5267`).
- **Strong-coupling lattice BKT Néel gap & $\chi$-convergence audit (`via_c_chi_convergence.csv`):**
  - Second-order Schrieffer–Wolff projection onto the neutral doublet at $J \gg v_A$ yields an effective XXZ chain with exact anisotropy $\Delta_{\text{ef}} = (v_A^2 + v_B^2)/(2 v_A v_B) = 1 + (v_A - v_B)^2/(2 v_A v_B) > 1$, so **velocity asymmetry $v_A \neq v_B$ itself drives a lattice BKT Néel transition**.
  - Bond-dimension convergence audit ($\chi \in \{24, 32, 44, 56\}$ at $J = 2.50$):
    - Symmetric critical control ($v_B/v_A = 1.00, \Delta_{\text{ef}} = 1.00$): 5-digit convergence (`E24 = -5.455193 -> -5.455213`, `c_eff = +1.232 -> +1.238`, `C_Neel(d=4) = +0.10427`).
    - Deep Néel phase ($v_B/v_A = 0.25, \Delta_{\text{ef}} = 2.125$): `C_Neel(d=4) = +0.435..+0.445` (**converged within 1%, $4.26\times$ above control**); at $\chi = 56$ on $N_r = 24$, DMRG spontaneously selects one of the two quasi-degenerate Néel states, dropping $S_{24}$ by $\sim\ln 2$ — a textbook signature of spontaneous $\mathbb{Z}_2$ symmetry breaking.
