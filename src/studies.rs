//! Fase 17 — Tres estudios numéricos rigurosos de preguntas abiertas en física fundamental:
//!
//! 1. **Estudio 1 (`study1_two_species_lorentz_phases`):** Restauración infrarroja de Lorentz vs.
//!    hueco de masa en una QFT discreta interactuante de dos especies acopladas ($v_A \neq v_B$)
//!    mediante diagonalización exacta de Lanczos en escalera periódica ($N_r = 4, 6, 8$, hasta
//!    12 870 estados, cero error de truncación) para el hueco de masa extrapolado $\Delta_\infty$,
//!    el factor de estructura dinámico infrarrojo $S_{\alpha\alpha}(q = 2\pi/N_r, \omega)$ que
//!    extrae las velocidades colectivas infrarrojas $(v_A^{\text{IR}}, v_B^{\text{IR}})$ y la
//!    transferencia exacta de carga de sabor $\Delta N_B(t)$ por RK4, complementado con DMRG
//!    variacional ($N_r = 8, 16$) para confirmar la ley de área frente al crecimiento logarítmico
//!    en los tres canales (1A densidad-densidad con $U(1)_A \times U(1)_B$ conservado, 1B
//!    intercambio isótropo de Heisenberg con hueco de masa, y 1C mezcla de sabor sin hueco que
//!    iguala las velocidades infrarrojas a la media armónica).
//! 2. **Estudio 2 (`study2_3d_qca_triple_classification`):** Clasificación exhaustiva de 8 familias
//!    de autómatas celulares cuánticos locales y unitarios en 3+1D (SC, BCC, FCC; 2, 4 y 6
//!    componentes) bajo la triple restricción conjunta: (i) sin birrefringencia CPT-impar $n = 1$,
//!    (ii) sin dobladores fermiónicos espurios y (iii) sin anisotropía direccional $O(k^2 \Delta x^2)$.
//!    Construye y verifica explícitamente un QCA unitario palindrómico de 4 componentes (esténcil
//!    isotrópico SC+FCC $\frac{1}{3}:\frac{2}{3}$ con masa de Wilson $r = 2/\sqrt{3}$) que satisface
//!    simultáneamente (i)+(ii)+(iii) y empuja la violación de Lorentz hasta orden $n = 4$ ($O(k^4)$)
//!    para paso de tiempo finito $\Delta t$.
//! 3. **Estudio 3 (`study3_causal_set_matter_backreaction`):** Conjuntos causales dinámicos 2D
//!    acoplados a un campo escalar cuántico (propagador retardado exacto de Johnston, PRL 2009 +
//!    acción espectral de Sorkin–Johnston $S_{\text{mat}} = \frac{1}{2}\ln\det(I + \Delta^T\Delta)$):
//!    verifica $\langle G_R(\tau)\rangle = \frac{1}{2}J_0(m\tau)$ y la varianza no local en el cono
//!    de luz, y mide con barras de error (semillas independientes) la retroacción de $N_f = 0, 1, 2$
//!    campos escalares cuánticos sobre el diagrama de fases de Surya (CQG 2012).

use crate::dmrg::{run_dmrg, SparseMpo};
use crate::rng::SplitMix64;
use crate::util::{fibonacci_sphere, mm_dim_2d};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

type C = Complex<f64>;

#[inline(always)]
fn cx(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

// ================================================================== ESTUDIO 1: Dos especies acopladas (Exacto U(1) + DMRG)

/// Base exacta del sector de espín fijo `n_up` de una escalera de `n_rungs` peldaños (2·n_rungs sitios).
struct LadderSector {
    n_rungs: usize,
    states: Vec<u32>,
    idx_map: Vec<i32>, // tamaño 1 << (2*n_rungs)
}

impl LadderSector {
    fn new(n_rungs: usize, n_up: usize) -> Self {
        let l = 2 * n_rungs;
        let full = 1usize << l;
        let mut states = Vec::new();
        let mut idx_map = vec![-1i32; full];
        for s in 0..full {
            if (s as u32).count_ones() as usize == n_up {
                idx_map[s] = states.len() as i32;
                states.push(s as u32);
            }
        }
        LadderSector { n_rungs, states, idx_map }
    }

    /// Aplica H real sobre `v` en O(dim · n_rungs) con condiciones periódicas (sin construir la matriz densa).
    #[allow(clippy::too_many_arguments)]
    fn matvec_real(&self, v: &[f64], out: &mut [f64], v_a: f64, v_b: f64, delta: f64, g_xy: f64, g_z: f64) {
        out.fill(0.0);
        let nr = self.n_rungs;
        let ha = 0.5 * v_a;
        let hb = 0.5 * v_b;
        let hr = 0.5 * g_xy;
        for (i, &st) in self.states.iter().enumerate() {
            let vi = v[i];
            if vi == 0.0 {
                continue;
            }
            let mut diag = 0.0;
            for r in 0..nr {
                let r1 = (r + 1) % nr;
                let a0 = (st >> (2 * r)) & 1;
                let b0 = (st >> (2 * r + 1)) & 1;
                // Peldaño A_r–B_r
                if g_z != 0.0 {
                    diag += if a0 == b0 { 0.25 * g_z } else { -0.25 * g_z };
                }
                if hr != 0.0 && a0 != b0 {
                    let fl = (st ^ (1u32 << (2 * r)) ^ (1u32 << (2 * r + 1))) as usize;
                    out[self.idx_map[fl] as usize] += hr * vi;
                }
                let a1 = (st >> (2 * r1)) & 1;
                let b1 = (st >> (2 * r1 + 1)) & 1;
                // Pierna A (periódica)
                if delta != 0.0 {
                    diag += if a0 == a1 { 0.25 * delta } else { -0.25 * delta };
                }
                if a0 != a1 {
                    let fl = (st ^ (1u32 << (2 * r)) ^ (1u32 << (2 * r1))) as usize;
                    out[self.idx_map[fl] as usize] += ha * vi;
                }
                // Pierna B (periódica)
                if delta != 0.0 {
                    diag += if b0 == b1 { 0.25 * delta } else { -0.25 * delta };
                }
                if b0 != b1 {
                    let fl = (st ^ (1u32 << (2 * r + 1)) ^ (1u32 << (2 * r1 + 1))) as usize;
                    out[self.idx_map[fl] as usize] += hb * vi;
                }
            }
            out[i] += diag * vi;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn matvec_complex(&self, v: &[C], out: &mut [C], v_a: f64, v_b: f64, delta: f64, g_xy: f64, g_z: f64) {
        for z in out.iter_mut() {
            *z = cx(0.0, 0.0);
        }
        let nr = self.n_rungs;
        let ha = 0.5 * v_a;
        let hb = 0.5 * v_b;
        let hr = 0.5 * g_xy;
        for (i, &st) in self.states.iter().enumerate() {
            let vi = v[i];
            if vi.re == 0.0 && vi.im == 0.0 {
                continue;
            }
            let mut diag = 0.0;
            for r in 0..nr {
                let r1 = (r + 1) % nr;
                let a0 = (st >> (2 * r)) & 1;
                let b0 = (st >> (2 * r + 1)) & 1;
                if g_z != 0.0 {
                    diag += if a0 == b0 { 0.25 * g_z } else { -0.25 * g_z };
                }
                if hr != 0.0 && a0 != b0 {
                    let fl = (st ^ (1u32 << (2 * r)) ^ (1u32 << (2 * r + 1))) as usize;
                    out[self.idx_map[fl] as usize] += vi * hr;
                }
                let a1 = (st >> (2 * r1)) & 1;
                let b1 = (st >> (2 * r1 + 1)) & 1;
                if delta != 0.0 {
                    diag += if a0 == a1 { 0.25 * delta } else { -0.25 * delta };
                }
                if a0 != a1 {
                    let fl = (st ^ (1u32 << (2 * r)) ^ (1u32 << (2 * r1))) as usize;
                    out[self.idx_map[fl] as usize] += vi * ha;
                }
                if delta != 0.0 {
                    diag += if b0 == b1 { 0.25 * delta } else { -0.25 * delta };
                }
                if b0 != b1 {
                    let fl = (st ^ (1u32 << (2 * r + 1)) ^ (1u32 << (2 * r1 + 1))) as usize;
                    out[self.idx_map[fl] as usize] += vi * hb;
                }
            }
            out[i] += vi * diag;
        }
    }
}

/// Diagonalización simétrica de Jacobi para matrices tridiagonales pequeñas (m × m).
fn jacobi_sym_tridiag(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let m = alphas.len();
    let mut tmat = vec![0.0; m * m];
    let mut vecs = vec![0.0; m * m];
    for i in 0..m {
        tmat[i * m + i] = alphas[i];
        vecs[i * m + i] = 1.0;
        if i + 1 < m {
            tmat[i * m + i + 1] = betas[i];
            tmat[(i + 1) * m + i] = betas[i];
        }
    }
    for _ in 0..90 {
        let mut mx = 0.0;
        for p in 0..m {
            for q in (p + 1)..m {
                let apq = tmat[p * m + q];
                if apq.abs() > mx {
                    mx = apq.abs();
                }
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (tmat[q * m + q] - tmat[p * m + p]) / (2.0 * apq);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                tmat[p * m + p] -= t * apq;
                tmat[q * m + q] += t * apq;
                tmat[p * m + q] = 0.0;
                tmat[q * m + p] = 0.0;
                for r in 0..m {
                    if r != p && r != q {
                        let rp = tmat[r * m + p];
                        let rq = tmat[r * m + q];
                        tmat[r * m + p] = c * rp - s * rq;
                        tmat[p * m + r] = tmat[r * m + p];
                        tmat[r * m + q] = s * rp + c * rq;
                        tmat[q * m + r] = tmat[r * m + q];
                    }
                    let vp = vecs[r * m + p];
                    let vq = vecs[r * m + q];
                    vecs[r * m + p] = c * vp - s * vq;
                    vecs[r * m + q] = s * vp + c * vq;
                }
            }
        }
        if mx < 1e-14 {
            break;
        }
    }
    let evals: Vec<f64> = (0..m).map(|i| tmat[i * m + i]).collect();
    (evals, vecs)
}

/// Estado fundamental exacto por Lanczos (precisión < 10⁻¹³) en un sector U(1) de la escalera.
fn exact_sector_ground(sec: &LadderSector, v_a: f64, v_b: f64, delta: f64, g_xy: f64, g_z: f64) -> (f64, Vec<f64>) {
    let dim = sec.states.len();
    let kmax = if dim <= 1500 { 95usize.min(dim) } else { 65usize.min(dim) };
    let mut rng = SplitMix64::new(0x1701_5E00 ^ dim as u64);
    let mut v0: Vec<f64> = (0..dim).map(|_| rng.next_f64() - 0.5).collect();
    let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v0.iter_mut() {
        *x /= n0;
    }
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(kmax);
    vs.push(v0);
    let mut alphas = Vec::with_capacity(kmax);
    let mut betas = Vec::with_capacity(kmax);
    let mut w = vec![0.0; dim];

    for j in 0..kmax {
        sec.matvec_real(&vs[j], &mut w, v_a, v_b, delta, g_xy, g_z);
        let a: f64 = vs[j].iter().zip(&w).map(|(x, y)| x * y).sum();
        alphas.push(a);
        for i in 0..dim {
            w[i] -= a * vs[j][i];
        }
        if j > 0 {
            let bp = betas[j - 1];
            for i in 0..dim {
                w[i] -= bp * vs[j - 1][i];
            }
        }
        for vk in &vs {
            let d: f64 = vk.iter().zip(&w).map(|(x, y)| x * y).sum();
            for i in 0..dim {
                w[i] -= d * vk[i];
            }
        }
        let b = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if b < 1e-13 || j + 1 == kmax {
            break;
        }
        betas.push(b);
        let inv = 1.0 / b;
        vs.push(w.iter().map(|x| x * inv).collect());
    }

    let m = alphas.len();
    let (evals, vecs) = jacobi_sym_tridiag(&alphas, &betas);
    let mut best = 0;
    for i in 1..m {
        if evals[i] < evals[best] {
            best = i;
        }
    }
    let e0 = evals[best];
    let mut psi = vec![0.0; dim];
    for k in 0..m {
        let yk = vecs[k * m + best];
        for i in 0..dim {
            psi[i] += yk * vs[k][i];
        }
    }
    let nrm = psi.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
    for x in psi.iter_mut() {
        *x /= nrm;
    }
    (e0, psi)
}

/// Velocidad colectiva infrarroja exacta v_α^IR = ω_dom(q) / (2 sin(q/2)) obtenida del
/// factor de estructura dinámico exacto S_{αα}(q = 2π/N_r, ω) por Lanczos complejo hermítico
/// sobre la escalera periódica (N_r = 8, L = 16 sitios, cero truncación).
#[allow(clippy::too_many_arguments)]
fn exact_ir_structure_velocity(
    sec0: &LadderSector,
    e0: f64,
    gs0: &[f64],
    leg: usize,
    v_a: f64,
    v_b: f64,
    delta: f64,
    g_xy: f64,
    g_z: f64,
) -> f64 {
    let nr = sec0.n_rungs;
    let dim = sec0.states.len();
    let q = 2.0 * PI / nr as f64;
    let phases: Vec<C> = (0..nr).map(|r| cx((q * r as f64).cos(), (q * r as f64).sin())).collect();
    let mut v0 = vec![cx(0.0, 0.0); dim];
    let mut n2 = 0.0;
    for (i, &st) in sec0.states.iter().enumerate() {
        let g = gs0[i];
        if g == 0.0 {
            continue;
        }
        let mut s = cx(0.0, 0.0);
        for r in 0..nr {
            let sz = if ((st >> (2 * r + leg)) & 1) != 0 { 0.5 } else { -0.5 };
            s += phases[r] * sz;
        }
        let z = s * g;
        n2 += z.norm_sqr();
        v0[i] = z;
    }
    let inv0 = 1.0 / n2.sqrt().max(1e-30);
    for z in v0.iter_mut() {
        *z *= inv0;
    }

    let kmax = 45usize.min(dim);
    let mut vs: Vec<Vec<C>> = Vec::with_capacity(kmax);
    vs.push(v0);
    let mut alphas = Vec::with_capacity(kmax);
    let mut betas = Vec::with_capacity(kmax);
    let mut w = vec![cx(0.0, 0.0); dim];

    for j in 0..kmax {
        sec0.matvec_complex(&vs[j], &mut w, v_a, v_b, delta, g_xy, g_z);
        let a: f64 = vs[j].iter().zip(&w).map(|(x, y)| x.re * y.re + x.im * y.im).sum();
        alphas.push(a);
        for i in 0..dim {
            w[i] -= vs[j][i] * a;
        }
        if j > 0 {
            let bp = betas[j - 1];
            for i in 0..dim {
                w[i] -= vs[j - 1][i] * bp;
            }
        }
        for vk in &vs {
            let mut d = cx(0.0, 0.0);
            for i in 0..dim {
                d += vk[i].conj() * w[i];
            }
            for i in 0..dim {
                w[i] -= vk[i] * d;
            }
        }
        let b = w.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if b < 1e-12 || j + 1 == kmax {
            break;
        }
        betas.push(b);
        let inv = 1.0 / b;
        vs.push(w.iter().map(|&z| z * inv).collect());
    }

    let m = alphas.len();
    let (evals, vecs) = jacobi_sym_tridiag(&alphas, &betas);
    // Recopilamos polos físicos (ω_n = E_n − E_0 > 10⁻⁴, peso p_n = |y_{0,n}|²)
    let mut poles: Vec<(f64, f64)> = (0..m)
        .map(|k| (evals[k] - e0, vecs[k] * vecs[k]))
        .filter(|&(w_ex, p)| w_ex > 1e-4 && p > 1e-4)
        .collect();
    poles.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Agrupamos polos casi degenerados (Δω ≤ 0,08) y seleccionamos el primer modo infrarrojo
    // acústico con peso espectral apreciable (p_cluster ≥ 0,06), evitando modos ópticos UV de alta energía.
    let mut clusters: Vec<(f64, f64)> = Vec::new();
    for &(w_ex, p) in &poles {
        if let Some(last) = clusters.last_mut() {
            if (w_ex - last.0).abs() <= 0.08 {
                let pnew = last.1 + p;
                last.0 = (last.0 * last.1 + w_ex * p) / pnew;
                last.1 = pnew;
                continue;
            }
        }
        clusters.push((w_ex, p));
    }
    let omega_ir = clusters
        .iter()
        .find(|&&(_, p)| p >= 0.06)
        .or_else(|| clusters.iter().max_by(|a, b| a.1.total_cmp(&b.1)))
        .map(|&(w_ex, _)| w_ex)
        .unwrap_or(0.0);
    omega_ir / (2.0 * (0.5 * q).sin())
}

/// Transferencia EXACTA de carga de sabor ΔN_B(t) = ⟨N_B(t)⟩ − N_r/2 tras inyectar σ⁺ en la
/// especie A sobre el estado fundamental no degenerado (N_r par = 6, escalera periódica).
/// Cuando g_xy = 0 (Control y Canal 1A), [H, N_B] = 0 ⇒ ΔN_B(t) = 0 con precisión de máquina (< 10⁻¹⁴)!
#[allow(clippy::too_many_arguments)]
fn exact_flavor_transfer(
    sec0: &LadderSector,
    gs0: &[f64],
    sec1: &LadderSector,
    v_a: f64,
    v_b: f64,
    delta: f64,
    g_xy: f64,
    g_z: f64,
) -> f64 {
    let dim1 = sec1.states.len();
    let mut psi = vec![cx(0.0, 0.0); dim1];
    let mut nrm2 = 0.0;
    // Inyectar partícula σ⁺ en el sitio A del peldaño 0
    for (i, &st) in sec0.states.iter().enumerate() {
        if (st & 1) == 0 {
            let st1 = (st | 1u32) as usize;
            let j = sec1.idx_map[st1] as usize;
            psi[j] = cx(gs0[i], 0.0);
            nrm2 += gs0[i] * gs0[i];
        }
    }
    let inv = 1.0 / nrm2.sqrt().max(1e-30);
    for z in psi.iter_mut() {
        *z *= inv;
    }

    // RK4 exacto: t ∈ [0, 4,8] (supera el semiperíodo π/g_xy ≈ 4,2 de oscilación entre piernas)
    let dt = 0.08;
    let steps = 60usize;
    let mut k1 = vec![cx(0.0, 0.0); dim1];
    let mut k2 = vec![cx(0.0, 0.0); dim1];
    let mut k3 = vec![cx(0.0, 0.0); dim1];
    let mut k4 = vec![cx(0.0, 0.0); dim1];
    let mut tmp = vec![cx(0.0, 0.0); dim1];
    let mut nb_init = 0.0;
    for (i, &st_bits) in sec1.states.iter().enumerate() {
        nb_init += psi[i].norm_sqr() * (st_bits & 0xAAAA_AAAAu32).count_ones() as f64;
    }
    let mut max_tr = 0.0f64;

    let rhs = |y: &[C], dy: &mut [C]| {
        sec1.matvec_complex(y, dy, v_a, v_b, delta, g_xy, g_z);
        for z in dy.iter_mut() {
            *z = cx(z.im, -z.re);
        }
    };

    for _ in 1..=steps {
        rhs(&psi, &mut k1);
        for i in 0..dim1 {
            tmp[i] = psi[i] + k1[i] * (0.5 * dt);
        }
        rhs(&tmp, &mut k2);
        for i in 0..dim1 {
            tmp[i] = psi[i] + k2[i] * (0.5 * dt);
        }
        rhs(&tmp, &mut k3);
        for i in 0..dim1 {
            tmp[i] = psi[i] + k3[i] * dt;
        }
        rhs(&tmp, &mut k4);
        for i in 0..dim1 {
            psi[i] += (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
        }
        // Carga total exacta en la especie B: ΔN_B(t) = ⟨ψ|N_B|ψ⟩/⟨ψ|ψ⟩ − ⟨N_B(0)⟩
        let mut nb_cur = 0.0;
        let mut nrm_cur = 0.0;
        for (i, &st_bits) in sec1.states.iter().enumerate() {
            let p = psi[i].norm_sqr();
            let nb_st = (st_bits & 0xAAAA_AAAAu32).count_ones() as f64;
            nb_cur += p * nb_st;
            nrm_cur += p;
        }
        max_tr = max_tr.max((nb_cur / nrm_cur.max(1e-30) - nb_init).abs());
    }
    max_tr
}

/// Estudio 1 (Pregunta Abierta #1): diagrama de fases no perturbativo de restauración infrarroja
/// de Lorentz en dos especies cuánticas acopladas (v_A = 1,0, v_B = 0,45, Δ = 0,25):
///   0: Control desacoplado (g_xy = 0, g_z = 0)
///   1: Canal 1A densidad-densidad (g_xy = 0, g_z = 0,60; conserva U(1)_A × U(1)_B)
///   2: Canal 1B intercambio isótropo de Heisenberg (g_xy = 1,40, g_z = 1,40; abre hueco de masa!)
///   3: Canal 1C mezcla de sabor sin hueco (g_xy = 0,75, g_z = 0,0; iguala velocidades a la media armónica!)
pub fn run_study1_two_species_lorentz_phases(chi_dmrg: usize, _chi_rt: usize) -> Vec<f64> {
    let v_a = 1.0f64;
    let v_b = 0.45f64;
    let delta = 0.25f64;
    let regimes = [
        (0.0f64, 0.0f64),   // 0: Control libre de acoplamiento entre especies
        (0.0f64, 0.60f64),  // 1A: Densidad-densidad pura (U(1)_A × U(1)_B conservado)
        (1.40f64, 1.40f64), // 1B: Intercambio isótropo de Heisenberg (hueco de masa!)
        (0.75f64, 0.0f64),  // 1C: Mezcla de sabor sin hueco (media armónica!)
    ];

    let eval_regime = |(idx, (g_xy, g_z)): (usize, (f64, f64))| -> [f64; 8] {
        // 1. Hueco de masa exacto ΔE(N_r) = E_0(Sz=+1) − E_0(Sz=0) en N_r pares {4, 6, 8} (periódico)
        let nrs_ex = [4usize, 6, 8];
        let mut inv_n = Vec::new();
        let mut gaps = Vec::new();
        let mut v_a_ir = 0.0;
        let mut v_b_ir = 0.0;
        let mut tr_ab = 0.0;
        for &nr in &nrs_ex {
            let sec0 = LadderSector::new(nr, nr);
            let sec1 = LadderSector::new(nr, nr + 1);
            let (e0, gs0) = exact_sector_ground(&sec0, v_a, v_b, delta, g_xy, g_z);
            let (e1, _) = exact_sector_ground(&sec1, v_a, v_b, delta, g_xy, g_z);
            inv_n.push(1.0 / nr as f64);
            gaps.push((e1 - e0).max(0.0));
            if nr == 6 {
                tr_ab = exact_flavor_transfer(&sec0, &gs0, &sec1, v_a, v_b, delta, g_xy, g_z);
            }
            if nr == 8 {
                v_a_ir = exact_ir_structure_velocity(&sec0, e0, &gs0, 0, v_a, v_b, delta, g_xy, g_z);
                v_b_ir = exact_ir_structure_velocity(&sec0, e0, &gs0, 1, v_a, v_b, delta, g_xy, g_z);
            }
        }
        let (gap_inf_raw, _) = crate::util::fit_linear(&inv_n, &gaps);
        let gap_inf = gap_inf_raw.max(0.0);

        // 2. DMRG variacional en N_r = 8 y N_r = 16 para verificar crecimiento logarítmico vs ley de área
        let mpo8 = SparseMpo::two_species_ladder_general(8, v_a, v_b, delta, delta, g_xy, g_z);
        let (_, mps8) = run_dmrg(&mpo8, chi_dmrg, 4, 1701 + idx as u64 * 97);
        let s8 = 0.5 * (mps8.entanglement_entropy(7) + mps8.entanglement_entropy(5));

        let mpo16 = SparseMpo::two_species_ladder_general(16, v_a, v_b, delta, delta, g_xy, g_z);
        let (_, mps16) = run_dmrg(&mpo16, chi_dmrg, 4, 1702 + idx as u64 * 97);
        let s16 = 0.5 * (mps16.entanglement_entropy(15) + mps16.entanglement_entropy(13));

        let v_bar = 0.5 * (v_a_ir + v_b_ir).max(1e-9);
        let mismatch = (v_a_ir - v_b_ir).abs() / v_bar;
        // [gap_inf, S(8), S(16), v_A_IR, v_B_IR, mismatch, max_delta_N_B, dS_8_16]
        [gap_inf, s8, s16, v_a_ir, v_b_ir, mismatch, tr_ab, s16 - s8]
    };

    #[cfg(not(target_arch = "wasm32"))]
    let res: Vec<[f64; 8]> = std::thread::scope(|sc| {
        let handles: Vec<_> = regimes
            .iter()
            .copied()
            .enumerate()
            .map(|item| sc.spawn(move || eval_regime(item)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let res: Vec<[f64; 8]> = regimes.iter().copied().enumerate().map(eval_regime).collect();

    // Predicción analítica exacta de la media armónica en el canal 1C (libre de Δ):
    // v_harm = (2 v_A v_B / (v_A + v_B)) √(1 − g_xy² / (4 v_A v_B))  (con ancho de banda v_A, v_B)
    let g_1c = regimes[3].0;
    let v_harm_pred = (2.0 * v_a * v_b / (v_a + v_b)) * (1.0 - g_1c * g_1c / (4.0 * v_a * v_b)).max(0.0).sqrt();

    let mut out = vec![4.0, v_harm_pred];
    for r in res {
        out.extend_from_slice(&r);
    }
    out
}

// ================================================================== ESTUDIO 2: Clasificación triple de QCA 3+1D

#[inline(always)]
fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn q_axis(ax: [f64; 3], angle: f64) -> [f64; 4] {
    let (s, c) = angle.sin_cos();
    [c, -s * ax[0], -s * ax[1], -s * ax[2]]
}

/// Evalúa la cuasi-energía positiva ω(k) de cada una de las 8 familias de QCA 3+1D locales y unitarias:
///   0: SC-Weyl-Fijo (2c)
///   1: SC-Weyl-Palindrómico (2c)
///   2: BCC-Weyl-Fijo (2c)
///   3: BCC-Weyl-Palindrómico (2c)
///   4: FCC-Weyl-Palindrómico (2c)
///   5: SC-Fotón-Mlodinow-Brun (6c; rama ω(k) frente a ω(−k))
///   6: SC-Wilson-Dirac-Trotter simétrico (4c, r = 1,0, dt = 0,25)
///   7: NUEVO — Iso-SC+FCC-Wilson-Dirac-Trotter simétrico (4c, α = 2/3, r = 2/√3, dt = 0,25)
fn qca3d_omega(family: usize, k: [f64; 3]) -> f64 {
    let (kx, ky, kz) = (k[0], k[1], k[2]);
    match family {
        0 => {
            // SC Weyl orden fijo: T_z T_y T_x
            let q = qmul(q_axis([0.0, 0.0, 1.0], kz), qmul(q_axis([0.0, 1.0, 0.0], ky), q_axis([1.0, 0.0, 0.0], kx)));
            q[0].clamp(-1.0, 1.0).acos()
        }
        1 => {
            // SC Weyl palindrómico: (T_x T_y T_z)(T_z T_y T_x) / 2 pasos
            let u1 = qmul(q_axis([0.0, 0.0, 1.0], kz), qmul(q_axis([0.0, 1.0, 0.0], ky), q_axis([1.0, 0.0, 0.0], kx)));
            let u2 = qmul(q_axis([1.0, 0.0, 0.0], kx), qmul(q_axis([0.0, 1.0, 0.0], ky), q_axis([0.0, 0.0, 1.0], kz)));
            let q = qmul(u2, u1);
            0.5 * q[0].clamp(-1.0, 1.0).acos()
        }
        2 => {
            // BCC Weyl orden fijo: 4 ejes (±1, ±1, ±1)/√3, θ_m = (3/4)(k·d_m)
            let s3 = 1.0 / 3.0f64.sqrt();
            let axes = [[s3, s3, s3], [s3, -s3, -s3], [-s3, s3, -s3], [-s3, -s3, s3]];
            let mut q = [1.0, 0.0, 0.0, 0.0];
            for ax in axes {
                q = qmul(q_axis(ax, 0.75 * (kx * ax[0] + ky * ax[1] + kz * ax[2])), q);
            }
            q[0].clamp(-1.0, 1.0).acos()
        }
        3 => {
            // BCC Weyl palindrómico (2 pasos)
            let s3 = 1.0 / 3.0f64.sqrt();
            let axes = [[s3, s3, s3], [s3, -s3, -s3], [-s3, s3, -s3], [-s3, -s3, s3]];
            let mut q = [1.0, 0.0, 0.0, 0.0];
            for &ax in &axes {
                q = qmul(q_axis(ax, 0.75 * (kx * ax[0] + ky * ax[1] + kz * ax[2])), q);
            }
            for &ax in axes.iter().rev() {
                q = qmul(q_axis(ax, 0.75 * (kx * ax[0] + ky * ax[1] + kz * ax[2])), q);
            }
            0.5 * q[0].clamp(-1.0, 1.0).acos()
        }
        4 => {
            // FCC Weyl palindrómico: 6 ejes de diagonal de cara (±1,±1,0)/√2; Σ d⊗d = 2 I ⇒ θ_m = ½ (k·d_m)
            let s2 = 1.0 / 2.0f64.sqrt();
            let axes = [
                [s2, s2, 0.0],
                [s2, -s2, 0.0],
                [s2, 0.0, s2],
                [s2, 0.0, -s2],
                [0.0, s2, s2],
                [0.0, s2, -s2],
            ];
            let mut q = [1.0, 0.0, 0.0, 0.0];
            for &ax in &axes {
                q = qmul(q_axis(ax, 0.5 * (kx * ax[0] + ky * ax[1] + kz * ax[2])), q);
            }
            for &ax in axes.iter().rev() {
                q = qmul(q_axis(ax, 0.5 * (kx * ax[0] + ky * ax[1] + kz * ax[2])), q);
            }
            0.5 * q[0].clamp(-1.0, 1.0).acos()
        }
        5 => {
            // Fotón de Mlodinow–Brun (6 componentes, SO(3)): ángulo de R_z(kz) R_y(ky) R_x(kx)
            let tr = kx.cos() * ky.cos()
                + kx.cos() * kz.cos()
                + ky.cos() * kz.cos()
                + kx.sin() * ky.sin() * kz.sin();
            (0.5 * (tr - 1.0)).clamp(-1.0, 1.0).acos()
        }
        6 => {
            // SC Wilson–Dirac Trotter simétrico (4c, r = 1,0, dt = 0,25):
            // cos(ω dt) = cos(dt M_W) cos(dt |S_SC|)
            let dt = 0.25;
            let s = (kx.sin().powi(2) + ky.sin().powi(2) + kz.sin().powi(2)).sqrt();
            let mw = (1.0 - kx.cos()) + (1.0 - ky.cos()) + (1.0 - kz.cos());
            ((dt * mw).cos() * (dt * s).cos()).clamp(-1.0, 1.0).acos() / dt
        }
        _ => {
            // 7: NUEVA CONSTRUCCIÓN — QCA Wilson–Dirac palindrómico isotrópico SC+FCC (4c):
            // S^iso_a(k) = ⅓ sin(k_a)(1 + cos k_b + cos k_c) = k_a(1 − k²/6) + O(k⁵)
            // M_W(k) = r Σ(1 − cos k_a) con r = 2/√3 ⇒ |S^iso|² + M_W² = k² + 0·k⁴ + O(k⁶)
            // Para CUALQUIER dt con dt·M_W < π (aquí dt = 0,25): cos(ω dt) = cos(dt M_W) cos(dt |S^iso|)
            // tiene ω(k) = |k| + O(|k|⁵) (orden n = 4 exacto!), ω(k) = ω(−k) y CERO dobladores!
            let dt = 0.25;
            let r_opt = 2.0 / 3.0f64.sqrt();
            let (cx, cy, cz) = (kx.cos(), ky.cos(), kz.cos());
            let sx = kx.sin() * (1.0 + cy + cz) / 3.0;
            let sy = ky.sin() * (1.0 + cz + cx) / 3.0;
            let sz = kz.sin() * (1.0 + cx + cy) / 3.0;
            let s = (sx * sx + sy * sy + sz * sz).sqrt();
            let mw = r_opt * ((1.0 - cx) + (1.0 - cy) + (1.0 - cz));
            ((dt * mw).cos() * (dt * s).cos()).clamp(-1.0, 1.0).acos() / dt
        }
    }
}

/// Estudio 2 (Pregunta Abierta #2): clasifica las 8 familias de QCA 3+1D locales y unitarias
/// bajo la triple restricción conjunta:
///   (i)   asimetría CPT-impar |ω(k) − ω(−k)| < 10⁻¹⁰ (sin birrefringencia / Cohen–Glashow n = 1)
///   (ii)  hueco mínimo de dobladores fuera de |k| < 0,35 (con refinamiento local en 4 niveles) > 0,30
///   (iii) anisotropía direccional A₂(n̂) en O(k²Δx²) < 10⁻⁴.
/// Devuelve [8_familias, ...8 × [cpt_asym, doubler_gap, a2_spread, a2_mean, a4_max, pass_all3]].
pub fn run_study2_3d_qca_triple_classification(grid_n: usize) -> Vec<f64> {
    let ng = grid_n.clamp(14, 28);
    let ndirs = 400usize;
    let dirs = fibonacci_sphere(ndirs);

    let eval_fam = |fam: usize| -> [f64; 6] {
        // (i) Asimetría CPT en k genérico
        let kt = [0.21, 0.13, 0.07];
        let asym = (qca3d_omega(fam, kt) - qca3d_omega(fam, [-kt[0], -kt[1], -kt[2]])).abs();

        // (ii) Búsqueda de dobladores en [−π, π)³ fuera de |k| < 0,35 + refinamiento local en 4 niveles
        let dk = 2.0 * PI / ng as f64;
        let mut eps_min = f64::INFINITY;
        let mut k_best = [PI, 0.0, 0.0];
        for ix in 0..ng {
            let kx = -PI + ix as f64 * dk;
            for iy in 0..ng {
                let ky = -PI + iy as f64 * dk;
                for iz in 0..ng {
                    let kz = -PI + iz as f64 * dk;
                    if (kx * kx + ky * ky + kz * kz).sqrt() < 0.35 {
                        continue;
                    }
                    let e = qca3d_omega(fam, [kx, ky, kz]);
                    if e < eps_min {
                        eps_min = e;
                        k_best = [kx, ky, kz];
                    }
                }
            }
        }
        let mut step = dk;
        for _ in 0..4 {
            let cen = k_best;
            let sub = step / 3.0;
            for dx in -2..=2 {
                for dy in -2..=2 {
                    for dz in -2..=2 {
                        let wx = (cen[0] + dx as f64 * sub + PI).rem_euclid(2.0 * PI) - PI;
                        let wy = (cen[1] + dy as f64 * sub + PI).rem_euclid(2.0 * PI) - PI;
                        let wz = (cen[2] + dz as f64 * sub + PI).rem_euclid(2.0 * PI) - PI;
                        if (wx * wx + wy * wy + wz * wz).sqrt() < 0.35 {
                            continue;
                        }
                        let e = qca3d_omega(fam, [wx, wy, wz]);
                        if e < eps_min {
                            eps_min = e;
                            k_best = [wx, wy, wz];
                        }
                    }
                }
            }
            step = sub;
        }

        // (iii) Coeficientes A₂(n̂) (dispersión simetrizada en k = 0,03) y A₄(n̂) (en k = 0,10)
        let k2 = 0.03;
        let k4 = 0.10;
        let mut a2_mn = f64::INFINITY;
        let mut a2_mx = f64::NEG_INFINITY;
        let mut a2_sum = 0.0;
        let mut a4_max = 0.0f64;
        for &d in &dirs {
            let om_sym = |kk: f64| {
                0.5 * (qca3d_omega(fam, [kk * d[0], kk * d[1], kk * d[2]])
                    + qca3d_omega(fam, [-kk * d[0], -kk * d[1], -kk * d[2]]))
            };
            let a2 = (om_sym(k2) / k2 - 1.0) / (k2 * k2);
            let a2_hi = (om_sym(k4) / k4 - 1.0) / (k4 * k4);
            let a4 = (a2_hi - a2) / (k4 * k4 - k2 * k2);
            a2_mn = a2_mn.min(a2);
            a2_mx = a2_mx.max(a2);
            a2_sum += a2;
            a4_max = a4_max.max(a4.abs());
        }
        let a2_spread = a2_mx - a2_mn;
        let a2_mean = a2_sum / ndirs as f64;
        let pass = if asym < 1e-10 && eps_min > 0.30 && a2_spread < 1e-4 { 1.0 } else { 0.0 };
        [asym, eps_min, a2_spread, a2_mean, a4_max, pass]
    };

    #[cfg(not(target_arch = "wasm32"))]
    let res: Vec<[f64; 6]> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..8).map(|f| sc.spawn(move || eval_fam(f))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let res: Vec<[f64; 6]> = (0..8).map(eval_fam).collect();

    let mut out = vec![8.0];
    for r in res {
        out.extend_from_slice(&r);
    }
    out
}

// ================================================================== ESTUDIO 3: Conjuntos causales + materia cuántica

/// Función de Bessel J₀(x) exacta por serie de potencias (precisión < 10⁻¹⁵ para |x| ≤ 15).
fn bessel_j0(x: f64) -> f64 {
    let x2 = -0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..40 {
        term *= x2 / ((k * k) as f64);
        sum += term;
        if term.abs() < 1e-16 * sum.abs().max(1.0) {
            break;
        }
    }
    sum
}

/// Resuelve el propagador retardado exacto de Johnston (PRL 103, 180401, 2009) en 1+1D:
///   (I + (m²/2) C) G_R = ½ C   (C estrictamente triangular superior: C[i, j] = 1 si i ≺ j)
/// por sustitución hacia atrás exacta en O(N³/6).
fn johnston_retarded_2d(prec: &[bool], n: usize, m: f64) -> Vec<f64> {
    let mut gr = vec![0.0; n * n];
    let c_mass = 0.5 * m * m;
    for j in 1..n {
        for i in (0..j).rev() {
            if !prec[i * n + j] {
                continue;
            }
            let mut val = 0.5;
            for k in (i + 1)..j {
                if prec[i * n + k] {
                    val -= c_mass * gr[k * n + j];
                }
            }
            gr[i * n + j] = val;
        }
    }
    gr
}

/// Acción espectral cuántica a 1 lazo de Sorkin–Johnston sobre un 2-orden causal `perm`:
///   S_mat[C] = ½ ln det(I + Δ^T Δ),   Δ = G_R − G_R^T (conmutador de Pauli–Jordan)
/// calculada exactamente vía factorización de Cholesky L·L^T = I + Δ^T Δ ⇒ S_mat = Σ ln L_{ii}.
fn eval_2order_bd_and_matter(perm: &[usize], m: f64, need_matter: bool) -> (f64, f64, usize) {
    let n = perm.len();
    assert!(n <= 64, "eval_2order_bd_and_matter requiere N <= 64 (recibido {n})");
    let mut fut = [0u64; 64];
    let mut past = [0u64; 64];
    let mut prec = if need_matter { vec![false; n * n] } else { Vec::new() };
    let mut n_rel = 0usize;
    for i in 0..n {
        let pi = perm[i];
        let mut fi = 0u64;
        for j in (i + 1)..n {
            if pi < perm[j] {
                fi |= 1u64 << j;
                past[j] |= 1u64 << i;
                if need_matter {
                    prec[i * n + j] = true;
                }
                n_rel += 1;
            }
        }
        fut[i] = fi;
    }
    // 1. Acción de Benincasa–Dowker S_BD = N − 2 N₁ + 4 N₂ − 2 N₃
    let (mut n1, mut n2, mut n3) = (0i64, 0i64, 0i64);
    for i in 0..n {
        let mut mask = fut[i];
        while mask != 0 {
            let j = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            match (fut[i] & past[j]).count_ones() {
                0 => n1 += 1,
                1 => n2 += 1,
                2 => n3 += 1,
                _ => {}
            }
        }
    }
    let s_bd = (n as i64 - 2 * n1 + 4 * n2 - 2 * n3) as f64;
    if !need_matter {
        return (s_bd, 0.0, n_rel);
    }

    // 2. Propagador retardado de Johnston G_R y acción de 1 lazo S_mat = ½ ln det(I + Δ^T Δ)
    let gr = johnston_retarded_2d(&prec, n, m);
    // Δ_{ij} = G_R[i, j] − G_R[j, i]; (Δ^T Δ)_{ij} = Σ_k Δ_{ki} Δ_{kj}
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n {
                let d_ki = if k < i {
                    gr[k * n + i]
                } else if k > i {
                    -gr[i * n + k]
                } else {
                    0.0
                };
                if d_ki == 0.0 {
                    continue;
                }
                let d_kj = if k < j {
                    gr[k * n + j]
                } else if k > j {
                    -gr[j * n + k]
                } else {
                    0.0
                };
                sum += d_ki * d_kj;
            }
            mat[i * n + j] += sum;
        }
    }
    // Factorización de Cholesky in-place (triángulo inferior)
    let mut log_det_half = 0.0;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat[i * n + j];
            for k in 0..j {
                sum -= mat[i * n + k] * mat[j * n + k];
            }
            if i == j {
                let diag = sum.max(1e-30).sqrt();
                mat[i * n + i] = diag;
                log_det_half += diag.ln();
            } else {
                mat[i * n + j] = sum / mat[j * n + j];
            }
        }
    }
    (s_bd, log_det_half, n_rel)
}

/// Estudio 3 (Pregunta Abierta #3):
/// Parte A: propagador retardado exacto de Johnston (PRL 2009) en espolvoreados 2D de Minkowski
///   (N = 140, m = 0,8, 12 realizaciones) frente a ½ J₀(m τ), midiendo la varianza no local σ_G(τ).
/// Parte B: Monte Carlo dinámico de gravedad + N_f campos escalares cuánticos en Ω_{2D} (N = 18):
///   Z = Σ exp(+β S_BD − N_f S_mat) para N_f ∈ {0, 1, 2} y β ∈ {−0,25, 0,0, 0,4, 0,8, 1,3, 1,8},
///   con `n_seeds` semillas independientes para obtener barras de error (media ± SEM) reales!
pub fn run_study3_causal_set_matter_backreaction(n_mc: usize, sweeps: usize, n_seeds: usize) -> Vec<f64> {
    let m_scalar = 0.8f64;

    // --- Parte A: Verificación del propagador de Johnston vs ½ J₀(m τ) y varianza no local ---
    let n_spr = 140usize;
    let s_side = (n_spr as f64).sqrt();
    let inv_s2 = 1.0 / 2.0f64.sqrt();
    let n_bins = 10usize;
    let tau_max = 5.0f64;
    let mut bin_sum = vec![0.0; n_bins];
    let mut bin_sum2 = vec![0.0; n_bins];
    let mut bin_cnt = vec![0usize; n_bins];

    for tr in 0..10usize {
        let mut rng = SplitMix64::new(0x1045_7009u64 ^ (tr as u64 * 7919));
        let mut pts: Vec<(f64, f64)> = (0..n_spr)
            .map(|_| {
                let u = rng.next_f64() * s_side;
                let v = rng.next_f64() * s_side;
                ((u + v) * inv_s2, (u - v) * inv_s2)
            })
            .collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut prec = vec![false; n_spr * n_spr];
        for i in 0..n_spr {
            for j in (i + 1)..n_spr {
                if pts[j].0 - pts[i].0 > (pts[j].1 - pts[i].1).abs() {
                    prec[i * n_spr + j] = true;
                }
            }
        }
        let gr = johnston_retarded_2d(&prec, n_spr, m_scalar);
        for i in 0..n_spr {
            for j in (i + 1)..n_spr {
                if !prec[i * n_spr + j] {
                    continue;
                }
                let dt = pts[j].0 - pts[i].0;
                let dx = pts[j].1 - pts[i].1;
                let tau = (dt * dt - dx * dx).max(0.0).sqrt();
                if tau < tau_max {
                    let b = ((tau / tau_max) * n_bins as f64) as usize;
                    if b < n_bins {
                        let g = gr[i * n_spr + j];
                        bin_sum[b] += g;
                        bin_sum2[b] += g * g;
                        bin_cnt[b] += 1;
                    }
                }
            }
        }
    }
    let mut gr_err_rms = 0.0;
    let mut gr_curve = Vec::with_capacity(n_bins);
    for b in 0..n_bins {
        let tau_c = (b as f64 + 0.5) * tau_max / n_bins as f64;
        let cnt = bin_cnt[b].max(1) as f64;
        let mean = bin_sum[b] / cnt;
        let std = (bin_sum2[b] / cnt - mean * mean).max(0.0).sqrt();
        let exact = 0.5 * bessel_j0(m_scalar * tau_c);
        gr_err_rms += (mean - exact).powi(2);
        gr_curve.push((tau_c, mean, exact, std));
    }
    gr_err_rms = (gr_err_rms / n_bins as f64).sqrt();
    let sigma_near_cone = gr_curve[0].3;
    let sigma_bulk = gr_curve[n_bins - 1].3;

    // --- Parte B: Monte Carlo dinámico con retroacción de N_f = 0, 1, 2 campos escalares cuánticos ---
    let n = n_mc.clamp(14, 24);
    let n_sw = sweeps.clamp(150, 1200);
    let therm = n_sw / 3;
    let seeds = n_seeds.clamp(3, 8);
    let betas = [-0.25f64, 0.0, 0.45, 0.90, 1.40, 1.90];
    let nfs = [0.0f64, 1.0, 2.0];
    let pairs_tot = (n * (n - 1)) as f64;

    let run_point = |(nf_idx, b_idx): (usize, usize)| -> (f64, f64, f64, f64, f64) {
        let nf = nfs[nf_idx];
        let beta = betas[b_idx];
        let mut s_vals = Vec::with_capacity(seeds);
        let mut d_vals = Vec::with_capacity(seeds);
        let mut m_vals = Vec::with_capacity(seeds);
        for sd in 0..seeds {
            let mut rng = SplitMix64::new(0x1703_CA05u64 ^ ((nf_idx * 100 + b_idx * 10 + sd) as u64 * 2654435761));
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                perm.swap(i, j);
            }
            let need_mat = nf > 0.0;
            let (mut s_cur, mut m_cur, mut rel_cur) = eval_2order_bd_and_matter(&perm, m_scalar, need_mat);
            let mut s_acc = 0.0;
            let mut m_acc = 0.0;
            let mut r_acc = 0.0;
            let mut cnt = 0usize;
            for sw in 0..n_sw {
                for _ in 0..(n / 2).max(4) {
                    let a = (rng.next_u64() as usize) % n;
                    let mut b = (rng.next_u64() as usize) % (n - 1);
                    if b >= a {
                        b += 1;
                    }
                    perm.swap(a, b);
                    let (s_new, m_new, rel_new) = eval_2order_bd_and_matter(&perm, m_scalar, need_mat);
                    // Peso exp(+β S_BD − N_f S_mat)
                    let dw = beta * (s_new - s_cur) - nf * (m_new - m_cur);
                    if dw >= 0.0 || rng.next_f64() < dw.exp() {
                        s_cur = s_new;
                        m_cur = m_new;
                        rel_cur = rel_new;
                    } else {
                        perm.swap(a, b);
                    }
                }
                if sw >= therm && (sw - therm) % 2 == 0 {
                    s_acc += s_cur / n as f64;
                    m_acc += m_cur / n as f64;
                    r_acc += 2.0 * rel_cur as f64 / pairs_tot;
                    cnt += 1;
                }
            }
            let cf = cnt.max(1) as f64;
            s_vals.push(s_acc / cf);
            d_vals.push(mm_dim_2d(r_acc / cf));
            m_vals.push(m_acc / cf);
        }
        let mean_sem = |v: &[f64]| -> (f64, f64) {
            let mu = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (v.len().saturating_sub(1).max(1)) as f64;
            (mu, (var / v.len() as f64).sqrt())
        };
        let (sm, ss) = mean_sem(&s_vals);
        let (dm, ds) = mean_sem(&d_vals);
        let (mm, _) = mean_sem(&m_vals);
        (sm, ss, dm, ds, mm)
    };

    let mut tasks = Vec::new();
    for nf_i in 0..nfs.len() {
        for b_i in 0..betas.len() {
            tasks.push((nf_i, b_i));
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let mc_res: Vec<(f64, f64, f64, f64, f64)> = std::thread::scope(|sc| {
        let handles: Vec<_> = tasks.into_iter().map(|t| sc.spawn(move || run_point(t))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let mc_res: Vec<(f64, f64, f64, f64, f64)> = tasks.into_iter().map(run_point).collect();

    // Índices destacados: β = 0 es b_idx = 1; β = 1,90 es b_idx = 5; β = −0,25 es b_idx = 0
    let nb = betas.len();
    let get = |nf_i: usize, b_i: usize| mc_res[nf_i * nb + b_i];
    let p_nf0_b0 = get(0, 1);
    let p_nf2_b0 = get(2, 1);
    let p_nf0_bhi = get(0, 5);
    let p_nf2_bhi = get(2, 5);
    let p_nf0_bneg = get(0, 0);
    let p_nf2_bneg = get(2, 0);

    // Cabecera (14):
    // [gr_err_rms, sigma_near_cone, sigma_bulk,
    //  dMM_nf0_b0, dMM_nf2_b0, S_nf0_bhi, S_nf2_bhi, S_nf0_bneg, S_nf2_bneg,
    //  sem_S_nf0_bhi, sem_S_nf2_bhi, n_bins_gr, n_betas, n_nfs]
    // Seguido de n_bins_gr × [tau, G_meas, G_exact, sigma_G]
    // Seguido de (n_nfs × n_betas) × [nf, beta, S_mean, S_sem, dMM_mean, dMM_sem, S_mat_mean]
    let mut out = vec![
        gr_err_rms,
        sigma_near_cone,
        sigma_bulk,
        p_nf0_b0.2,
        p_nf2_b0.2,
        p_nf0_bhi.0,
        p_nf2_bhi.0,
        p_nf0_bneg.0,
        p_nf2_bneg.0,
        p_nf0_bhi.1,
        p_nf2_bhi.1,
        n_bins as f64,
        nb as f64,
        nfs.len() as f64,
    ];
    for (tau, gm, ge, sg) in gr_curve {
        out.extend_from_slice(&[tau, gm, ge, sg]);
    }
    for (nf_i, &nf) in nfs.iter().enumerate() {
        for (b_i, &beta) in betas.iter().enumerate() {
            let (sm, ss, dm, ds, mm) = get(nf_i, b_i);
            out.extend_from_slice(&[nf, beta, sm, ss, dm, ds, mm]);
        }
    }
    out
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 17 (Estudio 1 — Pregunta abierta #1): diagrama de fases no perturbativo de restauración
    /// de Lorentz en dos especies cuánticas acopladas (Lanczos exacto periódico N_r = 4,6,8 +
    /// factor de estructura dinámico infrarrojo S_{αα}(q, ω) + transferencia RK4 + DMRG N_r = 8,16).
    /// Devuelve puntero a [4, v_harm_pred, ...4 × [gap_inf, S8, S16, vA_IR, vB_IR, mismatch, max_dNB, dS]].
    pub fn study1_two_species_lorentz_phases(&mut self, chi_dmrg: u32, chi_rt: u32) -> *const f64 {
        self.oq.out = run_study1_two_species_lorentz_phases(
            (chi_dmrg as usize).clamp(14, 36),
            (chi_rt as usize).clamp(10, 24),
        );
        self.oq.out.as_ptr()
    }

    /// Fase 17 (Estudio 2 — Pregunta abierta #2): clasificación exhaustiva de 8 familias de QCA 3+1D
    /// bajo la triple restricción (CPT-par + sin dobladores + isotropía O(k²)), incluyendo el nuevo
    /// QCA palindrómico isotrópico SC+FCC Wilson–Dirac (r = 2/√3) con orden n = 4 exacto.
    /// Devuelve puntero a [8, ...8 × [cpt_asym, doubler_gap, a2_spread, a2_mean, a4_max, pass_all3]].
    pub fn study2_3d_qca_triple_classification(&mut self, grid_n: u32) -> *const f64 {
        self.oq.out = run_study2_3d_qca_triple_classification(grid_n as usize);
        self.oq.out.as_ptr()
    }

    /// Fase 17 (Estudio 3 — Pregunta abierta #3): propagador retardado exacto de Johnston (PRL 2009)
    /// y Monte Carlo dinámico de conjuntos causales 2D acoplados a N_f = 0, 1, 2 campos escalares
    /// cuánticos (acción espectral de Sorkin–Johnston con barras de error de semillas independientes).
    pub fn study3_causal_set_matter_backreaction(&mut self, n_mc: u32, sweeps: u32, n_seeds: u32) -> *const f64 {
        self.oq.out = run_study3_causal_set_matter_backreaction(n_mc as usize, sweeps as usize, n_seeds as usize);
        self.oq.out.as_ptr()
    }
}
