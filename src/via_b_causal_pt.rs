//! Fase 18 — Vía B: Gravedad cuántica de conjuntos causales 2D acoplada a materia cuántica de
//! Sorkin–Johnston y control de Ising mediante *Parallel Tempering* (intercambio de réplicas en β),
//! soporte hasta N = 128 elementos con máscaras `u128`, empaquetado de columnas para Cholesky rápido,
//! diagnóstico riguroso (τ_int de Madras–Sokal, *block bootstrap*, susceptibilidad χ_S, calor
//! específico C_v y cumulante de Binder) y escalado de tamaño finito β_c(N, N_f).

use crate::research_io::{block_bootstrap_mean_err, integrated_autocorr_time, lin_fit_r2, susceptibility_cv_binder};
#[cfg(not(target_arch = "wasm32"))]
use crate::research_io::{read_research_file, write_research_file};
use crate::rng::SplitMix64;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Evalúa en un 2-orden `perm` (hasta N = 128 elementos con `u128`):
///   - S_BD = N − 2 N₁ + 4 N₂ − 2 N₃ (acción exacta de Benincasa–Dowker),
///   - número de relaciones causales `n_rel`,
///   - acción cuántica de 1 lazo de Sorkin–Johnston S_mat = ½ ln det(I + Δ^T Δ) (solo si `need_matter`).
fn eval_2order_fast(perm: &[usize], m_scalar: f64, need_matter: bool) -> (f64, f64, usize) {
    let n = perm.len();
    let mut fut = [0u128; 128];
    let mut past = [0u128; 128];
    let mut n_rel = 0usize;
    for i in 0..n {
        let pi = perm[i];
        let mut fi = 0u128;
        for j in (i + 1)..n {
            if pi < perm[j] {
                fi |= 1u128 << j;
                past[j] |= 1u128 << i;
                n_rel += 1;
            }
        }
        fut[i] = fi;
    }
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

    // 1. Propagador retardado exacto de Johnston (PRL 2009): (I + ½ m² C) G_R = ½ C
    // Aprovechamos `fut[i] & past[j]` para iterar SOLO sobre k con i ≺ k ≺ j!
    let mut gr = vec![0.0; n * n];
    let c_mass = 0.5 * m_scalar * m_scalar;
    for j in 1..n {
        let pj = past[j];
        for i in (0..j).rev() {
            if ((pj >> i) & 1) == 0 {
                continue;
            }
            let mut val = 0.5;
            let mut mid = fut[i] & pj;
            while mid != 0 {
                let k = mid.trailing_zeros() as usize;
                mid &= mid - 1;
                val -= c_mass * gr[k * n + j];
            }
            gr[i * n + j] = val;
        }
    }

    // 2. Empaquetado por columnas de Δ = G_R − G_R^T: d_col[i * n + k] = Δ[k, i]
    let mut d_col = vec![0.0; n * n];
    for i in 0..n {
        let mut mask = fut[i];
        while mask != 0 {
            let j = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let g = gr[i * n + j];
            d_col[j * n + i] = g;
            d_col[i * n + j] = -g;
        }
    }

    // 3. M = I + Δ^T Δ (triángulo inferior) y factorización de Cholesky in-place
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        let ci = &d_col[(i * n)..((i + 1) * n)];
        for j in 0..=i {
            let cj = &d_col[(j * n)..((j + 1) * n)];
            let mut sum = if i == j { 1.0 } else { 0.0 };
            for k in 0..n {
                sum += ci[k] * cj[k];
            }
            mat[i * n + j] = sum;
        }
    }
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

/// Evalúa S_BD, número de relaciones y energía de Ising sobre enlaces causales (N₁) para el control
/// de materia clásica de Ising (benchmark estilo Glaser–Surya).
fn eval_2order_with_ising(perm: &[usize], spins: &[i8]) -> (f64, f64, usize) {
    let n = perm.len();
    let mut fut = [0u128; 128];
    let mut past = [0u128; 128];
    let mut n_rel = 0usize;
    for i in 0..n {
        let pi = perm[i];
        let mut fi = 0u128;
        for j in (i + 1)..n {
            if pi < perm[j] {
                fi |= 1u128 << j;
                past[j] |= 1u128 << i;
                n_rel += 1;
            }
        }
        fut[i] = fi;
    }
    let (mut n1, mut n2, mut n3) = (0i64, 0i64, 0i64);
    let mut e_ising = 0i64;
    for i in 0..n {
        let si = spins[i] as i64;
        let mut mask = fut[i];
        while mask != 0 {
            let j = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            match (fut[i] & past[j]).count_ones() {
                0 => {
                    n1 += 1;
                    e_ising -= si * spins[j] as i64;
                }
                1 => n2 += 1,
                2 => n3 += 1,
                _ => {}
            }
        }
    }
    let s_bd = (n as i64 - 2 * n1 + 4 * n2 - 2 * n3) as f64;
    (s_bd, e_ising as f64, n_rel)
}

fn mm_dim_2d(r_frac: f64) -> f64 {
    let f = |d: f64| -> f64 {
        let lgamma = |z: f64| -> f64 {
            let p = [
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7,
            ];
            let mut x = 0.99999999999980993;
            let zm1 = z - 1.0;
            for (k, &c) in p.iter().enumerate() {
                x += c / (zm1 + (k + 1) as f64);
            }
            let t = zm1 + 7.5;
            0.5 * (2.0 * PI).ln() + (zm1 + 0.5) * t.ln() - t + x.ln()
        };
        (lgamma(d + 1.0) + lgamma(0.5 * d) - 2.0f64.ln() - lgamma(1.5 * d)).exp()
    };
    if r_frac <= 0.01 {
        return 6.0;
    }
    if r_frac >= 0.999 {
        return 1.0;
    }
    let (mut lo, mut hi) = (1.0, 6.0);
    for _ in 0..45 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > r_frac {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Resultado de un punto de la escalera de *Parallel Tempering* (1 réplica β_m).
#[derive(Clone, Debug)]
pub struct PtReplicaResult {
    pub beta: f64,
    pub s_mean: f64,
    pub s_err: f64,
    pub dmm_mean: f64,
    pub dmm_err: f64,
    pub chi_s: f64,
    pub cv: f64,
    pub binder: f64,
    pub tau_int: f64,
    pub mat_mean: f64,
}

/// Resultado de una corrida completa de *Parallel Tempering* para un (N, modo de materia).
#[derive(Clone, Debug)]
pub struct PtRunResult {
    pub n_elem: usize,
    pub matter_mode: i32, // 0 = gravedad pura, 1 = SJ N_f=1, 2 = SJ N_f=2, -1 = Ising J=0,45
    pub swap_acceptance: f64,
    pub min_pair_swap: f64,
    pub beta_c_est: f64,
    pub chi_peak: f64,
    pub beta_c_main: f64,
    pub chi_peak_main: f64,
    pub binder_min_main: f64,
    pub beta_zero_idx: usize,
    pub replicas: Vec<PtReplicaResult>,
}

/// Corre *Parallel Tempering* sobre `betas` con balance detallado exacto de intercambio de réplicas:
///   - `matter_mode == 0`: gravedad pura (N_f = 0)
///   - `matter_mode == 1, 2`: materia cuántica de Sorkin–Johnston con N_f = 1, 2 (m = 0,8)
///   - `matter_mode == -1`: materia clásica de Ising sobre enlaces causales (J_Ising = 0,45)
pub fn run_parallel_tempering_2d(
    n_elem: usize,
    betas: &[f64],
    sweeps: usize,
    matter_mode: i32,
    seed: u64,
) -> PtRunResult {
    let n = n_elem.clamp(12, 96);
    let m_rep = betas.len();
    let therm = sweeps / 3;
    let m_scalar = 0.8f64;
    let nf = if matter_mode > 0 { matter_mode as f64 } else { 0.0 };
    let is_ising = matter_mode < 0;
    let j_ising = 0.45f64;
    let pairs_tot = (n * (n - 1)) as f64;

    let mut rng = SplitMix64::new(seed ^ ((n as u64) << 24) ^ ((matter_mode as u64).wrapping_mul(0x9E37_79B9)));
    let mut perms: Vec<Vec<usize>> = Vec::with_capacity(m_rep);
    let mut spins: Vec<Vec<i8>> = Vec::with_capacity(m_rep);
    let mut s_cur = vec![0.0; m_rep];
    let mut m_cur = vec![0.0; m_rep];
    let mut r_cur = vec![0usize; m_rep];

    for r in 0..m_rep {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            p.swap(i, j);
        }
        let sp: Vec<i8> = (0..n).map(|_| if (rng.next_u64() & 1) == 0 { 1 } else { -1 }).collect();
        let (sb, mt, nr) = if is_ising {
            eval_2order_with_ising(&p, &sp)
        } else {
            eval_2order_fast(&p, m_scalar, matter_mode > 0)
        };
        perms.push(p);
        spins.push(sp);
        s_cur[r] = sb;
        m_cur[r] = mt;
        r_cur[r] = nr;
    }

    let n_meas = sweeps - therm;
    let mut s_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut sn_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut d_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut m_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut swap_tries = 0usize;
    let mut swap_accs = 0usize;
    let mut pair_tries = vec![0usize; m_rep.saturating_sub(1)];
    let mut pair_accs = vec![0usize; m_rep.saturating_sub(1)];

    let moves_per_sweep = (n / 2).max(6);
    for sw in 0..sweeps {
        // 1. Movimientos locales de Metropolis dentro de cada réplica
        for r in 0..m_rep {
            let beta = betas[r];
            for _ in 0..moves_per_sweep {
                let a = (rng.next_u64() as usize) % n;
                let mut b = (rng.next_u64() as usize) % (n - 1);
                if b >= a {
                    b += 1;
                }
                perms[r].swap(a, b);
                let (sn, mn, rn) = if is_ising {
                    eval_2order_with_ising(&perms[r], &spins[r])
                } else {
                    eval_2order_fast(&perms[r], m_scalar, matter_mode > 0)
                };
                let dw = if is_ising {
                    beta * (sn - s_cur[r]) - j_ising * (mn - m_cur[r])
                } else {
                    beta * (sn - s_cur[r]) - nf * (mn - m_cur[r])
                };
                if dw >= 0.0 || rng.next_f64() < dw.exp() {
                    s_cur[r] = sn;
                    m_cur[r] = mn;
                    r_cur[r] = rn;
                } else {
                    perms[r].swap(a, b);
                }
            }
            // Si hay espines de Ising, hacemos además un barrido de giro de espines a orden fijo
            if is_ising {
                for _ in 0..(n / 2) {
                    let idx = (rng.next_u64() as usize) % n;
                    spins[r][idx] = -spins[r][idx];
                    let (_, mn, _) = eval_2order_with_ising(&perms[r], &spins[r]);
                    let dw = -j_ising * (mn - m_cur[r]);
                    if dw >= 0.0 || rng.next_f64() < dw.exp() {
                        m_cur[r] = mn;
                    } else {
                        spins[r][idx] = -spins[r][idx];
                    }
                }
            }
        }

        // 2. Intercambio de réplicas adyacentes (Parallel Tempering con balance detallado exacto)
        // Como todas las réplicas comparten N_f (o J_Ising), la acción de materia se cancela exactamente!
        let start = sw % 2;
        let mut r = start;
        while r + 1 < m_rep {
            let dw = (betas[r + 1] - betas[r]) * (s_cur[r] - s_cur[r + 1]);
            swap_tries += 1;
            pair_tries[r] += 1;
            if dw >= 0.0 || rng.next_f64() < dw.exp() {
                perms.swap(r, r + 1);
                spins.swap(r, r + 1);
                s_cur.swap(r, r + 1);
                m_cur.swap(r, r + 1);
                r_cur.swap(r, r + 1);
                swap_accs += 1;
                pair_accs[r] += 1;
            }
            r += 2;
        }

        if sw >= therm {
            for r in 0..m_rep {
                s_hist[r].push(s_cur[r]);
                sn_hist[r].push(s_cur[r] / n as f64);
                d_hist[r].push(mm_dim_2d(2.0 * r_cur[r] as f64 / pairs_tot));
                m_hist[r].push(m_cur[r] / n as f64);
            }
        }
    }

    let mut replicas = Vec::with_capacity(m_rep);
    let mut best_chi = -1.0f64;
    let mut best_beta_c = 0.0;
    let mut best_chi_main = -1.0f64;
    let mut best_beta_c_main = 0.0;
    let mut binder_min_main = f64::INFINITY;
    let mut beta_zero_idx = 0usize;
    let mut best_zero_dist = f64::INFINITY;
    for r in 0..m_rep {
        let tau = integrated_autocorr_time(&sn_hist[r]);
        let blk = ((2.0 * tau).ceil() as usize).clamp(2, 24);
        let (sm, se) = block_bootstrap_mean_err(&sn_hist[r], blk, 160, seed ^ (r as u64 * 131));
        let (dm, de) = block_bootstrap_mean_err(&d_hist[r], blk, 160, seed ^ (r as u64 * 997));
        let (chi, cv, binder) = susceptibility_cv_binder(&s_hist[r], betas[r], n);
        let mm = m_hist[r].iter().sum::<f64>() / m_hist[r].len().max(1) as f64;
        if betas[r].abs() < best_zero_dist {
            best_zero_dist = betas[r].abs();
            beta_zero_idx = r;
        }
        // El pico de susceptibilidad en la rama positiva β ≥ 0 localiza β_c(N, N_f) de la transición de Surya
        if betas[r] >= 0.0 && chi > best_chi {
            best_chi = chi;
            best_beta_c = betas[r];
        }
        // Ventana principal de la transición variedad ↔ cristalina de Surya: β ∈ [0,00; 0,35]
        if betas[r] >= 0.0 && betas[r] <= 0.35 {
            if chi > best_chi_main {
                best_chi_main = chi;
                best_beta_c_main = betas[r];
            }
            if binder < binder_min_main {
                binder_min_main = binder;
            }
        }
        replicas.push(PtReplicaResult {
            beta: betas[r],
            s_mean: sm,
            s_err: se,
            dmm_mean: dm,
            dmm_err: de,
            chi_s: chi,
            cv,
            binder,
            tau_int: tau,
            mat_mean: mm,
        });
    }

    let min_pair_swap = pair_accs
        .iter()
        .zip(&pair_tries)
        .map(|(&a, &t)| a as f64 / t.max(1) as f64)
        .fold(1.0f64, f64::min);

    PtRunResult {
        n_elem: n,
        matter_mode,
        swap_acceptance: swap_accs as f64 / swap_tries.max(1) as f64,
        min_pair_swap,
        beta_c_est: best_beta_c,
        chi_peak: best_chi,
        beta_c_main: best_beta_c_main,
        chi_peak_main: best_chi_main,
        binder_min_main: if binder_min_main.is_finite() { binder_min_main } else { 0.0 },
        beta_zero_idx,
        replicas,
    }
}

/// Escalera optimizada de 12 temperaturas β que muestrea densamente la ventana crítica β ∈ [0; 0,50].
pub const VIA_B_BETAS: [f64; 12] = [-0.25, -0.10, 0.00, 0.06, 0.12, 0.18, 0.25, 0.34, 0.46, 0.65, 0.95, 1.35];

/// Escalera refinada de 20 réplicas β (Fase 19) concentrada en la transición de Surya β ∈ [0; 0,35]
/// para mantener alta tasa de intercambio incluso en N = 48–64 (índice 4 fijado en β = 0,00).
pub const VIA_B_BETAS_20: [f64; 20] = [
    -0.25, -0.17, -0.10, -0.05, 0.00, 0.03, 0.06, 0.09, 0.12, 0.15,
    0.18, 0.22, 0.27, 0.33, 0.41, 0.52, 0.66, 0.84, 1.08, 1.35,
];

/// Redistribuye los nodos interiores de un subintervalo [b[lo], b[hi]] para igualar las tasas de
/// intercambio locales p_r (estrategia adaptativa de Katzgraber–Trebst), manteniendo extremos fijos.
/// Como −ln(p_r) ∝ Δβ_r · ΔE_r ya integra el salto de energía en cada intervalo, redistribuir
/// uniformemente en la resistencia acumulada (−ln p_r)^0,75 concentra réplicas justo sobre la
/// transición de primer orden de Surya (evitando cuellos de botella de intercambio).
fn adapt_beta_segment(betas: &mut [f64], pair_acc: &[f64], lo: usize, hi: usize) {
    if hi <= lo + 1 {
        return;
    }
    let m = hi - lo;
    let mut cum = vec![0.0; m + 1];
    for k in 0..m {
        let r = lo + k;
        let p = pair_acc[r].clamp(0.01, 0.98);
        let resist = (-p.ln()).powf(0.75) + 0.15;
        cum[k + 1] = cum[k] + resist;
    }
    let tot = cum[m];
    if tot <= 1e-12 {
        return;
    }
    let old = betas[lo..=hi].to_vec();
    for k in 1..m {
        let target = tot * (k as f64) / (m as f64);
        let mut seg = 0usize;
        while seg + 1 < m && cum[seg + 1] < target {
            seg += 1;
        }
        let frac = ((target - cum[seg]) / (cum[seg + 1] - cum[seg]).max(1e-12)).clamp(0.0, 1.0);
        let b_new = old[seg] + frac * (old[seg + 1] - old[seg]);
        // Mezcla amortiguada (50 % nueva, 50 % previa) para estabilidad suave
        betas[lo + k] = 0.5 * old[k] + 0.5 * b_new;
    }
}

/// Serializa un *checkpoint* JSON ligero y reanudable de una corrida PT (solo nativo).
#[cfg(not(target_arch = "wasm32"))]
fn save_pt_checkpoint(
    tag: &str,
    n_elem: usize,
    matter_mode: i32,
    sweeps: usize,
    betas: &[f64],
    perms: &[Vec<usize>],
    spins: &[Vec<i8>],
    res: &PtRunResult,
) {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"n_elem\": {},\n  \"matter_mode\": {},\n  \"sweeps\": {},\n", n_elem, matter_mode, sweeps));
    s.push_str(&format!(
        "  \"swap_acc\": {:.6},\n  \"min_pair_swap\": {:.6},\n  \"beta_c_main\": {:.6},\n  \"chi_peak_main\": {:.6},\n  \"binder_min_main\": {:.6},\n",
        res.swap_acceptance, res.min_pair_swap, res.beta_c_main, res.chi_peak_main, res.binder_min_main
    ));
    s.push_str("  \"betas\": [");
    for (i, b) in betas.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{:.6}", b));
    }
    s.push_str("],\n  \"perms\": [\n");
    for (r, p) in perms.iter().enumerate() {
        s.push_str("    [");
        for (i, v) in p.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&v.to_string());
        }
        s.push_str(if r + 1 < perms.len() { "],\n" } else { "]\n" });
    }
    s.push_str("  ],\n  \"spins\": [\n");
    for (r, sp) in spins.iter().enumerate() {
        s.push_str("    [");
        for (i, v) in sp.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&v.to_string());
        }
        s.push_str(if r + 1 < spins.len() { "],\n" } else { "]\n" });
    }
    s.push_str("  ]\n}\n");
    let _ = write_research_file(&format!("checkpoints/{}.json", tag), &s);
}

/// Intenta cargar las permutaciones y espines iniciales desde un checkpoint existente (solo nativo).
#[cfg(not(target_arch = "wasm32"))]
fn load_pt_checkpoint_states(tag: &str, n_elem: usize, m_rep: usize) -> Option<(Vec<Vec<usize>>, Vec<Vec<i8>>)> {
    let txt = read_research_file(&format!("checkpoints/{}.json", tag))?;
    let parse_matrix = |key: &str| -> Option<Vec<Vec<i64>>> {
        let pos = txt.find(key)?;
        let rest = &txt[pos..];
        let start = rest.find('[')?;
        let mut depth = 0i32;
        let mut rows = Vec::new();
        let mut cur = Vec::new();
        let mut num = String::new();
        for ch in rest[start..].chars() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    if !num.is_empty() {
                        cur.push(num.parse::<i64>().ok()?);
                        num.clear();
                    }
                    if depth == 2 {
                        rows.push(std::mem::take(&mut cur));
                    }
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                ',' => {
                    if !num.is_empty() {
                        cur.push(num.parse::<i64>().ok()?);
                        num.clear();
                    }
                }
                '-' | '0'..='9' if depth == 2 => num.push(ch),
                _ => {}
            }
        }
        Some(rows)
    };
    let p_rows = parse_matrix("\"perms\"")?;
    let s_rows = parse_matrix("\"spins\"")?;
    if p_rows.len() != m_rep || s_rows.len() != m_rep {
        return None;
    }
    let mut perms = Vec::with_capacity(m_rep);
    let mut spins = Vec::with_capacity(m_rep);
    for r in 0..m_rep {
        if p_rows[r].len() != n_elem || s_rows[r].len() != n_elem {
            return None;
        }
        perms.push(p_rows[r].iter().map(|&v| v as usize).collect());
        spins.push(s_rows[r].iter().map(|&v| v as i8).collect());
    }
    Some((perms, spins))
}

/// Fase 19 (Vía B): *Parallel Tempering* con escalera adaptativa de 20 réplicas (Katzgraber–Trebst)
/// fijando β = 0 en `zero_idx` + *checkpoints* JSON reanudables en `datos_investigacion/checkpoints/`.
pub fn run_parallel_tempering_adaptive(
    n_elem: usize,
    base_betas: &[f64],
    zero_idx: usize,
    sweeps: usize,
    matter_mode: i32,
    seed: u64,
    ckpt_tag: Option<&str>,
) -> PtRunResult {
    let n = n_elem.clamp(12, 96);
    let m_rep = base_betas.len();
    let mut betas = base_betas.to_vec();
    let m_scalar = 0.8f64;
    let nf = if matter_mode > 0 { matter_mode as f64 } else { 0.0 };
    let is_ising = matter_mode < 0;
    let j_ising = 0.45f64;
    let pairs_tot = (n * (n - 1)) as f64;
    let mut rng = SplitMix64::new(seed ^ ((n as u64) << 24) ^ ((matter_mode as u64).wrapping_mul(0x9E37_79B9)));

    let (mut perms, mut spins) = {
        #[cfg(not(target_arch = "wasm32"))]
        let loaded = ckpt_tag.and_then(|tag| load_pt_checkpoint_states(tag, n, m_rep));
        #[cfg(target_arch = "wasm32")]
        let loaded: Option<(Vec<Vec<usize>>, Vec<Vec<i8>>)> = None;

        if let Some(st) = loaded {
            st
        } else {
            let mut ps = Vec::with_capacity(m_rep);
            let mut ss = Vec::with_capacity(m_rep);
            for _ in 0..m_rep {
                let mut p: Vec<usize> = (0..n).collect();
                for i in (1..n).rev() {
                    let j = (rng.next_u64() as usize) % (i + 1);
                    p.swap(i, j);
                }
                let sp: Vec<i8> = (0..n).map(|_| if (rng.next_u64() & 1) == 0 { 1 } else { -1 }).collect();
                ps.push(p);
                ss.push(sp);
            }
            (ps, ss)
        }
    };

    let mut s_cur = vec![0.0; m_rep];
    let mut m_cur = vec![0.0; m_rep];
    let mut r_cur = vec![0usize; m_rep];
    for r in 0..m_rep {
        let (sb, mt, nr) = if is_ising {
            eval_2order_with_ising(&perms[r], &spins[r])
        } else {
            eval_2order_fast(&perms[r], m_scalar, matter_mode > 0)
        };
        s_cur[r] = sb;
        m_cur[r] = mt;
        r_cur[r] = nr;
    }

    let moves_per_sweep = (n / 2).max(6);
    let do_sweep = |sw: usize,
                    betas_ref: &[f64],
                    perms: &mut [Vec<usize>],
                    spins: &mut [Vec<i8>],
                    s_cur: &mut [f64],
                    m_cur: &mut [f64],
                    r_cur: &mut [usize],
                    pair_tries: &mut [usize],
                    pair_accs: &mut [usize],
                    rng: &mut SplitMix64| {
        for r in 0..m_rep {
            let beta = betas_ref[r];
            for _ in 0..moves_per_sweep {
                let a = (rng.next_u64() as usize) % n;
                let mut b = (rng.next_u64() as usize) % (n - 1);
                if b >= a {
                    b += 1;
                }
                perms[r].swap(a, b);
                let (sn, mn, rn) = if is_ising {
                    eval_2order_with_ising(&perms[r], &spins[r])
                } else {
                    eval_2order_fast(&perms[r], m_scalar, matter_mode > 0)
                };
                let dw = if is_ising {
                    beta * (sn - s_cur[r]) - j_ising * (mn - m_cur[r])
                } else {
                    beta * (sn - s_cur[r]) - nf * (mn - m_cur[r])
                };
                if dw >= 0.0 || rng.next_f64() < dw.exp() {
                    s_cur[r] = sn;
                    m_cur[r] = mn;
                    r_cur[r] = rn;
                } else {
                    perms[r].swap(a, b);
                }
            }
            if is_ising {
                for _ in 0..(n / 2) {
                    let idx = (rng.next_u64() as usize) % n;
                    spins[r][idx] = -spins[r][idx];
                    let (_, mn, _) = eval_2order_with_ising(&perms[r], &spins[r]);
                    let dw = -j_ising * (mn - m_cur[r]);
                    if dw >= 0.0 || rng.next_f64() < dw.exp() {
                        m_cur[r] = mn;
                    } else {
                        spins[r][idx] = -spins[r][idx];
                    }
                }
            }
        }
        let mut r = sw % 2;
        while r + 1 < m_rep {
            let dw = (betas_ref[r + 1] - betas_ref[r]) * (s_cur[r] - s_cur[r + 1]);
            pair_tries[r] += 1;
            if dw >= 0.0 || rng.next_f64() < dw.exp() {
                perms.swap(r, r + 1);
                spins.swap(r, r + 1);
                s_cur.swap(r, r + 1);
                m_cur.swap(r, r + 1);
                r_cur.swap(r, r + 1);
                pair_accs[r] += 1;
            }
            r += 2;
        }
    };

    // Fase 1: calentamiento y ajuste adaptativo de la escalera β
    let warmup = (sweeps / 5).clamp(40, 220);
    let mut w_tries = vec![0usize; m_rep - 1];
    let mut w_accs = vec![0usize; m_rep - 1];
    for sw in 0..warmup {
        do_sweep(sw, &betas, &mut perms, &mut spins, &mut s_cur, &mut m_cur, &mut r_cur, &mut w_tries, &mut w_accs, &mut rng);
    }
    let w_rates: Vec<f64> = w_accs.iter().zip(&w_tries).map(|(&a, &t)| a as f64 / t.max(1) as f64).collect();
    adapt_beta_segment(&mut betas, &w_rates, 0, zero_idx);
    adapt_beta_segment(&mut betas, &w_rates, zero_idx, m_rep - 1);

    // Fase 2: producción con escalera adaptada (termalización corta + mediciones)
    let therm = (sweeps / 4).max(30);
    let n_meas = sweeps.saturating_sub(therm).max(40);
    let mut s_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut sn_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut d_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut m_hist = vec![Vec::with_capacity(n_meas); m_rep];
    let mut pair_tries = vec![0usize; m_rep - 1];
    let mut pair_accs = vec![0usize; m_rep - 1];

    for sw in 0..(therm + n_meas) {
        do_sweep(sw, &betas, &mut perms, &mut spins, &mut s_cur, &mut m_cur, &mut r_cur, &mut pair_tries, &mut pair_accs, &mut rng);
        if sw >= therm {
            for r in 0..m_rep {
                s_hist[r].push(s_cur[r]);
                sn_hist[r].push(s_cur[r] / n as f64);
                d_hist[r].push(mm_dim_2d(2.0 * r_cur[r] as f64 / pairs_tot));
                m_hist[r].push(m_cur[r] / n as f64);
            }
        }
    }

    let mut replicas = Vec::with_capacity(m_rep);
    let (mut best_chi, mut best_beta_c) = (-1.0f64, 0.0);
    let (mut best_chi_main, mut best_beta_c_main) = (-1.0f64, 0.0);
    let mut binder_min_main = f64::INFINITY;
    for r in 0..m_rep {
        let tau = integrated_autocorr_time(&sn_hist[r]);
        let blk = ((2.0 * tau).ceil() as usize).clamp(2, 24);
        let (sm, se) = block_bootstrap_mean_err(&sn_hist[r], blk, 160, seed ^ (r as u64 * 131));
        let (dm, de) = block_bootstrap_mean_err(&d_hist[r], blk, 160, seed ^ (r as u64 * 997));
        let (chi, cv, binder) = susceptibility_cv_binder(&s_hist[r], betas[r], n);
        let mm = m_hist[r].iter().sum::<f64>() / m_hist[r].len().max(1) as f64;
        if betas[r] >= 0.0 && chi > best_chi {
            best_chi = chi;
            best_beta_c = betas[r];
        }
        if betas[r] >= 0.0 && betas[r] <= 0.35 {
            if chi > best_chi_main {
                best_chi_main = chi;
                best_beta_c_main = betas[r];
            }
            if binder < binder_min_main {
                binder_min_main = binder;
            }
        }
        replicas.push(PtReplicaResult {
            beta: betas[r],
            s_mean: sm,
            s_err: se,
            dmm_mean: dm,
            dmm_err: de,
            chi_s: chi,
            cv,
            binder,
            tau_int: tau,
            mat_mean: mm,
        });
    }
    let tot_acc: usize = pair_accs.iter().sum();
    let tot_try: usize = pair_tries.iter().sum();
    let min_pair_swap = pair_accs
        .iter()
        .zip(&pair_tries)
        .map(|(&a, &t)| a as f64 / t.max(1) as f64)
        .fold(1.0f64, f64::min);

    let res = PtRunResult {
        n_elem: n,
        matter_mode,
        swap_acceptance: tot_acc as f64 / tot_try.max(1) as f64,
        min_pair_swap,
        beta_c_est: best_beta_c,
        chi_peak: best_chi,
        beta_c_main: best_beta_c_main,
        chi_peak_main: best_chi_main,
        binder_min_main: if binder_min_main.is_finite() { binder_min_main } else { 0.0 },
        beta_zero_idx: zero_idx,
        replicas,
    };

    #[cfg(not(target_arch = "wasm32"))]
    if let Some(tag) = ckpt_tag {
        save_pt_checkpoint(tag, n, matter_mode, sweeps, &betas, &perms, &spins, &res);
    }
    #[cfg(target_arch = "wasm32")]
    let _ = ckpt_tag;

    res
}

/// Ajuste de escalado de tamaño finito (FSS) por modelo de materia sobre múltiples tamaños N.
#[derive(Clone, Debug)]
pub struct FssModeFit {
    pub matter_mode: i32,
    pub gamma_chi: f64,      // exponente en χ_pico(N) ∝ N^γ (≈ 1 para transición de 1.er orden)
    pub r2_chi: f64,
    pub dmm_inf: f64,        // extrapolación 1/N de d_MM(β=0, N → ∞)
    pub dmm_slope: f64,      // pendiente b en d_MM(N) = d_∞ + b/N
    pub beta_c_inf: f64,     // extrapolación 1/N de β_c^main(N → ∞)
    pub min_swap_all_n: f64, // tasa de intercambio global mínima observada en el tamaño mayor
    pub binder_min_largest: f64,
}

/// Ejecuta la batería FSS completa con 20 réplicas adaptativas sobre los tamaños `sizes`
/// (con paralelismo total en nativo y checkpoints reanudables opcionales) y extrae los ajustes FSS.
pub fn run_via_b_fss_suite(sizes: &[usize], sweeps_per_size: &[usize], save_ckpt: bool) -> (Vec<PtRunResult>, Vec<FssModeFit>) {
    let modes = [0i32, 1, 2, -1];
    let mut tasks = Vec::with_capacity(sizes.len() * modes.len());
    for (si, &n) in sizes.iter().enumerate() {
        let sw = sweeps_per_size[si.min(sweeps_per_size.len() - 1)];
        for &m in &modes {
            tasks.push((n, m, sw));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    let runs: Vec<PtRunResult> = std::thread::scope(|sc| {
        let handles: Vec<_> = tasks
            .into_iter()
            .enumerate()
            .map(|(idx, (n, m, sw))| {
                sc.spawn(move || {
                    let tag = if save_ckpt {
                        Some(format!("via_b_N{}_m{}", n, if m < 0 { "ising".to_string() } else { m.to_string() }))
                    } else {
                        None
                    };
                    run_parallel_tempering_adaptive(
                        n,
                        &VIA_B_BETAS_20,
                        4,
                        sw,
                        m,
                        0x19B0_2026 ^ (idx as u64 * 104729),
                        tag.as_deref(),
                    )
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let runs: Vec<PtRunResult> = {
        let _ = save_ckpt;
        tasks
            .into_iter()
            .enumerate()
            .map(|(idx, (n, m, sw))| {
                run_parallel_tempering_adaptive(n, &VIA_B_BETAS_20, 4, sw, m, 0x19B0_2026 ^ (idx as u64 * 104729), None)
            })
            .collect()
    };

    let mut fits = Vec::with_capacity(modes.len());
    for &m in &modes {
        let mruns: Vec<&PtRunResult> = runs.iter().filter(|r| r.matter_mode == m).collect();
        let ln_n: Vec<f64> = mruns.iter().map(|r| (r.n_elem as f64).ln()).collect();
        let ln_chi: Vec<f64> = mruns.iter().map(|r| r.chi_peak_main.max(1e-6).ln()).collect();
        let inv_n: Vec<f64> = mruns.iter().map(|r| 1.0 / r.n_elem as f64).collect();
        let dmms: Vec<f64> = mruns.iter().map(|r| r.replicas[r.beta_zero_idx].dmm_mean).collect();
        let bcs: Vec<f64> = mruns.iter().map(|r| r.beta_c_main).collect();
        let (_, gamma_chi, r2_chi) = lin_fit_r2(&ln_n, &ln_chi);
        let (dmm_inf, dmm_slope, _) = lin_fit_r2(&inv_n, &dmms);
        let (beta_c_inf, _, _) = lin_fit_r2(&inv_n, &bcs);
        let min_swap_all_n = mruns.iter().map(|r| r.swap_acceptance).fold(1.0f64, f64::min);
        let binder_min_largest = mruns.last().map(|r| r.binder_min_main).unwrap_or(0.0);
        fits.push(FssModeFit {
            matter_mode: m,
            gamma_chi,
            r2_chi,
            dmm_inf,
            dmm_slope,
            beta_c_inf,
            min_swap_all_n,
            binder_min_largest,
        });
    }
    (runs, fits)
}

/// Versión compacta para WASM / test de la Fase 19 (Vía B): 20 réplicas adaptativas sobre
/// N ∈ {16, 22, 28} × 4 modelos con ajuste explícito de tamaño finito (γ, d_∞, β_c(∞), swap_acc).
pub fn run_via_b_adaptive_fss_compact(sweeps: usize) -> Vec<f64> {
    let sw = sweeps.clamp(160, 600);
    let sizes = [16usize, 22, 28];
    let sws = [sw, sw, (sw * 4 / 5).max(150)];
    let (runs, fits) = run_via_b_fss_suite(&sizes, &sws, false);
    // Salida:
    // Cabecera (2): [n_fits (4), n_runs (12)]
    // 4 × FssModeFit (8 floats): [mode, gamma_chi, r2_chi, dmm_inf, dmm_slope, beta_c_inf, min_swap, binder_min_largest]
    // 12 × RunSummary (8 floats): [N, mode, swap_acc, min_pair_swap, beta_c_main, chi_peak_main, dmm_b0, dmm_b0_err]
    let mut out = vec![fits.len() as f64, runs.len() as f64];
    for f in &fits {
        out.extend_from_slice(&[
            f.matter_mode as f64,
            f.gamma_chi,
            f.r2_chi,
            f.dmm_inf,
            f.dmm_slope,
            f.beta_c_inf,
            f.min_swap_all_n,
            f.binder_min_largest,
        ]);
    }
    for r in &runs {
        let r0 = &r.replicas[r.beta_zero_idx];
        out.extend_from_slice(&[
            r.n_elem as f64,
            r.matter_mode as f64,
            r.swap_acceptance,
            r.min_pair_swap,
            r.beta_c_main,
            r.chi_peak_main,
            r0.dmm_mean,
            r0.dmm_err,
        ]);
    }
    out
}

/// Ejecuta la **Vía B** completa (Parallel Tempering + escalado de tamaño finito + control Ising):
/// Compara gravedad pura (N_f = 0), Sorkin–Johnston (N_f = 1, 2) e Ising sobre enlaces en dos tamaños
/// (`n_small` y `n_large`, p. ej. 16 y 24 en WASM; 20, 28, 36 y 48 en `qsu_cli`).
pub fn run_via_b_causal_pt_scaling(n_small: usize, n_large: usize, sweeps: usize) -> Vec<f64> {
    let sizes = [n_small.clamp(14, 48), n_large.clamp(16, 64)];
    let modes = [0i32, 1, 2, -1]; // 0=puro, 1=SJ1, 2=SJ2, -1=Ising
    let mut tasks = Vec::with_capacity(sizes.len() * modes.len());
    for &n in &sizes {
        for &m in &modes {
            tasks.push((n, m));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    let runs: Vec<PtRunResult> = std::thread::scope(|sc| {
        let handles: Vec<_> = tasks
            .into_iter()
            .enumerate()
            .map(|(idx, (n, m))| {
                sc.spawn(move || {
                    run_parallel_tempering_2d(n, &VIA_B_BETAS, sweeps, m, 0x18B0_2026 ^ (idx as u64 * 7919))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let runs: Vec<PtRunResult> = tasks
        .into_iter()
        .enumerate()
        .map(|(idx, (n, m))| {
            run_parallel_tempering_2d(n, &VIA_B_BETAS, sweeps, m, 0x18B0_2026 ^ (idx as u64 * 7919))
        })
        .collect();

    // Formato de salida compacto:
    // Cabecera (3): [n_runs (8), n_betas (12), n_fields_per_rep (8)]
    // Por cada corrida (8): [n_elem, matter_mode, swap_acc, beta_c_est, chi_peak]
    //   seguido de n_betas × [beta, s_mean, s_err, dmm_mean, dmm_err, chi_s, binder, tau_int]
    let mut out = vec![runs.len() as f64, VIA_B_BETAS.len() as f64, 8.0];
    for run in &runs {
        out.extend_from_slice(&[
            run.n_elem as f64,
            run.matter_mode as f64,
            run.swap_acceptance,
            run.beta_c_est,
            run.chi_peak,
        ]);
        for rep in &run.replicas {
            out.extend_from_slice(&[
                rep.beta,
                rep.s_mean,
                rep.s_err,
                rep.dmm_mean,
                rep.dmm_err,
                rep.chi_s,
                rep.binder,
                rep.tau_int,
            ]);
        }
    }
    out
}

/// Resultado continuo del muestreo multicanónico / Wang–Landau en conjuntos causales 2D.
#[derive(Clone, Debug)]
pub struct WlResult {
    pub n_elem: usize,
    pub nf: usize,
    pub m_scalar: f64,
    pub beta_c_wl: f64,
    pub chi_peak_wl: f64,
    pub binder_min_wl: f64,
    pub latent_heat: f64,   // (S_+ − S_−) / N entre los dos picos de P_{β_c}(S)
    pub barrier_df: f64,    // altura de la barrera de energía libre ΔF en β_c (> 0 = 1.er orden!)
    pub dmm_b0: f64,
    pub visited_frac: f64,
}

/// Fase 20 (Vía B — Wang–Landau + fase multicanónica exacta con balance detallado):
/// Construye la densidad de estados logarítmica g(S_BD) sobre la ventana que abarca la fase de
/// variedad plana (S ≈ 0) y la fase cristalina de Surya (S/N > 0), eliminando por completo el
/// cuello de botella de intercambio de Parallel Tempering sobre la barrera de primer orden!
/// Después congela g(S) y realiza una corrida multicanónica de producción exacta para reponderar
/// de forma continua en β ∈ [0,00; 0,35] (paso Δβ = 0,002) y extraer β_c, χ_pico, B₄_mín, calor
/// latente Δs y barrera de energía libre ΔF.
pub fn run_via_b_wang_landau_2d(
    n_elem: usize,
    nf: usize,
    m_scalar: f64,
    steps_per_stage: usize,
    seed: u64,
) -> WlResult {
    let n = n_elem.clamp(16, 72);
    // Ventana física adaptada a N que abarca íntegramente la fase plana (S/N ≈ 0),
    // el valle de coexistencia y el pico cristalino de Surya en β_c sin bines inalcanzables!
    let s_top_mult: i64 = if n <= 24 { 12 } else if n <= 32 { 14 } else if n <= 40 { 15 } else { 17 };
    let s_lo: i64 = -(n as i64);
    let s_hi: i64 = (n as i64) * s_top_mult;
    let n_bins: usize = 28;
    let bin_w_f = (s_hi - s_lo) as f64 / n_bins as f64;
    let to_bin = |s: f64| -> Option<usize> {
        if s < s_lo as f64 || s >= s_hi as f64 {
            None
        } else {
            Some((((s - s_lo as f64) / bin_w_f).floor() as usize).min(n_bins - 1))
        }
    };
    let bin_center = |b: usize| -> f64 { s_lo as f64 + (b as f64 + 0.5) * bin_w_f };

    let mut rng = SplitMix64::new(seed ^ ((n as u64) << 20) ^ ((nf as u64) * 0x9E37_79B9));
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i + 1) as u64) as usize;
        perm.swap(i, j);
    }
    let need_mat = nf > 0;
    let nf_f = nf as f64;
    let pairs_tot = (n * (n - 1)) as f64;
    let (mut s_cur, mut m_cur, mut r_cur) = eval_2order_fast(&perm, m_scalar, need_mat);
    while to_bin(s_cur).is_none() {
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i + 1) as u64) as usize;
            perm.swap(i, j);
        }
        let e = eval_2order_fast(&perm, m_scalar, need_mat);
        s_cur = e.0;
        m_cur = e.1;
        r_cur = e.2;
    }
    let mut b_cur = to_bin(s_cur).unwrap();

    // 1. Wang–Landau controlado por planitud de histograma (Landau–Tsai–Exler / Belardinelli–Pereyra):
    //    Partimos de la pendiente lineal cercana a β_c(N) para evitar muros artificiales en bines no
    //    visitados, y reducimos ln f ← ln f / 2 SOLO cuando el histograma de etapa cubra ≥ 75 % de los
    //    bines (permitiendo hasta 4 etapas extra si aún se está aplanando la barrera global!).
    let beta0 = if n <= 24 { 0.122 } else if n <= 32 { 0.106 } else if n <= 40 { 0.095 } else { 0.088 };
    let mut g: Vec<f64> = (0..n_bins).map(|b| -beta0 * bin_center(b)).collect();
    let mut ln_f = 0.10f64;
    let mut h_st = vec![0u64; n_bins];
    let stage_budget = (steps_per_stage * 7).max(8000);
    let mut halves = 0usize;
    let mut total_st = 0usize;
    while halves < 7 && total_st < 13 {
        total_st += 1;
        h_st.fill(0);
        for step in 1..=stage_budget {
            let a = (rng.next_u64() % n as u64) as usize;
            let mut b = (rng.next_u64() % (n - 1) as u64) as usize;
            if b >= a {
                b += 1;
            }
            perm.swap(a, b);
            let (sn, mn, rn) = eval_2order_fast(&perm, m_scalar, need_mat);
            if let Some(bn) = to_bin(sn) {
                let dw = (g[b_cur] - g[bn]) - nf_f * (mn - m_cur);
                if dw >= 0.0 || rng.next_f64() < dw.exp() {
                    s_cur = sn;
                    m_cur = mn;
                    r_cur = rn;
                    b_cur = bn;
                } else {
                    perm.swap(a, b);
                }
            } else {
                perm.swap(a, b);
            }
            g[b_cur] += ln_f;
            h_st[b_cur] += 1;
            if step % 600 == 0 && step >= 1800 {
                let vis_cnt = h_st.iter().filter(|&&c| c > 0).count();
                if vis_cnt * 10 >= n_bins * 8 {
                    let sum_h: u64 = h_st.iter().sum();
                    let min_h = h_st.iter().copied().filter(|&c| c > 0).min().unwrap_or(0);
                    let mean_h = sum_h as f64 / vis_cnt as f64;
                    if (min_h as f64) >= 0.55 * mean_h {
                        break;
                    }
                }
            }
        }
        let vis_cnt = h_st.iter().filter(|&&c| c > 0).count();
        let g_ref = g[n_bins / 2];
        for gv in g.iter_mut() {
            *gv -= g_ref;
        }
        if vis_cnt * 10 >= n_bins * 8 || total_st >= 7 {
            ln_f *= 0.5;
            halves += 1;
        }
    }
    // Ajuste multicanónico final de Berg–Neuhaus antes de congelar g(S): elimina cualquier inclinación
    // residual para garantizar túneles simétricos durante la producción exacta con balance detallado!
    let vis_final = h_st.iter().filter(|&&c| c > 0).count().max(1);
    let mean_hf = h_st.iter().sum::<u64>() as f64 / vis_final as f64;
    if mean_hf > 5.0 {
        for b in 0..n_bins {
            if h_st[b] > 0 {
                g[b] += 0.65 * ((h_st[b] as f64) / mean_hf).ln();
            }
        }
        let g_ref = g[n_bins / 2];
        for gv in g.iter_mut() {
            *gv -= g_ref;
        }
    }

    // 2. Fase multicanónica de producción con g(S) CONGELADO (balance detallado 100 % exacto!):
    //    Reponderar g_eff(b) = g(b) + ln H(b) produce observables canónicos 100 % insesgados!
    let mut hist = vec![0u64; n_bins];
    let mut s_sum = vec![0.0f64; n_bins];
    let mut s2_sum = vec![0.0f64; n_bins];
    let mut d_sum = vec![0.0f64; n_bins];
    let prod_steps = (steps_per_stage * 14).max(20000);
    for _ in 0..prod_steps {
        let a = (rng.next_u64() % n as u64) as usize;
        let mut b = (rng.next_u64() % (n - 1) as u64) as usize;
        if b >= a {
            b += 1;
        }
        perm.swap(a, b);
        let (sn, mn, rn) = eval_2order_fast(&perm, m_scalar, need_mat);
        if let Some(bn) = to_bin(sn) {
            let dw = (g[b_cur] - g[bn]) - nf_f * (mn - m_cur);
            if dw >= 0.0 || rng.next_f64() < dw.exp() {
                s_cur = sn;
                m_cur = mn;
                r_cur = rn;
                b_cur = bn;
            } else {
                perm.swap(a, b);
            }
        } else {
            perm.swap(a, b);
        }
        hist[b_cur] += 1;
        s_sum[b_cur] += s_cur;
        s2_sum[b_cur] += s_cur * s_cur;
        d_sum[b_cur] += mm_dim_2d(2.0 * r_cur as f64 / pairs_tot);
    }

    // Corrección exacta de histograma residual: g_eff(b) = g(b) + ln(hist[b]) en los bines visitados
    let mut visited = Vec::new();
    for b in 0..n_bins {
        if hist[b] >= 5 {
            let h = hist[b] as f64;
            visited.push((b, g[b] + h.ln(), s_sum[b] / h, s2_sum[b] / h, d_sum[b] / h));
        }
    }
    let visited_frac = visited.len() as f64 / n_bins.max(1) as f64;

    // 3. Reponderación canónica continua en β ∈ [0,00; 0,35] (paso Δβ = 0,002)
    let eval_beta = |beta: f64| -> (f64, f64, f64, f64, Vec<(f64, f64)>) {
        let mut max_lw = f64::NEG_INFINITY;
        for &(_, ge, sm, _, _) in &visited {
            max_lw = max_lw.max(ge + beta * sm);
        }
        let mut z = 0.0;
        let mut m1 = 0.0;
        let mut dm = 0.0;
        let mut probs = Vec::with_capacity(visited.len());
        for &(_, ge, sm, _, dbar) in &visited {
            let w = (ge + beta * sm - max_lw).exp();
            z += w;
            m1 += w * sm;
            dm += w * dbar;
            probs.push((sm, w));
        }
        let mean_s = m1 / z.max(1e-30);
        let mean_d = dm / z.max(1e-30);
        let mut mu2 = 0.0;
        let mut mu4 = 0.0;
        for &(_, ge, sm, _, _) in &visited {
            let w = (ge + beta * sm - max_lw).exp() / z.max(1e-30);
            let d = sm - mean_s;
            let d2 = d * d;
            mu2 += w * d2;
            mu4 += w * d2 * d2;
        }
        let chi = mu2 / n as f64;
        let binder = if mu2 > 1e-20 { 1.0 - mu4 / (3.0 * mu2 * mu2) } else { 0.0 };
        for p in probs.iter_mut() {
            p.1 /= z.max(1e-30);
        }
        (mean_s / n as f64, chi, binder, mean_d, probs)
    };

    let (_, _, _, dmm_b0, _) = eval_beta(0.0);
    let mut best_chi = -1.0f64;
    let mut beta_c = 0.0f64;
    let mut binder_min = f64::INFINITY;
    for step in 0..=175 {
        let beta = step as f64 * 0.002;
        let (_, chi, b4, _, _) = eval_beta(beta);
        if chi > best_chi {
            best_chi = chi;
            beta_c = beta;
        }
        if b4 < binder_min {
            binder_min = b4;
        }
    }

    // 4. Calor latente Δs y barrera de energía libre ΔF en β_c (dos picos a ambos lados de ⟨S⟩_{β_c})
    let (mean_s_c, _, _, _, probs_c) = eval_beta(beta_c);
    let mut p_by_bin = vec![0.0f64; n_bins];
    for (idx, &(b, _, _, _, _)) in visited.iter().enumerate() {
        p_by_bin[b] = probs_c[idx].1;
    }
    let split_b = ((mean_s_c * n as f64 - s_lo as f64) / bin_w_f).floor().clamp(1.0, (n_bins - 2) as f64) as usize;
    let mut b_lo_pk = 0usize;
    let mut p_lo_pk = 0.0f64;
    for b in 0..=split_b {
        if p_by_bin[b] > p_lo_pk {
            p_lo_pk = p_by_bin[b];
            b_lo_pk = b;
        }
    }
    let mut b_hi_pk = split_b + 1;
    let mut p_hi_pk = 0.0f64;
    for b in (split_b + 1)..n_bins {
        if p_by_bin[b] > p_hi_pk {
            p_hi_pk = p_by_bin[b];
            b_hi_pk = b;
        }
    }
    let (latent_heat, barrier_df) = if p_lo_pk > 1e-6 && p_hi_pk > 1e-6 && b_hi_pk > b_lo_pk + 1 {
        let mut p_min = f64::INFINITY;
        for b in (b_lo_pk + 1)..b_hi_pk {
            if p_by_bin[b] > 0.0 && p_by_bin[b] < p_min {
                p_min = p_by_bin[b];
            }
        }
        let p_ref = p_lo_pk.min(p_hi_pk);
        let df = if p_min.is_finite() && p_min < p_ref {
            (p_ref / p_min).ln()
        } else {
            0.0
        };
        let lh = (bin_center(b_hi_pk) - bin_center(b_lo_pk)) / n as f64;
        (lh, df)
    } else {
        (0.0, 0.0)
    };

    WlResult {
        n_elem: n,
        nf,
        m_scalar,
        beta_c_wl: beta_c,
        chi_peak_wl: best_chi,
        binder_min_wl: if binder_min.is_finite() { binder_min } else { 0.0 },
        latent_heat,
        barrier_df,
        dmm_b0,
        visited_frac,
    }
}

/// Barrido en β = 0 de la masa escalar m ∈ {0,05; 0,25; 0,80; 2,00} (N_f = 1) y del número de
/// especies cuánticas N_f ∈ {0, 1, 2, 4, 8} (m = 0,80) mediante Metropolis directo en la fase plana
/// (donde τ_int ≈ 0,6, sin barreras metaestables!).
pub fn run_via_b_matter_mass_nf_sweep(n_elem: usize, sweeps: usize, seed: u64) -> Vec<[f64; 5]> {
    let n = n_elem.clamp(16, 48);
    let specs: [(usize, f64); 8] = [
        // Barrido de masa m (N_f = 1): casi sin masa → masivo
        (1, 0.05),
        (1, 0.25),
        (1, 0.80),
        (1, 2.00),
        // Barrido de especies N_f (m = 0,80): 0, 2, 4, 8
        (0, 0.80),
        (2, 0.80),
        (4, 0.80),
        (8, 0.80),
    ];
    let run_one = |(idx, (nf, ms)): (usize, (usize, f64))| -> [f64; 5] {
        let mut rng = SplitMix64::new(seed ^ ((idx as u64 + 1) * 0x9E37_79B9));
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i + 1) as u64) as usize;
            perm.swap(i, j);
        }
        let pairs_tot = (n * (n - 1)) as f64;
        let (_, mut m_cur, mut r_cur) = eval_2order_fast(&perm, ms, nf > 0);
        let therm = sweeps / 3;
        let moves = (n / 2).max(6);
        let mut d_hist = Vec::with_capacity(sweeps - therm);
        for sw in 0..sweeps {
            for _ in 0..moves {
                let a = (rng.next_u64() % n as u64) as usize;
                let mut b = (rng.next_u64() % (n - 1) as u64) as usize;
                if b >= a {
                    b += 1;
                }
                perm.swap(a, b);
                let (_, mn, rn) = eval_2order_fast(&perm, ms, nf > 0);
                let dw = -(nf as f64) * (mn - m_cur);
                if dw >= 0.0 || rng.next_f64() < dw.exp() {
                    m_cur = mn;
                    r_cur = rn;
                } else {
                    perm.swap(a, b);
                }
            }
            if sw >= therm {
                d_hist.push(mm_dim_2d(2.0 * r_cur as f64 / pairs_tot));
            }
        }
        let tau = integrated_autocorr_time(&d_hist);
        let blk = ((2.0 * tau).ceil() as usize).clamp(2, 20);
        let (dm, de) = block_bootstrap_mean_err(&d_hist, blk, 140, seed ^ (idx as u64 * 733));
        [nf as f64, ms, dm, de, tau]
    };

    #[cfg(not(target_arch = "wasm32"))]
    return std::thread::scope(|sc| {
        let hs: Vec<_> = specs.into_iter().enumerate().map(|t| sc.spawn(move || run_one(t))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    specs.into_iter().enumerate().map(run_one).collect()
}

/// Evalúa un único punto (N, N_f, m_scalar) en β = 0 mediante Metropolis directo con barras de
/// block bootstrap y τ_int de Madras–Sokal (usado para el escalado de tamaño finito de la
/// transición a gran N_f ∈ {0, 2, 4, 6, 8} en N ∈ {24, 36, 48}).
pub fn run_via_b_large_nf_point(
    n_elem: usize,
    nf: usize,
    m_scalar: f64,
    sweeps: usize,
    seed: u64,
) -> [f64; 5] {
    let n = n_elem.clamp(16, 64);
    let mut rng = SplitMix64::new(seed ^ ((n as u64) << 18) ^ ((nf as u64) * 0x9E37_79B9));
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i + 1) as u64) as usize;
        perm.swap(i, j);
    }
    let pairs_tot = (n * (n - 1)) as f64;
    let (_, mut m_cur, mut r_cur) = eval_2order_fast(&perm, m_scalar, nf > 0);
    let therm = sweeps / 3;
    let moves = (n / 2).max(6);
    let mut d_hist = Vec::with_capacity(sweeps - therm);
    for sw in 0..sweeps {
        for _ in 0..moves {
            let a = (rng.next_u64() % n as u64) as usize;
            let mut b = (rng.next_u64() % (n - 1) as u64) as usize;
            if b >= a {
                b += 1;
            }
            perm.swap(a, b);
            let (_, mn, rn) = eval_2order_fast(&perm, m_scalar, nf > 0);
            let dw = -(nf as f64) * (mn - m_cur);
            if dw >= 0.0 || rng.next_f64() < dw.exp() {
                m_cur = mn;
                r_cur = rn;
            } else {
                perm.swap(a, b);
            }
        }
        if sw >= therm {
            d_hist.push(mm_dim_2d(2.0 * r_cur as f64 / pairs_tot));
        }
    }
    let tau = integrated_autocorr_time(&d_hist);
    let blk = ((2.0 * tau).ceil() as usize).clamp(2, 24);
    let (dm, de) = block_bootstrap_mean_err(&d_hist, blk, 160, seed ^ 0xBEEF);
    [n as f64, nf as f64, dm, de, tau]
}

/// Versión compacta WASM / test de la Fase 20 (Vía B): Wang–Landau multicanónico en N = 20, 28
/// (gravedad pura + Sorkin–Johnston N_f = 1) + barrido de masa m y especies N_f = 0..8 en β = 0.
pub fn run_via_b_wl_and_matter_sweep(steps_per_stage: usize) -> Vec<f64> {
    let st = steps_per_stage.clamp(1200, 12000);
    let wl_tasks = [(20usize, 0usize), (28, 0), (20, 1), (28, 1)];
    #[cfg(not(target_arch = "wasm32"))]
    let wl_res: Vec<WlResult> = std::thread::scope(|sc| {
        let hs: Vec<_> = wl_tasks
            .into_iter()
            .enumerate()
            .map(|(i, (n, nf))| sc.spawn(move || run_via_b_wang_landau_2d(n, nf, 0.8, st, 0x20B0_1111 ^ (i as u64 * 7919))))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let wl_res: Vec<WlResult> = wl_tasks
        .into_iter()
        .enumerate()
        .map(|(i, (n, nf))| run_via_b_wang_landau_2d(n, nf, 0.8, st, 0x20B0_1111 ^ (i as u64 * 7919)))
        .collect();

    let sweep = run_via_b_matter_mass_nf_sweep(24, 420, 0x20B0_5555);
    // Salida:
    // Cabecera (2): [n_wl (4), n_sweep (8)]
    // 4 × [N, Nf, beta_c_wl, chi_peak_wl, binder_min_wl, latent_heat, barrier_df, dmm_b0, visited_frac] (9 floats)
    // 8 × [Nf, m_scalar, dmm_mean, dmm_err, tau_int] (5 floats)
    let mut out = vec![wl_res.len() as f64, sweep.len() as f64];
    for w in &wl_res {
        out.extend_from_slice(&[
            w.n_elem as f64,
            w.nf as f64,
            w.beta_c_wl,
            w.chi_peak_wl,
            w.binder_min_wl,
            w.latent_heat,
            w.barrier_df,
            w.dmm_b0,
            w.visited_frac,
        ]);
    }
    for s in &sweep {
        out.extend_from_slice(s);
    }
    out
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 18 (Vía B): Parallel Tempering sobre conjuntos causales 2D acoplados a materia cuántica
    /// de Sorkin–Johnston (N_f = 0, 1, 2) e Ising sobre enlaces en dos tamaños (escalado finito,
    /// susceptibilidad χ_S, Binder, block bootstrap y τ_int de Madras–Sokal).
    pub fn via_b_causal_pt_scaling(&mut self, n_small: u32, n_large: u32, sweeps: u32) -> *const f64 {
        self.oq.out = run_via_b_causal_pt_scaling(
            n_small as usize,
            n_large as usize,
            (sweeps as usize).clamp(150, 2000),
        );
        self.oq.out.as_ptr()
    }

    /// Fase 19 (Vía B): escalado de tamaño finito (FSS) con escalera adaptativa de 20 réplicas
    /// (Katzgraber–Trebst) en N ∈ {16, 22, 28} × 4 modelos de materia: ajusta χ_pico ∝ N^γ,
    /// extrapola d_MM(β=0, N → ∞) y β_c(∞), y mantiene alta tasa de intercambio de réplicas.
    pub fn via_b_adaptive_fss(&mut self, sweeps: u32) -> *const f64 {
        self.oq.out = run_via_b_adaptive_fss_compact(sweeps as usize);
        self.oq.out.as_ptr()
    }

    /// Fase 20 (Vía B — Wang–Landau multicanónico + barrido de masa m y N_f = 0..8):
    /// supera el cuello de botella sobre la barrera de primer orden midiendo β_c continuo, calor
    /// latente Δs y barrera de energía libre ΔF, y comprueba la estabilidad de d_MM en m → 0 y N_f = 8.
    pub fn via_b_wl_and_matter_sweep(&mut self, steps_per_stage: u32) -> *const f64 {
        self.oq.out = run_via_b_wl_and_matter_sweep(steps_per_stage as usize);
        self.oq.out.as_ptr()
    }
}
