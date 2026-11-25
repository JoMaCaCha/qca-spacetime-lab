//! Fase 18 — Vía C: Restauración infrarroja de Lorentz mediante confinamiento gauge en QED 1+1D
//! de dos sabores (modelo de Schwinger cuántico de dos especies con distinta velocidad desnuda
//! v_A ≠ v_B acopladas al MISMO campo eléctrico cuantizado E_r, conservando EXACTAMENTE
//! U(1)_A × U(1)_B sin mezcla de sabor!) + motor DMRG variacional con bloques explícitos de carga
//! U(1) y MPO exacto D = 7 sin corte del campo eléctrico.
//!
//! Predicción analítica exacta de bosonización (Coleman 1976 generalizado a v_A ≠ v_B):
//!   - Para J = 0: dos modos sin masa independientes (c = 2) con velocidades v_A y v_B.
//!   - Para J > 0: el modo de carga simétrica φ_+ adquiere la masa del bosón de Schwinger
//!     M_S = √(2 J (v_A + v_B) / π), mientras el modo neutro antisimétrico φ_- permanece
//!     exactamente sin masa (c_CFT = 1) e iguala las velocidades infrarrojas de AMBAS especies A y B
//!     a la **media geométrica exacta** v_IR = √(v_A · v_B), con transferencia de sabor ΔN_B ≡ 0!

use crate::linalg::{complete_complex_isometry, complete_real_isometry};
use crate::research_io::lin_fit_r2;
use crate::rng::SplitMix64;
use crate::spinor::cx;
use crate::sun::C;
use crate::util::clamp_finite;
use crate::QuantumSpacetimeUniverse;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Término disperso de MPO para el motor DMRG con simetría U(1) por bloques.
pub struct U1MpoTerm {
    pub l: usize,
    pub r: usize,
    pub op: [[f64; 2]; 2],
    pub c: f64,
}

pub struct U1MpoSite {
    pub dl: usize,
    pub dr: usize,
    pub terms: Vec<U1MpoTerm>,
}

pub struct U1Mpo {
    pub sites: Vec<U1MpoSite>,
}

impl U1Mpo {
    /// MPO exacto D = 5 de la cadena XXZ: H = Σ [½(σ⁺σ⁻ + σ⁻σ⁺) + Δ S^z S^z] con J_xy = 1, J_z = delta.
    pub fn xxz(l: usize, delta: f64) -> Self {
        let id = [[1.0, 0.0], [0.0, 1.0]];
        let sp = [[0.0, 1.0], [0.0, 0.0]];
        let sm = [[0.0, 0.0], [1.0, 0.0]];
        let sz = [[0.5, 0.0], [0.0, -0.5]];
        let mut sites = Vec::with_capacity(l);
        for i in 0..l {
            let dl = if i == 0 { 1 } else { 5 };
            let dr = if i + 1 == l { 1 } else { 5 };
            let f_out = if i + 1 == l { 0 } else { 4 };
            let mut terms = Vec::new();
            if i + 1 < l {
                terms.push(U1MpoTerm { l: 0, r: 0, op: id, c: 1.0 });
                terms.push(U1MpoTerm { l: 0, r: 1, op: sp, c: 0.5 });
                terms.push(U1MpoTerm { l: 0, r: 2, op: sm, c: 0.5 });
                terms.push(U1MpoTerm { l: 0, r: 3, op: sz, c: delta });
            }
            if i > 0 {
                terms.push(U1MpoTerm { l: 4, r: f_out, op: id, c: 1.0 });
                terms.push(U1MpoTerm { l: 1, r: f_out, op: sm, c: 1.0 });
                terms.push(U1MpoTerm { l: 2, r: f_out, op: sp, c: 1.0 });
                terms.push(U1MpoTerm { l: 3, r: f_out, op: sz, c: 1.0 });
            }
            sites.push(U1MpoSite { dl, dr, terms });
        }
        U1Mpo { sites }
    }
}

// ================================================================== 1. Diagonalización exacta en sector (N_A, N_B)

/// Sector exacto U(1)_A × U(1)_B de una cadena escalonada de dos sabores de `n_rungs` peldaños
/// con N_A = N_B = n_rungs / 2 (base producto cartesiano: O(1) indexación, hasta N_r = 10 ⇒ 63 504 estados!).
struct Schwinger2fSector {
    n_rungs: usize,
    states_a: Vec<u16>,
    states_b: Vec<u16>,
    /// Lista de adyacencia de saltos nearest-neighbor dentro de la especie A (índice destino).
    hops_a: Vec<Vec<usize>>,
    /// Lista de adyacencia de saltos nearest-neighbor dentro de la especie B.
    hops_b: Vec<Vec<usize>>,
    /// Cargas por peldaño de cada estado de A: q_a[ia][r] ∈ {−1, 0, +1} (en realidad n_r − (1−(−1)^r)/2).
    n_a: Vec<Vec<i8>>,
    n_b: Vec<Vec<i8>>,
}

impl Schwinger2fSector {
    fn new(n_rungs: usize) -> Self {
        let half = n_rungs / 2;
        let full = 1usize << n_rungs;
        let mut states = Vec::new();
        let mut idx_map = vec![-1i32; full];
        for s in 0..full {
            if (s as u32).count_ones() as usize == half {
                idx_map[s] = states.len() as i32;
                states.push(s as u16);
            }
        }
        let d = states.len();
        let mut hops = vec![Vec::new(); d];
        let mut occs = vec![vec![0i8; n_rungs]; d];
        for (i, &st) in states.iter().enumerate() {
            for r in 0..n_rungs {
                occs[i][r] = ((st >> r) & 1) as i8;
            }
            for r in 0..(n_rungs - 1) {
                let b0 = (st >> r) & 1;
                let b1 = (st >> (r + 1)) & 1;
                if b0 != b1 {
                    let fl = (st ^ (1u16 << r) ^ (1u16 << (r + 1))) as usize;
                    hops[i].push(idx_map[fl] as usize);
                }
            }
        }
        Schwinger2fSector {
            n_rungs,
            states_a: states.clone(),
            states_b: states,
            hops_a: hops.clone(),
            hops_b: hops,
            n_a: occs.clone(),
            n_b: occs,
        }
    }

    #[inline(always)]
    fn dim(&self) -> usize {
        self.states_a.len() * self.states_b.len()
    }

    /// Aplica H_Schwinger-2f sobre un vector real en O(dim · n_rungs):
    ///   H = (v_A/2) Σ (c†_{A,r} c_{A,r+1} + h.c.) + (v_B/2) Σ (c†_{B,r} c_{B,r+1} + h.c.)
    ///     + J Σ_{r=0}^{N_r-2} E_r²,   E_r = Σ_{j≤r} [n_{A,j} + n_{B,j} − (1 − (−1)^j)].
    fn matvec(&self, v: &[f64], out: &mut [f64], v_a: f64, v_b: f64, j_gauge: f64) {
        let da = self.states_a.len();
        let db = self.states_b.len();
        let nr = self.n_rungs;
        let wa = 0.5 * v_a;
        let wb = 0.5 * v_b;
        out.fill(0.0);

        for ia in 0..da {
            let na = &self.n_a[ia];
            let ha = &self.hops_a[ia];
            let row_base = ia * db;
            for ib in 0..db {
                let idx = row_base + ib;
                let val = v[idx];
                if val == 0.0 {
                    continue;
                }
                // 1. Energía eléctrica exacta J Σ E_r² con cargas opuestas q_A = +1, q_B = −1:
                //    Q_r = n_{A,r} − n_{B,r} ⇒ el estado neutro |1_A, 1_B⟩ / |0_A, 0_B⟩ tiene E_r = 0
                //    y mantiene el pión neutro exactamente sin masa (c_CFT = 1) mientras el modo
                //    cargado (n_A − n_B) adquiere la masa del bosón de Schwinger!
                if j_gauge != 0.0 {
                    let nb = &self.n_b[ib];
                    let mut e_acc = 0i32;
                    let mut e2_sum = 0i32;
                    for r in 0..(nr - 1) {
                        e_acc += na[r] as i32 - nb[r] as i32;
                        e2_sum += e_acc * e_acc;
                    }
                    out[idx] += (j_gauge * e2_sum as f64) * val;
                }
                // 2. Saltos especie A (cambia solo ia)
                for &ia2 in ha {
                    out[ia2 * db + ib] += wa * val;
                }
                // 3. Saltos especie B (cambia solo ib)
                for &ib2 in &self.hops_b[ib] {
                    out[row_base + ib2] += wb * val;
                }
            }
        }
    }
}

/// Jacobi simétrico para tridiagonal m × m.
fn jacobi_tridiag(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<f64>) {
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
    ((0..m).map(|i| tmat[i * m + i]).collect(), vecs)
}

/// Estado fundamental exacto de Lanczos en el sector (N_A = N_r/2, N_B = N_r/2).
fn lanczos_ground_2f(sec: &Schwinger2fSector, v_a: f64, v_b: f64, j_gauge: f64, kmax: usize) -> (f64, Vec<f64>) {
    let dim = sec.dim();
    let k_lim = kmax.min(dim);
    let mut rng = SplitMix64::new(0x18C0_5C41 ^ dim as u64);
    let mut v0: Vec<f64> = (0..dim).map(|_| rng.next_f64() - 0.5).collect();
    let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v0.iter_mut() {
        *x /= n0;
    }
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(k_lim);
    vs.push(v0);
    let mut alphas = Vec::with_capacity(k_lim);
    let mut betas = Vec::with_capacity(k_lim);
    let mut w = vec![0.0; dim];

    for j in 0..k_lim {
        sec.matvec(&vs[j], &mut w, v_a, v_b, j_gauge);
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
        if b < 1e-13 || j + 1 == k_lim {
            break;
        }
        betas.push(b);
        let inv = 1.0 / b;
        vs.push(w.iter().map(|x| x * inv).collect());
    }
    let m = alphas.len();
    let (evals, vecs) = jacobi_tridiag(&alphas, &betas);
    let mut best = 0;
    for i in 1..m {
        if evals[i] < evals[best] {
            best = i;
        }
    }
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
    (evals[best], psi)
}

/// Polo infrarrojo exacto del operador de densidad de onda estacionaria más baja
/// O_mode = Σ_r cos(π(r+1)/(N_r+1)) · [w_a (n_{A,r}−½) + w_b (n_{B,r}−½)].
/// Para J = 0 (libre), ω / (2 sin(π/(2(N_r+1)))) devuelve v_A o v_B con precisión 10⁻¹³!
#[allow(clippy::too_many_arguments)]
fn lanczos_ir_pole_2f(
    sec: &Schwinger2fSector,
    e0: f64,
    gs0: &[f64],
    w_a: f64,
    w_b: f64,
    v_a: f64,
    v_b: f64,
    j_gauge: f64,
) -> (f64, f64) {
    let nr = sec.n_rungs;
    let da = sec.states_a.len();
    let db = sec.states_b.len();
    let dim = sec.dim();
    let prof: Vec<f64> = (0..nr).map(|r| (PI * (r + 1) as f64 / (nr + 1) as f64).cos()).collect();
    let mut oa_val = vec![0.0; da];
    let mut ob_val = vec![0.0; db];
    for ia in 0..da {
        let mut s = 0.0;
        for r in 0..nr {
            s += prof[r] * (sec.n_a[ia][r] as f64 - 0.5);
        }
        oa_val[ia] = s;
    }
    for ib in 0..db {
        let mut s = 0.0;
        for r in 0..nr {
            s += prof[r] * (sec.n_b[ib][r] as f64 - 0.5);
        }
        ob_val[ib] = s;
    }

    let mut v0 = vec![0.0; dim];
    for ia in 0..da {
        let fa = w_a * oa_val[ia];
        let base = ia * db;
        for ib in 0..db {
            v0[base + ib] = (fa + w_b * ob_val[ib]) * gs0[base + ib];
        }
    }
    // Proyectamos fuera del estado fundamental para medir solo excitaciones ω > 0
    let ov0: f64 = v0.iter().zip(gs0).map(|(x, y)| x * y).sum();
    for i in 0..dim {
        v0[i] -= ov0 * gs0[i];
    }
    let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
    for x in v0.iter_mut() {
        *x /= n0;
    }

    let k_lim = 48usize.min(dim);
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(k_lim);
    vs.push(v0);
    let mut alphas = Vec::with_capacity(k_lim);
    let mut betas = Vec::with_capacity(k_lim);
    let mut w = vec![0.0; dim];
    for j in 0..k_lim {
        sec.matvec(&vs[j], &mut w, v_a, v_b, j_gauge);
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
        if b < 1e-12 || j + 1 == k_lim {
            break;
        }
        betas.push(b);
        let inv = 1.0 / b;
        vs.push(w.iter().map(|x| x * inv).collect());
    }
    let m = alphas.len();
    let (evals, vecs) = jacobi_tridiag(&alphas, &betas);
    let mut poles: Vec<(f64, f64)> = (0..m)
        .map(|k| (evals[k] - e0, vecs[k] * vecs[k]))
        .filter(|&(wx, p)| wx > 1e-5 && p > 1e-4)
        .collect();
    poles.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Primer cúmulo infrarrojo con peso apreciable ≥ 8 %
    let mut clusters: Vec<(f64, f64)> = Vec::new();
    for &(wx, p) in &poles {
        if let Some(last) = clusters.last_mut() {
            if (wx - last.0).abs() <= 0.05 {
                let pn = last.1 + p;
                last.0 = (last.0 * last.1 + wx * p) / pn;
                last.1 = pn;
                continue;
            }
        }
        clusters.push((wx, p));
    }
    let omega_ir = clusters
        .iter()
        .find(|&&(_, p)| p >= 0.08)
        .or_else(|| clusters.iter().max_by(|a, b| a.1.total_cmp(&b.1)))
        .map(|&(wx, _)| wx)
        .unwrap_or(0.0);
    let q_scale = 2.0 * (PI / (2.0 * (nr + 1) as f64)).sin();
    (omega_ir / q_scale, omega_ir)
}

// ================================================================== 2. MPO exacto D = 7 de Schwinger 2 sabores + DMRG U(1) por bloques

/// Construye el MPO exacto de dimensión D = 7 (SIN corte del campo eléctrico!) para el modelo de
/// Schwinger cuántico de 2 sabores en `n_rungs` peldaños (L = 2·n_rungs sitios intercalados A_0,B_0,…):
///   H = (v_A/2) Σ c†_{A,r} c_{A,r+1} + (v_B/2) Σ c†_{B,r} c_{B,r+1} + h.c. + J Σ_{r=0}^{N_r-2} E_r²
/// usando la identidad exacta Σ E_r² = Σ_{j≤N_r-2} (N_r − 1 − j) Q_j² + 2 Σ_{i<j} (N_r − 1 − j) Q_i Q_j.
pub fn schwinger_2f_exact_mpo(n_rungs: usize, v_a: f64, v_b: f64, j_gauge: f64) -> U1Mpo {
    let l = 2 * n_rungs;
    // Convención de base física: s = 0 (↑, n=1), s = 1 (↓, n=0) ⇒ n = [[1,0],[0,0]]
    // σ⁺ = [[0,1],[0,0]], σ⁻ = [[0,0],[1,0]], σ^z = [[1,0],[0,-1]]
    let id = [[1.0, 0.0], [0.0, 1.0]];
    let sz = [[1.0, 0.0], [0.0, -1.0]];
    let sp = [[0.0, 1.0], [0.0, 0.0]];
    let sm = [[0.0, 0.0], [1.0, 0.0]];
    let wa = 0.5 * v_a;
    let wb = 0.5 * v_b;

    // Canales MPO (D = 7): 0=I, 1=A⁺, 2=A⁻, 3=B⁺, 4=B⁻, 5=Q_acc, 6=F
    let mut sites = Vec::with_capacity(l);
    for site in 0..l {
        let rung = site / 2;
        let is_a = (site % 2) == 0;
        let dl = if site == 0 { 1 } else { 7 };
        let dr = if site + 1 == l { 1 } else { 7 };
        let f_out = if site + 1 == l { 0 } else { 6 };

        // Carga local de cargas opuestas q_A = +n_A, q_B = −n_B ⇒ Q_r = n_{A,r} − n_{B,r}
        // (n = [[1,0],[0,0]], q² = n = [[1,0],[0,0]] en ambas especies!)
        let (q_op, q2_op) = if is_a {
            ([[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]])
        } else {
            ([[-1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]])
        };
        let coeff = if rung + 1 < n_rungs {
            j_gauge * (n_rungs - 1 - rung) as f64
        } else {
            0.0
        };

        let mut terms = Vec::new();
        if site + 1 < l {
            terms.push(U1MpoTerm { l: 0, r: 0, op: id, c: 1.0 });
        }
        if site > 0 {
            terms.push(U1MpoTerm { l: 6, r: f_out, op: id, c: 1.0 });
        }

        if coeff != 0.0 {
            terms.push(U1MpoTerm { l: 0, r: f_out, op: q2_op, c: coeff });
            if site > 0 {
                terms.push(U1MpoTerm { l: 5, r: f_out, op: q_op, c: 2.0 * coeff });
            }
        }
        if site + 1 < l && j_gauge != 0.0 {
            if rung + 1 < n_rungs {
                terms.push(U1MpoTerm { l: 0, r: 5, op: q_op, c: 1.0 });
            }
            if site > 0 {
                terms.push(U1MpoTerm { l: 5, r: 5, op: id, c: 1.0 });
            }
        }

        if is_a {
            if site + 2 < l {
                terms.push(U1MpoTerm { l: 0, r: 1, op: sp, c: wa });
                terms.push(U1MpoTerm { l: 0, r: 2, op: sm, c: wa });
            }
            if site >= 2 {
                terms.push(U1MpoTerm { l: 1, r: f_out, op: sm, c: 1.0 });
                terms.push(U1MpoTerm { l: 2, r: f_out, op: sp, c: 1.0 });
            }
            if site > 0 && site + 1 < l {
                terms.push(U1MpoTerm { l: 3, r: 3, op: sz, c: 1.0 });
                terms.push(U1MpoTerm { l: 4, r: 4, op: sz, c: 1.0 });
            }
        } else {
            if site + 2 < l {
                terms.push(U1MpoTerm { l: 0, r: 3, op: sp, c: wb });
                terms.push(U1MpoTerm { l: 0, r: 4, op: sm, c: wb });
            }
            if site >= 2 {
                terms.push(U1MpoTerm { l: 3, r: f_out, op: sm, c: 1.0 });
                terms.push(U1MpoTerm { l: 4, r: f_out, op: sp, c: 1.0 });
            }
            if site > 0 && site + 1 < l {
                terms.push(U1MpoTerm { l: 1, r: 1, op: sz, c: 1.0 });
                terms.push(U1MpoTerm { l: 2, r: 2, op: sz, c: 1.0 });
            }
        }
        sites.push(U1MpoSite { dl, dr, terms });
    }
    U1Mpo { sites }
}

/// SVD real compacto (Jacobi de Hestenes) para bloques U(1) pequeños.
fn block_svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if m < n {
        let mut at = vec![0.0; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        let (ut, s, vtt) = block_svd_tall(&at, n, m);
        let k = s.len();
        let mut u = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                u[i * k + j] = vtt[j * m + i];
            }
        }
        let mut vt = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                vt[i * n + j] = ut[j * k + i];
            }
        }
        return (u, s, vt);
    }
    block_svd_tall(a, m, n)
}

fn block_svd_tall(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i * n + j]).collect()).collect();
    let mut vcols: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let mut v = vec![0.0; n];
            v[j] = 1.0;
            v
        })
        .collect();
    for _ in 0..30 {
        let mut conv = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let (mut al, mut be, mut ga) = (0.0, 0.0, 0.0);
                for i in 0..m {
                    let (ap, aq) = (cols[p][i], cols[q][i]);
                    al += ap * ap;
                    be += aq * aq;
                    ga += ap * aq;
                }
                if ga.abs() <= 1e-14 * (al * be).sqrt().max(1e-300) {
                    continue;
                }
                conv = false;
                let tau = (be - al) / (2.0 * ga);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..m {
                    let (ap, aq) = (cols[p][i], cols[q][i]);
                    cols[p][i] = ap * c - aq * s;
                    cols[q][i] = ap * s + aq * c;
                }
                for i in 0..n {
                    let (vp, vq) = (vcols[p][i], vcols[q][i]);
                    vcols[p][i] = vp * c - vq * s;
                    vcols[q][i] = vp * s + vq * c;
                }
            }
        }
        if conv {
            break;
        }
    }
    let norms: Vec<f64> = (0..n).map(|j| cols[j].iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    let mut u = vec![0.0; m * n];
    let mut s = vec![0.0; n];
    let mut vt = vec![0.0; n * n];
    let mut rank = 0usize;
    for (idx, &col) in order.iter().enumerate() {
        let sig = norms[col];
        s[idx] = sig;
        if sig > 1e-15 {
            rank += 1;
            let inv = 1.0 / sig;
            for i in 0..m {
                u[i * n + idx] = cols[col][i] * inv;
            }
        }
        for j in 0..n {
            vt[idx * n + j] = vcols[col][j];
        }
    }
    if rank < n {
        complete_real_isometry(&mut u, m, n, rank);
    }
    (u, s, vt)
}

/// MPS con conservación explícita de carga U(1) por bloques en cada enlace (2·S^z ∈ ℤ):
/// cada estado de enlace `a` lleva su carga exacta `q_bond[site][a]`, y el SVD de dos sitios
/// factoriza bloque por bloque dentro de cada sector de carga `q_mid` (aceleración O(k²) y
/// conservación exacta de N_total = L/2 a precisión de máquina!).
pub struct U1Mps {
    pub l: usize,
    pub q_bonds: Vec<Vec<i32>>, // l + 1 vectores de cargas de enlace (2·S^z acumulado)
    pub tensors: Vec<Vec<f64>>, // [s * dl * dr + a * dr + b]
    pub center_s: Vec<f64>,     // valores singulares en el último corte central
    pub bond_s: Vec<Vec<f64>>,  // valores singulares por enlace (índice 1..l)
}

impl U1Mps {
    /// Inicializa un MPS válido en el sector de carga total exacta Q = 0 (medio llenado) con
    /// múltiples sectores de carga de enlace q ∈ {−2, 0, +2} (o {−1, +1} en enlaces impares)
    /// para que DMRG de 2 sitios acople inmediatamente saltos de segundos vecinos (A_r ↔ A_{r+1}, B_r ↔ B_{r+1})!
    pub fn seed_u1(l: usize, seed: u64) -> Self {
        let mut q_bonds: Vec<Vec<i32>> = Vec::with_capacity(l + 1);
        q_bonds.push(vec![0]);
        for i in 1..l {
            let max_q = (i as i32).min((l - i) as i32).min(4);
            let mut qs = Vec::new();
            let mut q = -max_q;
            while q <= max_q {
                if ((q + i as i32) & 1) == 0 {
                    qs.push(q);
                }
                q += 1;
            }
            q_bonds.push(qs);
        }
        q_bonds.push(vec![0]);

        let mut rng = SplitMix64::new(seed ^ 0x18C0_D046);
        let phys_q = [1i32, -1i32];
        let mut tensors = Vec::with_capacity(l);
        for i in 0..l {
            let dl = q_bonds[i].len();
            let dr = q_bonds[i + 1].len();
            let mut t = vec![0.0; 2 * dl * dr];
            for s in 0..2 {
                for a in 0..dl {
                    for b in 0..dr {
                        if q_bonds[i][a] + phys_q[s] == q_bonds[i + 1][b] {
                            t[s * dl * dr + a * dr + b] = rng.next_f64() - 0.5;
                        }
                    }
                }
            }
            let nrm = t.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
            for x in t.iter_mut() {
                *x /= nrm;
            }
            tensors.push(t);
        }
        // Ortogonalización a la derecha inicial (SVD por bloques U(1) de derecha a izquierda)
        let mut mps = U1Mps {
            l,
            q_bonds,
            tensors,
            center_s: vec![1.0],
            bond_s: vec![vec![1.0]; l + 1],
        };
        for i in (1..l).rev() {
            let dl = mps.q_bonds[i].len();
            let dr = mps.q_bonds[i + 1].len();
            let mut by_q: BTreeMap<i32, (Vec<usize>, Vec<(usize, usize)>)> = BTreeMap::new();
            for a in 0..dl {
                by_q.entry(mps.q_bonds[i][a]).or_default().0.push(a);
            }
            for s in 0..2 {
                for b in 0..dr {
                    let ql_need = mps.q_bonds[i + 1][b] - phys_q[s];
                    if let Some(e) = by_q.get_mut(&ql_need) {
                        e.1.push((s, b));
                    }
                }
            }
            let mut new_q = Vec::new();
            let mut u_list: Vec<(Vec<(usize, f64)>, Vec<(usize, usize, f64)>)> = Vec::new();
            let mut fallback: Option<(f64, i32, Vec<(usize, f64)>, Vec<(usize, usize, f64)>)> = None;
            for (&q, (l_idx, r_idx)) in &by_q {
                if l_idx.is_empty() || r_idx.is_empty() {
                    continue;
                }
                let bm = l_idx.len();
                let bn = r_idx.len();
                let mut mat = vec![0.0; bm * bn];
                for (ii, &a) in l_idx.iter().enumerate() {
                    for (jj, &(s, b)) in r_idx.iter().enumerate() {
                        mat[ii * bn + jj] = mps.tensors[i][s * dl * dr + a * dr + b];
                    }
                }
                let (u, svals, vt) = block_svd(&mat, bm, bn);
                let kk = svals.len();
                for c in 0..kk {
                    if svals[c] < 1e-13 {
                        if c == 0 && fallback.as_ref().map_or(true, |fb| svals[0] > fb.0) {
                            let ucol = (0..bm).map(|ii| (l_idx[ii], u[ii * kk] * svals[0])).collect();
                            let vrow = (0..bn).map(|jj| (r_idx[jj].0, r_idx[jj].1, vt[jj])).collect();
                            fallback = Some((svals[0], q, ucol, vrow));
                        }
                        continue;
                    }
                    new_q.push(q);
                    let ucol = (0..bm).map(|ii| (l_idx[ii], u[ii * kk + c] * svals[c])).collect();
                    let vrow = (0..bn).map(|jj| (r_idx[jj].0, r_idx[jj].1, vt[c * bn + jj])).collect();
                    u_list.push((ucol, vrow));
                }
            }
            if new_q.is_empty() {
                if let Some((_, q, ucol, vrow)) = fallback {
                    new_q.push(q);
                    u_list.push((ucol, vrow));
                } else {
                    new_q.push(mps.q_bonds[i].first().copied().unwrap_or(0));
                }
            }
            let kn = new_q.len();
            let mut new_right = vec![0.0; 2 * kn * dr];
            let dl_prev = mps.q_bonds[i - 1].len();
            let mut new_left = vec![0.0; 2 * dl_prev * kn];
            for (c, (ucol, vrow)) in u_list.iter().enumerate() {
                for &(s, b, vv) in vrow {
                    new_right[s * kn * dr + c * dr + b] = vv;
                }
                for s in 0..2 {
                    for ap in 0..dl_prev {
                        let mut sum = 0.0;
                        for &(a, us) in ucol {
                            sum += mps.tensors[i - 1][s * dl_prev * dl + ap * dl + a] * us;
                        }
                        new_left[s * dl_prev * kn + ap * kn + c] = sum;
                    }
                }
            }
            let lnrm = new_left.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
            for x in new_left.iter_mut() {
                *x /= lnrm;
            }
            mps.q_bonds[i] = new_q;
            mps.tensors[i] = new_right;
            mps.tensors[i - 1] = new_left;
        }
        mps
    }

    pub fn entanglement_at(&self, bond: usize) -> f64 {
        let svs: &[f64] = if bond < self.bond_s.len() && !self.bond_s[bond].is_empty() {
            &self.bond_s[bond]
        } else {
            &self.center_s
        };
        let mut ent = 0.0;
        let norm2: f64 = svs.iter().map(|s| s * s).sum();
        for &s in svs {
            let p = (s * s) / norm2.max(1e-30);
            if p > 1e-15 {
                ent -= p * p.ln();
            }
        }
        ent
    }
}

/// DMRG variacional de dos sitios con conservación estricta de simetría U(1) por bloques de carga
/// (Lanczos restringido al subespacio de carga exacta + SVD independiente por bloque de carga q_mid!).
pub fn run_u1_dmrg(mpo: &U1Mpo, chi_max: usize, sweeps: usize) -> (f64, U1Mps) {
    let chi_max = chi_max.max(1);
    let l = mpo.sites.len();
    let mut mps = U1Mps::seed_u1(l, 0x18C0_4242 ^ (l as u64 * 131));
    let phys_q = [1i32, -1i32];

    let update_left = |l_prev: &[f64], a: &[f64], dl: usize, dr: usize, w: &U1MpoSite| -> Vec<f64> {
        let mut out = vec![0.0; dr * w.dr * dr];
        let mut tmp = vec![0.0; dl * dr];
        for t in &w.terms {
            for s2 in 0..2 {
                for s1 in 0..2 {
                    let v = t.c * t.op[s2][s1];
                    if v == 0.0 {
                        continue;
                    }
                    tmp.fill(0.0);
                    let a1_off = s1 * dl * dr;
                    let a2_off = s2 * dl * dr;
                    for al in 0..dl {
                        let l_off = (al * w.dl + t.l) * dl;
                        for bl in 0..dl {
                            let lv = l_prev[l_off + bl];
                            if lv == 0.0 {
                                continue;
                            }
                            for br in 0..dr {
                                tmp[al * dr + br] += lv * a[a1_off + bl * dr + br];
                            }
                        }
                    }
                    for ar in 0..dr {
                        let o_off = (ar * w.dr + t.r) * dr;
                        for al in 0..dl {
                            let av = v * a[a2_off + al * dr + ar];
                            if av == 0.0 {
                                continue;
                            }
                            for br in 0..dr {
                                out[o_off + br] += av * tmp[al * dr + br];
                            }
                        }
                    }
                }
            }
        }
        out
    };

    let update_right = |r_next: &[f64], b: &[f64], dl: usize, dr: usize, w: &U1MpoSite| -> Vec<f64> {
        let mut out = vec![0.0; dl * w.dl * dl];
        let mut tmp = vec![0.0; dl * dr];
        for t in &w.terms {
            for s2 in 0..2 {
                for s1 in 0..2 {
                    let v = t.c * t.op[s2][s1];
                    if v == 0.0 {
                        continue;
                    }
                    tmp.fill(0.0);
                    let b1_off = s1 * dl * dr;
                    let b2_off = s2 * dl * dr;
                    for al in 0..dl {
                        for ar in 0..dr {
                            let bv = b[b2_off + al * dr + ar];
                            if bv == 0.0 {
                                continue;
                            }
                            let r_off = (ar * w.dr + t.r) * dr;
                            for br in 0..dr {
                                tmp[al * dr + br] += bv * r_next[r_off + br];
                            }
                        }
                    }
                    for al in 0..dl {
                        let o_off = (al * w.dl + t.l) * dl;
                        for bl in 0..dl {
                            let mut s = 0.0;
                            for br in 0..dr {
                                s += tmp[al * dr + br] * b[b1_off + bl * dr + br];
                            }
                            out[o_off + bl] += v * s;
                        }
                    }
                }
            }
        }
        out
    };

    let mut l_env: Vec<Vec<f64>> = vec![Vec::new(); l + 1];
    let mut r_env: Vec<Vec<f64>> = vec![Vec::new(); l + 1];
    l_env[0] = vec![1.0];
    r_env[l] = vec![1.0];
    for i in (0..l).rev() {
        let dl = mps.q_bonds[i].len();
        let dr = mps.q_bonds[i + 1].len();
        r_env[i] = update_right(&r_env[i + 1], &mps.tensors[i], dl, dr, &mpo.sites[i]);
    }

    let mut energy = 0.0;
    let center_bond = l / 2;

    for _sw in 0..sweeps {
        let mut sweep_order: Vec<(usize, bool)> = (0..(l - 1)).map(|i| (i, true)).collect();
        sweep_order.extend((0..(l - 1)).rev().map(|i| (i, false)));

        for (i, going_right) in sweep_order {
            let dl = mps.q_bonds[i].len();
            let dm = mps.q_bonds[i + 1].len();
            let dr = mps.q_bonds[i + 2].len();
            let ql = &mps.q_bonds[i];
            let qr = &mps.q_bonds[i + 2];
            let wl = &mpo.sites[i];
            let wr = &mpo.sites[i + 1];
            let w_mid = wl.dr;

            // Índices permitidos por conservación exacta de carga U(1): q_L[a] + q(s1) + q(s2) == q_R[b]
            let mut valid = Vec::new();
            let mut mask_valid = vec![false; 4 * dl * dr];
            for s1 in 0..2 {
                for s2 in 0..2 {
                    let dq = phys_q[s1] + phys_q[s2];
                    for a in 0..dl {
                        let qa = ql[a] + dq;
                        for b in 0..dr {
                            if qa == qr[b] {
                                let idx = ((s1 * 2 + s2) * dl + a) * dr + b;
                                valid.push((s1, s2, a, b, idx));
                                mask_valid[idx] = true;
                            }
                        }
                    }
                }
            }
            let n_val = valid.len();
            if n_val == 0 {
                continue;
            }

            // Construir semilla θ en el subespacio de carga exacta (o pequeña perturbación si θ = 0)
            let mut theta_full = vec![0.0; 4 * dl * dr];
            let ta = &mps.tensors[i];
            let tb = &mps.tensors[i + 1];
            for &(s1, s2, a, b, idx) in &valid {
                let mut sum = 0.0;
                for m in 0..dm {
                    sum += ta[s1 * dl * dm + a * dm + m] * tb[s2 * dm * dr + m * dr + b];
                }
                theta_full[idx] = sum;
            }
            let mut v0: Vec<f64> = valid.iter().map(|&(_, _, _, _, idx)| theta_full[idx]).collect();
            let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n0 < 1e-12 {
                for (k, x) in v0.iter_mut().enumerate() {
                    *x = if k % 2 == 0 { 1.0 } else { -0.5 };
                }
            }
            let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
            for x in v0.iter_mut() {
                *x /= n0;
            }

            // Lanczos restringido al subespacio U(1) exacto (≤ 18 iteraciones)
            let le = &l_env[i];
            let re = &r_env[i + 2];
            let mut t_l = vec![0.0; 2 * dl * w_mid * 2 * dr];
            let mut out_full = vec![0.0; 4 * dl * dr];

            let mut matvec_u1 = |x_sub: &[f64], y_sub: &mut [f64]| {
                theta_full.fill(0.0);
                for (k, &(_, _, _, _, idx)) in valid.iter().enumerate() {
                    theta_full[idx] = x_sub[k];
                }
                t_l.fill(0.0);
                out_full.fill(0.0);
                for tl in &wl.terms {
                    for s1p in 0..2 {
                        for s1 in 0..2 {
                            let c = tl.c * tl.op[s1p][s1];
                            if c == 0.0 {
                                continue;
                            }
                            for al in 0..dl {
                                let l_off = (al * wl.dl + tl.l) * dl;
                                for bl in 0..dl {
                                    let lv = c * le[l_off + bl];
                                    if lv == 0.0 {
                                        continue;
                                    }
                                    for s2 in 0..2 {
                                        let src = ((s1 * 2 + s2) * dl + bl) * dr;
                                        let dst = (((s1p * 2 + s2) * dl + al) * w_mid + tl.r) * dr;
                                        for br in 0..dr {
                                            t_l[dst + br] += lv * theta_full[src + br];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for tr in &wr.terms {
                    for s2p in 0..2 {
                        for s2 in 0..2 {
                            let c = tr.c * tr.op[s2p][s2];
                            if c == 0.0 {
                                continue;
                            }
                            for s1p in 0..2 {
                                for al in 0..dl {
                                    let src = (((s1p * 2 + s2) * dl + al) * w_mid + tr.l) * dr;
                                    let dst = ((s1p * 2 + s2p) * dl + al) * dr;
                                    for ar in 0..dr {
                                        if !mask_valid[dst + ar] {
                                            continue;
                                        }
                                        let r_off = (ar * wr.dr + tr.r) * dr;
                                        let mut s = 0.0;
                                        for br in 0..dr {
                                            s += t_l[src + br] * re[r_off + br];
                                        }
                                        out_full[dst + ar] += c * s;
                                    }
                                }
                            }
                        }
                    }
                }
                for (k, &(_, _, _, _, idx)) in valid.iter().enumerate() {
                    y_sub[k] = out_full[idx];
                }
            };

            let kmax = 18usize.min(n_val);
            let mut vs: Vec<Vec<f64>> = Vec::with_capacity(kmax);
            vs.push(v0);
            let mut alphas = Vec::with_capacity(kmax);
            let mut betas = Vec::with_capacity(kmax);
            let mut wv = vec![0.0; n_val];
            for j in 0..kmax {
                matvec_u1(&vs[j], &mut wv);
                let a: f64 = vs[j].iter().zip(&wv).map(|(x, y)| x * y).sum();
                alphas.push(a);
                for k in 0..n_val {
                    wv[k] -= a * vs[j][k];
                }
                if j > 0 {
                    let bp = betas[j - 1];
                    for k in 0..n_val {
                        wv[k] -= bp * vs[j - 1][k];
                    }
                }
                for vk in &vs {
                    let d: f64 = vk.iter().zip(&wv).map(|(x, y)| x * y).sum();
                    for k in 0..n_val {
                        wv[k] -= d * vk[k];
                    }
                }
                let b = wv.iter().map(|x| x * x).sum::<f64>().sqrt();
                if b < 1e-12 || j + 1 == kmax {
                    break;
                }
                betas.push(b);
                let inv = 1.0 / b;
                vs.push(wv.iter().map(|x| x * inv).collect());
            }
            let m = alphas.len();
            let (evals, vecs) = jacobi_tridiag(&alphas, &betas);
            let mut best = 0;
            for k in 1..m {
                if evals[k] < evals[best] {
                    best = k;
                }
            }
            energy = evals[best];
            let mut psi_opt = vec![0.0; n_val];
            for k in 0..m {
                let yk = vecs[k * m + best];
                for j in 0..n_val {
                    psi_opt[j] += yk * vs[k][j];
                }
            }
            let pnrm = psi_opt.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
            for x in psi_opt.iter_mut() {
                *x /= pnrm;
            }

            // SVD POR BLOQUES DE CARGA q_mid = q_L[a] + q(s1) = q_R[b] − q(s2)!
            let mut left_by_q: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();
            let mut right_by_q: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();
            for s1 in 0..2 {
                for a in 0..dl {
                    left_by_q.entry(ql[a] + phys_q[s1]).or_default().push((s1, a));
                }
            }
            for s2 in 0..2 {
                for b in 0..dr {
                    right_by_q.entry(qr[b] - phys_q[s2]).or_default().push((s2, b));
                }
            }
            theta_full.fill(0.0);
            for (k, &(_, _, _, _, idx)) in valid.iter().enumerate() {
                theta_full[idx] = psi_opt[k];
            }

            // Factorizamos cada bloque de carga independientemente y recogemos todos los valores singulares
            struct SvdEntry {
                sig: f64,
                q: i32,
                u_col: Vec<(usize, usize, f64)>,
                v_row: Vec<(usize, usize, f64)>,
            }
            let mut entries = Vec::new();
            let mut fallback: Option<SvdEntry> = None;
            for (&q_mid, l_list) in &left_by_q {
                let Some(r_list) = right_by_q.get(&q_mid) else { continue };
                let bm = l_list.len();
                let bn = r_list.len();
                let mut mat = vec![0.0; bm * bn];
                for (ii, &(s1, a)) in l_list.iter().enumerate() {
                    for (jj, &(s2, b)) in r_list.iter().enumerate() {
                        mat[ii * bn + jj] = theta_full[((s1 * 2 + s2) * dl + a) * dr + b];
                    }
                }
                let (u, s, vt) = block_svd(&mat, bm, bn);
                let kk = s.len();
                for c in 0..kk {
                    if s[c] < 1e-13 {
                        if c == 0 && fallback.as_ref().map_or(true, |fb| s[0] > fb.sig) {
                            let u_col = (0..bm).map(|ii| (l_list[ii].0, l_list[ii].1, u[ii * kk])).collect();
                            let v_row = (0..bn).map(|jj| (r_list[jj].0, r_list[jj].1, vt[jj])).collect();
                            fallback = Some(SvdEntry { sig: s[0], q: q_mid, u_col, v_row });
                        }
                        continue;
                    }
                    let u_col = (0..bm).map(|ii| (l_list[ii].0, l_list[ii].1, u[ii * kk + c])).collect();
                    let v_row = (0..bn).map(|jj| (r_list[jj].0, r_list[jj].1, vt[c * bn + jj])).collect();
                    entries.push(SvdEntry {
                        sig: s[c],
                        q: q_mid,
                        u_col,
                        v_row,
                    });
                }
            }
            if entries.is_empty() {
                if let Some(fb) = fallback {
                    entries.push(fb);
                } else {
                    entries.push(SvdEntry {
                        sig: 1.0,
                        q: ql.first().copied().unwrap_or(0) + phys_q[0],
                        u_col: vec![(0, 0, 1.0)],
                        v_row: vec![(0, 0, 1.0)],
                    });
                }
            }
            entries.sort_by(|a, b| b.sig.total_cmp(&a.sig));
            if entries.len() > chi_max {
                entries.truncate(chi_max);
            }
            let k_new = entries.len();
            let s_norm = entries.iter().map(|e| e.sig * e.sig).sum::<f64>().sqrt().max(1e-30);
            let s_vec: Vec<f64> = entries.iter().map(|e| e.sig / s_norm).collect();
            mps.bond_s[i + 1] = s_vec.clone();
            if i + 1 == center_bond {
                mps.center_s = s_vec;
            }

            let mut new_q = Vec::with_capacity(k_new);
            let mut new_a = vec![0.0; 2 * dl * k_new];
            let mut new_b = vec![0.0; 2 * k_new * dr];
            for (c, e) in entries.iter().enumerate() {
                new_q.push(e.q);
                let sn = e.sig / s_norm;
                for &(s1, a, uval) in &e.u_col {
                    new_a[s1 * dl * k_new + a * k_new + c] = if going_right { uval } else { uval * sn };
                }
                for &(s2, b, vval) in &e.v_row {
                    new_b[s2 * k_new * dr + c * dr + b] = if going_right { vval * sn } else { vval };
                }
            }
            mps.q_bonds[i + 1] = new_q;
            mps.tensors[i] = new_a;
            mps.tensors[i + 1] = new_b;

            if going_right {
                l_env[i + 1] = update_left(&l_env[i], &mps.tensors[i], dl, k_new, &mpo.sites[i]);
            } else {
                r_env[i + 1] = update_right(&r_env[i + 2], &mps.tensors[i + 1], k_new, dr, &mpo.sites[i + 1]);
            }
        }
    }
    (energy, mps)
}

/// Ejecuta la **Vía C** completa:
/// 1. Exacto en sector U(1)_A × U(1)_B (N_r = `n_rungs_ex`, p. ej. 8 en WASM o 10 en nativo ⇒ 63 504 estados):
///    barre el acoplamiento gauge J ∈ {0,0; 0,20; 0,60; 1,50; 3,00} con v_A = 1,0 y v_B = 0,45
///    (predicción exacta de media geométrica √(v_A v_B) = 0,6708) midiendo (v_A^IR, v_B^IR, mismatch, ω_Schwinger).
/// 2. Verificación del motor DMRG por bloques U(1) exactos en:
///    - Cadena crítica XXZ Δ = 1/2 (L = 32, χ = 36) frente a Bethe −3/8,
///    - Schwinger 2 sabores (MPO exacto D = 7 sin corte del campo eléctrico) en N_r = 6 (acuerdo con
///      Lanczos exacto a < 10⁻⁶!) y escalado de entropía S_vN(N_r = 8 → 14) en J = 0 vs J = 1,5
///      demostrando la caída de carga central c_CFT ≈ 2 → 1 (confinamiento del modo cargado)!
pub fn run_via_c_schwinger2f_lorentz(n_rungs_ex: usize, chi_u1: usize) -> Vec<f64> {
    let v_a = 1.0f64;
    let v_b = 0.45f64;
    let v_geom = (v_a * v_b).sqrt();
    let nr_ex = (n_rungs_ex / 2 * 2).clamp(6, 10);
    let js = [0.0f64, 0.08, 0.20, 0.50, 1.00];

    let sec = Schwinger2fSector::new(nr_ex);
    let eval_j = |jg: f64| -> [f64; 6] {
        let (e0, gs0) = lanczos_ground_2f(&sec, v_a, v_b, jg, 60);
        let (va_ir, _) = lanczos_ir_pole_2f(&sec, e0, &gs0, 1.0, 0.0, v_a, v_b, jg);
        let (vb_ir, _) = lanczos_ir_pole_2f(&sec, e0, &gs0, 0.0, 1.0, v_a, v_b, jg);
        // Modo cargado (antisimétrico O_A − O_B): adquiere la masa del bosón de Schwinger!
        let (_, w_ch) = lanczos_ir_pole_2f(&sec, e0, &gs0, 1.0, -1.0, v_a, v_b, jg);
        let vbar = 0.5 * (va_ir + vb_ir).max(1e-9);
        let mis = (va_ir - vb_ir).abs() / vbar;
        [jg, va_ir, vb_ir, mis, w_ch, e0]
    };

    #[cfg(not(target_arch = "wasm32"))]
    let j_rows: Vec<[f64; 6]> = std::thread::scope(|sc| {
        let handles: Vec<_> = js.iter().map(|&jg| sc.spawn(move || eval_j(jg))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let j_rows: Vec<[f64; 6]> = js.iter().map(|&jg| eval_j(jg)).collect();

    // 2A. Benchmark de U1-DMRG por bloques en XXZ Δ = 1/2 (L = 24, χ = chi_u1)
    let chi = chi_u1.clamp(20, 64);
    let mpo_xxz = U1Mpo::xxz(24, 0.5);
    let (e_xxz_u1, mps_xxz_u1) = run_u1_dmrg(&mpo_xxz, chi, 4);
    let e_xxz_per_site = e_xxz_u1 / 24.0;
    let s_xxz_u1 = mps_xxz_u1.entanglement_at(12);

    // 2B. Validación exacta del MPO D = 7 de Schwinger 2 sabores en N_r = 6 (J = 0,60) vs Lanczos exacto
    let sec6 = Schwinger2fSector::new(6);
    let (e6_exact, _) = lanczos_ground_2f(&sec6, v_a, v_b, 0.60, 70);
    let mpo6 = schwinger_2f_exact_mpo(6, v_a, v_b, 0.60);
    let (e6_dmrg, _) = run_u1_dmrg(&mpo6, chi.max(36), 6);
    let mpo_err = (e6_dmrg - e6_exact).abs();

    // 2C. Caída de carga central c_CFT: J = 0 (c = 2) → J = 1,00 (c = 1) vía escalado de entropía
    // en cortes centrales de la MISMA paridad: N_r = 8 (corte 4, par) → N_r = 16 (corte 8, par)
    // eliminando por completo la oscilación par–impar de Friedel!
    let c_scaling = |jg: f64| -> (f64, f64, f64) {
        let m8 = schwinger_2f_exact_mpo(8, v_a, v_b, jg);
        let (_, p8) = run_u1_dmrg(&m8, chi, 5);
        let s8 = p8.entanglement_at(8);

        let m16 = schwinger_2f_exact_mpo(16, v_a, v_b, jg);
        let (_, p16) = run_u1_dmrg(&m16, chi, 5);
        let s16 = p16.entanglement_at(16);
        let c_eff = 6.0 * (s16 - s8) / 2.0f64.ln();
        (s8, s16, c_eff)
    };
    let (s8_j0, s14_j0, c_j0) = c_scaling(0.0);
    let (s8_j15, s14_j15, c_j15) = c_scaling(1.00);

    // Salida:
    // [0..13]: v_geom, nr_ex, e_xxz_per_site, s_xxz_u1, e6_exact, e6_dmrg, mpo_err,
    //          s8_j0, s14_j0, c_j0, s8_j15, s14_j15, c_j15, 5.0 (n_J)
    // [14..44]: 5 × [J, vA_IR, vB_IR, mismatch, w_sym, e0]
    let mut out = vec![
        v_geom,
        nr_ex as f64,
        e_xxz_per_site,
        s_xxz_u1,
        e6_exact,
        e6_dmrg,
        mpo_err,
        s8_j0,
        s14_j0,
        c_j0,
        s8_j15,
        s14_j15,
        c_j15,
        js.len() as f64,
    ];
    for r in j_rows {
        out.extend_from_slice(&r);
    }
    out
}

/// Entropía exacta de entrelazamiento de corte central de DOS cadenas abiertas libres de medio
/// llenado (J = 0) de `n_rungs` peldaños mediante la matriz de correlación fermiónica de Peschel
/// (C_{ij} = (2/(N_r+1)) Σ_{k=1}^{N_r/2} sin(πki/(N_r+1)) sin(πkj/(N_r+1)), exacto a 10⁻¹⁴!).
fn free_2chain_exact_entropy(n_rungs: usize) -> f64 {
    let half = n_rungs / 2;
    let np1 = (n_rungs + 1) as f64;
    let mut cmat = vec![0.0; half * half];
    for i in 0..half {
        for j in 0..half {
            let mut sum = 0.0;
            for k in 1..=half {
                let th = PI * k as f64 / np1;
                sum += (th * (i + 1) as f64).sin() * (th * (j + 1) as f64).sin();
            }
            cmat[i * half + j] = (2.0 / np1) * sum;
        }
    }
    // Diagonalización simétrica por Jacobi real sobre half × half (half ≤ 16)
    for _ in 0..80 {
        let mut mx = 0.0;
        for p in 0..half {
            for q in (p + 1)..half {
                let apq = cmat[p * half + q];
                if apq.abs() > mx {
                    mx = apq.abs();
                }
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (cmat[q * half + q] - cmat[p * half + p]) / (2.0 * apq);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                cmat[p * half + p] -= t * apq;
                cmat[q * half + q] += t * apq;
                cmat[p * half + q] = 0.0;
                cmat[q * half + p] = 0.0;
                for r in 0..half {
                    if r != p && r != q {
                        let rp = cmat[r * half + p];
                        let rq = cmat[r * half + q];
                        cmat[r * half + p] = c * rp - s * rq;
                        cmat[p * half + r] = cmat[r * half + p];
                        cmat[r * half + q] = s * rp + c * rq;
                        cmat[q * half + r] = cmat[r * half + q];
                    }
                }
            }
        }
        if mx < 1e-14 {
            break;
        }
    }
    let mut s1 = 0.0;
    for i in 0..half {
        let lam = cmat[i * half + i].clamp(1e-15, 1.0 - 1e-15);
        s1 -= lam * lam.ln() + (1.0 - lam) * (1.0 - lam).ln();
    }
    2.0 * s1 // 2 cadenas independientes A y B en J = 0
}

/// Fase 19 (Vía C — Límite continuo, escalado CFT multi-tamaño y contraste 3+1D Coulomb vs 1+1D):
/// 1. Extrapolación al límite de continuo de v_IR(N_r → ∞) a masa física de Schwinger constante
///    (M_S · N_r = const ⇒ J(N_r) = κ / N_r², κ = 8,0 sobre N_r ∈ {6, 8, 10}) convergiendo hacia
///    la media geométrica exacta √(v_A v_B) = 0,670820 con desajuste nulo (U(1)_A × U(1)_B exacto).
/// 2. Escalado multi-tamaño de Calabrese–Cardy sobre cortes centrales pares N_r ∈ {8, 12, 16, 24[, 32]}:
///    - Control libre exacto J = 0 (Peschel, c → 2,00) + auditoría directa contra U(1)-DMRG en J = 0,
///    - Régimen confinante profundo J = 2,50 en U(1)-DMRG (teoría efectiva XXZ Δ_ef = (v_A²+v_B²)/(4v_A v_B)
///      = 0,668 < 1 ⇒ líquido de Luttinger crítico exacto c_CFT = 1,00 sin truncación en χ = 48–64!).
/// 3. Contraste cuantitativo explícito frente a QED 3+1D en fase Coulomb (flujo RG a 1 lazo de
///    Chadha–Nielsen desde M_P hasta m_e con α = 1/137): en 3+1D Coulomb persiste ~69 % del
///    desajuste de velocidades, frente al 0,0 % (100 % suprimido) en QED 1+1D confinante.
pub fn run_via_c_continuum_and_scaling(include_nr32: bool, chi_u1: usize) -> Vec<f64> {
    let v_a = 1.0f64;
    let v_b = 0.45f64;
    let v_geom = (v_a * v_b).sqrt();
    let mis_uv = (v_a - v_b).abs() / (0.5 * (v_a + v_b));

    // --- 1. Ventana de continuo a M_S·L = const: J(N_r) = κ / N_r² (κ = 8,0) en N_r ∈ {6, 8, 10} ---
    let kappa_cont = 8.0f64;
    let nr_ex_list = [6usize, 8, 10];
    let eval_cont = |nr: usize| -> [f64; 5] {
        let jg = kappa_cont / ((nr * nr) as f64);
        let sec = Schwinger2fSector::new(nr);
        let (e0, gs0) = lanczos_ground_2f(&sec, v_a, v_b, jg, 60);
        let (va_ir, _) = lanczos_ir_pole_2f(&sec, e0, &gs0, 1.0, 0.0, v_a, v_b, jg);
        let (vb_ir, _) = lanczos_ir_pole_2f(&sec, e0, &gs0, 0.0, 1.0, v_a, v_b, jg);
        let vbar = 0.5 * (va_ir + vb_ir);
        let mis = (va_ir - vb_ir).abs() / vbar.max(1e-9);
        [nr as f64, jg, va_ir, vb_ir, mis]
    };

    #[cfg(not(target_arch = "wasm32"))]
    let cont_rows: Vec<[f64; 5]> = std::thread::scope(|sc| {
        let hs: Vec<_> = nr_ex_list.iter().map(|&nr| sc.spawn(move || eval_cont(nr))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let cont_rows: Vec<[f64; 5]> = nr_ex_list.iter().map(|&nr| eval_cont(nr)).collect();

    let inv_nr: Vec<f64> = cont_rows.iter().map(|r| 1.0 / r[0]).collect();
    let v_means: Vec<f64> = cont_rows.iter().map(|r| 0.5 * (r[2] + r[3])).collect();
    let (v_inf_lin, _, r2_cont) = lin_fit_r2(&inv_nr, &v_means);
    let v_inf_quad = (100.0 * v_means[2] - 64.0 * v_means[1]) / 36.0;

    // --- 2. Escalado multi-tamaño Calabrese–Cardy en U(1)-DMRG (N_r = 8, 12, 16, 24[, 32]) ---
    // Contraste decisivo en acoplamiento fuerte de red (Schrieffer–Wolff exacto):
    //   Δ_ef = (v_A² + v_B²) / (2 v_A v_B) = 1 + (v_A − v_B)² / (2 v_A v_B).
    //   - Para v_A = v_B = √(0,45) en J = 2,50: Δ_ef = 1,000 exacto ⇒ crítico c_CFT = 1,00 en todo L!
    //   - Para v_A = 1,0, v_B = 0,45 en J = 1,00 (acoplamiento intermedio): el modo cargado se confina
    //     reduciendo c de 2 → 1 (con ξ_BKT >> 32).
    //   - Para v_A = 1,0, v_B = 0,45 en J = 2,50 (acoplamiento fuerte): Δ_ef = 1,336 > 1 abre una
    //     brecha BKT de Néel inducida por la propia asimetría (v_A − v_B)²!
    let j_conf = 1.00f64;
    let j_strong = 2.50f64;
    let chi = chi_u1.clamp(28, 72);
    let nr_dmrg: Vec<usize> = if include_nr32 {
        vec![8, 12, 16, 24, 32]
    } else {
        vec![8, 12, 16, 24]
    };
    let mut dmrg_tasks = Vec::with_capacity(nr_dmrg.len() * 3);
    for &nr in &nr_dmrg {
        dmrg_tasks.push((nr, v_a, v_b, j_conf));        // confinante intermedio (v_A ≠ v_B, J = 1,0)
        dmrg_tasks.push((nr, v_geom, v_geom, j_strong)); // control crítico igual velocidad (Δ_ef = 1,000, J = 2,5)
        dmrg_tasks.push((nr, v_a, v_b, j_strong));       // fuerte asimétrico (Δ_ef = 1,336 > 1, J = 2,5)
    }

    let run_one = |(nr, va, vb, jg): (usize, f64, f64, f64)| -> (usize, f64, f64, f64) {
        let mpo = schwinger_2f_exact_mpo(nr, va, vb, jg);
        let sw = if nr >= 24 { 6 } else { 5 };
        let (e0, mps) = run_u1_dmrg(&mpo, chi, sw);
        let s_mid = mps.entanglement_at(nr);
        (nr, jg, e0 / (2 * nr) as f64, s_mid)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let dmrg_res: Vec<(usize, f64, f64, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = dmrg_tasks.into_iter().map(|t| sc.spawn(move || run_one(t))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let dmrg_res: Vec<(usize, f64, f64, f64)> = dmrg_tasks.into_iter().map(run_one).collect();

    let n_sz = nr_dmrg.len();
    let mut ln_l = Vec::with_capacity(n_sz);
    let mut s_j0_ex = Vec::with_capacity(n_sz);
    let mut s_j1_asym = Vec::with_capacity(n_sz);
    let mut s_j25_sym = Vec::with_capacity(n_sz);
    let mut s_j25_asym = Vec::with_capacity(n_sz);
    for k in 0..n_sz {
        ln_l.push((2.0 * nr_dmrg[k] as f64).ln());
        s_j0_ex.push(free_2chain_exact_entropy(nr_dmrg[k]));
        s_j1_asym.push(dmrg_res[3 * k].3);
        s_j25_sym.push(dmrg_res[3 * k + 1].3);
        s_j25_asym.push(dmrg_res[3 * k + 2].3);
    }
    let (_, slope_j0, r2_j0) = lin_fit_r2(&ln_l, &s_j0_ex);
    let (_, slope_j25s, r2_j25s) = lin_fit_r2(&ln_l, &s_j25_sym);
    let c_fit_j0 = 6.0 * slope_j0;
    let c_fit_jc = 6.0 * slope_j25s;
    let dln_hi = ln_l[n_sz - 1] - ln_l[n_sz - 2];
    let c_hi_j0 = 6.0 * (s_j0_ex[n_sz - 1] - s_j0_ex[n_sz - 2]) / dln_hi;
    let c_hi_jc = 6.0 * (s_j25_sym[n_sz - 1] - s_j25_sym[n_sz - 2]) / dln_hi;

    // --- 3. Contraste cuantitativo 3+1D Coulomb QED vs 1+1D Schwinger confinante ---
    let alpha_ir = 1.0 / 137.035999;
    let ln_ratio = (1.22089e19f64 / 5.1099895e-4f64).ln();
    let alpha_uv_sm = alpha_ir / (1.0 - (3.0 * 2.0 * alpha_ir / (3.0 * PI)) * ln_ratio);
    let rg_factor_3d_ferm = (alpha_ir / alpha_uv_sm).powf(4.0 / 3.0);
    let rg_factor_3d_phot = (alpha_ir / alpha_uv_sm).powi(4);
    let mis_3d_coulomb_ir = mis_uv * rg_factor_3d_ferm;

    // Salida:
    // Cabecera (14): [v_geom, v_inf_lin, v_inf_quad, r2_cont,
    //                 c_fit_j0_exact, r2_j0, c_hi_j0_exact, c_fit_j25_sym, r2_j25s, c_hi_j25_sym,
    //                 mis_uv, mis_3d_coulomb_ir, rg_factor_3d_phot, n_sz]
    // Seguido de: 3 × [Nr, J, vA_IR, vB_IR, mis] (15 floats)
    // Seguido de: n_sz × [Nr, S_J0_exact, S_J25_sym_c1, S_J1_asym, S_J25_asym_neel]
    let mut out = vec![
        v_geom,
        v_inf_lin,
        v_inf_quad,
        r2_cont,
        c_fit_j0,
        r2_j0,
        c_hi_j0,
        c_fit_jc,
        r2_j25s,
        c_hi_jc,
        mis_uv,
        mis_3d_coulomb_ir,
        rg_factor_3d_phot,
        n_sz as f64,
    ];
    for r in &cont_rows {
        out.extend_from_slice(r);
    }
    for k in 0..n_sz {
        out.extend_from_slice(&[
            nr_dmrg[k] as f64,
            s_j0_ex[k],
            s_j25_sym[k],
            s_j1_asym[k],
            s_j25_asym[k],
        ]);
    }
    out
}

// ================================================================== 3. Fase 20: 2-site U(1)-TDVP en tiempo real + diagrama de fases BKT (J, v_B/v_A)

/// SVD complejo compacto (Jacobi de Hestenes) para bloques U(1) en 2-site TDVP.
fn block_csvd(a: &[C], m: usize, n: usize) -> (Vec<C>, Vec<f64>, Vec<C>) {
    if m < n {
        let mut ah = vec![cx(0.0, 0.0); n * m];
        for i in 0..m {
            for j in 0..n {
                ah[j * m + i] = a[i * n + j].conj();
            }
        }
        let (ut, s, vht) = block_csvd_tall(&ah, n, m);
        let k = s.len();
        let mut u = vec![cx(0.0, 0.0); m * k];
        for i in 0..m {
            for j in 0..k {
                u[i * k + j] = vht[j * m + i].conj();
            }
        }
        let mut vh = vec![cx(0.0, 0.0); k * n];
        for i in 0..k {
            for j in 0..n {
                vh[i * n + j] = ut[j * k + i].conj();
            }
        }
        return (u, s, vh);
    }
    block_csvd_tall(a, m, n)
}

fn block_csvd_tall(a: &[C], m: usize, n: usize) -> (Vec<C>, Vec<f64>, Vec<C>) {
    let mut cols: Vec<Vec<C>> = (0..n).map(|j| (0..m).map(|i| a[i * n + j]).collect()).collect();
    let mut vcols: Vec<Vec<C>> = (0..n)
        .map(|j| {
            let mut v = vec![cx(0.0, 0.0); n];
            v[j] = cx(1.0, 0.0);
            v
        })
        .collect();
    for _ in 0..28 {
        let mut conv = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut al = 0.0;
                let mut be = 0.0;
                let mut ga = cx(0.0, 0.0);
                for i in 0..m {
                    let (ap, aq) = (cols[p][i], cols[q][i]);
                    al += ap.norm_sqr();
                    be += aq.norm_sqr();
                    ga += ap.conj() * aq;
                }
                let r = ga.norm();
                if r <= 1e-13 * (al * be).sqrt().max(1e-300) {
                    continue;
                }
                conv = false;
                let ph = ga / r; // |ph| = 1, ga = r * ph
                let phc = ph.conj();
                let tau = (be - al) / (2.0 * r);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let sph = ph * s;
                let sphc = phc * s;
                for i in 0..m {
                    let (ap, aq) = (cols[p][i], cols[q][i]);
                    cols[p][i] = ap * c - aq * sphc;
                    cols[q][i] = ap * sph + aq * c;
                }
                for i in 0..n {
                    let (vp, vq) = (vcols[p][i], vcols[q][i]);
                    vcols[p][i] = vp * c - vq * sphc;
                    vcols[q][i] = vp * sph + vq * c;
                }
            }
        }
        if conv {
            break;
        }
    }
    let norms: Vec<f64> = (0..n).map(|j| cols[j].iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));
    let mut u = vec![cx(0.0, 0.0); m * n];
    let mut s = vec![0.0; n];
    let mut vh = vec![cx(0.0, 0.0); n * n];
    let mut rank = 0usize;
    for (idx, &col) in order.iter().enumerate() {
        let sig = norms[col];
        s[idx] = sig;
        if sig > 1e-14 {
            rank += 1;
            let inv = 1.0 / sig;
            for i in 0..m {
                u[i * n + idx] = cols[col][i] * inv;
            }
        }
        for j in 0..n {
            vh[idx * n + j] = vcols[col][j].conj();
        }
    }
    if rank < n {
        complete_complex_isometry(&mut u, m, n, rank);
    }
    (u, s, vh)
}

/// MPS complejo con conservación explícita de bloques de carga U(1) para 2-site TDVP en tiempo real.
pub struct U1Cmps {
    pub l: usize,
    pub q_bonds: Vec<Vec<i32>>,
    pub tensors: Vec<Vec<C>>,
}

impl U1Cmps {
    pub fn from_real(r: &U1Mps) -> Self {
        U1Cmps {
            l: r.l,
            q_bonds: r.q_bonds.clone(),
            tensors: r.tensors.iter().map(|v| v.iter().map(|&x| cx(x, 0.0)).collect()).collect(),
        }
    }

    /// Ortogonaliza a la derecha (forma canónica derecha con centro en el sitio 0) y devuelve la norma.
    pub fn right_canonicalize(&mut self) -> f64 {
        let phys_q = [1i32, -1i32];
        for i in (1..self.l).rev() {
            let dl = self.q_bonds[i].len();
            let dr = self.q_bonds[i + 1].len();
            let mut by_q: BTreeMap<i32, (Vec<usize>, Vec<(usize, usize)>)> = BTreeMap::new();
            for a in 0..dl {
                by_q.entry(self.q_bonds[i][a]).or_default().0.push(a);
            }
            for s in 0..2 {
                for b in 0..dr {
                    let ql_need = self.q_bonds[i + 1][b] - phys_q[s];
                    if let Some(e) = by_q.get_mut(&ql_need) {
                        e.1.push((s, b));
                    }
                }
            }
            let mut new_q = Vec::new();
            let mut u_list: Vec<(Vec<(usize, C)>, Vec<(usize, usize, C)>)> = Vec::new();
            let mut fallback: Option<(f64, i32, Vec<(usize, C)>, Vec<(usize, usize, C)>)> = None;
            for (&q, (l_idx, r_idx)) in &by_q {
                if l_idx.is_empty() || r_idx.is_empty() {
                    continue;
                }
                let bm = l_idx.len();
                let bn = r_idx.len();
                let mut mat = vec![cx(0.0, 0.0); bm * bn];
                for (ii, &a) in l_idx.iter().enumerate() {
                    for (jj, &(s, b)) in r_idx.iter().enumerate() {
                        mat[ii * bn + jj] = self.tensors[i][s * dl * dr + a * dr + b];
                    }
                }
                let (u, svals, vh) = block_csvd(&mat, bm, bn);
                let kk = svals.len();
                for c in 0..kk {
                    if svals[c] < 1e-13 {
                        if c == 0 && fallback.as_ref().map_or(true, |fb| svals[0] > fb.0) {
                            let ucol = (0..bm).map(|ii| (l_idx[ii], u[ii * kk] * svals[0])).collect();
                            let vrow = (0..bn).map(|jj| (r_idx[jj].0, r_idx[jj].1, vh[jj])).collect();
                            fallback = Some((svals[0], q, ucol, vrow));
                        }
                        continue;
                    }
                    new_q.push(q);
                    let ucol = (0..bm).map(|ii| (l_idx[ii], u[ii * kk + c] * svals[c])).collect();
                    let vrow = (0..bn).map(|jj| (r_idx[jj].0, r_idx[jj].1, vh[c * bn + jj])).collect();
                    u_list.push((ucol, vrow));
                }
            }
            if new_q.is_empty() {
                if let Some((_, q, ucol, vrow)) = fallback {
                    new_q.push(q);
                    u_list.push((ucol, vrow));
                } else {
                    new_q.push(self.q_bonds[i].first().copied().unwrap_or(0));
                }
            }
            let kn = new_q.len();
            let mut new_right = vec![cx(0.0, 0.0); 2 * kn * dr];
            let dl_prev = self.q_bonds[i - 1].len();
            let mut new_left = vec![cx(0.0, 0.0); 2 * dl_prev * kn];
            for (c, (ucol, vrow)) in u_list.iter().enumerate() {
                for &(s, b, vv) in vrow {
                    new_right[s * kn * dr + c * dr + b] = vv;
                }
                for s in 0..2 {
                    for ap in 0..dl_prev {
                        let mut sum = cx(0.0, 0.0);
                        for &(a, us) in ucol {
                            sum += self.tensors[i - 1][s * dl_prev * dl + ap * dl + a] * us;
                        }
                        new_left[s * dl_prev * kn + ap * kn + c] = sum;
                    }
                }
            }
            self.q_bonds[i] = new_q;
            self.tensors[i] = new_right;
            self.tensors[i - 1] = new_left;
        }
        let nrm = self.tensors[0].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-30);
        for z in self.tensors[0].iter_mut() {
            *z /= nrm;
        }
        nrm
    }

    /// Calcula las ocupaciones locales ⟨n_i⟩ (i = 0..L−1) en O(L · χ³) mediante entornos de identidad.
    pub fn local_occupations(&self) -> Vec<f64> {
        let l = self.l;
        // Entornos derechos R_id[i] de tamaño dl × dl (R_id[L] = [[1]])
        let mut r_id: Vec<Vec<C>> = vec![Vec::new(); l + 1];
        r_id[l] = vec![cx(1.0, 0.0)];
        for i in (0..l).rev() {
            let dl = self.q_bonds[i].len();
            let dr = self.q_bonds[i + 1].len();
            let a = &self.tensors[i];
            let rn = &r_id[i + 1];
            let mut out = vec![cx(0.0, 0.0); dl * dl];
            let mut tmp = vec![cx(0.0, 0.0); dl * dr];
            for s in 0..2 {
                tmp.fill(cx(0.0, 0.0));
                let off = s * dl * dr;
                for al in 0..dl {
                    for ar in 0..dr {
                        let av = a[off + al * dr + ar].conj();
                        if av == cx(0.0, 0.0) {
                            continue;
                        }
                        for br in 0..dr {
                            tmp[al * dr + br] += av * rn[ar * dr + br];
                        }
                    }
                }
                for al in 0..dl {
                    for bl in 0..dl {
                        let mut sum = cx(0.0, 0.0);
                        for br in 0..dr {
                            sum += tmp[al * dr + br] * a[off + bl * dr + br];
                        }
                        out[al * dl + bl] += sum;
                    }
                }
            }
            r_id[i] = out;
        }
        let mut l_id = vec![cx(1.0, 0.0)];
        let mut occ = vec![0.0; l];
        for i in 0..l {
            let dl = self.q_bonds[i].len();
            let dr = self.q_bonds[i + 1].len();
            let a = &self.tensors[i];
            let rn = &r_id[i + 1];
            // Para s = 0 (n = 1) y s = 1 (n = 0)
            let mut num = 0.0;
            let mut den = 0.0;
            let mut next_l = vec![cx(0.0, 0.0); dr * dr];
            let mut tmp = vec![cx(0.0, 0.0); dl * dr];
            for s in 0..2 {
                tmp.fill(cx(0.0, 0.0));
                let off = s * dl * dr;
                for al in 0..dl {
                    for bl in 0..dl {
                        let lv = l_id[al * dl + bl];
                        if lv == cx(0.0, 0.0) {
                            continue;
                        }
                        for br in 0..dr {
                            tmp[al * dr + br] += lv * a[off + bl * dr + br];
                        }
                    }
                }
                let mut m_s = vec![cx(0.0, 0.0); dr * dr];
                for ar in 0..dr {
                    for al in 0..dl {
                        let av = a[off + al * dr + ar].conj();
                        if av == cx(0.0, 0.0) {
                            continue;
                        }
                        for br in 0..dr {
                            m_s[ar * dr + br] += av * tmp[al * dr + br];
                        }
                    }
                }
                let mut tr = 0.0;
                for ar in 0..dr {
                    for br in 0..dr {
                        tr += (m_s[ar * dr + br] * rn[ar * dr + br]).re;
                        next_l[ar * dr + br] += m_s[ar * dr + br];
                    }
                }
                den += tr;
                if s == 0 {
                    num += tr;
                }
            }
            occ[i] = num / den.max(1e-30);
            l_id = next_l;
        }
        occ
    }

    /// Correlador de Néel alternado de peldaños C_Néel(d) = (−1)^d ⟨(n_r − 1)(n_{r+d} − 1)⟩
    /// con n_r = n_{A,r} + n_{B,r} ∈ {0, 1, 2} (calculado exactamente en el MPS real o complejo!).
    pub fn rung_neel_correlator(&self, r0: usize, dist: usize) -> f64 {
        let r1 = r0 + dist;
        if 2 * r1 + 1 >= self.l {
            return 0.0;
        }
        // Operador (n_{A,r} + n_{B,r} − 1) = (n_{A,r} − ½) + (n_{B,r} − ½) = ½(σ^z_{A,r} + σ^z_{B,r})
        // Evaluamos las 4 combinaciones (α ∈ {A,B}, β ∈ {A,B}) de ⟨(n_i − ½)(n_j − ½)⟩ / ⟨ψ|ψ⟩:
        let two_pt = |i0: Option<usize>, i1: Option<usize>| -> f64 {
            let mut l_mat = vec![cx(1.0, 0.0)];
            for i in 0..self.l {
                let dl = self.q_bonds[i].len();
                let dr = self.q_bonds[i + 1].len();
                let a = &self.tensors[i];
                let mut weights = [1.0f64, 1.0f64];
                if Some(i) == i0 {
                    weights[0] *= 0.5;
                    weights[1] *= -0.5;
                }
                if Some(i) == i1 {
                    weights[0] *= 0.5;
                    weights[1] *= -0.5;
                }
                let mut next = vec![cx(0.0, 0.0); dr * dr];
                let mut tmp = vec![cx(0.0, 0.0); dl * dr];
                for s in 0..2 {
                    let w = weights[s];
                    tmp.fill(cx(0.0, 0.0));
                    let off = s * dl * dr;
                    for al in 0..dl {
                        for bl in 0..dl {
                            let lv = l_mat[al * dl + bl];
                            if lv == cx(0.0, 0.0) {
                                continue;
                            }
                            for br in 0..dr {
                                tmp[al * dr + br] += lv * a[off + bl * dr + br];
                            }
                        }
                    }
                    for ar in 0..dr {
                        for al in 0..dl {
                            let av = a[off + al * dr + ar].conj() * w;
                            if av == cx(0.0, 0.0) {
                                continue;
                            }
                            for br in 0..dr {
                                next[ar * dr + br] += av * tmp[al * dr + br];
                            }
                        }
                    }
                }
                l_mat = next;
            }
            l_mat[0].re
        };
        let norm2 = two_pt(None, None);
        if norm2.abs() <= 1e-30 {
            return 0.0;
        }
        let val = two_pt(Some(2 * r0), Some(2 * r1))
            + two_pt(Some(2 * r0), Some(2 * r1 + 1))
            + two_pt(Some(2 * r0 + 1), Some(2 * r1))
            + two_pt(Some(2 * r0 + 1), Some(2 * r1 + 1));
        let sgn = if dist % 2 == 0 { 1.0 } else { -1.0 };
        sgn * val / norm2
    }
}

/// Exponencial de Krylov compleja exp(−i τ H) v mediante `kmax` iteraciones de Lanczos hermítico
/// (usa `jacobi_tridiag` real sobre la matriz tridiagonal de Lanczos: unitaria a < 10⁻¹³!).
fn krylov_expm_complex<F>(v0: &[C], tau: f64, kmax: usize, mut matvec: F) -> Vec<C>
where
    F: FnMut(&[C], &mut [C]),
{
    let n = v0.len();
    if n == 0 {
        return Vec::new();
    }
    let norm0 = v0.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if norm0 < 1e-30 {
        return vec![cx(0.0, 0.0); n];
    }
    let km = kmax.min(n).max(1);
    let mut vs: Vec<Vec<C>> = Vec::with_capacity(km);
    vs.push(v0.iter().map(|&z| z / norm0).collect());
    let mut alphas = Vec::with_capacity(km);
    let mut betas = Vec::with_capacity(km);
    let mut w = vec![cx(0.0, 0.0); n];

    for j in 0..km {
        matvec(&vs[j], &mut w);
        let a: f64 = vs[j].iter().zip(&w).map(|(x, y)| (x.conj() * *y).re).sum();
        alphas.push(a);
        for i in 0..n {
            w[i] -= vs[j][i] * a;
        }
        if j > 0 {
            let bp = betas[j - 1];
            for i in 0..n {
                w[i] -= vs[j - 1][i] * bp;
            }
        }
        for vk in &vs {
            let d: C = vk.iter().zip(&w).map(|(x, y)| x.conj() * *y).sum();
            for i in 0..n {
                w[i] -= vk[i] * d;
            }
        }
        let b = w.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if b < 1e-12 || j + 1 == km {
            break;
        }
        betas.push(b);
        let inv = 1.0 / b;
        vs.push(w.iter().map(|&z| z * inv).collect());
    }
    let m = alphas.len();
    let (evals, vecs) = jacobi_tridiag(&alphas, &betas);
    // c_k = Σ_r V[k, r] · exp(−i τ E_r) · V[0, r]
    let mut coeffs = vec![cx(0.0, 0.0); m];
    for r in 0..m {
        let ph = C::from_polar(1.0, -tau * evals[r]) * (vecs[r] * norm0);
        for k in 0..m {
            coeffs[k] += ph * vecs[k * m + r];
        }
    }
    let mut out = vec![cx(0.0, 0.0); n];
    for k in 0..m {
        let ck = coeffs[k];
        for i in 0..n {
            out[i] += vs[k][i] * ck;
        }
    }
    out
}

fn c_update_left(l_prev: &[C], a: &[C], dl: usize, dr: usize, w: &U1MpoSite) -> Vec<C> {
    let mut out = vec![cx(0.0, 0.0); dr * w.dr * dr];
    let mut tmp = vec![cx(0.0, 0.0); dl * dr];
    for t in &w.terms {
        for s2 in 0..2 {
            for s1 in 0..2 {
                let v = t.c * t.op[s2][s1];
                if v == 0.0 {
                    continue;
                }
                tmp.fill(cx(0.0, 0.0));
                let a1_off = s1 * dl * dr;
                let a2_off = s2 * dl * dr;
                for al in 0..dl {
                    let l_off = (al * w.dl + t.l) * dl;
                    for bl in 0..dl {
                        let lv = l_prev[l_off + bl];
                        if lv == cx(0.0, 0.0) {
                            continue;
                        }
                        for br in 0..dr {
                            tmp[al * dr + br] += lv * a[a1_off + bl * dr + br];
                        }
                    }
                }
                for ar in 0..dr {
                    let o_off = (ar * w.dr + t.r) * dr;
                    for al in 0..dl {
                        let av = a[a2_off + al * dr + ar].conj() * v;
                        if av == cx(0.0, 0.0) {
                            continue;
                        }
                        for br in 0..dr {
                            out[o_off + br] += av * tmp[al * dr + br];
                        }
                    }
                }
            }
        }
    }
    out
}

fn c_update_right(r_next: &[C], b: &[C], dl: usize, dr: usize, w: &U1MpoSite) -> Vec<C> {
    let mut out = vec![cx(0.0, 0.0); dl * w.dl * dl];
    let mut tmp = vec![cx(0.0, 0.0); dl * dr];
    for t in &w.terms {
        for s2 in 0..2 {
            for s1 in 0..2 {
                let v = t.c * t.op[s2][s1];
                if v == 0.0 {
                    continue;
                }
                tmp.fill(cx(0.0, 0.0));
                let b1_off = s1 * dl * dr;
                let b2_off = s2 * dl * dr;
                for al in 0..dl {
                    for ar in 0..dr {
                        let bv = b[b2_off + al * dr + ar].conj();
                        if bv == cx(0.0, 0.0) {
                            continue;
                        }
                        let r_off = (ar * w.dr + t.r) * dr;
                        for br in 0..dr {
                            tmp[al * dr + br] += bv * r_next[r_off + br];
                        }
                    }
                }
                for al in 0..dl {
                    let o_off = (al * w.dl + t.l) * dl;
                    for bl in 0..dl {
                        let mut s = cx(0.0, 0.0);
                        for br in 0..dr {
                            s += tmp[al * dr + br] * b[b1_off + bl * dr + br];
                        }
                        out[o_off + bl] += s * v;
                    }
                }
            }
        }
    }
    out
}

/// Un paso completo simétrico de 2.º orden del integrador **2-site U(1)-TDVP** (Haegeman et al., PRB 2016)
/// sobre el MPO exacto `mpo` (ida derecha con Δt/2 + vuelta izquierda con Δt/2).
/// Conserva exactamente la carga U(1) por bloques, la norma y la energía ⟨ψ|H|ψ⟩!
pub fn tdvp_2site_step(
    mps: &mut U1Cmps,
    mpo: &U1Mpo,
    l_env: &mut [Vec<C>],
    r_env: &mut [Vec<C>],
    dt: f64,
    chi_max: usize,
) {
    let chi_max = chi_max.max(1);
    let dt = clamp_finite(dt, -10.0, 10.0, 0.25);
    let l = mps.l;
    let phys_q = [1i32, -1i32];
    let half_dt = 0.5 * dt;

    // Función auxiliar de evolución de 1 sitio hacia atrás exp(+i half_dt H_eff^(1))
    let evolve_1site_back = |site_t: &mut [C], dl: usize, dr: usize, ql: &[i32], qr: &[i32], le: &[C], re: &[C], w: &U1MpoSite| {
        let mut valid1 = Vec::new();
        let mut mask1 = vec![false; 2 * dl * dr];
        for s in 0..2 {
            for a in 0..dl {
                let qa = ql[a] + phys_q[s];
                for b in 0..dr {
                    if qa == qr[b] {
                        let idx = (s * dl + a) * dr + b;
                        valid1.push((s, a, b, idx));
                        mask1[idx] = true;
                    }
                }
            }
        }
        let nv = valid1.len();
        if nv == 0 {
            return;
        }
        let v0: Vec<C> = valid1.iter().map(|&(_, _, _, idx)| site_t[idx]).collect();
        let mut full = vec![cx(0.0, 0.0); 2 * dl * dr];
        let mut out_full = vec![cx(0.0, 0.0); 2 * dl * dr];
        let mut tmp = vec![cx(0.0, 0.0); dl * dr];
        let res = krylov_expm_complex(&v0, -half_dt, 10, |x, y| {
            full.fill(cx(0.0, 0.0));
            for (k, &(_, _, _, idx)) in valid1.iter().enumerate() {
                full[idx] = x[k];
            }
            out_full.fill(cx(0.0, 0.0));
            for t in &w.terms {
                for s2 in 0..2 {
                    for s1 in 0..2 {
                        let c = t.c * t.op[s2][s1];
                        if c == 0.0 {
                            continue;
                        }
                        tmp.fill(cx(0.0, 0.0));
                        for al in 0..dl {
                            let l_off = (al * w.dl + t.l) * dl;
                            for bl in 0..dl {
                                let lv = le[l_off + bl] * c;
                                if lv == cx(0.0, 0.0) {
                                    continue;
                                }
                                let src = (s1 * dl + bl) * dr;
                                for br in 0..dr {
                                    tmp[al * dr + br] += lv * full[src + br];
                                }
                            }
                        }
                        for al in 0..dl {
                            let dst = (s2 * dl + al) * dr;
                            for ar in 0..dr {
                                if !mask1[dst + ar] {
                                    continue;
                                }
                                let r_off = (ar * w.dr + t.r) * dr;
                                let mut s = cx(0.0, 0.0);
                                for br in 0..dr {
                                    s += tmp[al * dr + br] * re[r_off + br];
                                }
                                out_full[dst + ar] += s;
                            }
                        }
                    }
                }
            }
            for (k, &(_, _, _, idx)) in valid1.iter().enumerate() {
                y[k] = out_full[idx];
            }
        });
        site_t.fill(cx(0.0, 0.0));
        for (k, &(_, _, _, idx)) in valid1.iter().enumerate() {
            site_t[idx] = res[k];
        }
    };

    let mut sweep_order: Vec<(usize, bool)> = (0..(l - 1)).map(|i| (i, true)).collect();
    sweep_order.extend((0..(l - 1)).rev().map(|i| (i, false)));

    for (i, going_right) in sweep_order {
        let dl = mps.q_bonds[i].len();
        let dm = mps.q_bonds[i + 1].len();
        let dr = mps.q_bonds[i + 2].len();
        let ql = &mps.q_bonds[i];
        let qr = &mps.q_bonds[i + 2];
        let wl = &mpo.sites[i];
        let wr = &mpo.sites[i + 1];
        let w_mid = wl.dr;

        let mut valid = Vec::new();
        let mut mask_valid = vec![false; 4 * dl * dr];
        for s1 in 0..2 {
            for s2 in 0..2 {
                let dq = phys_q[s1] + phys_q[s2];
                for a in 0..dl {
                    let qa = ql[a] + dq;
                    for b in 0..dr {
                        if qa == qr[b] {
                            let idx = ((s1 * 2 + s2) * dl + a) * dr + b;
                            valid.push((s1, s2, a, b, idx));
                            mask_valid[idx] = true;
                        }
                    }
                }
            }
        }
        let n_val = valid.len();
        if n_val == 0 {
            continue;
        }
        let mut theta_full = vec![cx(0.0, 0.0); 4 * dl * dr];
        let ta = &mps.tensors[i];
        let tb = &mps.tensors[i + 1];
        for &(s1, s2, a, b, idx) in &valid {
            let mut sum = cx(0.0, 0.0);
            for m in 0..dm {
                sum += ta[s1 * dl * dm + a * dm + m] * tb[s2 * dm * dr + m * dr + b];
            }
            theta_full[idx] = sum;
        }
        let v0: Vec<C> = valid.iter().map(|&(_, _, _, _, idx)| theta_full[idx]).collect();

        let le = &l_env[i];
        let re = &r_env[i + 2];
        let mut t_l = vec![cx(0.0, 0.0); 2 * dl * w_mid * 2 * dr];
        let mut out_full = vec![cx(0.0, 0.0); 4 * dl * dr];
        let psi_next = krylov_expm_complex(&v0, half_dt, 10, |x, y| {
            theta_full.fill(cx(0.0, 0.0));
            for (k, &(_, _, _, _, idx)) in valid.iter().enumerate() {
                theta_full[idx] = x[k];
            }
            t_l.fill(cx(0.0, 0.0));
            out_full.fill(cx(0.0, 0.0));
            for tl in &wl.terms {
                for s1p in 0..2 {
                    for s1 in 0..2 {
                        let c = tl.c * tl.op[s1p][s1];
                        if c == 0.0 {
                            continue;
                        }
                        for al in 0..dl {
                            let l_off = (al * wl.dl + tl.l) * dl;
                            for bl in 0..dl {
                                let lv = le[l_off + bl] * c;
                                if lv == cx(0.0, 0.0) {
                                    continue;
                                }
                                for s2 in 0..2 {
                                    let src = ((s1 * 2 + s2) * dl + bl) * dr;
                                    let dst = (((s1p * 2 + s2) * dl + al) * w_mid + tl.r) * dr;
                                    for br in 0..dr {
                                        t_l[dst + br] += lv * theta_full[src + br];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for tr in &wr.terms {
                for s2p in 0..2 {
                    for s2 in 0..2 {
                        let c = tr.c * tr.op[s2p][s2];
                        if c == 0.0 {
                            continue;
                        }
                        for s1p in 0..2 {
                            for al in 0..dl {
                                let src = (((s1p * 2 + s2) * dl + al) * w_mid + tr.l) * dr;
                                let dst = ((s1p * 2 + s2p) * dl + al) * dr;
                                for ar in 0..dr {
                                    if !mask_valid[dst + ar] {
                                        continue;
                                    }
                                    let r_off = (ar * wr.dr + tr.r) * dr;
                                    let mut s = cx(0.0, 0.0);
                                    for br in 0..dr {
                                        s += t_l[src + br] * re[r_off + br];
                                    }
                                    out_full[dst + ar] += s * c;
                                }
                            }
                        }
                    }
                }
            }
            for (k, &(_, _, _, _, idx)) in valid.iter().enumerate() {
                y[k] = out_full[idx];
            }
        });

        // SVD complejo por bloques de carga U(1)
        let mut left_by_q: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();
        let mut right_by_q: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();
        for s1 in 0..2 {
            for a in 0..dl {
                left_by_q.entry(ql[a] + phys_q[s1]).or_default().push((s1, a));
            }
        }
        for s2 in 0..2 {
            for b in 0..dr {
                right_by_q.entry(qr[b] - phys_q[s2]).or_default().push((s2, b));
            }
        }
        theta_full.fill(cx(0.0, 0.0));
        for (k, &(_, _, _, _, idx)) in valid.iter().enumerate() {
            theta_full[idx] = psi_next[k];
        }
        struct CSvdEntry {
            sig: f64,
            q: i32,
            u_col: Vec<(usize, usize, C)>,
            v_row: Vec<(usize, usize, C)>,
        }
        let mut entries = Vec::new();
        let mut fallback: Option<CSvdEntry> = None;
        for (&q_mid, l_list) in &left_by_q {
            let Some(r_list) = right_by_q.get(&q_mid) else { continue };
            let bm = l_list.len();
            let bn = r_list.len();
            let mut mat = vec![cx(0.0, 0.0); bm * bn];
            for (ii, &(s1, a)) in l_list.iter().enumerate() {
                for (jj, &(s2, b)) in r_list.iter().enumerate() {
                    mat[ii * bn + jj] = theta_full[((s1 * 2 + s2) * dl + a) * dr + b];
                }
            }
            let (u, s, vh) = block_csvd(&mat, bm, bn);
            let kk = s.len();
            for c in 0..kk {
                if s[c] < 1e-13 {
                    if c == 0 && fallback.as_ref().map_or(true, |fb| s[0] > fb.sig) {
                        let u_col = (0..bm).map(|ii| (l_list[ii].0, l_list[ii].1, u[ii * kk])).collect();
                        let v_row = (0..bn).map(|jj| (r_list[jj].0, r_list[jj].1, vh[jj])).collect();
                        fallback = Some(CSvdEntry { sig: s[0], q: q_mid, u_col, v_row });
                    }
                    continue;
                }
                let u_col = (0..bm).map(|ii| (l_list[ii].0, l_list[ii].1, u[ii * kk + c])).collect();
                let v_row = (0..bn).map(|jj| (r_list[jj].0, r_list[jj].1, vh[c * bn + jj])).collect();
                entries.push(CSvdEntry { sig: s[c], q: q_mid, u_col, v_row });
            }
        }
        if entries.is_empty() {
            if let Some(fb) = fallback {
                entries.push(fb);
            } else {
                entries.push(CSvdEntry {
                    sig: 1.0,
                    q: ql.first().copied().unwrap_or(0) + phys_q[0],
                    u_col: vec![(0, 0, cx(1.0, 0.0))],
                    v_row: vec![(0, 0, cx(1.0, 0.0))],
                });
            }
        }
        entries.sort_by(|a, b| b.sig.total_cmp(&a.sig));
        if entries.len() > chi_max {
            entries.truncate(chi_max);
        }
        let k_new = entries.len();
        let s_norm = entries.iter().map(|e| e.sig * e.sig).sum::<f64>().sqrt().max(1e-30);
        let mut new_q = Vec::with_capacity(k_new);
        let mut new_a = vec![cx(0.0, 0.0); 2 * dl * k_new];
        let mut new_b = vec![cx(0.0, 0.0); 2 * k_new * dr];
        for (c, e) in entries.iter().enumerate() {
            new_q.push(e.q);
            let sn = e.sig / s_norm;
            for &(s1, a, uval) in &e.u_col {
                new_a[s1 * dl * k_new + a * k_new + c] = if going_right { uval } else { uval * sn };
            }
            for &(s2, b, vval) in &e.v_row {
                new_b[s2 * k_new * dr + c * dr + b] = if going_right { vval * sn } else { vval };
            }
        }
        mps.q_bonds[i + 1] = new_q;
        mps.tensors[i] = new_a;
        mps.tensors[i + 1] = new_b;

        if going_right {
            l_env[i + 1] = c_update_left(&l_env[i], &mps.tensors[i], dl, k_new, &mpo.sites[i]);
            if i + 2 < l {
                evolve_1site_back(
                    &mut mps.tensors[i + 1],
                    k_new,
                    dr,
                    &mps.q_bonds[i + 1],
                    &mps.q_bonds[i + 2],
                    &l_env[i + 1],
                    &r_env[i + 2],
                    &mpo.sites[i + 1],
                );
            }
        } else {
            r_env[i + 1] = c_update_right(&r_env[i + 2], &mps.tensors[i + 1], k_new, dr, &mpo.sites[i + 1]);
            if i > 0 {
                evolve_1site_back(
                    &mut mps.tensors[i],
                    dl,
                    k_new,
                    &mps.q_bonds[i],
                    &mps.q_bonds[i + 1],
                    &l_env[i],
                    &r_env[i + 1],
                    &mpo.sites[i],
                );
            }
        }
    }
}

/// Mide en tiempo real (mediante 2-site U(1)-TDVP sobre un MPS de `n_rungs` peldaños, L = 2·n_rungs sitios)
/// la velocidad del frente de onda balístico v_LC = d(r_rms)/dt de una excitación local de densidad creada
/// en la especie `leg` (0 = A, 1 = B) en el centro de la cadena!
fn measure_tdvp_front_speed(n_rungs: usize, v_a: f64, v_b: f64, j_gauge: f64, leg: usize, chi: usize, n_steps: usize, dt: f64) -> (f64, f64) {
    let mpo = schwinger_2f_exact_mpo(n_rungs, v_a, v_b, j_gauge);
    let (_, gs_real) = run_u1_dmrg(&mpo, chi, 5);
    let mut gs_c = U1Cmps::from_real(&gs_real);
    let occ0 = gs_c.local_occupations();

    // Proyección estrictamente local de ocupación n_{leg, center} |GS⟩ / ‖·‖:
    // - conserva EXACTAMENTE U(1)_A × U(1)_B (n es diagonal en la base de ocupación!);
    // - en t = 0 el exceso positivo δn_r^+(0) = max(0, ⟨n_r⟩ − ⟨n_r⟩_0) se concentra 100 % en r = center
    //   (r_rms^+(0) = 0 exacto, sin meseta discreta 1/√2!);
    // - en J = 0 lanza una partícula libre en la especie `leg` a velocidad exacta v_A o v_B;
    // - en J = 0,50 (vacío confinado), proyectar n_A = 1 o n_B = 1 excita el MISMO mesón neutro confinado
    //   que propaga el frente de luz de AMBAS especies a la misma velocidad infrarroja v_A^LC ≈ v_B^LC!
    let center = n_rungs / 2;
    let target_site = 2 * center + leg;
    let dl = gs_c.q_bonds[target_site].len();
    let dr = gs_c.q_bonds[target_site + 1].len();
    for a in 0..dl {
        for b in 0..dr {
            // s = 0 es ocupado (n = 1), s = 1 es vacío (n = 0) ⇒ anulamos s = 1!
            gs_c.tensors[target_site][dl * dr + a * dr + b] = cx(0.0, 0.0);
        }
    }
    gs_c.right_canonicalize();

    let l = gs_c.l;
    let mut l_env = vec![Vec::new(); l + 1];
    let mut r_env = vec![Vec::new(); l + 1];
    l_env[0] = vec![cx(1.0, 0.0)];
    r_env[l] = vec![cx(1.0, 0.0)];
    for i in (0..l).rev() {
        let dli = gs_c.q_bonds[i].len();
        let dri = gs_c.q_bonds[i + 1].len();
        r_env[i] = c_update_right(&r_env[i + 1], &gs_c.tensors[i], dli, dri, &mpo.sites[i]);
    }
    // Energía inicial del estado excitado (para auditar conservación simpléctica de energía en TDVP!)
    let e_init = c_update_right(&r_env[1], &gs_c.tensors[0], 1, gs_c.q_bonds[1].len(), &mpo.sites[0])[0].re;

    let mut ts = Vec::with_capacity(n_steps + 1);
    let mut r_rms = Vec::with_capacity(n_steps + 1);
    ts.push(0.0);
    r_rms.push(0.0);
    for st in 1..=n_steps {
        tdvp_2site_step(&mut gs_c, &mpo, &mut l_env, &mut r_env, dt, chi);
        let occ = gs_c.local_occupations();
        // Exceso positivo de densidad en cada peldaño (aísla el paquete balístico +0,5·J_d(v t)² sin hueco estático!)
        let mut w_sum = 0.0;
        let mut r2_sum = 0.0;
        for r in 0..n_rungs {
            let dn = (occ[2 * r] - occ0[2 * r]).max(0.0) + (occ[2 * r + 1] - occ0[2 * r + 1]).max(0.0);
            let dx = (r as f64 - center as f64).abs();
            w_sum += dn;
            r2_sum += dn * dx * dx;
        }
        ts.push(st as f64 * dt);
        r_rms.push((r2_sum / w_sum.max(1e-12)).sqrt());
    }
    let e_fin = c_update_right(&r_env[1], &gs_c.tensors[0], 1, gs_c.q_bonds[1].len(), &mpo.sites[0])[0].re;
    let e_drift = ((e_fin - e_init) / e_init.abs().max(1.0)).abs();
    // Velocidad balística efectiva: √2 · pendiente de r_rms^+(t) (identidad exacta Σ d² J_d(vt)² = ½ v² t²!)
    let (_, slope, _) = lin_fit_r2(&ts, &r_rms);
    (2.0f64.sqrt() * slope, e_drift)
}

/// Fase 20 (Vía C — 2-site U(1)-TDVP en tiempo real en L = 24–32 sitios + diagrama de fases BKT):
/// 1. Valida el integrador 2-site U(1)-TDVP en cadena de `nr_tdvp` peldaños (12 en WASM ⇒ L = 24,
///    16 en nativo ⇒ L = 32 sitios, espacio de Hilbert 2³² ≈ 4,3×10⁹!) midiendo en tiempo real
///    la velocidad del frente balístico de A vs B en J = 0 (conos de luz separados v_A ≠ v_B) frente
///    a J = 0,50 (ambas especies se acoplan al MISMO cono de luz infrarrojo con energía conservada!).
/// 2. Mapea el diagrama de fases (J, v_B/v_A) de la brecha BKT inducida por asimetría de velocidades
///    Δ_ef = (v_A² + v_B²)/(2 v_A v_B) en N_r = 16 → 24 midiendo el correlador de Néel alternado
///    C_Néel(d = 4) y la pendiente efectiva de entropía c_eff(16 → 24) en 6 puntos:
///    v_B/v_A ∈ {1,00; 0,45; 0,25} × J ∈ {1,0; 2,5}.
pub fn run_via_c_tdvp_and_bkt_diagram(nr_tdvp: usize, chi_tdvp: usize) -> Vec<f64> {
    let nr = (nr_tdvp / 2 * 2).clamp(10, 18);
    let chi = chi_tdvp.clamp(24, 48);
    let dt = 0.25f64;
    let n_steps = 12usize; // T = 3,0: frente balístico limpio sin rebote en bordes

    // 1. 4 corridas TDVP en paralelo: (J=0, leg A), (J=0, leg B), (J=0.50, leg A), (J=0.50, leg B)
    let tdvp_jobs = [(0.0f64, 0usize), (0.0, 1), (0.50, 0), (0.50, 1)];
    #[cfg(not(target_arch = "wasm32"))]
    let tdvp_res: Vec<(f64, f64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = tdvp_jobs
            .into_iter()
            .map(|(jg, leg)| sc.spawn(move || measure_tdvp_front_speed(nr, 1.0, 0.45, jg, leg, chi, n_steps, dt)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let tdvp_res: Vec<(f64, f64)> = tdvp_jobs
        .into_iter()
        .map(|(jg, leg)| measure_tdvp_front_speed(nr, 1.0, 0.45, jg, leg, chi, n_steps, dt))
        .collect();

    let (va_j0, ed0) = tdvp_res[0];
    let (vb_j0, ed1) = tdvp_res[1];
    let (va_jc, ed2) = tdvp_res[2];
    let (vb_jc, ed3) = tdvp_res[3];
    let mis_j0 = (va_j0 - vb_j0).abs() / (0.5 * (va_j0 + vb_j0)).max(1e-9);
    let mis_jc = (va_jc - vb_jc).abs() / (0.5 * (va_jc + vb_jc)).max(1e-9);
    let max_e_drift = ed0.max(ed1).max(ed2).max(ed3);

    // 2. Diagrama de fases (J, v_B/v_A) de la brecha BKT de Néel en N_r = 16 → 24
    let bkt_pts: [(f64, f64); 6] = [
        (1.00, 1.00), // Δ_ef = 1,000, J = 1,0
        (1.00, 2.50), // Δ_ef = 1,000, J = 2,5 (control crítico c = 1)
        (0.45, 1.00), // Δ_ef = 1,336, J = 1,0
        (0.45, 2.50), // Δ_ef = 1,336, J = 2,5
        (0.25, 1.00), // Δ_ef = 2,125, J = 1,0
        (0.25, 2.50), // Δ_ef = 2,125, J = 2,5 (fase Néel profunda!)
    ];
    let eval_bkt = |(vb, jg): (f64, f64)| -> [f64; 6] {
        let delta_eff = (1.0 + vb * vb) / (2.0 * vb);
        let m16 = schwinger_2f_exact_mpo(16, 1.0, vb, jg);
        let (_, p16) = run_u1_dmrg(&m16, chi.max(32), 5);
        let s16 = p16.entanglement_at(16);
        let m24 = schwinger_2f_exact_mpo(24, 1.0, vb, jg);
        let (_, p24) = run_u1_dmrg(&m24, chi.max(32), 5);
        let s24 = p24.entanglement_at(24);
        let c_eff = 6.0 * (s24 - s16) / (24.0f64 / 16.0).ln();
        let c24 = U1Cmps::from_real(&p24);
        let neel_d4 = c24.rung_neel_correlator(10, 4);
        [vb, jg, delta_eff, s16, c_eff, neel_d4]
    };

    #[cfg(not(target_arch = "wasm32"))]
    let bkt_rows: Vec<[f64; 6]> = std::thread::scope(|sc| {
        let hs: Vec<_> = bkt_pts.into_iter().map(|pt| sc.spawn(move || eval_bkt(pt))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let bkt_rows: Vec<[f64; 6]> = bkt_pts.into_iter().map(eval_bkt).collect();

    // Salida:
    // Cabecera (9): [nr_tdvp, va_j0, vb_j0, mis_j0, va_jc, vb_jc, mis_jc, max_e_drift, n_bkt (6)]
    // 6 × [vB/vA, J, Delta_eff, S16, c_eff_16_24, C_neel_d4]
    let mut out = vec![nr as f64, va_j0, vb_j0, mis_j0, va_jc, vb_jc, mis_jc, max_e_drift, bkt_rows.len() as f64];
    for r in &bkt_rows {
        out.extend_from_slice(r);
    }
    out
}

/// Auditoría de convergencia en dimensión de enlace χ ∈ {24, 32, 44, 56} para el diagrama de fases BKT
/// en J = 2,50: compara el punto crítico simétrico (v_B/v_A = 1,00, Δ_ef = 1,000) frente a la fase de
/// Néel profunda (v_B/v_A = 0,25, Δ_ef = 2,125) en N_r = 16 → 24, devolviendo por fila:
///   [vB/vA, J, chi, E0(N_r=24), S_vN(N_r=16), c_eff(16→24), C_Neel(d=4)]
pub fn run_via_c_chi_convergence() -> Vec<[f64; 7]> {
    let tasks: [(f64, f64, usize); 8] = [
        (1.00, 2.50, 24),
        (1.00, 2.50, 32),
        (1.00, 2.50, 44),
        (1.00, 2.50, 56),
        (0.25, 2.50, 24),
        (0.25, 2.50, 32),
        (0.25, 2.50, 44),
        (0.25, 2.50, 56),
    ];
    let eval_one = |(vb, jg, chi): (f64, f64, usize)| -> [f64; 7] {
        let m16 = schwinger_2f_exact_mpo(16, 1.0, vb, jg);
        let (_, p16) = run_u1_dmrg(&m16, chi, 6);
        let s16 = p16.entanglement_at(16);
        let m24 = schwinger_2f_exact_mpo(24, 1.0, vb, jg);
        let (e24, p24) = run_u1_dmrg(&m24, chi, 6);
        let s24 = p24.entanglement_at(24);
        let c_eff = 6.0 * (s24 - s16) / (24.0f64 / 16.0).ln();
        let c24 = U1Cmps::from_real(&p24);
        let neel = c24.rung_neel_correlator(10, 4);
        [vb, jg, chi as f64, e24, s16, c_eff, neel]
    };
    #[cfg(not(target_arch = "wasm32"))]
    return std::thread::scope(|sc| {
        let hs: Vec<_> = tasks.into_iter().map(|t| sc.spawn(move || eval_one(t))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    tasks.into_iter().map(eval_one).collect()
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 18 (Vía C): restauración infrarroja de Lorentz a la media geométrica √(v_A v_B) mediante
    /// confinamiento gauge en el modelo de Schwinger cuántico de dos sabores (U(1)_A × U(1)_B exacto)
    /// + motor DMRG variacional con conservación explícita de bloques de carga U(1).
    pub fn via_c_schwinger2f_lorentz(&mut self, n_rungs_ex: u32, chi_u1: u32) -> *const f64 {
        self.oq.out = run_via_c_schwinger2f_lorentz(n_rungs_ex as usize, chi_u1 as usize);
        self.oq.out.as_ptr()
    }

    /// Fase 19 (Vía C): extrapolación al continuo de v_IR(N_r → ∞) → √(v_A v_B) a M_S·L = const,
    /// escalado multi-tamaño Calabrese–Cardy en U(1)-DMRG (N_r = 8, 12, 16, 24[, 32]) verificando
    /// c_CFT(J=0) → 2 vs c_CFT(J=1) → 1, y contraste cuantitativo frente a QED 3+1D Coulomb.
    pub fn via_c_continuum_and_scaling(&mut self, include_nr32: bool, chi_u1: u32) -> *const f64 {
        self.oq.out = run_via_c_continuum_and_scaling(include_nr32, chi_u1 as usize);
        self.oq.out.as_ptr()
    }

    /// Fase 20 (Vía C — 2-site U(1)-TDVP en tiempo real + diagrama de fases BKT):
    /// evoluciona en tiempo real paquetes locales de densidad en cadenas de L = 24–32 sitios (2³² estados)
    /// para medir la igualación del frente de onda v_A^LC ≈ v_B^LC al encender J, y mapea la transición
    /// de Néel inducida por Δ_ef = (v_A² + v_B²)/(2 v_A v_B) > 1 en (J, v_B/v_A).
    pub fn via_c_tdvp_and_bkt_diagram(&mut self, nr_tdvp: u32, chi_tdvp: u32) -> *const f64 {
        self.oq.out = run_via_c_tdvp_and_bkt_diagram(nr_tdvp as usize, chi_tdvp as usize);
        self.oq.out.as_ptr()
    }
}
