//! Fase 16 — DMRG variacional de dos sitios (Lanczos + MPO exacto en f64), escalado al continuo
//! (Bethe y carga central CFT de Calabrese–Cardy) y estudio del flujo RG de restauración de Lorentz
//! en teorías de dos especies acopladas (Collins et al., 2004 frente a Chadha y Nielsen, 1983).

use crate::linalg::complete_real_isometry;
use crate::util::clamp_finite;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

// ------------------------------------------------------------------ Álgebra lineal real de precisión (SVD + Lanczos)

/// SVD real exacto de A (m × n, fila mayor): A = U · diag(s) · V^T por rotaciones de Jacobi de Hestenes.
fn real_svd(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if m < n {
        let mut at = vec![0.0; n * m];
        for i in 0..m {
            for j in 0..n {
                at[j * m + i] = a[i * n + j];
            }
        }
        let (ut, s, vtt) = real_svd_tall(&at, n, m);
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
    real_svd_tall(a, m, n)
}

fn real_svd_tall(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i * n + j]).collect()).collect();
    let mut vcols: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let mut v = vec![0.0; n];
            v[j] = 1.0;
            v
        })
        .collect();

    for _ in 0..35 {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gam = 0.0;
                for i in 0..m {
                    let ap = cols[p][i];
                    let aq = cols[q][i];
                    alpha += ap * ap;
                    beta += aq * aq;
                    gam += ap * aq;
                }
                if gam.abs() <= 1e-14 * (alpha * beta).sqrt().max(1e-300) {
                    continue;
                }
                converged = false;
                let tau = (beta - alpha) / (2.0 * gam);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..m {
                    let ap = cols[p][i];
                    let aq = cols[q][i];
                    cols[p][i] = ap * c - aq * s;
                    cols[q][i] = ap * s + aq * c;
                }
                for i in 0..n {
                    let vp = vcols[p][i];
                    let vq = vcols[q][i];
                    vcols[p][i] = vp * c - vq * s;
                    vcols[q][i] = vp * s + vq * c;
                }
            }
        }
        if converged {
            break;
        }
    }

    let norms: Vec<f64> = (0..n).map(|j| cols[j].iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));

    let k = n;
    let mut u = vec![0.0; m * k];
    let mut s = vec![0.0; k];
    let mut vt = vec![0.0; k * n];
    let mut rank = 0usize;
    for (idx, &col) in order.iter().enumerate() {
        let sig = norms[col];
        s[idx] = sig;
        if sig > 1e-15 {
            rank += 1;
            let inv = 1.0 / sig;
            for i in 0..m {
                u[i * k + idx] = cols[col][i] * inv;
            }
        }
        for j in 0..n {
            vt[idx * n + j] = vcols[col][j];
        }
    }
    if rank < k {
        complete_real_isometry(&mut u, m, k, rank);
    }
    (u, s, vt)
}

/// Autovalor y autovector mínimo de una matriz tridiagonal real simétrica T (diag `a`, subdiag `b`).
fn tridiag_ground_state(a: &[f64], b: &[f64]) -> (f64, Vec<f64>) {
    let n = a.len();
    if n == 1 {
        return (a[0], vec![1.0]);
    }
    // Jacobi simétrico directo sobre la matriz pequeña (n ≤ 35)
    let mut mat = vec![0.0; n * n];
    let mut vecs = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = a[i];
        vecs[i * n + i] = 1.0;
        if i + 1 < n {
            mat[i * n + i + 1] = b[i];
            mat[(i + 1) * n + i] = b[i];
        }
    }
    for _ in 0..60 {
        let mut max_off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[p * n + q];
                if apq.abs() > max_off {
                    max_off = apq.abs();
                }
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (mat[q * n + q] - mat[p * n + p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = mat[p * n + p];
                let aqq = mat[q * n + q];
                mat[p * n + p] = app - t * apq;
                mat[q * n + q] = aqq + t * apq;
                mat[p * n + q] = 0.0;
                mat[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = mat[r * n + p];
                        let arq = mat[r * n + q];
                        mat[r * n + p] = c * arp - s * arq;
                        mat[p * n + r] = mat[r * n + p];
                        mat[r * n + q] = s * arp + c * arq;
                        mat[q * n + r] = mat[r * n + q];
                    }
                    let vrp = vecs[r * n + p];
                    let vrq = vecs[r * n + q];
                    vecs[r * n + p] = c * vrp - s * vrq;
                    vecs[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
    }
    let mut best = 0;
    for i in 1..n {
        if mat[i * n + i] < mat[best * n + best] {
            best = i;
        }
    }
    let e0 = mat[best * n + best];
    let v0: Vec<f64> = (0..n).map(|r| vecs[r * n + best]).collect();
    (e0, v0)
}

/// Eigensolucionador de Lanczos para el estado fundamental de un operador hermítico real `matvec`.
fn lanczos_ground<F>(dim: usize, init: &[f64], max_iter: usize, mut matvec: F) -> (f64, Vec<f64>)
where
    F: FnMut(&[f64], &mut [f64]),
{
    let kmax = max_iter.min(dim).max(1);
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(kmax);
    let mut v0 = init.to_vec();
    let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n0 < 1e-15 {
        for (i, x) in v0.iter_mut().enumerate() {
            *x = ((i * 7 + 3) % 13) as f64 - 6.0;
        }
    }
    let n0 = v0.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
    for x in v0.iter_mut() {
        *x /= n0;
    }
    vs.push(v0);

    let mut alphas = Vec::with_capacity(kmax);
    let mut betas = Vec::with_capacity(kmax);
    let mut w = vec![0.0; dim];

    for j in 0..kmax {
        w.fill(0.0);
        matvec(&vs[j], &mut w);
        let alpha: f64 = vs[j].iter().zip(&w).map(|(a, b)| a * b).sum();
        alphas.push(alpha);
        for i in 0..dim {
            w[i] -= alpha * vs[j][i];
        }
        if j > 0 {
            let bp = betas[j - 1];
            for i in 0..dim {
                w[i] -= bp * vs[j - 1][i];
            }
        }
        // Reortogonalización completa (Gram–Schmidt) para estabilidad numérica exacta
        for vk in &vs {
            let dot: f64 = vk.iter().zip(&w).map(|(a, b)| a * b).sum();
            for i in 0..dim {
                w[i] -= dot * vk[i];
            }
        }
        let beta = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if beta < 1e-13 || j + 1 == kmax {
            break;
        }
        betas.push(beta);
        let inv = 1.0 / beta;
        let next: Vec<f64> = w.iter().map(|x| x * inv).collect();
        vs.push(next);
    }

    let (e0, y) = tridiag_ground_state(&alphas, &betas);
    let mut out = vec![0.0; dim];
    for (k, &yk) in y.iter().enumerate() {
        for i in 0..dim {
            out[i] += yk * vs[k][i];
        }
    }
    let nrm = out.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
    for x in out.iter_mut() {
        *x /= nrm;
    }
    (e0, out)
}

// ------------------------------------------------------------------ MPO disperso real y DMRG de dos sitios

/// Operador de matriz producto (MPO) en cada sitio: lista dispersa de entradas `(w_left, w_right, s_out, s_in, val)`.
pub struct SparseMpo {
    pub l: usize,
    pub d_bond: usize,
    pub entries: Vec<Vec<(usize, usize, usize, usize, f64)>>,
}

impl SparseMpo {
    /// MPO exacto D = 5 para la cadena XXZ/Heisenberg:
    /// H = Σ [ (J_xy/2)(S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1}) + J_z S^z_i S^z_{i+1} ].
    pub fn xxz(l: usize, j_xy: f64, j_z: f64) -> Self {
        let mut entries = Vec::with_capacity(l);
        let half_xy = 0.5 * j_xy;
        for i in 0..l {
            let mut e = Vec::new();
            let push_i = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
                e.push((wl, wr, 0, 0, c));
                e.push((wl, wr, 1, 1, c));
            };
            let push_sp = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
                e.push((wl, wr, 0, 1, c));
            };
            let push_sm = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
                e.push((wl, wr, 1, 0, c));
            };
            let push_sz = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
                e.push((wl, wr, 0, 0, 0.5 * c));
                e.push((wl, wr, 1, 1, -0.5 * c));
            };
            if i == 0 {
                // Solo fila 4 activa en el borde izquierdo (mapeada a índice 0 en el entorno L[0] de tamaño 1)
                push_sm(&mut e, 0, 1, half_xy);
                push_sp(&mut e, 0, 2, half_xy);
                push_sz(&mut e, 0, 3, j_z);
                push_i(&mut e, 0, 4, 1.0);
            } else if i + 1 == l {
                // Solo columna 0 activa en el borde derecho
                push_i(&mut e, 0, 0, 1.0);
                push_sp(&mut e, 1, 0, 1.0);
                push_sm(&mut e, 2, 0, 1.0);
                push_sz(&mut e, 3, 0, 1.0);
            } else {
                push_i(&mut e, 0, 0, 1.0);
                push_sp(&mut e, 1, 0, 1.0);
                push_sm(&mut e, 2, 0, 1.0);
                push_sz(&mut e, 3, 0, 1.0);
                push_sm(&mut e, 4, 1, half_xy);
                push_sp(&mut e, 4, 2, half_xy);
                push_sz(&mut e, 4, 3, j_z);
                push_i(&mut e, 4, 4, 1.0);
            }
            entries.push(e);
        }
        SparseMpo { l, d_bond: 5, entries }
    }

    /// MPO exacto D = 8 para una escalera cuántica de dos especies acopladas (A en sitios pares 2r,
    /// B en impares 2r+1, longitud total L = 2·n_rungs):
    /// H = Σ_r [ v_A (S⁺_{A,r} S⁻_{A,r+1} + h.c.)/2 + Δ_A S^z_{A,r} S^z_{A,r+1}
    ///         + v_B (S⁺_{B,r} S⁻_{B,r+1} + h.c.)/2 + Δ_B S^z_{B,r} S^z_{B,r+1}
    ///         + g_ab · ( (S⁺_{A,r} S⁻_{B,r} + h.c.)/2 + S^z_{A,r} S^z_{B,r} ) ].
    pub fn two_species_ladder(n_rungs: usize, v_a: f64, v_b: f64, delta_a: f64, delta_b: f64, g_ab: f64) -> Self {
        Self::two_species_ladder_general(n_rungs, v_a, v_b, delta_a, delta_b, g_ab, g_ab)
    }

    /// MPO exacto D = 8 con acoplamientos de peldaño independientes g_xy (mezcla de sabor) y g_z (densidad-densidad).
    pub(crate) fn two_species_ladder_general(
        n_rungs: usize,
        v_a: f64,
        v_b: f64,
        delta_a: f64,
        delta_b: f64,
        g_xy: f64,
        g_z: f64,
    ) -> Self {
        let l = 2 * n_rungs;
        let mut entries = Vec::with_capacity(l);
        let push_i = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
            e.push((wl, wr, 0, 0, c));
            e.push((wl, wr, 1, 1, c));
        };
        let push_sp = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
            e.push((wl, wr, 0, 1, c));
        };
        let push_sm = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
            e.push((wl, wr, 1, 0, c));
        };
        let push_sz = |e: &mut Vec<(usize, usize, usize, usize, f64)>, wl: usize, wr: usize, c: f64| {
            e.push((wl, wr, 0, 0, 0.5 * c));
            e.push((wl, wr, 1, 1, -0.5 * c));
        };
        for i in 0..l {
            let mut e = Vec::new();
            let even = i % 2 == 0;
            if i == 0 {
                // Primer sitio (A_0): solo fila inicial (mapeada a 0)
                push_sm(&mut e, 0, 1, 1.0);
                push_sp(&mut e, 0, 2, 1.0);
                push_sz(&mut e, 0, 3, 1.0);
                push_i(&mut e, 0, 7, 1.0);
            } else if i + 1 == l {
                // Último sitio (B_{n-1}, impar): cierra B y el peldaño final
                push_i(&mut e, 0, 0, 1.0);
                push_sp(&mut e, 1, 0, 0.5 * g_xy);
                push_sm(&mut e, 2, 0, 0.5 * g_xy);
                push_sz(&mut e, 3, 0, g_z);
                push_sp(&mut e, 4, 0, 0.5 * v_b);
                push_sm(&mut e, 5, 0, 0.5 * v_b);
                push_sz(&mut e, 6, 0, delta_b);
            } else if even {
                // Sitio A_r interior: cierra A_{r-1}→A_r, pasa B_{r-1}, inicia A_r
                push_i(&mut e, 0, 0, 1.0);
                push_sp(&mut e, 1, 0, 0.5 * v_a);
                push_sm(&mut e, 2, 0, 0.5 * v_a);
                push_sz(&mut e, 3, 0, delta_a);
                push_i(&mut e, 4, 4, 1.0);
                push_i(&mut e, 5, 5, 1.0);
                push_i(&mut e, 6, 6, 1.0);
                push_sm(&mut e, 7, 1, 1.0);
                push_sp(&mut e, 7, 2, 1.0);
                push_sz(&mut e, 7, 3, 1.0);
                push_i(&mut e, 7, 7, 1.0);
            } else {
                // Sitio B_r interior: cierra peldaño A_r–B_r, pasa A_r hacia A_{r+1}, cierra B_{r-1}→B_r, inicia B_r
                push_i(&mut e, 0, 0, 1.0);
                push_sp(&mut e, 1, 0, 0.5 * g_xy);
                push_sm(&mut e, 2, 0, 0.5 * g_xy);
                push_sz(&mut e, 3, 0, g_z);
                push_i(&mut e, 1, 1, 1.0);
                push_i(&mut e, 2, 2, 1.0);
                push_i(&mut e, 3, 3, 1.0);
                push_sp(&mut e, 4, 0, 0.5 * v_b);
                push_sm(&mut e, 5, 0, 0.5 * v_b);
                push_sz(&mut e, 6, 0, delta_b);
                push_sm(&mut e, 7, 4, 1.0);
                push_sp(&mut e, 7, 5, 1.0);
                push_sz(&mut e, 7, 6, 1.0);
                push_i(&mut e, 7, 7, 1.0);
            }
            entries.push(e);
        }
        SparseMpo { l, d_bond: 8, entries }
    }
}

/// MPS real para DMRG variacional de dos sitios.
pub struct RealMps {
    pub l: usize,
    pub a: Vec<Vec<f64>>,
    pub dl: Vec<usize>,
    pub dr: Vec<usize>,
    pub bond_s: Vec<Vec<f64>>,
}

impl RealMps {
    /// Estado inicial aleatorio con dimensión de enlace χ₀ = 8 (evita quedar atrapado en el
    /// mínimo local clásico de Néel con S = 0 en cadenas críticas sin hueco).
    fn initial_neel_perturbed(l: usize, seed: u64) -> Self {
        let mut rng = crate::rng::SplitMix64::new(seed ^ 0xD009_1616);
        let mut a = Vec::with_capacity(l);
        let mut dl = vec![1; l];
        let mut dr = vec![1; l];
        let chi0 = 8usize;
        let bond_dim = |b: usize| -> usize {
            if b == 0 || b == l {
                1
            } else {
                chi0.min(1usize << b.min(l - b).min(10))
            }
        };
        for i in 0..l {
            let ldim = bond_dim(i);
            let rdim = bond_dim(i + 1);
            dl[i] = ldim;
            dr[i] = rdim;
            let mut t = vec![0.0; ldim * 2 * rdim];
            for v in t.iter_mut() {
                *v = rng.next_f64() - 0.5;
            }
            a.push(t);
        }
        let mut mps = RealMps {
            l,
            a,
            dl,
            dr,
            bond_s: vec![vec![1.0]; l.saturating_sub(1)],
        };
        mps.right_canonicalize();
        mps
    }

    fn right_canonicalize(&mut self) {
        for i in (1..self.l).rev() {
            let dl = self.dl[i];
            let cols = 2 * self.dr[i];
            let (u, s, vt) = real_svd(&self.a[i], dl, cols);
            let k = s.len();
            let sn = s.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
            self.bond_s[i - 1] = s.iter().map(|v| v / sn).collect();
            self.a[i] = vt;
            let dlp = self.dl[i - 1];
            let mut na = vec![0.0; dlp * 2 * k];
            for l in 0..dlp {
                for sp in 0..2 {
                    let src = (l * 2 + sp) * dl;
                    let dst = (l * 2 + sp) * k;
                    for c in 0..k {
                        let mut sum = 0.0;
                        for m in 0..dl {
                            sum += self.a[i - 1][src + m] * u[m * k + c];
                        }
                        na[dst + c] = sum * s[c];
                    }
                }
            }
            self.a[i - 1] = na;
            self.dl[i] = k;
            self.dr[i - 1] = k;
        }
        let n0 = self.a[0].iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        for v in self.a[0].iter_mut() {
            *v /= n0;
        }
    }

    pub fn entanglement_entropy(&self, cut: usize) -> f64 {
        if self.bond_s.is_empty() {
            return 0.0;
        }
        let mut ent = 0.0;
        for &sv in &self.bond_s[cut.min(self.bond_s.len() - 1)] {
            let p = sv * sv;
            if p > 1e-15 {
                ent -= p * p.ln();
            }
        }
        ent
    }
}

/// Actualiza el entorno izquierdo L[site+1] (forma `(d_w, dr, dr)`):
/// L_next[wr, r', r] = Σ_{wl, l', l, s', s} L[wl, l', l] · A[l', s', r'] · W[wl, wr, s', s] · A[l, s, r].
fn update_left_env(
    l_env: &[f64],
    dw_l: usize,
    dl: usize,
    a: &[f64],
    dr: usize,
    w_entries: &[(usize, usize, usize, usize, f64)],
    dw_r: usize,
) -> Vec<f64> {
    // T[wl, s, l', r] = Σ_l L[wl, l', l] · A[l, s, r]
    let mut t = vec![0.0; dw_l * 2 * dl * dr];
    for wl in 0..dw_l {
        for lp in 0..dl {
            let l_off = (wl * dl + lp) * dl;
            for l in 0..dl {
                let lv = l_env[l_off + l];
                if lv == 0.0 {
                    continue;
                }
                for s in 0..2 {
                    let a_off = (l * 2 + s) * dr;
                    let t_off = ((wl * 2 + s) * dl + lp) * dr;
                    for r in 0..dr {
                        t[t_off + r] += lv * a[a_off + r];
                    }
                }
            }
        }
    }
    // U[wr, s', l', r] = Σ_{wl, s} W[wl, wr, s', s] · T[wl, s, l', r]
    let mut u = vec![0.0; dw_r * 2 * dl * dr];
    for &(wl, wr, sp, s, val) in w_entries {
        let t_base = (wl * 2 + s) * dl * dr;
        let u_base = (wr * 2 + sp) * dl * dr;
        for idx in 0..(dl * dr) {
            u[u_base + idx] += val * t[t_base + idx];
        }
    }
    // L_next[wr, r', r] = Σ_{l', s'} A[l', s', r'] · U[wr, s', l', r]
    let mut out = vec![0.0; dw_r * dr * dr];
    for wr in 0..dw_r {
        for sp in 0..2 {
            let u_base = (wr * 2 + sp) * dl * dr;
            for lp in 0..dl {
                let a_off = (lp * 2 + sp) * dr;
                let u_off = u_base + lp * dr;
                for rp in 0..dr {
                    let av = a[a_off + rp];
                    if av == 0.0 {
                        continue;
                    }
                    let o_off = (wr * dr + rp) * dr;
                    for r in 0..dr {
                        out[o_off + r] += av * u[u_off + r];
                    }
                }
            }
        }
    }
    out
}

/// Actualiza el entorno derecho R[site] (forma `(d_w_l, dl, dl)`):
/// R_prev[wl, l', l] = Σ_{wr, r', r, s', s} R[wr, r', r] · A[l', s', r'] · W[wl, wr, s', s] · A[l, s, r].
fn update_right_env(
    r_env: &[f64],
    dw_r: usize,
    dr: usize,
    a: &[f64],
    dl: usize,
    w_entries: &[(usize, usize, usize, usize, f64)],
    dw_l: usize,
) -> Vec<f64> {
    // T[wr, s, l, r'] = Σ_r A[l, s, r] · R[wr, r', r]
    let mut t = vec![0.0; dw_r * 2 * dl * dr];
    for wr in 0..dw_r {
        for s in 0..2 {
            let t_base = (wr * 2 + s) * dl * dr;
            for l in 0..dl {
                let a_off = (l * 2 + s) * dr;
                for rp in 0..dr {
                    let r_off = (wr * dr + rp) * dr;
                    let mut sum = 0.0;
                    for r in 0..dr {
                        sum += a[a_off + r] * r_env[r_off + r];
                    }
                    t[t_base + l * dr + rp] = sum;
                }
            }
        }
    }
    // U[wl, s', l, r'] = Σ_{wr, s} W[wl, wr, s', s] · T[wr, s, l, r']
    let mut u = vec![0.0; dw_l * 2 * dl * dr];
    for &(wl, wr, sp, s, val) in w_entries {
        let t_base = (wr * 2 + s) * dl * dr;
        let u_base = (wl * 2 + sp) * dl * dr;
        for idx in 0..(dl * dr) {
            u[u_base + idx] += val * t[t_base + idx];
        }
    }
    // R_prev[wl, l', l] = Σ_{s', r'} A[l', s', r'] · U[wl, s', l, r']
    let mut out = vec![0.0; dw_l * dl * dl];
    for wl in 0..dw_l {
        for sp in 0..2 {
            let u_base = (wl * 2 + sp) * dl * dr;
            for lp in 0..dl {
                let a_off = (lp * 2 + sp) * dr;
                let o_off = (wl * dl + lp) * dl;
                for l in 0..dl {
                    let u_off = u_base + l * dr;
                    let mut sum = 0.0;
                    for rp in 0..dr {
                        sum += a[a_off + rp] * u[u_off + rp];
                    }
                    out[o_off + l] += sum;
                }
            }
        }
    }
    out
}

/// Búferes temporales reutilizables para `apply_two_site_heff` (evita realocar en cada iteración de Lanczos).
struct HeffScratch {
    t1: Vec<f64>,
    t2: Vec<f64>,
    t3: Vec<f64>,
}

impl HeffScratch {
    fn new() -> Self {
        HeffScratch { t1: Vec::new(), t2: Vec::new(), t3: Vec::new() }
    }

    #[inline]
    fn reset(buf: &mut Vec<f64>, len: usize) {
        if buf.len() < len {
            buf.resize(len, 0.0);
        }
        buf[..len].fill(0.0);
    }
}

/// Aplica el hamiltoniano efectivo de dos sitios H_eff(site, site+1) sobre Θ(l, s0, s1, r) en O(D·χ³).
#[allow(clippy::too_many_arguments)]
fn apply_two_site_heff(
    theta: &[f64],
    out: &mut [f64],
    l_env: &[f64],
    r_env: &[f64],
    w0: &[(usize, usize, usize, usize, f64)],
    w1: &[(usize, usize, usize, usize, f64)],
    dw_l: usize,
    dw_m: usize,
    dw_r: usize,
    dl: usize,
    dr: usize,
    scratch: &mut HeffScratch,
) {
    // Paso 1: T1[wl, lp, s0, s1, r] = Σ_l L[wl, lp, l] · Θ[l, s0, s1, r]
    HeffScratch::reset(&mut scratch.t1, dw_l * dl * 4 * dr);
    let t1 = &mut scratch.t1;
    for wl in 0..dw_l {
        for lp in 0..dl {
            let l_off = (wl * dl + lp) * dl;
            let dst = (wl * dl + lp) * 4 * dr;
            for l in 0..dl {
                let lv = l_env[l_off + l];
                if lv == 0.0 {
                    continue;
                }
                let src = l * 4 * dr;
                for k in 0..(4 * dr) {
                    t1[dst + k] += lv * theta[src + k];
                }
            }
        }
    }
    // Paso 2: T2[wm, lp, s0', s1, r] = Σ_{wl, s0} W0[wl, wm, s0', s0] · T1[wl, lp, s0, s1, r]
    HeffScratch::reset(&mut scratch.t2, dw_m * dl * 4 * dr);
    let t2 = &mut scratch.t2;
    for &(wl, wm, s0p, s0, val) in w0 {
        for lp in 0..dl {
            let src_b = ((wl * dl + lp) * 2 + s0) * 2 * dr;
            let dst_b = ((wm * dl + lp) * 2 + s0p) * 2 * dr;
            for k in 0..(2 * dr) {
                t2[dst_b + k] += val * t1[src_b + k];
            }
        }
    }
    // Paso 3: T3[wr, lp, s0', s1', r] = Σ_{wm, s1} W1[wm, wr, s1', s1] · T2[wm, lp, s0', s1, r]
    HeffScratch::reset(&mut scratch.t3, dw_r * dl * 4 * dr);
    let t3 = &mut scratch.t3;
    for &(wm, wr, s1p, s1, val) in w1 {
        for lp in 0..dl {
            for s0p in 0..2 {
                let src_b = (((wm * dl + lp) * 2 + s0p) * 2 + s1) * dr;
                let dst_b = (((wr * dl + lp) * 2 + s0p) * 2 + s1p) * dr;
                for r in 0..dr {
                    t3[dst_b + r] += val * t2[src_b + r];
                }
            }
        }
    }
    // Paso 4: out[lp, s0', s1', rp] = Σ_{wr, r} T3[wr, lp, s0', s1', r] · R[wr, rp, r]
    out.fill(0.0);
    for wr in 0..dw_r {
        for lp in 0..dl {
            for s01 in 0..4 {
                let src = ((wr * dl + lp) * 4 + s01) * dr;
                let dst = (lp * 4 + s01) * dr;
                for rp in 0..dr {
                    let r_off = (wr * dr + rp) * dr;
                    let mut sum = 0.0;
                    for r in 0..dr {
                        sum += t3[src + r] * r_env[r_off + r];
                    }
                    out[dst + rp] += sum;
                }
            }
        }
    }
}

/// Ejecuta DMRG variacional de dos sitios (Lanczos) para un MPO disperso arbitrario.
/// Devuelve `(energía_fundamental, mps_convergido)`.
pub fn run_dmrg(mpo: &SparseMpo, chi_max: usize, sweeps: usize, seed: u64) -> (f64, RealMps) {
    let chi_max = chi_max.max(1);
    let l = mpo.l;
    let dw = mpo.d_bond;
    let mut mps = RealMps::initial_neel_perturbed(l, seed);

    let dw_at = |bond: usize| if bond == 0 || bond == l { 1 } else { dw };

    let mut l_envs: Vec<Vec<f64>> = vec![Vec::new(); l + 1];
    let mut r_envs: Vec<Vec<f64>> = vec![Vec::new(); l + 1];
    l_envs[0] = vec![1.0];
    r_envs[l] = vec![1.0];
    for i in (0..l).rev() {
        r_envs[i] = update_right_env(&r_envs[i + 1], dw_at(i + 1), mps.dr[i], &mps.a[i], mps.dl[i], &mpo.entries[i], dw_at(i));
    }

    let mut e_best = f64::INFINITY;
    let mut heff_scratch = HeffScratch::new();
    let optimize_bond = |mps: &mut RealMps,
                         l_envs: &mut [Vec<f64>],
                         r_envs: &mut [Vec<f64>],
                         scratch: &mut HeffScratch,
                         i: usize,
                         move_right: bool,
                         chi_cap: usize|
     -> f64 {
        let dl = mps.dl[i];
        let dm = mps.dr[i];
        let dr = mps.dr[i + 1];
        // Tensor inicial de dos sitios Θ(l, s0, s1, r)
        let mut theta0 = vec![0.0; dl * 4 * dr];
        for ll in 0..dl {
            for s0 in 0..2 {
                for m in 0..dm {
                    let v0 = mps.a[i][(ll * 2 + s0) * dm + m];
                    if v0 == 0.0 {
                        continue;
                    }
                    for s1 in 0..2 {
                        let a1_off = (m * 2 + s1) * dr;
                        let t_off = ((ll * 2 + s0) * 2 + s1) * dr;
                        for r in 0..dr {
                            theta0[t_off + r] += v0 * mps.a[i + 1][a1_off + r];
                        }
                    }
                }
            }
        }
        let dw_l = dw_at(i);
        let dw_m = dw_at(i + 1);
        let dw_r = dw_at(i + 2);
        let (e0, theta_opt) = lanczos_ground(dl * 4 * dr, &theta0, 14, |x, y| {
            apply_two_site_heff(x, y, &l_envs[i], &r_envs[i + 2], &mpo.entries[i], &mpo.entries[i + 1], dw_l, dw_m, dw_r, dl, dr, scratch);
        });
        let rows = dl * 2;
        let cols = 2 * dr;
        let (u, svals, vt) = real_svd(&theta_opt, rows, cols);
        let k_full = svals.len();
        let smax = svals.first().copied().unwrap_or(0.0).max(1e-30);
        let mut chi = 0usize;
        for &sv in svals.iter().take(chi_cap.min(k_full)) {
            if sv > 1e-11 * smax {
                chi += 1;
            }
        }
        chi = chi.max(1);
        let sn = svals[..chi].iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        mps.bond_s[i] = svals[..chi].iter().map(|v| v / sn).collect();

        let mut na0 = vec![0.0; rows * chi];
        let mut na1 = vec![0.0; chi * cols];
        if move_right {
            for r in 0..rows {
                for c in 0..chi {
                    na0[r * chi + c] = u[r * k_full + c];
                }
            }
            for c in 0..chi {
                let sv = svals[c] / sn;
                for col in 0..cols {
                    na1[c * cols + col] = vt[c * cols + col] * sv;
                }
            }
            mps.a[i] = na0;
            mps.a[i + 1] = na1;
            mps.dr[i] = chi;
            mps.dl[i + 1] = chi;
            l_envs[i + 1] = update_left_env(&l_envs[i], dw_l, dl, &mps.a[i], chi, &mpo.entries[i], dw_m);
        } else {
            for r in 0..rows {
                for c in 0..chi {
                    na0[r * chi + c] = u[r * k_full + c] * (svals[c] / sn);
                }
            }
            for c in 0..chi {
                for col in 0..cols {
                    na1[c * cols + col] = vt[c * cols + col];
                }
            }
            mps.a[i] = na0;
            mps.a[i + 1] = na1;
            mps.dr[i] = chi;
            mps.dl[i + 1] = chi;
            r_envs[i + 1] = update_right_env(&r_envs[i + 2], dw_r, dr, &mps.a[i + 1], chi, &mpo.entries[i + 1], dw_m);
        }
        e0
    };

    for sw in 0..sweeps.max(1) {
        let chi_cap = if sw == 0 { (chi_max / 2).max(8).min(chi_max) } else { chi_max };
        for i in 0..(l - 1) {
            let e = optimize_bond(&mut mps, &mut l_envs, &mut r_envs, &mut heff_scratch, i, true, chi_cap);
            if e < e_best {
                e_best = e;
            }
        }
        for i in (0..(l - 1)).rev() {
            let e = optimize_bond(&mut mps, &mut l_envs, &mut r_envs, &mut heff_scratch, i, false, chi_cap);
            if e < e_best {
                e_best = e;
            }
        }
    }
    (e_best, mps)
}

/// Calcula la correlación de peldaño ⟨S^z_{2r} S^z_{2r+1}⟩ y la velocidad/mezcla de especies en el
/// estado fundamental DMRG de la escalera de dos especies.
fn measure_rung_zz_mean(mps: &RealMps) -> f64 {
    let l = mps.l;
    let mut r_env: Vec<Vec<f64>> = vec![Vec::new(); l + 1];
    r_env[l] = vec![1.0];
    for i in (0..l).rev() {
        let dl = mps.dl[i];
        let dr = mps.dr[i];
        let rn = &r_env[i + 1];
        let ai = &mps.a[i];
        let mut cur = vec![0.0; dl * dl];
        for s in 0..2 {
            let mut tmp = vec![0.0; dl * dr];
            for ll in 0..dl {
                for r in 0..dr {
                    let av = ai[(ll * 2 + s) * dr + r];
                    if av == 0.0 {
                        continue;
                    }
                    for rp in 0..dr {
                        tmp[ll * dr + rp] += av * rn[r * dr + rp];
                    }
                }
            }
            for lp in 0..dl {
                for ll in 0..dl {
                    let mut sum = 0.0;
                    for rp in 0..dr {
                        sum += ai[(lp * 2 + s) * dr + rp] * tmp[ll * dr + rp];
                    }
                    cur[lp * dl + ll] += sum;
                }
            }
        }
        r_env[i] = cur;
    }
    let norm2 = r_env[0][0].max(1e-30);
    let mut l_env = vec![1.0];
    let mut sum_zz = 0.0;
    let n_rungs = l / 2;
    for i in 0..l {
        let dl = mps.dl[i];
        let dr = mps.dr[i];
        let ai = &mps.a[i];
        if i % 2 == 0 && i + 1 < l {
            let dm = dr;
            let dr1 = mps.dr[i + 1];
            let a1 = &mps.a[i + 1];
            let rn = &r_env[i + 2];
            let mut val = 0.0;
            for s0 in 0..2 {
                let z0 = if s0 == 0 { 0.5 } else { -0.5 };
                for s1 in 0..2 {
                    let z1 = if s1 == 0 { 0.5 } else { -0.5 };
                    let mut t = vec![0.0; dl * dr1];
                    for ll in 0..dl {
                        for m in 0..dm {
                            let v0 = ai[(ll * 2 + s0) * dm + m];
                            if v0 == 0.0 {
                                continue;
                            }
                            for r in 0..dr1 {
                                t[ll * dr1 + r] += v0 * a1[(m * 2 + s1) * dr1 + r];
                            }
                        }
                    }
                    let mut lt = vec![0.0; dl * dr1];
                    for lp in 0..dl {
                        for ll in 0..dl {
                            let lv = l_env[lp * dl + ll];
                            for r in 0..dr1 {
                                lt[lp * dr1 + r] += lv * t[ll * dr1 + r];
                            }
                        }
                    }
                    let mut tr = 0.0;
                    for lp in 0..dl {
                        for rp in 0..dr1 {
                            let b = t[lp * dr1 + rp];
                            for r in 0..dr1 {
                                tr += b * lt[lp * dr1 + r] * rn[r * dr1 + rp];
                            }
                        }
                    }
                    val += z0 * z1 * tr;
                }
            }
            sum_zz += val / norm2;
        }
        // Avanzar l_env
        let mut nl = vec![0.0; dr * dr];
        for s in 0..2 {
            let mut m1 = vec![0.0; dl * dr];
            for lp in 0..dl {
                for ll in 0..dl {
                    let lv = l_env[lp * dl + ll];
                    for r in 0..dr {
                        m1[lp * dr + r] += lv * ai[(ll * 2 + s) * dr + r];
                    }
                }
            }
            for rp in 0..dr {
                for r in 0..dr {
                    let mut sum = 0.0;
                    for lp in 0..dl {
                        sum += ai[(lp * 2 + s) * dr + rp] * m1[lp * dr + r];
                    }
                    nl[rp * dr + r] += sum;
                }
            }
        }
        l_env = nl;
    }
    sum_zz / n_rungs.max(1) as f64
}

// ------------------------------------------------------------------ 1. Escalado al continuo + CFT en Heisenberg XXX

/// Ejecuta DMRG variacional en la cadena crítica interactuante XXZ en el punto de Razumov–Stroganov
/// (Δ = 1/2, H = Σ [S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + ½ S^z_i S^z_{i+1}], parámetro de Luttinger
/// K = 3/4 sin corrección logarítmica marginal SU(2)) para L ∈ {16, 24, 32, 48}, extrapolando e_∞
/// al valor exacto de Bethe −3/8 = −0,375000 y extrayendo la carga central CFT de Calabrese–Cardy
/// S_vN(ℓ) = (c/6) ln[(2L/π) sin(πℓ/L)] + s₀ (c = 1 exacto).
pub fn run_dmrg_xxz_cft_scaling(chi_max: usize, sweeps: usize) -> Vec<f64> {
    let ls = [16usize, 24, 32, 48];
    let run_l = |idx: usize, l: usize| -> (f64, Vec<f64>) {
        let mpo = SparseMpo::xxz(l, 1.0, 0.5);
        let sw_l = if l >= 32 { sweeps.max(6) } else { sweeps.max(5) };
        let (e0, mps) = run_dmrg(&mpo, chi_max, sw_l, 1601 + idx as u64 * 17);
        let prof: Vec<f64> = (0..(l - 1)).map(|cut| mps.entanglement_entropy(cut)).collect();
        (e0 / l as f64, prof)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let runs: Vec<(f64, Vec<f64>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = ls.iter().enumerate().map(|(i, &l)| sc.spawn(move || run_l(i, l))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let runs: Vec<(f64, Vec<f64>)> = ls.iter().enumerate().map(|(i, &l)| run_l(i, l)).collect();

    let e_per_site: Vec<f64> = runs.iter().map(|r| r.0).collect();
    let s_profile = runs.last().unwrap().1.clone();
    // Extrapolación de tamaño finito: e(L) = e_∞ + b/L + c/L² (mínimos cuadrados 3×3)
    let mut g = [[0.0f64; 3]; 3];
    let mut h = [0.0f64; 3];
    for (&l, &e) in ls.iter().zip(&e_per_site) {
        let inv = 1.0 / l as f64;
        let v = [1.0, inv, inv * inv];
        for r in 0..3 {
            h[r] += v[r] * e;
            for c in 0..3 {
                g[r][c] += v[r] * v[c];
            }
        }
    }
    let coef = crate::util::solve_n::<3>(g, h);
    let e_inf_extrap = coef[0];
    let e_bethe = -0.375f64; // −3/8 exacto (Yang–Yang / Razumov–Stroganov en Δ = 1/2)

    // Carga central CFT de Calabrese–Cardy por escalado de tamaño finito en el corte central
    // promediado par/impar S̄(L/2) = ½[S_L(L/2) + S_L(L/2 − 1)] frente a la distancia de cuerda
    // x̄(L) = ½[ln(2L/π) + ln((2L/π) cos(π/L))] a través de L ∈ {16, 24, 32, 48}:
    // cancela la oscilación de dimerización por bordes abiertos y los efectos de borde de ℓ pequeño.
    let mut x_fs = Vec::with_capacity(ls.len());
    let mut s_fs = Vec::with_capacity(ls.len());
    for (&l, (_, prof)) in ls.iter().zip(&runs) {
        let lf = l as f64;
        let s_even = prof[l / 2 - 1];
        let s_odd = prof[l / 2 - 2];
        let x_even = (2.0 * lf / PI).ln();
        let x_odd = ((2.0 * lf / PI) * (PI / lf).cos()).ln();
        x_fs.push(0.5 * (x_even + x_odd));
        s_fs.push(0.5 * (s_even + s_odd));
    }
    // Usamos L ∈ {24, 32, 48} para minimizar las correcciones subdominantes O(1/L^{2K}) de L = 16
    let (_, slope_fs) = crate::util::fit_linear(&x_fs[1..], &s_fs[1..]);
    let c_cft = 6.0 * slope_fs;

    // Perfil interior de L = 48 sin el término alternante de Friedel (−1)^ℓ / d_chord^K (K = 3/4 en Δ = 1/2)
    let l_big = *ls.last().unwrap() as f64;
    let mut g_cft = [[0.0f64; 3]; 3];
    let mut h_cft = [0.0f64; 3];
    let mut curve_pts = Vec::new();
    for cut in 3..(s_profile.len() - 3) {
        let ell = (cut + 1) as f64;
        let d_chord = (2.0 * l_big / PI) * (PI * ell / l_big).sin();
        let x = d_chord.ln();
        let alt = if (cut + 1) % 2 == 0 { 1.0 } else { -1.0 } / d_chord.powf(0.75);
        let v = [x, 1.0, alt];
        let y = s_profile[cut];
        for r in 0..3 {
            h_cft[r] += v[r] * y;
            for c in 0..3 {
                g_cft[r][c] += v[r] * v[c];
            }
        }
    }
    let cft_coef = crate::util::solve_n::<3>(g_cft, h_cft);
    let a_osc = cft_coef[2];
    for cut in 3..(s_profile.len() - 3) {
        let ell = (cut + 1) as f64;
        let d_chord = (2.0 * l_big / PI) * (PI * ell / l_big).sin();
        let x = d_chord.ln();
        let alt = if (cut + 1) % 2 == 0 { 1.0 } else { -1.0 } / d_chord.powf(0.75);
        let s_uni = s_profile[cut] - a_osc * alt;
        curve_pts.push((ell, s_uni, x));
    }
    let s_mid = s_profile[s_profile.len() / 2];

    // Cabecera: [e_inf_extrap, e_bethe, err_rel_bethe, c_cft, s_mid, e(16), e(24), e(32), e(48), n_pts, ...]
    let mut out = vec![
        e_inf_extrap,
        e_bethe,
        (e_inf_extrap / e_bethe - 1.0).abs(),
        c_cft,
        s_mid,
        e_per_site[0],
        e_per_site[1],
        e_per_site[2],
        e_per_site[3],
        curve_pts.len() as f64,
    ];
    for (ell, s, ch) in curve_pts {
        out.push(ell);
        out.push(s);
        out.push(ch);
    }
    out
}

// ------------------------------------------------------------------ 2. Dos especies acopladas + flujo RG Chadha–Nielsen

/// Investiga la restauración infrarroja de la simetría de Lorentz (Collins et al. 2004 vs
/// Chadha y Nielsen 1983):
/// 1) DMRG no perturbativo en una escalera 1+1D de dos especies con velocidades desnudas distintas
///    (v_A ≠ v_B): mide la energía de hibridación, las correlaciones de peldaño ⟨S^z_A S^z_B⟩ y la
///    entropía de entrelazamiento entre especies al pasar de g_AB = 0 a g_AB > 0.
/// 2) Integración exacta del flujo RG a un lazo de Chadha–Nielsen desde M_P = 1,22·10¹⁹ GeV hasta
///    m_e = 5,11·10⁻⁴ GeV para QED débil (α = 1/137) frente a un sector fuertemente acoplado con
///    dimensión anómala γ_* = 0,90, calculando el γ_* mínimo requerido para bajar δc de 0,02 a < 10⁻²⁰.
pub fn run_two_species_lorentz_rg(n_rungs: usize, v_a: f64, v_b: f64, g_ab: f64, chi_max: usize) -> Vec<f64> {
    let nr = n_rungs.clamp(6, 16);
    let chi = chi_max.clamp(12, 48);
    // 1. DMRG desacoplado (g_AB = 0) frente a acoplado (g_AB > 0)
    let mpo_free = SparseMpo::two_species_ladder(nr, v_a, v_b, 0.5 * v_a, 0.5 * v_b, 0.0);
    let (e_free, mps_free) = run_dmrg(&mpo_free, chi, 4, 1602);
    let zz_free = measure_rung_zz_mean(&mps_free);
    let s_free = mps_free.entanglement_entropy(nr - 1);

    let mpo_coup = SparseMpo::two_species_ladder(nr, v_a, v_b, 0.5 * v_a, 0.5 * v_b, g_ab);
    let (e_coup, mps_coup) = run_dmrg(&mpo_coup, chi, 4, 1603);
    let zz_coup = measure_rung_zz_mean(&mps_coup);
    let s_coup = mps_coup.entanglement_entropy(nr - 1);

    // Energía de ligadura / hibridación por peldaño (e_coup < e_free cuando g_AB > 0 forma singletes entre especies)
    let de_bind = (e_free - e_coup) / nr as f64;

    // 2. Flujo RG exacto de Chadha–Nielsen (1983) desde M_P = 1,22089e19 GeV hasta m_e = 5,11e-4 GeV
    //    dα/dt = (2/3π) α²,  d(δc)/dt = (8/3π) α · δc   (t = ln μ)
    //    Con α(m_e) = 1/137,036, α(M_P) se obtiene integrando hacia arriba desde m_e hasta M_P.
    let mu_ir = 0.51099895e-3f64;
    let mu_uv = 1.22089e19f64;
    let dt_tot = (mu_uv / mu_ir).ln(); // 51,528
    let n_steps = 200usize;
    let dt = dt_tot / n_steps as f64;
    let alpha_ir = 1.0 / 137.035999;
    let b_qed = 2.0 / (3.0 * PI);
    // α(μ) exacta a 1 lazo: 1/α(μ) = 1/α_IR − b_QED · ln(μ / μ_IR)
    let alpha_uv = 1.0 / (1.0 / alpha_ir - b_qed * dt_tot);
    // Integración RK4 de δc(t) bajando desde μ_UV (δc_0 = 0,0207 de Collins et al., Fase 15) hasta μ_IR
    let dc_uv = 0.0207f64;
    let mut dc_qed = dc_uv;
    let mut dc_strong = dc_uv;
    let gamma_strong = 0.90f64;
    let mut curve = Vec::with_capacity(n_steps + 1);
    for step in 0..=n_steps {
        let frac = step as f64 / n_steps as f64; // 0 = UV (M_P), 1 = IR (m_e)
        let log10_mu = mu_uv.log10() * (1.0 - frac) + mu_ir.log10() * frac;
        if step % 4 == 0 {
            curve.push((log10_mu, dc_qed.log10(), dc_strong.log10()));
        }
        if step < n_steps {
            let t_mid = dt_tot * (1.0 - (step as f64 + 0.5) / n_steps as f64);
            let a_mid = 1.0 / (1.0 / alpha_ir - b_qed * t_mid);
            let g_qed = 4.0 * b_qed * a_mid; // 8α/(3π)
            dc_qed *= (-g_qed * dt).exp();
            dc_strong *= (-gamma_strong * dt).exp();
        }
    }
    let ratio_qed = dc_qed / dc_uv;
    let ratio_qed_analytic = (alpha_ir / alpha_uv).powi(4);
    let gamma_min_required = (dc_uv / 1e-20).ln() / dt_tot; // ≈ 0,819

    // Cabecera (10): [de_bind, zz_free, zz_coup, s_free, s_coup, ratio_qed, ratio_qed_analytic, dc_strong_ir, gamma_min_required, n_pts, ...]
    let mut out = vec![
        de_bind,
        zz_free,
        zz_coup,
        s_free,
        s_coup,
        ratio_qed,
        ratio_qed_analytic,
        dc_strong,
        gamma_min_required,
        curve.len() as f64,
    ];
    for (lmu, lq, ls) in curve {
        out.push(lmu);
        out.push(lq);
        out.push(ls);
    }
    out
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 16 (DMRG variacional de dos sitios con Lanczos): escalado al continuo en la cadena
    /// crítica interactuante XXZ (Δ = 1/2, L = 16, 24, 32, 48) frente a la energía exacta de Bethe
    /// (−3/8 = −0,375000) y extracción de la carga central CFT de Calabrese–Cardy (c = 1).
    /// Devuelve puntero a [e_inf, e_bethe, err_rel, c_cft, S_centro, e16, e24, e32, e48, n_pts, ell, S, chord, …].
    pub fn dmrg_xxz_cft_scaling(&mut self, chi_max: u32, sweeps: u32) -> *const f64 {
        let chi = (chi_max as usize).clamp(12, 64);
        let sw = (sweeps as usize).clamp(2, 8);
        self.oq.out = run_dmrg_xxz_cft_scaling(chi, sw);
        self.oq.out.as_ptr()
    }

    /// Fase 16 (Collins et al. 2004 frente a Chadha y Nielsen 1983): DMRG de dos especies acopladas
    /// en 1+1D + integración exacta del flujo RG desde M_P hasta m_e en QED débil frente a sector fuerte.
    /// Devuelve puntero a [dE_lig, zz_libre, zz_acopl, S_libre, S_acopl, ratio_QED, ratio_QED_exact, dc_fuerte_IR, gamma_min, n_pts, log10_mu, log10_dc_QED, log10_dc_fuerte, …].
    pub fn two_species_lorentz_rg_study(&mut self, n_rungs: u32, v_a: f32, v_b: f32, g_ab: f32, chi_max: u32) -> *const f64 {
        self.oq.out = run_two_species_lorentz_rg(
            n_rungs as usize,
            clamp_finite(v_a as f64, 0.2, 2.0, 1.0),
            clamp_finite(v_b as f64, 0.2, 2.0, 0.5),
            clamp_finite(g_ab as f64, 0.0, 3.0, 0.5),
            chi_max as usize,
        );
        self.oq.out.as_ptr()
    }
}
