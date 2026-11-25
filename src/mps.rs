//! Fase 15 — Teoría cuántica de campos de muchos cuerpos en 1+1D mediante Matrix Product States
//! (MPS + TEBD + MPO de flujo eléctrico exacto):
//!
//! 1. **Modelo de Schwinger cuántico (QED 1+1D con fermiones interactuantes y campo eléctrico
//!    cuantizado):** red escalonada de Kogut–Susskind transformada por Jordan–Wigner,
//!    H = w Σ (σ⁺_n σ⁻_{n+1} + h.c.) + (m/2) Σ (−1)^n σ^z_n + J Σ E_n²,
//!    con ley de Gauss exacta E_n = Σ_{j≤n} [½(σ^z_j + (−1)^j) + q_ext,j].
//!    La evolución de exp(−i dt J Σ E_n²) y exp(−dτ J Σ E_n²) se aplica mediante un MPO exacto
//!    de dimensión de enlace D = 5 (flujos E ∈ {−2, −1, 0, +1, +2}) con compresión *zip-up*
//!    por SVD de Jacobi complejo (precisión ~10⁻¹⁵ sin dependencias externas).
//!    Reproduce:
//!      - **Confinamiento cuántico real:** potencial lineal V(d) = E₀(d) − E₀(0) con tensión de cuerda.
//!      - **Rotura cuántica de la cuerda en tiempo real** con crecimiento de la entropía de
//!        entrelazamiento de von Neumann S_vN(t) a través del corte central.
//! 2. **Renormalización de la velocidad del cono de luz por interacciones en 1+1D (MPS):**
//!    cadena cuántica interactuante (fermiones sin espín / modelo XXZ-Thirring) donde el acoplamiento
//!    Δ modifica la velocidad efectiva del frente de correlación respecto al límite libre Δ = 0.

use crate::linalg::complete_complex_isometry;
use crate::util::clamp_finite;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use wasm_bindgen::prelude::*;

type C = Complex<f64>;

#[inline(always)]
fn cx(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

/// SVD complejo exacto de una matriz A de m × n (fila mayor): A = U · diag(s) · V†,
/// con k = min(m, n), U (m × k), s (k, ordenados decrecientemente), V† (k × n).
/// Usa rotaciones de Jacobi unilaterales de Hestenes directamente sobre las columnas.
pub(crate) fn complex_svd(a: &[C], m: usize, n: usize) -> (Vec<C>, Vec<f64>, Vec<C>) {
    if m < n {
        // Transponemos conjugando: A† (n × m) = U' · S · V'† ⇒ A = V' · S · U'†
        let mut ad = vec![cx(0.0, 0.0); n * m];
        for i in 0..m {
            for j in 0..n {
                ad[j * m + i] = a[i * n + j].conj();
            }
        }
        let (u_t, s, vt_t) = complex_svd_tall(&ad, n, m);
        let k = s.len();
        // U = (V'†)† (m × k), V† = U'† (k × n)
        let mut u = vec![cx(0.0, 0.0); m * k];
        for i in 0..m {
            for j in 0..k {
                u[i * k + j] = vt_t[j * m + i].conj();
            }
        }
        let mut vt = vec![cx(0.0, 0.0); k * n];
        for i in 0..k {
            for j in 0..n {
                vt[i * n + j] = u_t[j * k + i].conj();
            }
        }
        return (u, s, vt);
    }
    complex_svd_tall(a, m, n)
}

/// SVD de Hestenes para matriz alta/cuadrada (m ≥ n).
fn complex_svd_tall(a: &[C], m: usize, n: usize) -> (Vec<C>, Vec<f64>, Vec<C>) {
    // Column-major para acceso contiguo por columna: cols[j][i] = A[i, j]
    let mut cols: Vec<Vec<C>> = (0..n).map(|j| (0..m).map(|i| a[i * n + j]).collect()).collect();
    let mut vcols: Vec<Vec<C>> = (0..n)
        .map(|j| {
            let mut v = vec![cx(0.0, 0.0); n];
            v[j] = cx(1.0, 0.0);
            v
        })
        .collect();

    let max_sweeps = 40;
    for _ in 0..max_sweeps {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gam = cx(0.0, 0.0);
                for i in 0..m {
                    let ap = cols[p][i];
                    let aq = cols[q][i];
                    alpha += ap.norm_sqr();
                    beta += aq.norm_sqr();
                    gam += ap.conj() * aq;
                }
                let r = gam.norm();
                if r <= 1e-14 * (alpha * beta).sqrt().max(1e-300) {
                    continue;
                }
                converged = false;
                let e_ip = gam / r; // e^{iφ}
                let e_im = e_ip.conj(); // e^{−iφ}
                let tau = (beta - alpha) / (2.0 * r);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let se_im = e_im * s;
                let se_ip = e_ip * s;
                for i in 0..m {
                    let ap = cols[p][i];
                    let aq = cols[q][i];
                    cols[p][i] = ap * c - aq * se_im;
                    cols[q][i] = ap * se_ip + aq * c;
                }
                for i in 0..n {
                    let vp = vcols[p][i];
                    let vq = vcols[q][i];
                    vcols[p][i] = vp * c - vq * se_im;
                    vcols[q][i] = vp * se_ip + vq * c;
                }
            }
        }
        if converged {
            break;
        }
    }

    let norms: Vec<f64> = (0..n).map(|j| cols[j].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| norms[j].total_cmp(&norms[i]));

    let k = n;
    let mut u = vec![cx(0.0, 0.0); m * k];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![cx(0.0, 0.0); k * n];
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
            vt[idx * n + j] = vcols[col][j].conj();
        }
    }
    if rank < k {
        complete_complex_isometry(&mut u, m, k, rank);
    }
    (u, s, vt)
}

/// Estado Producto de Matrices (MPS) de L sitios con dimensión física d = 2.
/// Cada tensor `a[i]` tiene forma `(dl, 2, dr)` almacenado en orden `(l * 2 + s) * dr + r`.
pub struct Mps {
    pub l: usize,
    pub a: Vec<Vec<C>>,
    pub dl: Vec<usize>,
    pub dr: Vec<usize>,
    /// Valores singulares en el último barrido por cada enlace `0..L-1`.
    pub bond_s: Vec<Vec<f64>>,
}

impl Mps {
    /// Estado producto computacional `bits[i] ∈ {0, 1}` (0 = |↑⟩ con σ^z = +1, 1 = |↓⟩ con σ^z = −1).
    pub fn from_product(bits: &[usize]) -> Self {
        let l = bits.len();
        let mut a = Vec::with_capacity(l);
        for &b in bits {
            let mut t = vec![cx(0.0, 0.0); 2];
            t[b.min(1)] = cx(1.0, 0.0);
            a.push(t);
        }
        Mps {
            l,
            a,
            dl: vec![1; l],
            dr: vec![1; l],
            bond_s: vec![vec![1.0]; l.saturating_sub(1)],
        }
    }

    /// Vacío de Néel de la red escalonada de Kogut–Susskind:
    /// par (n = 0, 2, …): |↓⟩ (σ^z = −1, Q = 0); impar (n = 1, 3, …): |↑⟩ (σ^z = +1, Q = 0).
    pub fn neel_vacuum(l: usize) -> Self {
        let bits: Vec<usize> = (0..l).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        Self::from_product(&bits)
    }

    /// Puerta de un sitio `u` (2 × 2, fila mayor) en el sitio `i`.
    pub fn apply_one_site(&mut self, i: usize, u: &[C; 4]) {
        let dl = self.dl[i];
        let dr = self.dr[i];
        let src = &self.a[i];
        let mut dst = vec![cx(0.0, 0.0); dl * 2 * dr];
        for l in 0..dl {
            for r in 0..dr {
                let x0 = src[(l * 2) * dr + r];
                let x1 = src[(l * 2 + 1) * dr + r];
                dst[(l * 2) * dr + r] = u[0] * x0 + u[1] * x1;
                dst[(l * 2 + 1) * dr + r] = u[2] * x0 + u[3] * x1;
            }
        }
        self.a[i] = dst;
    }

    /// Puerta de dos sitios `u` (4 × 4, fila mayor, índice `2*s0 + s1`) en el enlace `(i, i + 1)`,
    /// con compresión SVD hasta `chi_max` y umbral relativo `tol`.
    pub fn apply_two_site(&mut self, i: usize, u: &[C; 16], chi_max: usize, tol: f64) {
        let dl = self.dl[i];
        let dm = self.dr[i];
        let dr = self.dr[i + 1];
        let a0 = &self.a[i];
        let a1 = &self.a[i + 1];
        // Theta[l, s0, s1, r] = Σ_m A0[l, s0, m] A1[m, s1, r]
        let mut th = vec![cx(0.0, 0.0); dl * 4 * dr];
        for l in 0..dl {
            for s0 in 0..2 {
                for m in 0..dm {
                    let v0 = a0[(l * 2 + s0) * dm + m];
                    if v0.re == 0.0 && v0.im == 0.0 {
                        continue;
                    }
                    for s1 in 0..2 {
                        let base1 = (m * 2 + s1) * dr;
                        let base_t = ((l * 2 + s0) * 2 + s1) * dr;
                        for r in 0..dr {
                            th[base_t + r] += v0 * a1[base1 + r];
                        }
                    }
                }
            }
        }
        // Aplicar U (4 × 4) sobre (s0, s1) y reordenar como matriz (dl * 2) × (2 * dr)
        let rows = dl * 2;
        let cols = 2 * dr;
        let mut mat = vec![cx(0.0, 0.0); rows * cols];
        for l in 0..dl {
            for r in 0..dr {
                let mut x = [cx(0.0, 0.0); 4];
                for s in 0..4 {
                    x[s] = th[(l * 4 + s) * dr + r];
                }
                for out_s in 0..4 {
                    let mut val = cx(0.0, 0.0);
                    for in_s in 0..4 {
                        val += u[out_s * 4 + in_s] * x[in_s];
                    }
                    let s0 = out_s >> 1;
                    let s1 = out_s & 1;
                    mat[(l * 2 + s0) * cols + (s1 * dr + r)] = val;
                }
            }
        }
        let (umat, svals, vt) = complex_svd(&mat, rows, cols);
        let k_full = svals.len();
        let s0_max = svals.first().copied().unwrap_or(0.0).max(1e-30);
        let mut chi = 0usize;
        for &sv in svals.iter().take(chi_max.min(k_full)) {
            if sv > tol * s0_max {
                chi += 1;
            }
        }
        chi = chi.max(1);
        let s_norm = svals[..chi].iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        self.bond_s[i] = svals[..chi].iter().map(|v| v / s_norm).collect();
        // Repartimos √s entre ambos tensores (forma equilibrada) o s en el derecho
        let mut na0 = vec![cx(0.0, 0.0); dl * 2 * chi];
        for row in 0..rows {
            for c in 0..chi {
                na0[row * chi + c] = umat[row * k_full + c];
            }
        }
        let mut na1 = vec![cx(0.0, 0.0); chi * 2 * dr];
        for c in 0..chi {
            let sv = svals[c];
            for col in 0..cols {
                na1[c * cols + col] = vt[c * cols + col] * sv;
            }
        }
        self.a[i] = na0;
        self.a[i + 1] = na1;
        self.dr[i] = chi;
        self.dl[i + 1] = chi;
    }

    /// Normaliza el MPS canónicamente (barrido izquierda → derecha y derecha → izquierda),
    /// actualizando los valores singulares exactos de cada enlace para S_vN.
    pub fn canonicalize_and_normalize(&mut self) -> f64 {
        // Barrido izquierda → derecha (QR vía SVD)
        for i in 0..(self.l - 1) {
            let rows = self.dl[i] * 2;
            let cols = self.dr[i];
            let (u, s, vt) = complex_svd(&self.a[i], rows, cols);
            let k = s.len();
            let mut na = vec![cx(0.0, 0.0); rows * k];
            for r in 0..rows {
                for c in 0..k {
                    na[r * k + c] = u[r * k + c];
                }
            }
            // Absorber S · V† en a[i + 1]
            let dr_next = self.dr[i + 1];
            let mut nb = vec![cx(0.0, 0.0); k * 2 * dr_next];
            for c in 0..k {
                let sv = s[c];
                for m in 0..cols {
                    let f = vt[c * cols + m] * sv;
                    if f.re == 0.0 && f.im == 0.0 {
                        continue;
                    }
                    for s_idx in 0..2 {
                        let src_off = (m * 2 + s_idx) * dr_next;
                        let dst_off = (c * 2 + s_idx) * dr_next;
                        for r in 0..dr_next {
                            nb[dst_off + r] += f * self.a[i + 1][src_off + r];
                        }
                    }
                }
            }
            self.a[i] = na;
            self.a[i + 1] = nb;
            self.dr[i] = k;
            self.dl[i + 1] = k;
        }
        // Norma en el último sitio
        let last = self.l - 1;
        let norm = self.a[last].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-30);
        let inv = 1.0 / norm;
        for z in self.a[last].iter_mut() {
            *z *= inv;
        }
        // Barrido derecha → izquierda: obtiene los valores singulares exactos de Schmidt en cada enlace
        for i in (1..self.l).rev() {
            let dl = self.dl[i];
            let cols = 2 * self.dr[i];
            let (u, s, vt) = complex_svd(&self.a[i], dl, cols);
            let k = s.len();
            let s_nrm = s.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
            self.bond_s[i - 1] = s.iter().map(|v| v / s_nrm).collect();
            let mut nb = vec![cx(0.0, 0.0); k * cols];
            for c in 0..k {
                for col in 0..cols {
                    nb[c * cols + col] = vt[c * cols + col];
                }
            }
            // Absorber U · S en a[i - 1]
            let dl_prev = self.dl[i - 1];
            let mut na = vec![cx(0.0, 0.0); dl_prev * 2 * k];
            for l in 0..dl_prev {
                for s_idx in 0..2 {
                    let src_off = (l * 2 + s_idx) * dl;
                    let dst_off = (l * 2 + s_idx) * k;
                    for c in 0..k {
                        let sv = s[c];
                        let mut sum = cx(0.0, 0.0);
                        for m in 0..dl {
                            sum += self.a[i - 1][src_off + m] * u[m * k + c];
                        }
                        na[dst_off + c] = sum * sv;
                    }
                }
            }
            self.a[i] = nb;
            self.a[i - 1] = na;
            self.dl[i] = k;
            self.dr[i - 1] = k;
        }
        // Normalizar sitio 0
        let n0 = self.a[0].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-30);
        for z in self.a[0].iter_mut() {
            *z /= n0;
        }
        norm
    }

    /// Aplica el operador diagonal exacto de campo eléctrico cuantizado
    ///   exp(−α · Σ_{n=0}^{L−2} E_n²)   (α = i·dt·J en tiempo real; α = dτ·J en tiempo imaginario)
    /// mediante el MPO exacto de flujo eléctrico (E ∈ {−2, −1, 0, +1, +2}, D = 5) con compresión *zip-up*.
    pub fn apply_schwinger_electric_mpo(&mut self, alpha: C, q_ext: &[i32], chi_max: usize, tol: f64) {
        const E_MIN: i32 = -2;
        const E_MAX: i32 = 2;
        const D_FLUX: usize = (E_MAX - E_MIN + 1) as usize; // 5 estados de flujo
        let e_zero = (-E_MIN) as usize; // índice 2 corresponde a E = 0

        let mut c_rem = vec![cx(1.0, 0.0)]; // forma (chi_new=1, d_left=1, chi_old=1)
        let mut chi_new = 1usize;
        let mut d_left = 1usize;
        // En el sitio 0 solo entra el flujo E = 0 (un único estado)
        let mut left_fluxes: Vec<i32> = vec![0];

        for i in 0..self.l {
            let chi_old = self.dl[i];
            let chi_next = self.dr[i];
            let is_last = i + 1 == self.l;
            let d_right = if is_last { 1 } else { D_FLUX };
            let rows = chi_new * 2;
            let cols = d_right * chi_next;
            let mut tmat = vec![cx(0.0, 0.0); rows * cols];

            let stag = if i % 2 == 0 { 1i32 } else { -1i32 };
            let qe = q_ext[i];
            let ai = &self.a[i];

            for s in 0..2usize {
                let sz = if s == 0 { 1i32 } else { -1i32 };
                // Carga escalonada exacta Q_i = ½(σ^z + (−1)^i) + q_ext,i
                let qi = (sz + stag) / 2 + qe;
                for (el_idx, &e_in) in left_fluxes.iter().enumerate() {
                    let e_out = e_in + qi;
                    if is_last {
                        // Exigimos carga total nula al final de la cadena abierta (E_{L-1} = 0)
                        if e_out != 0 {
                            continue;
                        }
                    } else if !(E_MIN..=E_MAX).contains(&e_out) {
                        continue;
                    }
                    let er_idx = if is_last { 0 } else { (e_out - E_MIN) as usize };
                    let weight = if is_last {
                        cx(1.0, 0.0)
                    } else {
                        (-alpha * ((e_out * e_out) as f64)).exp()
                    };
                    for a_idx in 0..chi_new {
                        for b_idx in 0..chi_old {
                            let cv = c_rem[(a_idx * d_left + el_idx) * chi_old + b_idx];
                            if cv.re == 0.0 && cv.im == 0.0 {
                                continue;
                            }
                            let cw = cv * weight;
                            let row = a_idx * 2 + s;
                            let src_off = (b_idx * 2 + s) * chi_next;
                            let dst_off = row * cols + er_idx * chi_next;
                            for g in 0..chi_next {
                                tmat[dst_off + g] += cw * ai[src_off + g];
                            }
                        }
                    }
                }
            }

            if is_last {
                self.a[i] = tmat;
                self.dl[i] = chi_new;
                self.dr[i] = 1;
            } else {
                let (u, svals, vt) = complex_svd(&tmat, rows, cols);
                let k_full = svals.len();
                let s_max = svals.first().copied().unwrap_or(0.0).max(1e-30);
                let mut chi = 0usize;
                for &sv in svals.iter().take(chi_max.min(k_full)) {
                    if sv > tol * s_max {
                        chi += 1;
                    }
                }
                chi = chi.max(1);
                let mut na = vec![cx(0.0, 0.0); rows * chi];
                for r in 0..rows {
                    for c in 0..chi {
                        na[r * chi + c] = u[r * k_full + c];
                    }
                }
                let mut next_rem = vec![cx(0.0, 0.0); chi * cols];
                for c in 0..chi {
                    let sv = svals[c];
                    for col in 0..cols {
                        next_rem[c * cols + col] = vt[c * cols + col] * sv;
                    }
                }
                self.a[i] = na;
                self.dl[i] = chi_new;
                self.dr[i] = chi;
                chi_new = chi;
                d_left = D_FLUX;
                left_fluxes = (E_MIN..=E_MAX).collect();
                c_rem = next_rem;
            }
        }
        let _ = e_zero;
    }

    /// Entropía de entrelazamiento de von Neumann S_vN(cut) = −Σ λ² ln λ² en el enlace `cut`.
    pub fn entanglement_entropy(&self, cut: usize) -> f64 {
        if self.bond_s.is_empty() {
            return 0.0;
        }
        let s = &self.bond_s[cut.min(self.bond_s.len() - 1)];
        let mut ent = 0.0;
        for &sv in s {
            let p = sv * sv;
            if p > 1e-15 {
                ent -= p * p.ln();
            }
        }
        ent
    }

    /// Valores esperados locales ⟨σ^z_i⟩ en todos los sitios (requiere MPS normalizado).
    pub fn expect_sz(&self) -> Vec<f64> {
        // Entornos derechos R[i] de tamaño dl[i] × dl[i]
        let mut r_env: Vec<Vec<C>> = vec![Vec::new(); self.l + 1];
        r_env[self.l] = vec![cx(1.0, 0.0)];
        for i in (0..self.l).rev() {
            let dl = self.dl[i];
            let dr = self.dr[i];
            let rn = &r_env[i + 1];
            let ai = &self.a[i];
            let mut cur = vec![cx(0.0, 0.0); dl * dl];
            for s in 0..2 {
                // tmp[l, r'] = Σ_r A[l, s, r] * R[r, r']
                let mut tmp = vec![cx(0.0, 0.0); dl * dr];
                for l in 0..dl {
                    for r in 0..dr {
                        let a_val = ai[(l * 2 + s) * dr + r];
                        if a_val.re == 0.0 && a_val.im == 0.0 {
                            continue;
                        }
                        for rp in 0..dr {
                            tmp[l * dr + rp] += a_val * rn[r * dr + rp];
                        }
                    }
                }
                for lp in 0..dl {
                    for l in 0..dl {
                        let mut sum = cx(0.0, 0.0);
                        for rp in 0..dr {
                            sum += ai[(lp * 2 + s) * dr + rp].conj() * tmp[l * dr + rp];
                        }
                        cur[lp * dl + l] += sum;
                    }
                }
            }
            r_env[i] = cur;
        }

        let mut sz_out = vec![0.0f64; self.l];
        let mut l_env = vec![cx(1.0, 0.0)]; // dl[0] × dl[0] = 1 × 1
        let norm2 = r_env[0][0].re.max(1e-30);
        for i in 0..self.l {
            let dl = self.dl[i];
            let dr = self.dr[i];
            let ai = &self.a[i];
            let rn = &r_env[i + 1];
            let mut next_l = vec![cx(0.0, 0.0); dr * dr];
            let mut exp_val = 0.0;
            for s in 0..2 {
                let sign = if s == 0 { 1.0 } else { -1.0 };
                // M[l', r] = Σ_l L[l', l] A[l, s, r]
                let mut m1 = vec![cx(0.0, 0.0); dl * dr];
                for lp in 0..dl {
                    for l in 0..dl {
                        let lv = l_env[lp * dl + l];
                        if lv.re == 0.0 && lv.im == 0.0 {
                            continue;
                        }
                        for r in 0..dr {
                            m1[lp * dr + r] += lv * ai[(l * 2 + s) * dr + r];
                        }
                    }
                }
                // next_l_s[r', r] = Σ_{l'} A*[l', s, r'] M[l', r]
                let mut nl_s = vec![cx(0.0, 0.0); dr * dr];
                for rp in 0..dr {
                    for r in 0..dr {
                        let mut sum = cx(0.0, 0.0);
                        for lp in 0..dl {
                            sum += ai[(lp * 2 + s) * dr + rp].conj() * m1[lp * dr + r];
                        }
                        nl_s[rp * dr + r] = sum;
                        next_l[rp * dr + r] += sum;
                    }
                }
                // Tr(nl_s * rn^T)
                let mut tr = 0.0;
                for rp in 0..dr {
                    for r in 0..dr {
                        tr += (nl_s[rp * dr + r] * rn[r * dr + rp]).re;
                    }
                }
                exp_val += sign * tr;
            }
            sz_out[i] = exp_val / norm2;
            l_env = next_l;
        }
        sz_out
    }

    /// Energía exacta ⟨H_Schwinger⟩ evaluando el término cinético, de masa y eléctrico
    /// (incluyendo el término eléctrico exacto ⟨Σ E_n²⟩ vía contracción directa del flujo).
    pub fn schwinger_energy(&self, w: f64, m: f64, j_coup: f64, q_ext: &[i32]) -> (f64, Vec<f64>) {
        // Calculamos ⟨E_n⟩ y ⟨E_n²⟩ y ⟨σ⁺_n σ⁻_{n+1} + h.c.⟩ con precisión exacta
        let sz = self.expect_sz();
        let mut e_mass = 0.0;
        let mut e_field = vec![0.0f64; self.l - 1];
        let mut q_cum = 0.0;
        for i in 0..self.l {
            let stag = if i % 2 == 0 { 1.0 } else { -1.0 };
            e_mass += 0.5 * m * stag * sz[i];
            q_cum += 0.5 * (sz[i] + stag) + q_ext[i] as f64;
            if i + 1 < self.l {
                e_field[i] = q_cum;
            }
        }
        // Término cinético ⟨σ⁺_i σ⁻_{i+1} + σ⁻_i σ⁺_{i+1}⟩ y ⟨E_n²⟩ usando entornos exactos
        let e_kin = self.expect_xy_sum() * w;
        let e_elec = self.expect_electric_sq_sum(q_ext) * j_coup;
        (e_kin + e_mass + e_elec, e_field)
    }

    /// Suma exacta Σ_{i=0}^{L-2} ⟨σ⁺_i σ⁻_{i+1} + σ⁻_i σ⁺_{i+1}⟩.
    fn expect_xy_sum(&self) -> f64 {
        let mut r_env: Vec<Vec<C>> = vec![Vec::new(); self.l + 1];
        r_env[self.l] = vec![cx(1.0, 0.0)];
        for i in (0..self.l).rev() {
            let dl = self.dl[i];
            let dr = self.dr[i];
            let rn = &r_env[i + 1];
            let ai = &self.a[i];
            let mut cur = vec![cx(0.0, 0.0); dl * dl];
            for s in 0..2 {
                let mut tmp = vec![cx(0.0, 0.0); dl * dr];
                for l in 0..dl {
                    for r in 0..dr {
                        let av = ai[(l * 2 + s) * dr + r];
                        if av.re == 0.0 && av.im == 0.0 {
                            continue;
                        }
                        for rp in 0..dr {
                            tmp[l * dr + rp] += av * rn[r * dr + rp];
                        }
                    }
                }
                for lp in 0..dl {
                    for l in 0..dl {
                        let mut sum = cx(0.0, 0.0);
                        for rp in 0..dr {
                            sum += ai[(lp * 2 + s) * dr + rp].conj() * tmp[l * dr + rp];
                        }
                        cur[lp * dl + l] += sum;
                    }
                }
            }
            r_env[i] = cur;
        }
        let norm2 = r_env[0][0].re.max(1e-30);
        let mut total = 0.0;
        let mut l_env = vec![cx(1.0, 0.0)];
        for i in 0..(self.l - 1) {
            let dl = self.dl[i];
            let dm = self.dr[i];
            let dr = self.dr[i + 1];
            let a0 = &self.a[i];
            let a1 = &self.a[i + 1];
            let rn = &r_env[i + 2];
            // Dos términos: (01 ← 10) y (10 ← 01)
            for &(s0_bra, s1_bra, s0_ket, s1_ket) in &[(0usize, 1usize, 1usize, 0usize), (1, 0, 0, 1)] {
                // T_ket[l, r] = Σ_m A0[l, s0_ket, m] A1[m, s1_ket, r]
                let mut tk = vec![cx(0.0, 0.0); dl * dr];
                let mut tb = vec![cx(0.0, 0.0); dl * dr];
                for l in 0..dl {
                    for m in 0..dm {
                        let v0k = a0[(l * 2 + s0_ket) * dm + m];
                        let v0b = a0[(l * 2 + s0_bra) * dm + m];
                        for r in 0..dr {
                            tk[l * dr + r] += v0k * a1[(m * 2 + s1_ket) * dr + r];
                            tb[l * dr + r] += v0b * a1[(m * 2 + s1_bra) * dr + r];
                        }
                    }
                }
                // Σ_{l',l,r',r} L[l',l] tb*[l',r'] tk[l,r] R[r,r']
                let mut lt = vec![cx(0.0, 0.0); dl * dr];
                for lp in 0..dl {
                    for l in 0..dl {
                        let lv = l_env[lp * dl + l];
                        for r in 0..dr {
                            lt[lp * dr + r] += lv * tk[l * dr + r];
                        }
                    }
                }
                let mut val = cx(0.0, 0.0);
                for lp in 0..dl {
                    for rp in 0..dr {
                        let bc = tb[lp * dr + rp].conj();
                        for r in 0..dr {
                            val += bc * lt[lp * dr + r] * rn[r * dr + rp];
                        }
                    }
                }
                total += val.re;
            }
            // Avanzar l_env al sitio i + 1
            let mut next_l = vec![cx(0.0, 0.0); dm * dm];
            for s in 0..2 {
                let mut m1 = vec![cx(0.0, 0.0); dl * dm];
                for lp in 0..dl {
                    for l in 0..dl {
                        let lv = l_env[lp * dl + l];
                        for r in 0..dm {
                            m1[lp * dm + r] += lv * a0[(l * 2 + s) * dm + r];
                        }
                    }
                }
                for rp in 0..dm {
                    for r in 0..dm {
                        let mut sum = cx(0.0, 0.0);
                        for lp in 0..dl {
                            sum += a0[(lp * 2 + s) * dm + rp].conj() * m1[lp * dm + r];
                        }
                        next_l[rp * dm + r] += sum;
                    }
                }
            }
            l_env = next_l;
        }
        total / norm2
    }

    /// Suma exacta ⟨Σ_{n=0}^{L-2} E_n²⟩ mediante seguimiento de los momentos (1, E, E²) en el barrido izquierdo.
    fn expect_electric_sq_sum(&self, q_ext: &[i32]) -> f64 {
        // Propagamos tres matrices de entorno izquierdo: L0 = ⟨1⟩, L1 = ⟨E_n⟩, L2 = ⟨E_n²⟩
        // y un acumulador L_acc = ⟨Σ_{k≤n} E_k²⟩. Como E_n = E_{n-1} + Q_n:
        //   E_n   = E_{n-1} + Q_n
        //   E_n²  = E_{n-1}² + 2 Q_n E_{n-1} + Q_n²
        let mut l0 = vec![cx(1.0, 0.0)];
        let mut l1 = vec![cx(0.0, 0.0)];
        let mut l2 = vec![cx(0.0, 0.0)];
        let mut lacc = vec![cx(0.0, 0.0)];
        for i in 0..self.l {
            let dl = self.dl[i];
            let dr = self.dr[i];
            let ai = &self.a[i];
            let stag = if i % 2 == 0 { 1.0 } else { -1.0 };
            let qe = q_ext[i] as f64;
            let mut n0 = vec![cx(0.0, 0.0); dr * dr];
            let mut n1 = vec![cx(0.0, 0.0); dr * dr];
            let mut n2 = vec![cx(0.0, 0.0); dr * dr];
            let mut nacc = vec![cx(0.0, 0.0); dr * dr];
            let add_ch = |env: &[C], s: usize, w: f64, dst: &mut [C]| {
                if w == 0.0 {
                    return;
                }
                let mut m1 = vec![cx(0.0, 0.0); dl * dr];
                for lp in 0..dl {
                    for l in 0..dl {
                        let ev = env[lp * dl + l];
                        if ev.re == 0.0 && ev.im == 0.0 {
                            continue;
                        }
                        for r in 0..dr {
                            m1[lp * dr + r] += ev * ai[(l * 2 + s) * dr + r];
                        }
                    }
                }
                for rp in 0..dr {
                    for r in 0..dr {
                        let mut sum = cx(0.0, 0.0);
                        for lp in 0..dl {
                            sum += ai[(lp * 2 + s) * dr + rp].conj() * m1[lp * dr + r];
                        }
                        dst[rp * dr + r] += sum * w;
                    }
                }
            };
            for s in 0..2 {
                let sz = if s == 0 { 1.0 } else { -1.0 };
                let q = 0.5 * (sz + stag) + qe;
                add_ch(&l0, s, 1.0, &mut n0);
                add_ch(&l1, s, 1.0, &mut n1);
                add_ch(&l0, s, q, &mut n1);
                add_ch(&l2, s, 1.0, &mut n2);
                add_ch(&l1, s, 2.0 * q, &mut n2);
                add_ch(&l0, s, q * q, &mut n2);
                add_ch(&lacc, s, 1.0, &mut nacc);
            }
            if i + 1 < self.l {
                for k in 0..(dr * dr) {
                    nacc[k] += n2[k];
                }
            }
            l0 = n0;
            l1 = n1;
            l2 = n2;
            lacc = nacc;
        }
        lacc[0].re / l0[0].re.max(1e-30)
    }
}

/// Puerta de dos sitios exp(−α · [w (σ⁺σ⁻ + σ⁻σ⁺) + Δ σ^z σ^z]).
/// En base {|00⟩, |01⟩, |10⟩, |11⟩}:
/// - |00⟩, |11⟩ tienen fase exp(−α Δ).
/// - Subespacio {|01⟩, |10⟩} tiene H = −Δ I + w σ_x.
pub(crate) fn xxz_gate(alpha: C, w: f64, delta: f64) -> [C; 16] {
    let mut u = [cx(0.0, 0.0); 16];
    let diag = (-alpha * delta).exp();
    let mid_p = (-alpha * (-delta + w)).exp();
    let mid_m = (-alpha * (-delta - w)).exp();
    let ch = (mid_p + mid_m) * 0.5;
    let sh = (mid_p - mid_m) * 0.5;
    u[0] = diag;
    u[15] = diag;
    u[5] = ch;
    u[6] = sh;
    u[9] = sh;
    u[10] = ch;
    u
}

/// Puerta de un sitio exp(−α · (m/2) · (−1)^i · σ^z).
fn mass_gate(alpha: C, m: f64, site: usize) -> [C; 4] {
    let sgn = if site % 2 == 0 { 1.0 } else { -1.0 };
    let e_up = (-alpha * (0.5 * m * sgn)).exp();
    let e_dn = (alpha * (0.5 * m * sgn)).exp();
    [e_up, cx(0.0, 0.0), cx(0.0, 0.0), e_dn]
}

/// Un paso de Strang de 2º orden para el modelo de Schwinger cuántico en MPS:
/// `alpha = i·dt` (tiempo real) o `alpha = dτ` (enfriamiento en tiempo imaginario).
#[allow(clippy::too_many_arguments)]
fn schwinger_step(mps: &mut Mps, alpha: C, w: f64, m: f64, j_coup: f64, q_ext: &[i32], chi_max: usize) {
    let l = mps.l;
    let half = alpha * 0.5;
    let u_even = xxz_gate(half, w, 0.0);
    let u_odd = xxz_gate(alpha, w, 0.0);
    // 1. Mitad de masa + enlaces pares
    for i in 0..l {
        let g = mass_gate(half, m, i);
        mps.apply_one_site(i, &g);
    }
    for i in (0..(l - 1)).step_by(2) {
        mps.apply_two_site(i, &u_even, chi_max, 1e-10);
    }
    // 2. Campo eléctrico cuantizado exacto (MPO D = 5)
    if j_coup.abs() > 0.0 {
        mps.apply_schwinger_electric_mpo(alpha * j_coup, q_ext, chi_max, 1e-10);
    }
    // 3. Enlaces impares (paso completo)
    for i in (1..(l - 1)).step_by(2) {
        mps.apply_two_site(i, &u_odd, chi_max, 1e-10);
    }
    // 4. Enlaces pares + mitad de masa
    for i in (0..(l - 1)).step_by(2) {
        mps.apply_two_site(i, &u_even, chi_max, 1e-10);
    }
    for i in 0..l {
        let g = mass_gate(half, m, i);
        mps.apply_one_site(i, &g);
    }
}

/// Calcula el estado fundamental del modelo de Schwinger cuántico en MPS con cargas externas ±1
/// separadas `dist` enlaces (dist = 0 ⇒ vacío sin cargas). Devuelve `(energía_fundamental, E_centro, S_vN_centro)`.
pub fn schwinger_ground_state(l: usize, w: f64, m: f64, j_coup: f64, dist: usize, chi_max: usize, cool_steps: usize) -> (f64, f64, f64) {
    let mut q_ext = vec![0i32; l];
    if dist > 0 && dist < l {
        let xp = (l - dist) / 2;
        let xm = xp + dist;
        q_ext[xp] = 1;
        q_ext[xm] = -1;
    }
    let mut mps = Mps::neel_vacuum(l);
    // Enfriamiento en tiempo imaginario adaptativo (dτ grande → pequeño)
    for step in 0..cool_steps {
        let dtau = if step < cool_steps / 2 { 0.12 } else { 0.04 };
        schwinger_step(&mut mps, cx(dtau, 0.0), w, m, j_coup, &q_ext, chi_max);
        mps.canonicalize_and_normalize();
    }
    let (energy, ef) = mps.schwinger_energy(w, m, j_coup, &q_ext);
    let mid = (l / 2).saturating_sub(1);
    let s_mid = mps.entanglement_entropy(mid);
    (energy, ef[mid], s_mid)
}

/// Evolución en tiempo real de la rotura cuántica de la cuerda en el modelo de Schwinger (MPS):
/// parte del estado de cuerda sin romper (vacío de Néel con cargas externas ±1 a distancia `dist`)
/// y registra en cada paso `[t, E_centro(t), S_vN_centro(t), E_total(t)]`.
#[allow(clippy::too_many_arguments)]
pub fn schwinger_string_breaking_mps(l: usize, w: f64, m: f64, j_coup: f64, dist: usize, chi_max: usize, dt: f64, steps: usize) -> Vec<f64> {
    let mut q_ext = vec![0i32; l];
    let d = dist.clamp(2, l - 2);
    let xp = (l - d) / 2;
    let xm = xp + d;
    q_ext[xp] = 1;
    q_ext[xm] = -1;
    let mut mps = Mps::neel_vacuum(l);
    let mid = l / 2 - 1;
    let (e0, ef0) = mps.schwinger_energy(w, m, j_coup, &q_ext);
    let mut e_min = ef0[mid];
    let mut s_max = 0.0f64;
    let mut series = Vec::with_capacity(4 * (steps + 1));
    series.extend_from_slice(&[0.0, ef0[mid], 0.0, e0]);
    for step in 1..=steps {
        schwinger_step(&mut mps, cx(0.0, dt), w, m, j_coup, &q_ext, chi_max);
        mps.canonicalize_and_normalize();
        let (en, ef) = mps.schwinger_energy(w, m, j_coup, &q_ext);
        let s_vn = mps.entanglement_entropy(mid);
        e_min = e_min.min(ef[mid]);
        s_max = s_max.max(s_vn);
        series.extend_from_slice(&[step as f64 * dt, ef[mid], s_vn, en]);
    }
    let e_final = series[series.len() - 1];
    let ef_final = series[series.len() - 3];
    let s_final = series[series.len() - 2];
    let n_pts = (series.len() / 4) as f64;
    // Cabecera: [E_centro_0, E_centro_min, E_centro_fin, S_vN_max, S_vN_fin, E_tot_0, E_tot_fin, n_pts, ...series]
    let mut out = vec![ef0[mid], e_min, ef_final, s_max, s_final, e0, e_final, n_pts];
    out.extend_from_slice(&series);
    out
}

/// Renormalización de la velocidad del frente de luz en una cadena cuántica de muchos cuerpos
/// interactuante (fermiones sin espín 1+1D / modelo XXZ con interacción vecina `delta` en MPS):
/// H = Σ [w (σ⁺_i σ⁻_{i+1} + h.c.) + Δ σ^z_i σ^z_{i+1}].
/// En el límite libre (Δ = 0) la velocidad de Fermi de red es v_F = 2w; con interacción repulsiva
/// Δ > 0, la solución exacta de Bethe da v_ren(Δ) = v_F · (π/2) · sin(γ)/γ con cos(γ) = Δ/(2w),
/// que supera v_F (renormalización de la velocidad límite por interacciones de muchos cuerpos).
/// Mide directamente el frente de luz del conmutador/perturbación local en el vacío interactuante MPS.
pub fn mps_interacting_lightcone(l: usize, w: f64, delta: f64, chi_max: usize, dt: f64, steps: usize) -> Vec<f64> {
    // 1. Preparamos el estado fundamental interactuante (vacío a llenado mitad, Sz_tot = 0)
    let mut gs = Mps::neel_vacuum(l);
    for cool in 0..50 {
        let dtau = if cool < 25 { 0.08 } else { 0.03 };
        let u_even_im = xxz_gate(cx(0.5 * dtau, 0.0), w, delta);
        let u_odd_im = xxz_gate(cx(dtau, 0.0), w, delta);
        for i in (0..(l - 1)).step_by(2) {
            gs.apply_two_site(i, &u_even_im, chi_max, 1e-10);
        }
        for i in (1..(l - 1)).step_by(2) {
            gs.apply_two_site(i, &u_odd_im, chi_max, 1e-10);
        }
        for i in (0..(l - 1)).step_by(2) {
            gs.apply_two_site(i, &u_even_im, chi_max, 1e-10);
        }
        gs.canonicalize_and_normalize();
    }
    let sz_bg = gs.expect_sz();
    let s_gs = gs.entanglement_entropy(l / 2 - 1);

    // 2. Inyectamos una excitación fermiónica/espín localizada c†_centro = σ⁺_centro sobre el vacío
    //    interactuante y renormalizamos: conserva exactamente Σ Δ⟨σ^z_i⟩ = +1 (frente de luz limpio).
    let center = l / 2;
    let create_op = [cx(0.0, 0.0), cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0)];
    gs.apply_one_site(center, &create_op);
    gs.canonicalize_and_normalize();

    // 3. Evolucionamos en tiempo real y medimos la expansión balística de Δn_i(t) = ½(⟨σ^z_i⟩ − ⟨σ^z_i⟩_vac)
    let u_even = xxz_gate(cx(0.0, 0.5 * dt), w, delta);
    let u_odd = xxz_gate(cx(0.0, dt), w, delta);
    let mut ts = Vec::with_capacity(steps);
    let mut radii = Vec::with_capacity(steps);
    let mut s_fin = s_gs;
    for step in 1..=steps {
        for i in (0..(l - 1)).step_by(2) {
            gs.apply_two_site(i, &u_even, chi_max, 1e-10);
        }
        for i in (1..(l - 1)).step_by(2) {
            gs.apply_two_site(i, &u_odd, chi_max, 1e-10);
        }
        for i in (0..(l - 1)).step_by(2) {
            gs.apply_two_site(i, &u_even, chi_max, 1e-10);
        }
        gs.canonicalize_and_normalize();
        let sz = gs.expect_sz();
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..l {
            let dn = (0.5 * (sz[i] - sz_bg[i])).max(0.0);
            let dist = (i as f64 - center as f64).abs();
            num += dn * dist * dist;
            den += dn;
        }
        // En 1D para un paquete en el mar de Fermi libre, r_rms(t) = v_F·t/√2 ⇒ √2·r_rms mide v_F
        let r_eff = (2.0 * num / den.max(1e-30)).sqrt();
        ts.push(step as f64 * dt);
        radii.push(r_eff);
        if step == steps {
            s_fin = gs.entanglement_entropy(l / 2 - 1);
        }
    }
    // Ajuste lineal de √2·r_rms(t) en la ventana balística (evitando el transitorio inicial y el borde)
    let i0 = (steps / 5).max(1);
    let i1 = (steps * 3 / 4).max(i0 + 3).min(steps);
    let (_, v_front) = crate::util::fit_linear(&ts[i0..i1], &radii[i0..i1]);
    // Predicción exacta de Bethe para la velocidad de Luttinger (|Δ_XXZ| ≤ 1):
    // H = w (σ⁺σ⁻ + h.c.) + Δ σ^z σ^z = (w/2)(σ^x σ^x + σ^y σ^y) + Δ σ^z σ^z ⇒ Δ_XXZ = 2Δ/w, J = 2w
    // v_Bethe = J · (π/2) · sin(γ) / γ = w · π · sin(γ) / γ con cos(γ) = Δ_XXZ
    // (en Δ = 0 ⇒ γ = π/2 ⇒ v = 2w; en Δ = w/2 ⇒ γ → 0 ⇒ v = πw, des Cloizeaux–Pearson).
    let d_xxz = (2.0 * delta / w.max(1e-12)).clamp(-0.9999, 0.9999);
    let gam = d_xxz.acos();
    let v_bethe = if gam < 1e-6 { w * std::f64::consts::PI } else { w * std::f64::consts::PI * gam.sin() / gam };
    // [v_frente_medida, v_bethe, S_vacío, S_final, r_final]
    vec![v_front, v_bethe, s_gs, s_fin, *radii.last().unwrap_or(&0.0)]
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 15 (QFT de muchos cuerpos en 1+1D con MPS): potencial estático quark-antiquark V(d)
    /// del modelo de Schwinger cuántico (QED 1+1D con campo eléctrico cuantizado y ley de Gauss exacta).
    /// Devuelve [E0(d=0), E0(d=2), E0(d=4), E0(d=6), S_vN(d=0), S_vN(d=4), E_centro(d=4)] en cadena de L sitios.
    pub fn mps_schwinger_confinement(&self, l: u32, w: f32, m: f32, j_coup: f32, chi_max: u32) -> Vec<f64> {
        let n = (l as usize).clamp(8, 20) & !1;
        let chi = (chi_max as usize).clamp(4, 24);
        let wf = clamp_finite(w as f64, 0.05, 5.0, 1.0);
        let mf = clamp_finite(m as f64, -5.0, 5.0, 0.2);
        let jf = clamp_finite(j_coup as f64, 0.0, 10.0, 0.5);
        let (e0, _, s0) = schwinger_ground_state(n, wf, mf, jf, 0, chi, 28);
        let (e2, _, _) = schwinger_ground_state(n, wf, mf, jf, 2, chi, 28);
        let (e4, ef4, s4) = schwinger_ground_state(n, wf, mf, jf, 4, chi, 28);
        let d6 = 6.min(n - 2);
        let (e6, _, _) = schwinger_ground_state(n, wf, mf, jf, d6, chi, 28);
        vec![e0, e2 - e0, e4 - e0, e6 - e0, s0, s4, ef4]
    }

    /// Fase 15 (QFT de muchos cuerpos en 1+1D con MPS): rotura cuántica de la cuerda en tiempo real
    /// del modelo de Schwinger con cargas ±1 a distancia `dist`.
    /// Devuelve puntero a [E_c0, E_c_min, E_c_fin, S_vN_max, S_vN_fin, E_tot_0, E_tot_fin, n_pts, t, E_c, S_vN, E_tot, …].
    #[allow(clippy::too_many_arguments)]
    pub fn mps_schwinger_string_breaking(&mut self, l: u32, w: f32, m: f32, j_coup: f32, dist: u32, chi_max: u32, dt: f32, steps: u32) -> *const f64 {
        let n = (l as usize).clamp(8, 20) & !1;
        let chi = (chi_max as usize).clamp(4, 24);
        let st = (steps as usize).clamp(5, 200);
        let wf = clamp_finite(w as f64, 0.05, 5.0, 1.0);
        let mf = clamp_finite(m as f64, -5.0, 5.0, 0.2);
        let jf = clamp_finite(j_coup as f64, 0.0, 10.0, 0.5);
        let dtf = clamp_finite(dt as f64, 0.01, 0.2, 0.05);
        self.oq.out = schwinger_string_breaking_mps(n, wf, mf, jf, dist as usize, chi, dtf, st);
        self.oq.out.as_ptr()
    }

    /// Fase 15 (MPS interactuante 1+1D): mide la velocidad del cono de luz de una excitación local
    /// sobre el vacío interactuante del modelo XXZ/Thirring con acoplamiento `delta`.
    /// Devuelve [v_frente, v_bethe, S_vacío, S_final, r_final].
    pub fn mps_interacting_velocity(&self, l: u32, w: f32, delta: f32, chi_max: u32, dt: f32, steps: u32) -> Vec<f64> {
        let n = (l as usize).clamp(10, 24) & !1;
        let chi = (chi_max as usize).clamp(6, 24);
        let st = (steps as usize).clamp(10, 80);
        let wf = clamp_finite(w as f64, 0.05, 5.0, 1.0);
        let df = clamp_finite(delta as f64, -5.0, 5.0, 0.0);
        let dtf = clamp_finite(dt as f64, 0.02, 0.2, 0.05);
        mps_interacting_lightcone(n, wf, df, chi, dtf, st)
    }
}
