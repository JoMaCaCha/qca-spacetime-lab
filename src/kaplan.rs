//! Fermiones quirales en 3+1D desde una pared en 4+1D (Kaplan, §2.20), a momento físico fijo: para
//! cada k = (k₁, k₂, k₃) el problema es una caminata 1D en la quinta dirección s, con cinco matrices
//! 4×4 que anticonmutan (α_a = τ_x⊗σ_a, α₄ = τ_y⊗1, β = τ_z⊗1). Paso: ψ ← M(m(s))·T₄·T₃·T₂·T₁·ψ, con
//! T_a = exp(−iα_a·k_a) fijas y T₄ el desplazamiento en s en la base propia de α₄. Las paredes en
//! L_s/4 y 3L_s/4 atrapan cada una un Weyl de quiralidad (γ⁵ = τ_x⊗1) opuesta: el doblador que
//! exige Nielsen–Ninomiya está en la otra pared. Es una partícula clásica, no una simulación 4D, y
//! la anomalía (Callan y Harvey) es cuántica.

use crate::linalg::eig;
use crate::rng::SplitMix64;
use crate::spinor::{cx, dirac_matrix, m4_apply, m4_identity, m4_mul, M4};
use crate::sun::C;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct Kaplan {
    pub ls: usize,
    pub m0: f64,
    pub k: [f64; 3],
    pub fixed: bool,
    pub psi: Vec<C>,
    pub tmp: Vec<C>,
    pub t: u32,
    /// T₃·T₂·T₁ (pasos pares) y T₁·T₂·T₃ (impares, orden alternado).
    pub tk: [M4; 2],
    /// Base propia de α₄ (columnas: autovalor +1, +1, −1, −1).
    pub v4: M4,
    pub autocorr: Vec<C>,
    pub psi0: Vec<C>,
}

impl Kaplan {
    pub fn new() -> Self {
        Kaplan { ls: 64, m0: 0.3, k: [0.1, 0.0, 0.0], fixed: true, psi: Vec::new(), tmp: Vec::new(), t: 0, tk: [m4_identity(), m4_identity()], v4: m4_identity(), autocorr: Vec::new(), psi0: Vec::new() }
    }

    fn rebuild(&mut self) {
        let t = |a: usize, k: f64| -> M4 {
            let al = dirac_matrix(0, a);
            let mut m = m4_identity();
            for i in 0..4 {
                for j in 0..4 {
                    m[i][j] = m[i][j] * k.cos() - cx(0.0, k.sin()) * al[i][j];
                }
            }
            m
        };
        let (t1, t2, t3) = (t(0, self.k[0]), t(1, self.k[1]), t(2, self.k[2]));
        self.tk[0] = m4_mul(&t3, &m4_mul(&t2, &t1));
        self.tk[1] = m4_mul(&t1, &m4_mul(&t2, &t3));
        // α₄ = τ_y ⊗ 1: autovectores |τ_y ±⟩ ⊗ |σ⟩
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let ty = [[cx(s, 0.0), cx(s, 0.0)], [cx(0.0, s), cx(0.0, -s)]];
        let mut v = [[cx(0.0, 0.0); 4]; 4];
        for (col, (tsign, sg)) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            for tt in 0..2 {
                v[2 * tt + sg][col] = ty[tt][tsign];
            }
        }
        self.v4 = v;
        let n = 4 * self.ls;
        if self.psi.len() != n {
            self.psi = vec![cx(0.0, 0.0); n];
            self.tmp = vec![cx(0.0, 0.0); n];
        }
    }

    pub fn mass_at(&self, s: usize) -> f64 {
        if s >= self.ls / 4 && s < 3 * self.ls / 4 {
            self.m0
        } else {
            -self.m0
        }
    }

    fn dag(m: &M4) -> M4 {
        let mut r = [[cx(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                r[i][j] = m[j][i].conj();
            }
        }
        r
    }

    /// Desplazamiento en s en la base propia de α₄ (las dos componentes +1 avanzan, las −1 retroceden).
    fn shift4(&mut self) {
        let ls = self.ls;
        let vd = Self::dag(&self.v4);
        for s in 0..ls {
            let v = [self.psi[4 * s], self.psi[4 * s + 1], self.psi[4 * s + 2], self.psi[4 * s + 3]];
            let r = m4_apply(&vd, &v);
            for c in 0..4 {
                self.tmp[4 * s + c] = r[c];
            }
        }
        for s in 0..ls {
            let sm = (s + ls - 1) % ls;
            let sp = (s + 1) % ls;
            let v = [self.tmp[4 * sm], self.tmp[4 * sm + 1], self.tmp[4 * sp + 2], self.tmp[4 * sp + 3]];
            let r = m4_apply(&self.v4, &v);
            for c in 0..4 {
                self.psi[4 * s + c] = r[c];
            }
        }
    }

    fn apply_local(&mut self, m: &M4) {
        for s in 0..self.ls {
            let v = [self.psi[4 * s], self.psi[4 * s + 1], self.psi[4 * s + 2], self.psi[4 * s + 3]];
            let r = m4_apply(m, &v);
            for c in 0..4 {
                self.psi[4 * s + c] = r[c];
            }
        }
    }

    fn apply_mass(&mut self) {
        for s in 0..self.ls {
            let m = self.mass_at(s);
            let (e1, e2) = (Complex::from_polar(1.0, -m), Complex::from_polar(1.0, m));
            self.psi[4 * s] *= e1;
            self.psi[4 * s + 1] *= e1;
            self.psi[4 * s + 2] *= e2;
            self.psi[4 * s + 3] *= e2;
        }
    }

    pub fn step(&mut self) {
        let odd = !self.fixed && (self.t & 1) == 1;
        if odd {
            // M·T₁·T₂·T₃·T₄
            self.shift4();
            let m = self.tk[1];
            self.apply_local(&m);
        } else {
            let m = self.tk[0];
            self.apply_local(&m);
            self.shift4();
        }
        self.apply_mass();
        self.t = self.t.wrapping_add(1);
        if !self.psi0.is_empty() && self.autocorr.len() < crate::spectral::AUTOCORR_MAX {
            let c: C = self.psi0.iter().zip(self.psi.iter()).map(|(a, b)| a.conj() * b).sum();
            self.autocorr.push(c);
        }
    }

    /// Matriz 4L_s × 4L_s del paso (orden fijo) o del operador de dos pasos (alternado).
    pub fn step_matrix(&mut self) -> Vec<C> {
        let n = 4 * self.ls;
        let saved = (self.psi.clone(), self.t, std::mem::take(&mut self.psi0));
        let mut u = vec![cx(0.0, 0.0); n * n];
        let steps = if self.fixed { 1 } else { 2 };
        for col in 0..n {
            for v in self.psi.iter_mut() {
                *v = cx(0.0, 0.0);
            }
            self.psi[col] = cx(1.0, 0.0);
            self.t = 0;
            for _ in 0..steps {
                self.step();
            }
            for row in 0..n {
                u[row * n + col] = self.psi[row];
            }
        }
        self.psi = saved.0;
        self.t = saved.1;
        self.psi0 = saved.2;
        u
    }
}

/// ⟨γ⁵⟩, ⟨Σ·k̂⟩ y la fracción de la norma a menos de 6 celdas de cada pared de un vector de la tira.
fn mode_stats(v: &[C], ls: usize, k: [f64; 3]) -> (f64, f64, f64, f64) {
    let g5 = dirac_matrix(0, 3);
    let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt().max(1e-12);
    let mut hel = [[cx(0.0, 0.0); 4]; 4];
    for a in 0..3 {
        let sa = dirac_matrix(3, a);
        for i in 0..4 {
            for j in 0..4 {
                hel[i][j] += sa[i][j] * (k[a] / kn);
            }
        }
    }
    let (mut g, mut h, mut fa, mut fb, mut tot) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (wa, wb) = (ls / 4, 3 * ls / 4);
    for s in 0..ls {
        let x = [v[4 * s], v[4 * s + 1], v[4 * s + 2], v[4 * s + 3]];
        let n: f64 = x.iter().map(|c| c.norm_sqr()).sum();
        let gx = m4_apply(&g5, &x);
        let hx = m4_apply(&hel, &x);
        g += (0..4).map(|c| (x[c].conj() * gx[c]).re).sum::<f64>();
        h += (0..4).map(|c| (x[c].conj() * hx[c]).re).sum::<f64>();
        tot += n;
        let da = (s as i64 - wa as i64).abs().min(ls as i64 - (s as i64 - wa as i64).abs());
        let db = (s as i64 - wb as i64).abs().min(ls as i64 - (s as i64 - wb as i64).abs());
        if da <= 6 {
            fa += n;
        }
        if db <= 6 {
            fb += n;
        }
    }
    (g / tot, h / tot, fa / tot, fb / tot)
}

impl QuantumSpacetimeUniverse {
    /// Autoestados de la tira con |ε| < emax: [ε, pared (0, 1 o −1), ⟨γ⁵⟩, helicidad, fracción en
    /// la pared]. Los estados degenerados de las dos paredes se separan diagonalizando el
    /// proyector de la pared A dentro de cada subespacio degenerado.
    pub(crate) fn kaplan_modes(&mut self, emax: f64) -> Vec<[f64; 5]> {
        let ls = self.kap.ls;
        let n = 4 * ls;
        let k = self.kap.k;
        let per = if self.kap.fixed { 1.0 } else { 2.0 };
        let e = eig(self.kap.step_matrix(), n);
        let mut idx: Vec<usize> = (0..n).filter(|&i| (-e.values[i].arg() / per).abs() < emax).collect();
        idx.sort_by(|&a, &b| (-e.values[a].arg()).partial_cmp(&(-e.values[b].arg())).unwrap());
        let mut out = Vec::new();
        let mut i = 0;
        while i < idx.len() {
            let mut j = i + 1;
            let e0 = -e.values[idx[i]].arg();
            while j < idx.len() && (-e.values[idx[j]].arg() - e0).abs() < 1e-7 {
                j += 1;
            }
            // subespacio degenerado idx[i..j]: ortonormaliza y diagonaliza P_A
            let mut basis: Vec<Vec<C>> = Vec::new();
            for &q in &idx[i..j] {
                let mut v = e.vectors[q].clone();
                for b in &basis {
                    let ov: C = b.iter().zip(v.iter()).map(|(x, y)| x.conj() * y).sum();
                    for (vv, bb) in v.iter_mut().zip(b.iter()) {
                        *vv -= ov * bb;
                    }
                }
                let nn: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if nn > 1e-8 {
                    basis.push(v.iter().map(|c| c / nn).collect());
                }
            }
            let d = basis.len();
            let mut pa = vec![cx(0.0, 0.0); d * d];
            let wa = ls / 4;
            for a in 0..d {
                for b in 0..d {
                    let mut s = cx(0.0, 0.0);
                    for site in 0..ls {
                        let dd = (site as i64 - wa as i64).abs().min(ls as i64 - (site as i64 - wa as i64).abs());
                        if dd <= ls as i64 / 4 {
                            for c in 0..4 {
                                s += basis[a][4 * site + c].conj() * basis[b][4 * site + c];
                            }
                        }
                    }
                    pa[a * d + b] = s;
                }
            }
            let sub = eig(pa, d);
            for vec in sub.vectors.iter() {
                let mut v = vec![cx(0.0, 0.0); n];
                for a in 0..d {
                    for (vv, bb) in v.iter_mut().zip(basis[a].iter()) {
                        *vv += vec[a] * bb;
                    }
                }
                let (g, h, fa, fb) = mode_stats(&v, ls, k);
                let wall = if fa > 0.5 { 0.0 } else if fb > 0.5 { 1.0 } else { -1.0 };
                out.push([e0 / per, wall, g, h, fa.max(fb)]);
            }
            i = j;
        }
        out
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.20 (escenario 5): tira de 4+1D a momento físico fijo k, con paredes en L_s/4 y 3L_s/4.
    pub fn set_scenario_kaplan(&mut self, length: u32, m0: f32, k1: f32, k2: f32, k3: f32, fixed_order: bool) {
        if self.scenario != 5 {
            self.set_scenario(5);
        }
        self.kap.ls = (length as usize).clamp(16, 512) & !3;
        self.kap.m0 = m0.clamp(0.0, 1.5) as f64;
        self.kap.k = [k1 as f64, k2 as f64, k3 as f64];
        self.kap.fixed = fixed_order;
        self.kap.psi.clear();
        self.kap.rebuild();
        self.kap.t = 0;
        self.kap.psi0.clear();
        self.kap.autocorr.clear();
    }

    /// Paquete gaussiano en s sobre la pared 0 (L_s/4) o 1 (3L_s/4), con el espinor de su Weyl de
    /// energía positiva en el continuo: γ⁵ = ∓1 y helicidad ∓1 (levógiro en la pared 0, dextrógiro
    /// en la 1).
    pub fn inject_kaplan_packet(&mut self, wall: u32, sigma: f32) {
        if self.scenario != 5 {
            return;
        }
        let ls = self.kap.ls;
        let wall = wall.min(1) as f64;
        let center = if wall == 0.0 { ls / 4 } else { 3 * ls / 4 };
        let sg = (sigma as f64).max(0.5);
        // el espinor de la pared: autovector de iα₄β (−γ⁵) con el signo de esa pared, proyectado
        // sobre la helicidad positiva o negativa según la quiralidad (Weyl ±σ·k)
        let g5 = dirac_matrix(0, 3);
        let sign = if wall == 0.0 { -1.0 } else { 1.0 };
        let k = self.kap.k;
        let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt().max(1e-12);
        let mut chi = [cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0)];
        // (1 + sign·γ⁵)/2 y (1 + h·Σ·k̂)/2 con h = sign (energía positiva)
        let mut p = m4_identity();
        for i in 0..4 {
            for j in 0..4 {
                p[i][j] = (p[i][j] + g5[i][j] * sign) * 0.5;
            }
        }
        let mut hel = m4_identity();
        for a in 0..3 {
            let sa = dirac_matrix(3, a);
            for i in 0..4 {
                for j in 0..4 {
                    hel[i][j] += sa[i][j] * (sign * k[a] / kn);
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                hel[i][j] *= 0.5;
            }
        }
        let pp = m4_mul(&hel, &p);
        for start in [[cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(0.0, 0.0), cx(1.0, 0.0), cx(0.0, 0.0)]] {
            let r = m4_apply(&pp, &start);
            if r.iter().map(|c| c.norm_sqr()).sum::<f64>() > 0.1 {
                chi = r;
                break;
            }
        }
        let nn: f64 = chi.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for s in 0..ls {
            let d = crate::util::min_image(s as f64 - center as f64, ls as f64);
            let e = (-d * d / (2.0 * sg * sg)).exp();
            for c in 0..4 {
                self.kap.psi[4 * s + c] += chi[c] * (e / nn);
            }
        }
        let tot: f64 = self.kap.psi.iter().map(|c| c.norm_sqr()).sum();
        for v in self.kap.psi.iter_mut() {
            *v /= tot.sqrt();
        }
    }

    /// ⟨γ⁵⟩ de la parte de la norma a menos de 6 celdas de la pared 0 o 1.
    pub fn kaplan_chirality(&self, wall: u32) -> f64 {
        let ls = self.kap.ls;
        if self.kap.psi.len() != 4 * ls {
            return 0.0;
        }
        let w0 = if wall == 0 { ls / 4 } else { 3 * ls / 4 };
        let g5 = dirac_matrix(0, 3);
        let (mut g, mut tot) = (0.0, 0.0);
        for s in 0..ls {
            let d = (s as i64 - w0 as i64).abs().min(ls as i64 - (s as i64 - w0 as i64).abs());
            if d > 6 {
                continue;
            }
            let x = [self.kap.psi[4 * s], self.kap.psi[4 * s + 1], self.kap.psi[4 * s + 2], self.kap.psi[4 * s + 3]];
            let gx = m4_apply(&g5, &x);
            g += (0..4).map(|c| (x[c].conj() * gx[c]).re).sum::<f64>();
            tot += x.iter().map(|c| c.norm_sqr()).sum::<f64>();
        }
        if tot > 0.0 {
            g / tot
        } else {
            0.0
        }
    }

    /// Autoestados de la tira con |ε| < emax (§3.18): n × 5 valores [ε, pared, ⟨γ⁵⟩, helicidad,
    /// fracción a menos de 6 celdas de su pared].
    pub fn kaplan_spectrum(&mut self, emax: f32) -> Vec<f64> {
        if self.scenario != 5 {
            return Vec::new();
        }
        self.kaplan_modes(emax as f64).into_iter().flatten().collect()
    }

    /// Estado aleatorio en la tira y autocorrelación c(t) = ⟨ψ₀|ψ(t)⟩ en los pasos siguientes.
    pub fn kaplan_start_autocorrelation(&mut self, seed: u32) {
        let mut rng = SplitMix64::new(seed as u64 ^ 0x4B41);
        for v in self.kap.psi.iter_mut() {
            *v = cx(rng.gauss(), rng.gauss());
        }
        let tot: f64 = self.kap.psi.iter().map(|c| c.norm_sqr()).sum();
        for v in self.kap.psi.iter_mut() {
            *v /= tot.sqrt();
        }
        self.kap.psi0 = self.kap.psi.clone();
        self.kap.autocorr.clear();
        self.kap.autocorr.push(cx(1.0, 0.0));
    }

    /// S(ω) de la autocorrelación de la tira (Hann y relleno ×4): pares [ω, S] con ω ∈ [−π, π).
    pub fn kaplan_autocorrelation_spectrum(&mut self) -> Vec<f32> {
        let nt = self.kap.autocorr.len();
        if nt < 4 {
            return Vec::new();
        }
        let l = 4 * nt;
        let mut buf = vec![cx(0.0, 0.0); l];
        for t in 0..nt {
            let wv = 0.5 * (1.0 - (2.0 * PI * t as f64 / (nt - 1) as f64).cos());
            buf[t] = self.kap.autocorr[t] * wv;
        }
        let mut planner = rustfft::FftPlanner::<f64>::new();
        planner.plan_fft_inverse(l).process(&mut buf);
        let mut out = vec![0.0f32; 2 * l];
        for j in 0..l {
            let src = (j + l / 2) % l;
            let mut om = 2.0 * PI * src as f64 / l as f64;
            if om >= PI {
                om -= 2.0 * PI;
            }
            out[2 * j] = om as f32;
            out[2 * j + 1] = buf[src].norm_sqr() as f32;
        }
        out
    }

    /// Fracción de la norma a menos de 6 celdas de la pared 0 o 1.
    pub fn kaplan_wall_fraction(&self, wall: u32) -> f64 {
        let ls = self.kap.ls;
        if self.kap.psi.len() != 4 * ls {
            return 0.0;
        }
        let w0 = if wall == 0 { ls / 4 } else { 3 * ls / 4 };
        let (mut a, mut t) = (0.0, 0.0);
        for s in 0..ls {
            let e: f64 = (0..4).map(|c| self.kap.psi[4 * s + c].norm_sqr()).sum();
            let d = (s as i64 - w0 as i64).abs().min(ls as i64 - (s as i64 - w0 as i64).abs());
            if d <= 6 {
                a += e;
            }
            t += e;
        }
        if t > 0.0 {
            a / t
        } else {
            0.0
        }
    }

    /// |ψ(s)|² de la tira (L_s valores) para dibujarla.
    pub fn kaplan_density(&self) -> Vec<f64> {
        (0..self.kap.ls).map(|s| (0..4).map(|c| self.kap.psi.get(4 * s + c).map_or(0.0, |v| v.norm_sqr())).sum()).collect()
    }

    pub fn kaplan_length(&self) -> u32 {
        self.kap.ls as u32
    }
}
