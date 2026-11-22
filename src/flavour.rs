//! Generaciones y mezcla en el escenario Dirac (§2.22): un índice de sabor y la matriz de masas
//! M = U·diag(m_i)·U† en el paso de masa, exp(−i·M ⊗ σ_y), sobre la caminata de Dirac 2D de §2.8
//! (en una tira de altura 1 es la caminata 1D). Un paquete creado en un sabor (el estado que produce
//! el W) es una superposición de estados de masa con frecuencias distintas: su sabor oscila. Es una
//! interferencia de una sola partícula, clásica. Con tres sabores y una fase δ, P(α → β) ≠ P(β → α):
//! viola T, con el tamaño que fija el invariante de Jarlskog.

use crate::dirac2d::{c, dirac2d_matrix, dirac2d_omega, su2_band, C64};
use crate::fftn::wavenumber;
use crate::util::min_image;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::FRAC_1_SQRT_2;
use wasm_bindgen::prelude::*;

pub struct Flavour {
    pub active: bool,
    pub nf: usize,
    pub masses: Vec<f64>,
    /// U (nf × nf, fila mayor): columnas = estados de masa en la base de sabor.
    pub u: Vec<C64>,
    /// ψ por celda: [sabor][componente ↑/↓].
    pub psi: Vec<C64>,
    pub tmp: Vec<C64>,
    pub norms: Vec<f64>,
}

impl Flavour {
    pub fn new() -> Self {
        Flavour { active: false, nf: 0, masses: Vec::new(), u: Vec::new(), psi: Vec::new(), tmp: Vec::new(), norms: vec![0.0; 3] }
    }
}

/// Matriz de mezcla con la parametrización estándar: 2 sabores, [[c, s], [−s, c]]; 3 sabores,
/// U = R₂₃·U₁₃(δ)·R₁₂ con los ángulos (θ₁₂, θ₂₃, θ₁₃) en ese orden.
pub fn mixing_matrix(angles: &[f64], delta: f64, nf: usize) -> Vec<C64> {
    if nf == 2 {
        let (s, co) = angles.first().copied().unwrap_or(0.0).sin_cos();
        return vec![c(co, 0.0), c(s, 0.0), c(-s, 0.0), c(co, 0.0)];
    }
    let g = |k: usize| angles.get(k).copied().unwrap_or(0.0);
    let (s12, c12) = g(0).sin_cos();
    let (s23, c23) = g(1).sin_cos();
    let (s13, c13) = g(2).sin_cos();
    let e = C64::from_polar(1.0, delta);
    let ei = C64::from_polar(1.0, -delta);
    vec![
        c(c12 * c13, 0.0),
        c(s12 * c13, 0.0),
        ei * s13,
        c(-s12 * c23, 0.0) - e * (c12 * s23 * s13),
        c(c12 * c23, 0.0) - e * (s12 * s23 * s13),
        c(s23 * c13, 0.0),
        c(s12 * s23, 0.0) - e * (c12 * c23 * s13),
        c(-c12 * s23, 0.0) - e * (s12 * c23 * s13),
        c(c23 * c13, 0.0),
    ]
}

impl QuantumSpacetimeUniverse {
    /// Un paso: T_x (↑ de x − 1, ↓ de x + 1), H, el salto en y, H, y el paso de masa con mezcla
    /// (U†, la rotación exp(−i·m_k·σ_y) de cada estado de masa, U).
    pub(crate) fn flavour_step(&mut self) {
        let (w, h) = (self.w, self.h);
        let nf = self.fl.nf;
        let n = w * h;
        let s = FRAC_1_SQRT_2;
        let st = 2 * nf;
        let mut psi = std::mem::take(&mut self.fl.psi);
        let mut tmp = std::mem::take(&mut self.fl.tmp);
        // T_x y H
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let xm = y * w + if x == 0 { w - 1 } else { x - 1 };
                let xp = y * w + if x + 1 == w { 0 } else { x + 1 };
                for f in 0..nf {
                    let (u, d) = (psi[xm * st + 2 * f], psi[xp * st + 2 * f + 1]);
                    tmp[i * st + 2 * f] = (u + d) * s;
                    tmp[i * st + 2 * f + 1] = (u - d) * s;
                }
            }
        }
        // salto en y y H
        for y in 0..h {
            let ym = if y == 0 { h - 1 } else { y - 1 };
            let yp = if y + 1 == h { 0 } else { y + 1 };
            for x in 0..w {
                let i = y * w + x;
                for f in 0..nf {
                    let (a, b) = (tmp[(ym * w + x) * st + 2 * f], tmp[(yp * w + x) * st + 2 * f + 1]);
                    psi[i * st + 2 * f] = (a + b) * s;
                    psi[i * st + 2 * f + 1] = (a - b) * s;
                }
            }
        }
        // paso de masa: ψ_masa = U†ψ, exp(−i m σ_y) = [[cos m, −sin m], [sin m, cos m]], ψ = Uψ_masa
        let u = &self.fl.u;
        let cs: Vec<(f64, f64)> = self.fl.masses.iter().map(|m| (m.cos(), m.sin())).collect();
        let mut pm = [[c(0.0, 0.0); 2]; 3];
        for i in 0..n {
            for k in 0..nf {
                for comp in 0..2 {
                    let mut acc = c(0.0, 0.0);
                    for f in 0..nf {
                        acc += u[f * nf + k].conj() * psi[i * st + 2 * f + comp];
                    }
                    pm[k][comp] = acc;
                }
                let (co, si) = cs[k];
                let (a, b) = (pm[k][0], pm[k][1]);
                pm[k][0] = a * co - b * si;
                pm[k][1] = a * si + b * co;
            }
            for f in 0..nf {
                for comp in 0..2 {
                    let mut acc = c(0.0, 0.0);
                    for k in 0..nf {
                        acc += u[f * nf + k] * pm[k][comp];
                    }
                    psi[i * st + 2 * f + comp] = acc;
                }
            }
        }
        self.fl.psi = psi;
        self.fl.tmp = tmp;
        self.t = self.t.wrapping_add(1);
        self.flavour_mirror();
    }

    /// Normas por sabor y la copia en f32 (suma de sabores) para el dibujo.
    pub(crate) fn flavour_mirror(&mut self) {
        let nf = self.fl.nf;
        let n = self.w * self.h;
        let mut norms = [0.0f64; 3];
        for i in 0..n {
            let (mut a, mut b) = (c(0.0, 0.0), c(0.0, 0.0));
            for f in 0..nf {
                let (p, q) = (self.fl.psi[i * 2 * nf + 2 * f], self.fl.psi[i * 2 * nf + 2 * f + 1]);
                norms[f] += p.norm_sqr() + q.norm_sqr();
                a += p;
                b += q;
            }
            // el dibujo del Dirac 2D lee ↑ y ↓: se le da la suma de sabores (con su norma total)
            let e: f64 = (0..nf).map(|f| self.fl.psi[i * 2 * nf + 2 * f].norm_sqr() + self.fl.psi[i * 2 * nf + 2 * f + 1].norm_sqr()).sum();
            let ab = (a.norm_sqr() + b.norm_sqr()).max(1e-300);
            let sc = (e / ab).sqrt();
            self.buffer_read[4 * i] = (a.re * sc) as f32;
            self.buffer_read[4 * i + 1] = (a.im * sc) as f32;
            self.buffer_read[4 * i + 2] = (b.re * sc) as f32;
            self.buffer_read[4 * i + 3] = (b.im * sc) as f32;
        }
        self.fl.norms = norms.to_vec();
    }

    /// Envolvente gaussiana en x (uniforme en y) con portadora k₀, en Fourier: ψ̂(k) por celda de x.
    fn flavour_envelope(&self, x0: f64, sigma: f64, k0: f64) -> Vec<C64> {
        let w = self.w;
        let mut env: Vec<C64> = (0..w)
            .map(|x| {
                let dx = min_image(x as f64 - x0, w as f64);
                C64::from_polar((-dx * dx / (2.0 * sigma * sigma)).exp(), k0 * dx)
            })
            .collect();
        let mut planner = rustfft::FftPlanner::<f64>::new();
        planner.plan_fft_forward(w).process(&mut env);
        env
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.22: 2 o 3 sabores en el escenario Dirac 2D, M = U·diag(m)·U† en el paso de masa, con U de
    /// la parametrización estándar (2 sabores: un ángulo; 3: θ₁₂, θ₂₃, θ₁₃ y la fase δ). Masas
    /// vacías: vuelve al Dirac de un sabor.
    pub fn set_flavour_mixing(&mut self, masses: &[f32], angles: &[f32], delta: f32) {
        if self.scenario != 2 || self.dim != 2 || self.sw.active {
            self.set_message("La mezcla de sabores necesita el escenario Dirac en 2D (o en una tira 1D).");
            return;
        }
        let nf = masses.len();
        if nf < 2 || nf > 3 {
            self.fl = Flavour::new();
            return;
        }
        let ang: Vec<f64> = angles.iter().map(|a| *a as f64).collect();
        self.fl.nf = nf;
        self.fl.masses = masses.iter().map(|m| *m as f64).collect();
        self.fl.u = mixing_matrix(&ang, delta as f64, nf);
        self.fl.psi = vec![c(0.0, 0.0); self.w * self.h * 2 * nf];
        self.fl.tmp = self.fl.psi.clone();
        self.fl.active = true;
        self.clear();
    }

    /// Paquete de la banda positiva creado en el sabor α (el estado que produce el W): en la base de
    /// masas, U*_αi por la envolvente con el autovector de la banda +1 de la masa m_i en cada k.
    /// Gaussiano en x (σ) con portadora k₀, uniforme en y. Norma 1.
    pub fn inject_flavour_packet(&mut self, x: f32, sigma: f32, k0: f32, flavour: u32) {
        if !self.fl.active {
            return;
        }
        let (w, h, nf) = (self.w, self.h, self.fl.nf);
        let a = (flavour as usize).min(nf - 1);
        let env = self.flavour_envelope(x as f64, (sigma as f64).max(0.5), k0 as f64);
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let inv = planner.plan_fft_inverse(w);
        // componentes de masa en real: χ_i(x) = U*_αi·Σ_k env(k)·u_i(k)·e^{ikx}
        let mut fl = vec![vec![c(0.0, 0.0); w]; 2 * nf];
        for i in 0..nf {
            let mut up = vec![c(0.0, 0.0); w];
            let mut dn = vec![c(0.0, 0.0); w];
            for j in 0..w {
                let k = wavenumber(j, w);
                let (_, v) = su2_band(&dirac2d_matrix(self.fl.masses[i], k, 0.0), 1);
                up[j] = env[j] * v[0];
                dn[j] = env[j] * v[1];
            }
            inv.process(&mut up);
            inv.process(&mut dn);
            let ua = self.fl.u[a * nf + i].conj();
            // a la base de sabor: ψ_f = Σ_i U_fi·χ_i
            for f in 0..nf {
                let uf = self.fl.u[f * nf + i];
                for x in 0..w {
                    fl[2 * f][x] += uf * ua * up[x];
                    fl[2 * f + 1][x] += uf * ua * dn[x];
                }
            }
        }
        let tot: f64 = fl.iter().map(|v| v.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>() * h as f64;
        let sc = 1.0 / tot.max(1e-300).sqrt();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                for f in 0..nf {
                    self.fl.psi[i * 2 * nf + 2 * f] += fl[2 * f][x] * sc;
                    self.fl.psi[i * 2 * nf + 2 * f + 1] += fl[2 * f + 1][x] * sc;
                }
            }
        }
        self.flavour_mirror();
    }

    /// Norma por sabor (3 valores; el tercero es 0 con dos sabores).
    pub fn flavour_norms(&self) -> Vec<f64> {
        self.fl.norms.clone()
    }

    pub fn flavour_norms_ptr(&self) -> *const f64 {
        self.fl.norms.as_ptr()
    }

    /// P(α → β, t) exacta: Σ_k |env(k)|²·Σ_comp |Σ_i U_βi·U*_αi·e^{−iω_i(k)·t}·u_i(k)|², con ω_i la
    /// banda de la red de masa m_i y u_i su autovector (el mismo paquete que `inject_flavour_packet`).
    pub fn flavour_transition_exact(&self, alpha: u32, beta: u32, t: f32, sigma: f32, k0: f32) -> f64 {
        if !self.fl.active {
            return f64::NAN;
        }
        let (w, nf) = (self.w, self.fl.nf);
        let (a, b) = ((alpha as usize).min(nf - 1), (beta as usize).min(nf - 1));
        let env = self.flavour_envelope(0.0, (sigma as f64).max(0.5), k0 as f64);
        let (mut num, mut den) = (0.0, 0.0);
        for j in 0..w {
            let k = wavenumber(j, w);
            let wgt = env[j].norm_sqr();
            if wgt < 1e-30 {
                continue;
            }
            let mut amp = [c(0.0, 0.0); 2];
            let mut tot = 0.0;
            for i in 0..nf {
                let (_, v) = su2_band(&dirac2d_matrix(self.fl.masses[i], k, 0.0), 1);
                let om = dirac2d_omega(self.fl.masses[i], k, 0.0);
                let ph = self.fl.u[b * nf + i] * self.fl.u[a * nf + i].conj() * C64::from_polar(1.0, -om * t as f64);
                amp[0] += ph * v[0];
                amp[1] += ph * v[1];
                tot += self.fl.u[a * nf + i].norm_sqr();
            }
            num += wgt * (amp[0].norm_sqr() + amp[1].norm_sqr());
            den += wgt * tot;
        }
        num / den.max(1e-300)
    }

    /// J = Im(U_e1·U_μ2·U*_e2·U*_μ1) de la matriz de mezcla actual.
    pub fn flavour_jarlskog(&self) -> f64 {
        if self.fl.nf != 3 {
            return 0.0;
        }
        let u = &self.fl.u;
        (u[0] * u[4] * u[1].conj() * u[3].conj()).im
    }
}
