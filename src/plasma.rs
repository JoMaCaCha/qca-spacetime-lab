//! El condensado de Dirac cargado (§2.12, §3.13 y §3.16): un estado uniforme de la banda +1 en
//! reposo, con la carga media neutralizada (la ley de Gauss usa ρ − ⟨ρ⟩), sobre el U(1) dinámico.
//! Oscila a la frecuencia de plasma (Langmuir), se desestabiliza por la resonancia
//! ε₀ − ε_b(−k) = ω_fotón(k) cuando el fotón de red llega a la banda negativa, y con fotones
//! lentos (κ = 0,1) es estable: el fotón adquiere masa y el campo se expulsa (Meissner).
//! Es una onda clásica de una partícula con mucha norma, no un mar de Dirac.

use crate::fftn::FftN;
use crate::rng::SplitMix64;
use crate::spinor::{cx, m4_apply};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Norma de los paquetes que se inyecten después (por defecto 1).
    pub fn set_packet_norm(&mut self, amount: f32) {
        if amount.is_finite() && amount > 0.0 {
            self.packet_norm = amount as f64;
        }
    }

    /// ∂²ω/∂k_a² de la banda +1 en k = 0: 1/m\* del fondo de la banda (Dirac 3D).
    pub fn dirac_band_curvature(&self, axis: u32) -> f64 {
        let h = 1e-3;
        let mut kp = [0.0; 3];
        kp[(axis as usize).min(2)] = h;
        let mut km = [0.0; 3];
        km[(axis as usize).min(2)] = -h;
        (self.sw_omega(kp) - 2.0 * self.sw_omega([0.0; 3]) + self.sw_omega(km)) / (h * h)
    }

    /// Condensado de Dirac (§3.13): estado uniforme de la banda +1 en k = 0 con el espín de
    /// `set_packet_spin`, densidad ρ̄ por celda y una modulación δ·cos(2π·mode·x/W) proyectada sobre
    /// la banda +1; después, ruido blanco de amplitud relativa `noise` (sin proyectar, para que la
    /// banda −1 tenga una semilla). Reemplaza el estado.
    pub fn seed_dirac_condensate(&mut self, rho: f32, modulation: f32, mode: u32, noise: f32, seed: u32) {
        if !self.sw.active || self.sw.weyl {
            self.set_message("seed_dirac_condensate necesita el escenario Dirac 3D.");
            return;
        }
        let ns = self.sw.ns;
        let nco = self.ncol;
        let namp = ns * nco;
        let n = self.n;
        let rho = rho.max(0.0) as f64;
        let u0 = self.sw_band_spinor([0.0; 3], 1);
        let k = 2.0 * PI * mode as f64 / self.w as f64;
        let mut comps: Vec<Vec<Complex<f64>>> = vec![vec![cx(0.0, 0.0); n]; namp];
        for i in 0..n {
            let x = (i % self.w) as f64;
            let amp = rho.sqrt() * (1.0 + modulation as f64 * (k * x).cos());
            for s in 0..ns {
                comps[s * nco][i] = u0[s] * amp;
            }
        }
        if modulation != 0.0 {
            let mut fft = FftN::<f64>::new(self.w, self.h, self.d);
            for c in comps.iter_mut() {
                fft.forward(c);
            }
            let start = (self.t & 1) as usize;
            for i in 0..n {
                let (x, y, z) = self.coords(i);
                let kv = [crate::fftn::wavenumber(x, self.w), crate::fftn::wavenumber(y, self.h), if self.dim == 3 { crate::fftn::wavenumber(z, self.d) } else { self.sw.kz }];
                let p = self.sw_projector(start, kv, 1);
                for a in 0..nco {
                    let mut v = [cx(0.0, 0.0); 4];
                    for s in 0..ns {
                        v[s] = comps[s * nco + a][i];
                    }
                    let r = m4_apply(&p, &v);
                    for s in 0..ns {
                        comps[s * nco + a][i] = r[s];
                    }
                }
            }
            for c in comps.iter_mut() {
                fft.inverse(c);
            }
        }
        let mut rng = SplitMix64::new(seed as u64 ^ 0xC0DE);
        let sn = noise as f64 * rho.sqrt();
        for i in 0..n {
            for kk in 0..namp {
                let v = comps[kk][i];
                let (nr, ni) = if sn > 0.0 { (sn * rng.gauss(), sn * rng.gauss()) } else { (0.0, 0.0) };
                self.sw.psi[i * 2 * namp + 2 * kk] = v.re + nr;
                self.sw.psi[i * 2 * namp + 2 * kk + 1] = v.im + ni;
            }
        }
        self.sw.parts.clear();
        self.sw.part_centers.clear();
        self.sw.center = [0.5 * self.w as f64, 0.5 * self.h as f64, 0.5 * self.d as f64];
        self.sw_mirror();
    }

    /// Amplitud del modo de densidad Σ_r ρ(r)·cos(2π·mode·x/W)/Σ_r ρ (oscilación de Langmuir).
    pub fn density_mode_x(&self, mode: u32) -> f64 {
        let k = 2.0 * PI * mode as f64 / self.w as f64;
        let (mut a, mut t) = (0.0, 0.0);
        for i in 0..self.n {
            let r = self.matter_density_f64(i);
            a += r * (k * (i % self.w) as f64).cos();
            t += r;
        }
        if t > 0.0 {
            a / t
        } else {
            0.0
        }
    }

    /// Onda transversal: E_axis += amp·cos(2π·mode·x/W) en todos los enlaces de ese eje (axis ≠ 0:
    /// no tiene divergencia, así que la ley de Gauss no cambia).
    pub fn add_transverse_wave(&mut self, axis: u32, mode: u32, amp: f32) {
        let a = (axis as usize).clamp(1, self.dim - 1);
        let n = self.n;
        let k = 2.0 * PI * mode as f64 / self.w as f64;
        for i in 0..n {
            self.em.e[a * n + i] += amp as f64 * (k * (i % self.w) as f64).cos();
        }
    }

    /// Componente de Fourier Σ E_axis·cos(2π·mode·x/W)/N del campo (el modo de una onda transversal).
    pub fn field_mode_x(&self, axis: u32, mode: u32) -> f64 {
        let a = (axis as usize).min(self.dim - 1);
        let n = self.n;
        let k = 2.0 * PI * mode as f64 / self.w as f64;
        let mut s = 0.0;
        for i in 0..n {
            s += self.em.e[a * n + i] * (k * (i % self.w) as f64).cos();
        }
        s / n as f64
    }

    /// ⟨E_a⟩ sobre todos los enlaces del eje (el modo uniforme k = 0).
    pub fn field_mean(&self, axis: u32) -> f64 {
        let a = (axis as usize).min(self.dim - 1);
        let n = self.n;
        self.em.e[a * n..(a + 1) * n].iter().sum::<f64>() / n as f64
    }

    /// El modo de E_a que crece a lo largo de x (Dirac 3D, §3.16), con las instantáneas guardadas:
    /// [k_x, ω, ω_fotón(k_x)]. Toma el k con más potencia en la última instantánea y la frecuencia
    /// de la rotación de fase de Ê_a(k, t).
    pub fn analyze_growing_mode_x(&self, axis: u32) -> Vec<f64> {
        let nt = self.modes.frames.len();
        if nt < 8 {
            return vec![f64::NAN; 3];
        }
        let (w, n) = (self.w, self.n);
        let a = (axis as usize).min(self.dim - 1);
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(w);
        let mut lines: Vec<Vec<Complex<f64>>> = Vec::with_capacity(nt);
        for f in &self.modes.frames {
            let mut line = vec![cx(0.0, 0.0); w];
            for i in 0..n {
                line[i % w] += cx(f[a * n + i], 0.0);
            }
            fft.process(&mut line);
            lines.push(line);
        }
        let last = nt - 1;
        let mut best = (0.0f64, 1usize);
        for j in 1..w {
            let p = lines[last][j].norm_sqr();
            if p > best.0 {
                best = (p, j);
            }
        }
        let j = best.1;
        let kx = crate::fftn::wavenumber(j, w);
        let mut bw = (0.0f64, 0.0f64);
        let steps = 8192;
        for s in 0..steps {
            let om = -PI + 2.0 * PI * (s as f64 + 0.5) / steps as f64;
            let mut acc = cx(0.0, 0.0);
            for (t, l) in lines.iter().enumerate() {
                acc += l[j] * Complex::from_polar(1.0, om * t as f64);
            }
            if acc.norm() > bw.0 {
                bw = (acc.norm(), om);
            }
        }
        vec![kx.abs(), bw.1.abs(), self.lattice_photon_omega(kx as f32, 0.0, 0.0)]
    }

    /// Raíz de m + E(k) = ω_fotón(k) a lo largo de x (la resonancia de §3.16), por bisección en
    /// (0, π]; NaN si no hay cruce.
    pub fn dirac_photon_resonance_x(&self) -> f64 {
        let m = self.sw_omega([0.0; 3]);
        let f = |k: f64| m + self.sw_omega([k, 0.0, 0.0]) - self.lattice_photon_omega(k as f32, 0.0, 0.0);
        let mut prev = (1e-3, f(1e-3));
        let steps = 2000;
        for s in 1..=steps {
            let k = PI * s as f64 / steps as f64;
            let v = f(k);
            if v.signum() != prev.1.signum() {
                let (mut lo, mut hi) = (prev.0, k);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if f(mid).signum() == f(lo).signum() {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return 0.5 * (lo + hi);
            }
            prev = (k, v);
        }
        f64::NAN
    }
}
