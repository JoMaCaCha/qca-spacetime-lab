//! Por qué el condensado cargado de la caminata no sirve como Higgs (§2.12 y §3.16): un análisis
//! del modo que crece. Se guardan instantáneas de E durante el crecimiento, se busca el (k, ω) con
//! más potencia (FFT espacio-temporal) y se compara con la resonancia ε₀ − ε_b(−k) = ω_fotón(k)
//! (módulo 2π), con las bandas exactas de la caminata lineal y la dispersión del fotón de red.

use crate::linalg::{eig, C};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct ModeCapture {
    pub frames: Vec<Vec<f64>>,
    pub max_frames: usize,
}

impl ModeCapture {
    pub fn new() -> Self {
        ModeCapture { frames: Vec::new(), max_frames: 256 }
    }
}

#[inline]
fn wrap(a: f64) -> f64 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

impl QuantumSpacetimeUniverse {
    /// Matriz del paso lineal de Grover en el momento k: U(k) = D(k)·C(m), con D la recogida de
    /// cada canal (N ← y+1, S ← y−1, W ← x+1, E ← x−1, F ← z+1, B ← z−1).
    fn grover_matrix(&self, k: [f64; 3]) -> Vec<C> {
        let nc = self.nc_max;
        let (cm, sm) = ((self.mass as f64).cos(), (self.mass as f64).sin());
        let f = 2.0 / nc as f64;
        let src: [[f64; 3]; 6] = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let mut u = vec![Complex::new(0.0, 0.0); nc * nc];
        for r in 0..nc {
            let ph = Complex::from_polar(1.0, k[0] * src[r][0] + k[1] * src[r][1] + k[2] * src[r][2]);
            for c in 0..nc {
                // C(m) = cos m·I − i·sin m·G con G = (2/n_c)·J − I
                let g = f - if r == c { 1.0 } else { 0.0 };
                let coin = Complex::new(if r == c { cm } else { 0.0 }, -sm * g);
                u[r * nc + c] = ph * coin;
            }
        }
        u
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Cuasi-energías ε_b(k) (ordenadas) de la caminata de Grover lineal en k: U(k)·v = e^{−iε}·v.
    pub fn grover_bands(&self, kx: f32, ky: f32, kz: f32) -> Vec<f64> {
        let nc = self.nc_max;
        let e = eig(self.grover_matrix([kx as f64, ky as f64, kz as f64]), nc);
        let mut w: Vec<f64> = e.values.iter().map(|l| -l.arg()).collect();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w
    }

    /// Frecuencia del fotón de red en k: 2s·arcsin((√κ/s)·√(Σ_a sin²(k_a/2))).
    pub fn lattice_photon_omega(&self, kx: f32, ky: f32, kz: f32) -> f64 {
        let s = self.em.substeps.max(1) as f64;
        let k2 = ((kx as f64) / 2.0).sin().powi(2) + ((ky as f64) / 2.0).sin().powi(2) + if self.dim == 3 { ((kz as f64) / 2.0).sin().powi(2) } else { 0.0 };
        2.0 * s * ((self.em.kappa.sqrt() / s) * k2.sqrt()).min(1.0).asin()
    }

    /// Guarda una instantánea de E (todos los enlaces) para el análisis del modo que crece.
    pub fn capture_field_snapshot(&mut self) {
        if self.modes.frames.len() >= self.modes.max_frames {
            self.modes.frames.remove(0);
        }
        self.modes.frames.push(self.em.e.clone());
    }

    pub fn clear_field_snapshots(&mut self) {
        self.modes.frames.clear();
    }

    /// Modo que crece (2D): [k_x, k_y, ω, fracción longitudinal, ω_fotón(k), desajuste mínimo de
    /// la resonancia ε₀ − ε_b(±k) − ω (módulo 2π), banda]. ε₀ es la cuasi-energía del condensado,
    /// que se pasa medida (`eps0`).
    pub fn analyze_growing_mode(&self, eps0: f32) -> Vec<f64> {
        let nt = self.modes.frames.len();
        if nt < 8 || self.dim != 2 {
            return vec![f64::NAN; 7];
        }
        let (w, h, n) = (self.w, self.h, self.n);
        let mut fft = crate::fftn::FftN::<f64>::new(w, h, 1);
        // FFT espacial de E_x y E_y en cada instantánea
        let mut ex_k: Vec<Vec<Complex<f64>>> = Vec::with_capacity(nt);
        let mut ey_k: Vec<Vec<Complex<f64>>> = Vec::with_capacity(nt);
        for f in &self.modes.frames {
            let mut bx: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(f[i], 0.0)).collect();
            let mut by: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(f[n + i], 0.0)).collect();
            fft.forward(&mut bx);
            fft.forward(&mut by);
            ex_k.push(bx);
            ey_k.push(by);
        }
        // el k con más potencia en la última instantánea (el modo ya ha crecido)
        let last = nt - 1;
        let mut best = (0.0f64, 1usize);
        for i in 1..n {
            let p = ex_k[last][i].norm_sqr() + ey_k[last][i].norm_sqr();
            if p > best.0 {
                best = (p, i);
            }
        }
        let i = best.1;
        let (ix, iy) = (i % w, i / w);
        let kx = crate::fftn::wavenumber(ix, w);
        let ky = crate::fftn::wavenumber(iy, h);
        // fracción longitudinal: proyección de Ê sobre d = (1 − e^{−ik_x}, 1 − e^{−ik_y})
        let d = [Complex::new(1.0, 0.0) - Complex::from_polar(1.0, -kx), Complex::new(1.0, 0.0) - Complex::from_polar(1.0, -ky)];
        let dn = d[0].norm_sqr() + d[1].norm_sqr();
        let (mut lon, mut tot) = (0.0, 0.0);
        for t in 0..nt {
            let e = [ex_k[t][i], ey_k[t][i]];
            let dot = d[0].conj() * e[0] + d[1].conj() * e[1];
            lon += dot.norm_sqr() / dn.max(1e-300);
            tot += e[0].norm_sqr() + e[1].norm_sqr();
        }
        // frecuencia: rotación de fase de la componente dominante entre instantáneas
        let comp: Vec<Complex<f64>> = (0..nt).map(|t| if ex_k[last][i].norm() > ey_k[last][i].norm() { ex_k[t][i] } else { ey_k[t][i] }).collect();
        let _ = h;
        // E real: el modo k y el −k son conjugados; la fase de Ê(k) gira a −ω o +ω según el sentido.
        // Se mide la frecuencia con el pico de |Σ_t Ê_k(t)·e^{iωt}| en ω ∈ (−π, π].
        let mut bw = (0.0f64, 0.0f64);
        let steps = 4096;
        for j in 0..steps {
            let om = -PI + 2.0 * PI * (j as f64 + 0.5) / steps as f64;
            let mut s = Complex::new(0.0, 0.0);
            for (t, cv) in comp.iter().enumerate() {
                s += cv * Complex::from_polar(1.0, om * t as f64);
            }
            if s.norm() > bw.0 {
                bw = (s.norm(), om);
            }
        }
        let omega = bw.1.abs();
        let wph = self.lattice_photon_omega(kx as f32, ky as f32, 0.0);
        // resonancia: min_b,± |ε₀ − ε_b(±k) − ω| (módulo 2π)
        let mut mis = (f64::INFINITY, 0usize);
        for sgn in [-1.0f64, 1.0] {
            let bands = self.grover_bands((sgn * kx) as f32, (sgn * ky) as f32, 0.0);
            for (b, eb) in bands.iter().enumerate() {
                let r = wrap(eps0 as f64 - eb - omega).abs();
                if r < mis.0 {
                    mis = (r, b);
                }
            }
        }
        vec![kx, ky, omega, lon / tot.max(1e-300), wph, mis.0, mis.1 as f64]
    }
}
