//! Escenario "vórtices" (§2.7): la misma caminata con g < 0 sobre un fondo uniforme de la banda
//! simétrica es un superfluido. Los vórtices se comportan como cargas cuantizadas con interacción
//! logarítmica (dualidad partícula-vórtice en 2+1D) y el sonido de Bogoliubov como "fotones".

use crate::rng::SplitMix64;
use crate::util::{min_image, wrap_angle};
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub const MAX_VORTICES: usize = 4096;

pub struct VortexState {
    pub data: Vec<f32>,
    pub count: usize,
    pub charge: i32,
    pub imprinted: usize,
    pub rho_bg: f32,
}

impl VortexState {
    pub fn new() -> Self {
        VortexState { data: Vec::new(), count: 0, charge: 0, imprinted: 0, rho_bg: 0.5 }
    }
}

impl QuantumSpacetimeUniverse {
    /// f′(ρ̄) = |g|/(1 + ρ̄/E_max)², la pendiente de la no linealidad saturable.
    pub(crate) fn nonlinear_slope(&self, rho: f64) -> f64 {
        (self.g as f64).abs() / (1.0 + rho / self.e_max as f64).powi(2)
    }

    /// Fase arg Σψ de la celda.
    #[inline]
    pub(crate) fn cell_phase(&self, i: usize) -> f64 {
        let (sr, si) = self.sum_psi_at(i);
        (si as f64).atan2(sr as f64)
    }

    /// Detección de vórtices en el plano z (2D: z = 0): winding de cada plaqueta 2×2.
    pub(crate) fn detect_vortices(&mut self, z: usize) -> usize {
        if self.vort.data.len() != 3 * MAX_VORTICES {
            self.vort.data.resize(3 * MAX_VORTICES, 0.0);
        }
        let (w, h) = (self.w, self.h);
        let mut count = 0usize;
        let mut charge = 0i32;
        for y in 0..h {
            let yp = if y + 1 == h { 0 } else { y + 1 };
            for x in 0..w {
                let xp = if x + 1 == w { 0 } else { x + 1 };
                if self.has_walls {
                    // la amplitud es 0 en los muros: su fase no está definida
                    let c = [self.idx(x, y, z), self.idx(xp, y, z), self.idx(xp, yp, z), self.idx(x, yp, z)];
                    if c.iter().any(|&i| self.wall[i] != 0) {
                        continue;
                    }
                }
                let p00 = self.cell_phase(self.idx(x, y, z));
                let p10 = self.cell_phase(self.idx(xp, y, z));
                let p11 = self.cell_phase(self.idx(xp, yp, z));
                let p01 = self.cell_phase(self.idx(x, yp, z));
                let s = wrap_angle(p10 - p00) + wrap_angle(p11 - p10) + wrap_angle(p01 - p11) + wrap_angle(p00 - p01);
                let wnd = s / (2.0 * PI);
                let q = wnd.round() as i32;
                if q != 0 {
                    if count < MAX_VORTICES {
                        self.vort.data[3 * count] = x as f32 + 0.5;
                        self.vort.data[3 * count + 1] = y as f32 + 0.5;
                        self.vort.data[3 * count + 2] = q as f32;
                    }
                    count += 1;
                    charge += q;
                }
            }
        }
        self.vort.count = count.min(MAX_VORTICES);
        self.vort.charge = charge;
        self.vort.count
    }

    /// Máximo de |w − round(w)| sobre las plaquetas con winding no nulo (test 26).
    pub(crate) fn winding_max_error(&self) -> f64 {
        let (w, h) = (self.w, self.h);
        let mut worst = 0.0f64;
        for y in 0..h {
            let yp = if y + 1 == h { 0 } else { y + 1 };
            for x in 0..w {
                let xp = if x + 1 == w { 0 } else { x + 1 };
                let p00 = self.cell_phase(self.idx(x, y, 0));
                let p10 = self.cell_phase(self.idx(xp, y, 0));
                let p11 = self.cell_phase(self.idx(xp, yp, 0));
                let p01 = self.cell_phase(self.idx(x, yp, 0));
                let s = wrap_angle(p10 - p00) + wrap_angle(p11 - p10) + wrap_angle(p01 - p11) + wrap_angle(p00 - p01);
                let wnd = s / (2.0 * PI);
                if wnd.round() != 0.0 {
                    worst = worst.max((wnd - wnd.round()).abs());
                }
            }
        }
        worst
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fondo uniforme de la banda simétrica: ψ_k = √ρ̄/√(n_c·N)·(1 + ruido relativo complejo).
    pub fn seed_uniform_background(&mut self, rho: f32, noise: f32, seed: u32) {
        let rho = if rho.is_finite() { rho.max(0.0) as f64 } else { 0.5 };
        let noise = if noise.is_finite() { noise.max(0.0) as f64 } else { 0.0 };
        let mut rng = SplitMix64::new(seed as u64 ^ 0x5EED_BA5E);
        let namp = self.namp();
        let stride = self.stride();
        let a = (rho / namp.max(1) as f64).sqrt();
        for i in 0..self.n {
            let (re, im) = if self.has_walls && self.wall[i] != 0 {
                (0.0, 0.0)
            } else {
                ((a * (1.0 + noise * rng.gauss())) as f32, (a * noise * rng.gauss()) as f32)
            };
            for k in 0..namp {
                self.buffer_read[i * stride + 2 * k] = re;
                self.buffer_read[i * stride + 2 * k + 1] = im;
            }
        }
        self.vort.imprinted = 0;
        self.vort.rho_bg = rho as f32;
        self.t = 0;
        self.theta_ref = 0.0;
        self.sol.reset();
    }

    /// Imprime un vórtice de carga `charge` en (x, y): φ ← φ·r/√(r² + ξ²)·e^{i·q·θ}, con θ la suma
    /// de las fases de las imágenes periódicas (i, j) ∈ [−4, 4]² (fase compatible con el toro).
    /// Solo 2D (en 3D actúa en todos los planos z). La carga total en el toro es siempre 0:
    /// imprime configuraciones neutras.
    pub fn imprint_vortex(&mut self, x: f32, y: f32, charge: i32) {
        if !(x.is_finite() && y.is_finite()) || charge == 0 {
            return;
        }
        let max_q = if self.scenario == 10 { 4 } else { 1 };
        let q_clamped = charge.clamp(-max_q, max_q);
        if self.scenario == 10 {
            // escenario 10: perfil radial exacto F_|q|(r/ξ) del vórtice de carga q (Newton radial)
            self.imprint_vortex_radial(x as f64, y as f64, q_clamped);
            self.vort.imprinted += q_clamped.unsigned_abs() as usize;
            return;
        }
        let q = q_clamped as f64;
        let (w, h) = (self.w as f64, self.h as f64);
        let xi_raw = self.healing_length();
        let xi = if xi_raw.is_finite() && xi_raw > 0.0 { xi_raw } else { 1.0 };
        let stride = self.stride();
        let namp = self.namp();
        for i in 0..self.n {
            let (cx, cy, _) = self.coords(i);
            let (px, py) = (cx as f64, cy as f64);
            let mut th = 0.0f64;
            for jy in -4..=4 {
                for ix in -4..=4 {
                    th += (py - y as f64 - jy as f64 * h).atan2(px - x as f64 - ix as f64 * w);
                }
            }
            let dx = min_image(px - x as f64, w);
            let dy = min_image(py - y as f64, h);
            let r = (dx * dx + dy * dy).sqrt();
            let prof = r / (r * r + xi * xi).sqrt();
            let (c, s) = ((q * th).cos() * prof, (q * th).sin() * prof);
            for k in 0..namp {
                let re = self.buffer_read[i * stride + 2 * k] as f64;
                let im = self.buffer_read[i * stride + 2 * k + 1] as f64;
                self.buffer_read[i * stride + 2 * k] = (re * c - im * s) as f32;
                self.buffer_read[i * stride + 2 * k + 1] = (re * s + im * c) as f32;
            }
        }
        self.vort.imprinted += q_clamped.unsigned_abs() as usize;
    }

    /// Número de vórtices guardados en `vortex_data_ptr` (acotado a MAX_VORTICES = 4096).
    pub fn analyze_vortices(&mut self) -> u32 {
        self.detect_vortices(0);
        self.vort.count as u32
    }

    /// Número de vórtices actualmente almacenados en `vortex_data_ptr`.
    pub fn vortex_stored_count(&self) -> u32 {
        self.vort.count as u32
    }

    /// n × 3 floats: [x, y, carga].
    pub fn vortex_data_ptr(&self) -> *const f32 {
        self.vort.data.as_ptr()
    }

    pub fn total_topological_charge(&self) -> i32 {
        self.vort.charge
    }

    /// Carga total imprimida en valor absoluto desde el último fondo (para contar los espontáneos).
    pub fn imprinted_vortex_count(&self) -> u32 {
        self.vort.imprinted as u32
    }

    /// ξ = 1/√(2·m\*·ρ̄·f′) con ρ̄ = ⟨E⟩ (en celdas); 0 si no hay régimen superfluido (g ≥ 0 o m\* ≤ 0).
    pub fn healing_length(&self) -> f64 {
        let m_star = self.band_mass();
        if !m_star.is_finite() || m_star <= 0.0 || self.g >= 0.0 {
            return 0.0;
        }
        let rho = self.mean_density().max(1e-12);
        let fp = self.nonlinear_slope(rho).max(1e-15);
        1.0 / (2.0 * m_star * rho * fp).sqrt()
    }

    /// c_s = √(ρ̄·f′/m\*) (celdas/paso); 0 si g ≥ 0 o m\* ≤ 0.
    pub fn sound_speed(&self) -> f64 {
        let m_star = self.band_mass();
        if !m_star.is_finite() || m_star <= 0.0 || self.g >= 0.0 {
            return 0.0;
        }
        let rho = self.mean_density().max(0.0);
        (rho * self.nonlinear_slope(rho) / m_star).sqrt()
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Máximo de |w − round(w)| en las plaquetas con winding no nulo (debe ser ~10⁻⁷).
    pub fn vortex_winding_error(&self) -> f64 {
        self.winding_max_error()
    }
}
