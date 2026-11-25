//! Dirac en un espacio-tiempo conformemente plano (§2.15): con g = Ω²·η, la ecuación de Dirac sin
//! masa es invariante conforme y Dirac en g es Dirac plano con masa m·Ω(r). La caminata de Dirac 2D
//! con la masa por celda de §2.13 ya es ese espacio-tiempo curvo.
//! - Nordström en 2D: Ω = 1 + Φ. Una partícula en reposo cae con a = −∇Φ sin depender de m₀
//!   (principio de equivalencia), su frecuencia es m₀·(1 + Φ) (corrimiento al rojo), una partícula
//!   rápida pesa menos, −(m/E)²·∇Φ, y la luz no siente Φ. Es la gravedad escalar de Nordström,
//!   descartada por la observación (no desvía la luz).
//! - En 1+1D toda métrica es conformemente plana: la parte radial de Schwarzschild en coordenada
//!   tortuga, x* = r + 2M·ln(r/2M − 1), con Ω² = 1 − 2M/r(x*). Es solo la parte radial; Φ es un
//!   campo externo fijo y no se resuelven las ecuaciones de Einstein.

use crate::dirac2d::{c, dirac2d_matrix, su2_band};
use crate::util::min_image;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// r(x*) de la coordenada tortuga: x*/2M − 1 = u + ln u con u = r/2M − 1 (Newton).
pub fn tortoise_r(xs: f64, m: f64) -> f64 {
    if !(xs.is_finite() && m.is_finite()) || m <= 0.0 {
        return 0.0;
    }
    let cc = xs / (2.0 * m) - 1.0;
    if cc < -700.0 {
        return 2.0 * m;
    }
    let mut u = if cc > 1.0 { cc - cc.ln() } else { cc.exp().max(1e-300) };
    for _ in 0..100 {
        if !u.is_finite() || u <= 1e-300 {
            u = 1e-300;
            break;
        }
        let f = u + u.ln() - cc;
        let d = 1.0 + 1.0 / u;
        let nu = u - f / d;
        u = if !nu.is_finite() || nu <= 0.0 { (0.5 * u).max(1e-300) } else { nu };
        if f.abs() < 1e-14 * cc.abs().max(1.0) {
            break;
        }
    }
    2.0 * m * (1.0 + u)
}

/// Ω = √(1 − 2M/r(x*)).
pub fn schwarzschild_omega(xs: f64, m: f64) -> f64 {
    if !(xs.is_finite() && m.is_finite()) || m <= 0.0 {
        return 0.0;
    }
    let r = tortoise_r(xs, m);
    if r <= 0.0 {
        return 0.0;
    }
    (1.0 - 2.0 * m / r).max(0.0).sqrt()
}

impl QuantumSpacetimeUniverse {
    /// Ω(x, y) del potencial conforme actual.
    pub(crate) fn conformal_omega(&self, x: f64, y: f64) -> f64 {
        let (kind, a) = (self.conf_kind, self.conf_amp);
        match kind {
            1 => 1.0 + a * (2.0 * PI * x / self.w as f64).cos(),
            2 => {
                let dx = min_image(x - self.conf_pos[0], self.w as f64);
                let dy = min_image(y - self.conf_pos[1], self.h as f64);
                1.0 + a * (-(dx * dx + dy * dy) / (2.0 * self.conf_width * self.conf_width)).exp()
            }
            3 => schwarzschild_omega(x - self.conf_pos[0], a),
            _ => 1.0,
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.15 (escenario Dirac 2D): 0 = nada, 1 = pozo cos(2πx/W) (Φ = amplitude·cos), 2 = pozo
    /// gaussiano en (x, y) de anchura `width`, 3 = Schwarzschild radial en coordenada tortuga
    /// (amplitude = M, x = celda con x* = 0). La masa de Dirac pasa a ser m₀·Ω, con m₀ la masa actual.
    pub fn set_conformal_potential(&mut self, kind: u32, amplitude: f32, x: f32, y: f32, width: f32) {
        self.message.clear();
        if self.scenario != 2 || self.dim != 2 || self.sw.active {
            self.set_message("El potencial conforme necesita el escenario Dirac 2D (§2.15).");
            return;
        }
        if !(amplitude.is_finite() && x.is_finite() && y.is_finite() && width.is_finite()) {
            return;
        }
        self.conf_kind = kind.min(3);
        self.conf_amp = if self.conf_kind == 3 { (amplitude as f64).max(0.5) } else { (amplitude as f64).clamp(-0.5, 0.5) };
        self.conf_pos = [x as f64, y as f64];
        self.conf_width = (width as f64).max(0.5);
        if self.conf_kind == 0 {
            self.dmass.active = false;
            return;
        }
        self.conf_m0 = self.mass as f64;
        let n = self.w * self.h;
        if self.dmass.m.len() != n {
            self.dmass.m = vec![0.0; n];
            self.dmass.cs = vec![0.0; 2 * n];
        }
        for i in 0..n {
            let (px, py) = ((i % self.w) as f64, (i / self.w) as f64);
            self.dmass.m[i] = (self.conf_m0 * self.conformal_omega(px, py)) as f32;
        }
        self.dmass.active = true;
        self.dmass.yukawa = 0;
        self.dmass_refresh();
    }

    /// Ω en una celda (1 sin potencial conforme).
    pub fn conformal_omega_at(&self, x: f32, y: f32) -> f64 {
        if !(x.is_finite() && y.is_finite()) {
            return 1.0;
        }
        self.conformal_omega(x as f64, y as f64)
    }

    /// r(x*) de la coordenada tortuga con masa M.
    pub fn tortoise_radius(&self, x_star: f32, m: f32) -> f64 {
        tortoise_r(x_star as f64, m as f64)
    }

    /// x*(r) = r + 2M·ln(r/2M − 1).
    pub fn tortoise_coordinate(&self, r: f32, m: f32) -> f64 {
        let (r, m) = (r as f64, m as f64);
        if !(r.is_finite() && m.is_finite()) || m <= 0.0 || r <= 2.0 * m {
            return f64::NAN;
        }
        r + 2.0 * m * (r / (2.0 * m) - 1.0).ln()
    }

    /// Geodésica radial soltada en reposo en x*₀ (Schwarzschild, kind = 3): x*(t) en `steps` pasos
    /// de la ecuación v̇ = −Ω·Ω′/Ω₀² (equivale a d(x*)/dt = −√(1 − Ω²/Ω₀²)), con RK4 de 8 subpasos.
    pub fn schwarzschild_geodesic(&self, x_star0: f32, m: f32, steps: u32) -> Vec<f64> {
        let m = m as f64;
        if !(x_star0.is_finite() && m.is_finite()) || m <= 0.0 {
            return Vec::new();
        }
        let steps = steps.min(200_000);
        let om = |x: f64| schwarzschild_omega(x, m);
        let o0 = om(x_star0 as f64);
        if !o0.is_finite() || o0 <= 1e-15 {
            return vec![x_star0 as f64; steps as usize + 1];
        }
        let acc = |x: f64| {
            let h = 1e-3;
            let (a, b) = (om(x + h), om(x - h));
            -0.5 * (a * a - b * b) / (2.0 * h) / (o0 * o0)
        };
        let (mut x, mut v) = (x_star0 as f64, 0.0f64);
        let mut out = Vec::with_capacity(steps as usize + 1);
        out.push(x);
        let sub = 8;
        let dt = 1.0 / sub as f64;
        for _ in 0..steps {
            for _ in 0..sub {
                let (k1x, k1v) = (v, acc(x));
                let (k2x, k2v) = (v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x));
                let (k3x, k3v) = (v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x));
                let (k4x, k4v) = (v + dt * k3v, acc(x + dt * k3x));
                x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
                v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
            }
            out.push(x);
        }
        out
    }

    /// Dirac 2D: onda plana en y (k_y) con envolvente gaussiana en x de anchura σ y el espinor de
    /// la banda en (0, k_y) (§3.12: "una partícula rápida pesa menos").
    pub fn inject_dirac_plane_wave_y(&mut self, x: f32, sigma: f32, ky: f32, band: i32) {
        if self.scenario != 2 || self.dim != 2 || self.sw.active {
            return;
        }
        if !(x.is_finite() && sigma.is_finite() && ky.is_finite()) || sigma <= 0.0 {
            return;
        }
        let (x0, s, ky) = (x as f64, (sigma as f64).max(0.5), ky as f64);
        let sp = su2_band(&dirac2d_matrix(self.mass as f64, 0.0, ky), if band < 0 { -1 } else { 1 }).1;
        let n = self.w * self.h;
        let nco = self.ncol.max(1);
        let stride = self.stride();
        if self.buffer_read.len() < n * stride {
            return;
        }
        let mut norm = 0.0;
        let mut vals = vec![c(0.0, 0.0); n];
        for i in 0..n {
            let dx = min_image((i % self.w) as f64 - x0, self.w as f64);
            let e = (-dx * dx / (2.0 * s * s)).exp();
            vals[i] = crate::dirac2d::C64::from_polar(e, ky * (i / self.w) as f64);
            norm += e * e;
        }
        if norm <= 0.0 {
            return;
        }
        let a = 1.0 / norm.sqrt();
        for i in 0..n {
            for comp in 0..2 {
                let v = vals[i] * sp[comp] * a;
                let o = i * stride + 2 * (comp * nco);
                self.buffer_read[o] += v.re as f32;
                self.buffer_read[o + 1] += v.im as f32;
            }
        }
    }
}
