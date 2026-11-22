//! Modo gravedad (§2.6): un campo auxiliar Φ con fuente E − ⟨E⟩ que se acopla a la vibración,
//! ω_nl ← ω_nl + γ·Φ. Es un análogo **newtoniano escalar** (Newton–Schrödinger), no la
//! relatividad general.
//!
//! - Variante estática: −∇²_red Φ = 4π(E − ⟨E⟩) por FFT, anulando k = 0 (*Jeans swindle*).
//! - Variante onda: Φ_{t+1} = Φ_t + (1 − η)(Φ_t − Φ_{t−1}) + c_g²(∇²_red Φ_t + 4π(E − ⟨E⟩)),
//!   local y causal, con c_g ≤ 1/√dim.
//! - Expansión de Einstein–de Sitter en tiempo superconforme: γ_ef(τ) = γ₀·a(τ).
//! - Fuentes externas congeladas (una masa gaussiana o un campo sinusoidal) en `phi_ext`.

use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Umbral a partir del cual la expansión se detiene: γ_ef·(max Φ − min Φ) ≥ 0,5·m.
pub const EXPANSION_STOP_FRACTION: f64 = 0.5;

pub struct Gravity {
    pub mode: u32,
    pub gamma: f32,
    pub cg: f32,
    pub eta: f32,
    pub phi: Vec<f32>,
    pub phi_prev: Vec<f32>,
    pub phi_ext: Vec<f32>,
    pub has_ext: bool,
    /// 4π/λ(k), con λ(k) = Σ_a 2(1 − cos k_a) el autovalor de −∇²_red; 0 en k = 0.
    pub poisson_kernel: Vec<f32>,
    pub gbuf: Vec<Complex<f32>>,
    pub phi_min: f32,
    pub phi_max: f32,
    // expansión (§2.6)
    pub exp_on: bool,
    pub exp_stopped: bool,
    pub exp_tau: f64,
    pub exp_c: f64,
    pub exp_a: f64,
    pub exp_a_max: f64,
    pub gamma0: f32,
}

impl Gravity {
    pub fn new(w: usize, h: usize, d: usize) -> Self {
        let n = w * h * d;
        let mut kernel = vec![0.0f32; n];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let i = (z * h + y) * w + x;
                    let mut lam = 2.0 * (1.0 - (2.0 * PI * x as f64 / w as f64).cos()) + 2.0 * (1.0 - (2.0 * PI * y as f64 / h as f64).cos());
                    if d > 1 {
                        lam += 2.0 * (1.0 - (2.0 * PI * z as f64 / d as f64).cos());
                    }
                    kernel[i] = if lam > 1e-12 { (4.0 * PI / lam) as f32 } else { 0.0 };
                }
            }
        }
        Gravity {
            mode: 0,
            gamma: 1e-6,
            cg: 0.5,
            eta: 0.02,
            phi: vec![0.0; n],
            phi_prev: vec![0.0; n],
            phi_ext: vec![0.0; n],
            has_ext: false,
            poisson_kernel: kernel,
            gbuf: vec![Complex::new(0.0, 0.0); n],
            phi_min: 0.0,
            phi_max: 0.0,
            exp_on: false,
            exp_stopped: false,
            exp_tau: 0.0,
            exp_c: 0.0,
            exp_a: 1.0,
            exp_a_max: 10.0,
            gamma0: 1e-6,
        }
    }

    /// γ efectivo: γ₀·a(τ) con la expansión, γ sin ella.
    #[inline(always)]
    pub fn gamma_eff(&self) -> f32 {
        if self.exp_on {
            (self.gamma0 as f64 * self.exp_a) as f32
        } else {
            self.gamma
        }
    }
}

impl QuantumSpacetimeUniverse {
    /// Resuelve −∇²_red Φ = 4π(ρ − ⟨ρ⟩) por FFT para la densidad `rho` (N_celdas) y deja Φ en `out`.
    pub(crate) fn poisson_solve(&mut self, rho_source: PoissonSource, out_ext: bool) {
        let n = self.n;
        let mut buf = std::mem::take(&mut self.grav.gbuf);
        match rho_source {
            PoissonSource::Matter => {
                let mean = self.mean_density() as f32;
                for (i, v) in buf.iter_mut().enumerate().take(n) {
                    *v = Complex::new(self.density(i) - mean, 0.0);
                }
            }
            PoissonSource::Gaussian { x, y, z, mass, sigma } => {
                let dims = [self.w as f64, self.h as f64, self.d as f64];
                let mut tot = 0.0f64;
                for (i, v) in buf.iter_mut().enumerate().take(n) {
                    let (cx, cy, cz) = self.coords(i);
                    let dx = crate::util::min_image(cx as f64 - x, dims[0]);
                    let dy = crate::util::min_image(cy as f64 - y, dims[1]);
                    let dz = if self.dim == 3 { crate::util::min_image(cz as f64 - z, dims[2]) } else { 0.0 };
                    let r2 = dx * dx + dy * dy + dz * dz;
                    let e = (-r2 / (2.0 * sigma * sigma)).exp();
                    tot += e;
                    *v = Complex::new(e as f32, 0.0);
                }
                let s = (mass / tot) as f32;
                let mean = (mass / n as f64) as f32;
                for v in buf.iter_mut().take(n) {
                    v.re = v.re * s - mean;
                }
            }
        }
        self.fft.forward(&mut buf);
        for (v, k) in buf.iter_mut().zip(self.grav.poisson_kernel.iter()) {
            *v *= *k;
        }
        self.fft.inverse(&mut buf);
        if out_ext {
            for i in 0..n {
                self.grav.phi_ext[i] += buf[i].re;
            }
            self.grav.has_ext = true;
        } else {
            for i in 0..n {
                self.grav.phi[i] = buf[i].re;
            }
        }
        self.grav.gbuf = buf;
    }

    /// Paso 5 del paso de la materia: actualiza Φ a partir de la E resultante.
    pub(crate) fn gravity_after_step(&mut self) {
        match self.grav.mode {
            1 => self.poisson_solve(PoissonSource::Matter, false),
            2 => self.gravity_wave_step(),
            _ => {}
        }
        if self.grav.mode != 0 {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in 0..self.n {
                let p = self.grav.phi[i] + if self.grav.has_ext { self.grav.phi_ext[i] } else { 0.0 };
                lo = lo.min(p);
                hi = hi.max(p);
            }
            self.grav.phi_min = lo;
            self.grav.phi_max = hi;
        }
        if self.grav.exp_on && !self.grav.exp_stopped {
            self.grav.exp_tau += 1.0;
            let denom = 1.0 - 0.5 * self.grav.exp_c * self.grav.exp_tau;
            let mut a = if denom > 1e-6 { 1.0 / (denom * denom) } else { f64::INFINITY };
            if a >= self.grav.exp_a_max {
                a = self.grav.exp_a_max;
                self.grav.exp_stopped = true;
            }
            let phase = self.grav.gamma0 as f64 * a * (self.grav.phi_max - self.grav.phi_min) as f64;
            if phase >= EXPANSION_STOP_FRACTION * self.mass as f64 {
                self.grav.exp_stopped = true;
                self.set_message("Expansión detenida: γ_ef·(max Φ − min Φ) ya no es pequeña frente a m (§2.6).");
            }
            self.grav.exp_a = a;
        }
    }

    /// Variante onda: integrador de dos niveles con amortiguamiento η, local y causal.
    fn gravity_wave_step(&mut self) {
        let (w, h, d) = (self.w, self.h, self.d);
        let mean = self.mean_density() as f32;
        let c2 = self.grav.cg * self.grav.cg;
        let damp = 1.0 - self.grav.eta;
        let four_pi = 4.0 * std::f32::consts::PI;
        let phi = std::mem::take(&mut self.grav.phi);
        let mut prev = std::mem::take(&mut self.grav.phi_prev);
        // prev pasa a contener Φ_{t+1}; después se intercambian.
        for z in 0..d {
            let zm = if z == 0 { d - 1 } else { z - 1 };
            let zp = if z + 1 == d { 0 } else { z + 1 };
            for y in 0..h {
                let ym = if y == 0 { h - 1 } else { y - 1 };
                let yp = if y + 1 == h { 0 } else { y + 1 };
                for x in 0..w {
                    let xm = if x == 0 { w - 1 } else { x - 1 };
                    let xp = if x + 1 == w { 0 } else { x + 1 };
                    let i = (z * h + y) * w + x;
                    let p = phi[i];
                    let mut lap = phi[(z * h + y) * w + xm] + phi[(z * h + y) * w + xp] + phi[(z * h + ym) * w + x] + phi[(z * h + yp) * w + x] - 4.0 * p;
                    if d > 1 {
                        lap += phi[(zm * h + y) * w + x] + phi[(zp * h + y) * w + x] - 2.0 * p;
                    }
                    let src = four_pi * (self.density(i) - mean);
                    prev[i] = p + damp * (p - prev[i]) + c2 * (lap + src);
                }
            }
        }
        self.grav.phi = prev;
        self.grav.phi_prev = phi;
    }

    /// Φ total en la celda i (dinámico + externo).
    #[inline(always)]
    pub(crate) fn phi_total(&self, i: usize) -> f32 {
        let mut p = if self.grav.mode != 0 { self.grav.phi[i] } else { 0.0 };
        if self.grav.has_ext {
            p += self.grav.phi_ext[i];
        }
        p
    }

    /// Estado fundamental del límite continuo por evolución en tiempo imaginario (paso partido
    /// espectral): H = k²/(2m\*) + V − g·f(ρ) − γ·Φ, con Φ de Poisson (si la gravedad está activa).
    /// Parte del campo actual (proyectado sobre el espinor simétrico) y conserva su norma.
    pub(crate) fn imaginary_time_relax(&mut self, iterations: u32, dtau: f64) {
        let n = self.n;
        let nc = self.nc;
        let stride = 2 * nc;
        let m_star = self.band_mass().max(1e-6);
        let inv_sqrt = 1.0 / (nc as f64).sqrt();
        // φ = Σψ/√n_c en un búfer complejo f64 propio (solo se usa aquí, fuera de step)
        let mut phi: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let (sr, si) = self.sum_psi_at(i);
                Complex::new(sr as f64 * inv_sqrt, si as f64 * inv_sqrt)
            })
            .collect();
        let norm0: f64 = phi.iter().map(|v| v.norm_sqr()).sum();
        if norm0 <= 0.0 {
            return;
        }
        let mut fft = crate::fftn::FftN::<f64>::new(self.w, self.h, self.d);
        let mut kin = vec![0.0f64; n];
        for (i, k) in kin.iter_mut().enumerate() {
            let (x, y, z) = self.coords(i);
            let kx = crate::fftn::wavenumber(x, self.w);
            let ky = crate::fftn::wavenumber(y, self.h);
            let kz = if self.dim == 3 { crate::fftn::wavenumber(z, self.d) } else { 0.0 };
            *k = (-dtau * (kx * kx + ky * ky + kz * kz) / (2.0 * m_star)).exp();
        }
        let gamma = self.grav.gamma_eff() as f64;
        let use_grav = self.grav.mode != 0 || self.grav.has_ext;
        let mut pot = vec![0.0f64; n];
        let mut rho = vec![Complex::new(0.0, 0.0); n];
        let g = self.g as f64;
        let emax = self.e_max as f64;
        for _ in 0..iterations {
            // potencial V − g f(ρ) − γΦ
            let mean = norm0 / n as f64;
            if use_grav && self.grav.mode != 0 {
                for i in 0..n {
                    rho[i] = Complex::new(phi[i].norm_sqr() - mean, 0.0);
                }
                fft.forward(&mut rho);
                for i in 0..n {
                    rho[i] *= self.grav.poisson_kernel[i] as f64;
                }
                fft.inverse(&mut rho);
            }
            for i in 0..n {
                let r = phi[i].norm_sqr();
                let f = r / (1.0 + r / emax);
                let mut p = self.potential[i] as f64 - g * f;
                if use_grav {
                    let ph = if self.grav.mode != 0 { rho[i].re } else { 0.0 } + if self.grav.has_ext { self.grav.phi_ext[i] as f64 } else { 0.0 };
                    p -= gamma * ph;
                }
                pot[i] = p;
                phi[i] *= (-0.5 * dtau * p).exp();
            }
            fft.forward(&mut phi);
            for i in 0..n {
                phi[i] *= kin[i];
            }
            fft.inverse(&mut phi);
            for i in 0..n {
                phi[i] *= (-0.5 * dtau * pot[i]).exp();
                if self.has_walls && self.wall[i] != 0 {
                    phi[i] = Complex::new(0.0, 0.0);
                }
            }
            let nn: f64 = phi.iter().map(|v| v.norm_sqr()).sum();
            let s = (norm0 / nn).sqrt();
            for v in phi.iter_mut() {
                *v *= s;
            }
        }
        for i in 0..n {
            let b = i * stride;
            let (re, im) = ((phi[i].re * inv_sqrt) as f32, (phi[i].im * inv_sqrt) as f32);
            for k in 0..nc {
                self.buffer_read[b + 2 * k] = re;
                self.buffer_read[b + 2 * k + 1] = im;
            }
        }
        if self.grav.mode == 1 {
            self.poisson_solve(PoissonSource::Matter, false);
        }
    }
}

pub(crate) enum PoissonSource {
    Matter,
    Gaussian { x: f64, y: f64, z: f64, mass: f64, sigma: f64 },
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.6: 0 = desactivado (defecto), 1 = estática (FFT), 2 = onda. Al activarlo, Φ parte de la
    /// solución estática (sin transitorio).
    pub fn set_gravity(&mut self, mode: u32, gamma: f32) {
        if self.scenario == 2 && mode != 0 {
            self.set_message("En el escenario Dirac la gravedad de §2.6 está desactivada; la de los fermiones es la de Nordström (§2.15).");
            return;
        }
        if gamma.is_finite() {
            self.grav.gamma = gamma;
            if !self.grav.exp_on {
                self.grav.gamma0 = gamma;
            }
        }
        let mode = mode.min(2);
        if mode != 1 && self.grav.exp_on {
            self.grav.exp_on = false;
            self.set_message("La expansión solo funciona con la gravedad estática: se ha desactivado.");
        }
        let was = self.grav.mode;
        self.grav.mode = mode;
        if mode != 0 && (was == 0 || was != mode) {
            self.poisson_solve(PoissonSource::Matter, false);
            if mode == 2 {
                self.grav.phi_prev.copy_from_slice(&self.grav.phi);
            }
        }
        if mode == 0 {
            self.grav.phi.iter_mut().for_each(|v| *v = 0.0);
            self.grav.phi_prev.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    pub fn gravity_mode(&self) -> u32 {
        self.grav.mode
    }

    /// γ efectivo actual (γ₀·a con la expansión).
    pub fn gravity_gamma(&self) -> f32 {
        self.grav.gamma_eff()
    }

    /// c_g ≤ 1/√dim (se limita) y η ≥ 0; por defecto 0,5 y 0,02.
    pub fn set_gravity_speed(&mut self, cg: f32, damping: f32) {
        if cg.is_finite() {
            self.grav.cg = cg.clamp(0.0, 1.0 / (self.dim as f32).sqrt());
        }
        if damping.is_finite() {
            self.grav.eta = damping.clamp(0.0, 1.0);
        }
    }

    pub fn gravity_speed(&self) -> f32 {
        self.grav.cg
    }

    /// Φ (dinámico + externo) en una celda.
    pub fn gravity_potential_at(&self, x: i32, y: i32, z: i32) -> f32 {
        self.phi_total(self.wrap_cell(x, y, z))
    }

    /// max Φ − min Φ del último paso con gravedad.
    pub fn gravity_phi_range(&self) -> f32 {
        if self.grav.mode == 0 && !self.grav.has_ext {
            return 0.0;
        }
        if self.grav.mode == 0 {
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for i in 0..self.n {
                lo = lo.min(self.grav.phi_ext[i]);
                hi = hi.max(self.grav.phi_ext[i]);
            }
            return hi - lo;
        }
        self.grav.phi_max - self.grav.phi_min
    }

    /// Recalcula Φ estático con la materia actual (para medir sin dar un paso).
    pub fn refresh_gravity(&mut self) {
        if self.grav.mode != 0 {
            self.poisson_solve(PoissonSource::Matter, false);
        }
    }

    /// §2.6: universo de Einstein–de Sitter en tiempo superconforme (solo con la variante estática).
    /// Toma γ₀ = γ y ⟨E⟩ actuales y m\* de la banda, y pone a = 1.
    pub fn set_expansion(&mut self, on: bool, a_max: f32) {
        if !on {
            if self.grav.exp_on {
                self.grav.gamma = self.grav.gamma_eff();
            }
            self.grav.exp_on = false;
            return;
        }
        if self.grav.mode != 1 {
            self.set_message("La expansión necesita la gravedad estática (modo 1).");
            return;
        }
        let rho = self.mean_density();
        let m_star = self.band_mass().max(1e-9);
        self.grav.gamma0 = self.grav.gamma;
        self.grav.exp_c = (8.0 * PI * self.grav.gamma0 as f64 * rho / (3.0 * m_star)).sqrt();
        self.grav.exp_a = 1.0;
        self.grav.exp_tau = 0.0;
        self.grav.exp_a_max = if a_max.is_finite() && a_max > 1.0 { a_max as f64 } else { 10.0 };
        self.grav.exp_on = true;
        self.grav.exp_stopped = false;
    }

    /// a(τ) actual (1 si la expansión está apagada).
    pub fn scale_factor(&self) -> f64 {
        if self.grav.exp_on {
            self.grav.exp_a
        } else {
            1.0
        }
    }

    /// τ_∞ = 2/C de la expansión activa (pasos hasta la divergencia de a).
    pub fn expansion_tau_infinity(&self) -> f64 {
        if self.grav.exp_c > 0.0 {
            2.0 / self.grav.exp_c
        } else {
            f64::INFINITY
        }
    }

    pub fn expansion_stopped(&self) -> bool {
        self.grav.exp_stopped
    }

    /// Fuente externa congelada: Φ de una masa gaussiana (σ, masa) que se calcula una vez y se
    /// mantiene fija; se suma a Φ en la fase (necesita γ de `set_gravity`).
    pub fn add_frozen_source(&mut self, x: f32, y: f32, z: f32, mass: f32, sigma: f32) {
        if !(x.is_finite() && y.is_finite() && z.is_finite() && mass.is_finite() && sigma.is_finite()) || sigma <= 0.0 {
            return;
        }
        self.poisson_solve(PoissonSource::Gaussian { x: x as f64, y: y as f64, z: z as f64, mass: mass as f64, sigma: sigma as f64 }, true);
    }

    /// Campo externo Φ_ext = A·sin(2πx/W) (la fuerza lenta casi uniforme de §3.4); A = 0 lo quita.
    pub fn set_gravity_sine(&mut self, amplitude: f32) {
        let w = self.w as f64;
        for i in 0..self.n {
            let x = (i % self.w) as f64;
            self.grav.phi_ext[i] = (amplitude as f64 * (2.0 * PI * x / w).sin()) as f32;
        }
        self.grav.has_ext = amplitude != 0.0;
    }

    /// Quita las fuentes externas congeladas y el campo sinusoidal.
    pub fn clear_external_gravity(&mut self) {
        self.grav.phi_ext.iter_mut().for_each(|v| *v = 0.0);
        self.grav.has_ext = false;
    }

    /// Estado fundamental del límite continuo en tiempo imaginario (para cargar núcleos de
    /// Schrödinger–Poisson o de Thomas–Fermi en la caminata).
    pub fn relax_imaginary_time(&mut self, iterations: u32, dtau: f32) {
        self.imaginary_time_relax(iterations, dtau as f64);
    }

    /// Multiplica la amplitud dentro de una esfera por e^{ik·(r − c)} (empuja un solo objeto).
    #[allow(clippy::too_many_arguments)]
    pub fn boost_region(&mut self, cx: f32, cy: f32, cz: f32, radius: f32, kx: f32, ky: f32, kz: f32) {
        let stride = self.stride();
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let dx = crate::util::min_image(x as f64 - cx as f64, dims[0]);
            let dy = crate::util::min_image(y as f64 - cy as f64, dims[1]);
            let dz = if self.dim == 3 { crate::util::min_image(z as f64 - cz as f64, dims[2]) } else { 0.0 };
            if dx * dx + dy * dy + dz * dz > (radius as f64).powi(2) {
                continue;
            }
            let th = kx as f64 * dx + ky as f64 * dy + kz as f64 * dz;
            let (c, s) = (th.cos() as f32, th.sin() as f32);
            for k in 0..self.nc {
                let re = self.buffer_read[i * stride + 2 * k];
                let im = self.buffer_read[i * stride + 2 * k + 1];
                self.buffer_read[i * stride + 2 * k] = re * c - im * s;
                self.buffer_read[i * stride + 2 * k + 1] = re * s + im * c;
            }
        }
    }

    /// ⟨∂Φ/∂x_a⟩ ponderado con E alrededor de un punto (fuerza media sobre un objeto, por diferencias centradas).
    pub fn mean_gravity_gradient(&self, axis: u32, cx: f32, cy: f32, cz: f32, radius: f32) -> f64 {
        let a = (axis as usize).min(self.dim - 1);
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let (mut sw, mut sg) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let dx = crate::util::min_image(x as f64 - cx as f64, dims[0]);
            let dy = crate::util::min_image(y as f64 - cy as f64, dims[1]);
            let dz = if self.dim == 3 { crate::util::min_image(z as f64 - cz as f64, dims[2]) } else { 0.0 };
            if dx * dx + dy * dy + dz * dz > (radius as f64).powi(2) {
                continue;
            }
            let e = self.density(i) as f64;
            let (xi, yi, zi) = (x as i32, y as i32, z as i32);
            let (p, m) = match a {
                0 => (self.wrap_cell(xi + 1, yi, zi), self.wrap_cell(xi - 1, yi, zi)),
                1 => (self.wrap_cell(xi, yi + 1, zi), self.wrap_cell(xi, yi - 1, zi)),
                _ => (self.wrap_cell(xi, yi, zi + 1), self.wrap_cell(xi, yi, zi - 1)),
            };
            sg += e * 0.5 * (self.phi_total(p) as f64 - self.phi_total(m) as f64);
            sw += e;
        }
        if sw > 0.0 {
            sg / sw
        } else {
            0.0
        }
    }

    /// Perfil radial de E alrededor de un punto, en capas de anchura 1 (hasta 128 capas).
    pub fn compute_radial_profile(&mut self, cx: f32, cy: f32, cz: f32) -> u32 {
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let nb = self.radial.len() / 2;
        for v in self.radial.iter_mut() {
            *v = 0.0;
        }
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let dx = crate::util::min_image(x as f64 - cx as f64, dims[0]);
            let dy = crate::util::min_image(y as f64 - cy as f64, dims[1]);
            let dz = if self.dim == 3 { crate::util::min_image(z as f64 - cz as f64, dims[2]) } else { 0.0 };
            let b = (dx * dx + dy * dy + dz * dz).sqrt().round() as usize;
            if b < nb {
                self.radial[2 * b] += self.density(i) as f64;
                self.radial[2 * b + 1] += 1.0;
            }
        }
        let mut used = 0;
        for b in 0..nb {
            if self.radial[2 * b + 1] > 0.0 {
                self.radial[2 * b] /= self.radial[2 * b + 1];
                used = b + 1;
            }
        }
        used as u32
    }

    /// Pares [⟨E⟩ de la capa, número de celdas] de `compute_radial_profile`.
    pub fn radial_profile_ptr(&self) -> *const f64 {
        self.radial.as_ptr()
    }
}
