//! Producción clásica de bosones (§2.24 y §3.22). Los bosones del modelo estándar ya son campos
//! clásicos en esta app (§2.11 y §2.12); con ocupaciones grandes, su producción por resonancias o
//! inestabilidades es física clásica legítima (la aproximación clásico-estadística del
//! recalentamiento, con una semilla de ħ_ef/2 por modo).
//! - W y Z desde un Higgs homogéneo que oscila (resonancia paramétrica): la energía eléctrica por
//!   capa k̂² de W±, Z y fotón, y el exponente de Floquet de la ecuación lineal de cada capa con el
//!   φ(t) de la red, integrada con el mismo salto de rana.
//! - Higgs desde un condensado de W¹ uniforme (resonancia en ω_k = M_W).
//! - Gluones cargados desde un campo de color uniforme (Nielsen–Olesen) y W desde un campo
//!   electromagnético uniforme (Ambjørn–Olesen), con el flujo cuantizado B·L² = 4πm.
//! Convenios de la red: Ẽ^c = g·E^c (SU(2)) y E_B físico (U(1)_Y, la carga del Higgs es g′/2). La
//! componente neutra del Higgs ve la fase −(θ³ + g′·A)/2, así que el Z es (g·W³ + g′·B)/√(g² + g′²)
//! y el fotón (g′·W³ − g·B)/√(g² + g′²). Como el enlace de SU(2) avanza con θ̇ = +Ẽ y el de U(1)
//! con Ȧ = −E, en campos eléctricos E_Z = (Ẽ³ − g′·E_B)/√(g² + g′²) y E_γ = (g′·Ẽ³/g + g·E_B)/√(g² + g′²):
//! un fotón puro tiene E_B = Ẽ³/g′ y un Z puro, E_B = −g′·Ẽ³/g² (§3.22).

use crate::fftn::FftN;
use crate::rng::SplitMix64;
use crate::sun::Algebra;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct Bosons {
    /// δ₀ de la última `set_higgs_oscillation` (para el φ(t) de Floquet).
    pub delta0: f64,
    pub shell: Vec<f64>,
    /// Registro del condensado de W: cos(θ/2) y |Φ| medios antes de cada subpaso, y la amplitud del
    /// modo radial tras cada paso.
    pub record: bool,
    pub rec_chalf: Vec<f64>,
    pub rec_phi: Vec<f64>,
    pub rec_amp: Vec<f64>,
    pub wc_k: usize,
    pub wc_seed: f64,
}

impl Bosons {
    pub fn new() -> Self {
        Bosons { delta0: 0.0, shell: Vec::new(), record: false, rec_chalf: Vec::new(), rec_phi: Vec::new(), rec_amp: Vec::new(), wc_k: 1, wc_seed: 0.0 }
    }
}

/// Tasa de la envolvente: ajuste lineal de ln|y| en los máximos locales de |y| dentro de [t0, t1).
pub fn envelope_rate(ys: &[f64], t0: usize, t1: usize) -> f64 {
    let t1 = t1.min(ys.len().saturating_sub(1));
    let (mut xs, mut ls) = (Vec::new(), Vec::new());
    for t in t0.max(1)..t1 {
        let (a, b, c) = (ys[t - 1].abs(), ys[t].abs(), ys[t + 1].abs());
        if b >= a && b > c && b > 0.0 {
            xs.push(t as f64);
            ls.push(b.ln());
        }
    }
    if xs.len() < 2 {
        return f64::NAN;
    }
    crate::util::fit_linear(&xs, &ls).1
}

impl QuantumSpacetimeUniverse {
    fn bos_require_ew(&mut self) -> bool {
        if self.higgs.mode != 2 {
            self.set_message("Hace falta el escenario electrodébil: el doblete de Higgs de §2.12 (set_higgs con el modo 2).");
            return false;
        }
        true
    }

    /// Vacío limpio del doblete: Φ = (0, φ0), Π = 0, enlaces 1 y campos eléctricos y A a cero.
    fn bos_reset_ew(&mut self, phi0: f64) {
        let n = self.n;
        for i in 0..n {
            self.higgs.phi[4 * i..4 * i + 4].copy_from_slice(&[0.0, 0.0, phi0, 0.0]);
            self.higgs.pi[4 * i..4 * i + 4].copy_from_slice(&[0.0; 4]);
        }
        let id = Algebra::new(2).exp_minus_i(&[0.0; 3]);
        for l in 0..self.dim * n {
            self.ym.set_link(l, &id);
        }
        self.ym.e.iter_mut().for_each(|v| *v = 0.0);
        self.em.e.iter_mut().for_each(|v| *v = 0.0);
        self.em.a.iter_mut().for_each(|v| *v = 0.0);
        self.em.ph_valid = false;
        self.bos.record = false;
        self.t = 0;
    }

    /// Hace transversos (div = 0, con U = 1) Ẽ^c y E: resta el gradiente que resuelve la ley de Gauss.
    fn bos_transverse(&mut self) {
        let n = self.n;
        let dim = self.dim;
        let div = |f: &dyn Fn(usize) -> f64, i: usize, u: &QuantumSpacetimeUniverse| -> f64 {
            let mut d = 0.0;
            for a in 0..dim {
                d += f(a * n + i) - f(a * n + u.nb(i, a, -1));
            }
            d
        };
        let mut src3 = vec![0.0; 3 * n];
        for c in 0..3 {
            for i in 0..n {
                let e = &self.ym.e;
                src3[3 * i + c] = -div(&|l| e[3 * l + c], i, self);
            }
        }
        self.ym_add_gradient(&src3);
        let src: Vec<f64> = (0..n).map(|i| {
            let e = &self.em.e;
            -div(&|l| e[l], i, self)
        }).collect();
        self.em_add_gradient_field(&src);
    }

    /// Ángulos de un campo uniforme a lo largo de T³ en el plano xy con B·L_x·L_y = 4π·m: A_y = B·x y,
    /// en la costura, A_x(L_x − 1, y) = −B·L_x·y. Devuelve (ángulo en x, ángulo en y) por celda.
    fn bos_uniform_angles(&self, flux: u32) -> (Vec<f64>, Vec<f64>, f64) {
        let (w, h) = (self.w, self.h);
        let b = 4.0 * PI * flux as f64 / (w * h) as f64;
        let mut ax = vec![0.0; self.n];
        let mut ay = vec![0.0; self.n];
        for i in 0..self.n {
            let (x, y, _) = self.coords(i);
            ay[i] = b * x as f64;
            if x == w - 1 {
                ax[i] = -b * w as f64 * y as f64;
            }
        }
        (ax, ay, b)
    }

    /// Rotaciones aleatorias exp(−i·amp·n·T) de todos los enlaces SU(N), con Ẽ = 0 (Gauss exacta).
    fn bos_seed_links(&mut self, amp: f64, seed: u64) {
        let ng = self.ym.ng;
        let alg = Algebra::new(self.ym.n);
        let mut rng = SplitMix64::new(seed ^ 0xB05);
        for l in 0..self.dim * self.n {
            let th: Vec<f64> = (0..ng).map(|_| amp * rng.gauss()).collect();
            let u = self.ym.link(l).mul(&alg.exp_minus_i(&th));
            self.ym.set_link(l, &u);
        }
    }

    /// Antes de cada subpaso del Higgs: cos(θ/2) medio de los enlaces z y |Φ| medio.
    pub(crate) fn bos_record_substep(&mut self) {
        let n = self.n;
        let mut ch = 0.0;
        for i in 0..n {
            let u = self.ym.link(2 * n + i);
            ch += 0.5 * (u.get(0, 0) + u.get(1, 1)).re;
        }
        let mut ph = 0.0;
        for i in 0..n {
            ph += self.higgs.phi[4 * i..4 * i + 4].iter().map(|v| v * v).sum::<f64>().sqrt();
        }
        self.bos.rec_chalf.push(ch / n as f64);
        self.bos.rec_phi.push(ph / n as f64);
    }

    /// Tras cada paso: amplitud del modo radial (|Φ| − media) con k = k_index·2π/N_x a lo largo de x.
    pub(crate) fn bos_record_step(&mut self) {
        let a = self.bos_radial_amplitude(self.bos.wc_k);
        self.bos.rec_amp.push(a);
    }

    fn bos_radial_amplitude(&self, k_index: usize) -> f64 {
        let n = self.n;
        let k = 2.0 * PI * k_index as f64 / self.w as f64;
        let mods: Vec<f64> = (0..n).map(|i| self.higgs.phi[4 * i..4 * i + 4].iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
        let mean = mods.iter().sum::<f64>() / n as f64;
        let (mut c, mut s) = (0.0, 0.0);
        for i in 0..n {
            let (x, _, _) = self.coords(i);
            let d = mods[i] - mean;
            c += d * (k * x as f64).cos();
            s += d * (k * x as f64).sin();
        }
        2.0 * c.hypot(s) / n as f64
    }

    /// Campo eléctrico físico de una especie en el enlace l (0 = W¹, 1 = Z, 2 = fotón, 3 = W³).
    fn bos_species_e(&self, l: usize, species: u32) -> f64 {
        let (g, gp) = (self.higgs.g, self.higgs.gp);
        let nrm = (g * g + gp * gp).sqrt();
        let e3 = self.ym.e[3 * l + 2];
        match species {
            0 => self.ym.e[3 * l] / g,
            1 => (e3 - gp * self.em.e[l]) / nrm,
            2 => (gp * e3 / g + g * self.em.e[l]) / nrm,
            _ => e3 / g,
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.24, escenario electrodébil: Φ = (0, v + δ₀) uniforme, Π = 0, enlaces 1 y una semilla
    /// transversa de varianza ħ_ef/2 por enlace en los campos eléctricos físicos (Ẽ^c = g·E^c y E_B),
    /// proyectada para que la ley de Gauss se cumpla exactamente.
    pub fn set_higgs_oscillation(&mut self, delta0: f32, hbar_eff: f32, seed: u64) {
        if !self.bos_require_ew() {
            return;
        }
        let v = self.higgs.v;
        self.bos_reset_ew(v + delta0 as f64);
        let sig = (0.5 * hbar_eff as f64).max(0.0).sqrt();
        let g = self.higgs.g;
        let mut rng = SplitMix64::new(seed ^ 0x5EED);
        for x in self.ym.e.iter_mut() {
            *x = g * sig * rng.gauss();
        }
        for x in self.em.e.iter_mut() {
            *x = sig * rng.gauss();
        }
        self.bos_transverse();
        self.bos.delta0 = delta0 as f64;
        self.higgs_store_gauss_ref();
    }

    /// Energía eléctrica por capa k̂² = Σ_a 4·sen²(k_a/2) de W± (Ẽ¹, Ẽ²), Z y fotón (FFT de Ẽ y
    /// E_B): [número de capas, y por capa k̂², P_W, P_Z, P_γ].
    pub fn gauge_shell_power_ptr(&mut self) -> *const f64 {
        let (w, h, d, n, dim) = (self.w, self.h, self.d, self.n, self.dim);
        let mut fft = FftN::<f64>::new(w, h, d);
        let mut acc: BTreeMap<i64, [f64; 4]> = BTreeMap::new();
        let khat2: Vec<f64> = (0..n)
            .map(|i| {
                let (x, y, z) = self.coords(i);
                let s = |m: usize, l: usize| 4.0 * (PI * m as f64 / l as f64).sin().powi(2);
                s(x, w) + s(y, h) + if dim == 3 { s(z, d) } else { 0.0 }
            })
            .collect();
        let mut buf = vec![Complex::new(0.0, 0.0); n];
        for (col, species) in [(1usize, 0u32), (1, 4), (2, 1), (3, 2)] {
            for a in 0..dim {
                for i in 0..n {
                    let l = a * n + i;
                    let v = if species == 4 { self.ym.e[3 * l + 1] / self.higgs.g } else { self.bos_species_e(l, species) };
                    buf[i] = Complex::new(v, 0.0);
                }
                fft.forward(&mut buf);
                for i in 0..n {
                    let key = (khat2[i] * 1e8).round() as i64;
                    let e = acc.entry(key).or_insert([khat2[i], 0.0, 0.0, 0.0]);
                    e[col] += 0.5 * buf[i].norm_sqr() / n as f64;
                }
            }
        }
        let mut out = vec![acc.len() as f64];
        for (_, v) in acc {
            out.extend_from_slice(&v);
        }
        self.bos.shell = out;
        self.bos.shell.as_ptr()
    }

    /// Reparto de la energía: [total, Higgs (|Π|² + κ|DΦ|² + V), eléctrica de W±, de Z y de fotón].
    pub fn gauge_energy_split(&mut self) -> Vec<f64> {
        if self.higgs.mode != 2 {
            return vec![0.0; 5];
        }
        let total = self.higgs_compute_energy_density();
        let (n, dim) = (self.n, self.dim);
        let (v2, lam, c2) = (self.higgs.v * self.higgs.v, self.higgs.lambda, self.em.kappa);
        let mut eh = 0.0;
        for i in 0..n {
            let p2: f64 = self.higgs.pi[4 * i..4 * i + 4].iter().map(|v| v * v).sum();
            let f2: f64 = self.higgs.phi[4 * i..4 * i + 4].iter().map(|v| v * v).sum();
            let mut grad = 0.0;
            for a in 0..dim {
                grad += self.higgs_grad2(i, a);
            }
            eh += p2 + c2 * grad + lam * (f2 - v2).powi(2);
        }
        let mut es = [0.0f64; 3];
        for l in 0..dim * n {
            let w1 = self.ym.e[3 * l] / self.higgs.g;
            let w2 = self.ym.e[3 * l + 1] / self.higgs.g;
            es[0] += 0.5 * (w1 * w1 + w2 * w2);
            es[1] += 0.5 * self.bos_species_e(l, 1).powi(2);
            es[2] += 0.5 * self.bos_species_e(l, 2).powi(2);
        }
        vec![total, eh, es[0], es[1], es[2]]
    }

    /// §2.24: exponente de Floquet de la capa k̂² (0 = W, 1 = Z, 2 = fotón) con el φ(t) homogéneo de
    /// la red (Φ = (0, v + δ₀) en reposo), integrado con el mismo salto de rana de s subpasos sobre un
    /// número entero de períodos (el que mejor cae en pasos enteros entre 15 y 60).
    pub fn floquet_exponent(&self, khat2: f32, species: u32) -> f64 {
        let s = self.em.substeps.max(1) as usize;
        let dt = 1.0 / s as f64;
        let (kap, g, gp, v, lam) = (self.em.kappa, self.higgs.g, self.higgs.gp, self.higgs.v, self.higgs.lambda);
        let m2 = |phi: f64| -> f64 {
            match species {
                0 => kap * g * g * phi * phi / 2.0,
                1 => kap * (g * g + gp * gp) * phi * phi / 2.0,
                _ => 0.0,
            }
        };
        let phi0 = v + self.bos.delta0;
        // período de φ(t): máximos (π cambia de + a −)
        let (mut phi, mut pi) = (phi0, 0.0f64);
        let mut maxima: Vec<f64> = Vec::new();
        let mut t = 0.0;
        while maxima.len() < 12 && t < 1e5 {
            let pold = pi;
            pi += dt * (-2.0 * lam * (phi * phi - v * v) * phi);
            phi += dt * pi;
            t += dt;
            if pold > 0.0 && pi <= 0.0 {
                maxima.push(t - dt * pi / (pi - pold));
            }
        }
        if maxima.len() < 3 {
            return 0.0;
        }
        let period = (maxima[maxima.len() - 1] - maxima[0]) / (maxima.len() - 1) as f64;
        let mut best = (f64::INFINITY, 20usize);
        for np in 15..=60 {
            let x = np as f64 * period;
            let err = (x - x.round()).abs();
            if err < best.0 {
                best = (err, np);
            }
        }
        let nsteps = (best.1 as f64 * period).round() as usize;
        // monodromía de (W, E): E ← E − dt·(κk̂² + M²(φ))·W, W ← W + dt·E, con φ en el mismo salto de rana
        let mut m = [[1.0f64, 0.0], [0.0, 1.0]];
        for col in 0..2 {
            let (mut w, mut e) = if col == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
            let (mut phi, mut pi) = (phi0, 0.0f64);
            for _ in 0..nsteps * s {
                let om2 = kap * khat2 as f64 + m2(phi);
                pi += dt * (-2.0 * lam * (phi * phi - v * v) * phi);
                e -= dt * om2 * w;
                phi += dt * pi;
                w += dt * e;
            }
            m[0][col] = w;
            m[1][col] = e;
        }
        let tr = m[0][0] + m[1][1];
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let disc = tr * tr / 4.0 - det;
        let rho = if disc > 0.0 { (tr.abs() / 2.0 + disc.sqrt()).max(1.0) } else { det.abs().sqrt().max(1.0) };
        rho.ln() / nsteps as f64
    }

    /// Siembra un solo modo de gauge transverso en los enlaces y, con k = k_index·2π/N_x a lo largo
    /// de x: 0 = W¹, 1 = Z puro, 2 = fotón puro, 3 = W³ solo; `amplitude` es la del campo eléctrico
    /// físico. Gauss sigue exacta (div = 0).
    pub fn seed_gauge_mode(&mut self, species: u32, k_index: u32, amplitude: f32) {
        if !self.bos_require_ew() {
            return;
        }
        let n = self.n;
        let (g, gp) = (self.higgs.g, self.higgs.gp);
        let nrm = (g * g + gp * gp).sqrt();
        let k = 2.0 * PI * k_index as f64 / self.w as f64;
        for i in 0..n {
            let (x, _, _) = self.coords(i);
            let f = amplitude as f64 * (k * x as f64).cos();
            let l = n + i;
            match species {
                0 => self.ym.e[3 * l] += g * f,
                1 => {
                    self.ym.e[3 * l + 2] += g * g * f / nrm;
                    self.em.e[l] -= gp * f / nrm;
                }
                2 => {
                    self.ym.e[3 * l + 2] += g * gp * f / nrm;
                    self.em.e[l] += g * f / nrm;
                }
                _ => self.ym.e[3 * l + 2] += g * f,
            }
        }
        self.higgs_store_gauss_ref();
    }

    /// Amplitud de Fourier (k = k_index·2π/N_x a lo largo de x) del campo eléctrico físico de una
    /// especie en los enlaces y.
    pub fn gauge_mode_amplitude(&self, species: u32, k_index: u32) -> f64 {
        let n = self.n;
        let k = 2.0 * PI * k_index as f64 / self.w as f64;
        let (mut c, mut s) = (0.0, 0.0);
        for i in 0..n {
            let (x, _, _) = self.coords(i);
            let e = self.bos_species_e(n + i, species);
            c += e * (k * x as f64).cos();
            s += e * (k * x as f64).sin();
        }
        2.0 * c.hypot(s) / n as f64
    }

    /// §2.24: el vacío Φ = (0, v) con un W¹ uniforme en los enlaces z (U_z = exp(−iθ₀T¹), Ẽ = 0) y un
    /// modo radial del Higgs con k = higgs_mode_k·2π/N_x a lo largo de x y amplitud `seed`. Empieza a
    /// registrar θ(t), φ(t) y la amplitud del modo para la predicción lineal.
    pub fn set_w_condensate(&mut self, theta0: f32, higgs_mode_k: u32, seed: f32) {
        if !self.bos_require_ew() {
            return;
        }
        let v = self.higgs.v;
        self.bos_reset_ew(v);
        let n = self.n;
        let u = Algebra::new(2).exp_minus_i(&[theta0 as f64, 0.0, 0.0]);
        for i in 0..n {
            self.ym.set_link(2 * n + i, &u);
        }
        let k = 2.0 * PI * higgs_mode_k as f64 / self.w as f64;
        for i in 0..n {
            let (x, _, _) = self.coords(i);
            self.higgs.phi[4 * i + 2] = v + seed as f64 * (k * x as f64).cos();
        }
        self.higgs_store_gauss_ref();
        self.bos.wc_k = higgs_mode_k as usize;
        self.bos.wc_seed = seed as f64;
        self.bos.rec_chalf.clear();
        self.bos.rec_phi.clear();
        self.bos.rec_amp = vec![self.bos_radial_amplitude(higgs_mode_k as usize)];
        self.bos.record = true;
    }

    /// Serie registrada del condensado de W: [pasos, amplitud del modo en la red (pasos + 1 valores),
    /// la de la integración lineal con la serie θ(t), φ(t) de la red (pasos + 1), cos(θ/2) por paso].
    /// La ecuación lineal es ḧ = −[κk̂² + 2λ(3φ² − v²) + 2κ(1 − cos(θ/2))]·h, con el mismo salto de rana.
    pub fn w_condensate_series(&self) -> Vec<f64> {
        let s = self.em.substeps.max(1) as usize;
        let dt = 1.0 / s as f64;
        let steps = self.bos.rec_amp.len().saturating_sub(1);
        let (kap, v, lam) = (self.em.kappa, self.higgs.v, self.higgs.lambda);
        let k = 2.0 * PI * self.bos.wc_k as f64 / self.w as f64;
        let kh2 = 4.0 * (0.5 * k).sin().powi(2);
        let (mut h, mut p) = (self.bos.wc_seed, 0.0f64);
        let mut lin = vec![h.abs()];
        let mut ch = Vec::with_capacity(steps);
        for t in 0..steps {
            for j in 0..s {
                let idx = t * s + j;
                if idx >= self.bos.rec_phi.len() {
                    break;
                }
                let (c, f) = (self.bos.rec_chalf[idx], self.bos.rec_phi[idx]);
                let om2 = kap * kh2 + 2.0 * lam * (3.0 * f * f - v * v) + 2.0 * kap * (1.0 - c);
                p -= dt * om2 * h;
                h += dt * p;
            }
            lin.push(h.abs());
            ch.push(self.bos.rec_chalf.get(t * s).copied().unwrap_or(1.0));
        }
        let mut out = vec![steps as f64];
        out.extend_from_slice(&self.bos.rec_amp);
        out.extend_from_slice(&lin);
        out.extend_from_slice(&ch);
        out
    }

    /// Tasas de la envolvente entre los pasos t0 y t1: [en la red, en la integración lineal].
    pub fn w_condensate_rates(&self, t0: u32, t1: u32) -> Vec<f64> {
        let s = self.w_condensate_series();
        let steps = s[0] as usize;
        let lat = &s[1..steps + 2];
        let lin = &s[steps + 2..2 * steps + 3];
        vec![envelope_rate(lat, t0 as usize, t1 as usize), envelope_rate(lin, t0 as usize, t1 as usize)]
    }

    /// Deja de registrar el condensado de W.
    pub fn stop_w_condensate_record(&mut self) {
        self.bos.record = false;
    }

    /// §2.24: SU(N) puro (escenario materia, sin materia), B uniforme a lo largo de T³ en el plano xy
    /// (B·L_x·L_y = 4π·m, costura retorcida) y rotaciones aleatorias de los enlaces de amplitud
    /// `seed_amplitude` con Ẽ = 0 (la ley de Gauss se cumple exactamente).
    pub fn set_uniform_colour_b(&mut self, flux_quanta: u32, seed_amplitude: f32) {
        if self.scenario != 0 || self.dim != 3 {
            self.set_message("El campo de color uniforme es del escenario materia en 3D (SU(N) puro).");
            return;
        }
        if self.ncol < 2 {
            self.resize_colour(2);
        }
        self.higgs.mode = 0;
        self.clear();
        self.em.mode = 2;
        self.em.substeps = self.em.substeps.max(2);
        let (ax, ay, _) = self.bos_uniform_angles(flux_quanta);
        let n = self.n;
        let alg = Algebra::new(self.ym.n);
        let mut th = vec![0.0; self.ym.ng];
        for i in 0..n {
            th[2] = ax[i];
            self.ym.set_link(i, &alg.exp_minus_i(&th));
            th[2] = ay[i];
            self.ym.set_link(n + i, &alg.exp_minus_i(&th));
            th[2] = 0.0;
            self.ym.set_link(2 * n + i, &alg.exp_minus_i(&th));
        }
        self.ym.e.iter_mut().for_each(|v| *v = 0.0);
        self.bos_seed_links(seed_amplitude as f64, 0x6C00);
        self.ym_store_gauss_ref();
        self.t = 0;
    }

    /// Energía eléctrica ½Σ(Ẽ^c)² de las componentes cargadas respecto a T³ (SU(2): 1 y 2; SU(3):
    /// 1, 2, 4, 5, 6 y 7), en unidades de la red.
    pub fn charged_gluon_energy(&self) -> f64 {
        let ng = self.ym.ng;
        let charged: &[usize] = if ng == 3 { &[0, 1] } else { &[0, 1, 3, 4, 5, 6] };
        let mut e = 0.0;
        for l in 0..self.dim * self.n {
            for &c in charged {
                e += 0.5 * self.ym.e[l * ng + c].powi(2);
            }
        }
        e
    }

    /// §2.24: escenario electrodébil en el vacío Φ = (0, v) con un campo electromagnético uniforme
    /// (ángulo θ en T³ y A_Y = −θ/g′, que la componente neutra del Higgs no ve), con B·L_x·L_y = 4π·m,
    /// la U(1)_Y compacta (c = g′/2) y rotaciones aleatorias de los enlaces SU(2) de amplitud
    /// `seed_amplitude`.
    pub fn set_uniform_em_b(&mut self, flux_quanta: u32, seed_amplitude: f32) {
        if !self.bos_require_ew() {
            return;
        }
        let v = self.higgs.v;
        self.bos_reset_ew(v);
        self.em.compact = true;
        let (ax, ay, _) = self.bos_uniform_angles(flux_quanta);
        let n = self.n;
        let alg = Algebra::new(2);
        let gp = self.higgs.gp;
        for i in 0..n {
            self.ym.set_link(i, &alg.exp_minus_i(&[0.0, 0.0, ax[i]]));
            self.ym.set_link(n + i, &alg.exp_minus_i(&[0.0, 0.0, ay[i]]));
            self.em.a[i] = -ax[i] / gp;
            self.em.a[n + i] = -ay[i] / gp;
        }
        self.bos_seed_links(seed_amplitude as f64, 0xA0B);
        self.higgs_store_gauss_ref();
    }

    /// Energía eléctrica de los W cargados, ½Σ(Ẽ¹² + Ẽ²²)/g².
    pub fn charged_w_energy(&self) -> f64 {
        let g2 = self.higgs.g * self.higgs.g;
        let mut e = 0.0;
        for l in 0..self.dim * self.n {
            e += 0.5 * (self.ym.e[3 * l].powi(2) + self.ym.e[3 * l + 1].powi(2)) / g2;
        }
        e
    }

    /// Máximo de |Φ₁|, la componente cargada del Higgs (el condensado de Ambjørn–Olesen).
    pub fn charged_higgs_max(&self) -> f64 {
        (0..self.n).map(|i| self.higgs.phi[4 * i].hypot(self.higgs.phi[4 * i + 1])).fold(0.0, f64::max)
    }

    /// Máximo de |Ẽ|, |E| y |Π|: con un fondo estático y sin semilla, la fuerza tras un paso.
    pub fn field_max_abs(&self) -> f64 {
        let m = |v: &[f64]| v.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        m(&self.ym.e).max(m(&self.em.e)).max(m(&self.higgs.pi))
    }
}
