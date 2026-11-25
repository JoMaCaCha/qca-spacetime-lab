//! Mecanismo de Higgs clásico (§2.12): un campo escalar aparte, acoplado a los enlaces de §2.9
//! (U(1)) y de §2.11 (SU(2)). La materia de la caminata queda en pausa mientras está activo.
//! - Abeliano: φ ∈ ℂ con carga q, D_aφ = e^{−iqA_a(r)}·φ(r+â) − φ(r).
//! - Doblete SU(2)×U(1)_Y: Φ ∈ ℂ², D_aΦ = e^{−i(g′/2)A_a(r)}·U_a(r)†·Φ(r+â) − Φ(r).
//! - Adjunto (3D, monopolo de 't Hooft–Polyakov): Φ = φ^c·T^c, D_aΦ = U_a(r)†·Φ(r+â)·U_a(r) − Φ(r).
//! Leapfrog con Δt = 1/s: π ← π + Δt·F_φ; E ← E + Δt·(F_gauge − q·J); φ ← φ + Δt·π; A ← A − Δt·E
//! (U ← U·exp(−iΔt·Ẽ·T)). Las leyes de Gauss (div E + 2q·Im(φ*π) y sus versiones no abelianas)
//! se conservan exactamente. Es el Higgs clásico real, puesto a mano, como en el modelo estándar.

use crate::sun::{Mat, C};
use crate::util::min_image;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct HiggsField {
    /// 0 = desactivado, 1 = abeliano, 2 = doblete electrodébil, 3 = adjunto.
    pub mode: u32,
    pub v: f64,
    pub lambda: f64,
    /// Sesgo lineal explícito −ε·Re φ en el potencial abeliano (falso vacío; solo con q = 0).
    pub bias: f64,
    /// Doblete: acoplamiento SU(2); adjunto: g.
    pub g: f64,
    /// Doblete: acoplamiento U(1)_Y (la carga del Higgs es g′/2).
    pub gp: f64,
    /// f64 por celda en φ y π: 2 (abeliano), 4 (doblete), 3 (adjunto, real).
    pub nf: usize,
    pub phi: Vec<f64>,
    pub pi: Vec<f64>,
    /// Ley de Gauss de referencia (abeliano: n; doblete: 3n de SU(2) y n de U(1); adjunto: 3n).
    pub g_ref: Vec<f64>,
    pub pinned: Vec<u8>,
    pub pin_val: Vec<f64>,
    pub has_pins: bool,
    /// Enlaces congelados (capa de borde del monopolo): ni U ni Ẽ cambian.
    pub frozen_link: Vec<u8>,
    pub has_frozen: bool,
    /// Flujo fijo de cuerdas de Dirac por plaqueta (planos xy, xz, yz): monopolos externos.
    pub string_b: Vec<f64>,
    pub has_string: bool,
    /// cos y sin de q·A por enlace (trabajo de cada subpaso).
    pub ph: Vec<f64>,
    pub vortices: Vec<f32>,
    pub energy_density: Vec<f32>,
}

impl HiggsField {
    pub fn new() -> Self {
        HiggsField {
            mode: 0,
            v: 1.0,
            lambda: 0.00125,
            bias: 0.0,
            g: 0.3,
            gp: 0.18,
            nf: 2,
            phi: Vec::new(),
            pi: Vec::new(),
            g_ref: Vec::new(),
            pinned: Vec::new(),
            pin_val: Vec::new(),
            has_pins: false,
            frozen_link: Vec::new(),
            has_frozen: false,
            string_b: Vec::new(),
            has_string: false,
            ph: Vec::new(),
            vortices: Vec::with_capacity(3 * 256),
            energy_density: Vec::new(),
        }
    }
}

/// Índice del plano (a, b) con a < b: xy = 0, xz = 1, yz = 2.
#[inline(always)]
fn plane(a: usize, b: usize) -> usize {
    a + b - 1
}

#[inline(always)]
fn cx(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

impl QuantumSpacetimeUniverse {
    /// c² = κ: el escalar y el fotón comparten la velocidad c = √κ (§2.12).
    #[inline(always)]
    fn higgs_c2(&self) -> f64 {
        self.em.kappa
    }

    /// Carga U(1) que ve el Higgs: q (abeliano) o g′/2 (doblete).
    #[inline(always)]
    fn higgs_q(&self) -> f64 {
        self.em.q
    }

    /// B total de la plaqueta (a, b) en r: B[A] más el flujo de las cuerdas de Dirac.
    #[inline(always)]
    pub(crate) fn higgs_bt(&self, i: usize, a: usize, b: usize) -> f64 {
        let mut bb = self.plaquette_b(i, a, b);
        if self.higgs.has_string {
            let (lo, hi, sg) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
            bb += sg * self.higgs.string_b[plane(lo, hi) * self.n + i];
        }
        bb
    }

    /// Fuerza magnética κ·Σ_b [B(r) − B(r − b̂)] sobre E_a(r), con la energía compacta si toca.
    fn higgs_curl(&self, i: usize, a: usize) -> f64 {
        let q = self.higgs_q();
        let mut f = 0.0;
        for b in 0..self.dim {
            if b == a {
                continue;
            }
            let im = self.nb(i, b, -1);
            let (b0, b1) = (self.higgs_bt(i, a, b), self.higgs_bt(im, a, b));
            f += if self.em.compact && q.abs() > 1e-15 { ((q * b0).sin() - (q * b1).sin()) / q } else { b0 - b1 };
        }
        self.em.kappa * f
    }

    fn higgs_refresh_phases(&mut self) {
        let q = self.higgs_q();
        let links = self.dim * self.n;
        if self.higgs.ph.len() != 2 * links {
            self.higgs.ph = vec![0.0; 2 * links];
        }
        for l in 0..links {
            let (s, c) = (q * self.em.a[l]).sin_cos();
            self.higgs.ph[2 * l] = c;
            self.higgs.ph[2 * l + 1] = s;
        }
    }

    /// Reserva φ y π (y los auxiliares) para el modo dado.
    pub(crate) fn higgs_alloc(&mut self, nf: usize) {
        let n = self.n;
        self.higgs.nf = nf;
        self.higgs.phi = vec![0.0; nf * n];
        self.higgs.pi = vec![0.0; nf * n];
        self.higgs.pinned = vec![0; n];
        self.higgs.pin_val = vec![0.0; nf * n];
        self.higgs.has_pins = false;
        self.higgs.frozen_link = vec![0; self.dim * n];
        self.higgs.has_frozen = false;
        self.higgs.string_b = vec![0.0; 3 * n];
        self.higgs.has_string = false;
        self.higgs.energy_density = vec![0.0; n];
        self.higgs.g_ref = vec![0.0; if nf == 4 { 4 * n } else if nf == 3 { 3 * n } else { n }];
    }

    // ------------------------------------------------------------------ paso

    /// Un paso del Higgs con s subpasos (la materia queda en pausa).
    pub(crate) fn higgs_step(&mut self) {
        let s = self.em.substeps.max(1);
        let dt = 1.0 / s as f64;
        for _ in 0..s {
            if self.bos.record {
                self.bos_record_substep();
            }
            match self.higgs.mode {
                1 => self.higgs_abelian_substep(dt),
                2 => self.higgs_doublet_substep(dt),
                3 => self.higgs_adjoint_substep(dt),
                _ => {}
            }
        }
        if self.bos.record {
            self.bos_record_step();
        }
        if self.higgs.mode >= 2 {
            self.ym.reunit_counter = self.ym.reunit_counter.wrapping_add(1);
            if self.ym.reunit_counter % 100 == 0 {
                for l in 0..self.dim * self.n {
                    let mut u = self.ym.link(l);
                    u.reunitarize();
                    self.ym.set_link(l, &u);
                }
            }
        }
        self.em.ph_valid = false;
        self.t = self.t.wrapping_add(1);
    }

    fn higgs_apply_pins(&mut self) {
        if !self.higgs.has_pins {
            return;
        }
        let nf = self.higgs.nf;
        for i in 0..self.n {
            match self.higgs.pinned[i] {
                0 => {}
                1 => {
                    for k in 0..nf {
                        self.higgs.phi[i * nf + k] = self.higgs.pin_val[i * nf + k];
                        self.higgs.pi[i * nf + k] = 0.0;
                    }
                }
                _ => {
                    // solo la fase (abeliano): φ = |φ|·u con u fijo y π paralelo a u
                    let (ur, ui) = (self.higgs.pin_val[2 * i], self.higgs.pin_val[2 * i + 1]);
                    let a = (self.higgs.phi[2 * i] * ur + self.higgs.phi[2 * i + 1] * ui).max(0.0);
                    self.higgs.phi[2 * i] = a * ur;
                    self.higgs.phi[2 * i + 1] = a * ui;
                    let p = self.higgs.pi[2 * i] * ur + self.higgs.pi[2 * i + 1] * ui;
                    self.higgs.pi[2 * i] = p * ur;
                    self.higgs.pi[2 * i + 1] = p * ui;
                }
            }
        }
    }

    fn higgs_abelian_substep(&mut self, dt: f64) {
        let n = self.n;
        let dim = self.dim;
        let q = self.higgs_q();
        let c2 = self.higgs_c2();
        let v2 = self.higgs.v * self.higgs.v;
        let lam = self.higgs.lambda;
        let damp = 1.0 - self.em.eta;
        self.higgs_refresh_phases();
        let ph = std::mem::take(&mut self.higgs.ph);
        let phi = std::mem::take(&mut self.higgs.phi);
        let mut pi = std::mem::take(&mut self.higgs.pi);
        // π ← π + Δt·[c²·Σ_a (e^{−iqA_a(r)}φ(r+â) + e^{iqA_a(r−â)}φ(r−â) − 2φ) − 2λ(|φ|² − v²)φ]
        let dd = 2.0 * dim as f64;
        for i in 0..n {
            let (xr, xi) = (phi[2 * i], phi[2 * i + 1]);
            let (mut lr, mut li) = (-dd * xr, -dd * xi);
            for a in 0..dim {
                let ip = self.nb(i, a, 1);
                let im = self.nb(i, a, -1);
                let l = a * n + i;
                let (c, s) = (ph[2 * l], ph[2 * l + 1]);
                let (yr, yi) = (phi[2 * ip], phi[2 * ip + 1]);
                lr += c * yr + s * yi;
                li += c * yi - s * yr;
                let l2 = a * n + im;
                let (c2_, s2_) = (ph[2 * l2], ph[2 * l2 + 1]);
                let (zr, zi) = (phi[2 * im], phi[2 * im + 1]);
                lr += c2_ * zr - s2_ * zi;
                li += c2_ * zi + s2_ * zr;
            }
            let m2 = xr * xr + xi * xi - v2;
            pi[2 * i] = (pi[2 * i] + dt * (c2 * lr - 2.0 * lam * m2 * xr + 0.5 * self.higgs.bias)) * damp;
            pi[2 * i + 1] = (pi[2 * i + 1] + dt * (c2 * li - 2.0 * lam * m2 * xi)) * damp;
        }
        // E_a ← E_a + Δt·[κ·rot B − q·J_a], J_a = 2c²·Im(φ*(r)·e^{−iqA_a(r)}·φ(r+â))
        let mut e_new = std::mem::take(&mut self.em.j);
        for a in 0..dim {
            for i in 0..n {
                let l = a * n + i;
                let ip = self.nb(i, a, 1);
                let (c, s) = (ph[2 * l], ph[2 * l + 1]);
                let (xr, xi) = (phi[2 * i], phi[2 * i + 1]);
                let (yr, yi) = (phi[2 * ip], phi[2 * ip + 1]);
                let (wr, wi) = (c * yr + s * yi, c * yi - s * yr);
                let j = 2.0 * c2 * (xr * wi - xi * wr);
                e_new[l] = (self.em.e[l] + dt * (self.higgs_curl(i, a) - q * j)) * damp;
            }
        }
        std::mem::swap(&mut self.em.e, &mut e_new);
        self.em.j = e_new;
        let mut phi = phi;
        for k in 0..2 * n {
            phi[k] += dt * pi[k];
        }
        for l in 0..dim * n {
            self.em.a[l] -= dt * self.em.e[l];
        }
        self.higgs.phi = phi;
        self.higgs.pi = pi;
        self.higgs.ph = ph;
        self.higgs_apply_pins();
    }

    /// e^{−iqA_a(r)}·U_a(r)†·Φ(r+â) para el doblete.
    #[inline(always)]
    fn doublet_w(&self, i: usize, a: usize, phi: &[f64]) -> [C; 3] {
        let n = self.n;
        let ip = self.nb(i, a, 1);
        let l = a * n + i;
        let u = self.ym.link(l);
        let v = [cx(phi[4 * ip], phi[4 * ip + 1]), cx(phi[4 * ip + 2], phi[4 * ip + 3]), cx(0.0, 0.0)];
        let w = u.apply_dag(&v);
        let e = cx(self.higgs.ph[2 * l], -self.higgs.ph[2 * l + 1]);
        [w[0] * e, w[1] * e, cx(0.0, 0.0)]
    }

    pub(crate) fn higgs_doublet_substep(&mut self, dt: f64) {
        let n = self.n;
        let dim = self.dim;
        let q = self.higgs_q();
        let c2 = self.higgs_c2();
        let g2 = self.higgs.g * self.higgs.g;
        let v2 = self.higgs.v * self.higgs.v;
        let lam = self.higgs.lambda;
        let damp = 1.0 - self.em.eta;
        self.higgs_refresh_phases();
        let phi = std::mem::take(&mut self.higgs.phi);
        let mut pi = std::mem::take(&mut self.higgs.pi);
        let dd = 2.0 * dim as f64;
        for i in 0..n {
            let x = [cx(phi[4 * i], phi[4 * i + 1]), cx(phi[4 * i + 2], phi[4 * i + 3])];
            let mut lap = [x[0] * (-dd), x[1] * (-dd)];
            for a in 0..dim {
                let w = self.doublet_w(i, a, &phi);
                // e^{+iqA_a(r−â)}·U_a(r−â)·Φ(r−â)
                let im = self.nb(i, a, -1);
                let l2 = a * n + im;
                let u = self.ym.link(l2);
                let z = u.apply(&[cx(phi[4 * im], phi[4 * im + 1]), cx(phi[4 * im + 2], phi[4 * im + 3]), cx(0.0, 0.0)]);
                let e = cx(self.higgs.ph[2 * l2], self.higgs.ph[2 * l2 + 1]);
                for k in 0..2 {
                    lap[k] += w[k] + z[k] * e;
                }
            }
            let m2 = x[0].norm_sqr() + x[1].norm_sqr() - v2;
            for k in 0..2 {
                let f = lap[k] * c2 - x[k] * (2.0 * lam * m2);
                pi[4 * i + 2 * k] = (pi[4 * i + 2 * k] + dt * f.re) * damp;
                pi[4 * i + 2 * k + 1] = (pi[4 * i + 2 * k + 1] + dt * f.im) * damp;
            }
        }
        // fuerzas sobre los enlaces con los campos en t
        self.ym_compute_force();
        let mut e_b = std::mem::take(&mut self.em.j);
        let mut jt = [0.0f64; 8];
        for a in 0..dim {
            for i in 0..n {
                let l = a * n + i;
                let w = self.doublet_w(i, a, &phi);
                let x = [cx(phi[4 * i], phi[4 * i + 1]), cx(phi[4 * i + 2], phi[4 * i + 3]), cx(0.0, 0.0)];
                // J^c = 2c²·Im(Φ†·T^c·w); J^B = 2c²·Im(Φ†·w)
                let mut jb = 0.0;
                for k in 0..2 {
                    jb += (x[k].conj() * w[k]).im;
                }
                for c in 0..3 {
                    let mut s = cx(0.0, 0.0);
                    for &(r, cc, t) in &self.ym.alg.sparse[c] {
                        s += x[r].conj() * t * w[cc];
                    }
                    jt[c] = 2.0 * c2 * s.im;
                }
                for c in 0..3 {
                    let o = l * 3 + c;
                    self.ym.e[o] = (self.ym.e[o] + dt * (self.ym.force[o] - g2 * jt[c])) * damp;
                }
                e_b[l] = (self.em.e[l] + dt * (self.higgs_curl(i, a) - q * 2.0 * c2 * jb)) * damp;
            }
        }
        std::mem::swap(&mut self.em.e, &mut e_b);
        self.em.j = e_b;
        let mut phi = phi;
        for k in 0..4 * n {
            phi[k] += dt * pi[k];
        }
        for l in 0..dim * n {
            self.em.a[l] -= dt * self.em.e[l];
        }
        self.higgs_su2_link_update(dt);
        self.higgs.phi = phi;
        self.higgs.pi = pi;
        self.higgs_apply_pins();
    }

    /// U ← U·exp(−iΔt·Ẽ·T) en todos los enlaces que no estén congelados.
    fn higgs_su2_link_update(&mut self, dt: f64) {
        let links = self.dim * self.n;
        let ng = self.ym.ng;
        let mut th = [0.0f64; 8];
        for l in 0..links {
            if self.higgs.has_frozen && self.higgs.frozen_link[l] != 0 {
                continue;
            }
            for c in 0..ng {
                th[c] = self.ym.e[l * ng + c] * dt;
            }
            let ex = self.ym.alg.exp_minus_i(&th[..ng]);
            let u = self.ym.link(l).mul(&ex);
            self.ym.set_link(l, &u);
        }
    }

    /// X = U_a(r)†·Φ(r+â)·U_a(r) (componentes) para el Higgs adjunto.
    #[inline(always)]
    fn adjoint_x(&self, l: usize, phi_next: &[f64]) -> Mat {
        let u = self.ym.link(l);
        let p = self.ym.alg.combine(phi_next);
        u.dag_mul(&p.mul(&u))
    }

    fn higgs_adjoint_substep(&mut self, dt: f64) {
        let n = self.n;
        let dim = self.dim;
        let c2 = self.higgs_c2();
        let g2 = self.higgs.g * self.higgs.g;
        let v2 = self.higgs.v * self.higgs.v;
        let lam = self.higgs.lambda;
        let damp = 1.0 - self.em.eta;
        let phi = std::mem::take(&mut self.higgs.phi);
        let mut pi = std::mem::take(&mut self.higgs.pi);
        let mut tmp = [0.0f64; 8];
        let dd = 2.0 * dim as f64;
        for i in 0..n {
            if self.higgs.has_pins && self.higgs.pinned[i] != 0 {
                continue;
            }
            let mut lap = [-dd * phi[3 * i], -dd * phi[3 * i + 1], -dd * phi[3 * i + 2]];
            for a in 0..dim {
                let ip = self.nb(i, a, 1);
                let x = self.adjoint_x(a * n + i, &phi[3 * ip..3 * ip + 3]);
                self.ym.alg.components(&x, &mut tmp);
                for c in 0..3 {
                    lap[c] += tmp[c];
                }
                // U_a(r−â)·Φ(r−â)·U_a(r−â)†
                let im = self.nb(i, a, -1);
                let u = self.ym.link(a * n + im);
                let p = self.ym.alg.combine(&phi[3 * im..3 * im + 3]);
                let y = u.mul(&p).mul_dag(&u);
                self.ym.alg.components(&y, &mut tmp);
                for c in 0..3 {
                    lap[c] += tmp[c];
                }
            }
            let m2 = phi[3 * i] * phi[3 * i] + phi[3 * i + 1] * phi[3 * i + 1] + phi[3 * i + 2] * phi[3 * i + 2] - v2;
            for c in 0..3 {
                pi[3 * i + c] = (pi[3 * i + c] + dt * (c2 * lap[c] - lam * m2 * phi[3 * i + c])) * damp;
            }
        }
        self.ym_compute_force();
        for a in 0..dim {
            for i in 0..n {
                let l = a * n + i;
                if self.higgs.has_frozen && self.higgs.frozen_link[l] != 0 {
                    for c in 0..3 {
                        self.ym.e[l * 3 + c] = 0.0;
                    }
                    continue;
                }
                let ip = self.nb(i, a, 1);
                let x = self.adjoint_x(l, &phi[3 * ip..3 * ip + 3]);
                let p = self.ym.alg.combine(&phi[3 * i..3 * i + 3]);
                // J^c = −2i·c²·Tr(T^c·[X, Φ(r)]) = 2c²·Im Tr(T^c·[X, Φ])
                let comm = x.mul(&p).add(&p.mul(&x).scale(cx(-1.0, 0.0)));
                for c in 0..3 {
                    let j = 2.0 * c2 * self.ym.alg.tr_t(c, &comm).im;
                    let o = l * 3 + c;
                    self.ym.e[o] = (self.ym.e[o] + dt * (self.ym.force[o] - g2 * j)) * damp;
                }
            }
        }
        let mut phi = phi;
        for k in 0..3 * n {
            phi[k] += dt * pi[k];
        }
        self.higgs_su2_link_update(dt);
        self.higgs.phi = phi;
        self.higgs.pi = pi;
        self.higgs_apply_pins();
    }

    // ------------------------------------------------------------------ Gauss

    /// Ley de Gauss en la celda i: abeliano → [div E + 2q·Im(φ*π)]; doblete → [G^1, G^2, G^3, G_B];
    /// adjunto → G^c = [cov-div Ẽ]^c + g²·(φ × π)^c.
    pub(crate) fn higgs_gauss_at(&self, i: usize, out: &mut [f64]) {
        let n = self.n;
        match self.higgs.mode {
            1 => {
                let mut div = 0.0;
                for a in 0..self.dim {
                    div += self.em.e[a * n + i] - self.em.e[a * n + self.nb(i, a, -1)];
                }
                let (xr, xi) = (self.higgs.phi[2 * i], self.higgs.phi[2 * i + 1]);
                let (pr, pim) = (self.higgs.pi[2 * i], self.higgs.pi[2 * i + 1]);
                out[0] = div + 2.0 * self.higgs_q() * (xr * pim - xi * pr);
            }
            2 => {
                self.ym_covdiv_at(i, out);
                let x = [cx(self.higgs.phi[4 * i], self.higgs.phi[4 * i + 1]), cx(self.higgs.phi[4 * i + 2], self.higgs.phi[4 * i + 3]), cx(0.0, 0.0)];
                let p = [cx(self.higgs.pi[4 * i], self.higgs.pi[4 * i + 1]), cx(self.higgs.pi[4 * i + 2], self.higgs.pi[4 * i + 3]), cx(0.0, 0.0)];
                let g2 = self.higgs.g * self.higgs.g;
                for c in 0..3 {
                    let mut s = cx(0.0, 0.0);
                    for &(r, cc, t) in &self.ym.alg.sparse[c] {
                        s += x[r].conj() * t * p[cc];
                    }
                    out[c] += 2.0 * g2 * s.im;
                }
                let mut div = 0.0;
                for a in 0..self.dim {
                    div += self.em.e[a * n + i] - self.em.e[a * n + self.nb(i, a, -1)];
                }
                let mut s = 0.0;
                for k in 0..2 {
                    s += (x[k].conj() * p[k]).im;
                }
                out[3] = div + 2.0 * self.higgs_q() * s;
                if self.sm.active {
                    // fermiones del modelo estándar (§2.21): −g²·ρ₂ y −Σ g′Y_f·|ψ_f|²
                    let (ry, r2, _) = self.sm_charges_at(i);
                    for c in 0..3 {
                        out[c] -= g2 * r2[c];
                    }
                    out[3] -= ry;
                }
            }
            3 => {
                self.ym_covdiv_at(i, out);
                let f = &self.higgs.phi[3 * i..3 * i + 3];
                let p = &self.higgs.pi[3 * i..3 * i + 3];
                let g2 = self.higgs.g * self.higgs.g;
                out[0] += g2 * (f[1] * p[2] - f[2] * p[1]);
                out[1] += g2 * (f[2] * p[0] - f[0] * p[2]);
                out[2] += g2 * (f[0] * p[1] - f[1] * p[0]);
            }
            _ => {}
        }
    }

    /// Σ_a [Ẽ_a(r) − Ad(U_a(r−â))·Ẽ_a(r−â)] (sin materia).
    pub(crate) fn ym_covdiv_at(&self, i: usize, out: &mut [f64]) {
        let n = self.n;
        let ng = self.ym.ng;
        let mut tmp = [0.0f64; 8];
        for c in 0..ng {
            out[c] = 0.0;
        }
        for a in 0..self.dim {
            let im = self.nb(i, a, -1);
            for c in 0..ng {
                out[c] += self.ym.e[(a * n + i) * ng + c];
            }
            let u = self.ym.link(a * n + im);
            let x = self.ym.alg.combine(&self.ym.e[(a * n + im) * ng..(a * n + im + 1) * ng]);
            let ad = u.mul(&x).mul_dag(&u);
            self.ym.alg.components(&ad, &mut tmp);
            for c in 0..ng {
                out[c] -= tmp[c];
            }
        }
    }

    fn higgs_gauss_width(&self) -> usize {
        match self.higgs.mode {
            1 => 1,
            2 => 4,
            _ => 3,
        }
    }

    /// Guarda la ley de Gauss actual como referencia (tras sembrar un estado).
    pub(crate) fn higgs_store_gauss_ref(&mut self) {
        let w = self.higgs_gauss_width();
        let mut g = std::mem::take(&mut self.higgs.g_ref);
        g.resize(w * self.n, 0.0);
        let mut tmp = [0.0f64; 8];
        for i in 0..self.n {
            self.higgs_gauss_at(i, &mut tmp);
            g[i * w..(i + 1) * w].copy_from_slice(&tmp[..w]);
        }
        self.higgs.g_ref = g;
    }

    pub(crate) fn higgs_gauss_residual_parts(&self) -> [f64; 2] {
        let w = self.higgs_gauss_width();
        let mut tmp = [0.0f64; 8];
        let mut worst = [0.0f64; 2];
        for i in 0..self.n {
            self.higgs_gauss_at(i, &mut tmp);
            for c in 0..w {
                let d = (tmp[c] - self.higgs.g_ref[i * w + c]).abs();
                let k = if self.higgs.mode == 2 && c == 3 { 1 } else { 0 };
                worst[k] = worst[k].max(d);
            }
        }
        worst
    }

    // ------------------------------------------------------------------ energía

    /// |D_aφ|² del enlace (a, r) (abeliano y doblete).
    pub(crate) fn higgs_grad2(&self, i: usize, a: usize) -> f64 {
        let n = self.n;
        let ip = self.nb(i, a, 1);
        let l = a * n + i;
        let q = self.higgs_q();
        match self.higgs.mode {
            1 => {
                let (s, c) = (q * self.em.a[l]).sin_cos();
                let (yr, yi) = (self.higgs.phi[2 * ip], self.higgs.phi[2 * ip + 1]);
                let (wr, wi) = (c * yr + s * yi, c * yi - s * yr);
                (wr - self.higgs.phi[2 * i]).powi(2) + (wi - self.higgs.phi[2 * i + 1]).powi(2)
            }
            2 => {
                let u = self.ym.link(l);
                let v = [cx(self.higgs.phi[4 * ip], self.higgs.phi[4 * ip + 1]), cx(self.higgs.phi[4 * ip + 2], self.higgs.phi[4 * ip + 3]), cx(0.0, 0.0)];
                let w = u.apply_dag(&v);
                let e = C::from_polar(1.0, -q * self.em.a[l]);
                let mut s = 0.0;
                for k in 0..2 {
                    s += (w[k] * e - cx(self.higgs.phi[4 * i + 2 * k], self.higgs.phi[4 * i + 2 * k + 1])).norm_sqr();
                }
                s
            }
            3 => {
                let x = self.adjoint_x(l, &self.higgs.phi[3 * ip..3 * ip + 3]);
                let mut tmp = [0.0f64; 8];
                self.ym.alg.components(&x, &mut tmp);
                (0..3).map(|c| (tmp[c] - self.higgs.phi[3 * i + c]).powi(2)).sum()
            }
            _ => 0.0,
        }
    }

    /// Densidad de energía por celda (enlaces y plaquetas con esquina en la celda) en
    /// `higgs.energy_density`; devuelve la total.
    pub(crate) fn higgs_compute_energy_density(&mut self) -> f64 {
        let n = self.n;
        let dim = self.dim;
        let c2 = self.higgs_c2();
        let v2 = self.higgs.v * self.higgs.v;
        let lam = self.higgs.lambda;
        let q = self.higgs_q();
        let kap = self.em.kappa;
        let nf = self.higgs.nf;
        let inv_g2 = if self.higgs.mode >= 2 && self.higgs.g.abs() > 1e-15 { 1.0 / (self.higgs.g * self.higgs.g) } else { 0.0 };
        let mut dens = std::mem::take(&mut self.higgs.energy_density);
        dens.resize(n, 0.0);
        let mut total = 0.0;
        for i in 0..n {
            let mut e = 0.0;
            let p2: f64 = self.higgs.pi[nf * i..nf * (i + 1)].iter().map(|v| v * v).sum();
            let f2: f64 = self.higgs.phi[nf * i..nf * (i + 1)].iter().map(|v| v * v).sum();
            let mut grad = 0.0;
            for a in 0..dim {
                grad += self.higgs_grad2(i, a);
            }
            if self.higgs.mode == 3 {
                e += 0.5 * p2 + 0.5 * c2 * grad + 0.25 * lam * (f2 - v2).powi(2);
            } else {
                e += p2 + c2 * grad + lam * (f2 - v2).powi(2);
                if self.higgs.mode == 1 && self.higgs.bias != 0.0 {
                    e -= self.higgs.bias * self.higgs.phi[2 * i];
                }
            }
            // U(1)
            if self.higgs.mode <= 2 {
                for a in 0..dim {
                    e += 0.5 * self.em.e[a * n + i].powi(2);
                    for b in (a + 1)..dim {
                        let bb = self.higgs_bt(i, a, b);
                        e += if self.em.compact && q.abs() > 1e-15 { kap * (1.0 - (q * bb).cos()) / (q * q) } else { 0.5 * kap * bb * bb };
                    }
                }
            }
            // SU(2), normalizada con 1/g²
            if self.higgs.mode >= 2 {
                let mut eg = 0.0;
                for a in 0..dim {
                    for c in 0..3 {
                        eg += 0.5 * self.ym.e[(a * n + i) * 3 + c].powi(2);
                    }
                    for b in (a + 1)..dim {
                        eg += 2.0 * kap * (2.0 - self.ym_plaq_p(i, a, b).trace().re);
                    }
                }
                e += inv_g2 * eg;
            }
            dens[i] = e as f32;
            total += e;
        }
        self.higgs.energy_density = dens;
        total
    }

    // ------------------------------------------------------------------ vórtices

    /// Número de vórtice invariante gauge de la plaqueta xy en r (abeliano).
    pub(crate) fn higgs_winding(&self, i: usize) -> i32 {
        let n = self.n;
        let q = self.higgs_q();
        let ix = self.nb(i, 0, 1);
        let iy = self.nb(i, 1, 1);
        let ixy = self.nb(ix, 1, 1);
        let f = |k: usize| cx(self.higgs.phi[2 * k], self.higgs.phi[2 * k + 1]);
        // arg(φ*(r)·e^{−iqA_a(r)}·φ(r+â)) para cada borde orientado
        let link = |from: usize, to: usize, l: usize| -> f64 { (f(from).conj() * C::from_polar(1.0, -q * self.em.a[l]) * f(to)).arg() };
        let s = link(i, ix, i) + link(ix, ixy, n + ix) - link(iy, ixy, iy) - link(i, iy, n + i);
        // con la energía compacta, qB solo está definido módulo 2π (la costura del gauge de Landau
        // lleva qB − 2πn): se toma en (−π, π]
        let qb = q * self.higgs_bt(i, 0, 1);
        let qb = if self.em.compact { crate::util::wrap_angle(qb) } else { qb };
        ((s + qb) / (2.0 * PI)).round() as i32
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.12: 0 = desactivado, 1 = abeliano (U(1)), 2 = doblete SU(2)×U(1). Pone φ = v (o
    /// Φ = (0, v)), π = 0 y los campos a cero; la materia queda en pausa. Usa q, κ = c² y s del
    /// modo electromagnetismo (y g, g′ de `set_electroweak_couplings` en el doblete).
    pub fn set_higgs(&mut self, mode: u32, v: f32, lambda: f32) {
        self.message.clear();
        if mode == 3 {
            self.set_message("El Higgs adjunto se activa con set_adjoint_higgs (§2.12, 3D).");
            return;
        }
        if mode == 0 && self.sm.active {
            self.set_message("No se puede desactivar el Higgs mientras el modelo estándar esté activo.");
            return;
        }
        if mode != 0 && self.scenario != 0 {
            self.set_message("El campo de Higgs se usa en el escenario materia (la materia queda en pausa).");
            return;
        }
        self.higgs.v = crate::util::clamp_finite(v as f64, 0.0, 1e3, 1.0);
        self.higgs.lambda = crate::util::clamp_finite(lambda as f64, 0.0, 1e3, 0.0);
        match mode {
            0 => {
                self.higgs.mode = 0;
                return;
            }
            1 => {
                if self.ncol > 1 {
                    self.resize_colour(1);
                }
                self.higgs.mode = 1;
                self.higgs.bias = 0.0;
                self.higgs_alloc(2);
                let n = self.n;
                for i in 0..n {
                    self.higgs.phi[2 * i] = self.higgs.v;
                }
            }
            _ => {
                if self.ncol != 2 {
                    self.resize_colour(2);
                } else {
                    self.ym = crate::yangmills::YmField::new(2, self.dim, self.n);
                }
                self.higgs.mode = 2;
                self.higgs_alloc(4);
                self.em.q = 0.5 * self.higgs.gp;
                let n = self.n;
                for i in 0..n {
                    self.higgs.phi[4 * i + 2] = self.higgs.v;
                }
            }
        }
        self.clear();
        self.ensure_em_buffers();
        self.em.mode = 2;
        for v in self.em.a.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.e.iter_mut() {
            *v = 0.0;
        }
        self.em.has_ext = false;
        self.em.has_j_ext = false;
        for v in self.em.rho_ext.iter_mut() {
            *v = 0.0;
        }
        self.em.markers.clear();
        self.higgs_store_gauss_ref();
        self.em.ph_valid = false;
    }

    pub fn higgs_mode(&self) -> u32 {
        self.higgs.mode
    }

    /// Doblete: SU(2) con g y U(1)_Y con g′ (la carga U(1) del Higgs es g′/2).
    pub fn set_electroweak_couplings(&mut self, g: f32, g_prime: f32) {
        if !(g.is_finite() && g_prime.is_finite()) {
            return;
        }
        self.higgs.g = g.clamp(-10.0, 10.0) as f64;
        self.higgs.gp = g_prime.clamp(-10.0, 10.0) as f64;
        if self.higgs.mode == 2 {
            self.em.q = 0.5 * self.higgs.gp;
            self.higgs_store_gauss_ref();
        }
    }

    /// U(1) de Wilson, (κ/q²)·Σ(1 − cos qB): necesaria para meter flujo neto en el toro.
    pub fn set_compact_magnetic_energy(&mut self, compact: bool) {
        self.em.compact = compact;
    }

    pub fn compact_magnetic_energy(&self) -> bool {
        self.em.compact
    }

    /// n cuantos 2π/q repartidos uniformemente (gauge de Landau con costura). Exige la energía
    /// compacta: con la no compacta, el flujo total de cada plano es exactamente cero.
    pub fn set_uniform_flux(&mut self, n_flux: i32) {
        if !self.em.compact {
            self.set_message("El flujo neto en el toro exige la energía magnética compacta (§2.12).");
            return;
        }
        self.set_uniform_magnetic_field(n_flux);
    }

    /// 3D: dos monopolos magnéticos externos de carga ±2π/q unidos por una cuerda de Dirac fija a
    /// lo largo de z, en la columna (x, y), entre z y z + length. El Higgs no la ve (flujo 2π/q).
    pub fn add_monopole_pair(&mut self, x: i32, y: i32, z: i32, length: u32) {
        self.message.clear();
        if self.dim != 3 || self.higgs.mode != 1 {
            self.set_message("Los monopolos externos son del Higgs abeliano en 3D.");
            return;
        }
        let q = if self.higgs_q() != 0.0 { self.higgs_q() } else { 1.0 };
        let flux = 2.0 * PI / q;
        let len = length.min((self.d as u32).saturating_mul(4).max(1)).min(4096) as i32;
        for k in 0..len {
            let i = self.wrap_cell(x, y, z.wrapping_add(k));
            self.higgs.string_b[i] += flux;
        }
        self.higgs.has_string = true;
    }

    /// Flujo total (B[A] + cuerdas) por el plano z = cte, en unidades de 2π/q.
    pub fn higgs_plane_flux(&self, z: i32) -> f64 {
        let q = if self.higgs_q() != 0.0 { self.higgs_q() } else { 1.0 };
        let zz = if self.dim == 3 { crate::util::wrap_i(z, self.d) } else { 0 };
        let mut s = 0.0;
        for y in 0..self.h {
            for x in 0..self.w {
                s += self.higgs_bt(self.idx(x, y, zz), 0, 1);
            }
        }
        s * q / (2.0 * PI)
    }

    /// Cuenta los vórtices por el número invariante de plaqueta (en 3D, en el corte z = slice).
    pub fn higgs_vortex_count(&mut self) -> u32 {
        self.higgs.vortices.clear();
        if self.higgs.mode != 1 {
            return 0;
        }
        let z = if self.dim == 3 { self.slice } else { 0 };
        let mut count = 0u32;
        for y in 0..self.h {
            for x in 0..self.w {
                let i = self.idx(x, y, z);
                let w = self.higgs_winding(i);
                if w != 0 {
                    count += 1;
                    if self.higgs.vortices.len() < 3 * 4096 {
                        self.higgs.vortices.push(x as f32 + 0.5);
                        self.higgs.vortices.push(y as f32 + 0.5);
                        self.higgs.vortices.push(w as f32);
                    }
                }
            }
        }
        let _ = count;
        (self.higgs.vortices.len() / 3) as u32
    }

    /// n × 3 floats: [x, y, devanado] del último `higgs_vortex_count`.
    pub fn higgs_vortices_ptr(&self) -> *const f32 {
        self.higgs.vortices.as_ptr()
    }

    pub fn higgs_vortices_len(&self) -> u32 {
        self.higgs.vortices.len() as u32
    }

    /// Σ|π|² + c²Σ|Dφ|² + λΣ(|φ|² − v²)² + la energía de los campos gauge (SU(2) con 1/g²).
    /// En el adjunto: ½Σπ² + ½c²Σ|DΦ|² + λ/4·Σ(|φ|² − v²)² + la energía gauge.
    pub fn higgs_energy(&mut self) -> f64 {
        if self.higgs.mode == 0 {
            return 0.0;
        }
        self.higgs_compute_energy_density()
    }

    /// Energía dentro de la esfera (círculo en 2D) de radio r alrededor de (x, y, z).
    pub fn higgs_energy_in_sphere(&mut self, x: f32, y: f32, z: f32, radius: f32) -> f64 {
        if self.higgs.mode == 0 {
            return 0.0;
        }
        self.higgs_compute_energy_density();
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let r2 = (radius as f64).powi(2);
        let mut s = 0.0;
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let dx = min_image(px as f64 - x as f64, dims[0]);
            let dy = min_image(py as f64 - y as f64, dims[1]);
            let dz = if self.dim == 3 { min_image(pz as f64 - z as f64, dims[2]) } else { 0.0 };
            if dx * dx + dy * dy + dz * dz < r2 {
                s += self.higgs.energy_density[i] as f64;
            }
        }
        s
    }

    pub fn higgs_energy_density_ptr(&self) -> *const f32 {
        self.higgs.energy_density.as_ptr()
    }

    pub fn higgs_energy_density_len(&self) -> u32 {
        self.higgs.energy_density.len() as u32
    }

    /// Residuo de Gauss del Higgs: [SU(2) o abeliano, U(1) del doblete].
    pub fn higgs_gauss_residuals(&self) -> Vec<f64> {
        if self.higgs.mode == 0 {
            return vec![0.0, 0.0];
        }
        self.higgs_gauss_residual_parts().to_vec()
    }

    /// Componente k del campo de Higgs en una celda (abeliano: Re, Im; doblete: Re Φ₁, Im Φ₁,
    /// Re Φ₂, Im Φ₂; adjunto: φ¹, φ², φ³).
    pub fn higgs_phi_at(&self, x: i32, y: i32, z: i32, k: u32) -> f64 {
        if self.higgs.mode == 0 {
            return 0.0;
        }
        let i = self.wrap_cell(x, y, z);
        let k = (k as usize).min(self.higgs.nf - 1);
        self.higgs.phi[i * self.higgs.nf + k]
    }

    /// |φ| en una celda.
    pub fn higgs_abs_at(&self, x: i32, y: i32, z: i32) -> f64 {
        if self.higgs.mode == 0 {
            return 0.0;
        }
        let i = self.wrap_cell(x, y, z);
        let nf = self.higgs.nf;
        self.higgs.phi[i * nf..(i + 1) * nf].iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Masas del vacío: abeliano [m_γ, m_H, λ_L, ξ_H]; doblete [M_W, M_Z, cos θ_W, m_H];
    /// adjunto [m_W, m_H, 4πκv/g, 0].
    pub fn higgs_masses(&self) -> Vec<f64> {
        let c2 = self.higgs_c2();
        let v = self.higgs.v;
        let lam = self.higgs.lambda;
        match self.higgs.mode {
            1 => {
                let q = self.higgs_q();
                let mg = (2.0 * c2 * q * q * v * v).sqrt();
                let mh = (4.0 * lam * v * v).sqrt();
                let c = c2.sqrt();
                vec![mg, mh, if mg > 0.0 { c / mg } else { f64::INFINITY }, if mh > 0.0 { c / mh } else { f64::INFINITY }]
            }
            2 => {
                let (g, gp) = (self.higgs.g, self.higgs.gp);
                let mw = (c2 * g * g * v * v / 2.0).sqrt();
                let mz = (c2 * (g * g + gp * gp) * v * v / 2.0).sqrt();
                let gn = (g * g + gp * gp).sqrt();
                let cw = if gn > 1e-15 { g / gn } else { 1.0 };
                vec![mw, mz, cw, (4.0 * lam * v * v).sqrt()]
            }
            3 => {
                let g = self.higgs.g;
                let mono = if g.abs() > 1e-15 { 4.0 * PI * self.em.kappa * v / g } else { f64::INFINITY };
                vec![c2.sqrt() * g * v, (2.0 * lam).sqrt() * v, mono, 0.0]
            }
            _ => vec![0.0; 4],
        }
    }

    /// Siembra φ = v + amp_phi·ruido complejo y π = amp_pi·ruido (y lo guarda como referencia de
    /// Gauss): el estado aleatorio de los tests de Gauss.
    pub fn seed_higgs_random(&mut self, amp_phi: f32, amp_pi: f32, seed: u32) {
        if self.higgs.mode == 0 {
            return;
        }
        let mut rng = crate::rng::SplitMix64::new(seed as u64 ^ 0x4166_5300);
        let nf = self.higgs.nf;
        let v = self.higgs.v;
        for i in 0..self.n {
            for k in 0..nf {
                let base = match (self.higgs.mode, k) {
                    (1, 0) | (2, 2) => v,
                    (3, 2) => v,
                    _ => 0.0,
                };
                self.higgs.phi[i * nf + k] = base + amp_phi as f64 * rng.gauss();
                self.higgs.pi[i * nf + k] = amp_pi as f64 * rng.gauss();
            }
        }
        self.higgs_store_gauss_ref();
    }

    /// Ruido en φ (φ += amp·ruido) sin tocar π: la semilla de las relajaciones.
    pub fn add_higgs_noise(&mut self, amp: f32, seed: u32) {
        if self.higgs.mode == 0 {
            return;
        }
        let mut rng = crate::rng::SplitMix64::new(seed as u64 ^ 0x4e01_5e00);
        for v in self.higgs.phi.iter_mut() {
            *v += amp as f64 * rng.gauss();
        }
        self.higgs_store_gauss_ref();
    }

    /// Onda estacionaria de amplitud `amp` con k = 2π·k_index/W a lo largo de x.
    /// Abeliano: 0 = fotón (A_y), 1 = Higgs (|φ|), 2 = fase de φ.
    /// Doblete: 0 = W¹ (U_y), 1 = W³ (U_y), 2 = B (A_y), 3 = Higgs (Φ₂).
    pub fn seed_higgs_wave(&mut self, kind: u32, k_index: u32, amp: f32) {
        if self.higgs.mode == 0 || self.dim < 2 {
            return;
        }
        self.ensure_em_buffers();
        let n = self.n;
        let k = 2.0 * PI * k_index as f64 / self.w as f64;
        let amp = amp as f64;
        for i in 0..n {
            let (x, _, _) = self.coords(i);
            let c = (k * x as f64).cos();
            match (self.higgs.mode, kind) {
                (1, 0) | (2, 2) => self.em.a[n + i] += amp * c,
                (1, 1) => self.higgs.phi[2 * i] += amp * c,
                (1, 2) => {
                    let (s, co) = (amp * c).sin_cos();
                    let (r, im) = (self.higgs.phi[2 * i], self.higgs.phi[2 * i + 1]);
                    self.higgs.phi[2 * i] = r * co - im * s;
                    self.higgs.phi[2 * i + 1] = r * s + im * co;
                }
                (2, 0) | (2, 1) => {
                    let mut th = [0.0f64; 3];
                    th[if kind == 0 { 0 } else { 2 }] = amp * c;
                    let ex = self.ym.alg.exp_minus_i(&th);
                    let u = self.ym.link(n + i).mul(&ex);
                    self.ym.set_link(n + i, &u);
                }
                (2, 3) => self.higgs.phi[4 * i + 2] += amp * c,
                _ => {}
            }
        }
        self.em.ph_valid = false;
        self.higgs_store_gauss_ref();
    }

    /// Vórtice de devanado w en (x, y): φ ← φ·f(r)·e^{iwθ}, con θ la suma de imágenes del toro
    /// y f(r) = r/√(r² + core²). La carga total del toro debe ser 0 para que la fase sea periódica.
    pub fn imprint_higgs_vortex(&mut self, x: f32, y: f32, winding: i32, core: f32) {
        if self.higgs.mode != 1 {
            return;
        }
        let (w, h) = (self.w as f64, self.h as f64);
        let core2 = (core as f64).powi(2).max(1e-6);
        for i in 0..self.n {
            let (px, py, _) = self.coords(i);
            let mut th = 0.0;
            for a in -4i32..=4 {
                for b in -4i32..=4 {
                    th += (py as f64 - y as f64 - b as f64 * h).atan2(px as f64 - x as f64 - a as f64 * w);
                }
            }
            let dx = min_image(px as f64 - x as f64, w);
            let dy = min_image(py as f64 - y as f64, h);
            let r2 = dx * dx + dy * dy;
            let f = (r2 / (r2 + core2)).sqrt();
            let (s, c) = (winding as f64 * th).sin_cos();
            let (re, im) = (self.higgs.phi[2 * i], self.higgs.phi[2 * i + 1]);
            self.higgs.phi[2 * i] = f * (re * c - im * s);
            self.higgs.phi[2 * i + 1] = f * (re * s + im * c);
        }
        self.higgs_store_gauss_ref();
    }

    /// Fija φ (y π = 0) en las celdas a menos de `radius` de (x, y, z), con su valor actual:
    /// los núcleos fijados de §3.9.
    pub fn pin_higgs_cells(&mut self, x: f32, y: f32, z: f32, radius: f32) {
        if self.higgs.mode == 0 {
            return;
        }
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let nf = self.higgs.nf;
        let r2 = (radius as f64).powi(2);
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let dx = min_image(px as f64 - x as f64, dims[0]);
            let dy = min_image(py as f64 - y as f64, dims[1]);
            let dz = if self.dim == 3 { min_image(pz as f64 - z as f64, dims[2]) } else { 0.0 };
            if dx * dx + dy * dy + dz * dz <= r2 {
                self.higgs.pinned[i] = 1;
                for k in 0..nf {
                    self.higgs.pin_val[i * nf + k] = self.higgs.phi[i * nf + k];
                    self.higgs.pi[i * nf + k] = 0.0;
                }
                self.higgs.has_pins = true;
            }
        }
    }

    /// Fija solo la fase de φ (abeliano) en las celdas a menos de `radius` de (x, y): el devanado
    /// alrededor de la plaqueta, y con él la posición del vórtice, queda fijo, pero |φ| y los
    /// campos gauge se relajan libremente (fijar el valor entero sesga la energía del núcleo).
    pub fn pin_higgs_phase(&mut self, x: f32, y: f32, radius: f32) {
        if self.higgs.mode != 1 {
            return;
        }
        let dims = [self.w as f64, self.h as f64];
        let r2 = (radius as f64).powi(2);
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            if self.dim == 3 && pz != self.slice {
                continue;
            }
            let dx = min_image(px as f64 - x as f64, dims[0]);
            let dy = min_image(py as f64 - y as f64, dims[1]);
            if dx * dx + dy * dy <= r2 {
                let (re, im) = (self.higgs.phi[2 * i], self.higgs.phi[2 * i + 1]);
                let a = re.hypot(im).max(1e-300);
                self.higgs.pinned[i] = 2;
                self.higgs.pin_val[2 * i] = re / a;
                self.higgs.pin_val[2 * i + 1] = im / a;
                self.higgs.has_pins = true;
            }
        }
    }

    /// Perfil radial de |φ| alrededor de (x, y) (2D, abeliano o cualquier modo): pares
    /// [r₀, |φ|₀, r₁, |φ|₁, …] en los radios discretos de las celdas, hasta rmax.
    pub fn higgs_radial_profile(&self, x: f32, y: f32, rmax: f32) -> Vec<f64> {
        if self.higgs.mode == 0 {
            return Vec::new();
        }
        let (w, h) = (self.w as f64, self.h as f64);
        let z = if self.dim == 3 { self.slice } else { 0 };
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for py in 0..self.h {
            for px in 0..self.w {
                let r = min_image(px as f64 - x as f64, w).hypot(min_image(py as f64 - y as f64, h));
                if r <= rmax as f64 {
                    pts.push((r, self.higgs_abs_at(px as i32, py as i32, z as i32)));
                }
            }
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut out = Vec::new();
        let mut k = 0;
        while k < pts.len() {
            let r = pts[k].0;
            let (mut s, mut c) = (0.0, 0.0);
            while k < pts.len() && (pts[k].0 - r).abs() < 1e-9 {
                s += pts[k].1;
                c += 1.0;
                k += 1;
            }
            out.push(r);
            out.push(s / c);
        }
        out
    }

    /// Pone |φ| = a en una celda conservando su fase (abeliano).
    pub fn set_higgs_abs(&mut self, x: i32, y: i32, z: i32, a: f32) {
        if self.higgs.mode != 1 {
            return;
        }
        let i = self.wrap_cell(x, y, z);
        let (re, im) = (self.higgs.phi[2 * i], self.higgs.phi[2 * i + 1]);
        let m = re.hypot(im);
        let (ur, ui) = if m > 1e-300 { (re / m, im / m) } else { (1.0, 0.0) };
        self.higgs.phi[2 * i] = a as f64 * ur;
        self.higgs.phi[2 * i + 1] = a as f64 * ui;
    }

    pub fn clear_higgs_pins(&mut self) {
        for v in self.higgs.pinned.iter_mut() {
            *v = 0;
        }
        self.higgs.has_pins = false;
    }

    /// §3.9: empujón covariante de lo que hay en el rectángulo [x0, x1) × [y0, y1) (2D, abeliano):
    /// π = −v·Dφ (diferencia centrada covariante), E = −v × B, y la ley de Gauss restaurada con una
    /// corrección longitudinal (FFT).
    pub fn boost_higgs_region(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, vx: f32, vy: f32) {
        if self.higgs.mode != 1 || self.dim != 2 {
            return;
        }
        let n = self.n;
        let q = self.higgs_q();
        let (vx, vy) = (vx as f64, vy as f64);
        let inside = |x: usize, y: usize| -> bool {
            let (x, y) = (x as i32, y as i32);
            x >= x0 && x < x1 && y >= y0 && y < y1
        };
        let phi = self.higgs.phi.clone();
        let f = |k: usize| cx(phi[2 * k], phi[2 * k + 1]);
        for i in 0..n {
            let (x, y, _) = self.coords(i);
            if !inside(x, y) {
                continue;
            }
            // D_aφ centrada: [e^{−iqA_a(r)}φ(r+â) − e^{iqA_a(r−â)}φ(r−â)]/2
            let mut d = [cx(0.0, 0.0); 2];
            for a in 0..2 {
                let ip = self.nb(i, a, 1);
                let im = self.nb(i, a, -1);
                let fw = C::from_polar(1.0, -q * self.em.a[a * n + i]) * f(ip);
                let bw = C::from_polar(1.0, q * self.em.a[a * n + im]) * f(im);
                d[a] = (fw - bw) * 0.5;
            }
            let p = -(d[0] * vx + d[1] * vy);
            self.higgs.pi[2 * i] = p.re;
            self.higgs.pi[2 * i + 1] = p.im;
            // E = −v × B con B = B_xy (media de las plaquetas que tocan el enlace)
            let b_here = self.plaquette_b(i, 0, 1);
            let b_y = 0.5 * (b_here + self.plaquette_b(self.nb(i, 1, -1), 0, 1));
            let b_x = 0.5 * (b_here + self.plaquette_b(self.nb(i, 0, -1), 0, 1));
            self.em.e[i] = -vy * b_y;
            self.em.e[n + i] = vx * b_x;
        }
        // restaura G = G_ref con un campo longitudinal
        let mut src = vec![0.0f64; n];
        let mut tmp = [0.0f64; 8];
        for i in 0..n {
            self.higgs_gauss_at(i, &mut tmp);
            src[i] = -(tmp[0] - self.higgs.g_ref[i]);
        }
        self.em_add_gradient_field(&src);
    }

    // ------------------------------------------------------------------ monopolo de 't Hooft–Polyakov

    /// §2.12 (3D): SU(2) con un Higgs en la adjunta, acoplamiento g, v y λ. Pone φ = v·ẑ, π = 0,
    /// U = 1 y Ẽ = 0; la materia queda en pausa. Usa κ = c² y s del modo electromagnetismo.
    pub fn set_adjoint_higgs(&mut self, g: f32, v: f32, lambda: f32) {
        self.message.clear();
        if self.dim != 3 {
            self.set_message("El Higgs adjunto y el monopolo de 't Hooft–Polyakov son 3D.");
            return;
        }
        if self.scenario != 0 {
            self.set_message("El campo de Higgs se usa en el escenario materia (la materia queda en pausa).");
            return;
        }
        if self.ncol != 2 {
            self.resize_colour(2);
        } else {
            self.ym = crate::yangmills::YmField::new(2, self.dim, self.n);
        }
        self.higgs.g = crate::util::clamp_finite(g as f64, -10.0, 10.0, 0.5);
        self.higgs.v = crate::util::clamp_finite(v as f64, 0.0, 1e3, 1.0);
        self.higgs.lambda = crate::util::clamp_finite(lambda as f64, 0.0, 1e3, 0.0);
        self.higgs.mode = 3;
        self.higgs_alloc(3);
        for i in 0..self.n {
            self.higgs.phi[3 * i + 2] = self.higgs.v;
        }
        self.clear();
        self.ensure_em_buffers();
        self.em.mode = 2;
        self.higgs_store_gauss_ref();
    }

    /// Erizo de 't Hooft–Polyakov centrado en (x, y, z): φ = v·h(r)·r̂ y A_i^a = ε_aij·r̂_j·(1 − K)/(g·r),
    /// con U = exp(+i·g·A·T) a lo largo de cada enlace. Con λ = 0, h y K son la solución exacta de
    /// Prasad–Sommerfield (ξ = g·v·r); con λ > 0, perfiles de prueba que la relajación corrige. Una
    /// capa de 2 celdas en el borde de la caja queda congelada en esos campos.
    pub fn seed_thooft_polyakov(&mut self, x: f32, y: f32, z: f32) {
        if self.higgs.mode != 3 {
            self.set_message("Primero activa el Higgs adjunto (set_adjoint_higgs).");
            return;
        }
        let (c0, n) = ([x as f64, y as f64, z as f64], self.n);
        let (g, v) = (self.higgs.g, self.higgs.v);
        let bps = self.higgs.lambda == 0.0;
        let mh = (2.0 * self.higgs.lambda).sqrt() * v;
        let c = self.higgs_c2().sqrt();
        let profiles = |r: f64| -> (f64, f64) {
            let xi = g * v * r;
            if bps {
                if xi < 1e-6 {
                    return (xi / 3.0, 1.0);
                }
                (1.0 / xi.tanh() - 1.0 / xi, xi / xi.sinh())
            } else {
                let h = (r * mh.max(g * v) / (2.0 * c.max(1e-9))).tanh();
                let k = if xi < 1e-6 { 1.0 } else { xi / xi.sinh() };
                (h, k)
            }
        };
        // Higgs
        for i in 0..n {
            let (px, py, pz) = self.coords(i);
            let d = [px as f64 - c0[0], py as f64 - c0[1], pz as f64 - c0[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let (h, _) = profiles(r);
            for k in 0..3 {
                self.higgs.phi[3 * i + k] = if r > 1e-9 { v * h * d[k] / r } else { 0.0 };
                self.higgs.pi[3 * i + k] = 0.0;
            }
        }
        // enlaces: A en el punto medio del enlace
        for a in 0..3 {
            for i in 0..n {
                let (px, py, pz) = self.coords(i);
                let mut d = [px as f64 - c0[0], py as f64 - c0[1], pz as f64 - c0[2]];
                d[a] += 0.5;
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let mut th = [0.0f64; 3];
                if r > 1e-9 {
                    let (_, k) = profiles(r);
                    let rh = [d[0] / r, d[1] / r, d[2] / r];
                    // A_a^c = ε_{c a j}·r̂_j·(1 − K)/(g·r); θ^c = −g·A_a^c (U = exp(+igA·T) = exp(−iθ·T))
                    for cc in 0..3 {
                        let mut s = 0.0;
                        for j in 0..3 {
                            s += levi(cc, a, j) * rh[j];
                        }
                        th[cc] = -s * (1.0 - k) / r;
                    }
                }
                let u = self.ym.alg.exp_minus_i(&th);
                self.ym.set_link(a * n + i, &u);
                self.ym.e[(a * n + i) * 3] = 0.0;
                self.ym.e[(a * n + i) * 3 + 1] = 0.0;
                self.ym.e[(a * n + i) * 3 + 2] = 0.0;
            }
        }
        // capa congelada de 2 celdas en el borde de la caja
        let dims = [self.w, self.h, self.d];
        for i in 0..n {
            let (px, py, pz) = self.coords(i);
            let p = [px, py, pz];
            let edge = (0..3).any(|k| p[k] < 2 || p[k] + 2 >= dims[k]);
            if edge {
                self.higgs.pinned[i] = 1;
                for k in 0..3 {
                    self.higgs.pin_val[3 * i + k] = self.higgs.phi[3 * i + k];
                }
                for a in 0..3 {
                    self.higgs.frozen_link[a * n + i] = 1;
                }
                self.higgs.has_pins = true;
                self.higgs.has_frozen = true;
            }
        }
        self.higgs_store_gauss_ref();
    }

    /// Par monopolo-antimonopolo (ansatz de Nambu) a lo largo de z, centrado en (x, y, z) y
    /// separado d: φ̂ gira de un polo al otro (Θ = θ₁ − θ₂), |φ| = v·h(r₁)·h(r₂) y
    /// A_i^a = ε_abc·φ̂_b·∂_iφ̂_c/g (D φ̂ = 0 lejos de los núcleos). Los núcleos (r < 1,5) quedan
    /// fijados hasta `clear_higgs_pins`.
    pub fn seed_monopole_pair(&mut self, x: f32, y: f32, z: f32, d: f32) {
        if self.higgs.mode != 3 {
            self.set_message("Primero activa el Higgs adjunto (set_adjoint_higgs).");
            return;
        }
        let n = self.n;
        let (g, v) = (self.higgs.g, self.higgs.v);
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let p1 = [x as f64, y as f64, z as f64 + 0.5 * d as f64];
        let p2 = [x as f64, y as f64, z as f64 - 0.5 * d as f64];
        let mw = self.higgs_c2().sqrt() * g * v;
        let _ = dims;
        // Desplazamientos sin imagen mínima (el par está centrado en la caja): en las costuras
        // φ̂ = ẑ por los dos lados. El giro se amortigua lejos del eje, Θ·exp(−ρ²/2σ²) con σ = d/2,
        // para que la cola dipolar no deje un salto en las costuras x e y del toro; cerca de los
        // polos el campo (y su grado topológico) no cambia.
        let sig2 = 2.0 * (0.5 * d as f64).powi(2).max(1.0);
        let hat = |p: [f64; 3]| -> ([f64; 3], f64) {
            let r1 = [p[0] - p1[0], p[1] - p1[1], p[2] - p1[2]];
            let r2 = [p[0] - p2[0], p[1] - p2[1], p[2] - p2[2]];
            let n1 = (r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]).sqrt().max(1e-9);
            let n2 = (r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]).sqrt().max(1e-9);
            let th1 = (r1[2] / n1).clamp(-1.0, 1.0).acos();
            let th2 = (r2[2] / n2).clamp(-1.0, 1.0).acos();
            let rho2 = r1[0] * r1[0] + r1[1] * r1[1];
            let big = (th1 - th2) * (-rho2 / sig2).exp();
            let ph = r1[1].atan2(r1[0]);
            let dir = [big.sin() * ph.cos(), big.sin() * ph.sin(), big.cos()];
            let h = (n1 * mw).tanh() * (n2 * mw).tanh();
            (dir, h)
        };
        for i in 0..n {
            let (px, py, pz) = self.coords(i);
            let (dir, h) = hat([px as f64, py as f64, pz as f64]);
            for k in 0..3 {
                self.higgs.phi[3 * i + k] = v * h * dir[k];
                self.higgs.pi[3 * i + k] = 0.0;
            }
        }
        for a in 0..3 {
            for i in 0..n {
                let (px, py, pz) = self.coords(i);
                let p = [px as f64, py as f64, pz as f64];
                let q0 = p;
                let mut q1 = p;
                q1[a] += 1.0;
                let mut mid = p;
                mid[a] += 0.5;
                let (n0, _) = hat(q0);
                let (n1, _) = hat(q1);
                let (nm, _) = hat(mid);
                // Con D = ∂ − ig[A, ·], D φ̂ = 0 pide A_a^c = −ε_cbe·φ̂_b·∂_aφ̂_e/g (el mismo signo que
                // el erizo, A_i^a = ε_aij·r̂_j/(g·r)); ∂_aφ̂ ≈ φ̂(r+â) − φ̂(r) y θ^c = −g·A_a^c
                let dphi = [n1[0] - n0[0], n1[1] - n0[1], n1[2] - n0[2]];
                let mut th = [0.0f64; 3];
                for cc in 0..3 {
                    let mut s = 0.0;
                    for b in 0..3 {
                        for e in 0..3 {
                            s += levi(cc, b, e) * nm[b] * dphi[e];
                        }
                    }
                    th[cc] = s;
                }
                let u = self.ym.alg.exp_minus_i(&th);
                self.ym.set_link(a * n + i, &u);
                for c in 0..3 {
                    self.ym.e[(a * n + i) * 3 + c] = 0.0;
                }
            }
        }
        self.clear_higgs_pins();
        self.pin_higgs_cells(p1[0] as f32, p1[1] as f32, p1[2] as f32, 1.5);
        self.pin_higgs_cells(p2[0] as f32, p2[1] as f32, p2[2] as f32, 1.5);
        self.higgs_store_gauss_ref();
    }

    /// Flujo de φ̂·B por la superficie de un cubo de semilado `half` centrado en (x, y, z), en
    /// unidades de 4π/g (proyección abeliana de 't Hooft: el ángulo de cada plaqueta alrededor de
    /// φ̂ en su esquina). En el Higgs abeliano, el flujo de B (con cuerdas) en unidades de 2π/q.
    pub fn monopole_flux(&self, x: f32, y: f32, z: f32, half: u32) -> f64 {
        if self.dim != 3 || !(x.is_finite() && y.is_finite() && z.is_finite()) {
            return 0.0;
        }
        let h = half.min(128) as i32;
        let (x0, y0, z0) = (x.round() as i32 - h, y.round() as i32 - h, z.round() as i32 - h);
        let (x1, y1, z1) = (x0.wrapping_add(2 * h), y0.wrapping_add(2 * h), z0.wrapping_add(2 * h));
        let mut total = 0.0;
        // caras normales a cada eje: plano (a, b) con la normal c = a × b
        let faces: [(usize, usize); 3] = [(1, 2), (2, 0), (0, 1)];
        for (axis, &(a, b)) in faces.iter().enumerate() {
            for side in 0..2 {
                let fixed = if side == 0 { [x0, y0, z0][axis] } else { [x1, y1, z1][axis] };
                let sign = if side == 0 { -1.0 } else { 1.0 };
                let lo = [x0, y0, z0];
                let hi = [x1, y1, z1];
                for s in lo[a]..hi[a] {
                    for t in lo[b]..hi[b] {
                        let mut p = [0i32; 3];
                        p[axis] = fixed;
                        p[a] = s;
                        p[b] = t;
                        let i = self.wrap_cell(p[0], p[1], p[2]);
                        total += sign * self.plaquette_angle(i, a, b);
                    }
                }
            }
        }
        if self.higgs.mode == 3 {
            total / (4.0 * PI)
        } else {
            let q = if self.higgs_q() != 0.0 { self.higgs_q() } else { 1.0 };
            total * q / (2.0 * PI)
        }
    }

    /// Celda de |φ| mínimo a menos de `radius` de (x, y, z): [x, y, z, |φ|] (sigue los polos).
    pub fn higgs_min_abs_near(&self, x: f32, y: f32, z: f32, radius: f32) -> Vec<f64> {
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let r2 = (radius as f64).powi(2);
        let nf = self.higgs.nf.max(1);
        let mut best = (f64::INFINITY, 0usize);
        if self.higgs.mode == 0 {
            return vec![0.0; 4];
        }
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let dx = min_image(px as f64 - x as f64, dims[0]);
            let dy = min_image(py as f64 - y as f64, dims[1]);
            let dz = if self.dim == 3 { min_image(pz as f64 - z as f64, dims[2]) } else { 0.0 };
            if dx * dx + dy * dy + dz * dz > r2 {
                continue;
            }
            let a: f64 = self.higgs.phi[i * nf..(i + 1) * nf].iter().map(|v| v * v).sum();
            if a < best.0 {
                best = (a, i);
            }
        }
        let (px, py, pz) = self.coords(best.1);
        vec![px as f64, py as f64, pz as f64, best.0.sqrt()]
    }

    /// Razón Σ½c²|DΦ|² / Σ(1/g²)·2κ(2 − Re Tr P) en la corteza r0 < r < r1 alrededor de (x, y, z)
    /// (Bogomolny: 1).
    pub fn higgs_bogomolny_ratio(&self, x: f32, y: f32, z: f32, r0: f32, r1: f32) -> f64 {
        if self.higgs.mode != 3 {
            return 0.0;
        }
        let n = self.n;
        let c2 = self.higgs_c2();
        let kap = self.em.kappa;
        let g2 = self.higgs.g * self.higgs.g;
        if g2 <= 1e-30 {
            return 0.0;
        }
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let (mut gr, mut mg) = (0.0, 0.0);
        for i in 0..n {
            let (px, py, pz) = self.coords(i);
            let dx = min_image(px as f64 + 0.5 - x as f64, dims[0]);
            let dy = min_image(py as f64 + 0.5 - y as f64, dims[1]);
            let dz = min_image(pz as f64 + 0.5 - z as f64, dims[2]);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < r0 as f64 || r >= r1 as f64 {
                continue;
            }
            for a in 0..3 {
                gr += 0.5 * c2 * self.higgs_grad2(i, a);
                for b in (a + 1)..3 {
                    mg += 2.0 * kap * (2.0 - self.ym_plaq_p(i, a, b).trace().re) / g2;
                }
            }
        }
        if mg > 0.0 {
            gr / mg
        } else {
            0.0
        }
    }

    /// B de la plaqueta (a, b) con esquina en (x, y, z) incluidas las cuerdas de Dirac (abeliano).
    /// Con la energía compacta, q·B se toma en (−π, π] (el flujo físico).
    pub fn higgs_plaquette_b(&self, x: i32, y: i32, z: i32, a: u32, b: u32) -> f64 {
        let i = self.wrap_cell(x, y, z);
        let (a, b) = ((a as usize).min(self.dim - 1), (b as usize).min(self.dim - 1));
        if a == b {
            return 0.0;
        }
        let bb = self.higgs_bt(i, a, b);
        let q = self.higgs_q();
        if self.em.compact && q != 0.0 {
            crate::util::wrap_angle(q * bb) / q
        } else {
            bb
        }
    }

    /// Puntero a φ (f64, nf por celda) y a π.
    pub fn higgs_phi_ptr(&self) -> *const f64 {
        self.higgs.phi.as_ptr()
    }

    pub fn higgs_phi_len(&self) -> u32 {
        self.higgs.phi.len() as u32
    }

    pub fn higgs_pi_ptr(&self) -> *const f64 {
        self.higgs.pi.as_ptr()
    }

    pub fn higgs_pi_len(&self) -> u32 {
        self.higgs.pi.len() as u32
    }

    pub fn higgs_components(&self) -> u32 {
        self.higgs.nf as u32
    }

    /// Sesgo lineal −ε·Re φ en el potencial del Higgs abeliano (falso vacío en −v, verdadero en +v;
    /// pensado para campo escalar real con q = 0: con q ≠ 0 rompe la simetría gauge U(1)).
    pub fn set_higgs_bias(&mut self, eps: f32) {
        self.higgs.bias = if eps.is_finite() { eps as f64 } else { 0.0 };
    }

    /// Siembra una burbuja esférica de vacío verdadero (Re φ > 0) de radio `radius` y grosor de
    /// pared `wall` rodeada de falso vacío (Re φ < 0) en el Higgs abeliano (Im φ = 0, π = 0).
    /// La expansión o colapso posterior es dinámica clásica de campo; la nucleación por túnel
    /// corresponde al módulo cuántico `false_vacuum_lockdown`.
    pub fn seed_vacuum_bubble(&mut self, cx_pos: f32, cy_pos: f32, cz_pos: f32, radius: f32, wall: f32) {
        if self.higgs.mode != 1 {
            return;
        }
        let v = self.higgs.v;
        let lam = self.higgs.lambda;
        let eps = self.higgs.bias;
        let newton = |mut x: f64| -> f64 {
            for _ in 0..20 {
                let f = 4.0 * lam * x * (x * x - v * v) - eps;
                let fp = 4.0 * lam * (3.0 * x * x - v * v);
                if fp.abs() < 1e-12 {
                    break;
                }
                x -= f / fp;
            }
            x
        };
        let phi_t = newton(v);
        let phi_f = newton(-v);
        let mid = 0.5 * (phi_t + phi_f);
        let half = 0.5 * (phi_t - phi_f);
        let rr = (radius as f64).max(0.5);
        let ww = (wall as f64).max(0.5);
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let dx = min_image(px as f64 - cx_pos as f64, dims[0]);
            let dy = min_image(py as f64 - cy_pos as f64, dims[1]);
            let dz = if self.dim == 3 { min_image(pz as f64 - cz_pos as f64, dims[2]) } else { 0.0 };
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            self.higgs.phi[2 * i] = mid - half * ((r - rr) / ww).tanh();
            self.higgs.phi[2 * i + 1] = 0.0;
            self.higgs.pi[2 * i] = 0.0;
            self.higgs.pi[2 * i + 1] = 0.0;
        }
        self.higgs_store_gauss_ref();
    }

    /// Fracción de celdas en el vacío verdadero (Re φ > 0) en el Higgs abeliano.
    pub fn higgs_true_vacuum_fraction(&self) -> f64 {
        if self.higgs.mode != 1 || self.n == 0 {
            return 0.0;
        }
        let cnt = (0..self.n).filter(|&i| self.higgs.phi[2 * i] > 0.0).count();
        cnt as f64 / self.n as f64
    }
}

impl QuantumSpacetimeUniverse {
    /// Ángulo de la plaqueta xy para el dibujo (q·B módulo 2π con la energía compacta).
    pub(crate) fn higgs_plaquette_angle_xy(&self, i: usize) -> f64 {
        let a = self.plaquette_angle(i, 0, 1);
        if self.higgs.mode != 3 && self.em.compact {
            crate::util::wrap_angle(a)
        } else {
            a
        }
    }

    /// Ángulo de la plaqueta (a, b) en r: abeliano q·B_total; adjunto 2·atan2(p·φ̂, p₀) con
    /// P = p₀ + i·p·σ en el marco de r.
    fn plaquette_angle(&self, i: usize, a: usize, b: usize) -> f64 {
        if self.higgs.mode == 3 {
            let p = self.ym_plaq_p(i, a, b);
            // P = p₀ + i·(p₁σ₁ + p₂σ₂ + p₃σ₃): P₀₀ = p₀ + i·p₃, P₀₁ = p₂ + i·p₁
            let (p0, p1, p2, p3) = (p.a[0].re, p.a[1].im, p.a[1].re, p.a[0].im);
            let f = &self.higgs.phi[3 * i..3 * i + 3];
            let nrm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt().max(1e-12);
            let proj = (p1 * f[0] + p2 * f[1] + p3 * f[2]) / nrm;
            2.0 * proj.atan2(p0)
        } else {
            self.higgs_q() * self.higgs_bt(i, a, b)
        }
    }
}

/// Símbolo de Levi-Civita ε_ijk.
#[inline(always)]
fn levi(i: usize, j: usize, k: usize) -> f64 {
    match (i, j, k) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,
        _ => 0.0,
    }
}
