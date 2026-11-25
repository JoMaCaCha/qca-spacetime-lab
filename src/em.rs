//! Modo electromagnetismo (§2.9 y §2.10): un campo gauge U(1) clásico en los enlaces, en la
//! formulación hamiltoniana de Wilson (Kogut–Susskind), acoplado a la caminata con fases de
//! Peierls. A_a(r) vive en el enlace r → r + â; B_ab(r) = A_a(r) + A_b(r+â) − A_a(r+b̂) − A_b(r).
//! Dinámica de Maxwell en gauge temporal con leapfrog y subpasos:
//!   E_a ← E_a − q·J_a (una vez); s veces: E_a ← E_a + (κ/s)·Σ_b [B_ab(r) − B_ab(r − b̂)], A ← A − E/s.
//! La ley de Gauss div E = q(ρ − ⟨ρ⟩) + (ρ_ext − ⟨ρ_ext⟩) se conserva exactamente (salvo el
//! redondeo de la materia en f32). Es electromagnetismo clásico real, no QED.

use crate::fftn::FftN;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct EmField {
    pub mode: u32,
    pub q: f64,
    pub kappa: f64,
    pub substeps: u32,
    pub eta: f64,
    /// Amortiguamiento fuera de los subpasos (solo para el test 107, que comprueba que diverge).
    pub damp_outside: bool,
    pub a: Vec<f64>,
    pub e: Vec<f64>,
    pub j: Vec<f64>,
    pub j_ext: Vec<f64>,
    pub has_j_ext: bool,
    pub j_ext_omega: f64,
    /// Las corrientes externas se encienden linealmente durante estos pasos (0 = de golpe).
    pub j_ext_ramp: u32,
    pub rho_ext: Vec<f64>,
    pub has_ext: bool,
    pub uniform_e: [f64; 3],
    /// (cos qA, sin qA) por enlace, recalculados cuando A cambia.
    pub ph: Vec<f32>,
    pub ph_valid: bool,
    pub book_capacity: bool,
    pub absorb: Vec<f32>,
    pub has_absorb: bool,
    pub energy_density: Vec<f32>,
    pub markers: Vec<f32>,
    pub fft: Option<FftN<f64>>,
    pub cbuf: Vec<Complex<f64>>,
    pub compact: bool,
    /// Fotón oscuro U(1)_D con mezcla cinética ε y masa m_A' (Fase 14; vacío mientras dp_active = false).
    pub dp_active: bool,
    pub dp_eps: f64,
    pub dp_mass: f64,
    pub dp_a: Vec<f64>,
    pub dp_e: Vec<f64>,
    pub dp_force: Vec<f64>,
    /// Vecinos r + â y r − â de cada celda (tablas, para no dividir en los bucles internos).
    pub nbp: Vec<u32>,
    pub nbm: Vec<u32>,
}

impl EmField {
    pub fn new(w: usize, h: usize, d: usize) -> Self {
        let dim = if d > 1 { 3 } else { 2 };
        let n = w * h * d;
        let mut nbp = vec![0u32; dim * n];
        let mut nbm = vec![0u32; dim * n];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let i = (z * h + y) * w + x;
                    let id = |xx: usize, yy: usize, zz: usize| ((zz * h + yy) * w + xx) as u32;
                    nbp[i] = id((x + 1) % w, y, z);
                    nbm[i] = id((x + w - 1) % w, y, z);
                    nbp[n + i] = id(x, (y + 1) % h, z);
                    nbm[n + i] = id(x, (y + h - 1) % h, z);
                    if dim == 3 {
                        nbp[2 * n + i] = id(x, y, (z + 1) % d);
                        nbm[2 * n + i] = id(x, y, (z + d - 1) % d);
                    }
                }
            }
        }
        EmField {
            mode: 0,
            q: 4e-3,
            kappa: 0.25,
            substeps: 1,
            eta: 0.0,
            damp_outside: false,
            a: Vec::new(),
            e: Vec::new(),
            j: Vec::new(),
            j_ext: Vec::new(),
            has_j_ext: false,
            j_ext_omega: 0.0,
            j_ext_ramp: 0,
            rho_ext: Vec::new(),
            has_ext: false,
            uniform_e: [0.0; 3],
            ph: Vec::new(),
            ph_valid: false,
            book_capacity: true,
            absorb: Vec::new(),
            has_absorb: false,
            energy_density: Vec::new(),
            markers: Vec::new(),
            fft: None,
            cbuf: Vec::new(),
            compact: false,
            dp_active: false,
            dp_eps: 0.0,
            dp_mass: 0.0,
            dp_a: Vec::new(),
            dp_e: Vec::new(),
            dp_force: Vec::new(),
            nbp,
            nbm,
        }
    }
}

impl QuantumSpacetimeUniverse {
    /// Reserva de forma diferida los búferes del campo electromagnético U(1).
    pub(crate) fn ensure_em_buffers(&mut self) {
        let links = self.dim * self.n;
        let n = self.n;
        if self.em.a.len() != links {
            self.em.a.resize(links, 0.0);
        }
        if self.em.e.len() != links {
            self.em.e.resize(links, 0.0);
        }
        if self.em.j.len() != links {
            self.em.j.resize(links, 0.0);
        }
        if self.em.j_ext.len() != links {
            self.em.j_ext.resize(links, 0.0);
        }
        if self.em.rho_ext.len() != n {
            self.em.rho_ext.resize(n, 0.0);
        }
        if self.em.ph.len() != 2 * links {
            self.em.ph.resize(2 * links, 0.0);
            self.em.ph_valid = false;
        }
        if self.em.absorb.len() != n {
            self.em.absorb.resize(n, 0.0);
        }
        if self.em.energy_density.len() != n {
            self.em.energy_density.resize(n, 0.0);
        }
    }

    /// Vecino r ± â (tabla precalculada).
    #[inline(always)]
    pub(crate) fn nb(&self, i: usize, axis: usize, dir: i32) -> usize {
        if dir > 0 {
            self.em.nbp[axis * self.n + i] as usize
        } else {
            self.em.nbm[axis * self.n + i] as usize
        }
    }

    /// B_ab(r) de la plaqueta (a, b) con esquina en r para un potencial dado.
    #[inline(always)]
    pub(crate) fn plaquette_b_from(&self, field: &[f64], i: usize, a: usize, b: usize) -> f64 {
        let n = self.n;
        let ia = self.nb(i, a, 1);
        let ib = self.nb(i, b, 1);
        field[a * n + i] + field[b * n + ia] - field[a * n + ib] - field[b * n + i]
    }

    /// B_ab(r) de la plaqueta (a, b) con esquina en r.
    #[inline(always)]
    pub(crate) fn plaquette_b(&self, i: usize, a: usize, b: usize) -> f64 {
        if self.em.a.len() < self.dim * self.n {
            return 0.0;
        }
        self.plaquette_b_from(&self.em.a, i, a, b)
    }

    pub(crate) fn em_refresh_phases(&mut self) {
        self.ensure_em_buffers();
        let q = self.em.q;
        let links = self.dim * self.n;
        for l in 0..links {
            let th = q * self.em.a[l];
            self.em.ph[2 * l] = th.cos() as f32;
            self.em.ph[2 * l + 1] = th.sin() as f32;
        }
        self.em.ph_valid = true;
    }

    /// Corrientes de la materia (Grover) tras la moneda y antes del desplazamiento:
    /// J_a(r) = |ψ_{+a}(r)|² − |ψ_{−a}(r + â)|².
    fn grover_currents<const NC: usize>(&mut self) {
        let stride = 2 * NC;
        let n = self.n;
        // canales que avanzan en +a y en −a: x: E(3)/W(2); y: S(1)/N(0); z: B(5)/F(4)
        let plus = [3usize, 1, 5];
        let minus = [2usize, 0, 4];
        for a in 0..self.dim {
            for i in 0..n {
                let ip = self.nb(i, a, 1);
                let bp = i * stride + 2 * plus[a];
                let bm = ip * stride + 2 * minus[a];
                let p = &self.buffer_read;
                let jp = (p[bp] as f64).powi(2) + (p[bp + 1] as f64).powi(2);
                let jm = (p[bm] as f64).powi(2) + (p[bm + 1] as f64).powi(2);
                self.em.j[a * n + i] = jp - jm;
            }
        }
    }

    /// Desplazamiento *gather* con las fases de Peierls: lo que avanza en el sentido del enlace
    /// toma e^{+iqA}, lo que va en contra, e^{−iqA}.
    pub(crate) fn shift_pass_u1<const NC: usize>(&mut self) {
        let stride = 2 * NC;
        let (w, h, d) = (self.w, self.h, self.d);
        let n = self.n;
        let cap = self.e_cap;
        self.cap_stack.clear();
        let ph = std::mem::take(&mut self.em.ph);
        {
            let src = &self.buffer_read;
            let dst = &mut self.buffer_write;
            let mul = |dst: &mut [f32], o: usize, s: &[f32], so: usize, c: f32, sn: f32| {
                let (re, im) = (s[so], s[so + 1]);
                dst[o] = re * c - im * sn;
                dst[o + 1] = re * sn + im * c;
            };
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
                        let b = i * stride;
                        // N(0) ← (x, y+1) con e^{−iqA_y(r)}
                        let l = n + i;
                        mul(dst, b, src, ((z * h + yp) * w + x) * stride, ph[2 * l], -ph[2 * l + 1]);
                        // S(1) ← (x, y−1) con e^{+iqA_y(x, y−1)}
                        let l = n + (z * h + ym) * w + x;
                        mul(dst, b + 2, src, ((z * h + ym) * w + x) * stride + 2, ph[2 * l], ph[2 * l + 1]);
                        // W(2) ← (x+1, y) con e^{−iqA_x(r)}
                        let l = i;
                        mul(dst, b + 4, src, ((z * h + y) * w + xp) * stride + 4, ph[2 * l], -ph[2 * l + 1]);
                        // E(3) ← (x−1, y) con e^{+iqA_x(x−1, y)}
                        let l = (z * h + y) * w + xm;
                        mul(dst, b + 6, src, l * stride + 6, ph[2 * l], ph[2 * l + 1]);
                        if NC == 6 {
                            let l = 2 * n + i;
                            mul(dst, b + 8, src, ((zp * h + y) * w + x) * stride + 8, ph[2 * l], -ph[2 * l + 1]);
                            let l = 2 * n + (zm * h + y) * w + x;
                            mul(dst, b + 10, src, ((zm * h + y) * w + x) * stride + 10, ph[2 * l], ph[2 * l + 1]);
                        }
                        if cap > 0.0 {
                            let mut e = 0.0f32;
                            for k in 0..stride {
                                e += dst[b + k] * dst[b + k];
                            }
                            if e > cap * (1.0 + crate::walk::CAP_TOLERANCE) {
                                self.cap_stack.push_back(i as u32);
                                self.cap_flag[i] = 1;
                            }
                        }
                    }
                }
            }
        }
        self.em.ph = ph;
        std::mem::swap(&mut self.buffer_read, &mut self.buffer_write);
    }

    /// Paso de la materia de Grover con U(1): corrientes, desplazamiento con fases.
    pub(crate) fn grover_u1_step<const NC: usize>(&mut self) {
        self.ensure_em_buffers();
        if self.em.mode == 2 {
            self.grover_currents::<NC>();
        }
        if !self.em.ph_valid {
            self.em_refresh_phases();
        }
        self.shift_pass_u1::<NC>();
    }

    /// Dinámica del campo tras el paso de la materia (y, con U(1) estático, el E uniforme externo).
    pub(crate) fn em_after_step(&mut self) {
        if self.em.mode == 0 {
            return;
        }
        self.ensure_em_buffers();
        let n = self.n;
        let dim = self.dim;
        match self.em.mode {
            1 => {
                let ue = self.em.uniform_e;
                if ue.iter().any(|v| *v != 0.0) {
                    for a in 0..dim {
                        if ue[a] != 0.0 {
                            for v in self.em.a[a * n..(a + 1) * n].iter_mut() {
                                *v -= ue[a];
                            }
                        }
                    }
                    self.em.ph_valid = false;
                }
            }
            2 => {
                let q = self.em.q;
                for l in 0..dim * n {
                    self.em.e[l] -= q * self.em.j[l];
                }
                if self.em.has_j_ext {
                    let f = if self.em.j_ext_omega != 0.0 { (self.em.j_ext_omega * self.t as f64).sin() } else { 1.0 };
                    let f = if self.em.j_ext_ramp > 0 { f * (self.t as f64 / self.em.j_ext_ramp as f64).min(1.0) } else { f };
                    for l in 0..dim * n {
                        self.em.e[l] -= f * self.em.j_ext[l];
                    }
                }
                self.maxwell_substeps();
                self.em.ph_valid = false;
            }
            _ => {}
        }
    }

    /// s subpasos de Maxwell (rotacional, A ← A − E/s), con el amortiguamiento dentro de cada uno.
    /// Si el fotón oscuro U(1)_D está activo (Fase 14), evoluciona el hamiltoniano exacto acoplado
    /// con mezcla cinética ε y masa m_A', conservando exactamente la ley de Gauss visible.
    pub(crate) fn maxwell_substeps(&mut self) {
        self.ensure_em_buffers();
        if self.em.dp_active {
            self.maxwell_dark_photon_substeps();
            return;
        }
        let n = self.n;
        let dim = self.dim;
        let s = self.em.substeps.max(1);
        let ks = self.em.kappa / s as f64;
        let compact = self.em.compact && self.em.q.abs() >= 1e-15;
        let q = self.em.q;
        let mut force = std::mem::take(&mut self.em.j); // la corriente ya se usó: búfer de trabajo
        for _ in 0..s {
            for a in 0..dim {
                for i in 0..n {
                    let mut f = 0.0;
                    for b in 0..dim {
                        if b == a {
                            continue;
                        }
                        let im = self.nb(i, b, -1);
                        let (b0, b1) = if compact {
                            ((q * self.plaquette_b(i, a, b)).sin() / q, (q * self.plaquette_b(im, a, b)).sin() / q)
                        } else {
                            (self.plaquette_b(i, a, b), self.plaquette_b(im, a, b))
                        };
                        f += b0 - b1;
                    }
                    force[a * n + i] = f;
                }
            }
            let damp = if self.em.damp_outside { 1.0 } else { 1.0 - self.em.eta };
            for l in 0..dim * n {
                self.em.e[l] = (self.em.e[l] + ks * force[l]) * damp;
            }
            if self.em.has_absorb {
                for a in 0..dim {
                    for i in 0..n {
                        let f = 1.0 - self.em.absorb[i] as f64;
                        self.em.e[a * n + i] *= f;
                        self.em.a[a * n + i] *= f;
                    }
                }
            }
            for l in 0..dim * n {
                self.em.a[l] -= self.em.e[l] / s as f64;
            }
        }
        if self.em.damp_outside && self.em.eta > 0.0 {
            for l in 0..dim * n {
                self.em.e[l] *= 1.0 - self.em.eta;
            }
        }
        for v in force.iter_mut() {
            *v = 0.0;
        }
        self.em.j = force;
    }

    /// Subpasos de Maxwell acoplados a un fotón oscuro U(1)_D con mezcla cinética ε y masa m_A':
    /// H = (E² + E_D² − 2ε·E·E_D)/(2(1−ε²)) + (κ/2)(B² + B_D² + 2ε·B·B_D) + (m_A'²/2)·A_D².
    /// La divergencia del rotacional es idénticamente nula, por lo que div E = q·ρ + ρ_ext sigue exacta.
    fn maxwell_dark_photon_substeps(&mut self) {
        let n = self.n;
        let dim = self.dim;
        let s = self.em.substeps.max(1);
        let dt = 1.0 / s as f64;
        let ks = self.em.kappa * dt;
        let eps = self.em.dp_eps;
        let inv = 1.0 / (1.0 - eps * eps);
        let m2 = self.em.dp_mass * self.em.dp_mass;
        let mut force_v = std::mem::take(&mut self.em.j);
        let mut force_d = std::mem::take(&mut self.em.dp_force);
        force_d.resize(dim * n, 0.0);
        for _ in 0..s {
            for a in 0..dim {
                for i in 0..n {
                    let mut fv = 0.0;
                    let mut fd = 0.0;
                    for b in 0..dim {
                        if b == a {
                            continue;
                        }
                        let im = self.nb(i, b, -1);
                        fv += self.plaquette_b_from(&self.em.a, i, a, b) - self.plaquette_b_from(&self.em.a, im, a, b);
                        fd += self.plaquette_b_from(&self.em.dp_a, i, a, b) - self.plaquette_b_from(&self.em.dp_a, im, a, b);
                    }
                    let l = a * n + i;
                    force_v[l] = fv;
                    force_d[l] = fd;
                }
            }
            let damp = if self.em.damp_outside { 1.0 } else { 1.0 - self.em.eta };
            for l in 0..dim * n {
                let fv = force_v[l];
                let fd = force_d[l];
                self.em.e[l] = (self.em.e[l] + ks * (fv + eps * fd)) * damp;
                self.em.dp_e[l] = (self.em.dp_e[l] + ks * (fd + eps * fv) + dt * m2 * self.em.dp_a[l]) * damp;
            }
            if self.em.has_absorb {
                for a in 0..dim {
                    for i in 0..n {
                        let f = 1.0 - self.em.absorb[i] as f64;
                        let l = a * n + i;
                        self.em.e[l] *= f;
                        self.em.a[l] *= f;
                        self.em.dp_e[l] *= f;
                        self.em.dp_a[l] *= f;
                    }
                }
            }
            for l in 0..dim * n {
                let ev = self.em.e[l];
                let ed = self.em.dp_e[l];
                self.em.a[l] -= dt * inv * (ev - eps * ed);
                self.em.dp_a[l] -= dt * inv * (ed - eps * ev);
            }
        }
        if self.em.damp_outside && self.em.eta > 0.0 {
            let f = 1.0 - self.em.eta;
            for l in 0..dim * n {
                self.em.e[l] *= f;
                self.em.dp_e[l] *= f;
            }
        }
        for v in force_v.iter_mut() {
            *v = 0.0;
        }
        self.em.j = force_v;
        self.em.dp_force = force_d;
    }

    /// Anota un reparto del tope como corriente en el enlace por el que pasa (§2.5).
    pub(crate) fn em_book(&mut self, from: usize, to: usize, axis: usize, dir: i32, amount: f64) {
        self.ensure_em_buffers();
        let n = self.n;
        let q = self.em.q;
        if dir > 0 {
            // de r a r + â: corriente +amount en el enlace r → r + â
            self.em.e[axis * n + from] -= q * amount;
        } else {
            // de r a r − â: corriente −amount en el enlace (r − â) → r
            self.em.e[axis * n + to] += q * amount;
        }
    }

    pub(crate) fn em_fft(&mut self) -> FftN<f64> {
        match self.em.fft.take() {
            Some(f) => f,
            None => FftN::<f64>::new(self.w, self.h, self.d),
        }
    }

    /// Resuelve −∇²_red φ = src (media nula) y suma −∇φ a E (diferencias hacia delante en los enlaces).
    pub(crate) fn em_add_gradient_field(&mut self, src: &[f64]) {
        self.ensure_em_buffers();
        let n = self.n;
        let mut fft = self.em_fft();
        if self.em.cbuf.len() != n {
            self.em.cbuf = vec![Complex::new(0.0, 0.0); n];
        }
        let mean = src.iter().sum::<f64>() / n as f64;
        let mut buf = std::mem::take(&mut self.em.cbuf);
        for i in 0..n {
            buf[i] = Complex::new(src[i] - mean, 0.0);
        }
        fft.forward(&mut buf);
        for i in 0..n {
            let (x, y, z) = self.coords(i);
            let mut lam = 2.0 * (1.0 - (2.0 * PI * x as f64 / self.w as f64).cos()) + 2.0 * (1.0 - (2.0 * PI * y as f64 / self.h as f64).cos());
            if self.dim == 3 {
                lam += 2.0 * (1.0 - (2.0 * PI * z as f64 / self.d as f64).cos());
            }
            buf[i] = if lam > 1e-12 { buf[i] / lam } else { Complex::new(0.0, 0.0) };
        }
        fft.inverse(&mut buf);
        for a in 0..self.dim {
            for i in 0..n {
                let ip = self.nb(i, a, 1);
                self.em.e[a * n + i] -= buf[ip].re - buf[i].re;
            }
        }
        self.em.cbuf = buf;
        self.em.fft = Some(fft);
    }

    /// Densidad de la materia para la ley de Gauss (f64).
    pub(crate) fn matter_density_f64(&self, i: usize) -> f64 {
        if self.sw.active {
            return self.sw_density(i);
        }
        let stride = self.stride();
        let b = i * stride;
        let mut e = 0.0f64;
        for k in 0..stride {
            let v = self.buffer_read[b + k] as f64;
            e += v * v;
        }
        e
    }

    /// Inicializa E con la ley de Gauss de la materia (y de las cargas externas) y pone A = 0.
    pub(crate) fn em_init_gauss(&mut self) {
        self.ensure_em_buffers();
        let n = self.n;
        for v in self.em.e.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.a.iter_mut() {
            *v = 0.0;
        }
        let q = self.em.q;
        let mut src: Vec<f64> = (0..n).map(|i| q * self.matter_density_f64(i)).collect();
        if self.em.has_ext {
            for i in 0..n {
                src[i] += self.em.rho_ext[i];
            }
        }
        self.em_add_gradient_field(&src);
        self.em.ph_valid = false;
    }

    pub(crate) fn em_gauss_residual(&self) -> f64 {
        let n = self.n;
        if self.em.e.len() < self.dim * n || self.em.rho_ext.len() < n {
            return 0.0;
        }
        let q = self.em.q;
        let mut rho_mean = 0.0;
        let mut ext_mean = 0.0;
        for i in 0..n {
            rho_mean += self.matter_density_f64(i);
            ext_mean += self.em.rho_ext[i];
        }
        rho_mean /= n as f64;
        ext_mean /= n as f64;
        let mut worst = 0.0f64;
        for i in 0..n {
            let mut div = 0.0;
            for a in 0..self.dim {
                let im = self.nb(i, a, -1);
                div += self.em.e[a * n + i] - self.em.e[a * n + im];
            }
            let r = div - q * (self.matter_density_f64(i) - rho_mean) - (self.em.rho_ext[i] - ext_mean);
            worst = worst.max(r.abs());
        }
        worst
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.9: 0 = desactivado, 1 = estático (A fijo), 2 = dinámico (Maxwell). κ ≤ s²/dim (se limita).
    /// Al pasar a dinámico, E se inicializa con la ley de Gauss (FFT) y A = 0.
    pub fn set_em(&mut self, mode: u32, q: f32, kappa: f32) {
        self.message.clear();
        if self.sm.active {
            // el modelo estándar fija su U(1)_Y (q = g′/2 para el Higgs, κ = 0,25): se configura con set_scenario_sm
            return;
        }
        let mode = mode.min(2);
        if mode != 0 {
            self.ensure_em_buffers();
        }
        let q_old = self.em.q;
        if q.is_finite() {
            self.em.q = q as f64;
        }
        // Cambiar q en modo dinámico cambia la ley de Gauss: se corrige E con el campo de Gauss
        // de Δq·(ρ − ⟨ρ⟩) (U(1)) o se toma la nueva referencia (SU(N)), sin tocar B ni la radiación.
        if self.em.mode == 2 && mode == 2 && self.em.q != q_old {
            if self.ncol > 1 {
                self.ym_store_gauss_ref();
            } else {
                let dq = self.em.q - q_old;
                let src: Vec<f64> = (0..self.n).map(|i| dq * self.matter_density_f64(i)).collect();
                self.em_add_gradient_field(&src);
                self.em.ph_valid = false;
            }
        }
        if kappa.is_finite() {
            let s = self.em.substeps.max(1) as f64;
            self.em.kappa = (kappa as f64).clamp(0.0, s * s / self.dim as f64);
        }
        let was = self.em.mode;
        self.em.mode = mode;
        if mode == 2 && was != 2 {
            if self.ncol > 1 {
                self.ym_init_gauss();
            } else {
                self.em_init_gauss();
            }
        }
        if mode == 0 {
            self.em.uniform_e = [0.0; 3];
        }
        self.em.ph_valid = false;
    }

    pub fn em_mode(&self) -> u32 {
        self.em.mode
    }
    pub fn em_q(&self) -> f64 {
        self.em.q
    }
    pub fn em_kappa(&self) -> f64 {
        self.em.kappa
    }

    /// §2.9: subpasos del campo, s ∈ 1–4. La velocidad de los fotones sigue siendo √κ.
    pub fn set_field_substeps(&mut self, s: u32) {
        if self.sm.active {
            return;
        }
        self.em.substeps = s.clamp(1, 4);
        let sf = self.em.substeps as f64;
        self.em.kappa = self.em.kappa.min(sf * sf / self.dim as f64);
    }

    pub fn field_substeps(&self) -> u32 {
        self.em.substeps
    }

    /// Amortiguamiento η de E (y, con el Higgs, de π) dentro de cada subpaso; 0 = conservativo.
    pub fn set_relaxation(&mut self, eta: f32) {
        if eta.is_finite() {
            self.em.eta = (eta as f64).clamp(0.0, 1.0);
        }
    }

    /// Tubo de flujo Φ en la plaqueta (x, y) con su compensador −Φ en (x, y + H/2) y la cuerda de
    /// Dirac entre ellos (A_x = −Φ en los enlaces que la cruzan). En 3D, en todos los planos z.
    pub fn add_flux_tube(&mut self, x: i32, y: i32, flux: f32) {
        if !flux.is_finite() || self.h < 2 {
            return;
        }
        self.ensure_em_buffers();
        let x0 = crate::util::wrap_i(x, self.w);
        let y0 = crate::util::wrap_i(y, self.h);
        let q = if self.em.q != 0.0 { self.em.q } else { 1.0 };
        let a_val = -(flux as f64) / q;
        for z in 0..self.d {
            for dy in 1..=self.h / 2 {
                let yy = (y0 + dy) % self.h;
                let i = self.idx(x0, yy, z);
                self.em.a[i] += a_val;
            }
        }
        if self.em.markers.len() + 6 <= 3 * 4096 {
            self.em.markers.push(x0 as f32 + 0.5);
            self.em.markers.push(y0 as f32 + 0.5);
            self.em.markers.push(1.0);
            self.em.markers.push(x0 as f32 + 0.5);
            self.em.markers.push(((y0 + self.h / 2) % self.h) as f32 + 0.5);
            self.em.markers.push(-1.0);
        }
        self.em.ph_valid = false;
    }

    /// B uniforme con n cuantos de flujo: q·B = 2πn/(W·H) por plaqueta, gauge de Landau
    /// A_y = B·x más A_x = −B·W·y en la costura x = W − 1 → 0. En 3D, en cada plano z.
    pub fn set_uniform_magnetic_field(&mut self, n_flux: i32) {
        self.ensure_em_buffers();
        let n = self.n;
        let (w, h) = (self.w, self.h);
        let q = if self.em.q != 0.0 { self.em.q } else { 1.0 };
        let b = 2.0 * PI * n_flux as f64 / (q * (w * h) as f64);
        for z in 0..self.d {
            for y in 0..h {
                for x in 0..w {
                    let i = self.idx(x, y, z);
                    self.em.a[n + i] += b * x as f64;
                    if x == w - 1 {
                        self.em.a[i] -= b * (w as f64) * y as f64;
                    }
                }
            }
        }
        self.em.ph_valid = false;
    }

    /// B por plaqueta del campo uniforme de n cuantos (q·B = 2πn/(W·H)).
    pub fn uniform_field_b(&self, n_flux: i32) -> f64 {
        let q = if self.em.q != 0.0 { self.em.q } else { 1.0 };
        2.0 * PI * n_flux as f64 / (q * (self.w * self.h) as f64)
    }

    /// Suma un E uniforme (el modo k = 0). Dinámico: se suma a E. Estático: es un campo externo
    /// que hace crecer A como −E·t (fuerza constante sin reacción).
    pub fn set_uniform_electric_field(&mut self, ex: f32, ey: f32, ez: f32) {
        if !(ex.is_finite() && ey.is_finite() && ez.is_finite()) {
            return;
        }
        let n = self.n;
        let v = [ex as f64, ey as f64, if self.dim == 3 { ez as f64 } else { 0.0 }];
        if self.em.mode == 2 {
            self.ensure_em_buffers();
            for a in 0..self.dim {
                for i in 0..n {
                    self.em.e[a * n + i] += v[a];
                }
            }
        } else {
            self.em.uniform_e = v;
        }
    }

    /// Carga externa fija (gaussiana de σ = radius, total `charge`): un núcleo infinitamente
    /// pesado. Se suma −∇φ_ext a E y se guarda ρ_ext para el residuo de Gauss.
    pub fn add_static_charge(&mut self, x: i32, y: i32, z: i32, charge: f32, radius: f32) {
        if !(charge.is_finite() && radius.is_finite()) {
            return;
        }
        self.ensure_em_buffers();
        let n = self.n;
        let sigma = (radius as f64).max(0.3);
        let c = self.wrap_cell(x, y, z);
        let (cx, cy, cz) = self.coords(c);
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let mut src = vec![0.0f64; n];
        let mut tot = 0.0;
        for i in 0..n {
            let (px, py, pz) = self.coords(i);
            let dx = crate::util::min_image(px as f64 - cx as f64, dims[0]);
            let dy = crate::util::min_image(py as f64 - cy as f64, dims[1]);
            let dz = if self.dim == 3 { crate::util::min_image(pz as f64 - cz as f64, dims[2]) } else { 0.0 };
            let e = (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp();
            src[i] = e;
            tot += e;
        }
        if tot <= 0.0 {
            return;
        }
        let s = charge as f64 / tot;
        for v in src.iter_mut() {
            *v *= s;
        }
        for i in 0..n {
            self.em.rho_ext[i] += src[i];
        }
        self.em.has_ext = true;
        self.em_add_gradient_field(&src);
        if self.em.markers.len() + 3 <= 3 * 4096 {
            self.em.markers.push(cx as f32 + 0.5);
            self.em.markers.push(cy as f32 + 0.5);
            self.em.markers.push(if charge >= 0.0 { 2.0 } else { -2.0 });
        }
    }

    /// A = E = 0, sin cargas ni corrientes externas.
    pub fn clear_em(&mut self) {
        for v in self.em.a.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.e.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.j.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.rho_ext.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.j_ext.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.dp_a.iter_mut() {
            *v = 0.0;
        }
        for v in self.em.dp_e.iter_mut() {
            *v = 0.0;
        }
        self.em.has_ext = false;
        self.em.has_j_ext = false;
        self.em.uniform_e = [0.0; 3];
        self.em.markers.clear();
        self.em.ph_valid = false;
        if self.ncol > 1 {
            let nn = self.ym.n;
            let m2 = 2 * nn * nn;
            for l in 0..self.dim * self.n {
                for k in 0..m2 {
                    self.ym.u[l * m2 + k] = 0.0;
                }
                for i in 0..nn {
                    self.ym.u[l * m2 + 2 * (i * nn + i)] = 1.0;
                }
            }
            self.ym.e.iter_mut().for_each(|v| *v = 0.0);
            self.ym.j.iter_mut().for_each(|v| *v = 0.0);
            self.ym.rho_ext.iter_mut().for_each(|v| *v = 0.0);
            self.ym.g_ref.iter_mut().for_each(|v| *v = 0.0);
            if self.em.mode == 2 {
                self.ym_init_gauss();
            }
        } else if self.em.mode == 2 {
            // En modo dinámico, E = 0 con materia cargada violaría la ley de Gauss: se reinicializa.
            self.em_init_gauss();
        }
    }

    /// Residuo de Gauss: max|div E − q(ρ − ⟨ρ⟩) − (ρ_ext − ⟨ρ_ext⟩)|.
    pub fn gauss_residual(&self) -> f64 {
        if self.higgs.mode != 0 {
            let r = self.higgs_gauss_residual_parts();
            return r[0].max(r[1]);
        }
        if self.em.mode == 0 {
            return 0.0;
        }
        if self.ncol > 1 {
            return self.ym_gauss_residual();
        }
        self.em_gauss_residual()
    }

    /// Energía del campo: ½Σ E(t−½)·E(t+½) + κ Σ B²/2 (o la compacta (1 − cos qB)/q²): la energía
    /// del leapfrog, con E(t+½) = E(t−½) + κ·F/s.
    pub fn em_energy(&mut self) -> f64 {
        if self.ncol > 1 {
            return self.ym_energy();
        }
        let n = self.n;
        if self.em.a.len() < self.dim * n || self.em.e.len() < self.dim * n {
            return 0.0;
        }
        let s = self.em.substeps.max(1) as f64;
        let sync = self.em.mode == 2;
        let q = self.em.q;
        let compact = self.em.compact && q.abs() >= 1e-15;
        let mut e2 = 0.0;
        for a in 0..self.dim {
            for i in 0..n {
                let e = self.em.e[a * n + i];
                if sync {
                    let mut f = 0.0;
                    for b in 0..self.dim {
                        if b == a {
                            continue;
                        }
                        let im = self.nb(i, b, -1);
                        f += if compact {
                            ((q * self.plaquette_b(i, a, b)).sin() - (q * self.plaquette_b(im, a, b)).sin()) / q
                        } else {
                            self.plaquette_b(i, a, b) - self.plaquette_b(im, a, b)
                        };
                    }
                    e2 += e * (e + self.em.kappa * f / s);
                    continue;
                }
                e2 += e * e;
            }
        }
        let mut b2 = 0.0;
        for i in 0..n {
            for a in 0..self.dim {
                for b in (a + 1)..self.dim {
                    let bb = self.plaquette_b(i, a, b);
                    b2 += if compact { 2.0 * (1.0 - (q * bb).cos()) / (q * q) } else { bb * bb };
                }
            }
        }
        0.5 * e2 + 0.5 * self.em.kappa * b2
    }

    /// rms de E por enlace (el HUD avisa si crece de forma exponencial).
    /// rms de E por enlace (SU(N): |Ẽ| por enlace, sumando las componentes de color).
    pub fn em_e_rms(&self) -> f64 {
        if self.ncol > 1 {
            let s: f64 = self.ym.e.iter().map(|v| v * v).sum();
            return (s / (self.dim * self.n).max(1) as f64).sqrt();
        }
        let s: f64 = self.em.e.iter().map(|v| v * v).sum();
        (s / self.em.e.len().max(1) as f64).sqrt()
    }

    /// Densidad de energía del campo por celda (½|E|² de sus enlaces + κB²/2 de sus plaquetas).
    pub fn compute_field_energy_density(&mut self) {
        if self.ncol > 1 {
            self.compute_colour_field_densities();
            return;
        }
        self.ensure_em_buffers();
        let n = self.n;
        for i in 0..n {
            let mut s = 0.0;
            for a in 0..self.dim {
                s += 0.5 * self.em.e[a * n + i].powi(2);
                for b in (a + 1)..self.dim {
                    s += 0.5 * self.em.kappa * self.plaquette_b(i, a, b).powi(2);
                }
            }
            self.em.energy_density[i] = s as f32;
        }
    }

    pub fn field_energy_density_ptr(&self) -> *const f32 {
        if self.ncol > 1 {
            return self.ym.energy_density.as_ptr();
        }
        self.em.energy_density.as_ptr()
    }

    /// Transformación gauge aleatoria: χ = amplitude·ξ, ψ → e^{iqχ}ψ, A_a → A_a + χ(r+â) − χ(r).
    pub fn apply_random_gauge_transform(&mut self, amplitude: f32, seed: u32) {
        if !amplitude.is_finite() {
            return;
        }
        if self.ncol > 1 {
            self.ym_random_gauge(amplitude, seed);
            return;
        }
        self.ensure_em_buffers();
        let n = self.n;
        let mut rng = crate::rng::SplitMix64::new(seed as u64 ^ 0x6A06E);
        let chi: Vec<f64> = (0..n).map(|_| amplitude as f64 * rng.gauss()).collect();
        let q = self.em.q;
        if self.higgs.mode == 1 {
            // φ → e^{iqχ}φ y π → e^{iqχ}π (la ley de Gauss y |φ| no cambian)
            for i in 0..n {
                let (s, c) = (q * chi[i]).sin_cos();
                for f in [&mut self.higgs.phi, &mut self.higgs.pi] {
                    let (re, im) = (f[2 * i], f[2 * i + 1]);
                    f[2 * i] = re * c - im * s;
                    f[2 * i + 1] = re * s + im * c;
                }
            }
        }
        if self.sw.active {
            let st = 2 * self.sw_namp();
            let mut bufs = vec![std::mem::take(&mut self.sw.psi)];
            bufs.extend(std::mem::take(&mut self.sw.parts));
            for b in bufs.iter_mut() {
                for i in 0..n {
                    let (s, c) = (q * chi[i]).sin_cos();
                    for k in 0..st / 2 {
                        let (re, im) = (b[i * st + 2 * k], b[i * st + 2 * k + 1]);
                        b[i * st + 2 * k] = re * c - im * s;
                        b[i * st + 2 * k + 1] = re * s + im * c;
                    }
                }
            }
            self.sw.psi = bufs.remove(0);
            self.sw.parts = bufs;
            self.sw_mirror();
        } else {
            let stride = self.stride();
            let namp = self.namp();
            for i in 0..n {
                let (c, s) = ((q * chi[i]).cos(), (q * chi[i]).sin());
                for k in 0..namp {
                    let re = self.buffer_read[i * stride + 2 * k] as f64;
                    let im = self.buffer_read[i * stride + 2 * k + 1] as f64;
                    self.buffer_read[i * stride + 2 * k] = (re * c - im * s) as f32;
                    self.buffer_read[i * stride + 2 * k + 1] = (re * s + im * c) as f32;
                }
            }
        }
        for a in 0..self.dim {
            for i in 0..n {
                let ip = self.nb(i, a, 1);
                self.em.a[a * n + i] += chi[ip] - chi[i];
            }
        }
        self.em.ph_valid = false;
    }

    /// Corriente externa fija en un enlace (axis 0/1/2): se resta de E en cada paso, como q·J.
    pub fn add_static_current(&mut self, x: i32, y: i32, z: i32, axis: u32, current: f32) {
        if !current.is_finite() {
            return;
        }
        self.ensure_em_buffers();
        let a = (axis as usize).min(self.dim - 1);
        let i = self.wrap_cell(x, y, z);
        self.em.j_ext[a * self.n + i] += current as f64;
        self.em.has_j_ext = true;
    }

    /// Las corrientes externas se encienden linealmente durante `steps` pasos desde t = 0 (0 = de golpe).
    pub fn set_current_ramp(&mut self, steps: u32) {
        self.em.j_ext_ramp = steps;
    }

    /// Hace oscilar las corrientes externas como sin(ωt) (antena dipolar); ω = 0 las deja fijas.
    pub fn set_current_oscillation(&mut self, omega: f32) {
        if omega.is_finite() {
            self.em.j_ext_omega = omega as f64;
        }
    }

    /// Capa absorbente de anchura `width` en los bordes del toro: E y A se multiplican por
    /// (1 − f) en cada subpaso, con f creciendo cuadráticamente hasta `strength`.
    pub fn set_absorbing_layer(&mut self, width: u32, strength: f32) {
        if !strength.is_finite() {
            return;
        }
        let strength = (strength as f64).clamp(0.0, 1.0);
        if width == 0 || strength == 0.0 {
            self.em.absorb.iter_mut().for_each(|v| *v = 0.0);
            self.em.has_absorb = false;
            return;
        }
        self.ensure_em_buffers();
        let wd = width as f64;
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let mut d = f64::INFINITY;
            let dims = [self.w, self.h, self.d];
            for (a, c) in [x, y, z].into_iter().enumerate() {
                if a == 2 && self.dim == 2 {
                    continue;
                }
                let l = dims[a];
                d = d.min(c as f64 + 0.5).min(l as f64 - c as f64 - 0.5);
            }
            self.em.absorb[i] = if d < wd { (strength * ((wd - d) / wd).powi(2)) as f32 } else { 0.0 };
        }
        self.em.has_absorb = true;
    }

    /// B_ab de la plaqueta (a, b) en una celda (a, b ∈ 0..dim).
    pub fn plaquette_field(&self, x: i32, y: i32, z: i32, a: u32, b: u32) -> f64 {
        let (a, b) = ((a as usize).min(self.dim - 1), (b as usize).min(self.dim - 1));
        if a == b {
            return 0.0;
        }
        self.plaquette_b(self.wrap_cell(x, y, z), a, b)
    }

    /// E_a en el enlace r → r + â.
    pub fn link_e(&self, x: i32, y: i32, z: i32, axis: u32) -> f64 {
        if self.em.e.len() < self.dim * self.n {
            return 0.0;
        }
        let a = (axis as usize).min(self.dim - 1);
        self.em.e[a * self.n + self.wrap_cell(x, y, z)]
    }

    /// A_a en el enlace r → r + â.
    pub fn link_a(&self, x: i32, y: i32, z: i32, axis: u32) -> f64 {
        if self.em.a.len() < self.dim * self.n {
            return 0.0;
        }
        let a = (axis as usize).min(self.dim - 1);
        self.em.a[a * self.n + self.wrap_cell(x, y, z)]
    }

    /// Fija A_a en un enlace (pulsos de fotones y montajes del laboratorio).
    pub fn set_link_a(&mut self, x: i32, y: i32, z: i32, axis: u32, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.ensure_em_buffers();
        let a = (axis as usize).min(self.dim - 1);
        let i = self.wrap_cell(x, y, z);
        self.em.a[a * self.n + i] = value;
        self.em.ph_valid = false;
    }

    /// Pulso gaussiano de A_axis(x) (σ = sigma, amplitud amp) centrado en x0, uniforme en y, z.
    pub fn add_photon_pulse(&mut self, x0: f32, sigma: f32, amp: f32, axis: u32) {
        if !(x0.is_finite() && sigma.is_finite() && amp.is_finite()) || sigma <= 0.0 {
            return;
        }
        self.ensure_em_buffers();
        let a = (axis as usize).min(self.dim - 1);
        let w = self.w as f64;
        for i in 0..self.n {
            let x = (i % self.w) as f64;
            let dx = crate::util::min_image(x - x0 as f64, w);
            self.em.a[a * self.n + i] += amp as f64 * (-(dx * dx) / (2.0 * (sigma as f64).powi(2))).exp();
        }
        self.em.ph_valid = false;
    }

    /// Copia el estado del campo U(1) de otro universo con la misma malla (modo, q, κ, s, A, E,
    /// cargas/corrientes externas, capa absorbente y fotón oscuro). Sirve para comparar dos evoluciones.
    pub fn copy_em_from(&mut self, other: &QuantumSpacetimeUniverse) {
        if (other.w, other.h, other.d) != (self.w, self.h, self.d) {
            return;
        }
        self.em.mode = other.em.mode;
        self.em.q = other.em.q;
        self.em.kappa = other.em.kappa;
        self.em.substeps = other.em.substeps;
        self.em.eta = other.em.eta;
        self.em.damp_outside = other.em.damp_outside;
        self.em.book_capacity = other.em.book_capacity;
        self.em.has_ext = other.em.has_ext;
        self.em.has_j_ext = other.em.has_j_ext;
        self.em.j_ext_omega = other.em.j_ext_omega;
        self.em.j_ext_ramp = other.em.j_ext_ramp;
        self.em.has_absorb = other.em.has_absorb;
        self.em.uniform_e = other.em.uniform_e;
        self.em.compact = other.em.compact;
        self.em.dp_active = other.em.dp_active;
        self.em.dp_eps = other.em.dp_eps;
        self.em.dp_mass = other.em.dp_mass;
        self.em.dp_a = other.em.dp_a.clone();
        self.em.dp_e = other.em.dp_e.clone();
        self.em.dp_force = vec![0.0; other.em.dp_force.len()];
        let links = self.dim * self.n;
        if other.em.a.len() == links {
            self.ensure_em_buffers();
            self.em.a.copy_from_slice(&other.em.a);
            self.em.e.copy_from_slice(&other.em.e);
            self.em.rho_ext.copy_from_slice(&other.em.rho_ext);
            self.em.j_ext.copy_from_slice(&other.em.j_ext);
            self.em.absorb.copy_from_slice(&other.em.absorb);
        } else {
            self.em.a.iter_mut().for_each(|v| *v = 0.0);
            self.em.e.iter_mut().for_each(|v| *v = 0.0);
            self.em.rho_ext.iter_mut().for_each(|v| *v = 0.0);
            self.em.j_ext.iter_mut().for_each(|v| *v = 0.0);
            self.em.absorb.iter_mut().for_each(|v| *v = 0.0);
        }
        self.em.ph_valid = false;
    }

    /// Marcadores de tubos de flujo y cargas externas: n × 3 floats [x, y, tipo].
    pub fn em_markers_ptr(&self) -> *const f32 {
        self.em.markers.as_ptr()
    }

    pub fn em_markers_len(&self) -> u32 {
        (self.em.markers.len() / 3) as u32
    }

    /// Fase 14: activa o desactiva el campo de fotón oscuro U(1)_D con mezcla cinética `eps`
    /// (|ε| < 0,95) y masa `mass` en la dinámica de Maxwell (modo EM 2). Pone a cero A_D y E_D.
    pub fn set_dark_photon(&mut self, on: bool, eps: f32, mass: f32) {
        self.em.dp_active = on;
        self.em.dp_eps = if eps.is_finite() { (eps as f64).clamp(-0.95, 0.95) } else { 0.0 };
        self.em.dp_mass = if mass.is_finite() { (mass as f64).max(0.0) } else { 0.0 };
        let links = self.dim * self.n;
        if on {
            self.ensure_em_buffers();
            self.em.dp_a.resize(links, 0.0);
            self.em.dp_e.resize(links, 0.0);
            self.em.dp_force.resize(links, 0.0);
            self.em.dp_a.fill(0.0);
            self.em.dp_e.fill(0.0);
        }
    }

    /// Energía exacta del sistema fotón visible + fotón oscuro U(1)_D:
    /// [E_total, E_vis_diag, E_dark_diag, E_dark_masa, E_mezcla].
    pub fn dark_photon_energy(&self) -> Vec<f64> {
        let n = self.n;
        let dim = self.dim;
        if self.em.a.len() < dim * n || self.em.e.len() < dim * n {
            return vec![0.0; 5];
        }
        let kap = self.em.kappa;
        if !self.em.dp_active {
            let mut ev = 0.0;
            for l in 0..dim * n {
                ev += 0.5 * self.em.e[l] * self.em.e[l];
            }
            for i in 0..n {
                for a in 0..dim {
                    for b in (a + 1)..dim {
                        let bb = self.plaquette_b(i, a, b);
                        ev += 0.5 * kap * bb * bb;
                    }
                }
            }
            return vec![ev, ev, 0.0, 0.0, 0.0];
        }
        if self.em.dp_a.len() < dim * n || self.em.dp_e.len() < dim * n {
            return vec![0.0; 5];
        }
        let eps = self.em.dp_eps;
        let inv = 1.0 / (1.0 - eps * eps);
        let m2 = self.em.dp_mass * self.em.dp_mass;
        let (mut ev_e, mut ed_e, mut emix_e, mut ed_m) = (0.0, 0.0, 0.0, 0.0);
        for l in 0..dim * n {
            let e = self.em.e[l];
            let ed = self.em.dp_e[l];
            let ad = self.em.dp_a[l];
            ev_e += 0.5 * inv * e * e;
            ed_e += 0.5 * inv * ed * ed;
            emix_e -= inv * eps * e * ed;
            ed_m += 0.5 * m2 * ad * ad;
        }
        let (mut ev_b, mut ed_b, mut emix_b) = (0.0, 0.0, 0.0);
        for i in 0..n {
            for a in 0..dim {
                for b in (a + 1)..dim {
                    let bv = self.plaquette_b_from(&self.em.a, i, a, b);
                    let bd = self.plaquette_b_from(&self.em.dp_a, i, a, b);
                    ev_b += 0.5 * kap * bv * bv;
                    ed_b += 0.5 * kap * bd * bd;
                    emix_b += kap * eps * bv * bd;
                }
            }
        }
        let e_vis = ev_e + ev_b;
        let e_dark = ed_e + ed_b + ed_m;
        let e_mix = emix_e + emix_b;
        vec![e_vis + e_dark + e_mix, e_vis, e_dark, ed_m, e_mix]
    }

    /// Potencial vectorial oscuro A_D en el enlace r → r + â.
    pub fn dark_link_a(&self, x: i32, y: i32, z: i32, axis: u32) -> f64 {
        if !self.em.dp_active || self.em.dp_a.len() < self.dim * self.n {
            return 0.0;
        }
        let a = (axis as usize).min(self.dim - 1);
        self.em.dp_a[a * self.n + self.wrap_cell(x, y, z)]
    }

    /// Campo eléctrico oscuro E_D en el enlace r → r + â.
    pub fn dark_link_e(&self, x: i32, y: i32, z: i32, axis: u32) -> f64 {
        if !self.em.dp_active || self.em.dp_e.len() < self.dim * self.n {
            return 0.0;
        }
        let a = (axis as usize).min(self.dim - 1);
        self.em.dp_e[a * self.n + self.wrap_cell(x, y, z)]
    }
}
