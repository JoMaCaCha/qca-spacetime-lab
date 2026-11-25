//! Escenario "Dirac" en 2D (§2.8): un espinor de 2 componentes por celda y el paso de
//! D'Ariano y Perinotti, ψ ← M(m)·T_y·T_x·ψ, con
//! T_x: ↑ avanza +x y ↓ avanza −x, es decir exp(−iσ_z k_x);
//! T_y = H·[↑ +y, ↓ −y]·H, es decir exp(−iσ_x k_y);
//! M(m) = exp(−i m σ_y) = [[cos m, −sin m], [sin m, cos m]].
//! Su hamiltoniano efectivo es h = σ_z k_x + σ_x k_y + m σ_y: Dirac en 2+1D con c = 1 celda/paso.
//! Dispersión exacta: cos ω = cos m·cos k_x·cos k_y + sin m·sin k_x·sin k_y.
//! Es el espín ½ de una sola partícula (función de onda), no un campo cuantizado.

use crate::util::min_image;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use wasm_bindgen::prelude::*;

pub type C64 = Complex<f64>;

/// Matriz 2×2 compleja.
pub type M2 = [[C64; 2]; 2];

#[inline]
pub fn c(re: f64, im: f64) -> C64 {
    Complex::new(re, im)
}

pub fn mat2_mul(a: &M2, b: &M2) -> M2 {
    let mut r = [[c(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    r
}

/// Operador del paso de Dirac 2D en Fourier, U(k) = M(m)·T_y(k_y)·T_x(k_x).
pub fn dirac2d_matrix(m: f64, kx: f64, ky: f64) -> M2 {
    let tx: M2 = [[C64::from_polar(1.0, -kx), c(0.0, 0.0)], [c(0.0, 0.0), C64::from_polar(1.0, kx)]];
    let ty: M2 = [[c(ky.cos(), 0.0), c(0.0, -ky.sin())], [c(0.0, -ky.sin()), c(ky.cos(), 0.0)]];
    let mm: M2 = [[c(m.cos(), 0.0), c(-m.sin(), 0.0)], [c(m.sin(), 0.0), c(m.cos(), 0.0)]];
    mat2_mul(&mm, &mat2_mul(&ty, &tx))
}

/// Autovalores e^{−iω} de una matriz 2×2 unitaria con det = 1 y el autovector de la banda pedida:
/// banda +1 → ω ∈ (0, π) (energía positiva), banda −1 → −ω. Devuelve (ω de esa banda, autovector).
pub fn su2_band(u: &M2, band: i32) -> (f64, [C64; 2]) {
    let tr = u[0][0] + u[1][1];
    let cw = (0.5 * tr.re).clamp(-1.0, 1.0);
    let w = cw.acos(); // ω ≥ 0
    let omega = if band >= 0 { w } else { -w };
    let lam = C64::from_polar(1.0, -omega);
    let v = eigvec2(u, lam);
    (omega, v)
}

/// Autovector normalizado de u para el autovalor lam.
pub fn eigvec2(u: &M2, lam: C64) -> [C64; 2] {
    let a = u[0][0] - lam;
    let b = u[0][1];
    let cc = u[1][0];
    let d = u[1][1] - lam;
    // (a b; c d)·v = 0: elegir la fila de mayor norma para mejor condicionamiento
    let r0 = a.norm_sqr() + b.norm_sqr();
    let r1 = cc.norm_sqr() + d.norm_sqr();
    let v = if r0 >= r1 && r0 > 1e-24 {
        [b, -a]
    } else if r1 > 1e-24 {
        [d, -cc]
    } else {
        [c(1.0, 0.0), c(0.0, 0.0)]
    };
    let n = (v[0].norm_sqr() + v[1].norm_sqr()).sqrt();
    if n < 1e-300 {
        return [c(1.0, 0.0), c(0.0, 0.0)];
    }
    [v[0] / n, v[1] / n]
}

/// Cuasi-energía positiva de la banda +1 en k (con la masa m).
pub fn dirac2d_omega(m: f64, kx: f64, ky: f64) -> f64 {
    let cw = (m.cos() * kx.cos() * ky.cos() + m.sin() * kx.sin() * ky.sin()).clamp(-1.0, 1.0);
    cw.acos()
}

/// Fase de Berry de la banda +1 en un círculo |k| = radius (lazo de Wilson discreto).
pub fn dirac2d_berry(m: f64, radius: f64, points: usize) -> f64 {
    let mut prod = c(1.0, 0.0);
    let first = su2_band(&dirac2d_matrix(m, radius, 0.0), 1).1;
    let mut prev = first;
    for j in 1..=points {
        let th = 2.0 * PI * j as f64 / points as f64;
        let v = if j == points { first } else { su2_band(&dirac2d_matrix(m, radius * th.cos(), radius * th.sin()), 1).1 };
        let ov = prev[0].conj() * v[0] + prev[1].conj() * v[1];
        prod *= ov;
        prev = v;
    }
    let ph = -prod.arg();
    let x = ph.rem_euclid(2.0 * PI);
    // la fase está definida módulo 2π; se devuelve su representante en [0, 2π) más cercano a ±π…
    if x > PI {
        2.0 * PI - x
    } else {
        x
    }
}

impl QuantumSpacetimeUniverse {
    /// Un paso de Dirac 2D: ψ ← e^{−iV}·M·T_y·T_x·ψ, con H aplicada en el intermedio. Con el campo
    /// U(1), las fases de Peierls se aplican en las dos recogidas y las corrientes se toman antes
    /// de cada salto (§2.10): J_x = |ψ↑(x)|² − |ψ↓(x+1)|², J_y = |a(y)|² − |b(y+1)|².
    pub(crate) fn dirac2d_step(&mut self) {
        let (w, h) = (self.w, self.h);
        let n = self.n;
        let s = FRAC_1_SQRT_2 as f32;
        let gauge = self.em.mode != 0;
        let dynamic = self.em.mode == 2;
        if gauge {
            self.ensure_em_buffers();
            if !self.em.ph_valid {
                self.em_refresh_phases();
            }
        }
        let ph = std::mem::take(&mut self.em.ph);
        if dynamic {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let ip = y * w + if x + 1 == w { 0 } else { x + 1 };
                    let p = &self.buffer_read;
                    let ju = (p[4 * i] as f64).powi(2) + (p[4 * i + 1] as f64).powi(2);
                    let jd = (p[4 * ip + 2] as f64).powi(2) + (p[4 * ip + 3] as f64).powi(2);
                    self.em.j[i] = ju - jd;
                }
            }
        }
        // pasada 1: T_x (recogida con e^{±iqA_x}) y H → buffer_write = (a, b)
        {
            let src = &self.buffer_read;
            let dst = &mut self.buffer_write;
            for y in 0..h {
                for x in 0..w {
                    let xm = if x == 0 { w - 1 } else { x - 1 };
                    let xp = if x + 1 == w { 0 } else { x + 1 };
                    let c = y * w + x;
                    let i = c * 4;
                    let iu = (y * w + xm) * 4; // ↑ viene de x − 1
                    let id = (y * w + xp) * 4 + 2; // ↓ viene de x + 1
                    let (mut ur, mut ui, mut dr, mut di) = (src[iu], src[iu + 1], src[id], src[id + 1]);
                    if gauge {
                        let lu = y * w + xm; // enlace (x − 1) → x: e^{+iqA_x(x−1)}
                        let (cu, su) = (ph[2 * lu], ph[2 * lu + 1]);
                        let (r, im) = (ur * cu - ui * su, ur * su + ui * cu);
                        ur = r;
                        ui = im;
                        let ld = c; // enlace x → x + 1: e^{−iqA_x(x)}
                        let (cd, sd) = (ph[2 * ld], -ph[2 * ld + 1]);
                        let (r, im) = (dr * cd - di * sd, dr * sd + di * cd);
                        dr = r;
                        di = im;
                    }
                    dst[i] = s * (ur + dr);
                    dst[i + 1] = s * (ui + di);
                    dst[i + 2] = s * (ur - dr);
                    dst[i + 3] = s * (ui - di);
                }
            }
        }
        if dynamic {
            for y in 0..h {
                let yp = if y + 1 == h { 0 } else { y + 1 };
                for x in 0..w {
                    let i = y * w + x;
                    let ip = yp * w + x;
                    let p = &self.buffer_write;
                    let ja = (p[4 * i] as f64).powi(2) + (p[4 * i + 1] as f64).powi(2);
                    let jb = (p[4 * ip + 2] as f64).powi(2) + (p[4 * ip + 3] as f64).powi(2);
                    self.em.j[n + i] = ja - jb;
                }
            }
        }
        // pasada 2: recogida en y (a de y − 1, b de y + 1), H, M y e^{−iV} → buffer_read
        let mass_field = self.dirac_mass_active();
        let (cm0, sm0) = (self.cos_m, self.sin_m);
        let has_v = self.has_potential;
        {
            for y in 0..h {
                let ym = if y == 0 { h - 1 } else { y - 1 };
                let yp = if y + 1 == h { 0 } else { y + 1 };
                for x in 0..w {
                    let c = y * w + x;
                    let i = c * 4;
                    let ia = (ym * w + x) * 4;
                    let ib = (yp * w + x) * 4 + 2;
                    let src = &self.buffer_write;
                    let (mut ar, mut ai, mut br, mut bi) = (src[ia], src[ia + 1], src[ib], src[ib + 1]);
                    if gauge {
                        let la = n + ym * w + x; // enlace (y − 1) → y: e^{+iqA_y(y−1)}
                        let (ca, sa) = (ph[2 * la], ph[2 * la + 1]);
                        let (r, im) = (ar * ca - ai * sa, ar * sa + ai * ca);
                        ar = r;
                        ai = im;
                        let lb = n + c; // enlace y → y + 1: e^{−iqA_y(y)}
                        let (cb, sb) = (ph[2 * lb], -ph[2 * lb + 1]);
                        let (r, im) = (br * cb - bi * sb, br * sb + bi * cb);
                        br = r;
                        bi = im;
                    }
                    let ur = s * (ar + br);
                    let ui = s * (ai + bi);
                    let dr = s * (ar - br);
                    let di = s * (ai - bi);
                    let (cm, sm) = if mass_field { self.dirac_cs(c) } else { (cm0, sm0) };
                    let (mut o0, mut o1, mut o2, mut o3) = (cm * ur - sm * dr, cm * ui - sm * di, sm * ur + cm * dr, sm * ui + cm * di);
                    if has_v {
                        let v = self.potential[c];
                        if v != 0.0 {
                            let (cv, sv) = crate::util::fast_cos_sin(v);
                            // multiplicar por e^{−iV}
                            let (a0, a1) = (o0 * cv + o1 * sv, o1 * cv - o0 * sv);
                            let (a2, a3) = (o2 * cv + o3 * sv, o3 * cv - o2 * sv);
                            o0 = a0;
                            o1 = a1;
                            o2 = a2;
                            o3 = a3;
                        }
                    }
                    let dst = &mut self.buffer_read;
                    dst[i] = o0;
                    dst[i + 1] = o1;
                    dst[i + 2] = o2;
                    dst[i + 3] = o3;
                }
            }
        }
        self.em.ph = ph;
        self.t = self.t.wrapping_add(1);
    }

    /// Construye un paquete gaussiano de Dirac 2D (norma 1) con el espinor de la banda pedida en k₀.
    pub(crate) fn add_dirac_packet(&mut self, x0: f64, y0: f64, sigma: f64, kx: f64, ky: f64, band: i32) {
        let spinor = if band == 0 {
            [c(1.0, 0.0), c(0.0, 0.0)]
        } else {
            su2_band(&dirac2d_matrix(self.mass as f64, kx, ky), band.signum()).1
        };
        let (w, h) = (self.w as f64, self.h as f64);
        let mut norm = 0.0f64;
        let rc = 4.0 * sigma;
        for i in 0..self.w * self.h {
            let (x, y, _) = self.coords(i);
            let dx = min_image(x as f64 - x0, w);
            let dy = min_image(y as f64 - y0, h);
            if dx.abs() > rc || dy.abs() > rc {
                continue;
            }
            let e = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            norm += e * e;
        }
        if norm <= 0.0 {
            return;
        }
        let amp = 1.0 / norm.sqrt();
        let (ax, ay) = self.packet_gauge_phase(x0, y0);
        let nco = self.ncol.max(1);
        let stride = self.stride();
        if self.buffer_read.len() < self.w * self.h * stride {
            return;
        }
        for i in 0..self.w * self.h {
            let (x, y, _) = self.coords(i);
            let dx = min_image(x as f64 - x0, w);
            let dy = min_image(y as f64 - y0, h);
            if dx.abs() > rc || dy.abs() > rc {
                continue;
            }
            let e = amp * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            let ph = C64::from_polar(e, (kx + ax) * dx + (ky + ay) * dy);
            let b = i * stride;
            for comp in 0..2 {
                let v = ph * spinor[comp];
                let o = b + 2 * (comp * nco);
                self.buffer_read[o] += v.re as f32;
                self.buffer_read[o + 1] += v.im as f32;
            }
        }
    }

    /// q·A(r₀) promediado en los enlaces de la celda central (paquetes con momento cinético, §2.10):
    /// la envolvente se multiplica por e^{iq·A(r₀)·(r − r₀)}.
    pub(crate) fn packet_gauge_phase(&self, x0: f64, y0: f64) -> (f64, f64) {
        if self.em.mode == 0 || self.em.a.len() < 2 * self.n {
            return (0.0, 0.0);
        }
        let c = self.wrap_cell(x0.round() as i32, y0.round() as i32, 0);
        let n = self.n;
        let q = self.em.q;
        let ax = 0.5 * (self.em.a[c] + self.em.a[self.nb(c, 0, -1)]);
        let ay = 0.5 * (self.em.a[n + c] + self.em.a[n + self.nb(c, 1, -1)]);
        (q * ax, q * ay)
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Paquete gaussiano de Dirac (norma 1) con el espinor propio de U(k₀) de la banda ±1, o el
    /// espinor ↑ puro con band = 0 (mezcla las dos bandas: Zitterbewegung). Escenario Dirac 2D.
    pub fn inject_dirac_packet(&mut self, x: f32, y: f32, sigma: f32, kx: f32, ky: f32, band: i32) {
        self.message.clear();
        if self.scenario != 2 || self.dim != 2 {
            self.set_message("inject_dirac_packet necesita el escenario Dirac en 2D.");
            return;
        }
        if !(x.is_finite() && y.is_finite() && sigma.is_finite() && kx.is_finite() && ky.is_finite()) || sigma <= 0.0 {
            return;
        }
        self.add_dirac_packet(x as f64, y as f64, sigma as f64, kx as f64, ky as f64, band.clamp(-1, 1));
    }

    /// Cuasi-energía positiva de la banda +1 del paso de Dirac 2D en k, con la masa actual.
    pub fn dirac_dispersion(&self, kx: f32, ky: f32) -> f64 {
        if !(kx.is_finite() && ky.is_finite()) {
            return f64::NAN;
        }
        dirac2d_omega(self.mass as f64, kx as f64, ky as f64)
    }

    /// Fase de Berry de la banda +1 en un círculo |k| = radius (lazo de Wilson con `points` puntos).
    pub fn dirac_berry_phase(&self, radius: f32, points: u32) -> f64 {
        if !radius.is_finite() {
            return f64::NAN;
        }
        dirac2d_berry(self.mass as f64, radius as f64, points.clamp(8, 100_000) as usize)
    }

    /// ⟨σ_a⟩ total (a = 0: x, 1: y, 2: z) del espinor de Dirac 2D (normalizado por la norma).
    pub fn dirac_spin(&self, axis: u32) -> f64 {
        let nco = self.ncol.max(1);
        let stride = self.stride();
        if self.nc < 2 || self.buffer_read.len() < self.w * self.h * stride {
            return 0.0;
        }
        let (mut s, mut n) = (0.0f64, 0.0f64);
        for i in 0..self.w * self.h {
            let b = i * stride;
            for a in 0..nco {
                let ou = b + 2 * a;
                let od = b + 2 * (nco + a);
                let u = c(self.buffer_read[ou] as f64, self.buffer_read[ou + 1] as f64);
                let d = c(self.buffer_read[od] as f64, self.buffer_read[od + 1] as f64);
                n += u.norm_sqr() + d.norm_sqr();
                s += match axis {
                    0 => 2.0 * (u.conj() * d).re,
                    1 => 2.0 * (u.conj() * d).im,
                    _ => u.norm_sqr() - d.norm_sqr(),
                };
            }
        }
        if n > 0.0 {
            s / n
        } else {
            0.0
        }
    }
}
