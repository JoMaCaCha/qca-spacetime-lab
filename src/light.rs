//! Escenario luz (§2.19): un fermión sin masa de 2 componentes con una velocidad de la luz local,
//! para métricas estáticas y diagonales ds² = dt² − n(r)²·dx² (que no son conformemente planas) y
//! para métricas uniformes que dependen del tiempo.
//! - Pieza 1D: U = S_z(k)·exp(−iθσ_x); dos pasos cerca de k = π/2 dan un Dirac sin masa de velocidad
//!   cos θ con eje n_θ = (0, sin θ, cos θ).
//! - Ronda: W = [R_y·U_y²·R_y†]·[R_x·U_x²·R_x†], con R_x = cos(θ/2) − i·sin(θ/2)·σ_x (el eje pasa a
//!   σ_z), U_y con el desplazamiento en la base σ_x y moneda exp(−iθσ_z), y R_y = cos(θ/2) −
//!   i·sin(θ/2)·σ_z. Cerca de k = (π/2, π/2): un cono isótropo de velocidad cos θ/2 por paso.
//! - Con θ(r): c(r) = cos θ(r)/2 y n(r) = c₀/cos θ(r). Es la propagación de la luz en una métrica
//!   dada: no resuelve las ecuaciones de Einstein y no sirve para una partícula con masa (§3.17).

use crate::fftn::{wavenumber, FftN};
use crate::linalg::eig;
use crate::spinor::cx;
use crate::sun::C;
use crate::util::min_image;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use wasm_bindgen::prelude::*;

pub struct Light {
    pub theta0: f64,
    /// θ por celda de cada media ronda (x e y) y sus funciones precalculadas.
    pub th: [Vec<f64>; 2],
    pub cs: [Vec<f64>; 2],
    pub sn: [Vec<f64>; 2],
    pub ch: [Vec<f64>; 2],
    pub sh: [Vec<f64>; 2],
    pub psi: Vec<C>,
    pub tmp: Vec<C>,
    pub gm: f64,
    pub src: [f64; 2],
    pub soft: f64,
    /// 0 = nada, 1 = expansión, 2 = onda gravitacional.
    pub metric_kind: u32,
    pub metric_amp: f64,
    pub metric_period: f64,
    pub rounds: u32,
    /// Centroide sin envolver (seguido de ronda en ronda) y el último centroide circular.
    pub unwrapped: [f64; 2],
    pub last: [f64; 2],
    pub tracking: bool,
}

impl Light {
    pub fn new() -> Self {
        Light {
            theta0: 0.3,
            th: [Vec::new(), Vec::new()],
            cs: [Vec::new(), Vec::new()],
            sn: [Vec::new(), Vec::new()],
            ch: [Vec::new(), Vec::new()],
            sh: [Vec::new(), Vec::new()],
            psi: Vec::new(),
            tmp: Vec::new(),
            gm: 0.0,
            src: [0.0; 2],
            soft: 4.0,
            metric_kind: 0,
            metric_amp: 0.0,
            metric_period: 1.0,
            rounds: 0,
            unwrapped: [0.0; 2],
            last: [0.0; 2],
            tracking: false,
        }
    }

    fn set_theta(&mut self, half: usize, i: usize, th: f64) {
        self.th[half][i] = th;
        self.cs[half][i] = th.cos();
        self.sn[half][i] = th.sin();
        self.ch[half][i] = (0.5 * th).cos();
        self.sh[half][i] = (0.5 * th).sin();
    }

    /// θ_x y θ_y uniformes de la métrica que depende del tiempo, en el instante t (rondas).
    pub fn metric_thetas(&self, t: f64) -> (f64, f64) {
        let c0 = self.theta0.cos();
        match self.metric_kind {
            1 => {
                let a = (1.0 + t / self.metric_period.max(1e-9)).powf(2.0 / 3.0).min(self.metric_amp.max(1.0));
                let th = (c0 / a).clamp(-1.0, 1.0).acos();
                (th, th)
            }
            2 => {
                let h = self.metric_amp * (2.0 * PI * t / self.metric_period.max(1e-9)).sin();
                ((c0 / (1.0 + h).sqrt()).clamp(-1.0, 1.0).acos(), (c0 / (1.0 - h).sqrt()).clamp(-1.0, 1.0).acos())
            }
            _ => (self.theta0, self.theta0),
        }
    }
}

type M2 = [[C; 2]; 2];

fn m2_mul(a: &M2, b: &M2) -> M2 {
    let mut r = [[cx(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    r
}

/// Operador de una ronda en Fourier con θ_x y θ_y uniformes.
pub fn light_round_matrix(thx: f64, thy: f64, k: [f64; 2]) -> M2 {
    let z = cx(0.0, 0.0);
    let rot = |th: f64, sz: bool| -> M2 {
        // cos(θ/2) − i·sin(θ/2)·σ (σ_x o σ_z)
        let (c, s) = ((0.5 * th).cos(), (0.5 * th).sin());
        if sz {
            [[cx(c, -s), z], [z, cx(c, s)]]
        } else {
            [[cx(c, 0.0), cx(0.0, -s)], [cx(0.0, -s), cx(c, 0.0)]]
        }
    };
    let dag = |m: &M2| -> M2 { [[m[0][0].conj(), m[1][0].conj()], [m[0][1].conj(), m[1][1].conj()]] };
    // U_x = S_x·exp(−iθσ_x)
    let sx: M2 = [[C::from_polar(1.0, -k[0]), z], [z, C::from_polar(1.0, k[0])]];
    let cxm: M2 = [[cx(thx.cos(), 0.0), cx(0.0, -thx.sin())], [cx(0.0, -thx.sin()), cx(thx.cos(), 0.0)]];
    let ux = m2_mul(&sx, &cxm);
    // U_y = H·diag(e^{−ik_y}, e^{ik_y})·H·exp(−iθσ_z)
    let s = FRAC_1_SQRT_2;
    let hm: M2 = [[cx(s, 0.0), cx(s, 0.0)], [cx(s, 0.0), cx(-s, 0.0)]];
    let dy: M2 = [[C::from_polar(1.0, -k[1]), z], [z, C::from_polar(1.0, k[1])]];
    let sy = m2_mul(&hm, &m2_mul(&dy, &hm));
    let czm: M2 = [[C::from_polar(1.0, -thy), z], [z, C::from_polar(1.0, thy)]];
    let uy = m2_mul(&sy, &czm);
    let rx = rot(thx, false);
    let ry = rot(thy, true);
    let wx = m2_mul(&rx, &m2_mul(&ux, &m2_mul(&ux, &dag(&rx))));
    let wy = m2_mul(&ry, &m2_mul(&uy, &m2_mul(&uy, &dag(&ry))));
    m2_mul(&wy, &wx)
}

/// Cuasi-energía (por ronda) y autovector de la banda superior (la de mayor ω = −arg λ).
pub fn light_upper(thx: f64, thy: f64, k: [f64; 2]) -> (f64, [C; 2]) {
    let w = light_round_matrix(thx, thy, k);
    let e = eig(vec![w[0][0], w[0][1], w[1][0], w[1][1]], 2);
    let (a, b) = (-e.values[0].arg(), -e.values[1].arg());
    let i = if a >= b { 0 } else { 1 };
    (a.max(b), [e.vectors[i][0], e.vectors[i][1]])
}

impl QuantumSpacetimeUniverse {
    pub(crate) fn light_alloc(&mut self) {
        let n = self.w * self.h;
        let th0 = self.light.theta0;
        for half in 0..2 {
            self.light.th[half] = vec![0.0; n];
            self.light.cs[half] = vec![0.0; n];
            self.light.sn[half] = vec![0.0; n];
            self.light.ch[half] = vec![0.0; n];
            self.light.sh[half] = vec![0.0; n];
            for i in 0..n {
                self.light.set_theta(half, i, th0);
            }
        }
        self.light.psi = vec![cx(0.0, 0.0); n];
        self.light.psi.extend(vec![cx(0.0, 0.0); n]);
        self.light.tmp = vec![cx(0.0, 0.0); 2 * n];
    }

    /// Una ronda W = [R_y·U_y²·R_y†]·[R_x·U_x²·R_x†] (4 pasos de caminata).
    pub(crate) fn light_step(&mut self) {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        if self.light.metric_kind != 0 {
            let (tx, ty) = self.light.metric_thetas(self.light.rounds as f64 + 0.5);
            for i in 0..n {
                self.light.set_theta(0, i, tx);
                self.light.set_theta(1, i, ty);
            }
        }
        let mut psi = std::mem::take(&mut self.light.psi);
        let mut tmp = std::mem::take(&mut self.light.tmp);
        let l = &self.light;
        // --- mitad x: R_x†, (moneda, desplazamiento)², R_x
        let rot_x = |p: &mut [C], i: usize, dagger: bool| {
            let (c, s) = (l.ch[0][i], if dagger { l.sh[0][i] } else { -l.sh[0][i] });
            // cos(θ/2) ∓ i·sin(θ/2)·σ_x
            let (a, b) = (p[i], p[n + i]);
            p[i] = a * c + b * cx(0.0, s);
            p[n + i] = b * c + a * cx(0.0, s);
        };
        for i in 0..n {
            rot_x(&mut psi, i, true);
        }
        for _ in 0..2 {
            for i in 0..n {
                let (c, s) = (l.cs[0][i], l.sn[0][i]);
                let (a, b) = (psi[i], psi[n + i]);
                psi[i] = a * c + b * cx(0.0, -s);
                psi[n + i] = b * c + a * cx(0.0, -s);
            }
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let xm = y * w + if x == 0 { w - 1 } else { x - 1 };
                    let xp = y * w + if x + 1 == w { 0 } else { x + 1 };
                    tmp[i] = psi[xm];
                    tmp[n + i] = psi[n + xp];
                }
            }
            std::mem::swap(&mut psi, &mut tmp);
        }
        for i in 0..n {
            rot_x(&mut psi, i, false);
        }
        // --- mitad y: R_y†, (moneda e^{−iθσ_z}, desplazamiento en la base σ_x)², R_y
        let rot_y = |p: &mut [C], i: usize, dagger: bool| {
            let half = if dagger { 0.5 * l.th[1][i] } else { -0.5 * l.th[1][i] };
            p[i] *= C::from_polar(1.0, half);
            p[n + i] *= C::from_polar(1.0, -half);
        };
        for i in 0..n {
            rot_y(&mut psi, i, true);
        }
        let s = FRAC_1_SQRT_2;
        for _ in 0..2 {
            for i in 0..n {
                let th = l.th[1][i];
                let (a, b) = (psi[i] * C::from_polar(1.0, -th), psi[n + i] * C::from_polar(1.0, th));
                // a la base σ_x: (+, −) = H·(a, b)
                psi[i] = (a + b) * s;
                psi[n + i] = (a - b) * s;
            }
            for y in 0..h {
                let ym = if y == 0 { h - 1 } else { y - 1 };
                let yp = if y + 1 == h { 0 } else { y + 1 };
                for x in 0..w {
                    let i = y * w + x;
                    let (p, m) = (psi[ym * w + x], psi[n + yp * w + x]);
                    tmp[i] = (p + m) * s;
                    tmp[n + i] = (p - m) * s;
                }
            }
            std::mem::swap(&mut psi, &mut tmp);
        }
        for i in 0..n {
            rot_y(&mut psi, i, false);
        }
        self.light.psi = psi;
        self.light.tmp = tmp;
        self.light.rounds += 1;
        self.t = self.t.wrapping_add(4);
        if self.light.tracking {
            let c = self.light_circular_centroid();
            for a in 0..2 {
                let l = [w as f64, h as f64][a];
                self.light.unwrapped[a] += min_image(c[a] - self.light.last[a], l);
            }
            self.light.last = c;
        }
    }

    /// Centroide circular de |ψ|².
    pub(crate) fn light_circular_centroid(&self) -> [f64; 2] {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let (mut cxs, mut sxs, mut cys, mut sys) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let e = self.light.psi[i].norm_sqr() + self.light.psi[n + i].norm_sqr();
            if e == 0.0 {
                continue;
            }
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let (ax, ay) = (2.0 * PI * x / w as f64, 2.0 * PI * y / h as f64);
            cxs += e * ax.cos();
            sxs += e * ax.sin();
            cys += e * ay.cos();
            sys += e * ay.sin();
        }
        [(sxs.atan2(cxs) * w as f64 / (2.0 * PI)).rem_euclid(w as f64), (sys.atan2(cys) * h as f64 / (2.0 * PI)).rem_euclid(h as f64)]
    }

    fn light_inject(&mut self, x0: f64, y0: f64, sx: f64, sy: f64, k0: [f64; 2]) {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let th0 = self.light.theta0;
        let (_, u0) = light_upper(th0, th0, k0);
        let mut a: Vec<C> = vec![cx(0.0, 0.0); n];
        let mut b: Vec<C> = vec![cx(0.0, 0.0); n];
        for i in 0..n {
            let dx = min_image((i % w) as f64 - x0, w as f64);
            let dy = min_image((i / w) as f64 - y0, h as f64);
            let arg = dx * dx / (2.0 * sx * sx) + dy * dy / (2.0 * sy * sy);
            if arg > 40.0 {
                continue;
            }
            let e = C::from_polar((-arg).exp(), k0[0] * dx + k0[1] * dy);
            a[i] = e * u0[0];
            b[i] = e * u0[1];
        }
        let mut fft = FftN::<f64>::new(w, h, 1);
        fft.forward(&mut a);
        fft.forward(&mut b);
        for i in 0..n {
            let k = [wavenumber(i % w, w), wavenumber(i / w, h)];
            let (_, v) = light_upper(th0, th0, k);
            let ov = v[0].conj() * a[i] + v[1].conj() * b[i];
            a[i] = v[0] * ov;
            b[i] = v[1] * ov;
        }
        fft.inverse(&mut a);
        fft.inverse(&mut b);
        let nrm: f64 = a.iter().chain(b.iter()).map(|c| c.norm_sqr()).sum();
        let s = 1.0 / nrm.max(1e-300).sqrt();
        for i in 0..n {
            self.light.psi[i] += a[i] * s;
            self.light.psi[n + i] += b[i] * s;
        }
        let c = self.light_circular_centroid();
        self.light.last = c;
        self.light.unwrapped = c;
        self.light.tracking = true;
    }

    /// Amplitud de la componente de Fourier k de ψ proyectada sobre el autovector superior de la
    /// ronda con los θ actuales: (re, im).
    pub(crate) fn light_mode_amplitude(&self, k: [f64; 2]) -> C {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let (tx, ty) = self.light.metric_thetas(self.light.rounds as f64);
        let (tx, ty) = if self.light.metric_kind == 0 { (self.light.theta0, self.light.theta0) } else { (tx, ty) };
        let (_, v) = light_upper(tx, ty, k);
        let (mut a, mut b) = (cx(0.0, 0.0), cx(0.0, 0.0));
        for i in 0..n {
            let ph = C::from_polar(1.0, -(k[0] * (i % w) as f64 + k[1] * (i / w) as f64));
            a += self.light.psi[i] * ph;
            b += self.light.psi[n + i] * ph;
        }
        v[0].conj() * a + v[1].conj() * b
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.19 (escenario 4): n(r) = 1 + 2GM/√(r² + a²) centrado en (x, y), θ(r) = arccos(c₀/n) con
    /// c₀ = cos θ₀ (θ₀ = 0,3); gm = 0: medio uniforme.
    pub fn set_light_index(&mut self, gm: f32, x: f32, y: f32, softening: f32) {
        if self.scenario != 4 {
            return;
        }
        let (w, h) = (self.w, self.h);
        self.light.gm = gm as f64;
        self.light.src = [x as f64, y as f64];
        self.light.soft = (softening as f64).max(0.1);
        self.light.metric_kind = 0;
        let c0 = self.light.theta0.cos();
        for i in 0..w * h {
            let th = if gm == 0.0 {
                self.light.theta0
            } else {
                let dx = min_image((i % w) as f64 - x as f64, w as f64);
                let dy = min_image((i / w) as f64 - y as f64, h as f64);
                let nn = 1.0 + 2.0 * gm as f64 / (dx * dx + dy * dy + self.light.soft * self.light.soft).sqrt();
                (c0 / nn).clamp(-1.0, 1.0).acos()
            };
            self.light.set_theta(0, i, th);
            self.light.set_theta(1, i, th);
        }
    }

    /// n(r) en una celda (escenario luz).
    pub fn light_index_at(&self, x: i32, y: i32) -> f64 {
        if self.scenario != 4 {
            return 1.0;
        }
        let i = self.wrap_cell(x, y, 0);
        self.light.theta0.cos() / self.light.cs[0][i].max(1e-12)
    }

    /// §2.19: paquete con portadora (π/2 + q, π/2), proyectado sobre la banda superior de W(k).
    pub fn inject_light_packet(&mut self, x: f32, y: f32, sigma_x: f32, sigma_y: f32, q: f32) {
        if self.scenario != 4 {
            return;
        }
        self.light_inject(x as f64, y as f64, (sigma_x as f64).max(0.5), (sigma_y as f64).max(0.5), [0.5 * PI + q as f64, 0.5 * PI]);
    }

    /// La misma, con portadora (π/2, π/2 + q): viaja en y.
    pub fn inject_light_packet_y(&mut self, x: f32, y: f32, sigma: f32, q: f32) {
        if self.scenario != 4 {
            return;
        }
        let s = (sigma as f64).max(0.5);
        self.light_inject(x as f64, y as f64, s, s, [0.5 * PI, 0.5 * PI + q as f64]);
    }

    /// §2.19, uniforme y dependiente del tiempo: 0 = nada; 1 = expansión a(t) = (1 + t/t₁)^(2/3)
    /// con t₁ = period (amplitude = a máximo); 2 = onda gravitacional h(t) = amplitude·sin(2πt/period).
    /// t en rondas desde ahora.
    pub fn set_light_metric_t(&mut self, kind: u32, amplitude: f32, period: f32) {
        if self.scenario != 4 {
            return;
        }
        self.light.metric_kind = kind.min(2);
        self.light.metric_amp = amplitude as f64;
        self.light.metric_period = (period as f64).max(1e-6);
        self.light.rounds = 0;
        if kind == 0 {
            self.set_light_index(0.0, 0.0, 0.0, 4.0);
        }
    }

    /// Centroide sin envolver del paquete (seguido de ronda en ronda desde la inyección).
    pub fn light_centroid(&self, axis: u32) -> f64 {
        self.light.unwrapped[(axis as usize).min(1)]
    }

    pub fn light_rounds(&self) -> u32 {
        self.light.rounds
    }

    /// Fase de la componente de Fourier k proyectada sobre el autovector superior de la ronda con
    /// los θ actuales (la frecuencia del modo central, §3.17, es su ritmo de giro).
    pub fn light_mode_phase(&self, kx: f32, ky: f32) -> f64 {
        self.light_mode_amplitude([kx as f64, ky as f64]).arg()
    }

    /// Velocidad del cono (por paso) en la dirección (dx, dy) con θ uniforme, por autovalores:
    /// ω(|q|)/(4|q|) con |q| = 0,02 alrededor de (π/2, π/2).
    pub fn light_cone_speed(&self, theta: f32, dx: f32, dy: f32) -> f64 {
        let nn = ((dx * dx + dy * dy) as f64).sqrt().max(1e-12);
        let q = 0.02;
        let k = [0.5 * PI + q * dx as f64 / nn, 0.5 * PI + q * dy as f64 / nn];
        light_upper(theta as f64, theta as f64, k).0 / (4.0 * q)
    }

    /// Cuasi-energía por ronda de la banda superior en k con los θ de la métrica en el instante t.
    pub fn light_band_energy(&self, kx: f32, ky: f32, t: f32) -> f64 {
        let (tx, ty) = self.light.metric_thetas(t as f64);
        light_upper(tx, ty, [kx as f64, ky as f64]).0
    }

    /// Cuasi-energía por ronda de la banda superior en k con θ_x y θ_y dados.
    pub fn light_band_energy_th(&self, thx: f32, thy: f32, kx: f32, ky: f32) -> f64 {
        light_upper(thx as f64, thy as f64, [kx as f64, ky as f64]).0
    }

    /// Fracción de la norma en la banda superior (θ uniformes del instante actual).
    pub fn light_upper_weight(&self) -> f64 {
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let (tx, ty) = if self.light.metric_kind == 0 { (self.light.theta0, self.light.theta0) } else { self.light.metric_thetas(self.light.rounds as f64) };
        let mut a: Vec<C> = self.light.psi[..n].to_vec();
        let mut b: Vec<C> = self.light.psi[n..].to_vec();
        let mut fft = FftN::<f64>::new(w, h, 1);
        fft.forward(&mut a);
        fft.forward(&mut b);
        let (mut up, mut tot) = (0.0, 0.0);
        for i in 0..n {
            let k = [wavenumber(i % w, w), wavenumber(i / w, h)];
            let (_, v) = light_upper(tx, ty, k);
            up += (v[0].conj() * a[i] + v[1].conj() * b[i]).norm_sqr();
            tot += a[i].norm_sqr() + b[i].norm_sqr();
        }
        if tot > 0.0 {
            up / tot
        } else {
            0.0
        }
    }
}
