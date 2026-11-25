//! Gravedad análoga en el superfluido (§2.28, §3.26), primera parte: la resonancia paramétrica de
//! Floquet del fondo repulsivo (§2.7), la tira acústica W × 1 (horizonte negro a la entrada de la
//! región supersónica, blanco a la salida: el láser de agujero negro), el BdG del continuo con la
//! banda de la caminata y el Hawking espontáneo por lotes (vacío de Wigner truncado propagado como
//! covarianza exacta, sin ruido).
//!
//! Convenciones (las de `walk.rs`): ψ ← S·e^{−iθ}·C(m)·ψ con θ = V − g·f(ρ), f(ρ) = ρ/(1 + ρ/E_max),
//! C(m) = cos m·I − i·sin m·G y desplazamiento *gather* (canal 2 ← x + 1, 3 ← x − 1). En la tira de
//! altura 1 los canales N y S no se mueven (k_y = 0). El autovalor de U(k) = S(k)·C es e^{−iω(k)}.

use crate::linalg::{self, C};
use crate::rng::SplitMix64;
use crate::util::{clamp_finite, wrap_angle};
use crate::QuantumSpacetimeUniverse;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

#[inline]
fn cz() -> C {
    C::new(0.0, 0.0)
}

#[inline]
fn cis(a: f64) -> C {
    C::new(a.cos(), a.sin())
}

/// Moneda de Grover C(m) = cos m·I − i·sin m·((2/n_c)·J − I) como matriz (fila mayor).
pub(crate) fn coin_matrix(m: f64, nc: usize) -> Vec<C> {
    let f = 2.0 / nc as f64;
    let mut c = vec![cz(); nc * nc];
    for i in 0..nc {
        for j in 0..nc {
            let g = f - if i == j { 1.0 } else { 0.0 };
            c[i * nc + j] = C::new(if i == j { m.cos() } else { 0.0 }, -m.sin() * g);
        }
    }
    c
}

/// Fases del desplazamiento en el momento k: canal 0 ← y + 1, 1 ← y − 1, 2 ← x + 1, 3 ← x − 1,
/// 4 ← z + 1, 5 ← z − 1. dim = 1 es la tira de altura 1 (N y S sin fase).
pub(crate) fn shift_phases(k: [f64; 3], dim: u32) -> Vec<C> {
    let one = C::new(1.0, 0.0);
    match dim {
        1 => vec![one, one, cis(k[0]), cis(-k[0])],
        2 => vec![cis(k[1]), cis(-k[1]), cis(k[0]), cis(-k[0])],
        _ => vec![cis(k[1]), cis(-k[1]), cis(k[0]), cis(-k[0]), cis(k[2]), cis(-k[2])],
    }
}

fn step_matrix(coin: &[C], k: [f64; 3], dim: u32) -> Vec<C> {
    let nc = if dim == 3 { 6 } else { 4 };
    let s = shift_phases(k, dim);
    let mut u = coin.to_vec();
    for i in 0..nc {
        for j in 0..nc {
            u[i * nc + j] *= s[i];
        }
    }
    u
}

/// Sigue la banda focalizante (la del espinor simétrico en k = 0, autovalor e^{−im}) a lo largo de
/// `path` (valores de k_x crecientes en módulo desde 0) por solapamiento de autovectores.
/// Devuelve, para cada k del camino, el espinor unitario (Σ_c u_c real y positiva) y λ = e^{−iω}.
pub(crate) fn track_band(m: f64, dim: u32, path: &[f64]) -> Vec<(Vec<C>, C)> {
    let nc = if dim == 3 { 6 } else { 4 };
    let coin = coin_matrix(m, nc);
    let mut v: Vec<C> = vec![C::new(1.0 / (nc as f64).sqrt(), 0.0); nc];
    let mut lam = cis(-m);
    let mut k_prev = 0.0f64;
    let mut out = Vec::with_capacity(path.len());
    for &k in path {
        if !k.is_finite() {
            out.push((v.clone(), lam));
            continue;
        }
        let nsub = (((k - k_prev).abs() / 0.01).ceil() as usize).clamp(1, 10_000);
        if k != k_prev {
            for s in 1..=nsub {
                let kk = k_prev + (k - k_prev) * s as f64 / nsub as f64;
                let e = linalg::eig(step_matrix(&coin, [kk, 0.0, 0.0], dim), nc);
                let mut best = 0;
                let mut bo = -1.0;
                for (j, vec) in e.vectors.iter().enumerate() {
                    let o: C = vec.iter().zip(&v).map(|(a, b)| a.conj() * b).sum();
                    if o.norm() > bo {
                        bo = o.norm();
                        best = j;
                    }
                }
                v = e.vectors[best].clone();
                lam = e.values[best];
                let s: C = v.iter().sum();
                if s.norm() > 1e-12 {
                    let ph = s.conj() / s.norm();
                    for c in v.iter_mut() {
                        *c *= ph;
                    }
                }
            }
        }
        k_prev = k;
        out.push((v.clone(), lam));
    }
    out
}

/// Espinor unitario y autovalor de la banda focalizante en k_x.
pub(crate) fn focusing_band(m: f64, dim: u32, kx: f64) -> (Vec<C>, C) {
    track_band(m, dim, &[kx]).pop().unwrap()
}

/// Velocidad de grupo ∂ω/∂k de la banda focalizante en k_x (diferencia centrada).
pub(crate) fn band_group_velocity(m: f64, dim: u32, kx: f64) -> f64 {
    let h = 1e-4;
    let l1 = focusing_band(m, dim, kx + h).1;
    let l0 = focusing_band(m, dim, kx - h).1;
    // λ = e^{−iω}: ω₁ − ω₀ = −arg(λ₁/λ₀)
    -(l1 / l0).arg() / (2.0 * h)
}

/// Bloque de Floquet–Bogoliubov 2n_c × 2n_c del fondo ψ₀ = u·e^{ik₀x} en el marco rotante, con
/// δψ = e^{ik₀x}·(a·e^{iqx} + b̄·e^{−iqx}):
/// a′ = e^{iω₀}·S(k₀ + q)·C·[a + iΘ′·u·(u†a + uᵀb)],
/// b′ = e^{−iω₀}·S̄(k₀ − q)·C̄·[b − iΘ′·ū·(u†a + uᵀb)],
/// con Θ′ = g·f′(ρ̄) (f′ sin |g|) y |u|² = ρ̄. La parte local es exp(iΘ′·P) = 1 + iΘ′·P (P² = 0).
pub(crate) fn bogoliubov_block(coin: &[C], u: &[C], w0: f64, th: f64, kp: [f64; 3], km: [f64; 3], dim: u32) -> Vec<C> {
    let nc = u.len();
    let n2 = 2 * nc;
    let sp = shift_phases(kp, dim);
    let sm = shift_phases(km, dim);
    let (ep, em) = (cis(w0), cis(-w0));
    let ith = C::new(0.0, th);
    let mut inner = vec![cz(); n2 * n2];
    for r in 0..nc {
        for c in 0..nc {
            let d = C::new(if r == c { 1.0 } else { 0.0 }, 0.0);
            inner[r * n2 + c] = d + ith * u[r] * u[c].conj();
            inner[r * n2 + nc + c] = ith * u[r] * u[c];
            inner[(nc + r) * n2 + c] = -ith * u[r].conj() * u[c].conj();
            inner[(nc + r) * n2 + nc + c] = d - ith * u[r].conj() * u[c];
        }
    }
    let mut m = vec![cz(); n2 * n2];
    for r in 0..nc {
        let (fa, fb) = (ep * sp[r], em * sm[r].conj());
        for c in 0..n2 {
            let mut s1 = cz();
            let mut s2 = cz();
            for k in 0..nc {
                s1 += coin[r * nc + k] * inner[k * n2 + c];
                s2 += coin[r * nc + k].conj() * inner[(nc + k) * n2 + c];
            }
            m[r * n2 + c] = fa * s1;
            m[(nc + r) * n2 + c] = fb * s2;
        }
    }
    m
}

/// Máximo de ln|λ| por paso del Floquet–Bogoliubov del fondo uniforme (§2.7), con el q donde se
/// alcanza. Rejilla q_j = 2π(j + ½)/nq por eje (evita el bloque de Goldstone en q = 0) y las
/// simetrías exactas del bloque: q → −q (pareja conjugada), reflexiones transversales al flujo y,
/// sin flujo, las permutaciones de ejes.
pub(crate) fn parametric_rate_impl(m: f64, g: f64, rho: f64, emax: f64, dim: u32, nq: usize, k0: f64) -> (f64, [f64; 3]) {
    parametric_rate_grid(m, g, rho, emax, dim, nq, k0, 0.5)
}

/// Lo mismo con la rejilla q_j = 2π(j + offset)/nq (offset = 0: los q de una tira de nq celdas,
/// sin el bloque de Goldstone j = 0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn parametric_rate_grid(m: f64, g: f64, rho: f64, emax: f64, dim: u32, nq: usize, k0: f64, offset: f64) -> (f64, [f64; 3]) {
    if !m.is_finite() || !g.is_finite() || !rho.is_finite() || !emax.is_finite() || !k0.is_finite() || !offset.is_finite() || rho <= 0.0 || emax <= 0.0 {
        return (0.0, [0.0; 3]);
    }
    let dim = dim.clamp(1, 3);
    let k0 = wrap_angle(k0);
    let nc = if dim == 3 { 6 } else { 4 };
    let coin = coin_matrix(m, nc);
    let (u1, lam) = focusing_band(m, dim, k0);
    let u: Vec<C> = u1.iter().map(|c| c * rho.sqrt()).collect();
    let w0 = -lam.arg();
    let th = g / (1.0 + rho / emax).powi(2);
    let nq = match dim {
        1 => nq.clamp(2, 65_536),
        2 => nq.clamp(2, 2048),
        _ => nq.clamp(2, 128),
    };
    let half = nq.div_ceil(2);
    let qv = |j: usize| 2.0 * PI * (j as f64 + offset) / nq as f64;
    let mut best = (0.0f64, [0.0f64; 3]);
    let mut eval = |q: [f64; 3]| {
        if q.iter().all(|v| v.abs() < 1e-15) {
            return;
        }
        let blk = bogoliubov_block(&coin, &u, w0, th, [k0 + q[0], q[1], q[2]], [k0 - q[0], -q[1], -q[2]], dim);
        let ev = linalg::eigvals(blk, 2 * nc);
        let r = ev.iter().map(|l| l.norm().ln()).fold(f64::NEG_INFINITY, f64::max);
        if r > best.0 {
            best = (r, q);
        }
    };
    match dim {
        1 => {
            for j in 0..half {
                eval([qv(j), 0.0, 0.0]);
            }
        }
        2 => {
            if k0 == 0.0 {
                for jy in 0..half {
                    for jx in 0..=jy {
                        eval([qv(jx), qv(jy), 0.0]);
                    }
                }
            } else {
                for jy in 0..half {
                    for jx in 0..half {
                        eval([qv(jx), qv(jy), 0.0]);
                    }
                }
            }
        }
        _ => {
            if k0 == 0.0 {
                for jz in 0..half {
                    for jy in 0..=jz {
                        for jx in 0..=jy {
                            eval([qv(jx), qv(jy), qv(jz)]);
                        }
                    }
                }
            } else {
                for jx in 0..half {
                    for jz in 0..half {
                        for jy in 0..=jz {
                            eval([qv(jx), qv(jy), qv(jz)]);
                        }
                    }
                }
            }
        }
    }
    (best.0.max(0.0), best.1)
}

/// Smoothstep 3u² − 2u³ en [0, 1].
fn smoothstep(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// La tira acústica W × 1 (el sector k_y = 0 de la caminata 2D), en f64.
pub struct Strip {
    pub w: usize,
    pub mass: f64,
    cm: f64,
    sm: f64,
    pub emax: f64,
    pub rho: f64,
    pub n_flow: u32,
    pub k0: f64,
    pub g_sub: f64,
    pub g_super: f64,
    pub length: f64,
    /// g(x) del perfil completo (supersónico en la región central).
    pub g: Vec<f64>,
    /// V(x) = (g(x) − g_sub)·f(ρ̄): la onda plana es una solución estacionaria exacta.
    pub v: Vec<f64>,
    pub psi: Vec<[C; 4]>,
    phi: Vec<[C; 4]>,
    /// Espinor del fondo (|u|² = ρ̄).
    pub u: [C; 4],
    /// ω(k₀) de la banda (λ = e^{−iω₀}).
    pub omega0: f64,
    /// ω(2πj/W) de la banda focalizante, j = 0..W.
    pub band: Vec<f64>,
    pub vg: f64,
    pub m_star: f64,
    pub t: u64,
    /// Historia de la densidad para la cascada espacio-tiempo: filas de W f32.
    pub hist: Vec<f32>,
    pub hist_rows: usize,
    pub hist_every: u32,
}

impl Strip {
    pub fn new(w: usize, n_flow: u32, g_sub: f64, g_super: f64, length: f64, mass: f64, emax: f64) -> Strip {
        let w = w.max(8);
        let rho: f64 = 0.5;
        let k0 = 2.0 * PI * n_flow as f64 / w as f64;
        // banda en la rejilla: j = 0..W/2 hacia k > 0 y j = W − 1..W/2 hacia k < 0
        let half = w / 2;
        let pos: Vec<f64> = (0..=half).map(|j| 2.0 * PI * j as f64 / w as f64).collect();
        let neg: Vec<f64> = (1..(w - half)).map(|j| -2.0 * PI * j as f64 / w as f64).collect();
        let tp = track_band(mass, 1, &pos);
        let tn = track_band(mass, 1, &neg);
        let mut band = vec![0.0; w];
        for (j, (_, l)) in tp.iter().enumerate() {
            band[j] = -l.arg();
        }
        for (j, (_, l)) in tn.iter().enumerate() {
            band[w - 1 - j] = -l.arg();
        }
        let (u1, lam) = focusing_band(mass, 1, k0);
        let u = [u1[0] * rho.sqrt(), u1[1] * rho.sqrt(), u1[2] * rho.sqrt(), u1[3] * rho.sqrt()];
        let f = rho / (1.0 + rho / emax);
        let mut g = vec![g_sub; w];
        let mut v = vec![0.0; w];
        let (x1, x2) = (0.5 * w as f64 - 0.5 * length, 0.5 * w as f64 + 0.5 * length);
        for x in 0..w {
            let xf = x as f64;
            let prof = if length > 0.0 { 0.5 * (((xf - x1) / 2.0).tanh() - ((xf - x2) / 2.0).tanh()) } else { 0.0 };
            g[x] = g_sub + (g_super - g_sub) * prof;
            v[x] = (g[x] - g_sub) * f;
        }
        let psi: Vec<[C; 4]> = (0..w)
            .map(|x| {
                let ph = cis(k0 * x as f64);
                [u[0] * ph, u[1] * ph, u[2] * ph, u[3] * ph]
            })
            .collect();
        Strip {
            w,
            mass,
            cm: mass.cos(),
            sm: mass.sin(),
            emax,
            rho,
            n_flow,
            k0,
            g_sub,
            g_super,
            length,
            g,
            v,
            phi: psi.clone(),
            psi,
            u,
            omega0: -lam.arg(),
            band,
            vg: band_group_velocity(mass, 1, k0),
            m_star: 2.0 * mass.tan(),
            t: 0,
            hist: Vec::new(),
            hist_rows: 0,
            hist_every: 50,
        }
    }

    /// f′(ρ̄) sin |g|.
    fn fprime(&self) -> f64 {
        1.0 / (1.0 + self.rho / self.emax).powi(2)
    }

    /// c_s = √(ρ̄·|g|·f′/m\*).
    pub fn sound(&self, g: f64) -> f64 {
        (self.rho * g.abs() * self.fprime() / self.m_star).sqrt()
    }

    pub fn xi(&self, g: f64) -> f64 {
        1.0 / (2.0 * self.m_star * self.rho * g.abs() * self.fprime()).sqrt()
    }

    /// Fondo ψ₀(x) = u·e^{ik₀x}.
    fn background(&self, x: usize) -> [C; 4] {
        let ph = cis(self.k0 * x as f64);
        [self.u[0] * ph, self.u[1] * ph, self.u[2] * ph, self.u[3] * ph]
    }

    /// Un paso de la caminata no lineal: e^{−iθ}·C(m) local y desplazamiento *gather*.
    pub fn step(&mut self) {
        let w = self.w;
        let (cm, sm) = (self.cm, self.sm);
        let isn = C::new(0.0, sm);
        for x in 0..w {
            let p = self.psi[x];
            let e: f64 = p.iter().map(|c| c.norm_sqr()).sum();
            let f = e / (1.0 + e / self.emax);
            let ph = cis(-(self.v[x] - self.g[x] * f));
            let s = (p[0] + p[1] + p[2] + p[3]) * 0.5;
            for c in 0..4 {
                self.phi[x][c] = (p[c] * cm - isn * (s - p[c])) * ph;
            }
        }
        for x in 0..w {
            let xp = if x + 1 == w { 0 } else { x + 1 };
            let xm = if x == 0 { w - 1 } else { x - 1 };
            self.psi[x] = [self.phi[x][0], self.phi[x][1], self.phi[xp][2], self.phi[xm][3]];
        }
        self.t += 1;
        if self.hist_rows > 0 && self.t % self.hist_every as u64 == 0 {
            // desplaza la cascada una fila y añade la densidad relativa actual al final
            let w = self.w;
            self.hist.copy_within(w.., 0);
            let n = self.hist.len();
            for x in 0..w {
                let e: f64 = self.psi[x].iter().map(|c| c.norm_sqr()).sum();
                self.hist[n - w + x] = (e / self.rho - 1.0) as f32;
            }
        }
    }

    pub fn add_noise(&mut self, amp: f64, seed: u64) {
        let mut rng = SplitMix64::new(seed ^ 0xA11A_0617);
        let a = amp * (self.rho / 4.0).sqrt() / 2f64.sqrt();
        for p in self.psi.iter_mut() {
            for c in p.iter_mut() {
                *c += C::new(a * rng.gauss(), a * rng.gauss());
            }
        }
    }

    /// ‖ψ − e^{iφ}·ψ₀‖ con la fase global φ ajustada (absorbe e^{−iΩt}).
    pub fn deviation(&self) -> f64 {
        let mut ov = cz();
        for x in 0..self.w {
            let b = self.background(x);
            for c in 0..4 {
                ov += b[c].conj() * self.psi[x][c];
            }
        }
        let ph = if ov.norm() > 0.0 { ov / ov.norm() } else { C::new(1.0, 0.0) };
        let mut s = 0.0;
        for x in 0..self.w {
            let b = self.background(x);
            for c in 0..4 {
                s += (self.psi[x][c] - ph * b[c]).norm_sqr();
            }
        }
        s.sqrt()
    }

    pub fn norm(&self) -> f64 {
        self.psi.iter().map(|p| p.iter().map(|c| c.norm_sqr()).sum::<f64>()).sum()
    }

    pub fn density(&self, x: usize) -> f64 {
        self.psi[x].iter().map(|c| c.norm_sqr()).sum()
    }

    /// Modo de Fourier dominante de δn = n − ρ̄ (q > 0): [q, |δn_q|/W].
    pub fn dominant_mode(&self) -> (f64, f64) {
        let w = self.w;
        let mut buf: Vec<C> = (0..w).map(|x| C::new(self.density(x) - self.rho, 0.0)).collect();
        let fft = FftPlanner::new().plan_fft_forward(w);
        fft.process(&mut buf);
        let mut best = (0.0, 0.0);
        for (j, b) in buf.iter().enumerate().take(w / 2 + 1).skip(1) {
            let a = b.norm() / w as f64;
            if a > best.1 {
                best = (2.0 * PI * j as f64 / w as f64, a);
            }
        }
        best
    }

    /// Posiciones de los horizontes (c_s(x) = v): [entrada (negro), salida (blanco)]; NaN si no hay.
    pub fn horizons(&self, s: f64) -> (f64, f64) {
        let w = self.w;
        let v = self.vg.abs();
        let c = |x: usize| self.sound(self.g_sub + (self.g[x] - self.g_sub) * s);
        let (mut xb, mut xw) = (f64::NAN, f64::NAN);
        for x in 0..w - 1 {
            let (c0, c1) = (c(x), c(x + 1));
            if c0 > v && c1 <= v && xb.is_nan() {
                xb = x as f64 + (c0 - v) / (c0 - c1);
            }
            if c0 <= v && c1 > v && xw.is_nan() {
                xw = x as f64 + (v - c0) / (c1 - c0);
            }
        }
        (xb, xw)
    }

    /// Paso en que se forma el horizonte con una rampa smoothstep de `ramp` pasos: el primero en que
    /// el mínimo de c_s(x) baja de v (0 con un salto brusco).
    pub fn horizon_formation(&self, ramp: u32) -> f64 {
        if ramp == 0 {
            return 0.0;
        }
        let v = self.vg.abs();
        for n in 0..=ramp {
            let sr = smoothstep(n as f64 / ramp as f64);
            let cmin = self.g.iter().map(|g| self.sound(self.g_sub + (g - self.g_sub) * sr)).fold(f64::INFINITY, f64::min);
            if cmin <= v {
                return n as f64;
            }
        }
        f64::NAN
    }

    /// Tasa del BdG del continuo con la banda de la caminata por iteración de potencia (§2.28):
    /// i∂_t(a, b) = [[A + β, β], [−β, −(B + β)]](a, b), paso partido espectral de dt = 1 (la parte
    /// local, exp(−iβ·M) = 1 − iβ·M porque M² = 0), renormalizando cada 100 pasos. Devuelve la tasa
    /// media de la segunda mitad.
    pub fn bdg_power_rate(&self, steps: u32, seed: u64) -> f64 {
        let w = self.w;
        let n = self.n_flow as usize % w;
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(w);
        let inv = planner.plan_fft_inverse(w);
        let fp = self.fprime();
        let beta: Vec<f64> = self.g.iter().map(|g| self.rho * g.abs() * fp).collect();
        let ka: Vec<C> = (0..w).map(|j| cis(-(self.band[(n + j) % w] - self.band[n]))).collect();
        let kb: Vec<C> = (0..w).map(|j| cis(self.band[(n + w - j) % w] - self.band[n])).collect();
        let mut rng = SplitMix64::new(seed ^ 0xB0D6);
        let mut a: Vec<C> = (0..w).map(|_| C::new(rng.gauss(), rng.gauss())).collect();
        let mut b: Vec<C> = (0..w).map(|_| C::new(rng.gauss(), rng.gauss())).collect();
        let inv_w = 1.0 / w as f64;
        let steps = steps.clamp(200, 200_000) as usize;
        let mut log_sum = 0.0;
        let mut counted = 0usize;
        let mut since = 0usize;
        for s in 1..=steps {
            for x in 0..w {
                let ib = C::new(0.0, beta[x]) * (a[x] + b[x]);
                a[x] -= ib;
                b[x] += ib;
            }
            fwd.process(&mut a);
            fwd.process(&mut b);
            for j in 0..w {
                a[j] *= ka[j] * inv_w;
                b[j] *= kb[j] * inv_w;
            }
            inv.process(&mut a);
            inv.process(&mut b);
            since += 1;
            if s % 100 == 0 || s == steps {
                let nn = (a.iter().chain(b.iter()).map(|c| c.norm_sqr()).sum::<f64>()).sqrt();
                if !(nn > 1e-300 && nn.is_finite()) {
                    return 0.0;
                }
                if s > steps / 2 {
                    log_sum += nn.ln();
                    counted += since;
                }
                since = 0;
                for c in a.iter_mut().chain(b.iter_mut()) {
                    *c /= nn;
                }
            }
        }
        if counted == 0 {
            0.0
        } else {
            log_sum / counted as f64
        }
    }
}

/// Cálculo por lotes de G(x, x′) = ⟨δn(x)·δn(x′)⟩ en el paso t (§2.28): las filas del funcional δn
/// se propagan hacia atrás con la jacobiana exacta del paso (marco rotante del fondo) y se combinan
/// con el vacío de Bogoliubov del fondo uniforme, construido por bloques en q.
pub struct HawkingJob {
    pub engine: u32,
    pub ramp: u32,
    pub t: u32,
    /// Filas x calculadas.
    pub rows: Vec<usize>,
    pub next: usize,
    w: usize,
    /// Modos de norma simpléctica positiva: índice de q y (a, b) con n_ch componentes cada uno.
    mode_q: Vec<usize>,
    mode_a: Vec<C>,
    mode_b: Vec<C>,
    nch: usize,
    /// Coeficientes c_{x,j} (parte real e imaginaria) de cada fila calculada.
    cre: Vec<f64>,
    cim: Vec<f64>,
    // propagación
    ramp_s: Vec<f64>,
    th_base: Vec<f64>,
    th_delta: Vec<f64>,
    u: [C; 4],
    d_ns: C,
    d_e: C,
    d_w: C,
    kappa_c: C,
    ka: Vec<C>,
    rho: f64,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
    pub xi: f64,
    pub tr_sigma: f64,
    pub n_modes: usize,
}

impl HawkingJob {
    pub fn new(st: &Strip, engine: u32, ramp: u32, t: u32, row_stride: usize) -> HawkingJob {
        let w = st.w;
        let fp = st.fprime();
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(w);
        let inv = planner.plan_fft_inverse(w);
        let ramp_s: Vec<f64> = (0..t as usize).map(|n| if ramp == 0 { 1.0 } else { smoothstep(n as f64 / ramp as f64) }).collect();
        let th_base = vec![st.g_sub * fp; w];
        let th_delta: Vec<f64> = st.g.iter().map(|g| (g - st.g_sub) * fp).collect();
        let n = st.n_flow as usize % w;
        let rows: Vec<usize> = (0..w).step_by(row_stride.max(1)).collect();
        let (mut mode_q, mut mode_a, mut mode_b) = (Vec::new(), Vec::new(), Vec::new());
        let mut tr_sigma = 0.0;
        let nch;
        if engine == 0 {
            nch = 4;
            let coin = coin_matrix(st.mass, 4);
            let th = st.g_sub * fp;
            let h = 1.0 / 2f64.sqrt();
            // base reducida: (N + S)/√2, E, W (la banda plana N − S no se acopla ni pesa en δn)
            let basis: [[f64; 4]; 3] = [[h, h, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
            for j in 0..w {
                let q = 2.0 * PI * j as f64 / w as f64;
                let blk = bogoliubov_block(&coin, &st.u, st.omega0, th, [st.k0 + q, 0.0, 0.0], [st.k0 - q, 0.0, 0.0], 1);
                let mut red = vec![cz(); 36];
                for r in 0..6 {
                    for c in 0..6 {
                        let mut s = cz();
                        for i in 0..4 {
                            for k in 0..4 {
                                let pr = basis[r % 3][i];
                                let pc = basis[c % 3][k];
                                if pr != 0.0 && pc != 0.0 {
                                    s += blk[(4 * (r / 3) + i) * 8 + 4 * (c / 3) + k] * (pr * pc);
                                }
                            }
                        }
                        red[r * 6 + c] = s;
                    }
                }
                let e = linalg::eig(red, 6);
                let mut sel: Vec<Vec<C>> = Vec::new();
                for (lam, y) in e.values.iter().zip(e.vectors.iter()) {
                    if j == 0 && (lam - C::new(1.0, 0.0)).norm() < 1e-5 {
                        continue; // par de Goldstone (bloque de Jordan en q = 0)
                    }
                    let mut z = vec![cz(); 8];
                    for r in 0..6 {
                        for i in 0..4 {
                            z[4 * (r / 3) + i] += y[r] * basis[r % 3][i];
                        }
                    }
                    let nrm: f64 = z[..4].iter().map(|c| c.norm_sqr()).sum::<f64>() - z[4..].iter().map(|c| c.norm_sqr()).sum::<f64>();
                    if nrm > 1e-9 {
                        sel.push(z);
                    }
                }
                eta_gram_schmidt(&mut sel, 4);
                for z in sel {
                    tr_sigma += z.iter().map(|c| c.norm_sqr()).sum::<f64>();
                    mode_q.push(j);
                    mode_a.extend_from_slice(&z[..4]);
                    mode_b.extend_from_slice(&z[4..]);
                }
            }
            // la banda plana: un modo de norma 1 por q, sin b
            tr_sigma += w as f64;
        } else {
            nch = 1;
            let beta = st.rho * st.g_sub.abs() * fp;
            for j in 1..w {
                let ea = cis(-(st.band[(n + j) % w] - st.band[n]));
                let eb = cis(st.band[(n + w - j) % w] - st.band[n]);
                let ib = C::new(0.0, beta);
                let m = vec![ea * (C::new(1.0, 0.0) - ib), ea * (-ib), eb * ib, eb * (C::new(1.0, 0.0) + ib)];
                let e = linalg::eig(m, 2);
                let mut sel: Vec<Vec<C>> = Vec::new();
                for y in e.vectors.iter() {
                    let nrm = y[0].norm_sqr() - y[1].norm_sqr();
                    if nrm > 1e-9 {
                        sel.push(y.clone());
                    }
                }
                eta_gram_schmidt(&mut sel, 1);
                for z in sel {
                    tr_sigma += z.iter().map(|c| c.norm_sqr()).sum::<f64>();
                    mode_q.push(j);
                    mode_a.push(z[0]);
                    mode_b.push(z[1]);
                }
            }
        }
        let nm = mode_q.len();
        let m = st.mass;
        let ka: Vec<C> = (0..w).map(|j| cis(-(st.band[(n + j) % w] - st.band[n]))).collect();
        HawkingJob {
            engine,
            ramp,
            t,
            next: 0,
            w,
            cre: vec![0.0; rows.len() * nm],
            cim: vec![0.0; rows.len() * nm],
            rows,
            mode_q,
            mode_a,
            mode_b,
            nch,
            ramp_s,
            th_base,
            th_delta,
            u: st.u,
            d_ns: cis(m + st.omega0),
            d_e: cis(m + st.omega0 + st.k0),
            d_w: cis(m + st.omega0 - st.k0),
            kappa_c: C::new(0.0, -0.5 * m.sin()) * cis(-m),
            ka,
            rho: st.rho,
            fwd,
            inv,
            xi: st.xi(st.g_sub),
            tr_sigma: tr_sigma / (8.0 * w as f64),
            n_modes: nm,
        }
    }

    /// Propaga hacia atrás la fila x (caminata) y devuelve Â′(q, canal) = Σ_x α̂(x)·e^{iqx}
    /// para los canales N (= S), E y W.
    fn walk_row(&self, x0: usize) -> [Vec<C>; 3] {
        let w = self.w;
        let u = self.u;
        let ub = [u[0].conj(), u[2].conj(), u[3].conj()];
        // α̂ inicial = δ_{x,x0}·ū (N = S)
        let mut an_r = vec![0.0f64; w];
        let mut an_i = vec![0.0f64; w];
        let mut ae_r = vec![0.0f64; w];
        let mut ae_i = vec![0.0f64; w];
        let mut aw_r = vec![0.0f64; w];
        let mut aw_i = vec![0.0f64; w];
        an_r[x0] = ub[0].re;
        an_i[x0] = ub[0].im;
        ae_r[x0] = ub[1].re;
        ae_i[x0] = ub[1].im;
        aw_r[x0] = ub[2].re;
        aw_i[x0] = ub[2].im;
        let (dn, de, dw, kc) = (self.d_ns, self.d_e, self.d_w, self.kappa_c);
        let (un, ue, uw) = (u[0], u[2], u[3]);
        for n in (0..self.t as usize).rev() {
            let sr = self.ramp_s[n];
            // dispersión (traspuesta del gather): E ← x − 1, W ← x + 1
            ae_r.rotate_right(1);
            ae_i.rotate_right(1);
            aw_r.rotate_left(1);
            aw_i.rotate_left(1);
            for x in 0..w {
                let bn = C::new(an_r[x], an_i[x]) * dn;
                let be = C::new(ae_r[x], ae_i[x]) * de;
                let bw = C::new(aw_r[x], aw_i[x]) * dw;
                let kap = kc * (bn * 2.0 + be + bw);
                let gn = bn + kap;
                let ge = be + kap;
                let gw = bw + kap;
                let p = (gn * un * 2.0 + ge * ue + gw * uw).im;
                let s = 2.0 * (self.th_base[x] + self.th_delta[x] * sr) * p;
                an_r[x] = gn.re - s * un.re;
                an_i[x] = gn.im + s * un.im;
                ae_r[x] = ge.re - s * ue.re;
                ae_i[x] = ge.im + s * ue.im;
                aw_r[x] = gw.re - s * uw.re;
                aw_i[x] = gw.im + s * uw.im;
            }
        }
        let mut out: [Vec<C>; 3] = [
            (0..w).map(|x| C::new(an_r[x], an_i[x])).collect(),
            (0..w).map(|x| C::new(ae_r[x], ae_i[x])).collect(),
            (0..w).map(|x| C::new(aw_r[x], aw_i[x])).collect(),
        ];
        for v in out.iter_mut() {
            self.inv.process(v);
        }
        out
    }

    /// Lo mismo con el continuo de una componente: α ← Kᵀα y α ← α + 2β·Im α.
    fn continuum_row(&self, x0: usize) -> Vec<C> {
        let w = self.w;
        let mut a = vec![cz(); w];
        a[x0] = C::new(self.rho.sqrt(), 0.0);
        let inv_w = 1.0 / w as f64;
        for n in (0..self.t as usize).rev() {
            let sr = self.ramp_s[n];
            self.inv.process(&mut a);
            for j in 0..w {
                a[j] *= self.ka[j] * inv_w;
            }
            self.fwd.process(&mut a);
            for x in 0..w {
                let beta = -self.rho * (self.th_base[x] + self.th_delta[x] * sr);
                a[x].re += 2.0 * beta * a[x].im;
            }
        }
        self.inv.process(&mut a);
        a
    }

    /// Calcula hasta `count` filas más; devuelve la fracción hecha.
    pub fn advance(&mut self, count: usize) -> f64 {
        let w = self.w;
        let nm = self.n_modes;
        let inv_sqrt_w = 1.0 / (w as f64).sqrt();
        let end = (self.next + count).min(self.rows.len());
        for r in self.next..end {
            let x0 = self.rows[r];
            if self.engine == 0 {
                let ah = self.walk_row(x0);
                for j in 0..nm {
                    let q = self.mode_q[j];
                    let qm = (w - q) % w;
                    let a = &self.mode_a[4 * j..4 * j + 4];
                    let b = &self.mode_b[4 * j..4 * j + 4];
                    let mut c = ah[0][q] * (a[0] + a[1]) + ah[1][q] * a[2] + ah[2][q] * a[3];
                    c += ah[0][qm].conj() * (b[0] + b[1]) + ah[1][qm].conj() * b[2] + ah[2][qm].conj() * b[3];
                    c *= inv_sqrt_w;
                    self.cre[r * nm + j] = c.re;
                    self.cim[r * nm + j] = c.im;
                }
            } else {
                let ah = self.continuum_row(x0);
                for j in 0..nm {
                    let q = self.mode_q[j];
                    let qm = (w - q) % w;
                    let c = (ah[q] * self.mode_a[j] + ah[qm].conj() * self.mode_b[j]) * inv_sqrt_w;
                    self.cre[r * nm + j] = c.re;
                    self.cim[r * nm + j] = c.im;
                }
            }
        }
        self.next = end;
        self.next as f64 / self.rows.len().max(1) as f64
    }

    pub fn done(&self) -> bool {
        self.next >= self.rows.len()
    }

    /// Mapa ξ_sub·G/ρ̄ (filas × filas calculadas).
    pub fn map(&self) -> Vec<f64> {
        let nr = self.rows.len();
        let nm = self.n_modes;
        let scale = self.xi / self.rho;
        let mut g = vec![0.0; nr * nr];
        for r1 in 0..nr {
            let (a1, b1) = (&self.cre[r1 * nm..(r1 + 1) * nm], &self.cim[r1 * nm..(r1 + 1) * nm]);
            for r2 in r1..nr {
                let (a2, b2) = (&self.cre[r2 * nm..(r2 + 1) * nm], &self.cim[r2 * nm..(r2 + 1) * nm]);
                let mut s = 0.0;
                for j in 0..nm {
                    s += a1[j] * a2[j] + b1[j] * b2[j];
                }
                g[r1 * nr + r2] = s * scale;
                g[r2 * nr + r1] = s * scale;
            }
        }
        g
    }
}

/// Gram–Schmidt con el producto simpléctico ⟨z, z′⟩ = a†a′ − b†b′ (definido positivo en los modos
/// de norma positiva): ortonormaliza los modos de un bloque (casi degenerados incluidos).
fn eta_gram_schmidt(modes: &mut [Vec<C>], nch: usize) {
    let ip = |x: &[C], y: &[C]| -> C {
        let mut s = cz();
        for i in 0..nch {
            s += x[i].conj() * y[i];
        }
        for i in nch..2 * nch {
            s -= x[i].conj() * y[i];
        }
        s
    };
    for i in 0..modes.len() {
        for j in 0..i {
            let (head, tail) = modes.split_at_mut(i);
            let p = ip(&head[j], &tail[0]);
            for k in 0..2 * nch {
                tail[0][k] -= head[j][k] * p;
            }
        }
        let nn = ip(&modes[i], &modes[i]).re.max(1e-300).sqrt();
        for c in modes[i].iter_mut() {
            *c /= nn;
        }
    }
}

/// Estado del escenario 10 (gravedad análoga).
pub struct Analog {
    /// Montaje: 0 = láser de agujero negro (tira), 1 = Hawking espontáneo (por lotes),
    /// 2 = vórtice de carga 2 en un disco, 3 = reconexión de anillos (3D), 4 = Unruh de fermiones.
    pub setup: u32,
    pub strip: Option<Strip>,
    pub job: Option<HawkingJob>,
    pub map: Vec<f64>,
    pub map_n: usize,
    pub out: Vec<f64>,
    /// Centro y radio del disco de muros.
    pub disk: Option<(f64, f64, f64)>,
    /// Puntos de las líneas de vórtice (x, y, z, componente) del último análisis.
    pub line_pts: Vec<f32>,
    pub line_info: Vec<f64>,
}

impl Analog {
    pub fn new() -> Analog {
        Analog {
            setup: 0,
            strip: None,
            job: None,
            map: Vec::new(),
            map_n: 0,
            out: Vec::new(),
            disk: None,
            line_pts: Vec::new(),
            line_info: vec![0.0; 4],
        }
    }
}

impl QuantumSpacetimeUniverse {
    /// Paso del escenario 10 cuando la tira está activa.
    pub(crate) fn analog_strip_step(&mut self) {
        if let Some(st) = self.an.strip.as_mut() {
            if self.an.setup == 0 {
                st.step();
            }
        }
        self.t = self.t.wrapping_add(1);
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.7: máximo de ln|λ| por paso del Floquet–Bogoliubov del fondo uniforme (bloques de
    /// 2n_c × 2n_c en q, nq por eje; dim = 1 es la tira k_y = 0, con nq ≥ 4096; flujo k₀ a lo largo
    /// de x). La rejilla es q_j = 2π(j + ½)/nq.
    pub fn parametric_rate(&self, m: f32, g: f32, rho: f32, dim: u32, nq: u32, k0: f32) -> f64 {
        parametric_rate_impl(m as f64, g as f64, rho as f64, self.e_max as f64, dim, nq as usize, k0 as f64).0
    }

    /// [tasa, q_x, q_y, q_z, tasa/(c_s/ξ)] con c_s/ξ = √2·ρ̄f′ (independiente de m\*).
    pub fn parametric_resonance(&self, m: f32, g: f32, rho: f32, dim: u32, nq: u32, k0: f32) -> Vec<f64> {
        let (r, q) = parametric_rate_impl(m as f64, g as f64, rho as f64, self.e_max as f64, dim, nq as usize, k0 as f64);
        let rf = if rho.is_finite() && g.is_finite() && rho > 0.0 && self.e_max > 0.0 {
            rho as f64 * (g as f64).abs() / (1.0 + rho as f64 / self.e_max as f64).powi(2)
        } else {
            0.0
        };
        let ratio = if rf > 1e-30 { r / (2f64.sqrt() * rf) } else { 0.0 };
        vec![r, q[0], q[1], q[2], ratio]
    }

    /// §2.28 (escenario 10): tira W × 1, onda plana de la banda en k₀ = 2πn/W con ρ̄ = 0,5, g(x)
    /// con bordes tanh de anchura 2 y V(x) = (g(x) − g_sub)·f(ρ̄). La región supersónica, de longitud
    /// `length`, está centrada en la tira.
    pub fn acoustic_strip_init(&mut self, width: u32, n_flow: u32, g_sub: f32, g_super: f32, length: f32, mass: f32) {
        let w = (width as usize).clamp(16, 16_384);
        let mass = clamp_finite(mass as f64, 0.05, 1.55, 1.4) as f32;
        let g_sub_f = clamp_finite(g_sub as f64, -10.0, 10.0, -0.02);
        let g_super_f = clamp_finite(g_super as f64, -10.0, 10.0, -0.0025);
        let len_f = clamp_finite(length as f64, 0.0, w as f64, 0.0);
        let emax = clamp_finite(self.e_max as f64, 1e-6, 1e12, 1.0);
        let st = Strip::new(w, n_flow % w as u32, g_sub_f, g_super_f, len_f, mass as f64, emax);
        let (v, cs, csp) = (st.vg, st.sound(st.g_sub), st.sound(st.g_super));
        if csp > v.abs() && len_f > 0.0 {
            self.set_message(&format!("La región central no es supersónica: v = {:.4} y c_s = {:.4}.", v, csp));
        } else if cs < v.abs() {
            self.set_message(&format!("La región exterior ya es supersónica: v = {:.4} y c_s = {:.4}.", v, cs));
        }
        self.set_mass(mass);
        self.g = g_sub_f as f32;
        self.an.strip = Some(st);
        self.an.job = None;
        self.t = 0;
    }

    /// Avanza la tira n pasos sin pasar por `step` (para los cálculos por lotes en otro universo).
    pub fn strip_step(&mut self, n: u32) {
        let n = n.min(100_000);
        if let Some(st) = self.an.strip.as_mut() {
            for _ in 0..n {
                st.step();
            }
        }
    }

    /// Paso de la tira.
    pub fn strip_time(&self) -> f64 {
        self.an.strip.as_ref().map(|s| s.t as f64).unwrap_or(0.0)
    }

    /// Ruido complejo gaussiano de amplitud relativa `amplitude` en los cuatro canales de la tira.
    pub fn strip_add_noise(&mut self, amplitude: f32, seed: u32) {
        let amp = clamp_finite(amplitude as f64, 0.0, 10.0, 0.0);
        if let Some(st) = self.an.strip.as_mut() {
            st.add_noise(amp, seed as u64);
        }
    }

    /// Activa la cascada espacio-tiempo de la tira: `rows` filas, una cada `every` pasos.
    pub fn strip_history(&mut self, rows: u32, every: u32) {
        if let Some(st) = self.an.strip.as_mut() {
            let max_rows = (8_388_608 / st.w.max(1)).min(4096);
            st.hist_rows = (rows as usize).min(max_rows);
            st.hist_every = every.max(1);
            st.hist = vec![0.0; st.hist_rows * st.w];
        }
    }

    /// ‖ψ − e^{iφ}·e^{−iΩt}·ψ₀‖ con la fase global φ ajustada (tira).
    pub fn background_deviation(&self) -> f64 {
        self.an.strip.as_ref().map(|s| s.deviation()).unwrap_or(f64::NAN)
    }

    /// §2.28: tasa del BdG continuo de la tira (banda de la caminata) por iteración de potencia con
    /// paso partido espectral; renormaliza cada 100 pasos.
    pub fn bdg_power_rate(&mut self, steps: u32) -> f64 {
        self.an.strip.as_ref().map(|s| s.bdg_power_rate(steps, 12345)).unwrap_or(f64::NAN)
    }

    /// Densidad de la tira (W f64).
    pub fn strip_density(&self) -> Vec<f64> {
        self.an.strip.as_ref().map(|s| (0..s.w).map(|x| s.density(x)).collect()).unwrap_or_default()
    }

    /// c_s(x) local de la tira con el perfil completo (W f64).
    pub fn strip_sound_profile(&self) -> Vec<f64> {
        self.an.strip.as_ref().map(|s| s.g.iter().map(|&g| s.sound(g)).collect()).unwrap_or_default()
    }

    /// [W, v, c_sub, c_super, Mach_sub, Mach_super, ξ_sub, ξ_super, x_h negro, x_h blanco, t, norma,
    /// (las tasas paramétricas con los q de la propia tira, 2πj/W)
    /// m, m\*, k₀, tasa paramétrica sub, tasa paramétrica super, longitud supersónica].
    pub fn strip_info(&self) -> Vec<f64> {
        let Some(s) = self.an.strip.as_ref() else {
            return Vec::new();
        };
        let (cs, cp) = (s.sound(s.g_sub), s.sound(s.g_super));
        let (xb, xw) = s.horizons(1.0);
        // los q de la propia tira (2πj/W): una resonancia más estrecha que 2π/W no tiene modo en ella
        let pr_sub = parametric_rate_grid(s.mass, s.g_sub, s.rho, s.emax, 1, s.w, s.k0, 0.0).0;
        let pr_sup = parametric_rate_grid(s.mass, s.g_super, s.rho, s.emax, 1, s.w, s.k0, 0.0).0;
        vec![
            s.w as f64,
            s.vg,
            cs,
            cp,
            s.vg.abs() / cs,
            s.vg.abs() / cp,
            s.xi(s.g_sub),
            s.xi(s.g_super),
            xb,
            xw,
            s.t as f64,
            s.norm(),
            s.mass,
            s.m_star,
            s.k0,
            pr_sub,
            pr_sup,
            s.length,
        ]
    }

    /// Modo de Fourier dominante de δn en la tira: [q, |δn_q|].
    pub fn strip_dominant_mode(&self) -> Vec<f64> {
        self.an.strip.as_ref().map(|s| {
            let (q, a) = s.dominant_mode();
            vec![q, a]
        }).unwrap_or_default()
    }

    /// Prepara el cálculo por lotes de ξ_sub·G/ρ̄ en el paso t (0 = caminata, 1 = continuo) con una
    /// rampa smoothstep de `ramp` pasos (0 = salto brusco), calculando una de cada `row_stride` filas.
    pub fn hawking_begin(&mut self, engine: u32, ramp: u32, t: u32, row_stride: u32) -> bool {
        let Some(st) = self.an.strip.as_ref() else {
            self.set_message("Hawking por lotes: primero acoustic_strip_init.");
            return false;
        };
        let t = t.min(50_000);
        let min_stride = st.w.div_ceil(1024).max(1);
        let stride = (row_stride.max(1) as usize).max(min_stride);
        self.an.job = Some(HawkingJob::new(st, engine.min(1), ramp, t, stride));
        true
    }

    /// Calcula `rows` filas más; devuelve la fracción hecha (1 = terminado: el mapa está listo).
    pub fn hawking_advance(&mut self, rows: u32) -> f64 {
        let Some(job) = self.an.job.as_mut() else {
            return 1.0;
        };
        let f = job.advance(rows.max(1) as usize);
        if job.done() {
            self.an.map = job.map();
            self.an.map_n = job.rows.len();
        }
        f
    }

    /// Lado del mapa (filas calculadas).
    pub fn hawking_map_size(&self) -> u32 {
        self.an.map_n as u32
    }

    pub fn hawking_map_ptr(&self) -> *const f64 {
        self.an.map.as_ptr()
    }

    /// Filas x del mapa (u32).
    pub fn hawking_rows(&self) -> Vec<u32> {
        self.an.job.as_ref().map(|j| j.rows.iter().map(|&x| x as u32).collect()).unwrap_or_default()
    }

    /// Promedio por rayos del mapa desde el horizonte negro (§3.26): para cada velocidad aguas abajo
    /// v_d, la media de ξ·G/ρ̄ en los puntos (x_h − v_u·s, x_h + v_d·s) con s entre el paso de formación
    /// del horizonte y el paso t (el lóbulo es causal), sin los puntos con |x′ − x| < 6ξ_sub (las
    /// correlaciones de corto alcance de la diagonal). Devuelve [v_d del mínimo, mínimo, x_h, t_h,
    /// v_d₀, media₀, v_d₁, media₁, …] con v_d de `vd_min` a `vd_max` en pasos de `dvd`.
    pub fn hawking_lobe(&self, v_up: f64, vd_min: f64, vd_max: f64, dvd: f64) -> Vec<f64> {
        let (Some(st), Some(job)) = (self.an.strip.as_ref(), self.an.job.as_ref()) else {
            return Vec::new();
        };
        let nm = self.an.map_n;
        if nm == 0 || !job.done() || self.an.map.len() < nm * nm {
            return Vec::new();
        }
        if !v_up.is_finite() || !vd_min.is_finite() || !vd_max.is_finite() || !dvd.is_finite() || dvd <= 0.0 || vd_max < vd_min {
            return Vec::new();
        }
        let stride = if job.rows.len() > 1 { (job.rows[1] - job.rows[0]) as f64 } else { 1.0 };
        let xh = st.horizons(1.0).0;
        let th = st.horizon_formation(job.ramp);
        let smax = job.t as f64 - th;
        let dmin = 6.0 * st.xi(st.g_sub);
        let map = &self.an.map;
        let at = |x: f64, y: f64| -> f64 {
            // bilineal en la rejilla de filas calculadas
            let (u, v) = (x / stride, y / stride);
            let i0 = (u.floor().max(0.0) as usize).min(nm - 1);
            let j0 = (v.floor().max(0.0) as usize).min(nm - 1);
            let fu = (u - i0 as f64).clamp(0.0, 1.0);
            let fv = (v - j0 as f64).clamp(0.0, 1.0);
            let (i1, j1) = ((i0 + 1).min(nm - 1), (j0 + 1).min(nm - 1));
            map[i0 * nm + j0] * (1.0 - fu) * (1.0 - fv) + map[i1 * nm + j0] * fu * (1.0 - fv) + map[i0 * nm + j1] * (1.0 - fu) * fv + map[i1 * nm + j1] * fu * fv
        };
        let mut out = vec![f64::NAN, f64::INFINITY, xh, th];
        let nv = (((vd_max - vd_min) / dvd.max(1e-9)).round() as usize).min(10_000);
        let wmax = (st.w - 1) as f64;
        for k in 0..=nv {
            let vd = vd_min + k as f64 * dvd;
            let (mut sum, mut cnt) = (0.0, 0usize);
            let mut s = 0.0;
            while s < smax {
                let (x, xp) = (xh - v_up * s, xh + vd * s);
                if xp - x >= dmin && x >= 0.0 && x <= wmax && xp >= 0.0 && xp <= wmax {
                    sum += at(x, xp);
                    cnt += 1;
                }
                s += 5.0;
            }
            let mean = if cnt > 0 { sum / cnt as f64 } else { f64::NAN };
            if mean < out[1] {
                out[0] = vd;
                out[1] = mean;
            }
            out.push(vd);
            out.push(mean);
        }
        out
    }

    /// [Tr Σ₀/(2N), modos de norma positiva, rampa, canales] del vacío del último cálculo.
    pub fn hawking_vacuum_info(&self) -> Vec<f64> {
        self.an.job.as_ref().map(|j| vec![j.tr_sigma, j.n_modes as f64, j.ramp as f64, j.nch as f64]).unwrap_or_default()
    }

    /// §2.28, por lotes: W×W f64 de ξ_sub·G/ρ̄ en el paso t (0 = caminata, 1 = continuo), vacío de
    /// Bogoliubov por bloques, rampa smoothstep de `ramp` pasos; propaga hacia atrás las W filas de δn.
    pub fn hawking_correlations(&mut self, engine: u32, ramp: u32, t: u32) -> *const f64 {
        if self.hawking_begin(engine, ramp, t, 1) {
            let rows = self.an.job.as_ref().map(|j| j.rows.len()).unwrap_or(0) as u32;
            self.hawking_advance(rows);
        }
        self.an.map.as_ptr()
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Montaje del escenario 10 con los valores de §3.26: 0 = láser de agujero negro (tira de 512,
    /// m = 1,4, ruido de 10⁻⁷), 1 = Hawking espontáneo (tira de 512, m = 1,35; por lotes),
    /// 2 = vórtice de carga 2 en un disco (2D, m = 1,4, ξ = 8, R = 76, semilla cuadrupolar de 10⁻³),
    /// 3 = reconexión de anillos (3D, m = 0,5, g = −0,03, R = 20), 4 = Unruh de fermiones (sin malla).
    pub fn set_analog_setup(&mut self, setup: u32) {
        self.message.clear();
        if self.scenario != 10 {
            self.set_message("Los montajes de gravedad análoga son del escenario 10.");
            return;
        }
        let setup = setup.min(4);
        if setup == 2 && self.dim != 2 {
            self.set_message("El vórtice en un disco es 2D: cambia a una malla 2D (160² o mayor).");
            return;
        }
        if setup == 3 && self.dim != 3 {
            self.set_message("La reconexión de anillos es 3D: cambia a una malla 3D (96³).");
            return;
        }
        self.clear_barriers();
        self.an = Analog::new();
        self.an.setup = setup;
        self.clear();
        self.e_max = 1.0;
        self.alpha = 0.0;
        self.e_cap = 0.0;
        match self.an.setup {
            0 => {
                self.acoustic_strip_init(512, 16, -0.02, -0.0025, 10.0, 1.4);
                self.strip_add_noise(1e-7, 1);
                let rows = self.h.clamp(16, 1024) as u32;
                self.strip_history(rows, 100);
            }
            1 => {
                self.acoustic_strip_init(512, 7, -0.01, -0.01 / 16.0, 240.0, 1.35);
            }
            2 => {
                self.set_mass(1.4);
                self.g = -0.00303;
                self.seed_uniform_background(0.5, 0.0, 1);
                let (cx, cy) = (self.w as f32 / 2.0 - 0.5, self.h as f32 / 2.0 - 0.5);
                let r = (76.0f32).min(self.w.min(self.h) as f32 / 2.0 - 2.0);
                self.add_disk_wall(cx, cy, r);
                self.imprint_vortex(cx, cy, 2);
                self.seed_disk_multipole(2, 1e-3);
            }
            3 => {
                self.set_mass(0.5);
                self.g = -0.03;
                self.seed_uniform_background(0.5, 0.0, 1);
                let (cx, cy, cz) = (self.w as f32 / 2.0, self.h as f32 / 2.0, self.d as f32 / 2.0);
                let sep = (22.0f32).min(self.d as f32 / 4.0);
                let r = (20.0f32).min(self.w.min(self.h) as f32 / 4.0);
                // el anillo de abajo avanza hacia +z (signo −1) y el de arriba hacia −z (signo +1)
                self.imprint_vortex_ring(cx, cy - 9.0, cz - sep, r, -1);
                self.imprint_vortex_ring(cx, cy + 9.0, cz + sep, r, 1);
            }
            _ => {}
        }
        self.t = 0;
    }

    pub fn analog_setup(&self) -> u32 {
        self.an.setup
    }

    /// Para el HUD del disco y de los anillos: [m, m\*, g, ρ̄, ξ, c_s, cuanto de circulación 2π/m\*,
    /// tasa paramétrica del fondo con nq valores de q por eje (0 = no la calcula)].
    pub fn analog_fluid_info(&self, nq: u32) -> Vec<f64> {
        let rho = self.vort.rho_bg as f64;
        let ms = self.band_mass();
        let fp = self.nonlinear_slope(rho);
        let dim = if self.dim == 3 { 3 } else { 2 };
        let pr = if nq == 0 || !(rho > 0.0 && rho.is_finite()) {
            f64::NAN
        } else {
            parametric_rate_impl(self.mass as f64, self.g as f64, rho, self.e_max as f64, dim, nq as usize, 0.0).0
        };
        let cs = if rho > 0.0 && ms > 1e-30 && fp >= 0.0 { (rho * fp / ms).sqrt() } else { f64::NAN };
        let circ = if ms > 1e-30 { 2.0 * PI / ms } else { f64::NAN };
        vec![self.mass as f64, ms, self.g as f64, rho, self.analog_xi(), cs, circ, pr]
    }

    /// Fase 14 (Micheli, Oshima y Takahashi, PRD 2026): evolución gaussiana de dos modos (k, −k)
    /// acoplados por compresión paramétrica λ y canales de Lindblad (amortiguamiento térmico γ_damp,
    /// ocupación n_th y difusión de cuadratura γ_diff).
    /// Devuelve puntero a [pureza_fin, r_eff_fin, r_libre_fin, nu_menos, nu_tilde_menos, E_N, n_pts,
    /// t0, pur0, r0, EN0, …] (guardado en `self.an.out`).
    #[allow(clippy::too_many_arguments)]
    pub fn gaussian_desqueezing_run(&mut self, omega: f32, lambda_sq: f32, gamma_damp: f32, n_th: f32, gamma_diff: f32, t_max: f32, steps: u32) -> *const f64 {
        let res = gaussian_two_mode_evolve(
            clamp_finite(omega as f64, -1e3, 1e3, 0.0),
            clamp_finite(lambda_sq as f64, -10.0, 10.0, 0.5),
            clamp_finite(gamma_damp as f64, 0.0, 1e3, 0.0),
            clamp_finite(n_th as f64, 0.0, 1e6, 0.0),
            clamp_finite(gamma_diff as f64, 0.0, 1e3, 0.0),
            clamp_finite(t_max as f64, 0.05, 50.0, 2.0),
            (steps as usize).clamp(20, 4000),
        );
        self.an.out = res;
        self.an.out.as_ptr()
    }

    /// Fase 14 (Micheli et al., PRD 2026): construye dos estados gaussianos abiertos de dos modos
    /// con la MISMA pureza γ (ajustada por bisección a < 10⁻¹⁰):
    ///   A: compresión fuerte `lambda_a` + disipación térmica (`gamma_a`, `n_th`) → sigue entrelazado.
    ///   B: compresión más débil / des-comprimido `lambda_b` + difusión de cuadratura → separable.
    /// Devuelve [γ_A, r_A, ν̃_−A, E_NA, γ_B, r_B, ν̃_−B, E_NB, γ_diff_B].
    pub fn gaussian_same_purity_comparison(&self, lambda_a: f32, lambda_b: f32, gamma_a: f32, n_th: f32, t_max: f32) -> Vec<f64> {
        let la = clamp_finite(lambda_a as f64, 0.05, 10.0, 1.0);
        let lb = clamp_finite(lambda_b as f64, 0.0, 10.0, 0.2);
        let ga = clamp_finite(gamma_a as f64, 1e-4, 1e3, 0.1);
        let nt = clamp_finite(n_th as f64, 0.0, 1e6, 0.0);
        let tm = clamp_finite(t_max as f64, 0.2, 20.0, 2.0);
        let steps = 400usize;
        let sig_a = gaussian_two_mode_final(0.0, la, ga, nt, 0.0, tm, steps);
        let inv_a = gaussian_invariants(&sig_a);
        let target_pur = inv_a.0;
        // Bisección en γ_diff para que el estado B (con compresión lb) tenga exactamente pureza target_pur
        let mut lo = 0.0f64;
        let mut hi = 50.0f64;
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let s = gaussian_two_mode_final(0.0, lb, 0.0, 0.0, mid, tm, steps);
            let pur = gaussian_invariants(&s).0;
            if pur > target_pur {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let gdiff = 0.5 * (lo + hi);
        let sig_b = gaussian_two_mode_final(0.0, lb, 0.0, 0.0, gdiff, tm, steps);
        let inv_b = gaussian_invariants(&sig_b);
        vec![inv_a.0, inv_a.1, inv_a.3, inv_a.4, inv_b.0, inv_b.1, inv_b.3, inv_b.4, gdiff]
    }

    /// Diagnóstico gaussiano de un par de Bogoliubov con parámetro de compresión `r_sq` (p. ej.
    /// parejas de Hawking con tanh r = exp(−πω/κ_H)) sometido a un canal atenuador/térmico de
    /// transmitancia `eta` ∈ (0, 1] y ocupación térmica `n_th`.
    /// Devuelve [pureza, r_eff, nu_menos, nu_tilde_menos, E_N].
    pub fn analog_bogoliubov_entanglement(&self, r_sq: f32, eta: f32, n_th: f32) -> Vec<f64> {
        let r = clamp_finite(r_sq as f64, 0.0, 8.0, 0.5);
        let et = clamp_finite(eta as f64, 1e-6, 1.0, 1.0);
        let nt = clamp_finite(n_th as f64, 0.0, 1e6, 0.0);
        let ch = (2.0 * r).cosh();
        let sh = (2.0 * r).sinh();
        let a = 0.5 * et * ch + (1.0 - et) * (nt + 0.5);
        let c = 0.5 * et * sh;
        let mut s = [0.0f64; 16];
        s[0] = a;
        s[5] = a;
        s[10] = a;
        s[15] = a;
        s[2] = c;
        s[8] = c;
        s[7] = -c;
        s[13] = -c;
        let (pur, reff, nu_m, nut_m, en) = gaussian_invariants(&s);
        vec![pur, reff, nu_m, nut_m, en]
    }
}

/// Determinante 4 × 4 (fila mayor).
fn det4(m: &[f64; 16]) -> f64 {
    let at = |r: usize, c: usize| m[r * 4 + c];
    let det3 = |r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize| -> f64 {
        at(r0, c0) * (at(r1, c1) * at(r2, c2) - at(r1, c2) * at(r2, c1))
            - at(r0, c1) * (at(r1, c0) * at(r2, c2) - at(r1, c2) * at(r2, c0))
            + at(r0, c2) * (at(r1, c0) * at(r2, c1) - at(r1, c1) * at(r2, c0))
    };
    at(0, 0) * det3(1, 2, 3, 1, 2, 3) - at(0, 1) * det3(1, 2, 3, 0, 2, 3) + at(0, 2) * det3(1, 2, 3, 0, 1, 3) - at(0, 3) * det3(1, 2, 3, 0, 1, 2)
}

/// Autovalores de una matriz simétrica 4 × 4 mediante Jacobi.
fn sym4_eigvals(mut a: [f64; 16]) -> [f64; 4] {
    for _ in 0..30 {
        for p in 0..4 {
            for q in (p + 1)..4 {
                let apq = a[p * 4 + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (a[q * 4 + q] - a[p * 4 + p]) / (2.0 * apq);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                a[p * 4 + p] -= t * apq;
                a[q * 4 + q] += t * apq;
                a[p * 4 + q] = 0.0;
                a[q * 4 + p] = 0.0;
                for r in 0..4 {
                    if r != p && r != q {
                        let arp = a[r * 4 + p];
                        let arq = a[r * 4 + q];
                        a[r * 4 + p] = c * arp - s * arq;
                        a[p * 4 + r] = a[r * 4 + p];
                        a[r * 4 + q] = s * arp + c * arq;
                        a[q * 4 + r] = a[r * 4 + q];
                    }
                }
            }
        }
    }
    let mut vals = [a[0], a[5], a[10], a[15]];
    vals.sort_by(f64::total_cmp);
    vals
}

/// Invariantes gaussianos exactos de una matriz de covarianza 4 × 4 (orden q₁, p₁, q₂, p₂; vacío = ½ I₄):
/// (pureza γ, compresión efectiva r_eff, autovalor simpléctico menor ν_−, transpuesta parcial ν̃_−, negatividad logarítmica E_N).
fn gaussian_invariants(sig: &[f64; 16]) -> (f64, f64, f64, f64, f64) {
    let det_a = sig[0] * sig[5] - sig[1] * sig[4];
    let det_b = sig[10] * sig[15] - sig[11] * sig[14];
    let det_c = sig[2] * sig[7] - sig[3] * sig[6];
    let i4 = det4(sig).max(1e-30);
    let purity = (1.0 / (4.0 * i4.sqrt())).min(1.0);
    let delta = det_a + det_b + 2.0 * det_c;
    let disc = (delta * delta - 4.0 * i4).max(0.0).sqrt();
    let nu_minus = (0.5 * (delta - disc).max(0.0)).sqrt();
    let t_delta = det_a + det_b - 2.0 * det_c;
    let t_disc = (t_delta * t_delta - 4.0 * i4).max(0.0).sqrt();
    let nu_tilde = (0.5 * (t_delta - t_disc).max(0.0)).sqrt();
    let log_neg = if nu_tilde < 0.5 { -(2.0 * nu_tilde).log2() } else { 0.0 };
    let ev = sym4_eigvals(*sig);
    let r_eff = 0.25 * (ev[3].max(1e-15) / ev[0].max(1e-15)).ln().max(0.0);
    (purity, r_eff, nu_minus, nu_tilde, log_neg)
}

/// Ecuación de Lyapunov RK4 para dΣ/dt = A Σ + Σ Aᵀ + D sobre 4 × 4 simétrica.
fn gaussian_lyapunov_rhs(sig: &[f64; 16], a: &[f64; 16], d: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = d[i * 4 + j];
            for k in 0..4 {
                s += a[i * 4 + k] * sig[k * 4 + j] + sig[i * 4 + k] * a[j * 4 + k];
            }
            out[i * 4 + j] = s;
        }
    }
    out
}

fn gaussian_two_mode_matrices(omega: f64, lambda_sq: f64, gamma_damp: f64, n_th: f64, gamma_diff: f64) -> ([f64; 16], [f64; 16]) {
    let g2 = -0.5 * gamma_damp;
    let a = [
        g2, omega, lambda_sq, 0.0,
        -omega, g2, 0.0, -lambda_sq,
        lambda_sq, 0.0, g2, omega,
        0.0, -lambda_sq, -omega, g2,
    ];
    let d_th = gamma_damp * (n_th + 0.5) + gamma_diff;
    let d = [
        d_th, 0.0, 0.0, 0.0,
        0.0, d_th, 0.0, 0.0,
        0.0, 0.0, d_th, 0.0,
        0.0, 0.0, 0.0, d_th,
    ];
    (a, d)
}

fn gaussian_two_mode_final(omega: f64, lambda_sq: f64, gamma_damp: f64, n_th: f64, gamma_diff: f64, t_max: f64, steps: usize) -> [f64; 16] {
    let (a, d) = gaussian_two_mode_matrices(omega, lambda_sq, gamma_damp, n_th, gamma_diff);
    let mut sig = [0.0f64; 16];
    sig[0] = 0.5;
    sig[5] = 0.5;
    sig[10] = 0.5;
    sig[15] = 0.5;
    let dt = t_max / steps as f64;
    for _ in 0..steps {
        let k1 = gaussian_lyapunov_rhs(&sig, &a, &d);
        let mut t2 = [0.0; 16];
        for i in 0..16 { t2[i] = sig[i] + 0.5 * dt * k1[i]; }
        let k2 = gaussian_lyapunov_rhs(&t2, &a, &d);
        let mut t3 = [0.0; 16];
        for i in 0..16 { t3[i] = sig[i] + 0.5 * dt * k2[i]; }
        let k3 = gaussian_lyapunov_rhs(&t3, &a, &d);
        let mut t4 = [0.0; 16];
        for i in 0..16 { t4[i] = sig[i] + dt * k3[i]; }
        let k4 = gaussian_lyapunov_rhs(&t4, &a, &d);
        for i in 0..16 {
            sig[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
    sig
}

fn gaussian_two_mode_evolve(omega: f64, lambda_sq: f64, gamma_damp: f64, n_th: f64, gamma_diff: f64, t_max: f64, steps: usize) -> Vec<f64> {
    let (a, d) = gaussian_two_mode_matrices(omega, lambda_sq, gamma_damp, n_th, gamma_diff);
    let mut sig = [0.0f64; 16];
    sig[0] = 0.5;
    sig[5] = 0.5;
    sig[10] = 0.5;
    sig[15] = 0.5;
    let dt = t_max / steps as f64;
    let sample_every = (steps / 60).max(1);
    let mut series = Vec::new();
    let i0 = gaussian_invariants(&sig);
    series.extend_from_slice(&[0.0, i0.0, i0.1, i0.4]);
    for step in 1..=steps {
        let k1 = gaussian_lyapunov_rhs(&sig, &a, &d);
        let mut t2 = [0.0; 16];
        for i in 0..16 { t2[i] = sig[i] + 0.5 * dt * k1[i]; }
        let k2 = gaussian_lyapunov_rhs(&t2, &a, &d);
        let mut t3 = [0.0; 16];
        for i in 0..16 { t3[i] = sig[i] + 0.5 * dt * k2[i]; }
        let k3 = gaussian_lyapunov_rhs(&t3, &a, &d);
        let mut t4 = [0.0; 16];
        for i in 0..16 { t4[i] = sig[i] + dt * k3[i]; }
        let k4 = gaussian_lyapunov_rhs(&t4, &a, &d);
        for i in 0..16 {
            sig[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        if step % sample_every == 0 || step == steps {
            let inv = gaussian_invariants(&sig);
            series.extend_from_slice(&[step as f64 * dt, inv.0, inv.1, inv.4]);
        }
    }
    let fin = gaussian_invariants(&sig);
    let sig_free = gaussian_two_mode_final(omega, lambda_sq, 0.0, 0.0, 0.0, t_max, steps);
    let r_free = gaussian_invariants(&sig_free).1;
    let n_pts = (series.len() / 4) as f64;
    let mut out = vec![fin.0, fin.1, r_free, fin.2, fin.3, fin.4, n_pts];
    out.extend_from_slice(&series);
    out
}
