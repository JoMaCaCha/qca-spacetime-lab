//! Skyrmiones (§2.26, escenario 8): bariones como solitones topológicos de un campo clásico de piones
//! U(x) ∈ SU(2), con φ un vector unitario de 4 componentes (U = φ₀ + iφ·τ). En unidades de Manton la
//! energía es E = ∫ [Σ_i ∂_iφ·∂_iφ + ½((Σ_i a_ii)² − Σ_ij a_ij²)] d³x, con a_ij = ∂_iφ·∂_jφ, y cumple
//! E ≥ 12π²·|B|; la carga es B = (1/2π²)·∫ det[φ, ∂₁φ, ∂₂φ, ∂₃φ] d³x.
//! - En la red, la derivada centrada vale 0 en k = π (los dameros no cuestan energía y el solitón se
//!   desenrolla): la energía es la media de la calculada con diferencias hacia delante y hacia atrás.
//! - Relajación: L-BFGS sobre ψ con φ = ψ/|ψ|, gradiente analítico y el vacío fijo en las dos capas
//!   del borde.
//! - El perfil radial del erizo (problema de contorno con la cola C/r² de piones sin masa) da la masa y
//!   el momento de inercia de la cuantización de Adkins, Nappi y Witten (1983).
//! Es una teoría efectiva de piones, no QCD; el campo de piones se pone a mano.

use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct Skyrme {
    pub active: bool,
    pub n: usize,
    pub h: f64,
    pub half: f64,
    /// ψ (4 por punto); φ = ψ/|ψ|.
    pub psi: Vec<f64>,
    pub free: Vec<bool>,
    pub baryons: u32,
    pub dens: Vec<f32>,
    pub centered: bool,
    pub lbfgs: Option<Lbfgs>,
    pub energy: f64,
    pub quant: Vec<f64>,
}

impl Skyrme {
    pub fn new() -> Self {
        Skyrme { active: false, n: 0, h: 0.15, half: 5.0, psi: Vec::new(), free: Vec::new(), baryons: 1, dens: Vec::new(), centered: false, lbfgs: None, energy: 0.0, quant: Vec::new() }
    }

    #[inline(always)]
    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.n + y) * self.n + x
    }

    fn phi_at(&self, i: usize) -> [f64; 4] {
        let p = &self.psi[4 * i..4 * i + 4];
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]).sqrt().max(1e-300);
        [p[0] / r, p[1] / r, p[2] / r, p[3] / r]
    }
}

#[inline(always)]
fn dot4(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Densidad de energía y su derivada respecto de los tres vectores d_i (las derivadas de φ):
/// e = Σ a_ii + ½(S² − Σ a_ij²), ∂e/∂d_i = 2d_i + 2(S·d_i − Σ_j a_ij·d_j).
#[inline(always)]
fn skyrme_density(d: &[[f64; 4]; 3], grad: Option<&mut [[f64; 4]; 3]>) -> f64 {
    let mut a = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            a[i][j] = dot4(&d[i], &d[j]);
            a[j][i] = a[i][j];
        }
    }
    let s = a[0][0] + a[1][1] + a[2][2];
    let mut t = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            t += a[i][j] * a[i][j];
        }
    }
    if let Some(g) = grad {
        for i in 0..3 {
            for c in 0..4 {
                let mut v = (2.0 + 2.0 * s) * d[i][c];
                for j in 0..3 {
                    v -= 2.0 * a[i][j] * d[j][c];
                }
                g[i][c] = v;
            }
        }
    }
    s + 0.5 * (s * s - t)
}

impl Skyrme {
    /// Energía (media de diferencias hacia delante y hacia atrás, o centrada) y gradiente respecto de ψ.
    pub fn energy_grad(&self, psi: &[f64], grad: Option<&mut [f64]>) -> f64 {
        let n = self.n;
        let h = self.h;
        let np = n * n * n;
        let phi: Vec<[f64; 4]> = (0..np)
            .map(|i| {
                let p = &psi[4 * i..4 * i + 4];
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]).sqrt().max(1e-300);
                [p[0] / r, p[1] / r, p[2] / r, p[3] / r]
            })
            .collect();
        let want = grad.is_some();
        let mut gphi = if want { vec![[0.0f64; 4]; np] } else { Vec::new() };
        let mut e = 0.0;
        // las dos capas del borde son vacío fijo: basta recorrer el interior 1..n−1, donde existen los
        // seis vecinos (en la capa 0 todas las diferencias se anulan)
        let strides = [1usize, n, n * n];
        let (versions, w): (&[i64], f64) = if self.centered { (&[0], 1.0) } else { (&[1, -1], 0.5) };
        let h3 = h * h * h;
        for z in 1..n - 1 {
            for y in 1..n - 1 {
                for x in 1..n - 1 {
                    let i = self.idx(x, y, z);
                    for &s in versions {
                        let mut d = [[0.0f64; 4]; 3];
                        let mut nb = [(0usize, 0usize, 0.0f64); 3];
                        for a in 0..3 {
                            let st = strides[a];
                            let (p, m, f) = match s {
                                0 => (i + st, i - st, 1.0 / (2.0 * h)),
                                1 => (i + st, i, 1.0 / h),
                                _ => (i, i - st, 1.0 / h),
                            };
                            let (pp, pm) = (&phi[p], &phi[m]);
                            for c in 0..4 {
                                d[a][c] = (pp[c] - pm[c]) * f;
                            }
                            nb[a] = (p, m, f);
                        }
                        let mut g = [[0.0f64; 4]; 3];
                        let ed = skyrme_density(&d, if want { Some(&mut g) } else { None });
                        e += w * ed * h3;
                        if want {
                            for a in 0..3 {
                                let (p, m, f) = nb[a];
                                let k = w * h3 * f;
                                for c in 0..4 {
                                    gphi[p][c] += k * g[a][c];
                                    gphi[m][c] -= k * g[a][c];
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(gr) = grad {
            for i in 0..np {
                let p = &psi[4 * i..4 * i + 4];
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]).sqrt().max(1e-300);
                let f = &phi[i];
                let g = &gphi[i];
                let pg = dot4(f, g);
                for c in 0..4 {
                    gr[4 * i + c] = if self.free[i] { (g[c] - f[c] * pg) / r } else { 0.0 };
                }
            }
        }
        e
    }

    /// B = (1/2π²)·Σ det[φ, ∂₁φ, ∂₂φ, ∂₃φ]·h³ con derivadas centradas de cuarto orden (error O(h⁴)).
    pub fn charge(&self) -> f64 {
        let n = self.n;
        let h = self.h;
        let mut b = 0.0;
        for z in 2..n - 2 {
            for y in 2..n - 2 {
                for x in 2..n - 2 {
                    let f = self.phi_at(self.idx(x, y, z));
                    let dd = |a: usize| -> [f64; 4] {
                        let at = |s: i64| {
                            let mut c = [x as i64, y as i64, z as i64];
                            c[a] += s;
                            self.phi_at(self.idx(c[0] as usize, c[1] as usize, c[2] as usize))
                        };
                        let (p1, p2, m1, m2) = (at(1), at(2), at(-1), at(-2));
                        [0, 1, 2, 3].map(|c| (-p2[c] + 8.0 * p1[c] - 8.0 * m1[c] + m2[c]) / (12.0 * h))
                    };
                    let rows = [f, dd(0), dd(1), dd(2)];
                    b += det4(&rows) * h * h * h;
                }
            }
        }
        b / (2.0 * PI * PI)
    }

    /// Densidad de energía por punto (media de las dos versiones), para dibujarla.
    pub fn refresh_density(&mut self) {
        let n = self.n;
        let h = self.h;
        let np = n * n * n;
        let mut dens = vec![0.0f32; np];
        for z in 1..n - 1 {
            for y in 1..n - 1 {
                for x in 1..n - 1 {
                    let i = self.idx(x, y, z);
                    let f = self.phi_at(i);
                    let mut acc = 0.0;
                    for s in [1i64, -1] {
                        let mut d = [[0.0f64; 4]; 3];
                        for a in 0..3 {
                            let mut c = [x as i64, y as i64, z as i64];
                            c[a] += s;
                            let p = self.phi_at(self.idx(c[0] as usize, c[1] as usize, c[2] as usize));
                            for k in 0..4 {
                                d[a][k] = (p[k] - f[k]) / (s as f64 * h);
                            }
                        }
                        acc += 0.5 * skyrme_density(&d, None);
                    }
                    dens[i] = acc as f32;
                }
            }
        }
        self.dens = dens;
    }
}

fn det4(m: &[[f64; 4]; 4]) -> f64 {
    let mut a = *m;
    let mut det = 1.0;
    for c in 0..4 {
        let mut piv = c;
        for r in c + 1..4 {
            if a[r][c].abs() > a[piv][c].abs() {
                piv = r;
            }
        }
        if a[piv][c].abs() < 1e-300 {
            return 0.0;
        }
        if piv != c {
            a.swap(piv, c);
            det = -det;
        }
        det *= a[c][c];
        for r in c + 1..4 {
            let f = a[r][c] / a[c][c];
            for k in c..4 {
                a[r][k] -= f * a[c][k];
            }
        }
    }
    det
}

/// L-BFGS con memoria m y búsqueda lineal con retroceso (condición de Armijo).
pub struct Lbfgs {
    pub m: usize,
    pub s: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
    pub x: Vec<f64>,
    pub g: Vec<f64>,
    pub f: f64,
}

fn vdot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Lbfgs {
    pub fn new(x: Vec<f64>, f: f64, g: Vec<f64>) -> Self {
        Lbfgs { m: 10, s: Vec::new(), y: Vec::new(), x, g, f }
    }

    /// Una iteración: dirección por la doble recursión y retroceso desde el paso 1.
    pub fn iterate(&mut self, fun: &dyn Fn(&[f64], &mut [f64]) -> f64) -> bool {
        let k = self.s.len();
        let mut q = self.g.clone();
        let mut alpha = vec![0.0; k];
        for i in (0..k).rev() {
            let rho = 1.0 / vdot(&self.y[i], &self.s[i]);
            alpha[i] = rho * vdot(&self.s[i], &q);
            for (qq, yy) in q.iter_mut().zip(&self.y[i]) {
                *qq -= alpha[i] * yy;
            }
        }
        let gamma = if k > 0 { vdot(&self.s[k - 1], &self.y[k - 1]) / vdot(&self.y[k - 1], &self.y[k - 1]) } else { 1e-3 / vdot(&self.g, &self.g).sqrt().max(1e-300) };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..k {
            let rho = 1.0 / vdot(&self.y[i], &self.s[i]);
            let beta = rho * vdot(&self.y[i], &q);
            for (qq, ss) in q.iter_mut().zip(&self.s[i]) {
                *qq += (alpha[i] - beta) * ss;
            }
        }
        // dirección de descenso d = −q
        let gd = -vdot(&self.g, &q);
        if gd >= 0.0 {
            self.s.clear();
            self.y.clear();
            return false;
        }
        let mut t = 1.0;
        let mut xn = vec![0.0; self.x.len()];
        let mut gn = vec![0.0; self.x.len()];
        for _ in 0..30 {
            for i in 0..xn.len() {
                xn[i] = self.x[i] - t * q[i];
            }
            let fnew = fun(&xn, &mut gn);
            if fnew <= self.f + 1e-4 * t * gd {
                let s: Vec<f64> = xn.iter().zip(&self.x).map(|(a, b)| a - b).collect();
                let y: Vec<f64> = gn.iter().zip(&self.g).map(|(a, b)| a - b).collect();
                if vdot(&s, &y) > 1e-300 {
                    self.s.push(s);
                    self.y.push(y);
                    if self.s.len() > self.m {
                        self.s.remove(0);
                        self.y.remove(0);
                    }
                }
                self.x = xn;
                self.g = gn;
                self.f = fnew;
                return true;
            }
            t *= 0.5;
        }
        self.s.clear();
        self.y.clear();
        false
    }
}

/// Perfil radial del erizo: minimiza E = 4π∫[r²f′² + 2sen²f(1 + f′²) + sen⁴f/r²]dr en [0, R] con
/// f(0) = π y la cola C/r² fuera (energía 8π·f(R)²·R, que impone f′(R) = −2f(R)/R). Devuelve (f, dr).
pub fn hedgehog_profile(r_max: f64, npts: usize) -> (Vec<f64>, f64) {
    let dr = r_max / (npts - 1) as f64;
    let mut f: Vec<f64> = (0..npts).map(|i| PI * (-(i as f64 * dr) / 1.2).exp().max(1e-12)).collect();
    // energía discreta (puntos medios) y su gradiente
    let fun = |f: &[f64], g: &mut [f64]| -> f64 {
        let n = f.len();
        for v in g.iter_mut() {
            *v = 0.0;
        }
        let mut e = 0.0;
        for i in 0..n - 1 {
            let r = (i as f64 + 0.5) * dr;
            let fm = 0.5 * (f[i] + f[i + 1]);
            let fp = (f[i + 1] - f[i]) / dr;
            let (s, c) = fm.sin_cos();
            let s2 = s * s;
            let dens = r * r * fp * fp + 2.0 * s2 * (1.0 + fp * fp) + s2 * s2 / (r * r);
            e += 4.0 * PI * dens * dr;
            // derivadas respecto de fm y fp
            let dfm = 4.0 * PI * dr * (2.0 * 2.0 * s * c * (1.0 + fp * fp) + 4.0 * s2 * s * c / (r * r));
            let dfp = 4.0 * PI * dr * (2.0 * r * r * fp + 4.0 * s2 * fp);
            g[i] += 0.5 * dfm - dfp / dr;
            g[i + 1] += 0.5 * dfm + dfp / dr;
        }
        let rr = (n - 1) as f64 * dr;
        e += 8.0 * PI * f[n - 1] * f[n - 1] * rr;
        g[n - 1] += 16.0 * PI * f[n - 1] * rr;
        g[0] = 0.0;
        e
    };
    let mut g = vec![0.0; npts];
    let e0 = fun(&f, &mut g);
    let mut opt = Lbfgs::new(f.clone(), e0, g);
    opt.m = 20;
    for _ in 0..20000 {
        let fx = opt.f;
        if !opt.iterate(&fun) {
            if !opt.iterate(&fun) {
                break;
            }
        }
        if (fx - opt.f).abs() < 1e-15 * fx.abs() {
            break;
        }
    }
    f = opt.x;
    // Newton sobre las ecuaciones discretas (jacobiano tridiagonal por diferencias en tres colores),
    // hasta que el gradiente se anule: la energía converge mucho antes que la cola del perfil
    let n = f.len();
    let mut g = vec![0.0; n];
    for _ in 0..60 {
        fun(&f, &mut g);
        let gmax = g[1..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if gmax < 1e-13 {
            break;
        }
        let eps = 1e-7;
        let (mut lo, mut di, mut up) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut gp = vec![0.0; n];
        for c in 0..3 {
            let mut fp = f.clone();
            for i in (1..n).filter(|i| i % 3 == c) {
                fp[i] += eps;
            }
            fun(&fp, &mut gp);
            for i in (1..n).filter(|i| i % 3 == c) {
                di[i] = (gp[i] - g[i]) / eps;
                if i + 1 < n {
                    lo[i + 1] = (gp[i + 1] - g[i + 1]) / eps;
                }
                if i > 1 {
                    up[i - 1] = (gp[i - 1] - g[i - 1]) / eps;
                }
            }
        }
        // Thomas en los índices 1..n (f₀ = π fijo)
        let mut cp = vec![0.0; n];
        let mut dp = vec![0.0; n];
        for i in 1..n {
            let m = di[i] - if i > 1 { lo[i] * cp[i - 1] } else { 0.0 };
            cp[i] = up[i] / m;
            dp[i] = (-g[i] - if i > 1 { lo[i] * dp[i - 1] } else { 0.0 }) / m;
        }
        let mut dx = vec![0.0; n];
        for i in (1..n).rev() {
            dx[i] = dp[i] - if i + 1 < n { cp[i] * dx[i + 1] } else { 0.0 };
        }
        for i in 1..n {
            f[i] += dx[i];
        }
    }
    (f, dr)
}

/// Cuantización semiclásica de Adkins, Nappi y Witten con el perfil radial: [E/12π², Λ, e, F_π (MeV),
/// masa clásica (MeV), radio isoescalar (fm), C de la cola].
pub fn anw_quantization(r_max: f64, npts: usize) -> Vec<f64> {
    let (f, dr) = hedgehog_profile(r_max, npts);
    let n = f.len();
    let (mut e, mut lam, mut r2) = (0.0, 0.0, 0.0);
    for i in 0..n - 1 {
        let r = (i as f64 + 0.5) * dr;
        let fm = 0.5 * (f[i] + f[i + 1]);
        let fp = (f[i + 1] - f[i]) / dr;
        let s2 = fm.sin().powi(2);
        e += 4.0 * PI * (r * r * fp * fp + 2.0 * s2 * (1.0 + fp * fp) + s2 * s2 / (r * r)) * dr;
        // Λ de ANW en sus unidades (r̃ = 2r): 8∫r²sen²f(1 + f′² + sen²f/r²)dr
        lam += 8.0 * r * r * s2 * (1.0 + fp * fp + s2 / (r * r)) * dr;
        // ⟨r²⟩ de la densidad bariónica, −(2/π)∫r²sen²f·f′dr (en unidades de Manton)
        r2 += -(2.0 / PI) * r * r * s2 * fp * dr;
    }
    let rr = (n - 1) as f64 * dr;
    e += 8.0 * PI * f[n - 1] * f[n - 1] * rr;
    let c_tail = f[n - 1] * rr * rr;
    // cola C/r² fuera de R: aporta 8∫_R^∞ r²·(C/r²)² dr = 8C²/R al momento de inercia
    lam += 8.0 * c_tail * c_tail / rr;
    let e_norm = e / (12.0 * PI * PI);
    // M = E·F_π/(4e) = m̃·F_π/e
    let mtil = e / 4.0;
    let (mn, md) = (938.9, 1232.0);
    let lambda_mev = 3.0 / (2.0 * (md - mn));
    let mass = mn - 3.0 / (8.0 * lambda_mev);
    let fpi_over_e = mass / mtil;
    let e3fpi = (2.0 * PI / 3.0) * lam / lambda_mev;
    let ee = (e3fpi / fpi_over_e).powf(0.25);
    let fpi = fpi_over_e * ee;
    let radius = (4.0 * r2).sqrt() / (ee * fpi) * 197.327;
    vec![e_norm, lam, ee, fpi, mass, radius, c_tail]
}

impl QuantumSpacetimeUniverse {
    pub(crate) fn skyrme_fun(&self) -> impl Fn(&[f64], &mut [f64]) -> f64 + '_ {
        move |x: &[f64], g: &mut [f64]| self.sky.energy_grad(x, Some(g))
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.26 (escenario 8): erizo (B = 1) o dos erizos a ±1,4 en x con isorrotación relativa de π
    /// alrededor de z (el canal atractivo, B = 2), perfil inicial tosco f = π·e^{−r/1,2}, en una malla
    /// cúbica de ±half_box con paso h. El vacío queda fijo en las dos capas del borde.
    pub fn skyrme_init(&mut self, baryons: u32, h: f32, half_box: f32) {
        let h = (h as f64).max(0.05);
        let half = (half_box as f64).max(1.0);
        let n = (2.0 * half / h).round() as usize + 1;
        let mut sky = Skyrme::new();
        sky.n = n;
        sky.h = h;
        sky.half = half;
        sky.baryons = baryons.clamp(1, 2);
        let np = n * n * n;
        sky.psi = vec![0.0; 4 * np];
        sky.free = vec![false; np];
        let hedgehog = |x: f64, y: f64, z: f64| -> [f64; 4] {
            let r = (x * x + y * y + z * z).sqrt();
            let f = PI * (-r / 1.2).exp();
            let (s, c) = f.sin_cos();
            if r < 1e-12 {
                return [c, 0.0, 0.0, s];
            }
            [c, s * x / r, s * y / r, s * z / r]
        };
        // producto de cuaterniones (φ₀ + iφ·τ)(χ₀ + iχ·τ)
        let qmul = |a: [f64; 4], b: [f64; 4]| -> [f64; 4] {
            [
                a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                a[0] * b[1] + b[0] * a[1] - (a[2] * b[3] - a[3] * b[2]),
                a[0] * b[2] + b[0] * a[2] - (a[3] * b[1] - a[1] * b[3]),
                a[0] * b[3] + b[0] * a[3] - (a[1] * b[2] - a[2] * b[1]),
            ]
        };
        for zi in 0..n {
            for yi in 0..n {
                for xi in 0..n {
                    let (x, y, z) = (-half + xi as f64 * h, -half + yi as f64 * h, -half + zi as f64 * h);
                    let i = (zi * n + yi) * n + xi;
                    let border = [xi, yi, zi].iter().any(|&c| c < 2 || c >= n - 2);
                    sky.free[i] = !border;
                    let phi = if border {
                        [1.0, 0.0, 0.0, 0.0]
                    } else if sky.baryons == 1 {
                        hedgehog(x, y, z)
                    } else {
                        // U₁(x − a)·C·U₀(x + a)·C†, C = rotación de π alrededor de z: (π₁, π₂, π₃) → (−π₁, −π₂, π₃)
                        let a = hedgehog(x - 1.4, y, z);
                        let b = hedgehog(x + 1.4, y, z);
                        qmul(a, [b[0], -b[1], -b[2], b[3]])
                    };
                    sky.psi[4 * i..4 * i + 4].copy_from_slice(&phi);
                }
            }
        }
        sky.active = true;
        self.sky = sky;
        let mut g = vec![0.0; 4 * np];
        let f = self.sky.energy_grad(&self.sky.psi, Some(&mut g));
        self.sky.energy = f;
        self.sky.lbfgs = Some(Lbfgs::new(self.sky.psi.clone(), f, g));
        self.sky.refresh_density();
    }

    /// L-BFGS con la energía de diferencias hacia delante y hacia atrás (o centrada, para ver el
    /// desenrollado); devuelve E/(12π²·B).
    pub fn skyrme_relax(&mut self, iterations: u32) -> f64 {
        if !self.sky.active {
            return f64::NAN;
        }
        let mut opt = match self.sky.lbfgs.take() {
            Some(o) => o,
            None => {
                let mut g = vec![0.0; self.sky.psi.len()];
                let f = self.sky.energy_grad(&self.sky.psi, Some(&mut g));
                Lbfgs::new(self.sky.psi.clone(), f, g)
            }
        };
        {
            let fun = self.skyrme_fun();
            for _ in 0..iterations {
                if !opt.iterate(&fun) && !opt.iterate(&fun) {
                    break;
                }
            }
        }
        self.sky.psi = opt.x.clone();
        self.sky.energy = opt.f;
        self.sky.lbfgs = Some(opt);
        self.sky.refresh_density();
        self.sky.energy / (12.0 * PI * PI * self.sky.baryons as f64)
    }

    /// Usa la derivada centrada (segundo orden) en lugar de la media de las dos laterales: con ella los
    /// dameros no cuestan energía y el solitón se desenrolla.
    pub fn skyrme_set_centered(&mut self, centered: bool) {
        self.sky.centered = centered;
        let mut g = vec![0.0; self.sky.psi.len()];
        let f = self.sky.energy_grad(&self.sky.psi, Some(&mut g));
        self.sky.energy = f;
        self.sky.lbfgs = Some(Lbfgs::new(self.sky.psi.clone(), f, g));
    }

    /// B = (1/2π²)∫det[φ, ∂₁φ, ∂₂φ, ∂₃φ] (derivadas centradas).
    pub fn skyrme_charge(&self) -> f64 {
        if !self.sky.active {
            return 0.0;
        }
        self.sky.charge()
    }

    /// E/12π² de la configuración actual (por barión).
    pub fn skyrme_energy(&self) -> f64 {
        self.sky.energy / (12.0 * PI * PI * self.sky.baryons as f64)
    }

    /// Segundos momentos de la densidad de energía respecto de su centro: [⟨x²⟩, ⟨y²⟩, ⟨z²⟩, densidad en
    /// el centro, densidad máxima].
    pub fn skyrme_moments(&self) -> Vec<f64> {
        let s = &self.sky;
        if !s.active {
            return vec![0.0; 5];
        }
        let n = s.n;
        let (mut w, mut m) = (0.0, [0.0f64; 3]);
        let mut dmax = 0.0f64;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let d = s.dens[(z * n + y) * n + x] as f64;
                    let c = [-s.half + x as f64 * s.h, -s.half + y as f64 * s.h, -s.half + z as f64 * s.h];
                    w += d;
                    for a in 0..3 {
                        m[a] += d * c[a] * c[a];
                    }
                    dmax = dmax.max(d);
                }
            }
        }
        let c = n / 2;
        vec![m[0] / w, m[1] / w, m[2] / w, s.dens[(c * n + c) * n + c] as f64, dmax]
    }

    /// Lado de la malla de Skyrme (puntos).
    pub fn skyrme_grid(&self) -> u32 {
        self.sky.n as u32
    }

    /// Perfil radial del erizo (R = 30, 6001 puntos) y la cuantización de Adkins, Nappi y Witten:
    /// [E/12π², Λ, e, F_π, masa clásica, radio isoescalar, C].
    pub fn skyrme_radial_quantization(&mut self) -> *const f64 {
        self.sky.quant = anw_quantization(30.0, 6001);
        self.sky.quant.as_ptr()
    }
}
