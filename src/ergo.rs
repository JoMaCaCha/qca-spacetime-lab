//! Gravedad análoga en el superfluido (§2.28), segunda parte: la ergorregión y la división de un
//! vórtice multicuantizado en un disco con muros (BdG radial: perfil por Newton y tasa por
//! Crank–Nicolson usado como iteración de potencia), y la reconexión de anillos de vórtice en 3D
//! (líneas localizadas con precisión de subcelda y separadas en componentes conexas).
//!
//! Unidades del problema radial: longitudes en ξ = 1/√(2m\*ρ̄f′) y tiempos en 1/(ρ̄f′), con
//! φ = ψ/√ρ̄: i∂_τφ = −∇²φ + N(|φ|²)φ, N(s) = c₀·[f(ρ̄s) − f(ρ̄)], c₀ = (1 + ρ̄/E_max)²/ρ̄ (N′(1) = 1).

use crate::util::{clamp_finite, min_image, wrap_angle};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

type C = Complex<f64>;

/// Malla radial escalonada r_i = (i + ½)h con una pared (Dirichlet) en r = R = n·h.
struct Radial {
    n: usize,
    h: f64,
    rho: f64,
    emax: f64,
}

impl Radial {
    fn r(&self, i: usize) -> f64 {
        (i as f64 + 0.5) * self.h
    }

    fn nl(&self, s: f64) -> f64 {
        let c0 = (1.0 + self.rho / self.emax).powi(2) / self.rho;
        let f = |x: f64| x / (1.0 + x / self.emax);
        c0 * (f(self.rho * s) - f(self.rho))
    }

    fn nl_prime(&self, s: f64) -> f64 {
        let c0 = (1.0 + self.rho / self.emax).powi(2) / self.rho;
        c0 * self.rho / (1.0 + self.rho * s / self.emax).powi(2)
    }

    /// Coeficientes (abajo, diagonal, arriba) de −Δ_n en volúmenes finitos (con la pared).
    fn lap(&self, i: usize, n_ang: f64) -> (f64, f64, f64) {
        let h2 = self.h * self.h;
        let r = self.r(i);
        let rp = (i as f64 + 1.0) * self.h;
        let rm = i as f64 * self.h;
        let lo = if i > 0 { -rm / (r * h2) } else { 0.0 };
        let mut up = -rp / (r * h2);
        let mut di = (rp + rm) / (r * h2) + n_ang * n_ang / (r * r);
        if i + 1 == self.n {
            // fantasma F_n = −F_{n−1}: la amplitud se anula en la pared
            di -= up;
            up = 0.0;
        }
        (lo, di, up)
    }

    /// Perfil estacionario F(r) del vórtice de carga κ por Newton (Jacobiana tridiagonal).
    fn profile(&self, kappa: f64) -> Vec<f64> {
        let n = self.n;
        let rw = n as f64 * self.h;
        let mut f: Vec<f64> = (0..n)
            .map(|i| {
                let r = self.r(i);
                let core = if kappa > 0.0 { (r / (r * r + 2.0).sqrt()).powf(kappa) } else { 1.0 };
                core * ((rw - r) / 2f64.sqrt()).tanh()
            })
            .collect();
        let (mut a, mut b, mut c, mut d, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for _ in 0..60 {
            let mut res = 0.0f64;
            for i in 0..n {
                let (lo, di, up) = self.lap(i, kappa);
                let s = f[i] * f[i];
                let fm = if i > 0 { f[i - 1] } else { 0.0 };
                let fp = if i + 1 < n { f[i + 1] } else { 0.0 };
                let r = lo * fm + di * f[i] + up * fp + self.nl(s) * f[i];
                res = res.max(r.abs());
                a[i] = lo;
                b[i] = di + self.nl(s) + 2.0 * s * self.nl_prime(s);
                c[i] = up;
                d[i] = -r;
            }
            // Thomas
            for i in 1..n {
                if b[i - 1].abs() < 1e-30 {
                    return f;
                }
                let m = a[i] / b[i - 1];
                b[i] -= m * c[i - 1];
                d[i] -= m * d[i - 1];
            }
            if b[n - 1].abs() < 1e-30 {
                return f;
            }
            x[n - 1] = d[n - 1] / b[n - 1];
            for i in (0..n - 1).rev() {
                if b[i].abs() < 1e-30 {
                    return f;
                }
                x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
            }
            for i in 0..n {
                f[i] += x[i];
            }
            if res < 1e-12 {
                break;
            }
        }
        f
    }
}

/// Radio de la ergorregión (en ξ): el r más externo con v_θ ≥ c_s local, con v_θ/c̄ = √2·κ/r y
/// c(r)/c̄ = F·(1 + ρ̄/E)/(1 + ρ̄F²/E).
fn ergo_radius(rad: &Radial, f: &[f64], kappa: f64) -> f64 {
    let ratio = |i: usize| -> f64 {
        let r = rad.r(i);
        let c = f[i] * (1.0 + rad.rho / rad.emax) / (1.0 + rad.rho * f[i] * f[i] / rad.emax);
        2f64.sqrt() * kappa / r - c
    };
    for i in (1..rad.n).rev() {
        let (a, b) = (ratio(i - 1), ratio(i));
        if a >= 0.0 && b < 0.0 {
            let t = a / (a - b);
            return rad.r(i - 1) + t * rad.h;
        }
    }
    f64::NAN
}

/// Tasa Im ω (en ρ̄f′) del modo ℓ por Crank–Nicolson como iteración de potencia: (u, v) con momentos
/// angulares κ + ℓ y κ − ℓ, i∂_τ(u, v) = [[H₊, P], [−P, −H₋]](u, v), H± = −Δ_{κ±ℓ} + N + F²N′,
/// P = F²N′. Devuelve la pendiente de ln‖(u, v)‖ en la segunda mitad (corregida con la
/// transformada de Cayley del cociente de Rayleigh cuando el modo domina).
fn bdg_rate(rad: &Radial, f: &[f64], kappa: f64, ell: f64, dt: f64, steps: usize) -> f64 {
    let n = rad.n;
    let dim = 2 * n;
    // matriz L en bandas (orden u_0, v_0, u_1, v_1, …): ancho de banda 2 a cada lado
    let mut band = vec![[C::new(0.0, 0.0); 5]; dim]; // columnas j − i = −2..2
    for i in 0..n {
        let s = f[i] * f[i];
        let pot = rad.nl(s) + s * rad.nl_prime(s);
        let p = s * rad.nl_prime(s);
        let (lo_u, di_u, up_u) = rad.lap(i, kappa + ell);
        let (lo_v, di_v, up_v) = rad.lap(i, kappa - ell);
        let (ru, rv) = (2 * i, 2 * i + 1);
        band[ru][2] = C::new(di_u + pot, 0.0);
        band[ru][3] = C::new(p, 0.0); // (u_i, v_i)
        if i > 0 {
            band[ru][0] = C::new(lo_u, 0.0);
        }
        if i + 1 < n {
            band[ru][4] = C::new(up_u, 0.0);
        }
        band[rv][2] = C::new(-(di_v + pot), 0.0);
        band[rv][1] = C::new(-p, 0.0); // (v_i, u_i)
        if i > 0 {
            band[rv][0] = C::new(-lo_v, 0.0);
        }
        if i + 1 < n {
            band[rv][4] = C::new(-up_v, 0.0);
        }
    }
    let half = C::new(0.0, 0.5 * dt);
    // A = I + i·dt/2·L (a factorizar), B = I − i·dt/2·L
    let mut amat = vec![[C::new(0.0, 0.0); 5]; dim];
    let mut bmat = vec![[C::new(0.0, 0.0); 5]; dim];
    for r in 0..dim {
        for k in 0..5 {
            amat[r][k] = half * band[r][k];
            bmat[r][k] = -half * band[r][k];
        }
        amat[r][2] += C::new(1.0, 0.0);
        bmat[r][2] += C::new(1.0, 0.0);
    }
    // LU en bandas sin pivoteo (A es diagonalmente dominante para dt pequeño)
    for r in 0..dim {
        let piv = amat[r][2];
        if !piv.re.is_finite() || !piv.im.is_finite() || piv.norm_sqr() < 1e-30 {
            return 0.0;
        }
        for dr in 1..=2 {
            let rr = r + dr;
            if rr >= dim {
                break;
            }
            let k = 2 - dr; // columna r en la fila rr
            let m = amat[rr][k] / piv;
            amat[rr][k] = m;
            for c in 1..=2 {
                let col = r + c;
                if col >= dim {
                    break;
                }
                let kk = col + 2 - rr;
                amat[rr][kk] = amat[rr][kk] - m * amat[r][2 + c];
            }
        }
    }
    let mut psi: Vec<C> = (0..dim)
        .map(|i| {
            let x = (i as f64 * 0.7548776662).fract();
            C::new(x - 0.5, ((i as f64 * 0.5698402910).fract()) - 0.5)
        })
        .collect();
    let mut rhs = vec![C::new(0.0, 0.0); dim];
    let mut logn = Vec::with_capacity(steps / 10 + 1);
    let mut acc = 0.0f64;
    let mut last_ratio = C::new(1.0, 0.0);
    for s in 1..=steps {
        for r in 0..dim {
            let mut v = C::new(0.0, 0.0);
            for k in 0..5 {
                let col = r as isize + k as isize - 2;
                if col >= 0 && (col as usize) < dim {
                    v += bmat[r][k] * psi[col as usize];
                }
            }
            rhs[r] = v;
        }
        // sustitución hacia delante y hacia atrás
        for r in 0..dim {
            let mut v = rhs[r];
            for dr in 1..=2 {
                if r >= dr {
                    v -= amat[r][2 - dr] * rhs[r - dr];
                }
            }
            rhs[r] = v;
        }
        for r in (0..dim).rev() {
            let mut v = rhs[r];
            for c in 1..=2 {
                if r + c < dim {
                    v -= amat[r][2 + c] * rhs[r + c];
                }
            }
            let piv = amat[r][2];
            if piv.norm_sqr() < 1e-30 {
                return 0.0;
            }
            rhs[r] = v / piv;
        }
        if s == steps {
            let num: C = psi.iter().zip(&rhs).map(|(a, b)| a.conj() * b).sum();
            let den: f64 = psi.iter().map(|a| a.norm_sqr()).sum();
            if den > 1e-30 && den.is_finite() {
                last_ratio = num / den;
            }
        }
        std::mem::swap(&mut psi, &mut rhs);
        if s % 10 == 0 {
            let nn: f64 = psi.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
            if !nn.is_finite() || nn <= 1e-30 {
                return 0.0;
            }
            acc += nn.ln();
            logn.push((s as f64 * dt, acc));
            for a in psi.iter_mut() {
                *a /= nn;
            }
        }
    }
    if logn.len() < 4 {
        return 0.0;
    }
    let tail = &logn[logn.len() / 2..];
    let xs: Vec<f64> = tail.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = tail.iter().map(|p| p.1).collect();
    let (_, slope) = crate::util::fit_linear(&xs, &ys);
    if slope > 2e-3 {
        // ω = (2i/dt)·(λ − 1)/(λ + 1) con λ el cociente de Rayleigh del último paso
        let one = C::new(1.0, 0.0);
        let denom = last_ratio + one;
        if denom.norm_sqr() > 1e-30 {
            let w = C::new(0.0, 2.0 / dt) * (last_ratio - one) / denom;
            if w.im.is_finite() && (w.im - slope).abs() < 0.2 * slope {
                return w.im;
            }
        }
    }
    slope.max(0.0)
}

/// [r_e/ξ, Im ω/(ρ̄f′)] del vórtice de carga κ en un disco de R/ξ (ρ̄ = 0,5, E_max).
pub(crate) fn radial_bdg(charge: u32, rd_over_xi: f64, ell: u32, emax: f64) -> (f64, f64) {
    let h = 0.025;
    let n = ((rd_over_xi / h).round() as usize).max(40);
    let rad = Radial { n, h: rd_over_xi / n as f64, rho: 0.5, emax };
    let kappa = charge as f64;
    let f = rad.profile(kappa);
    let re = ergo_radius(&rad, &f, kappa);
    let rate = bdg_rate(&rad, &f, kappa, ell as f64, 0.25, 24000);
    (re, rate)
}

/// Perfil radial libre (pared lejana a 40ξ) del vórtice de carga κ, en una tabla r/ξ → F.
pub(crate) fn free_vortex_profile(charge: u32, emax: f64) -> (f64, Vec<f64>) {
    let h = 0.05;
    let rad = Radial { n: 800, h, rho: 0.5, emax };
    let f = rad.profile(charge as f64);
    // normaliza a 1 lejos del núcleo (antes de la capa de la pared)
    let fref = f[(20.0 / h) as usize].max(1e-12);
    (h, f.iter().map(|v| v / fref).collect())
}

/// Perfil de la capa de la pared (κ = 0) en un disco de R/ξ.
fn wall_profile(rd_over_xi: f64, emax: f64) -> (f64, Vec<f64>) {
    let h = 0.025;
    let n = ((rd_over_xi / h).round() as usize).max(40);
    let rad = Radial { n, h: rd_over_xi / n as f64, rho: 0.5, emax };
    (rad.h, rad.profile(0.0))
}

fn table_at(h: f64, tab: &[f64], r: f64) -> f64 {
    if tab.is_empty() || !r.is_finite() || !h.is_finite() || h <= 0.0 {
        return 1.0;
    }
    let x = r / h - 0.5;
    if x <= 0.0 {
        return tab[0] * (r / (0.5 * h)).min(1.0);
    }
    let i = x.floor() as usize;
    if i + 1 >= tab.len() {
        return *tab.last().unwrap();
    }
    let t = x - i as f64;
    tab[i] * (1.0 - t) + tab[i + 1] * t
}

impl QuantumSpacetimeUniverse {
    /// ξ con la densidad del fondo ρ̄ (no la media de la malla, que incluye muros y núcleos).
    pub(crate) fn analog_xi(&self) -> f64 {
        let rho = self.vort.rho_bg.max(1e-6) as f64;
        let denom = 2.0 * self.band_mass() * rho * self.nonlinear_slope(rho);
        if !denom.is_finite() || denom <= 1e-18 {
            return 1.0;
        }
        1.0 / denom.sqrt()
    }

    /// Multiplica la fase del campo por el perfil F_|q|(r/ξ) del vórtice libre y e^{iqθ} (θ con las
    /// imágenes periódicas): la versión del escenario 10 de `imprint_vortex`, para |q| ≤ 4.
    pub(crate) fn imprint_vortex_radial(&mut self, x: f64, y: f64, q: i32) {
        let xi = self.analog_xi();
        let (h, tab) = free_vortex_profile(q.unsigned_abs(), self.e_max as f64);
        let (w, hh) = (self.w as f64, self.h as f64);
        let stride = self.stride();
        let nc = self.nc;
        for i in 0..self.n {
            let (cx, cy, _) = self.coords(i);
            let (px, py) = (cx as f64, cy as f64);
            let mut th = 0.0f64;
            for jy in -4..=4 {
                for ix in -4..=4 {
                    th += (py - y - jy as f64 * hh).atan2(px - x - ix as f64 * w);
                }
            }
            let dx = min_image(px - x, w);
            let dy = min_image(py - y, hh);
            let r = (dx * dx + dy * dy).sqrt();
            let prof = table_at(h, &tab, r / xi);
            let (c, s) = ((q as f64 * th).cos() * prof, (q as f64 * th).sin() * prof);
            for k in 0..nc {
                let re = self.buffer_read[i * stride + 2 * k] as f64;
                let im = self.buffer_read[i * stride + 2 * k + 1] as f64;
                self.buffer_read[i * stride + 2 * k] = (re * c - im * s) as f32;
                self.buffer_read[i * stride + 2 * k + 1] = (re * s + im * c) as f32;
            }
        }
    }

    /// Fase arg Σψ y Σψ de la celda (f64).
    fn sum_psi64(&self, i: usize) -> C {
        let (a, b) = self.sum_psi_at(i);
        C::new(a as f64, b as f64)
    }
}

/// Cero de la interpolación bilineal de f en el cuadrado unidad (f00, f10, f01, f11).
fn bilinear_zero(f00: C, f10: C, f01: C, f11: C) -> (f64, f64) {
    let a = f00;
    let b = f10 - f00;
    let c = f01 - f00;
    let d = f11 - f10 - f01 + f00;
    // Im[(A + Ct)·conj(B + Dt)] = 0: cuadrática en t
    let q2 = (c * d.conj()).im;
    let q1 = (a * d.conj() + c * b.conj()).im;
    let q0 = (a * b.conj()).im;
    let mut roots = Vec::new();
    if q2.abs() < 1e-14 * (q1.abs() + q0.abs()).max(1e-300) {
        if q1.abs() > 1e-300 {
            roots.push(-q0 / q1);
        }
    } else {
        let disc = q1 * q1 - 4.0 * q2 * q0;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            roots.push((-q1 + sq) / (2.0 * q2));
            roots.push((-q1 - sq) / (2.0 * q2));
        }
    }
    let mut best = (0.5, 0.5);
    let mut bd = f64::INFINITY;
    for t in roots {
        let den = b + d * t;
        if den.norm_sqr() < 1e-300 {
            continue;
        }
        let s = -((a + c * t) * den.conj()).re / den.norm_sqr();
        let dist = (s - 0.5).abs().max((t - 0.5).abs());
        if dist < bd {
            bd = dist;
            best = (s.clamp(0.0, 1.0), t.clamp(0.0, 1.0));
        }
    }
    best
}

fn uf_find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.28: muro (§2.4) en las celdas con r ≥ radius alrededor de (cx, cy) (2D, o un cilindro en
    /// 3D). Además multiplica el campo por el perfil de la capa de curación junto a la pared (κ = 0,
    /// por Newton radial), para que el fondo empiece estacionario con ρ̄ en el interior.
    pub fn add_disk_wall(&mut self, cx: f32, cy: f32, radius: f32) {
        if !(cx.is_finite() && cy.is_finite() && radius.is_finite()) || radius <= 1.0 {
            return;
        }
        let max_rad = (self.w.max(self.h) as f64 * 2.0).max(2.0);
        let (cx, cy, rad) = (cx as f64, cy as f64, (radius as f64).min(max_rad));
        let xi = self.analog_xi().max(1e-3);
        let (h, tab) = wall_profile((rad / xi).clamp(1.0, 200.0), self.e_max as f64);
        let stride = self.stride();
        for i in 0..self.n {
            let (x, y, _) = self.coords(i);
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            let r = (dx * dx + dy * dy).sqrt();
            if r >= rad {
                self.wall[i] = 1;
                for k in 0..stride {
                    self.buffer_read[i * stride + k] = 0.0;
                }
            } else {
                let p = table_at(h, &tab, r / xi) as f32;
                for k in 0..stride {
                    self.buffer_read[i * stride + k] *= p;
                }
            }
        }
        self.has_walls = true;
        self.an.disk = Some((cx, cy, rad));
    }

    /// |Σ (E_ref − E)·e^{−r²/radius²}·e^{ilθ}| alrededor del centro del disco, con E_ref el máximo
    /// de E dentro del disco.
    pub fn vortex_multipole(&self, l: u32, radius: f32) -> f64 {
        let (cx, cy, rd) = self.an.disk.unwrap_or((self.w as f64 / 2.0 - 0.5, self.h as f64 / 2.0 - 0.5, f64::INFINITY));
        let rad = clamp_finite(radius as f64, 1e-4, 1e6, 1.0);
        let r2s = (rad * rad).max(1e-9);
        let mut eref = 0.0f64;
        for i in 0..self.n {
            if self.has_walls && self.wall[i] != 0 {
                continue;
            }
            eref = eref.max(self.density(i) as f64);
        }
        let mut s = C::new(0.0, 0.0);
        for i in 0..self.n {
            if self.has_walls && self.wall[i] != 0 {
                continue;
            }
            let (x, y, _) = self.coords(i);
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            let r2 = dx * dx + dy * dy;
            if r2 >= rd * rd {
                continue;
            }
            let th = dy.atan2(dx);
            let wgt = (eref - self.density(i) as f64) * (-r2 / r2s).exp();
            s += C::new((l as f64 * th).cos(), (l as f64 * th).sin()) * wgt;
        }
        s.norm()
    }

    /// Semilla multipolar: ψ ← ψ·(1 + amplitude·cos(lθ)·e^{−r²/(2ξ)²}) alrededor del centro del disco.
    pub fn seed_disk_multipole(&mut self, l: u32, amplitude: f32) {
        let amp = clamp_finite(amplitude as f64, -10.0, 10.0, 0.0);
        let (cx, cy, _) = self.an.disk.unwrap_or((self.w as f64 / 2.0 - 0.5, self.h as f64 / 2.0 - 0.5, f64::INFINITY));
        let xi = self.analog_xi();
        let stride = self.stride();
        for i in 0..self.n {
            let (x, y, _) = self.coords(i);
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            let r2 = dx * dx + dy * dy;
            let th = dy.atan2(dx);
            let f = (1.0 + amp * (l as f64 * th).cos() * (-r2 / (4.0 * xi * xi)).exp()) as f32;
            for k in 0..stride {
                self.buffer_read[i * stride + k] *= f;
            }
        }
    }

    /// §2.28: [r_e/ξ, Im ω/(ρ̄f′)] con ρ̄ = 0,5: perfil por Newton radial y Crank–Nicolson como
    /// iteración de potencia (paso 0,25/(ρ̄f′)).
    pub fn vortex_radial_bdg(&mut self, charge: u32, rd_over_xi: f32, l: u32) -> *const f64 {
        let rd = clamp_finite(rd_over_xi as f64, 2.0, 40.0, 10.0);
        let (re, rate) = radial_bdg(charge.clamp(1, 4), rd, l.clamp(1, 64), self.e_max as f64);
        self.an.out = vec![re, rate];
        self.an.out.as_ptr()
    }

    /// §2.28, 3D: anillo de radio R en el plano z = z₀ centrado en (x₀, y₀); fase
    /// sign·[atan2(z − z₀, ρ − R) − atan2(z − z₀, ρ + R)] y núcleo d/√(d² + ξ²), con d la distancia al
    /// núcleo. Multiplica el campo actual. Las coordenadas son las de la malla, sin imagen mínima: la
    /// costura de fase de cada anillo queda en el borde de la caja (z = 0) y, con dos anillos de
    /// circulaciones opuestas, las dos costuras casi se cancelan. Con la imagen mínima la costura de un
    /// anillo (un salto de ~1,6 rad) quedaría junto al otro y nuclearía lazos espurios.
    pub fn imprint_vortex_ring(&mut self, x: f32, y: f32, z: f32, radius: f32, sign: i32) {
        if self.dim != 3 || !(x.is_finite() && y.is_finite() && z.is_finite() && radius.is_finite()) || radius <= 0.0 {
            self.set_message("Los anillos de vórtice necesitan una malla 3D.");
            return;
        }
        let xi = self.analog_xi();
        let sg = if sign < 0 { -1.0 } else { 1.0 };
        let rr = radius as f64;
        let stride = self.stride();
        let nc = self.nc;
        for i in 0..self.n {
            let (cx, cy, cz) = self.coords(i);
            let dx = cx as f64 - x as f64;
            let dy = cy as f64 - y as f64;
            let dz = cz as f64 - z as f64;
            let rho = (dx * dx + dy * dy).sqrt();
            let ph = sg * (dz.atan2(rho - rr) - dz.atan2(rho + rr));
            let dd = ((rho - rr).powi(2) + dz * dz).sqrt();
            let amp = dd / (dd * dd + xi * xi).sqrt();
            let (c, s) = (ph.cos() * amp, ph.sin() * amp);
            for k in 0..nc {
                let re = self.buffer_read[i * stride + 2 * k] as f64;
                let im = self.buffer_read[i * stride + 2 * k + 1] as f64;
                self.buffer_read[i * stride + 2 * k] = (re * c - im * s) as f32;
                self.buffer_read[i * stride + 2 * k + 1] = (re * s + im * c) as f32;
            }
        }
    }

    /// Distancia mínima entre las dos componentes mayores de las líneas de vórtice (subcelda): en
    /// cada plaqueta con winding, el cero de la interpolación bilineal de Σ_k ψ_k; las plaquetas de un
    /// mismo cubo se unen, y también los cubos vecinos por una cara que contienen línea (a la
    /// resolución de la red son la misma componente). 0 si solo hay una componente, NaN si no hay líneas. Guarda
    /// [componentes, puntos, puntos fuera de las dos mayores, distancia] en `vortex_line_info`.
    pub fn vortex_line_distance(&mut self) -> f64 {
        if self.dim != 3 {
            return f64::NAN;
        }
        let (w, h, d) = (self.w, self.h, self.d);
        let n = self.n;
        let sum: Vec<C> = (0..n).map(|i| self.sum_psi64(i)).collect();
        let phase: Vec<f64> = sum.iter().map(|s| s.im.atan2(s.re)).collect();
        let at = |x: usize, y: usize, z: usize| (z * h + y) * w + x;
        // plaquetas: id = 3·celda + orientación (0: xy, 1: yz, 2: zx)
        let mut pts: Vec<[f64; 3]> = Vec::new();
        let mut cube_first = vec![u32::MAX; n];
        let mut parent: Vec<u32> = Vec::new();
        for z in 0..d {
            let zp = (z + 1) % d;
            for y in 0..h {
                let yp = (y + 1) % h;
                for x in 0..w {
                    let xp = (x + 1) % w;
                    for o in 0..3 {
                        // esquinas (00, 10, 11, 01) en el plano de la orientación
                        let (c00, c10, c11, c01) = match o {
                            0 => (at(x, y, z), at(xp, y, z), at(xp, yp, z), at(x, yp, z)),
                            1 => (at(x, y, z), at(x, yp, z), at(x, yp, zp), at(x, y, zp)),
                            _ => (at(x, y, z), at(x, y, zp), at(xp, y, zp), at(xp, y, z)),
                        };
                        let s = wrap_angle(phase[c10] - phase[c00]) + wrap_angle(phase[c11] - phase[c10]) + wrap_angle(phase[c01] - phase[c11]) + wrap_angle(phase[c00] - phase[c01]);
                        if (s / (2.0 * PI)).round() == 0.0 {
                            continue;
                        }
                        let (sx, sy) = bilinear_zero(sum[c00], sum[c10], sum[c01], sum[c11]);
                        let (xf, yf, zf) = (x as f64, y as f64, z as f64);
                        let p = match o {
                            0 => [xf + sx, yf + sy, zf],
                            1 => [xf, yf + sx, zf + sy],
                            _ => [xf + sy, yf, zf + sx],
                        };
                        let id = pts.len() as u32;
                        pts.push(p);
                        parent.push(id);
                        // los dos cubos que comparten la plaqueta
                        let other = match o {
                            0 => at(x, y, (z + d - 1) % d),
                            1 => at((x + w - 1) % w, y, z),
                            _ => at(x, (y + h - 1) % h, z),
                        };
                        for cube in [at(x, y, z), other] {
                            if cube_first[cube] == u32::MAX {
                                cube_first[cube] = id;
                            } else {
                                let a = uf_find(&mut parent, cube_first[cube]);
                                let b = uf_find(&mut parent, id);
                                if a != b {
                                    parent[b as usize] = a;
                                }
                            }
                        }
                    }
                }
            }
        }
        // a la resolución de la red, dos líneas en cubos vecinos (por una cara) son la misma
        // componente: así se ve el instante de la reconexión aunque caiga entre dos pasos
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let c = at(x, y, z);
                    if cube_first[c] == u32::MAX {
                        continue;
                    }
                    for nb in [at((x + 1) % w, y, z), at(x, (y + 1) % h, z), at(x, y, (z + 1) % d)] {
                        if cube_first[nb] != u32::MAX {
                            let a = uf_find(&mut parent, cube_first[c]);
                            let b = uf_find(&mut parent, cube_first[nb]);
                            if a != b {
                                parent[b as usize] = a;
                            }
                        }
                    }
                }
            }
        }
        let np = pts.len();
        if np == 0 {
            self.an.line_info = vec![0.0, 0.0, 0.0, f64::NAN];
            self.an.line_pts.clear();
            return f64::NAN;
        }
        let roots: Vec<u32> = (0..np as u32).map(|i| uf_find(&mut parent, i)).collect();
        let mut sizes = std::collections::HashMap::new();
        for &r in &roots {
            *sizes.entry(r).or_insert(0usize) += 1;
        }
        let mut comps: Vec<(u32, usize)> = sizes.into_iter().collect();
        comps.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let rank = |r: u32| comps.iter().position(|c| c.0 == r).unwrap_or(usize::MAX);
        self.an.line_pts.clear();
        for (i, p) in pts.iter().enumerate() {
            let k = rank(roots[i]);
            self.an.line_pts.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32, k.min(255) as f32]);
        }
        let outside: usize = comps.iter().skip(2).map(|c| c.1).sum();
        let dist = if comps.len() < 2 {
            0.0
        } else {
            let (ra, rb) = (comps[0].0, comps[1].0);
            let a: Vec<&[f64; 3]> = pts.iter().zip(&roots).filter(|(_, &r)| r == ra).map(|(p, _)| p).collect();
            let b: Vec<&[f64; 3]> = pts.iter().zip(&roots).filter(|(_, &r)| r == rb).map(|(p, _)| p).collect();
            let mut best = f64::INFINITY;
            for p in &a {
                for q in &b {
                    let dx = min_image(p[0] - q[0], w as f64);
                    let dy = min_image(p[1] - q[1], h as f64);
                    let dz = min_image(p[2] - q[2], d as f64);
                    best = best.min(dx * dx + dy * dy + dz * dz);
                }
            }
            best.sqrt()
        };
        self.an.line_info = vec![comps.len() as f64, np as f64, outside as f64, dist];
        dist
    }

    /// [componentes, puntos, puntos fuera de las dos mayores, distancia] del último análisis.
    pub fn vortex_line_info(&self) -> Vec<f64> {
        self.an.line_info.clone()
    }

    /// Puntos de las líneas del último análisis: [x, y, z, rango de su componente] (f32).
    pub fn vortex_line_points(&self) -> Vec<f32> {
        self.an.line_pts.clone()
    }
}
