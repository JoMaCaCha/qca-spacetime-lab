//! Fermiones quirales en paredes de dominio y masa de Yukawa (§2.13): la caminata de Dirac 2D con
//! una masa m(r) en cada celda (M(m) = exp(−imσ_y) sigue siendo local y unitaria). Donde m cambia
//! de signo, la pared atrapa un único modo que se propaga en un solo sentido (Jackiw–Rebbi).
//! Con Yukawa, m(r) = y·φ(r) sale de un Higgs real φ con potencial λ/4·(φ² − v²)², c = 1 y
//! s = 2 subpasos; en el modo dinámico, el fermión empuja al Higgs con −y·ψ†σ_yψ.
//! Los espectros exactos de una tira (para cada k_y) se calculan aparte, sin simulación.

use crate::dirac2d::c;
use crate::linalg::{eig, Eigen, C};
use crate::util::min_image;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct DiracMass {
    pub active: bool,
    /// m(r) por celda.
    pub m: Vec<f32>,
    /// cos m(r) y sin m(r) por celda.
    pub cs: Vec<f32>,
    /// 0 = masa fija, 1 = m = y·φ con el Higgs estático, 2 = Higgs dinámico.
    pub yukawa: u32,
    pub y: f64,
    pub v: f64,
    pub mh: f64,
    pub phi: Vec<f64>,
    pub pi: Vec<f64>,
}

impl DiracMass {
    pub fn new() -> Self {
        DiracMass { active: false, m: Vec::new(), cs: Vec::new(), yukawa: 0, y: 0.3, v: 1.0, mh: 0.3, phi: Vec::new(), pi: Vec::new() }
    }
}

/// Matriz 2W×2W del paso U(k_y) = M(m(x))·T_y(k_y)·T_x en una tira periódica de anchura W
/// (índice 2x + componente).
fn strip_matrix(masses: &[f64], ky: f64) -> Vec<C> {
    let w = masses.len();
    let n = 2 * w;
    let mut u = vec![c(0.0, 0.0); n * n];
    let (cy, sy) = (ky.cos(), ky.sin());
    for x in 0..w {
        // T_x: ↑ en x viene de x − 1; ↓ en x viene de x + 1
        let xm = (x + w - 1) % w;
        let xp = (x + 1) % w;
        let (cm, sm) = (masses[x].cos(), masses[x].sin());
        // M·T_y = [[cm, −sm], [sm, cm]]·[[cy, −i·sy], [−i·sy, cy]]
        let mt = [[c(cm * cy, 0.0) + c(0.0, sm * sy), c(0.0, -cm * sy) - c(sm * cy, 0.0)], [c(sm * cy, 0.0) - c(0.0, cm * sy), c(0.0, -sm * sy) + c(cm * cy, 0.0)]];
        for r in 0..2 {
            // columna de ↑(x − 1) y de ↓(x + 1)
            u[(2 * x + r) * n + 2 * xm] += mt[r][0];
            u[(2 * x + r) * n + 2 * xp + 1] += mt[r][1];
        }
    }
    u
}

/// Espectro exacto de la tira: autovalores e^{−iω} y autovectores. En k_y = 0 las ramas quirales
/// de dos paredes opuestas se cruzan en ω = 0 y son degeneradas, así que cualquier mezcla de las
/// dos es autovector: se usa k_y = 10⁻⁴, que las separa (2·10⁻⁴·cos m) y da los modos localizados.
fn strip_eigen(masses: &[f64], ky: f64) -> Eigen {
    let n = 2 * masses.len();
    let ky = if ky.abs() < 1e-4 { 1e-4 } else { ky };
    eig(strip_matrix(masses, ky), n)
}

/// ω ∈ (−π, π] de un autovalor e^{−iω}.
fn omega_of(l: C) -> f64 {
    -l.arg()
}

/// Perfil |ψ(x)|² de un autovector de la tira.
fn mode_profile(v: &[C]) -> Vec<f64> {
    (0..v.len() / 2).map(|x| v[2 * x].norm_sqr() + v[2 * x + 1].norm_sqr()).collect()
}

/// Centroide circular (en x) y tasa de decaimiento (ajuste de ln|ψ|² a ambos lados) de un modo.
fn mode_stats(p: &[f64]) -> (f64, f64) {
    let w = p.len() as f64;
    let (mut sc, mut ss) = (0.0, 0.0);
    for (x, v) in p.iter().enumerate() {
        let th = 2.0 * PI * x as f64 / w;
        sc += v * th.cos();
        ss += v * th.sin();
    }
    let xc = (ss.atan2(sc) / (2.0 * PI) * w).rem_euclid(w);
    let peak = p.iter().enumerate().fold((0usize, 0.0f64), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0;
    // decaimiento: pendiente de ln|ψ|² entre 3 y 15 celdas del pico, a cada lado
    let wi = p.len();
    let mut slopes = Vec::new();
    for dir in [-1i32, 1] {
        let (mut ts, mut ls) = (Vec::new(), Vec::new());
        for d in 3..16usize.min(wi / 4) {
            let x = ((peak as i32 + dir * d as i32).rem_euclid(wi as i32)) as usize;
            if p[x] > 1e-28 {
                ts.push(d as f64);
                ls.push(p[x].ln());
            }
        }
        if ts.len() > 4 {
            let (_, b) = crate::util::fit_linear(&ts, &ls);
            slopes.push(-0.5 * b);
        }
    }
    let decay = if slopes.is_empty() { 0.0 } else { slopes.iter().sum::<f64>() / slopes.len() as f64 };
    (xc, decay)
}

/// Estados de la tira con |ω| < omega_max: [ω, centroide x, decaimiento, fracción a menos de
/// `near` celdas del centroide] por estado.
fn strip_modes(masses: &[f64], ky: f64, omega_max: f64, near: f64) -> Vec<f64> {
    let e = strip_eigen(masses, ky);
    let w = masses.len() as f64;
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..e.values.len()).filter(|&k| omega_of(e.values[k]).abs() < omega_max).collect();
    idx.sort_by(|&a, &b| omega_of(e.values[a]).partial_cmp(&omega_of(e.values[b])).unwrap());
    for k in idx {
        let p = mode_profile(&e.vectors[k]);
        let (xc, decay) = mode_stats(&p);
        let tot: f64 = p.iter().sum();
        let close: f64 = p.iter().enumerate().filter(|(x, _)| min_image(*x as f64 - xc, w).abs() <= near).map(|(_, v)| v).sum();
        out.extend_from_slice(&[omega_of(e.values[k]), xc, decay, close / tot.max(1e-300)]);
    }
    out
}

/// Perfil de masa sintético: m_in en la franja central de anchura `inner` y m_out fuera.
fn synthetic_profile(width: usize, m_out: f64, m_in: f64, inner: f64) -> Vec<f64> {
    let xc = width as f64 / 2.0;
    (0..width).map(|x| if (x as f64 + 0.5 - xc).abs() < inner / 2.0 { m_in } else { m_out }).collect()
}

impl QuantumSpacetimeUniverse {
    #[inline(always)]
    pub(crate) fn dirac_mass_active(&self) -> bool {
        self.dmass.active
    }

    #[inline(always)]
    pub(crate) fn dirac_cs(&self, i: usize) -> (f32, f32) {
        (self.dmass.cs[2 * i], self.dmass.cs[2 * i + 1])
    }

    /// Activa el campo de masa con la masa global en todas las celdas (si no lo estaba).
    fn dmass_ensure(&mut self) {
        if !self.dmass.active || self.dmass.m.len() != self.n {
            self.dmass.m = vec![self.mass; self.n];
            self.dmass.cs = vec![0.0; 2 * self.n];
            self.dmass.active = true;
        }
    }

    pub(crate) fn dmass_refresh(&mut self) {
        for i in 0..self.n {
            let (s, c) = self.dmass.m[i].sin_cos();
            self.dmass.cs[2 * i] = c;
            self.dmass.cs[2 * i + 1] = s;
        }
    }

    /// m(r) = y·φ(r) desde el Higgs de Yukawa.
    fn yukawa_to_mass(&mut self) {
        let y = self.dmass.y;
        for i in 0..self.n {
            self.dmass.m[i] = (y * self.dmass.phi[i]) as f32;
        }
        self.dmass_refresh();
    }

    /// Paso del Higgs real de Yukawa (s = 2, c = 1): π ← π + Δt·[∇²φ − λ(φ² − v²)φ − y·ψ†σ_yψ].
    pub(crate) fn yukawa_step(&mut self) {
        if self.dmass.yukawa != 2 {
            return;
        }
        let n = self.n;
        let (v, mh, y) = (self.dmass.v, self.dmass.mh, self.dmass.y);
        let lam = if v > 0.0 { mh * mh / (2.0 * v * v) } else { 0.0 };
        let damp = 1.0 - self.em.eta;
        let dt = 0.5;
        // ψ†σ_yψ = 2·Im(conj(ψ↑)·ψ↓) (escenario Dirac 2D, sin color)
        let src: Vec<f64> = (0..n)
            .map(|i| {
                let b = &self.buffer_read[4 * i..4 * i + 4];
                2.0 * (b[0] as f64 * b[3] as f64 - b[1] as f64 * b[2] as f64)
            })
            .collect();
        for _ in 0..2 {
            for i in 0..n {
                let f = self.dmass.phi[i];
                let mut lap = -4.0 * f;
                for a in 0..2 {
                    lap += self.dmass.phi[self.nb(i, a, 1)] + self.dmass.phi[self.nb(i, a, -1)];
                }
                self.dmass.pi[i] = (self.dmass.pi[i] + dt * (lap - lam * (f * f - v * v) * f - y * src[i])) * damp;
            }
            for i in 0..n {
                self.dmass.phi[i] += dt * self.dmass.pi[i];
            }
        }
        self.yukawa_to_mass();
    }

    /// Perfil de masa de la fila y (para el espectro exacto de la tira).
    fn mass_row(&self, row: usize) -> Vec<f64> {
        let y = row.min(self.h - 1);
        (0..self.w).map(|x| if self.dmass.active { self.dmass.m[y * self.w + x] as f64 } else { self.mass as f64 }).collect()
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.13: masa de Dirac m en el rectángulo [x, x + w) × [y, y + h) (m < 0 crea paredes).
    pub fn paint_dirac_mass(&mut self, x: i32, y: i32, w: u32, h: u32, m: f32) {
        self.dmass_ensure();
        for dy in 0..h as i32 {
            for dx in 0..w as i32 {
                let i = self.wrap_cell(x + dx, y + dy, 0);
                self.dmass.m[i] = m;
            }
        }
        self.dmass.yukawa = 0;
        self.dmass_refresh();
    }

    /// m = +m en x < W/2 y −m en el resto: paredes en x = 0 y x = W/2.
    pub fn set_domain_walls(&mut self, m: f32) {
        self.dmass_ensure();
        for i in 0..self.n {
            let (x, _, _) = self.coords(i);
            self.dmass.m[i] = if x < self.w / 2 { m } else { -m };
        }
        self.dmass.yukawa = 0;
        self.dmass_refresh();
    }

    /// Paredes cerradas centradas en la malla (m = −m₀ dentro y +m₀ fuera): 0 = disco de radio
    /// `size` con m₀·tanh((r − R)/width); 1 = cuadrado de semilado `size`; 2 = estrella
    /// r(φ) = size·(1 + 0,2·cos 3φ + 0,1·sin 5φ). 3 = canal de masa m₀ fuera y 0 dentro, de
    /// anchura `size` a lo largo de y (control no quiral).
    pub fn set_dirac_mass_shape(&mut self, kind: u32, m0: f32, size: f32, width: f32) {
        self.dmass_ensure();
        let (cx0, cy0) = (self.w as f64 / 2.0, self.h as f64 / 2.0);
        let (m0, r0) = (m0 as f64, size as f64);
        for i in 0..self.n {
            let (x, y, _) = self.coords(i);
            let dx = x as f64 + 0.5 - cx0;
            let dy = y as f64 + 0.5 - cy0;
            let m = match kind {
                0 => m0 * ((dx.hypot(dy) - r0) / (width as f64).max(1e-6)).tanh(),
                1 => {
                    if dx.abs() < r0 && dy.abs() < r0 {
                        -m0
                    } else {
                        m0
                    }
                }
                2 => {
                    let ph = dy.atan2(dx);
                    let rb = r0 * (1.0 + 0.2 * (3.0 * ph).cos() + 0.1 * (5.0 * ph).sin());
                    if dx.hypot(dy) < rb {
                        -m0
                    } else {
                        m0
                    }
                }
                _ => {
                    if dx.abs() < r0 / 2.0 {
                        0.0
                    } else {
                        m0
                    }
                }
            };
            self.dmass.m[i] = m as f32;
        }
        self.dmass.yukawa = 0;
        self.dmass_refresh();
    }

    /// Vuelve a la masa global uniforme.
    pub fn clear_dirac_mass(&mut self) {
        self.dmass.active = false;
        self.dmass.yukawa = 0;
    }

    pub fn dirac_mass_at(&self, x: i32, y: i32) -> f32 {
        if !self.dmass.active {
            return self.mass;
        }
        self.dmass.m[self.wrap_cell(x, y, 0)]
    }

    /// 0 = masa fija, 1 = m = y·φ con el Higgs real estático, 2 = dinámico (el fermión empuja al
    /// Higgs). λ = m_H²/(2v²), así que m_H² = 2λv². Al activarlo, φ = v uniforme.
    pub fn set_yukawa(&mut self, mode: u32, y: f32, v: f32, m_h: f32) {
        if self.scenario != 2 || self.dim != 2 {
            self.set_message("La masa de Yukawa con paredes es del escenario Dirac 2D (§2.13).");
            return;
        }
        self.dmass.y = y as f64;
        self.dmass.v = v as f64;
        self.dmass.mh = m_h as f64;
        if mode == 0 {
            self.dmass.yukawa = 0;
            return;
        }
        self.dmass_ensure();
        if self.dmass.phi.len() != self.n || self.dmass.yukawa == 0 {
            self.dmass.phi = vec![v as f64; self.n];
            self.dmass.pi = vec![0.0; self.n];
        }
        self.dmass.yukawa = mode.min(2);
        self.yukawa_to_mass();
    }

    /// Kink del Higgs en x1 y antikink en x2 (este con velocidad u, contraído por γ), con x1 < x2:
    /// φ = v·[tanh(m_H·(x − x1)/2) − tanh(γ·m_H·(x − x2)/2) − 1] (−v fuera del par y +v dentro,
    /// continuo en la costura) y π = −u·∂_x del antikink.
    pub fn seed_higgs_kinks(&mut self, x1: i32, x2: i32, velocity: f32) {
        if self.dmass.yukawa == 0 {
            self.set_message("Primero activa la masa de Yukawa (set_yukawa).");
            return;
        }
        let (v, mh) = (self.dmass.v, self.dmass.mh);
        let u = (velocity as f64).clamp(-0.99, 0.99);
        let gam = 1.0 / (1.0 - u * u).sqrt();
        let (x1, x2) = (x1.min(x2) as f64, x1.max(x2) as f64);
        for i in 0..self.n {
            let (x, _, _) = self.coords(i);
            let a1 = 0.5 * mh * (x as f64 - x1);
            let a2 = 0.5 * gam * mh * (x as f64 - x2);
            self.dmass.phi[i] = v * (a1.tanh() - a2.tanh() - 1.0);
            self.dmass.pi[i] = u * v * 0.5 * gam * mh / a2.cosh().powi(2);
        }
        self.yukawa_to_mass();
    }

    pub fn yukawa_phi_at(&self, x: i32, y: i32) -> f64 {
        if self.dmass.phi.len() != self.n {
            return 0.0;
        }
        self.dmass.phi[self.wrap_cell(x, y, 0)]
    }

    /// Paso por cero de φ (media sobre las filas) más cercano a x, interpolado: la posición del kink.
    pub fn yukawa_kink_position(&self, x: f32) -> f64 {
        if self.dmass.phi.len() != self.n {
            return 0.0;
        }
        let w = self.w;
        let col: Vec<f64> = (0..w).map(|xx| (0..self.h).map(|y| self.dmass.phi[y * w + xx]).sum::<f64>() / self.h as f64).collect();
        let mut best = (f64::INFINITY, x as f64);
        for xx in 0..w {
            let (a, b) = (col[xx], col[(xx + 1) % w]);
            if a == 0.0 || a * b < 0.0 {
                let z = xx as f64 + if a == b { 0.0 } else { a / (a - b) };
                let d = min_image(z - x as f64, w as f64).abs();
                if d < best.0 {
                    best = (d, z);
                }
            }
        }
        best.1
    }

    /// Anchura del kink cercano a x: ajuste de φ = ±v·tanh((x − x₀)/ℓ) en ±8 celdas; devuelve ℓ
    /// (la predicción es 2c/m_H).
    pub fn yukawa_kink_width(&self, x: f32) -> f64 {
        let x0 = self.yukawa_kink_position(x);
        let w = self.w;
        let v = self.dmass.v;
        let col: Vec<f64> = (0..w).map(|xx| (0..self.h).map(|y| self.dmass.phi[y * w + xx]).sum::<f64>() / self.h as f64).collect();
        // atanh(φ/v) es lineal en x − x₀ con pendiente ±1/ℓ
        let (mut ts, mut ls) = (Vec::new(), Vec::new());
        for d in -8i32..=8 {
            let xx = ((x0.floor() as i32 + d).rem_euclid(w as i32)) as usize;
            let r = (col[xx] / v).clamp(-0.98, 0.98);
            if r.abs() < 0.95 {
                ts.push(min_image(xx as f64 - x0, w as f64));
                ls.push(r.atanh());
            }
        }
        if ts.len() < 3 {
            return 0.0;
        }
        let (_, b) = crate::util::fit_linear(&ts, &ls);
        1.0 / b.abs()
    }

    /// Paquete (norma 1) en el modo quiral exacto de la pared más cercana a x: el autovector de la
    /// tira (perfil de masa de la fila y, con k_y) con |ω| mínimo y centroide más cercano a x,
    /// por una gaussiana de anchura σ en y.
    pub fn inject_wall_packet(&mut self, x: i32, y: f32, sigma: f32, ky: f32) {
        if self.scenario != 2 || self.dim != 2 || self.ncol > 1 {
            return;
        }
        let row = (y.round() as i32).rem_euclid(self.h as i32) as usize;
        let masses = self.mass_row(row);
        let e = strip_eigen(&masses, ky as f64);
        // borde del volumen a este k_y: cos ω = √((cos m·cos k_y)² + (sin m·sin k_y)²) en su máximo
        let mmax = masses.iter().map(|m| m.abs()).fold(0.0, f64::max).max(1e-3);
        let (cy, sy) = ((ky as f64).cos(), (ky as f64).sin());
        let gap = ((mmax.cos() * cy).powi(2) + (mmax.sin() * sy).powi(2)).sqrt().min(1.0).acos();
        let w = self.w as f64;
        let mut best = (f64::INFINITY, 0usize);
        for k in 0..e.values.len() {
            let om = omega_of(e.values[k]);
            if om.abs() >= 0.95 * gap {
                continue;
            }
            let (xc, _) = mode_stats(&mode_profile(&e.vectors[k]));
            let score = min_image(xc - x as f64, w).abs() + 50.0 * om.abs();
            if score < best.0 {
                best = (score, k);
            }
        }
        if !best.0.is_finite() {
            return;
        }
        let v = &e.vectors[best.1];
        let hh = self.h as f64;
        let sig = (sigma as f64).max(0.5);
        // con el campo U(1), k_y es el momento cinético: la fase lleva además q·A_y de cada columna
        // (en la fila del paquete), como en `packet_gauge_phase`
        let qay: Vec<f64> = (0..self.w).map(|px| if self.em.mode != 0 { self.packet_gauge_phase(px as f64, y as f64).1 } else { 0.0 }).collect();
        let mut norm = 0.0f64;
        let mut vals = vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64); self.n];
        for i in 0..self.n {
            let (px, py, _) = self.coords(i);
            let dy = min_image(py as f64 - y as f64, hh);
            let env = (-dy * dy / (2.0 * sig * sig)).exp();
            if env < 1e-12 {
                continue;
            }
            let ph = C::from_polar(env, (ky as f64 + qay[px]) * dy);
            let (u, d) = (v[2 * px] * ph, v[2 * px + 1] * ph);
            vals[i] = (u.re, u.im, d.re, d.im);
            norm += u.norm_sqr() + d.norm_sqr();
        }
        let s = 1.0 / norm.sqrt().max(1e-300);
        for i in 0..self.n {
            let (a, b, cc, d) = vals[i];
            self.buffer_read[4 * i] += (a * s) as f32;
            self.buffer_read[4 * i + 1] += (b * s) as f32;
            self.buffer_read[4 * i + 2] += (cc * s) as f32;
            self.buffer_read[4 * i + 3] += (d * s) as f32;
        }
    }

    /// Fracción de la norma a menos de `half_width` celdas (en x) de x.
    pub fn wall_fraction(&self, x: i32, half_width: u32) -> f64 {
        let w = self.w as f64;
        let (mut near, mut tot) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            let (px, _, _) = self.coords(i);
            let e = self.density(i) as f64;
            tot += e;
            if min_image(px as f64 - x as f64, w).abs() <= half_width as f64 {
                near += e;
            }
        }
        if tot > 0.0 {
            near / tot
        } else {
            0.0
        }
    }

    /// Distancia (en saltos) de cada celda a la pared más cercana (celda con una vecina de masa de
    /// signo opuesto); u32::MAX si no hay paredes.
    fn wall_distance(&self) -> Vec<u32> {
        let n = self.n;
        let mut dist = vec![u32::MAX; n];
        let mut queue = std::collections::VecDeque::new();
        if !self.dmass.active {
            return dist;
        }
        for i in 0..n {
            let m = self.dmass.m[i];
            let wall = (0..2).any(|a| m * self.dmass.m[self.nb(i, a, 1)] < 0.0 || m * self.dmass.m[self.nb(i, a, -1)] < 0.0);
            if wall {
                dist[i] = 0;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            for a in 0..2 {
                for dir in [-1, 1] {
                    let j = self.nb(i, a, dir);
                    if dist[j] == u32::MAX {
                        dist[j] = dist[i] + 1;
                        queue.push_back(j);
                    }
                }
            }
        }
        dist
    }

    /// Pone a cero la materia a más de `half_width` saltos de cualquier pared (estados de prueba en
    /// la pared, por ejemplo para su espectro por autocorrelación).
    pub fn mask_near_dirac_wall(&mut self, half_width: u32) {
        let dist = self.wall_distance();
        let stride = self.stride();
        for i in 0..self.n {
            if dist[i] > half_width {
                for k in 0..stride {
                    self.buffer_read[i * stride + k] = 0.0;
                }
            }
        }
    }

    /// Fracción de la norma a menos de `half_width` saltos de cualquier pared.
    pub fn near_wall_fraction(&self, half_width: u32) -> f64 {
        let dist = self.wall_distance();
        let (mut near, mut tot) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            let e = self.density(i) as f64;
            tot += e;
            if dist[i] <= half_width {
                near += e;
            }
        }
        if tot > 0.0 {
            near / tot
        } else {
            0.0
        }
    }

    /// Espectro exacto (ω ∈ (−π, π], ordenados) de una tira periódica de anchura `width` con masa
    /// m_out y una franja central de anchura `inner` con m_in (paredes: m_in = −m_out, inner = W/2).
    pub fn dirac_strip_spectrum(&self, width: u32, m_out: f32, m_in: f32, inner: f32, ky: f32) -> Vec<f64> {
        let masses = synthetic_profile(width.max(4) as usize, m_out as f64, m_in as f64, inner as f64);
        let e = strip_eigen(&masses, ky as f64);
        let mut w: Vec<f64> = e.values.iter().map(|l| omega_of(*l)).collect();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w
    }

    /// Estados de la tira sintética con |ω| < omega_max: [ω, centroide x, decaimiento, fracción a
    /// menos de `near` celdas] por estado.
    #[allow(clippy::too_many_arguments)]
    pub fn dirac_strip_modes(&self, width: u32, m_out: f32, m_in: f32, inner: f32, ky: f32, omega_max: f32, near: f32) -> Vec<f64> {
        let masses = synthetic_profile(width.max(4) as usize, m_out as f64, m_in as f64, inner as f64);
        strip_modes(&masses, ky as f64, omega_max as f64, near as f64)
    }

    /// Perfil |ψ(x)|² del estado de la tira sintética con ω más cercano a `omega`.
    pub fn dirac_strip_mode_profile(&self, width: u32, m_out: f32, m_in: f32, inner: f32, ky: f32, omega: f32) -> Vec<f64> {
        let masses = synthetic_profile(width.max(4) as usize, m_out as f64, m_in as f64, inner as f64);
        let e = strip_eigen(&masses, ky as f64);
        let k = (0..e.values.len()).min_by(|&a, &b| (omega_of(e.values[a]) - omega as f64).abs().partial_cmp(&(omega_of(e.values[b]) - omega as f64).abs()).unwrap()).unwrap_or(0);
        mode_profile(&e.vectors[k])
    }

    /// Estados con |ω| < omega_max del perfil de masa real de la fila `row` (paredes pintadas o Yukawa).
    pub fn dirac_row_modes(&self, row: u32, ky: f32, omega_max: f32, near: f32) -> Vec<f64> {
        let masses = self.mass_row(row as usize);
        strip_modes(&masses, ky as f64, omega_max as f64, near as f64)
    }
}
