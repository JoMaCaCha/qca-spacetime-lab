//! Motor 3D sin dobladores (§2.25, escenario 7): Trotter simétrico (Strang) del hamiltoniano de
//! Wilson–Dirac H = Σ_x (m + 3r)·ψ†βψ + Σ_{x,j} [ψ†_{x+j}·M_j·e^{iqA_j(x)}·ψ_x + h.c.], con
//! M_j = (iα_j − r·β)/2, que en momentos da H(k) = Σ_j α_j sen k_j + β·(m + r·Σ_j(1 − cos k_j)).
//! - Cada término de enlace actúa sobre dos sitios; los enlaces de una dirección se separan en pares e
//!   impares (ladrillo), así que exp(−iτh) de una capa es un producto de compuertas de 8×8 exactas.
//! - Un paso: las seis capas (x par, x impar, y par, y impar, z par, z impar) con τ = dt/2, la
//!   compuerta local exp(−i·dt·(m + 3r)·β) y las seis capas en orden inverso. Error O(dt³) por paso.
//! - Acoplamiento U(1): g(θ) = D·g₀·D†, D = diag(1, e^{iθ}). La carga que mueve cada compuerta por su
//!   enlace es el cambio de |ψ_{x+j}|², así que la continuidad es exacta y Gauss también con E ← E − q·j;
//!   el campo sigue a Maxwell con c = 1 (E += dt·rot B, A −= dt·E).
//! Los dobladores reciben la masa de Wilson m + 2r·(número de componentes π) y, con dt·E_máx < π, las
//! cuasi-energías no llegan a ±π: no hay pseudo-dobladores. El precio: avanza dt celdas por paso, no una.
//! El término de Wilson mezcla L y R: sirve para teorías vectoriales, no para el W quiral.

use crate::linalg::{eig, expm_c64};
use crate::rng::SplitMix64;
use crate::spinor::{cx, dirac_matrix, m4_zero, M4};
use crate::sun::C;
use crate::util::{clamp_finite, min_image};
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub type G8 = [[C; 8]; 8];

/// exp(A) de una matriz 8×8 delegada a `crate::linalg::expm_c64`.
fn expm8(a: &G8) -> G8 {
    let mut flat = [cx(0.0, 0.0); 64];
    for i in 0..8 {
        for j in 0..8 {
            flat[i * 8 + j] = a[i][j];
        }
    }
    let out = expm_c64(&flat, 8);
    let mut res = [[cx(0.0, 0.0); 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            res[i][j] = out[i * 8 + j];
        }
    }
    res
}

/// Compuertas del motor: g₀ = exp(−iτ·h_j) por dirección (τ = dt/2) y el paso local.
#[derive(Clone)]
pub struct TwdGates {
    pub m: f64,
    pub r: f64,
    pub dt: f64,
    pub g0: [G8; 3],
}

impl TwdGates {
    pub fn new(m: f64, r: f64, dt: f64) -> Self {
        let beta = dirac_matrix(2, 3);
        let mut g0 = [[[cx(0.0, 0.0); 8]; 8]; 3];
        for (j, g) in g0.iter_mut().enumerate() {
            let alpha = dirac_matrix(0, j);
            // M_j = (iα_j − r·β)/2; h = [[0, M†], [M, 0]] en la base (ψ_x, ψ_{x+j})
            let mut mm: M4 = m4_zero();
            for a in 0..4 {
                for b in 0..4 {
                    mm[a][b] = (alpha[a][b] * cx(0.0, 1.0) - beta[a][b] * r) * 0.5;
                }
            }
            let mut a8 = [[cx(0.0, 0.0); 8]; 8];
            for a in 0..4 {
                for b in 0..4 {
                    // −iτ·h
                    a8[4 + a][b] = mm[a][b] * cx(0.0, -0.5 * dt);
                    a8[a][4 + b] = mm[b][a].conj() * cx(0.0, -0.5 * dt);
                }
            }
            *g = expm8(&a8);
        }
        TwdGates { m, r, dt, g0 }
    }

    /// exp(−i·dt·(m + 3r)·β) (diagonal: β = diag(1, 1, −1, −1)).
    pub fn local(&self, m: f64) -> [C; 4] {
        let w = self.dt * (m + 3.0 * self.r);
        let (e1, e2) = (C::from_polar(1.0, -w), C::from_polar(1.0, w));
        [e1, e1, e2, e2]
    }
}

/// Caja periódica con torsión de Bloch: (L_x, L_y, L_z) sitios, twist_j = e^{iK_j} al cruzar el borde.
#[derive(Clone)]
pub struct TwdBox {
    pub dims: [usize; 3],
    pub twist: [C; 3],
}

impl TwdBox {
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
    #[inline(always)]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }
    #[inline(always)]
    pub fn coords(&self, i: usize) -> [usize; 3] {
        [i % self.dims[0], (i / self.dims[0]) % self.dims[1], i / (self.dims[0] * self.dims[1])]
    }
    /// Vecino en +ĵ y si cruza el borde.
    #[inline(always)]
    pub fn up(&self, i: usize, j: usize) -> (usize, bool) {
        let mut c = self.coords(i);
        c[j] += 1;
        let wrap = c[j] == self.dims[j];
        if wrap {
            c[j] = 0;
        }
        (self.idx(c[0], c[1], c[2]), wrap)
    }
}

/// Una capa de compuertas: enlaces (x, x + ĵ) con x_j ≡ p (mod 2). `ph` da e^{iθ} por enlace
/// (índice j·n + x; vacío = 1) y, si `cur` existe, acumula la carga que cruza cada enlace.
pub fn twd_layer(psi: &mut [C], bx: &TwdBox, j: usize, p: usize, g0: &G8, ph: &[C], cur: Option<&mut [f64]>) {
    let n = bx.n();
    let mut cur = cur;
    for i in 0..n {
        let c = bx.coords(i);
        if c[j] % 2 != p {
            continue;
        }
        let (k, wrap) = bx.up(i, j);
        let tw = if wrap { bx.twist[j] } else { cx(1.0, 0.0) };
        let e = if ph.is_empty() { cx(1.0, 0.0) } else { ph[j * n + i] };
        // D†·v con D = diag(1, e^{iθ}); la torsión de Bloch lleva ψ_{x+j} al marco de la celda
        let mut v = [cx(0.0, 0.0); 8];
        for a in 0..4 {
            v[a] = psi[4 * i + a];
            v[4 + a] = psi[4 * k + a] * tw * e.conj();
        }
        let before: f64 = (0..4).map(|a| psi[4 * k + a].norm_sqr()).sum();
        let mut w = [cx(0.0, 0.0); 8];
        for a in 0..8 {
            let mut s = cx(0.0, 0.0);
            for b in 0..8 {
                s += g0[a][b] * v[b];
            }
            w[a] = s;
        }
        let back = e * tw.conj();
        for a in 0..4 {
            psi[4 * i + a] = w[a];
            psi[4 * k + a] = w[4 + a] * back;
        }
        if let Some(cu) = cur.as_deref_mut() {
            let after: f64 = (0..4).map(|a| psi[4 * k + a].norm_sqr()).sum();
            cu[j * n + i] += after - before;
        }
    }
}

/// Un paso de Strang: seis capas con τ = dt/2, el paso local con la masa m y las seis en orden inverso.
pub fn twd_step(psi: &mut [C], bx: &TwdBox, gates: &TwdGates, m: f64, ph: &[C], cur: Option<&mut [f64]>) {
    let order = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)];
    let mut cur = cur;
    for &(j, p) in &order {
        twd_layer(psi, bx, j, p, &gates.g0[j], ph, cur.as_deref_mut());
    }
    let loc = gates.local(m);
    for (i, z) in psi.iter_mut().enumerate() {
        *z *= loc[i % 4];
    }
    for &(j, p) in order.iter().rev() {
        twd_layer(psi, bx, j, p, &gates.g0[j], ph, cur.as_deref_mut());
    }
}

/// El paso de "orden simple" (sin simetrizar): seis capas con τ = dt (dos veces dt/2) y el local.
pub fn twd_step_simple(psi: &mut [C], bx: &TwdBox, gates: &TwdGates, m: f64) {
    let order = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)];
    for &(j, p) in &order {
        twd_layer(psi, bx, j, p, &gates.g0[j], &[], None);
        twd_layer(psi, bx, j, p, &gates.g0[j], &[], None);
    }
    let loc = gates.local(m);
    for (i, z) in psi.iter_mut().enumerate() {
        *z *= loc[i % 4];
    }
}

/// La celda de 2×2×2 (32 componentes) con momento de celda K (torsión e^{iK_j}).
pub fn cell_box(k: [f64; 3]) -> TwdBox {
    TwdBox { dims: [2, 2, 2], twist: [C::from_polar(1.0, k[0]), C::from_polar(1.0, k[1]), C::from_polar(1.0, k[2])] }
}

/// Matriz (fila mayor) del paso en una caja pequeña, aplicándolo a la base.
pub fn box_step_matrix(bx: &TwdBox, gates: &TwdGates, m: f64, ph: &[C], simple: bool) -> Vec<C> {
    let d = 4 * bx.n();
    let mut out = vec![cx(0.0, 0.0); d * d];
    for col in 0..d {
        let mut v = vec![cx(0.0, 0.0); d];
        v[col] = cx(1.0, 0.0);
        if simple {
            twd_step_simple(&mut v, bx, gates, m);
        } else {
            twd_step(&mut v, bx, gates, m, ph, None);
        }
        for r in 0..d {
            out[r * d + col] = v[r];
        }
    }
    out
}

/// Autovalores y una base ortonormal de autovectores de un operador unitario (Gram–Schmidt dentro de
/// los grupos degenerados), ordenados por cuasi-energía ε = −arg λ.
pub fn unitary_eigen(mat: Vec<C>, d: usize) -> (Vec<f64>, Vec<Vec<C>>) {
    let e = eig(mat, d);
    let eps: Vec<f64> = e.values.iter().map(|l| -l.arg()).collect();
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| eps[i].total_cmp(&eps[j]));
    let mut vecs: Vec<Vec<C>> = Vec::with_capacity(d);
    let mut vals = Vec::with_capacity(d);
    for &i in &order {
        let mut v = e.vectors[i].clone();
        for b in &vecs {
            let ov: C = b.iter().zip(&v).map(|(x, y)| x.conj() * y).sum();
            for (vv, bb) in v.iter_mut().zip(b) {
                *vv -= ov * bb;
            }
        }
        let nr: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
        v.iter_mut().for_each(|z| *z /= nr);
        vecs.push(v);
        vals.push(eps[i]);
    }
    (vals, vecs)
}

fn fib_dirs(n: usize) -> Vec<[f64; 3]> {
    crate::pairs::fibonacci_dirs(n)
}

/// Estado del escenario 7 en la malla del universo.
pub struct Twd {
    pub active: bool,
    pub gates: TwdGates,
    pub q: f64,
    pub dynamic: bool,
    pub psi: Vec<C>,
    pub e: Vec<f64>,
    pub a: Vec<f64>,
    pub j: Vec<f64>,
    pub g_ref: Vec<f64>,
    pub out: Vec<f64>,
}

impl Twd {
    pub fn new() -> Self {
        Twd { active: false, gates: TwdGates::new(0.1, 1.0, 0.3), q: 0.0, dynamic: false, psi: Vec::new(), e: Vec::new(), a: Vec::new(), j: Vec::new(), g_ref: Vec::new(), out: Vec::new() }
    }
}

impl QuantumSpacetimeUniverse {
    fn twd_ready(&self) -> bool {
        self.twd.active
            && self.twd.psi.len() == 4 * self.n
            && self.twd.e.len() == 3 * self.n
            && self.twd.a.len() == 3 * self.n
    }

    fn twd_box(&self) -> TwdBox {
        TwdBox { dims: [self.w, self.h, self.d], twist: [cx(1.0, 0.0); 3] }
    }

    fn twd_phases(&self) -> Vec<C> {
        if self.twd.q == 0.0 {
            return Vec::new();
        }
        self.twd.a.iter().map(|a| C::from_polar(1.0, self.twd.q * a)).collect()
    }

    /// Un paso del escenario 7: fermiones (con las corrientes) y, con el campo dinámico, E ← E − q·j y
    /// Maxwell con c = 1 (E += dt·rot B, A −= dt·E).
    pub(crate) fn twd_step_universe(&mut self) {
        if !self.twd_ready() {
            return;
        }
        let bx = self.twd_box();
        let ph = self.twd_phases();
        let n = self.n;
        let mut psi = std::mem::take(&mut self.twd.psi);
        let mut cur = std::mem::take(&mut self.twd.j);
        cur.iter_mut().for_each(|v| *v = 0.0);
        let m = self.twd.gates.m;
        twd_step(&mut psi, &bx, &self.twd.gates, m, &ph, if self.twd.dynamic { Some(&mut cur) } else { None });
        self.twd.psi = psi;
        if self.twd.dynamic {
            let (q, dt) = (self.twd.q, self.twd.gates.dt);
            for l in 0..3 * n {
                self.twd.e[l] -= q * cur[l];
            }
            // E_a += dt·Σ_b [B_ab(r) − B_ab(r − b̂)], B_ab = A_a(r) + A_b(r + â) − A_a(r + b̂) − A_b(r)
            let a = &self.twd.a;
            let bab = |i: usize, aa: usize, b: usize| -> f64 {
                let ia = self.nb(i, aa, 1);
                let ib = self.nb(i, b, 1);
                a[aa * n + i] + a[b * n + ia] - a[aa * n + ib] - a[b * n + i]
            };
            let mut force = vec![0.0; 3 * n];
            for aa in 0..3 {
                for i in 0..n {
                    let mut f = 0.0;
                    for b in 0..3 {
                        if b == aa {
                            continue;
                        }
                        let im = self.nb(i, b, -1);
                        f += bab(i, aa, b) - bab(im, aa, b);
                    }
                    force[aa * n + i] = f;
                }
            }
            for l in 0..3 * n {
                self.twd.e[l] += dt * force[l];
                self.twd.a[l] -= dt * self.twd.e[l];
            }
        }
        self.twd.j = cur;
        self.t = self.t.wrapping_add(1);
    }

    fn twd_density(&self, i: usize) -> f64 {
        (0..4).map(|a| self.twd.psi[4 * i + a].norm_sqr()).sum()
    }

    fn twd_gauss_at(&self, i: usize) -> f64 {
        let n = self.n;
        let mut div = 0.0;
        for a in 0..3 {
            div += self.twd.e[a * n + i] - self.twd.e[a * n + self.nb(i, a, -1)];
        }
        div - self.twd.q * self.twd_density(i)
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.25 (escenario 7): Trotter simétrico de Wilson–Dirac en compuertas de dos sitios; dt ≤ 0,5 (se
    /// admite hasta 1 con un aviso, para ver cómo aparecen los pseudo-dobladores).
    /// Con q ≠ 0 y `field_dynamic`, el U(1) es dinámico (Maxwell con c = 1); con q ≠ 0 y sin él, A es un
    /// fondo fijo. La malla tiene que ser 3D con lados pares.
    pub fn set_scenario_twd(&mut self, m: f32, r: f32, dt: f32, q: f32, field_dynamic: bool) {
        self.message.clear();
        if self.dim != 3 || self.w % 2 != 0 || self.h % 2 != 0 || self.d % 2 != 0 {
            self.set_message("El motor sin dobladores necesita una malla 3D con lados pares (§2.25).");
            return;
        }
        if self.scenario != 7 {
            self.set_scenario(7);
        }
        let n = self.n;
        let m = clamp_finite(m as f64, -10.0, 10.0, 0.1);
        let r = clamp_finite(r as f64, 0.0, 10.0, 1.0);
        let dt = clamp_finite(dt as f64, 0.01, 1.0, 0.3);
        let q = clamp_finite(q as f64, -10.0, 10.0, 0.0);
        if dt > 0.5 {
            self.set_message("dt > 0,5: las cuasi-energías de los dobladores se acercan a ±π (pseudo-dobladores); el motor necesita dt ≲ 0,5.");
        }
        self.twd.gates = TwdGates::new(m, r, dt);
        self.twd.q = q;
        self.twd.dynamic = field_dynamic && q != 0.0;
        self.twd.psi = vec![cx(0.0, 0.0); 4 * n];
        self.twd.e = vec![0.0; 3 * n];
        self.twd.a = vec![0.0; 3 * n];
        self.twd.j = vec![0.0; 3 * n];
        self.twd.g_ref = vec![0.0; n];
        self.twd.active = true;
        self.t = 0;
    }

    /// Paquete gaussiano de la banda positiva del Wilson–Dirac continuo en k₀ (P₊ aplicado a (1, 0, 0,
    /// 0)), de norma `amount`; guarda la ley de Gauss como referencia.
    #[allow(clippy::too_many_arguments)]
    pub fn inject_twd_packet(&mut self, x: f32, y: f32, z: f32, sigma: f32, kx: f32, ky: f32, kz: f32, amount: f32) {
        if !self.twd_ready() {
            return;
        }
        let x = clamp_finite(x as f64, -1e6, 1e6, 0.0);
        let y = clamp_finite(y as f64, -1e6, 1e6, 0.0);
        let z = clamp_finite(z as f64, -1e6, 1e6, 0.0);
        let k = [
            clamp_finite(kx as f64, -1e6, 1e6, 0.0),
            clamp_finite(ky as f64, -1e6, 1e6, 0.0),
            clamp_finite(kz as f64, -1e6, 1e6, 0.0),
        ];
        let amount = clamp_finite(amount as f64, 0.0, 1e6, 1.0);
        let gm = &self.twd.gates;
        let mw = gm.m + gm.r * k.iter().map(|v| 1.0 - v.cos()).sum::<f64>();
        let beta = dirac_matrix(2, 3);
        let mut hk = m4_zero();
        for a in 0..4 {
            for b in 0..4 {
                hk[a][b] = beta[a][b] * mw;
                for (j, kj) in k.iter().enumerate() {
                    hk[a][b] += dirac_matrix(0, j)[a][b] * kj.sin();
                }
            }
        }
        let en = (mw * mw + k.iter().map(|v| v.sin().powi(2)).sum::<f64>()).sqrt().max(1e-12);
        // P₊·(1, 0, 0, 0) = ½(e₀ + H·e₀/E)
        let mut sp = [cx(0.0, 0.0); 4];
        for a in 0..4 {
            sp[a] = (hk[a][0] / en + if a == 0 { cx(1.0, 0.0) } else { cx(0.0, 0.0) }) * 0.5;
        }
        let s2: f64 = sp.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        if s2 <= 1e-30 {
            return;
        }
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let sg = clamp_finite(sigma as f64, 0.5, 1e4, 1.0);
        let mut amps = vec![cx(0.0, 0.0); self.n];
        let mut norm = 0.0;
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let (dx, dy, dz) = (min_image(px as f64 - x, w), min_image(py as f64 - y, h), min_image(pz as f64 - z, d));
            let arg = (dx * dx + dy * dy + dz * dz) / (2.0 * sg * sg);
            if arg > 40.0 {
                continue;
            }
            amps[i] = C::from_polar((-arg).exp(), k[0] * dx + k[1] * dy + k[2] * dz);
            norm += amps[i].norm_sqr();
        }
        if norm <= 1e-300 {
            return;
        }
        let sc = amount.sqrt() / norm.sqrt() / s2;
        for i in 0..self.n {
            for a in 0..4 {
                self.twd.psi[4 * i + a] += amps[i] * sp[a] * sc;
            }
        }
        self.twd_store_gauss();
    }

    /// Guarda la ley de Gauss actual como referencia.
    pub fn twd_store_gauss(&mut self) {
        if !self.twd_ready() {
            return;
        }
        let g: Vec<f64> = (0..self.n).map(|i| self.twd_gauss_at(i)).collect();
        self.twd.g_ref = g;
    }

    /// Fondo U(1) aleatorio: A uniforme en [−amp, amp] por enlace (con E = 0).
    pub fn twd_random_field(&mut self, amp: f32, seed: u32) {
        if !self.twd_ready() || !amp.is_finite() {
            return;
        }
        let mut rng = SplitMix64::new(seed as u64 ^ 0x7D);
        for v in self.twd.a.iter_mut() {
            *v = amp as f64 * (2.0 * rng.next_f64() - 1.0);
        }
        self.twd_store_gauss();
    }

    /// Transformación gauge aleatoria (ψ → e^{iqχ}ψ, A → A + χ(r + â) − χ(r)).
    pub fn twd_random_gauge(&mut self, amp: f32, seed: u32) {
        if !self.twd_ready() || !amp.is_finite() {
            return;
        }
        let mut rng = SplitMix64::new(seed as u64 ^ 0x6A);
        let chi: Vec<f64> = (0..self.n).map(|_| amp as f64 * rng.gauss()).collect();
        let q = self.twd.q;
        for i in 0..self.n {
            let p = C::from_polar(1.0, q * chi[i]);
            for a in 0..4 {
                self.twd.psi[4 * i + a] *= p;
            }
        }
        for a in 0..3 {
            for i in 0..self.n {
                let ip = self.nb(i, a, 1);
                self.twd.a[a * self.n + i] += chi[ip] - chi[i];
            }
        }
    }

    pub fn twd_norm(&self) -> f64 {
        if !self.twd_ready() {
            return 0.0;
        }
        (0..self.n).map(|i| self.twd_density(i)).sum()
    }

    /// Máximo de |G − G_ref| (Gauss).
    pub fn twd_gauss_residual(&self) -> f64 {
        if !self.twd_ready() {
            return 0.0;
        }
        (0..self.n).map(|i| (self.twd_gauss_at(i) - self.twd.g_ref.get(i).copied().unwrap_or(0.0)).abs()).fold(0.0, f64::max)
    }

    /// Distancia máxima entre las densidades de dos universos del escenario 7 (copias gauge).
    pub fn twd_density_distance(&self, other: &QuantumSpacetimeUniverse) -> f64 {
        if !self.twd_ready() || !other.twd_ready() || other.n != self.n {
            return f64::NAN;
        }
        (0..self.n).map(|i| (self.twd_density(i) - other.twd_density(i)).abs()).fold(0.0, f64::max)
    }

    /// Cuasi-energías por paso (32, ordenadas) del operador en la celda de 2×2×2 con momento de celda
    /// K; `simple` = orden sin simetrizar.
    pub fn twd_cell_spectrum(&self, kx: f32, ky: f32, kz: f32, simple: bool) -> Vec<f64> {
        let g = &self.twd.gates;
        let bx = cell_box([
            clamp_finite(kx as f64, -1e6, 1e6, 0.0),
            clamp_finite(ky as f64, -1e6, 1e6, 0.0),
            clamp_finite(kz as f64, -1e6, 1e6, 0.0),
        ]);
        let mat = box_step_matrix(&bx, g, g.m, &[], simple);
        let mut v: Vec<f64> = eig(mat, 32).values.iter().map(|l| -l.arg()).collect();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Energías del Wilson–Dirac continuo en el momento por sitio k (las dos ramas, por espín).
    pub fn twd_wilson_dirac_energy(&self, kx: f32, ky: f32, kz: f32) -> f64 {
        let g = &self.twd.gates;
        let k = [
            clamp_finite(kx as f64, -1e6, 1e6, 0.0),
            clamp_finite(ky as f64, -1e6, 1e6, 0.0),
            clamp_finite(kz as f64, -1e6, 1e6, 0.0),
        ];
        let mw = g.m + g.r * k.iter().map(|v| 1.0 - v.cos()).sum::<f64>();
        (mw * mw + k.iter().map(|v| v.sin().powi(2)).sum::<f64>()).sqrt()
    }

    /// §3.23: Higgs → f f̄ con el motor y el mar lleno: m(t) = m_f + amplitude·cos(m_h·t), celda de 2×2×2,
    /// capa de 12 direcciones × 60 radios en |k| ± 0,06 alrededor de la resonancia. Pares por sitio y
    /// por unidad de tiempo.
    pub fn twd_pair_rate_higgs(&mut self, m_f: f32, m_h: f32, amplitude: f32, dt: f32, t_phys: f32) -> f64 {
        if !m_f.is_finite() || !m_h.is_finite() || !amplitude.is_finite() || !dt.is_finite() || dt <= 0.0 || !t_phys.is_finite() || t_phys <= 0.0 {
            return f64::NAN;
        }
        let (mf, mh, dm) = (m_f as f64, m_h as f64, amplitude as f64);
        let dt = clamp_finite(dt as f64, 1e-3, 1.0, 0.1);
        let gates = TwdGates::new(mf, 1.0, dt);
        let steps = ((t_phys as f64 / dt).round() as usize).clamp(1, 50_000);
        let kres = ((0.5 * mh).powi(2) - mf * mf).max(1e-6).sqrt();
        let (qmin, nr, dirs) = ((kres - 0.06).max(0.002), 60usize, fib_dirs(12));
        let dq = 0.12 / nr as f64;
        let mut total = 0.0;
        for dir in &dirs {
            for ir in 0..nr {
                let q = qmin + (ir as f64 + 0.5) * dq;
                let kc = [2.0 * q * dir[0], 2.0 * q * dir[1], 2.0 * q * dir[2]];
                let bx = cell_box(kc);
                // operadores fijos: las capas hacia delante (P) y hacia atrás (Q)
                let (pm, qm) = half_matrices(&bx, &gates);
                let loc0 = gates.local(mf);
                let u0 = compose(&qm, &loc0, &pm);
                let (eps, vecs) = unitary_eigen(u0, 32);
                // mar físico: los dos estados de la rama negativa más cercanos a 0
                let mut neg: Vec<usize> = (0..32).filter(|&i| eps[i] < 0.0).collect();
                neg.sort_by(|&a, &b| eps[b].total_cmp(&eps[a]));
                let mut occ: Vec<Vec<C>> = neg.iter().take(2).map(|&i| vecs[i].clone()).collect();
                for t in 0..steps {
                    let loc = gates.local(mf + dm * (mh * t as f64 * dt).cos());
                    for v in occ.iter_mut() {
                        let a = matvec(&pm, v);
                        let b: Vec<C> = a.iter().enumerate().map(|(i, z)| z * loc[i % 4]).collect();
                        *v = matvec(&qm, &b);
                    }
                }
                let mut nq = 0.0;
                for v in &occ {
                    for (i, b) in vecs.iter().enumerate() {
                        if eps[i] > 0.0 {
                            let ov: C = b.iter().zip(v).map(|(x, y)| x.conj() * y).sum();
                            nq += ov.norm_sqr();
                        }
                    }
                }
                total += 4.0 * PI / dirs.len() as f64 * q * q * dq / (2.0 * PI).powi(3) * nq;
            }
        }
        total / (steps as f64 * dt)
    }

    /// §3.23: probabilidad de par por espín de un modo con k⊥ fijo cuando un E uniforme barre k_z de
    /// −1,2 a +1,2 (q·A_z(t) = −q·E·t en los enlaces z).
    pub fn twd_landau_zener(&mut self, m: f32, q_e: f32, k_perp: f32, dt: f32) -> f64 {
        if !m.is_finite() || !q_e.is_finite() || (q_e as f64).abs() < 1e-12 || !k_perp.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return f64::NAN;
        }
        let dt = clamp_finite(dt as f64, 1e-3, 1.0, 0.05);
        let qe = q_e as f64;
        let gates = TwdGates::new(m as f64, 1.0, dt);
        let k0 = [k_perp as f64, 0.0, -1.2];
        let bx = cell_box([2.0 * k0[0], 2.0 * k0[1], 2.0 * k0[2]]);
        let steps = ((2.4 / (qe.abs() * dt)).round() as usize).clamp(1, 50_000);
        let ph_of = |theta: f64| -> Vec<C> {
            let mut ph = vec![cx(1.0, 0.0); 3 * 8];
            for i in 0..8 {
                ph[16 + i] = C::from_polar(1.0, theta);
            }
            ph
        };
        let u0 = box_step_matrix(&bx, &gates, gates.m, &ph_of(0.0), false);
        let (eps, vecs) = unitary_eigen(u0, 32);
        let mut neg: Vec<usize> = (0..32).filter(|&i| eps[i] < 0.0).collect();
        neg.sort_by(|&a, &b| eps[b].total_cmp(&eps[a]));
        let mut occ: Vec<Vec<C>> = neg.iter().take(2).map(|&i| vecs[i].clone()).collect();
        // q·A_z(t) = −q·E·t: la fase del enlace z es e^{iqA} = e^{−iqE·t}
        let mut qa = 0.0;
        for _ in 0..steps {
            let ph = ph_of(qa);
            for v in occ.iter_mut() {
                twd_step(v, &bx, &gates, gates.m, &ph, None);
            }
            qa -= qe * dt;
        }
        let u1 = box_step_matrix(&bx, &gates, gates.m, &ph_of(qa), false);
        let (eps1, vecs1) = unitary_eigen(u1, 32);
        // rama física positiva: los dos estados positivos más cercanos a 0
        let mut pos: Vec<usize> = (0..32).filter(|&i| eps1[i] > 0.0).collect();
        pos.sort_by(|&a, &b| eps1[a].total_cmp(&eps1[b]));
        let mut p = 0.0;
        for v in &occ {
            for &i in pos.iter().take(2) {
                let ov: C = vecs1[i].iter().zip(v).map(|(x, y)| x.conj() * y).sum();
                p += ov.norm_sqr();
            }
        }
        p / 2.0
    }

    /// §3.23, reacción en 3D: el mar lleno en una rejilla de cells³ momentos de celda ((2·cells)³ sitios,
    /// condiciones antiperiódicas), E homogéneo a lo largo de z con E ← E − q·j_z y A_z ← A_z − dt·E.
    /// Devuelve [pasos + 1, E(t)…, energía total por sitio…]: el campo ½E² más la energía de Floquet de
    /// los fermiones, Σ ε/dt de los autovalores e^{−iε} del paso (cada 10 pasos; NaN en los demás).
    pub fn twd_homogeneous_run(&mut self, m: f32, q: f32, e0: f32, cells: u32, t_phys: f32, dt: f32) -> *const f64 {
        let m = clamp_finite(m as f64, -10.0, 10.0, 0.3);
        let q = clamp_finite(q as f64, -10.0, 10.0, 0.3);
        let e0 = clamp_finite(e0 as f64, -10.0, 10.0, 0.01);
        let dt = clamp_finite(dt as f64, 1e-3, 1.0, 0.2);
        let t_phys = clamp_finite(t_phys as f64, 0.0, 50_000.0, 200.0);
        let gates = TwdGates::new(m, 1.0, dt);
        let nc = (cells as usize).clamp(1, 32);
        let steps = ((t_phys / dt).round() as usize).clamp(0, 50_000);
        let nsites = (8 * nc * nc * nc) as f64;
        // mar: los 16 estados negativos del operador libre en cada K
        let mut sectors: Vec<(TwdBox, Vec<Vec<C>>)> = Vec::new();
        for ix in 0..nc {
            for iy in 0..nc {
                for iz in 0..nc {
                    // condiciones antiperiódicas (K = 2π(n + ½)/n_c): con las periódicas, la energía del
                    // mar de un toro finito tiene una curvatura negativa frente a un A uniforme (un
                    // artefacto de tamaño finito que decae con el tamaño) y el campo crece
                    let sh = 0.5;
                    let kk = [2.0 * PI * (ix as f64 + sh) / nc as f64, 2.0 * PI * (iy as f64 + sh) / nc as f64, 2.0 * PI * (iz as f64 + sh) / nc as f64];
                    let bx = cell_box(kk);
                    let (eps, vecs) = unitary_eigen(box_step_matrix(&bx, &gates, m, &[], false), 32);
                    let occ: Vec<Vec<C>> = (0..32).filter(|&i| eps[i] < 0.0).map(|i| vecs[i].clone()).collect();
                    sectors.push((bx, occ));
                }
            }
        }
        let ph_of = |theta: f64| -> Vec<C> {
            let mut ph = vec![cx(1.0, 0.0); 24];
            for i in 0..8 {
                ph[16 + i] = C::from_polar(1.0, theta);
            }
            ph
        };
        let energy = |sectors: &Vec<(TwdBox, Vec<Vec<C>>)>, qa: f64, e: f64| -> f64 {
            let ph = ph_of(qa);
            let mut ef = 0.0;
            for (bx, occ) in sectors {
                let (eps, vecs) = unitary_eigen(box_step_matrix(bx, &gates, m, &ph, false), 32);
                for v in occ {
                    for (j, b) in vecs.iter().enumerate() {
                        let ov: C = b.iter().zip(v).map(|(x, y)| x.conj() * y).sum();
                        ef += eps[j] / dt * ov.norm_sqr();
                    }
                }
            }
            ef / nsites + 0.5 * e * e
        };
        let (mut e, mut a) = (e0, 0.0f64);
        let mut es = vec![e];
        let mut en = vec![energy(&sectors, q * a, e)];
        for t in 0..steps {
            let ph = ph_of(q * a);
            let mut moved = 0.0;
            for (bx, occ) in sectors.iter_mut() {
                for v in occ.iter_mut() {
                    let mut cur = vec![0.0; 24];
                    twd_step(v, bx, &gates, m, &ph, Some(&mut cur));
                    moved += cur[16..24].iter().sum::<f64>();
                }
            }
            // carga por enlace z y por paso (8 enlaces z por celda)
            let jz = moved / nsites;
            e -= q * jz;
            a -= dt * e;
            es.push(e);
            if t % 10 == 9 || t + 1 == steps {
                en.push(energy(&sectors, q * a, e));
            } else {
                en.push(f64::NAN);
            }
        }
        let mut out = vec![es.len() as f64];
        out.extend_from_slice(&es);
        out.extend_from_slice(&en);
        self.twd.out = out;
        self.twd.out.as_ptr()
    }

    /// §3.23, anomalía en 3+1D (E ∥ B): plano transversal de L×L sitios en espacio real con `flux`
    /// cuantos (qB·L² = 2π·flux, gauge de Landau con costura), Bloch en z con nz momentos de celda (2
    /// sitios en z), mar lleno, E constante a lo largo de z. Devuelve dj_z/dt (corriente por sitio y por
    /// unidad de tiempo) dividida entre (qB/2π)(qE/π).
    pub fn twd_anomaly_run(&mut self, l: u32, flux: u32, q_e: f32, m: f32, dt: f32, nz: u32, t_phys: f32) -> f64 {
        if flux == 0 || !q_e.is_finite() || (q_e as f64).abs() < 1e-12 || !m.is_finite() || !dt.is_finite() || dt <= 0.0 || !t_phys.is_finite() || t_phys <= 0.0 {
            return f64::NAN;
        }
        let l = ((l as usize).clamp(2, 16) & !1).max(2);
        let dt = clamp_finite(dt as f64, 1e-3, 1.0, 0.2);
        let qe = q_e as f64;
        let gates = TwdGates::new(m as f64, 1.0, dt);
        let qb = 2.0 * PI * flux as f64 / (l * l) as f64;
        let steps = ((t_phys as f64 / dt).round() as usize).clamp(2, 20_000);
        let nzz = (nz as usize).clamp(1, 64);
        let mut sectors: Vec<(TwdBox, Vec<Vec<C>>)> = Vec::new();
        let base_ph = |bx: &TwdBox, theta_z: f64| -> Vec<C> {
            let n = bx.n();
            let mut ph = vec![cx(1.0, 0.0); 3 * n];
            for i in 0..n {
                let c = bx.coords(i);
                ph[n + i] = C::from_polar(1.0, qb * c[0] as f64);
                if c[0] == l - 1 {
                    ph[i] = C::from_polar(1.0, -qb * l as f64 * c[1] as f64);
                }
                ph[2 * n + i] = C::from_polar(1.0, theta_z);
            }
            ph
        };
        for iz in 0..nzz {
            let kz = 2.0 * PI * iz as f64 / nzz as f64;
            let bx = TwdBox { dims: [l, l, 2], twist: [cx(1.0, 0.0), cx(1.0, 0.0), C::from_polar(1.0, kz)] };
            let ph = base_ph(&bx, 0.0);
            let d = 4 * bx.n();
            let (eps, vecs) = unitary_eigen(box_step_matrix(&bx, &gates, gates.m, &ph, false), d);
            let occ: Vec<Vec<C>> = (0..d).filter(|&i| eps[i] < 0.0).map(|i| vecs[i].clone()).collect();
            sectors.push((bx, occ));
        }
        let nsites = (l * l * 2 * nzz) as f64;
        let mut js = Vec::with_capacity(steps);
        let mut qa = 0.0;
        for _ in 0..steps {
            let mut moved = 0.0;
            for (bx, occ) in sectors.iter_mut() {
                let ph = base_ph(bx, qa);
                let n = bx.n();
                for v in occ.iter_mut() {
                    let mut cur = vec![0.0; 3 * n];
                    twd_step(v, bx, &gates, gates.m, &ph, Some(&mut cur));
                    moved += cur[2 * n..].iter().sum::<f64>();
                }
            }
            js.push(moved / (nsites * dt));
            qa -= qe * dt;
        }
        // pendiente en la segunda mitad
        let t0 = steps / 2;
        let ts: Vec<f64> = (t0..steps).map(|t| (t + 1) as f64 * dt).collect();
        let slope = crate::util::fit_linear(&ts, &js[t0..]).1;
        slope / (qb / (2.0 * PI) * qe / PI)
    }
}

/// P (capas hacia delante con τ = dt/2) y Q (hacia atrás) de una caja pequeña, sin campo.
fn half_matrices(bx: &TwdBox, g: &TwdGates) -> (Vec<C>, Vec<C>) {
    let d = 4 * bx.n();
    let order = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)];
    let mut p = vec![cx(0.0, 0.0); d * d];
    let mut q = vec![cx(0.0, 0.0); d * d];
    for col in 0..d {
        let mut v = vec![cx(0.0, 0.0); d];
        v[col] = cx(1.0, 0.0);
        let mut w = v.clone();
        for &(j, pp) in &order {
            twd_layer(&mut v, bx, j, pp, &g.g0[j], &[], None);
        }
        for &(j, pp) in order.iter().rev() {
            twd_layer(&mut w, bx, j, pp, &g.g0[j], &[], None);
        }
        for r in 0..d {
            p[r * d + col] = v[r];
            q[r * d + col] = w[r];
        }
    }
    (p, q)
}

fn matvec(m: &[C], v: &[C]) -> Vec<C> {
    let d = v.len();
    let mut out = vec![cx(0.0, 0.0); d];
    for r in 0..d {
        let row = &m[r * d..(r + 1) * d];
        let mut s = cx(0.0, 0.0);
        for (a, b) in row.iter().zip(v) {
            s += a * b;
        }
        out[r] = s;
    }
    out
}

/// Q·diag(loc)·P.
fn compose(q: &[C], loc: &[C; 4], p: &[C]) -> Vec<C> {
    let d = (p.len() as f64).sqrt() as usize;
    let mut lp = p.to_vec();
    for r in 0..d {
        for c in 0..d {
            lp[r * d + c] *= loc[r % 4];
        }
    }
    let mut out = vec![cx(0.0, 0.0); d * d];
    for r in 0..d {
        for k in 0..d {
            let a = q[r * d + k];
            if a == cx(0.0, 0.0) {
                continue;
            }
            for c in 0..d {
                out[r * d + c] += a * lp[k * d + c];
            }
        }
    }
    out
}
