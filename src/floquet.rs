//! Quiralidad por resonancia (§2.27, escenario 9): el Floquet anómalo de Rudner, Lindner, Berg y
//! Levin (2013) en una red cuadrada bipartita o en una alfombra de Sierpinski, y el índice que dice
//! por qué la geometría sola no quita los dobladores.
//! - Cuatro pasos de transferencia completa (ψ_A, ψ_B) → (−iψ_B, −iψ_A) por los enlaces +x, +y, −x y
//!   −y devuelven cada partícula del interior a su sitio tras recorrer una plaqueta. En un borde falta un
//!   enlace y la partícula avanza: un modo de borde quiral con el interior trivial. En un fractal cada
//!   agujero es una frontera con su propio canal quiral. Un quinto paso con fases aleatorias en cada
//!   sitio (desorden) no frena el modo de borde.
//! - Un hamiltoniano con simetría quiral es H = [[0, D], [D†, 0]] con D cuadrada, sea cual sea la red:
//!   su índice, dim ker D − dim ker D†, es 0. Una sola especie en N_φ cuantos de flujo necesitaría
//!   índice N_φ, así que ni una red aleatoria ni un fractal dan una sola especie quiral.

use crate::linalg::hermitian_eigvals;
use crate::rng::SplitMix64;
use crate::sun::C;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// ¿Está el sitio (x, y) en la alfombra de Sierpinski de lado 3^nivel?
pub fn in_carpet(mut x: usize, mut y: usize) -> bool {
    while x > 0 || y > 0 {
        if x % 3 == 1 && y % 3 == 1 {
            return false;
        }
        x /= 3;
        y /= 3;
    }
    true
}

/// Máscara de sitios: 0 = red cuadrada L×L, 1 = alfombra de nivel n (lado 3ⁿ).
pub fn lattice_mask(lattice: u32, size_or_level: u32) -> (usize, Vec<bool>) {
    if lattice == 0 {
        let l = (size_or_level as usize).clamp(2, 512);
        (l, vec![true; l * l])
    } else {
        let lev = size_or_level.clamp(1, 6);
        let l = 3usize.pow(lev);
        let mut m = vec![false; l * l];
        for y in 0..l {
            for x in 0..l {
                m[y * l + x] = in_carpet(x, y);
            }
        }
        (l, m)
    }
}

pub struct Floquet {
    pub active: bool,
    pub l: usize,
    pub mask: Vec<bool>,
    pub psi: Vec<C>,
    pub tmp: Vec<C>,
    pub disorder: Vec<f64>,
    pub periods: u32,
}

impl Floquet {
    pub fn new() -> Self {
        Floquet { active: false, l: 0, mask: Vec::new(), psi: Vec::new(), tmp: Vec::new(), disorder: Vec::new(), periods: 0 }
    }

    /// Un período: los cuatro pasos de transferencia (A en r con B en r + b_j, b = +x, +y, −x, −y) y,
    /// con desorden, un quinto paso de fases por sitio. Los sitios sin pareja (borde, agujeros) se quedan.
    pub fn period(&mut self) {
        let l = self.l as i64;
        let dirs = [(1i64, 0i64), (0, 1), (-1, 0), (0, -1)];
        for &(dx, dy) in &dirs {
            self.tmp.copy_from_slice(&self.psi);
            for y in 0..l {
                for x in 0..l {
                    if (x + y) % 2 != 0 {
                        continue;
                    }
                    let a = (y * l + x) as usize;
                    if !self.mask[a] {
                        continue;
                    }
                    let (bx, by) = (x + dx, y + dy);
                    if bx < 0 || by < 0 || bx >= l || by >= l {
                        continue;
                    }
                    let b = (by * l + bx) as usize;
                    if !self.mask[b] {
                        continue;
                    }
                    let mi = Complex::new(0.0, -1.0);
                    self.tmp[a] = mi * self.psi[b];
                    self.tmp[b] = mi * self.psi[a];
                }
            }
            std::mem::swap(&mut self.psi, &mut self.tmp);
        }
        if !self.disorder.is_empty() {
            for (z, ph) in self.psi.iter_mut().zip(&self.disorder) {
                *z *= C::from_polar(1.0, *ph);
            }
        }
        self.periods += 1;
    }
}

/// Matriz D (N×N, fila mayor) del fermión ingenuo H = Σ (−i/2)·σ_a·e^{iθ}·t_l·ψ†_{x+a}ψ_x + h.c. en la
/// base de quiralidad (σ_z): H = [[0, D], [D†, 0]]. Flujo uniforme por el gauge de Landau θ_y = B·x con
/// B = 2π·N_φ/L²; `jitter` multiplica cada salto por 1 + jitter·u con u ∈ [−1, 1].
pub fn naive_d(l: usize, mask: &[bool], flux: u32, jitter: f64, seed: u64) -> (Vec<C>, usize) {
    let sites: Vec<usize> = (0..l * l).filter(|&i| mask[i]).collect();
    let mut id = vec![usize::MAX; l * l];
    for (k, &i) in sites.iter().enumerate() {
        id[i] = k;
    }
    let n = sites.len();
    let b = 2.0 * PI * flux as f64 / (l * l) as f64;
    let mut rng = SplitMix64::new(seed);
    let mut d = vec![Complex::new(0.0, 0.0); n * n];
    for &i in &sites {
        let (x, y) = (i % l, i / l);
        for a in 0..2 {
            let (nx, ny) = if a == 0 { (x + 1, y) } else { (x, y + 1) };
            if nx >= l || ny >= l {
                continue;
            }
            let j = ny * l + nx;
            if !mask[j] {
                continue;
            }
            let t = 1.0 + jitter * (2.0 * rng.next_f64() - 1.0);
            let th = if a == 1 { b * x as f64 } else { 0.0 };
            let e = C::from_polar(t, th);
            let (ki, kj) = (id[i], id[j]);
            // bloque (σ_z = +, σ_z = −): σ_x → 1, σ_y → −i; ⟨x+a|H|x⟩ = (−i/2)σ_a·e^{iθ}
            let (fwd, back) = if a == 0 { (Complex::new(0.0, -0.5), Complex::new(0.0, 0.5)) } else { (Complex::new(-0.5, 0.0), Complex::new(0.5, 0.0)) };
            d[kj * n + ki] += fwd * e;
            d[ki * n + kj] += back * e.conj();
        }
    }
    (d, n)
}

/// Recuento de modos casi nulos (|E| < eps) por quiralidad: [n₊, n₋, n₊ − n₋, total]. n₊ son los
/// autovalores de D†D por debajo de eps² y n₋ los de DD†: los pares ±E se compensan y la diferencia es
/// el índice.
pub fn chiral_counts(lattice: u32, size_or_level: u32, flux: u32, jitter: f64, eps: f64) -> [f64; 4] {
    let (l, mask) = lattice_mask(lattice, size_or_level);
    let (d, n) = naive_d(l, &mask, flux, jitter, 0xC417);
    let mut dd = vec![Complex::new(0.0, 0.0); n * n];
    let mut ddt = vec![Complex::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let (mut s1, mut s2) = (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
            for k in 0..n {
                s1 += d[k * n + i].conj() * d[k * n + j];
                s2 += d[i * n + k] * d[j * n + k].conj();
            }
            dd[i * n + j] = s1;
            ddt[i * n + j] = s2;
        }
    }
    let e1 = hermitian_eigvals(dd, n);
    let e2 = hermitian_eigvals(ddt, n);
    let np = e1.iter().filter(|v| **v < eps * eps).count() as f64;
    let nm = e2.iter().filter(|v| **v < eps * eps).count() as f64;
    [np, nm, np - nm, np + nm]
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.27 (escenario 9): 0 = red cuadrada (lado size_or_level), 1 = alfombra de Sierpinski de ese
    /// nivel; `disorder` ∈ [0, 1] da un quinto paso con fases aleatorias en [−π·disorder, π·disorder].
    pub fn floquet_anomalous_init(&mut self, lattice: u32, size_or_level: u32, disorder: f32) {
        let (l, mask) = lattice_mask(lattice, size_or_level);
        let mut f = Floquet::new();
        f.l = l;
        f.mask = mask;
        f.psi = vec![Complex::new(0.0, 0.0); l * l];
        f.tmp = f.psi.clone();
        if disorder > 0.0 {
            let mut rng = SplitMix64::new(0xD150);
            f.disorder = (0..l * l).map(|_| PI * disorder as f64 * (2.0 * rng.next_f64() - 1.0)).collect();
        }
        f.active = true;
        self.flq = f;
    }

    /// Pone una partícula (norma 1) en el sitio (x, y).
    pub fn floquet_inject(&mut self, x: u32, y: u32) {
        let f = &mut self.flq;
        if !f.active {
            return;
        }
        f.psi.iter_mut().for_each(|z| *z = Complex::new(0.0, 0.0));
        let (x, y) = ((x as usize).min(f.l - 1), (y as usize).min(f.l - 1));
        f.psi[y * f.l + x] = Complex::new(1.0, 0.0);
        f.periods = 0;
    }

    /// Avanza `periods` períodos.
    pub fn floquet_periods(&mut self, periods: u32) {
        for _ in 0..periods {
            self.flq.period();
        }
    }

    /// [x, y, probabilidad] del sitio más probable, y la norma total.
    pub fn floquet_max_site(&self) -> Vec<f64> {
        let f = &self.flq;
        let (mut best, mut bi) = (0.0f64, 0usize);
        let mut tot = 0.0;
        for (i, z) in f.psi.iter().enumerate() {
            let p = z.norm_sqr();
            tot += p;
            if p > best {
                best = p;
                bi = i;
            }
        }
        vec![(bi % f.l.max(1)) as f64, (bi / f.l.max(1)) as f64, best, tot]
    }

    /// Lado de la red del escenario 9.
    pub fn floquet_size(&self) -> u32 {
        self.flq.l as u32
    }

    /// ¿Hay sitio en (x, y)?
    pub fn floquet_site(&self, x: u32, y: u32) -> bool {
        let f = &self.flq;
        (x as usize) < f.l && (y as usize) < f.l && f.mask[y as usize * f.l + x as usize]
    }

    /// Suma de las quiralidades de los modos casi nulos (|E| < 0,05) del fermión ingenuo con
    /// `flux_quanta` cuantos: 0 = red cuadrada 27×27, 1 = alfombra de nivel 3; `jitter`, saltos aleatorios.
    pub fn chiral_index(&self, lattice: u32, flux_quanta: u32, jitter: f32) -> f64 {
        chiral_counts(lattice, if lattice == 0 { 27 } else { 3 }, flux_quanta, jitter as f64, 0.05)[2]
    }

    /// [modos casi nulos de quiralidad +, de quiralidad −, suma de quiralidades, total].
    pub fn chiral_mode_counts(&self, lattice: u32, flux_quanta: u32, jitter: f32) -> Vec<f64> {
        chiral_counts(lattice, if lattice == 0 { 27 } else { 3 }, flux_quanta, jitter as f64, 0.05).to_vec()
    }
}
