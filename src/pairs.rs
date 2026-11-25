//! Laboratorio de pares (§2.23 y §3.21): el mar de Dirac lleno sobre campos clásicos de fondo.
//! La caminata es lineal en el campo fermiónico: un estado de Slater sigue siendo de Slater bajo una
//! evolución lineal, así que basta evolucionar cada modo ocupado con la misma caminata (el método de
//! las funciones de modo, Aarts y Smit, 1999). Es exacto con campos de fondo dados e incluye la
//! estadística de Fermi: la ocupación de cada modo nunca pasa de 1.
//! - 1D: la caminata sin dobladores de Gupta y Short, U = W·A·B (θ = 0 es la convencional), en k con
//!   campos homogéneos (Schwinger, anomalía, bosón de Schwinger, polarización del vacío, quarks en un
//!   campo de color) y en un anillo en espacio real con la reacción E ← E − q·j (rotura de la cuerda).
//! - 3D: el Dirac de §2.16 y los Weyl de §2.17 con el orden alternado, nodo por nodo (Higgs → f f̄,
//!   W¹ → ℓν y Z → f f̄), integrando |q| alrededor de cada nodo en 24 direcciones.
//! No usa la malla del universo. Solo cuantiza fermiones libres en campos clásicos; el W, el Z y el
//! Higgs son condensados clásicos y su "desintegración" es producción de pares a primer orden.

use crate::dirac2d::{su2_band, C64, M2};
use crate::linalg::eig;
use crate::rng::SplitMix64;
use crate::spinor::{band_projector_s, cx, m4_identity, m4_mul, m4_zero, pauli_eigenbasis, SpinorWalk, M4};
use crate::sun::Algebra;
use crate::util::{clamp_finite, fibonacci_sphere};
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

// ------------------------------------------------------------------ caminata 1D de Gupta y Short

/// U = W·A·B con B = Π_b + Π_b̄·S† (la parte b̄ salta a la izquierda), A = Π_a·S + Π_ā (la parte a
/// salta a la derecha) y W = exp(−i·m·σ_x); Π_a = |↑_θ⟩⟨↑_θ|, Π_b = |↑_−θ⟩⟨↑_−θ|, con
/// |↑_θ⟩ = cos(θ/2)|↑⟩ − i·sen(θ/2)|↓⟩. En el continuo, H = cos θ·k·σ_z + m·σ_x.
#[derive(Clone, Copy)]
pub struct Lazy {
    pub a: [C64; 2],
    pub b: [C64; 2],
    pub w: M2,
}

#[inline(always)]
fn braket(v: &[C64; 2], p: &[C64; 2]) -> C64 {
    v[0].conj() * p[0] + v[1].conj() * p[1]
}

impl Lazy {
    pub fn new(m: f64, theta: f64) -> Self {
        let ket = |t: f64| [cx((0.5 * t).cos(), 0.0), cx(0.0, -(0.5 * t).sin())];
        let (s, c) = m.sin_cos();
        Lazy { a: ket(theta), b: ket(-theta), w: [[cx(c, 0.0), cx(0.0, -s)], [cx(0.0, -s), cx(c, 0.0)]] }
    }

    /// Un paso de un modo homogéneo con κ = k − q·A: devuelve (lo que cruza a la derecha en A, lo
    /// que cruza a la izquierda en B), las dos partes de la corriente.
    #[inline(always)]
    pub fn step(&self, psi: &mut [C64; 2], kappa: f64) -> (f64, f64) {
        let pb = braket(&self.b, psi);
        let bbar = [psi[0] - self.b[0] * pb, psi[1] - self.b[1] * pb];
        let left = bbar[0].norm_sqr() + bbar[1].norm_sqr();
        let f = C64::from_polar(1.0, kappa) - cx(1.0, 0.0);
        let phi = [psi[0] + bbar[0] * f, psi[1] + bbar[1] * f];
        let pa = braket(&self.a, &phi);
        let right = pa.norm_sqr();
        let g = (C64::from_polar(1.0, -kappa) - cx(1.0, 0.0)) * pa;
        let chi = [phi[0] + self.a[0] * g, phi[1] + self.a[1] * g];
        psi[0] = self.w[0][0] * chi[0] + self.w[0][1] * chi[1];
        psi[1] = self.w[1][0] * chi[0] + self.w[1][1] * chi[1];
        (right, left)
    }

    /// El operador de un paso en κ.
    pub fn matrix(&self, kappa: f64) -> M2 {
        let mut m = [[cx(0.0, 0.0); 2]; 2];
        for j in 0..2 {
            let mut v = [cx(0.0, 0.0); 2];
            v[j] = cx(1.0, 0.0);
            self.step(&mut v, kappa);
            m[0][j] = v[0];
            m[1][j] = v[1];
        }
        m
    }

    /// Cuasi-energía positiva ε(κ) ∈ [0, π] (det U = 1).
    pub fn omega(&self, kappa: f64) -> f64 {
        su2_band(&self.matrix(kappa), 1).0
    }
}

/// c_ef del ajuste ε² = m² + c²·k² + d·k⁴ en k ∈ (0; 0,04].
pub fn lazy_c_eff(m: f64, theta: f64) -> f64 {
    let lw = Lazy::new(m, theta);
    let e0 = lw.omega(0.0);
    // mínimos cuadrados de (ε² − ε₀²)/k² = c² + d·k²
    let (mut sx, mut sy, mut sxx, mut sxy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for j in 1..=20 {
        let k = 0.002 * j as f64;
        let e = lw.omega(k);
        let y = (e * e - e0 * e0) / (k * k);
        let x = k * k;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        n += 1.0;
    }
    let d = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    ((sy - d * sx) / n).max(0.0).sqrt()
}

/// Mar de la banda negativa del operador libre (A = 0) en los L valores de k de la red.
fn sea_1d(lw: &Lazy, modes: usize) -> Vec<[C64; 2]> {
    (0..modes)
        .map(|j| {
            let k = 2.0 * PI * j as f64 / modes as f64;
            su2_band(&lw.matrix(k), -1).1
        })
        .collect()
}

/// Ocupación de la banda positiva del operador instantáneo en κ.
#[inline]
fn occupation(lw: &Lazy, psi: &[C64; 2], kappa: f64) -> f64 {
    let v = su2_band(&lw.matrix(kappa), 1).1;
    braket(&v, psi).norm_sqr()
}

// ------------------------------------------------------------------ anillo 1D en espacio real

/// La cuerda entre dos cargas externas ±q en un anillo de N celdas, con el mar lleno (N modos de 2N
/// amplitudes) y la reacción E ← E − q·j, A ← A − E (Gauss exacta, como en §2.9).
pub struct Ring {
    pub n: usize,
    pub lw: Lazy,
    pub q: f64,
    pub xp: usize,
    pub xm: usize,
    pub d: usize,
    pub rho_ext: Vec<f64>,
    pub e: Vec<f64>,
    pub a: Vec<f64>,
    pub modes: Vec<Vec<[C64; 2]>>,
    pub tmp: Vec<[C64; 2]>,
    pub j: Vec<f64>,
    pub t: u32,
    /// Sin reacción (para el mar inerte con fases aleatorias).
    pub backreaction: bool,
}

impl Ring {
    pub fn new(n: usize, m: f64, theta: f64, q: f64, d: usize) -> Self {
        let lw = Lazy::new(m, theta);
        let d = d.clamp(1, n - 1);
        let xp = (n - d) / 2;
        let xm = xp + d;
        let mut rho_ext = vec![0.0; n];
        rho_ext[xp] = q;
        rho_ext[xm] = -q;
        // E(x) − E(x − 1) = ρ_ext(x): E = q en los enlaces x_+ … x_− − 1
        let mut e = vec![0.0; n];
        for v in e.iter_mut().take(xm).skip(xp) {
            *v = q;
        }
        let modes = (0..n)
            .map(|jk| {
                let k = 2.0 * PI * jk as f64 / n as f64;
                let u = su2_band(&lw.matrix(k), -1).1;
                let s = 1.0 / (n as f64).sqrt();
                (0..n).map(|x| {
                    let ph = C64::from_polar(s, k * x as f64);
                    [u[0] * ph, u[1] * ph]
                }).collect()
            })
            .collect();
        Ring { n, lw, q, xp, xm, d, rho_ext, e, a: vec![0.0; n], modes, tmp: vec![[cx(0.0, 0.0); 2]; n], j: vec![0.0; n], t: 0, backreaction: true }
    }

    /// Un paso de un modo con las fases de enlace; acumula la corriente en `j` si `cur`.
    fn step_mode(lw: &Lazy, psi: &mut [[C64; 2]], tmp: &mut [[C64; 2]], ph: &[C64], j: &mut [f64], cur: bool) {
        let n = psi.len();
        // B: φ(x) = Π_b·ψ(x) + e^{−iqA(x)}·Π_b̄·ψ(x + 1)
        for x in 0..n {
            let xp = if x + 1 == n { 0 } else { x + 1 };
            let p = psi[x];
            let pb = braket(&lw.b, &p);
            let stay = [lw.b[0] * pb, lw.b[1] * pb];
            let r = psi[xp];
            let rb = braket(&lw.b, &r);
            let hop = [r[0] - lw.b[0] * rb, r[1] - lw.b[1] * rb];
            if cur {
                j[x] -= hop[0].norm_sqr() + hop[1].norm_sqr();
            }
            let e = ph[x].conj();
            tmp[x] = [stay[0] + hop[0] * e, stay[1] + hop[1] * e];
        }
        // A: χ(x) = Π_ā·φ(x) + e^{iqA(x−1)}·Π_a·φ(x − 1); luego W
        for x in 0..n {
            let xm = if x == 0 { n - 1 } else { x - 1 };
            let p = tmp[x];
            let pa = braket(&lw.a, &p);
            let stay = [p[0] - lw.a[0] * pa, p[1] - lw.a[1] * pa];
            let pm = braket(&lw.a, &tmp[xm]) * ph[xm];
            let chi = [stay[0] + lw.a[0] * pm, stay[1] + lw.a[1] * pm];
            psi[x] = [lw.w[0][0] * chi[0] + lw.w[0][1] * chi[1], lw.w[1][0] * chi[0] + lw.w[1][1] * chi[1]];
            if cur {
                j[x] += pa.norm_sqr();
            }
        }
    }

    pub fn step(&mut self) {
        let n = self.n;
        let ph: Vec<C64> = self.a.iter().map(|a| C64::from_polar(1.0, self.q * a)).collect();
        for v in self.j.iter_mut() {
            *v = 0.0;
        }
        let mut tmp = std::mem::take(&mut self.tmp);
        for psi in self.modes.iter_mut() {
            Self::step_mode(&self.lw, psi, &mut tmp, &ph, &mut self.j, true);
        }
        self.tmp = tmp;
        if self.backreaction {
            for x in 0..n {
                self.e[x] -= self.q * self.j[x];
                self.a[x] -= self.e[x];
            }
        }
        self.t += 1;
    }

    pub fn density(&self) -> Vec<f64> {
        let mut d = vec![0.0; self.n];
        for psi in &self.modes {
            for (x, p) in psi.iter().enumerate() {
                d[x] += p[0].norm_sqr() + p[1].norm_sqr();
            }
        }
        d
    }

    /// [campo medio dentro de la cuerda, carga inducida a menos de 15 celdas de +q y de −q, residuo de
    /// Gauss, energía del campo ½·ΣE²].
    pub fn observables(&self) -> [f64; 5] {
        let n = self.n;
        let dens = self.density();
        let (lo, hi) = if self.d >= 24 { (self.xp + 10, self.xm - 10) } else { (self.xp, self.xm) };
        let inside = self.e[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
        let near = |c: usize| -> f64 {
            let mut s = 0.0;
            for dx in -15i64..=15 {
                let x = (c as i64 + dx).rem_euclid(n as i64) as usize;
                s += dens[x] - 1.0;
            }
            self.q * s
        };
        let mut gauss = 0.0f64;
        for x in 0..n {
            let xm = if x == 0 { n - 1 } else { x - 1 };
            gauss = gauss.max((self.e[x] - self.e[xm] - self.rho_ext[x] - self.q * (dens[x] - 1.0)).abs());
        }
        let energy = 0.5 * self.e.iter().map(|v| v * v).sum::<f64>();
        [inside, near(self.xp), near(self.xm), gauss, energy]
    }

    /// Pares y energía de los fermiones: se diagonaliza el operador de un paso de 2N×2N con los
    /// enlaces del momento; pares = Σ_modos |P₊ψ|², energía = Σ_modos ⟨ψ|ε|ψ⟩ (ε ∈ (−π, π]).
    pub fn pairs_energy(&self) -> [f64; 2] {
        let n = self.n;
        let nn = 2 * n;
        let ph: Vec<C64> = self.a.iter().map(|a| C64::from_polar(1.0, self.q * a)).collect();
        let mut mat = vec![cx(0.0, 0.0); nn * nn];
        let mut tmp = vec![[cx(0.0, 0.0); 2]; n];
        let mut jj = vec![0.0; n];
        for col in 0..nn {
            let mut v = vec![[cx(0.0, 0.0); 2]; n];
            v[col / 2][col % 2] = cx(1.0, 0.0);
            Self::step_mode(&self.lw, &mut v, &mut tmp, &ph, &mut jj, false);
            for x in 0..n {
                for s in 0..2 {
                    mat[(2 * x + s) * nn + col] = v[x][s];
                }
            }
        }
        let e = eig(mat, nn);
        // base ortonormal de cada subespacio (Gram–Schmidt por si hay degeneraciones)
        let mut order: Vec<usize> = (0..nn).collect();
        let eps: Vec<f64> = e.values.iter().map(|l| -l.arg()).collect();
        order.sort_by(|&i, &j| eps[i].total_cmp(&eps[j]));
        let mut basis: Vec<Vec<C64>> = Vec::with_capacity(nn);
        for &i in &order {
            let mut v = e.vectors[i].clone();
            for b in &basis {
                let ov: C64 = b.iter().zip(&v).map(|(x, y)| x.conj() * y).sum();
                for (vv, bb) in v.iter_mut().zip(b) {
                    *vv -= ov * bb;
                }
            }
            let nr: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
            v.iter_mut().for_each(|z| *z /= nr);
            basis.push(v);
        }
        let (mut pairs, mut energy) = (0.0, 0.0);
        for psi in &self.modes {
            for (bi, &i) in order.iter().enumerate() {
                let b = &basis[bi];
                let mut ov = cx(0.0, 0.0);
                for x in 0..n {
                    ov += b[2 * x].conj() * psi[x][0] + b[2 * x + 1].conj() * psi[x][1];
                }
                let w = ov.norm_sqr();
                energy += eps[i] * w;
                if eps[i] > 0.0 {
                    pairs += w;
                }
            }
        }
        [pairs, energy]
    }
}

// ------------------------------------------------------------------ 3D por nodos

/// Los 64 nodos del operador de dos pasos: node = 16·i_x + 4·i_y + i_z con k_a ∈ {0, π/2, π, −π/2}.
pub fn node_k(node: u32) -> [f64; 3] {
    let v = [0.0, 0.5 * PI, PI, -0.5 * PI];
    let n = node.min(63) as usize;
    [v[n / 16], v[(n / 4) % 4], v[n % 4]]
}

/// Nodo de cuasi-energía π (un número impar de componentes ±π/2).
pub fn node_odd(node: u32) -> bool {
    let n = node.min(63) as usize;
    [n / 16, (n / 4) % 4, n % 4].iter().filter(|&&i| i == 1 || i == 3).count() % 2 == 1
}

/// n direcciones casi uniformes (esfera de Fibonacci).
pub fn fibonacci_dirs(n: usize) -> Vec<[f64; 3]> {
    fibonacci_sphere(n)
}

/// Secuencias de subpasos de la caminata alterna en 2 componentes de espín: matrices de cambio de
/// base de cada eje y signos de los saltos (R: [+1, −1]; L: [−1, +1], el salto es exp(+iσ·k)).
struct Seq2 {
    order: [[usize; 3]; 2],
    mat_in: [[[[C64; 2]; 2]; 3]; 2],
    mat_out: [[[C64; 2]; 2]; 2],
    sign: [i32; 2],
}

fn m2_mul(a: &M2, b: &M2) -> M2 {
    let mut r = [[cx(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    r
}

fn m2_dag(a: &M2) -> M2 {
    [[a[0][0].conj(), a[1][0].conj()], [a[0][1].conj(), a[1][1].conj()]]
}

impl Seq2 {
    fn new(right: bool) -> Self {
        let order = [[0, 1, 2], [2, 1, 0]];
        let id: M2 = [[cx(1.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(1.0, 0.0)]];
        let mut mat_in = [[id; 3]; 2];
        let mut mat_out = [id; 2];
        for p in 0..2 {
            let mut prev = id;
            for (s, &a) in order[p].iter().enumerate() {
                let v = pauli_eigenbasis(a);
                mat_in[p][s] = m2_mul(&m2_dag(&v), &prev);
                prev = v;
            }
            mat_out[p] = prev;
        }
        Seq2 { order, mat_in, mat_out, sign: if right { [1, -1] } else { [-1, 1] } }
    }
}

/// Un proceso homogéneo en 3D: el paso es ψ ← post·T(t)·pre·ψ en `dim` componentes, con pre y post
/// fijos por k y paridad, y T(t) la parte que depende del tiempo (masa, o el salto en z con el campo).
#[derive(Clone, Copy)]
pub enum Process3 {
    /// Higgs → f f̄: Dirac de 4 componentes, m(t) = m_f + δm·cos(m_H·t).
    Higgs { mf: f64, mh: f64, dm: f64 },
    /// W¹ → ℓν: doblete L sin masa (espín ⊗ isospín), U_z = exp(−iθ(t)·T¹), θ = θ₀·cos(M·t).
    W { mw: f64, theta0: f64 },
    /// Z → f f̄: un Weyl (L o R) sin masa con la fase abeliana φ(t) = φ₀·cos(M·t) en los enlaces z.
    Z { right: bool, mz: f64, phi0: f64 },
}

impl Process3 {
    fn dim(&self) -> usize {
        match self {
            Process3::Higgs { .. } | Process3::W { .. } => 4,
            Process3::Z { .. } => 2,
        }
    }
}

/// pre/post de la paridad p en k: Higgs, todos los saltos (pre) y nada (post); W y Z, lo anterior al
/// salto en z (incluida la entrada en su base) y lo posterior (con la vuelta a la base estándar).
fn split_matrices(proc_: &Process3, sw: &SpinorWalk, s2: &Seq2, p: usize, k: [f64; 3]) -> (M4, M4, [C64; 2]) {
    let hop = |kk: f64, sg: i32| -> C64 { C64::from_polar(1.0, -(sg as f64) * kk) };
    match proc_ {
        Process3::Higgs { .. } => {
            let mut u = m4_identity();
            for (s, sub) in sw.seq[p].iter().enumerate() {
                u = m4_mul(&sw.mat_in[p][s], &u);
                let mut d = m4_identity();
                for c in 0..4 {
                    d[c][c] = hop(k[sub.axis], sub.sign[c] as i32);
                }
                u = m4_mul(&d, &u);
            }
            (m4_mul(&sw.mat_out[p], &u), m4_identity(), [cx(1.0, 0.0); 2])
        }
        Process3::W { .. } | Process3::Z { .. } => {
            let iso = if let Process3::W { .. } = proc_ { 2 } else { 1 };
            // matrices de espín 2×2 antes y después del salto en z
            let id: M2 = [[cx(1.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(1.0, 0.0)]];
            let (mut pre, mut post) = (id, id);
            let mut after = false;
            for (s, &a) in s2.order[p].iter().enumerate() {
                let into = s2.mat_in[p][s];
                if a == 2 {
                    pre = m2_mul(&into, &pre);
                    after = true;
                    continue;
                }
                let d: M2 = [[hop(k[a], s2.sign[0]), cx(0.0, 0.0)], [cx(0.0, 0.0), hop(k[a], s2.sign[1])]];
                if after {
                    post = m2_mul(&d, &m2_mul(&into, &post));
                } else {
                    pre = m2_mul(&d, &m2_mul(&into, &pre));
                }
            }
            post = m2_mul(&s2.mat_out[p], &post);
            let zph = [hop(k[2], s2.sign[0]), hop(k[2], s2.sign[1])];
            // espín ⊗ isospín (índice s·iso + i)
            let mut pre4 = m4_zero();
            let mut post4 = m4_zero();
            for s in 0..2 {
                for t in 0..2 {
                    for i in 0..iso {
                        pre4[s * iso + i][t * iso + i] = pre[s][t];
                        post4[s * iso + i][t * iso + i] = post[s][t];
                    }
                }
            }
            (pre4, post4, zph)
        }
    }
}

/// Integra un nodo: devuelve los pares por volumen y por paso, Σ_q w·n(q)/T.
pub fn node_rate(proc_: Process3, node: u32, qmin: f64, qmax: f64, dq: f64, ndirs: usize, steps: u32) -> f64 {
    if !qmin.is_finite() || !qmax.is_finite() || !dq.is_finite() || dq <= 0.0 || qmax < qmin {
        return 0.0;
    }
    let dim = proc_.dim();
    let mut sw = SpinorWalk::new();
    sw.ns = 4;
    sw.weyl = false;
    sw.fixed_order = false;
    sw.rebuild();
    let (s2, zsign) = match proc_ {
        Process3::W { .. } => (Seq2::new(false), [-1, 1]),
        Process3::Z { right, .. } => {
            let s = Seq2::new(right);
            let sg = s.sign;
            (s, sg)
        }
        _ => (Seq2::new(true), [1, -1]),
    };
    let k0 = node_k(node);
    let odd = node_odd(node);
    let ndirs = ndirs.clamp(1, 256);
    let dirs = fibonacci_dirs(ndirs);
    let dq = dq.max(1e-5);
    let nr = (((qmax - qmin) / dq).round().max(1.0) as usize).min(4096);
    let free = match proc_ {
        Process3::Higgs { mf, mh, .. } => Process3::Higgs { mf, mh, dm: 0.0 },
        Process3::W { mw, .. } => Process3::W { mw, theta0: 0.0 },
        Process3::Z { right, mz, .. } => Process3::Z { right, mz, phi0: 0.0 },
    };
    let steps = steps.min(200_000);
    let steps = steps.saturating_add(steps & 1).max(2);
    let mut total = 0.0;
    for dir in &dirs {
        for ir in 0..nr {
            let q = qmin + (ir as f64 + 0.5) * dq;
            let k = [k0[0] + q * dir[0], k0[1] + q * dir[1], k0[2] + q * dir[2]];
            let mut pre = [m4_identity(); 2];
            let mut post = [m4_identity(); 2];
            let mut zph = [[cx(1.0, 0.0); 2]; 2];
            for p in 0..2 {
                let (a, b, z) = split_matrices(&proc_, &sw, &s2, p, k);
                pre[p] = a;
                post[p] = b;
                zph[p] = z;
            }
            // operador libre de dos pasos y su banda negativa (relativa al nodo)
            let step_free = |p: usize, v: &mut [C64; 4]| {
                *v = mat4_apply(&pre[p], v, dim);
                apply_t_field(&free, 0, &zph[p], v, &zsign);
                *v = mat4_apply(&post[p], v, dim);
            };
            let mut w2 = m4_zero();
            for col in 0..dim {
                let mut v = [cx(0.0, 0.0); 4];
                v[col] = cx(1.0, 0.0);
                step_free(0, &mut v);
                step_free(1, &mut v);
                for r in 0..dim {
                    w2[r][col] = v[r];
                }
            }
            let band_neg = if odd { 1 } else { -1 };
            let pm = match band_projector_s(&w2, dim, band_neg) {
                Some(p) => p,
                None => crate::spinor::band_projector_eig(&w2, dim, band_neg),
            };
            // base ortonormal del mar (columnas de P₋)
            let mut occ: Vec<[C64; 4]> = Vec::new();
            for col in 0..dim {
                let mut v = [cx(0.0, 0.0); 4];
                for r in 0..dim {
                    v[r] = pm[r][col];
                }
                for o in &occ {
                    let ov: C64 = (0..dim).map(|r| o[r].conj() * v[r]).sum();
                    for r in 0..dim {
                        v[r] -= ov * o[r];
                    }
                }
                let nr2: f64 = (0..dim).map(|r| v[r].norm_sqr()).sum();
                if nr2 > 1e-6 && occ.len() < dim / 2 {
                    let s = 1.0 / nr2.sqrt();
                    for r in 0..dim {
                        v[r] *= s;
                    }
                    occ.push(v);
                }
            }
            for t in 0..steps {
                let p = (t & 1) as usize;
                for v in occ.iter_mut() {
                    *v = mat4_apply(&pre[p], v, dim);
                    apply_t_field(&proc_, t, &zph[p], v, &zsign);
                    *v = mat4_apply(&post[p], v, dim);
                }
            }
            // n = Σ |P₊ψ|² = Σ (1 − |P₋ψ|²)
            let mut nq = 0.0;
            for v in &occ {
                let pv = mat4_apply(&pm, v, dim);
                let inside: f64 = (0..dim).map(|r| pv[r].norm_sqr()).sum();
                nq += 1.0 - inside;
            }
            let w = 4.0 * PI / ndirs as f64 * q * q * dq / (2.0 * PI).powi(3);
            total += w * nq;
        }
    }
    total / steps as f64
}

#[inline(always)]
fn mat4_apply(m: &M4, v: &[C64; 4], dim: usize) -> [C64; 4] {
    let mut r = [cx(0.0, 0.0); 4];
    for i in 0..dim {
        let mut s = cx(0.0, 0.0);
        for j in 0..dim {
            s += m[i][j] * v[j];
        }
        r[i] = s;
    }
    r
}

/// T(t) con el campo: masa (Higgs), U(t) del W o la fase del Z en el salto en z; `zsign` dice qué
/// componente de espín avanza en z (+1) y cuál retrocede (−1).
#[inline(always)]
fn apply_t_field(proc_: &Process3, t: u32, zph: &[C64; 2], v: &mut [C64; 4], zsign: &[i32; 2]) {
    match *proc_ {
        Process3::Higgs { mf, mh, dm } => {
            let m = mf + dm * (mh * t as f64).cos();
            let (e1, e2) = (C64::from_polar(1.0, -m), C64::from_polar(1.0, m));
            v[0] *= e1;
            v[1] *= e1;
            v[2] *= e2;
            v[3] *= e2;
        }
        Process3::W { mw, theta0 } => {
            let th = theta0 * (mw * t as f64).cos();
            let (sn, cs) = (0.5 * th).sin_cos();
            for s in 0..2 {
                // U = cos(θ/2) − i·sin(θ/2)·σ_x a lo que avanza; U† a lo que retrocede
                let sg = if zsign[s] > 0 { -1.0 } else { 1.0 };
                let (a, b) = (v[2 * s], v[2 * s + 1]);
                let na = a * cs + b * cx(0.0, sg * sn);
                let nb = b * cs + a * cx(0.0, sg * sn);
                v[2 * s] = na * zph[s];
                v[2 * s + 1] = nb * zph[s];
            }
        }
        Process3::Z { mz, phi0, .. } => {
            let ph = phi0 * (mz * t as f64).cos();
            for s in 0..2 {
                v[s] *= zph[s] * C64::from_polar(1.0, zsign[s] as f64 * ph);
            }
        }
    }
}

// ------------------------------------------------------------------ fracciones del Z

/// Fracciones a nivel árbol con s² = sin²θ_W: Γ_f ∝ Σ_{L,R}(T³ − s²Q)², ×3 en los quarks, tres
/// generaciones sin el top, fermiones sin masa. [invisible, hadrones, leptones cargados,
/// Γ(νν̄)/Γ(ℓ⁺ℓ⁻)].
pub fn z_branching(s2: f64) -> [f64; 4] {
    let g = |t3: f64, q: f64| (t3 - s2 * q).powi(2) + (s2 * q).powi(2);
    let nu = 0.25;
    let lep = g(-0.5, -1.0);
    let up = 3.0 * g(0.5, 2.0 / 3.0);
    let dn = 3.0 * g(-0.5, -1.0 / 3.0);
    let inv = 3.0 * nu;
    let had = 2.0 * up + 3.0 * dn;
    let chl = 3.0 * lep;
    let tot = inv + had + chl;
    [inv / tot, had / tot, chl / tot, nu / lep]
}

// ------------------------------------------------------------------ estado del laboratorio

pub struct PairLab {
    pub ring: Option<Ring>,
    pub out: Vec<f64>,
    pub profiles: Vec<f64>,
    pub sea_run: Vec<f64>,
    pub max_occ: f64,
    pub z_branch: Vec<f64>,
}

impl PairLab {
    pub fn new() -> Self {
        PairLab { ring: None, out: vec![0.0; 5], profiles: Vec::new(), sea_run: Vec::new(), max_occ: 0.0, z_branch: vec![0.0; 4] }
    }
}

impl QuantumSpacetimeUniverse {
    /// Copia los observables y los perfiles del anillo a los búferes que lee JS.
    fn string_breaking_refresh(&mut self) {
        if let Some(r) = self.pairs.ring.as_ref() {
            self.pairs.out = r.observables().to_vec();
            let d = r.density();
            let mut pr = r.e.clone();
            pr.extend(d.iter().map(|v| v - 1.0));
            self.pairs.profiles = pr;
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.23: Schwinger con el mar lleno en 1D (campo uniforme, `modes` valores de k). Tasa de pares
    /// por celda y paso: la pendiente del número de pares entre los barridos de k de 1,5 y 3,0, es
    /// decir, entre q·E·t = 1,5 y 3,0 (cada modo cruza cada hueco una vez como mucho, así que no hay
    /// interferencia de Stückelberg entre cruces, y la resta quita los transitorios de los modos que
    /// están cruzando al final). theta = 0: caminata convencional; theta > m/2: la de Gupta y Short.
    pub fn pair_rate_schwinger(&mut self, m: f32, q_e: f32, modes: u32, theta: f32) -> f64 {
        if !q_e.is_finite() || (q_e as f64).abs() < 1e-6 {
            return 0.0;
        }
        let mf = clamp_finite(m as f64, -10.0, 10.0, 0.1);
        let th = clamp_finite(theta as f64, -PI, PI, 0.0);
        let lw = Lazy::new(mf, th);
        let qe = (q_e as f64).abs();
        let l = (modes as usize).clamp(8, 4096);
        let mut sea = sea_1d(&lw, l);
        let t2 = ((3.0 / qe).round() as u32).min(200_000);
        let t1 = ((1.5 / qe).round() as u32).min(t2.saturating_sub(1));
        if t2 <= t1 {
            return 0.0;
        }
        let mut qa = 0.0f64;
        let mut n1 = 0.0;
        let mut maxo = 0.0f64;
        let count = |sea: &[[C64; 2]], qa: f64, maxo: &mut f64| -> f64 {
            let mut s = 0.0;
            for (j, p) in sea.iter().enumerate() {
                let k = 2.0 * PI * j as f64 / l as f64;
                let o = occupation(&lw, p, k - qa);
                *maxo = maxo.max(o);
                s += o;
            }
            s / l as f64
        };
        for t in 0..t2 {
            for (j, p) in sea.iter_mut().enumerate() {
                let k = 2.0 * PI * j as f64 / l as f64;
                lw.step(p, k - qa);
            }
            // A ← A − E: q·A baja q·E por paso (k − q·A barre la zona a ritmo q·E)
            qa -= qe;
            if t + 1 == t1 {
                n1 = count(&sea, qa, &mut maxo);
            }
        }
        let n2 = count(&sea, qa, &mut maxo);
        self.pairs.max_occ = maxo;
        (n2 - n1) / (t2 - t1) as f64
    }

    /// Ocupación máxima de un modo en la última `pair_rate_schwinger` (nunca pasa de 1: Pauli).
    pub fn pair_max_occupation(&self) -> f64 {
        self.pairs.max_occ
    }

    /// Velocidad de la banda en el hueco: ajuste de ε² = m² + c²·k² cerca de k = 0.
    pub fn lazy_walk_c_eff(&self, m: f32, theta: f32) -> f64 {
        lazy_c_eff(clamp_finite(m as f64, -10.0, 10.0, 0.1), clamp_finite(theta as f64, -PI, PI, 0.0))
    }

    /// §2.23: mar lleno en 1D (4096 modos) con un campo homogéneo E(0) = e0: devuelve j(t) (corriente
    /// por celda) y E(t), 2·steps valores. Con `backreaction`, E ← E − q·j; siempre A ← A − E.
    pub fn homogeneous_sea_run(&mut self, m: f32, theta: f32, q: f32, e0: f32, backreaction: bool, steps: u32) -> *const f64 {
        let lw = Lazy::new(clamp_finite(m as f64, -10.0, 10.0, 0.1), clamp_finite(theta as f64, -PI, PI, 0.0));
        let l = 4096usize;
        let mut sea = sea_1d(&lw, l);
        let q = clamp_finite(q as f64, -10.0, 10.0, 0.1);
        let mut e = clamp_finite(e0 as f64, -10.0, 10.0, 0.0);
        let mut a = 0.0f64;
        let steps = (steps as usize).min(200_000);
        let mut out = vec![0.0; 2 * steps];
        for t in 0..steps {
            let mut j = 0.0;
            for (jk, p) in sea.iter_mut().enumerate() {
                let k = 2.0 * PI * jk as f64 / l as f64;
                let (r, lft) = lw.step(p, k - q * a);
                j += r - lft;
            }
            j /= l as f64;
            if backreaction {
                e -= q * j;
            }
            a -= e;
            out[t] = j;
            out[steps + t] = e;
        }
        self.pairs.sea_run = out;
        self.pairs.sea_run.as_ptr()
    }

    /// §3.21: el mar de la caminata convencional en un anillo de `cells` celdas con fases de enlace
    /// aleatorias en [−π, π) que cambian cada paso: máximo de |n − 1| y de |j| en `steps` pasos.
    pub fn sea_inertness(&mut self, cells: u32, steps: u32, seed: u64) -> f64 {
        let n = (cells as usize).clamp(4, 512) & !1;
        let steps = steps.min(200_000);
        let mut ring = Ring::new(n, 0.1, 0.0, 1.0, n / 2);
        ring.backreaction = false;
        let mut rng = SplitMix64::new(seed);
        let mut worst = 0.0f64;
        for _ in 0..steps {
            for v in ring.a.iter_mut() {
                *v = PI * (2.0 * rng.next_f64() - 1.0);
            }
            ring.step();
            let d = ring.density();
            for x in 0..n {
                worst = worst.max((d[x] - 1.0).abs()).max(ring.j[x].abs());
            }
        }
        worst
    }

    /// §2.23: anillo 1D con el mar lleno (cells modos de 2 componentes), cargas ±q a distancia d,
    /// E = q entre ellas, y la reacción E ← E − q·j, A ← A − E en cada paso. Vacío libre al empezar.
    pub fn string_breaking_start(&mut self, cells: u32, m: f32, theta: f32, q: f32, d: u32) {
        let n = (cells as usize).clamp(8, 512) & !1;
        let mf = clamp_finite(m as f64, -10.0, 10.0, 0.1);
        let th = clamp_finite(theta as f64, -PI, PI, 0.0);
        let qf = clamp_finite(q as f64, -10.0, 10.0, 0.2);
        self.pairs.ring = Some(Ring::new(n, mf, th, qf, d as usize));
        self.string_breaking_refresh();
    }

    /// Avanza y devuelve [campo medio dentro, carga inducida junto a +q y −q, residuo de Gauss,
    /// energía del campo].
    pub fn string_breaking_step(&mut self, steps: u32) -> *const f64 {
        let steps = steps.min(200_000);
        if let Some(r) = self.pairs.ring.as_mut() {
            for _ in 0..steps {
                r.step();
            }
        }
        self.string_breaking_refresh();
        self.pairs.out.as_ptr()
    }

    /// E(x) y n(x) − 1 del anillo (2N valores) para dibujarlos.
    pub fn string_breaking_profiles_ptr(&self) -> *const f64 {
        self.pairs.profiles.as_ptr()
    }

    /// Celdas del anillo y posiciones de las cargas: [N, x₊, x₋, paso].
    pub fn string_breaking_info(&self) -> Vec<f64> {
        match &self.pairs.ring {
            Some(r) => vec![r.n as f64, r.xp as f64, r.xm as f64, r.t as f64],
            None => vec![0.0; 4],
        }
    }

    /// Pares y energía de los fermiones (diagonalizando el operador de 2N×2N con los enlaces del
    /// momento): [pares, Σ ε de los modos ocupados].
    pub fn string_breaking_pairs(&self) -> Vec<f64> {
        match &self.pairs.ring {
            Some(r) => r.pairs_energy().to_vec(),
            None => vec![0.0; 2],
        }
    }

    /// §2.23, 3D homogéneo: kind 0 = Higgs → f f̄ (Dirac, m(t) = m_f + amplitude·cos(m_boson·t)),
    /// 1 = W¹ → ℓν (doblete L sin masa, θ(t) = amplitude·cos(m_boson·t)); node 0–63 (16·i_x + 4·i_y +
    /// i_z con k_a ∈ {0, π/2, π, −π/2}); 24 direcciones, |q| ∈ [0,01; 0,40] con paso 0,0004. Pares
    /// por celda y paso.
    pub fn pair_rate_decay(&mut self, kind: u32, node: u32, m_f: f32, m_boson: f32, amplitude: f32, steps: u32) -> f64 {
        let mf = clamp_finite(m_f as f64, -10.0, 10.0, 0.1);
        let mb = clamp_finite(m_boson as f64, -10.0, 10.0, 0.4);
        let amp = clamp_finite(amplitude as f64, -10.0, 10.0, 0.0);
        let p = if kind == 0 { Process3::Higgs { mf, mh: mb, dm: amp } } else { Process3::W { mw: mb, theta0: amp } };
        node_rate(p, node, 0.01, 0.40, 0.0004, 24, steps)
    }

    /// La misma integral en una ventana de |q| elegida (para una ventana estrecha alrededor de la
    /// resonancia, o un cálculo más rápido en el navegador).
    #[allow(clippy::too_many_arguments)]
    pub fn pair_rate_decay_window(&mut self, kind: u32, node: u32, m_f: f32, m_boson: f32, amplitude: f32, steps: u32, qmin: f32, qmax: f32, dq: f32) -> f64 {
        let mf = clamp_finite(m_f as f64, -10.0, 10.0, 0.1);
        let mb = clamp_finite(m_boson as f64, -10.0, 10.0, 0.4);
        let amp = clamp_finite(amplitude as f64, -10.0, 10.0, 0.0);
        let p = match kind {
            0 => Process3::Higgs { mf, mh: mb, dm: amp },
            1 => Process3::W { mw: mb, theta0: amp },
            2 => Process3::Z { right: false, mz: mb, phi0: amp },
            _ => Process3::Z { right: true, mz: mb, phi0: amp },
        };
        node_rate(p, node, qmin as f64, qmax as f64, dq as f64, 24, steps)
    }

    /// §2.23: Weyl sin masa (L o R) con una fase abeliana homogénea φ(t) = φ₀·cos(M·t) en los enlaces
    /// z (la φ_f del Z); pares por celda y paso en el nodo.
    pub fn pair_rate_z(&mut self, node: u32, right_handed: bool, m_z: f32, phase_amplitude: f32, steps: u32) -> f64 {
        let mz = clamp_finite(m_z as f64, -10.0, 10.0, 0.4);
        let phi0 = clamp_finite(phase_amplitude as f64, -10.0, 10.0, 0.0);
        node_rate(Process3::Z { right: right_handed, mz, phi0 }, node, 0.01, 0.40, 0.0004, 24, steps)
    }

    /// Fracciones del Z a nivel árbol: [invisible, hadrones, leptones cargados, Γ(νν̄)/Γ(ℓ⁺ℓ⁻)].
    pub fn z_branching_ptr(&mut self, s2w: f32) -> *const f64 {
        self.pairs.z_branch = z_branching(clamp_finite(s2w as f64, 0.0, 1.0, 0.231)).to_vec();
        self.pairs.z_branch.as_ptr()
    }

    /// §2.23: quarks en un campo de color constante en 1D, SU(3) completo: U = exp(−i·θ(t)·n·T) con
    /// θ = g·E·t, n de 8 componentes (se normaliza), la caminata de Gupta y Short (m, theta) y el mar
    /// lleno de cada color (`modes` valores de k). Tasa de pares por celda y paso: la pendiente entre
    /// los barridos de k de 1,5 y 3,0 del canal de mayor |λ| (|λ|·g·E·t = 1,5 y 3,0).
    pub fn pair_rate_colour_schwinger(&mut self, n: &[f32], g_e: f32, m: f32, theta: f32, modes: u32) -> f64 {
        if !g_e.is_finite() || (g_e as f64).abs() < 1e-6 {
            return 0.0;
        }
        let alg = Algebra::new(3);
        let mut nv = [0.0f64; 8];
        for (i, v) in n.iter().take(8).enumerate() {
            nv[i] = if v.is_finite() { *v as f64 } else { 0.0 };
        }
        let nn = nv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nn < 1e-12 {
            return 0.0;
        }
        nv.iter_mut().for_each(|x| *x /= nn);
        // autovalores y autovectores de n·T (para medir; la evolución usa el SU(3) completo)
        let nt = alg.combine(&nv);
        let mut a = vec![cx(0.0, 0.0); 9];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] = nt.get(i, j);
            }
        }
        let e = eig(a, 3);
        let lam: Vec<f64> = e.values.iter().map(|l| l.re).collect();
        let vecs = e.vectors.clone();
        let lmax = lam.iter().fold(0.0f64, |m, l| m.max(l.abs())).max(1e-6);
        let ge = (g_e as f64).abs();
        let lw = Lazy::new(clamp_finite(m as f64, -10.0, 10.0, 0.1), clamp_finite(theta as f64, -PI, PI, 0.0));
        let l = (modes as usize).clamp(8, 4096);
        // mar: banda negativa ⊗ cada color de la base estándar
        let base = sea_1d(&lw, l);
        let mut sea: Vec<[[C64; 3]; 2]> = Vec::with_capacity(3 * l);
        for b in &base {
            for c in 0..3 {
                let mut v = [[cx(0.0, 0.0); 3]; 2];
                v[0][c] = b[0];
                v[1][c] = b[1];
                sea.push(v);
            }
        }
        let t2 = ((3.0 / (ge * lmax)).round() as u32).min(200_000);
        let t1 = ((1.5 / (ge * lmax)).round() as u32).min(t2.saturating_sub(1));
        if t2 <= t1 {
            return 0.0;
        }
        let count = |sea: &[[[C64; 3]; 2]], th: f64| -> f64 {
            // en el canal i (autovector e_i, autovalor λ_i) el salto a la derecha lleva e^{−iλ_iθ}, que
            // es e^{iqA} con q·A = −λ_i·θ: κ = k − q·A = k + λ_i·θ
            let mut s = 0.0;
            for (idx, v) in sea.iter().enumerate() {
                let k = 2.0 * PI * (idx / 3) as f64 / l as f64;
                for i in 0..3 {
                    let ev = &vecs[i];
                    let proj = [ev[0].conj() * v[0][0] + ev[1].conj() * v[0][1] + ev[2].conj() * v[0][2], ev[0].conj() * v[1][0] + ev[1].conj() * v[1][1] + ev[2].conj() * v[1][2]];
                    s += occupation(&lw, &proj, k + lam[i] * th);
                }
            }
            s / l as f64
        };
        let mut n1 = 0.0;
        let mut th = 0.0f64;
        for t in 0..t2 {
            let mut thv = [0.0f64; 8];
            for c in 0..8 {
                thv[c] = th * nv[c];
            }
            let u = alg.exp_minus_i(&thv);
            for (idx, v) in sea.iter_mut().enumerate() {
                let k = 2.0 * PI * (idx / 3) as f64 / l as f64;
                // B: Π_b + Π_b̄ ⊗ e^{ik}·U†
                let mut bbar = [[cx(0.0, 0.0); 3]; 2];
                for c in 0..3 {
                    let pb = lw.b[0].conj() * v[0][c] + lw.b[1].conj() * v[1][c];
                    for s in 0..2 {
                        bbar[s][c] = v[s][c] - lw.b[s] * pb;
                    }
                }
                let eik = C64::from_polar(1.0, k);
                let mut phi = [[cx(0.0, 0.0); 3]; 2];
                for s in 0..2 {
                    let hop = u.apply_dag(&bbar[s]);
                    for c in 0..3 {
                        phi[s][c] = v[s][c] - bbar[s][c] + hop[c] * eik;
                    }
                }
                // A: Π_a ⊗ e^{−ik}·U + Π_ā
                let mut pa = [cx(0.0, 0.0); 3];
                for c in 0..3 {
                    pa[c] = lw.a[0].conj() * phi[0][c] + lw.a[1].conj() * phi[1][c];
                }
                let hop = u.apply(&pa);
                let emik = eik.conj();
                let mut chi = [[cx(0.0, 0.0); 3]; 2];
                for s in 0..2 {
                    for c in 0..3 {
                        chi[s][c] = phi[s][c] + lw.a[s] * (hop[c] * emik - pa[c]);
                    }
                }
                for c in 0..3 {
                    v[0][c] = lw.w[0][0] * chi[0][c] + lw.w[0][1] * chi[1][c];
                    v[1][c] = lw.w[1][0] * chi[0][c] + lw.w[1][1] * chi[1][c];
                }
            }
            th += ge;
            if t + 1 == t1 {
                n1 = count(&sea, th);
            }
        }
        let n2 = count(&sea, th);
        (n2 - n1) / (t2 - t1) as f64
    }

    /// Autovalores de n·T (n normalizado), ordenados.
    pub fn colour_channel_charges(&self, n: &[f32]) -> Vec<f64> {
        let alg = Algebra::new(3);
        let mut nv = [0.0f64; 8];
        for (i, v) in n.iter().take(8).enumerate() {
            nv[i] = if v.is_finite() { *v as f64 } else { 0.0 };
        }
        let nn = nv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        nv.iter_mut().for_each(|x| *x /= nn);
        let nt = alg.combine(&nv);
        let mut a = vec![cx(0.0, 0.0); 9];
        for i in 0..3 {
            for j in 0..3 {
                a[i * 3 + j] = nt.get(i, j);
            }
        }
        let mut l: Vec<f64> = eig(a, 3).values.iter().map(|v| v.re).collect();
        l.sort_by(f64::total_cmp);
        l
    }

    /// Fase 14 (Video 6 — polarización del vacío dependiente de la frecuencia Π(ω) = χ′(ω) + iχ″(ω)):
    /// somete el mar de Dirac 1D de Gupta y Short (m, theta) a un campo eléctrico oscilante débil
    /// E(t) = E₀·cos(ωt) con rampa suave, y extrae χ′(ω) (apantallamiento en fase con ∂E/∂t),
    /// χ″(ω) (absorción en fase con E, ligada por el teorema óptico a la tasa real de pares) y la
    /// tasa directa de creación de pares Γ_par(ω).
    /// Devuelve [chi_re, chi_im, gamma_pair, chi_im_from_pairs, omega_eff].
    #[allow(clippy::too_many_arguments)]
    pub fn vacuum_polarization_spectrum(&self, m: f32, theta: f32, q: f32, omega: f32, e0: f32, modes: u32, steps: u32) -> Vec<f64> {
        let mf = clamp_finite(m as f64, 0.02, 1.2, 0.15);
        let th = clamp_finite(theta as f64, -PI, PI, 0.0);
        let qf = clamp_finite((q as f64).abs(), 1e-6, 10.0, 0.1);
        let om = clamp_finite(omega as f64, 1e-3, PI - 0.05, 0.5);
        let amp_e = clamp_finite((e0 as f64).abs(), 1e-7, 10.0, 1e-3);
        let l = (modes as usize).clamp(32, 2048);
        let lw = Lazy::new(mf, th);
        let mut sea = sea_1d(&lw, l);
        let w_lat = 2.0 * (0.5 * om).sin();
        let a0 = amp_e / w_lat;
        let t_total = (steps as usize).clamp(120, 200_000);
        let t_ramp = (t_total / 3).max(40);
        // Ajuste por mínimos cuadrados de J(t) = c0 + ac·cos(ω t_mid) + as·sin(ω t_mid) tras la rampa
        let mut m_mat = [[0.0f64; 3]; 3];
        let mut rhs = [0.0f64; 3];
        let mut occ_start = 0.0;
        let mut occ_end = 0.0;
        let t_occ1 = t_ramp + (t_total - t_ramp) / 4;
        let t_occ2 = t_total - 1;
        let count_occ = |s: &[[C64; 2]], a_val: f64| -> f64 {
            let mut sum = 0.0;
            for (j, v) in s.iter().enumerate() {
                let k = 2.0 * PI * j as f64 / l as f64;
                sum += occupation(&lw, v, k - qf * a_val);
            }
            sum / l as f64
        };
        for t in 0..t_total {
            let s_env = if t < t_ramp {
                let u = 0.5 * PI * t as f64 / t_ramp as f64;
                u.sin().powi(2)
            } else {
                1.0
            };
            let a_t = -a0 * s_env * (om * t as f64).sin();
            let mut j_sum = 0.0;
            for (j, v) in sea.iter_mut().enumerate() {
                let k = 2.0 * PI * j as f64 / l as f64;
                let (r, lf) = lw.step(v, k - qf * a_t);
                j_sum += r - lf;
            }
            let j_ind = qf * j_sum / l as f64;
            if t == t_occ1 {
                occ_start = count_occ(&sea, a_t);
            }
            if t == t_occ2 {
                occ_end = count_occ(&sea, a_t);
            }
            if t >= t_ramp {
                // La corriente j(t) de Lazy::step usa A(t) = −a0·sin(ωt), en fase con t entero
                let tm = t as f64;
                let b = [1.0, (om * tm).cos(), (om * tm).sin()];
                for r in 0..3 {
                    rhs[r] += b[r] * j_ind;
                    for c in 0..3 {
                        m_mat[r][c] += b[r] * b[c];
                    }
                }
            }
        }
        // Resolvemos 3 × 3 (eliminación gaussiana simple)
        let mut aug = [[0.0f64; 4]; 3];
        for r in 0..3 {
            for c in 0..3 {
                aug[r][c] = m_mat[r][c];
            }
            aug[r][3] = rhs[r];
        }
        for p in 0..3 {
            let mut piv = p;
            for r in (p + 1)..3 {
                if aug[r][p].abs() > aug[piv][p].abs() {
                    piv = r;
                }
            }
            aug.swap(p, piv);
            let d = aug[p][p].abs().max(1e-30);
            let sgn = if aug[p][p] >= 0.0 { 1.0 } else { -1.0 };
            for r in (p + 1)..3 {
                let f = aug[r][p] / (sgn * d);
                for c in p..4 {
                    aug[r][c] -= f * aug[p][c];
                }
            }
        }
        let mut sol = [0.0f64; 3];
        for r in (0..3).rev() {
            let mut s = aug[r][3];
            for c in (r + 1)..3 {
                s -= aug[r][c] * sol[c];
            }
            sol[r] = if aug[r][r].abs() > 1e-15 { s / aug[r][r] } else { 0.0 };
        }
        let ac = sol[1];
        let as_ = sol[2];
        let chi_re = -as_ / (w_lat * amp_e);
        let chi_im = ac / (w_lat * amp_e);
        let dt_occ = (t_occ2.saturating_sub(t_occ1)).max(1) as f64;
        let gamma_pair = (occ_end - occ_start).max(0.0) / dt_occ;
        let chi_im_opt = 2.0 * gamma_pair / (amp_e * amp_e);
        vec![chi_re, chi_im, gamma_pair, chi_im_opt, om]
    }

    /// Fase 14 (Video 6 — relación de dispersión de Kramers–Kronig y efecto de un sesgo sistemático
    /// en la región del umbral 2m, análogo de KLOE frente a CMD-3 en a_μ^HVP):
    /// calcula en toda la zona de Brillouin de la caminata 1D de Gupta y Short (L = 1024):
    ///   - χ′_KK(0) = (2/π) ∫ dω χ″_red(ω)/ω (integral dispersiva exacta sobre los pares virtuales)
    ///   - χ′_cont(0) = q²·c_ef/(6π m²) (continuo de QED 1+1D)
    ///   - fracción de χ′_KK(0) procedente de la ventana de umbral ω ∈ [2m, 4m]
    ///   - desplazamiento relativo Δχ′/χ′ al escalar un +4 % el espectro absortivo en [2m, 4m]
    ///   - 2·m (umbral de creación de pares) y c_ef.
    /// Devuelve [chi0_kk, chi0_cont, frac_threshold, shift_4pct_rel, two_m, c_ef].
    pub fn kramers_kronig_vp_check(&self, m: f32, theta: f32, q: f32) -> Vec<f64> {
        let mf = clamp_finite(m as f64, 0.03, 1.0, 0.15);
        let th = clamp_finite(theta as f64, -PI, PI, 0.0);
        let qf = clamp_finite((q as f64).abs(), 1e-6, 10.0, 0.1);
        let lw = Lazy::new(mf, th);
        let c_ef = lazy_c_eff(mf, th);
        let chi_cont = qf * qf * c_ef / (6.0 * PI * mf * mf);
        let l = 1024usize;
        let dk_diff = 1e-5;
        let mut chi_kk = 0.0;
        let mut chi_thr = 0.0;
        let e0 = lw.omega(0.0);
        let two_m = 2.0 * e0;
        for j in 0..l {
            let k = 2.0 * PI * j as f64 / l as f64;
            let u0 = lw.matrix(k);
            let up = lw.matrix(k + dk_diff);
            let um = lw.matrix(k - dk_diff);
            let (eps, vp) = su2_band(&u0, 1);
            let (_, vm) = su2_band(&u0, -1);
            // U† · (∂U/∂k) · |u_−⟩
            let mut du_vm = [cx(0.0, 0.0); 2];
            for r in 0..2 {
                for c in 0..2 {
                    let du = (up[r][c] - um[r][c]) / (2.0 * dk_diff);
                    du_vm[r] += du * vm[c];
                }
            }
            let mut udag_du_vm = [cx(0.0, 0.0); 2];
            for r in 0..2 {
                for c in 0..2 {
                    udag_du_vm[r] += u0[c][r].conj() * du_vm[c];
                }
            }
            let mat2 = braket(&vp, &udag_du_vm).norm_sqr();
            let w_lat = (2.0 * eps.sin()).abs().max(1e-12);
            let w_pair = 2.0 * eps;
            // Contribución de este modo a (2/π) ∫ χ″(ω)/ω dω
            let contrib = (qf * qf / l as f64) * 2.0 * mat2 / (w_lat * w_lat * w_pair);
            chi_kk += contrib;
            if w_pair <= 2.0 * two_m {
                chi_thr += contrib;
            }
        }
        let frac_thr = if chi_kk > 0.0 { chi_thr / chi_kk } else { 0.0 };
        let shift_4pct = 0.04 * frac_thr;
        vec![chi_kk, chi_cont, frac_thr, shift_4pct, two_m, c_ef]
    }

    /// Fase 14 (Video 3 — LHCb/CMS 2026): tasas de producción de pares a nivel árbol de un bosón Z′
    /// hipotético en un nodo 3D del autómata: Z′ → b_L s̄_L + s_L b̄_L (con N_c = 3 colores y
    /// acoplamiento g_bs) frente a Z′ → μ_L⁺ μ_L⁻ (acoplamiento g_mu).
    /// Devuelve [tasa_bs_total, tasa_mu, razon_medida, razon_teorica_6_gbs2_gmu2].
    pub fn pair_rate_zprime(&self, node: u32, m_zp: f32, g_bs: f32, g_mu: f32, steps: u32) -> Vec<f64> {
        let mz = clamp_finite(m_zp as f64, 0.1, 1.0, 0.4);
        let gb = clamp_finite((g_bs as f64).abs(), 1e-6, 0.1, 1e-3);
        let gm = clamp_finite((g_mu as f64).abs(), 1e-6, 0.1, 1e-3);
        let st = steps.clamp(100, 2000);
        // Ventana estrecha alrededor de la resonancia |q| ≈ M_Z'/2 para precisión y rapidez
        let q0 = 0.5 * mz;
        let qmin = (q0 - 0.05).max(0.01);
        let qmax = (q0 + 0.05).min(0.6);
        let dq = 0.0004;
        // En Process3::W, U = exp(−i (θ₀/2) cos(M t) σ_x): con θ₀ = 2 g_bs reproduce la mezcla
        // b_L ↔ s_L con dos canales (b s̄ y s b̄), multiplicada por N_c = 3 colores.
        let r_bs_1col = node_rate(Process3::W { mw: mz, theta0: 2.0 * gb }, node, qmin, qmax, dq, 12, st);
        let r_bs = 3.0 * r_bs_1col;
        let r_mu = node_rate(Process3::Z { right: false, mz, phi0: gm }, node, qmin, qmax, dq, 12, st);
        let ratio = if r_mu > 0.0 { r_bs / r_mu } else { f64::NAN };
        let expected = 6.0 * (gb * gb) / (gm * gm);
        vec![r_bs, r_mu, ratio, expected]
    }
}
