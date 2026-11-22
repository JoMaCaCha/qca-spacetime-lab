//! Contraste con la física real (§2.30): la cinemática de las caminatas a escala de Planck.
//!
//! - Para cada caminata sin masa, |v_g| − 1 = A₁(n̂)·kΔx + A₂(n̂)·(kΔx)² + … en 2000 direcciones de
//!   Fibonacci (ajuste en kΔx = 0,01–0,04), y la asimetría ω(k) − ω(−k), que decide si hay violación de
//!   Lorentz lineal. Con las cotas de LHAASO (GRB 221009A), la cota sobre Δx/ℓ_P.
//! - La banda simétrica de la caminata de Grover con k ≫ m: su velocidad tiende a ‖n̂‖∞ (anisotropía de
//!   orden 1 que depende de k/m, no de kΔx).
//! - Si ω(k) ≠ ω(−k), el fotón que es su propia antipartícula es birrefringente (helicidades con ω(k) y
//!   ω(−k)): el campo real del fotón de Mlodinow y Brun y la cota de polarización de Kislat y Krawczynski.

use crate::analog::{coin_matrix, shift_phases};
use crate::linalg::{self, C};
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Energía de Planck (GeV) y cotas de LHAASO (subluminales) para n = 1 y n = 2.
const E_PLANCK: f64 = 1.22089e19;
const E_QG1: f64 = 1.0e20;
const E_QG2: f64 = 6.9e11;

type M2 = [[C; 2]; 2];

fn m2_mul(a: &M2, b: &M2) -> M2 {
    let mut r = [[C::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    r
}

/// exp(−iσ_a k) = cos k − i·sin k·σ_a.
fn t_axis(a: usize, k: f64) -> M2 {
    let (c, s) = (k.cos(), k.sin());
    let z = C::new(0.0, 0.0);
    let mi = C::new(0.0, -s);
    match a {
        0 => [[C::new(c, 0.0), mi], [mi, C::new(c, 0.0)]],
        1 => [[C::new(c, 0.0), C::new(-s, 0.0)], [C::new(s, 0.0), C::new(c, 0.0)]],
        _ => [[C::new(c, 0.0) + mi, z], [z, C::new(c, 0.0) - mi]],
    }
}

/// ω de F = a₀ − i·a·σ ∈ U(2) (normalizado a SU(2)): atan2(|a|, a₀), preciso para ω pequeño.
fn su2_omega(f: &M2) -> f64 {
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let ph = det.sqrt();
    let g = [[f[0][0] / ph, f[0][1] / ph], [f[1][0] / ph, f[1][1] / ph]];
    let a0 = 0.5 * (g[0][0] + g[1][1]).re;
    // a_x = (i/2)·tr(σ_x F), a_y = (i/2)·tr(σ_y F), a_z = (i/2)·tr(σ_z F)
    let i = C::new(0.0, 0.5);
    let ax = (i * (g[0][1] + g[1][0])).re;
    let ay = (i * (C::new(0.0, -1.0) * g[1][0] + C::new(0.0, 1.0) * g[0][1])).re;
    let az = (i * (g[0][0] - g[1][1])).re;
    (ax * ax + ay * ay + az * az).sqrt().atan2(a0)
}

fn proj(a: usize, s: f64) -> M2 {
    let half = |z: C| z * 0.5;
    let one = C::new(1.0, 0.0);
    let zero = C::new(0.0, 0.0);
    let sig: M2 = match a {
        0 => [[zero, one], [one, zero]],
        1 => [[zero, C::new(0.0, -1.0)], [C::new(0.0, 1.0), zero]],
        _ => [[one, zero], [zero, -one]],
    };
    [[half(one + sig[0][0] * s), half(sig[0][1] * s)], [half(sig[1][0] * s), half(one + sig[1][1] * s)]]
}

/// Bombeo U_j^± = P_j^±·e^{∓ik} + P_j^∓.
fn pump(a: usize, plus: bool, k: f64) -> M2 {
    let (pm, other) = if plus { (proj(a, 1.0), proj(a, -1.0)) } else { (proj(a, -1.0), proj(a, 1.0)) };
    let ph = if plus { C::new(k.cos(), -k.sin()) } else { C::new(k.cos(), k.sin()) };
    let mut r = other;
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] += pm[i][j] * ph;
        }
    }
    r
}

/// Cuasi-energía por paso de la banda positiva de cada caminata (k en unidades de 1/Δx).
pub(crate) fn walk_omega(walk: u32, k: [f64; 3]) -> f64 {
    match walk {
        0 => su2_omega(&m2_mul(&t_axis(2, k[2]), &m2_mul(&t_axis(1, k[1]), &t_axis(0, k[0])))),
        1 => {
            let f = m2_mul(&t_axis(2, k[2]), &m2_mul(&t_axis(1, k[1]), &t_axis(0, k[0])));
            let b = m2_mul(&t_axis(0, k[0]), &m2_mul(&t_axis(1, k[1]), &t_axis(2, k[2])));
            0.5 * su2_omega(&m2_mul(&b, &f))
        }
        2 => {
            let h = 0.5 * k[2];
            let seq = [pump(2, true, h), pump(1, true, k[1]), pump(2, false, h), pump(0, true, k[0]), pump(2, true, h), pump(1, false, k[1]), pump(2, false, h), pump(0, false, k[0])];
            let mut u: M2 = [[C::new(1.0, 0.0), C::new(0.0, 0.0)], [C::new(0.0, 0.0), C::new(1.0, 0.0)]];
            for p in seq.iter() {
                u = m2_mul(p, &u);
            }
            su2_omega(&u)
        }
        3 => {
            // fotón de red: salto de rana con κ = 1 y s = 2 subpasos
            let kk: f64 = k.iter().map(|q| 4.0 * (0.5 * q).sin().powi(2)).sum::<f64>().sqrt();
            let tau = 0.5;
            (2.0 / tau) * (0.5 * tau * kk).asin()
        }
        4 => {
            let s2: f64 = k.iter().map(|q| q.sin().powi(2)).sum();
            let w: f64 = k.iter().map(|q| 1.0 - q.cos()).sum();
            (s2 + w * w).sqrt()
        }
        _ => {
            // fotón de Mlodinow y Brun (2025), ec. (42)
            let (c, s) = (k.map(f64::cos), k.map(f64::sin));
            let x = (c[0] * c[1] + c[0] * c[2] + c[1] * c[2] + s[0] * s[1] * s[2] - 1.0) / 2.0;
            x.clamp(-1.0, 1.0).acos()
        }
    }
}

fn speed(walk: u32, k: [f64; 3]) -> f64 {
    let h = 1e-6;
    let mut g = [0.0; 3];
    for a in 0..3 {
        let (mut kp, mut km) = (k, k);
        kp[a] += h;
        km[a] -= h;
        g[a] = (walk_omega(walk, kp) - walk_omega(walk, km)) / (2.0 * h);
    }
    (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt()
}

fn fibonacci(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f64 + 0.5;
            let phi = (1.0 - 2.0 * t / n as f64).acos();
            let th = PI * (1.0 + 5f64.sqrt()) * t;
            [th.cos() * phi.sin(), th.sin() * phi.sin(), phi.cos()]
        })
        .collect()
}

/// [n, A_n mín, A_n máx, A_n rms, ω(k) − ω(−k), Δx/ℓ_P con el rms, Δx/ℓ_P en la peor dirección].
pub(crate) fn lorentz_coefficients(walk: u32, ndirs: usize) -> Vec<f64> {
    let ks = [0.01, 0.02, 0.03, 0.04];
    // mínimos cuadrados de y = A₁k + A₂k² + A₃k³ (ecuaciones normales 3 × 3, fijas)
    let mut g = [[0.0f64; 3]; 3];
    for &k in &ks {
        let v = [k, k * k, k * k * k];
        for i in 0..3 {
            for j in 0..3 {
                g[i][j] += v[i] * v[j];
            }
        }
    }
    let dirs = fibonacci(ndirs);
    let (mut a1, mut a2) = (Vec::with_capacity(ndirs), Vec::with_capacity(ndirs));
    for d in &dirs {
        let mut h = [0.0f64; 3];
        for &k in &ks {
            let y = speed(walk, [k * d[0], k * d[1], k * d[2]]) - 1.0;
            let v = [k, k * k, k * k * k];
            for i in 0..3 {
                h[i] += v[i] * y;
            }
        }
        let x = crate::util::solve_n::<3>(g, h);
        a1.push(x[0]);
        a2.push(x[1]);
    }
    let stats = |a: &[f64]| -> (f64, f64, f64) {
        let mn = a.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rms = (a.iter().map(|v| v * v).sum::<f64>() / a.len() as f64).sqrt();
        (mn, mx, rms)
    };
    let kt = [0.21, 0.13, 0.07];
    let asym = walk_omega(walk, kt) - walk_omega(walk, [-kt[0], -kt[1], -kt[2]]);
    let (s1, s2) = (stats(&a1), stats(&a2));
    // el orden lo decide la simetría exacta ω(k) = ω(−k), no el ajuste (que tiene ruido de ~10⁻⁶)
    let linear = asym.abs() > 1e-10;
    if linear {
        let worst = s1.0.abs().max(s1.1.abs());
        let b = birefringence_bound(&dirs, &a1, 800);
        vec![1.0, s1.0, s1.1, s1.2, asym, E_PLANCK / (s1.2 * E_QG1), E_PLANCK / (worst * E_QG1), b[0], b[1], b[2]]
    } else {
        let worst = s2.0.abs().max(s2.1.abs());
        let bound = |a: f64| E_PLANCK / (E_QG2 * (a / 1.5).sqrt());
        vec![2.0, s2.0, s2.1, s2.2, asym, bound(s2.2), bound(worst), f64::NAN, f64::NAN, 0.0]
    }
}

/// Cotas al 95 % de Kislat y Krawczynski (PRD 95, 083013, 2017, tabla IV) sobre los coeficientes
/// k_(V)jm^(5) de la SME, en GeV⁻¹: (j, m, parte real, cota). Los de m < 0 se siguen de k_j,−m = (−1)^m k_jm*.
const KK_LIMITS: [(usize, usize, bool, f64); 16] = [
    (0, 0, true, 5.0e-26),
    (1, 0, true, 6.5e-26),
    (1, 1, true, 7.8e-27),
    (1, 1, false, 2.9e-26),
    (2, 0, true, 3.0e-26),
    (2, 1, true, 7.5e-27),
    (2, 1, false, 2.4e-26),
    (2, 2, true, 2.1e-26),
    (2, 2, false, 1.7e-26),
    (3, 0, true, 2.1e-26),
    (3, 1, true, 1.4e-26),
    (3, 1, false, 1.0e-26),
    (3, 2, true, 8.3e-27),
    (3, 2, false, 1.4e-26),
    (3, 3, true, 9.7e-27),
    (3, 3, false, 5.8e-27),
];

/// Y_jm(n̂) complejos (Condon–Shortley) con j ≤ 3 y m ≥ 0, en el orden (0,0), (1,0), (1,1), (2,0), …, (3,3).
fn ylm_all(n: [f64; 3]) -> [C; 10] {
    let (x, y, z) = (n[0], n[1], n[2]);
    let w = C::new(x, y);
    let w2 = w * w;
    let r = |c: f64| C::new(c, 0.0);
    let k = |num: f64| (num / PI).sqrt();
    [
        r(0.5 / PI.sqrt()),
        r(k(3.0 / 4.0) * z),
        w * (-k(3.0 / 8.0)),
        r(k(5.0 / 16.0) * (3.0 * z * z - 1.0)),
        w * (-k(15.0 / 8.0) * z),
        w2 * k(15.0 / 32.0),
        r(k(7.0 / 16.0) * (5.0 * z * z * z - 3.0 * z)),
        w * (-k(21.0 / 64.0) * (5.0 * z * z - 1.0)),
        w2 * (k(105.0 / 32.0) * z),
        w2 * w * (-k(35.0 / 64.0)),
    ]
}

/// Índice de (j, m) en `ylm_all`.
fn ylm_index(j: usize, m: usize) -> usize {
    j * (j + 1) / 2 + m
}

/// Orientación al azar (método de Shoemake) a partir de tres uniformes.
fn random_rotation(u: [f64; 3]) -> [[f64; 3]; 3] {
    let (a, b) = ((1.0 - u[0]).sqrt(), u[0].sqrt());
    let (q0, q1, q2, q3) = (a * (2.0 * PI * u[1]).sin(), a * (2.0 * PI * u[1]).cos(), b * (2.0 * PI * u[2]).sin(), b * (2.0 * PI * u[2]).cos());
    [
        [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
        [2.0 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 - q0 * q1)],
        [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
    ]
}

/// Birrefringencia de un fotón autoconjugado con término lineal A₁(n̂) (coeficiente de la velocidad de grupo): sus
/// helicidades van con ω(k) y ω(−k), separadas Δω = A₁·k²·Δx, y en la SME la separación es 2k²·Σ_jm Y_jm·k_(V)jm^(5),
/// así que Σ_jm Y_jm(n̂)·k_(V)jm^(5) = A₁(n̂)·Δx/2. Proyecta A₁/2 (muestreado en las direcciones de Fibonacci, en el
/// marco de la red) sobre j ≤ 3 para `norients` orientaciones de la red respecto al marco ecuatorial y aplica
/// cada una de las 16 cotas de Kislat y Krawczynski. Devuelve [Δx/ℓ_P mediana, Δx/ℓ_P en la peor orientación,
/// norma (Σ_jm |k_jm|²)^½ por unidad de Δx].
fn birefringence_bound(dirs: &[[f64; 3]], a1: &[f64], norients: usize) -> [f64; 3] {
    let wq = 4.0 * PI / dirs.len() as f64;
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut uni = || {
        rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let project = |rot: &[[f64; 3]; 3]| -> [C; 10] {
        let mut kjm = [C::new(0.0, 0.0); 10];
        for (d, &a) in dirs.iter().zip(a1) {
            let s = [0, 1, 2].map(|i| rot[i][0] * d[0] + rot[i][1] * d[1] + rot[i][2] * d[2]);
            for (acc, y) in kjm.iter_mut().zip(ylm_all(s)) {
                *acc += y.conj() * (0.5 * a * wq);
            }
        }
        kjm
    };
    let ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let k0 = project(&ident);
    let mut norm2 = 0.0;
    for j in 0..4 {
        for m in 0..=j {
            norm2 += if m == 0 { 1.0 } else { 2.0 } * k0[ylm_index(j, m)].norm_sqr();
        }
    }
    let mut bounds = Vec::with_capacity(norients);
    for _ in 0..norients {
        let rot = random_rotation([uni(), uni(), uni()]);
        let kjm = project(&rot);
        let mut best = f64::INFINITY;
        for &(j, m, re, lim) in KK_LIMITS.iter() {
            let c = kjm[ylm_index(j, m)];
            let v = if re { c.re } else { c.im }.abs();
            if v > 1e-12 {
                best = best.min(lim / v);
            }
        }
        bounds.push(best * E_PLANCK);
    }
    bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    [bounds[bounds.len() / 2], bounds[bounds.len() - 1], norm2.sqrt()]
}

type M3 = [[f64; 3]; 3];

/// Rotación de ángulo t alrededor del eje a.
fn rot3(a: usize, t: f64) -> M3 {
    let (c, s) = (t.cos(), t.sin());
    match a {
        0 => [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        1 => [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        _ => [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

fn m3_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|l| a[i][l] * b[l][j]).sum();
        }
    }
    r
}

/// Paso del fotón de Mlodinow y Brun sobre el vector de Riemann–Silberstein F = E + iB: R_z(k_z)·R_y(k_y)·R_x(k_x),
/// cuyo ángulo de rotación es la ω de su ec. (42).
fn mb_rotation(k: [f64; 3]) -> M3 {
    m3_mul(&rot3(2, k[2]), &m3_mul(&rot3(1, k[1]), &rot3(0, k[0])))
}

/// Campo real (E y B reales) del fotón de Mlodinow y Brun en el par de modos k y −k, con polarización inicial
/// lineal y transversal. Devuelve [ritmo de rotación de la polarización lineal medido (rad/paso),
/// (ω(k) − ω(−k))/2, helicidad k̂·J del modo de frecuencia ω(k), helicidad del modo de frecuencia ω(−k)].
pub(crate) fn photon_rotation(k: [f64; 3], steps: usize) -> [f64; 4] {
    let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt().max(1e-12);
    let kh = [k[0] / kn, k[1] / kn, k[2] / kn];
    let mk = [-k[0], -k[1], -k[2]];
    let (wk, wm) = (walk_omega(5, k), walk_omega(5, mk));
    let (rk, rm) = (mb_rotation(k), mb_rotation(mk));
    let to_c = |r: &M3| -> Vec<C> { r.iter().flat_map(|row| row.iter().map(|&v| C::new(v, 0.0))).collect() };
    // autovector de R con autovalor más cercano a `target`
    let pick = |r: &M3, target: C| -> Vec<C> {
        let e = linalg::eig(to_c(r), 3);
        let mut best = 0;
        for j in 1..3 {
            if (e.values[j] - target).norm() < (e.values[best] - target).norm() {
                best = j;
            }
        }
        e.vectors[best].clone()
    };
    // modos de frecuencia positiva de E en k: e^{−iω(k)} de R(k) y el conjugado del e^{+iω(−k)} de R(−k)
    let up = pick(&rk, C::new(wk.cos(), -wk.sin()));
    let um: Vec<C> = pick(&rm, C::new(wm.cos(), wm.sin())).iter().map(|c| c.conj()).collect();
    let helicity = |u: &[C]| -> f64 {
        // (k̂·J)u = i·k̂ × u
        let cr = [kh[1] * u[2] - kh[2] * u[1], kh[2] * u[0] - kh[0] * u[2], kh[0] * u[1] - kh[1] * u[0]];
        let num: C = (0..3).map(|i| u[i].conj() * C::new(0.0, 1.0) * cr[i]).sum();
        let den: f64 = u.iter().map(|c| c.norm_sqr()).sum();
        num.re / den
    };
    let proj = |u: &[C], v: &[C]| -> Vec<C> {
        let num: C = u.iter().zip(v).map(|(a, b)| a.conj() * b).sum();
        let den: f64 = u.iter().map(|c| c.norm_sqr()).sum();
        u.iter().map(|a| *a * (num / den)).collect()
    };
    // polarización inicial: E(x) = e1·cos(k·x), B = 0
    let zz = if kh[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let mut e1 = [kh[1] * zz[2] - kh[2] * zz[1], kh[2] * zz[0] - kh[0] * zz[2], kh[0] * zz[1] - kh[1] * zz[0]];
    let n1 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
    e1 = e1.map(|v| v / n1);
    let e2 = [kh[1] * e1[2] - kh[2] * e1[1], kh[2] * e1[0] - kh[0] * e1[2], kh[0] * e1[1] - kh[1] * e1[0]];
    let mut fk = e1.map(|v| C::new(0.5 * v, 0.0));
    let mut fm = fk;
    let apply = |r: &M3, f: &[C; 3]| -> [C; 3] { [0, 1, 2].map(|i| (0..3).map(|j| f[j] * r[i][j]).sum()) };
    let (mut ts, mut angs) = (Vec::new(), Vec::new());
    let mut prev = 0.0f64;
    for t in 0..=steps {
        if t % 10 == 0 {
            let fmc: Vec<C> = fm.iter().map(|c| c.conj()).collect();
            let (p1, p2) = (proj(&up, &fk), proj(&um, &fmc));
            let fpos: Vec<C> = p1.iter().zip(&p2).map(|(a, b)| *a + *b).collect();
            let a: C = (0..3).map(|i| fpos[i] * e1[i]).sum();
            let b: C = (0..3).map(|i| fpos[i] * e2[i]).sum();
            let (s1, s2) = (a.norm_sqr() - b.norm_sqr(), 2.0 * (a.conj() * b).re);
            // ángulo doble del eje mayor, desenrollado
            let mut th2 = s2.atan2(s1);
            while th2 - prev > PI {
                th2 -= 2.0 * PI;
            }
            while th2 - prev < -PI {
                th2 += 2.0 * PI;
            }
            prev = th2;
            ts.push(t as f64);
            angs.push(0.5 * th2);
        }
        fk = apply(&rk, &fk);
        fm = apply(&rm, &fm);
    }
    let (_, slope) = crate::util::fit_linear(&ts, &angs);
    [slope, 0.5 * (wk - wm), helicity(&up), helicity(&um)]
}

/// Velocidad radial de la banda simétrica de la caminata de Grover en |k| = k_over_m·m a lo largo de n̂:
/// se sigue la banda desde k = 0 (espinor simétrico) por solapamiento de autovectores.
pub(crate) fn grover_speed(dim: u32, m: f64, k_over_m: f64, n: [f64; 3]) -> f64 {
    let dim = if dim == 3 { 3 } else { 2 };
    let nc = 2 * dim as usize;
    let coin = coin_matrix(m, nc);
    let norm = (n[0] * n[0] + n[1] * n[1] + if dim == 3 { n[2] * n[2] } else { 0.0 }).sqrt().max(1e-300);
    let nh = [n[0] / norm, n[1] / norm, if dim == 3 { n[2] / norm } else { 0.0 }];
    let kmax = k_over_m * m;
    let steps = 2000usize;
    let dk = kmax * 1.01 / steps as f64;
    let mut v: Vec<C> = vec![C::new(1.0 / (nc as f64).sqrt(), 0.0); nc];
    let mut om_prev = m;
    let mut omegas = Vec::with_capacity(steps + 1);
    for s in 0..=steps {
        let kk = s as f64 * dk;
        let ph = shift_phases([kk * nh[0], kk * nh[1], kk * nh[2]], dim);
        let mut u = coin.clone();
        for i in 0..nc {
            for j in 0..nc {
                u[i * nc + j] *= ph[i];
            }
        }
        let e = linalg::eig(u, nc);
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
        // ω continuo (sin saltos de 2π)
        let mut om = -e.values[best].arg();
        while om - om_prev > PI {
            om -= 2.0 * PI;
        }
        while om - om_prev < -PI {
            om += 2.0 * PI;
        }
        om_prev = om;
        omegas.push(om);
    }
    let i = ((kmax / dk).round() as usize).clamp(1, steps - 1);
    (omegas[i + 1] - omegas[i - 1]) / (2.0 * dk)
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.30: walk = 0 Weyl de orden fijo, 1 Weyl alternado, 2 bombeos de Higashikawa, 3 fotón de red
    /// (κ = 1, s = 2), 4 Wilson–Dirac (dt → 0), 5 fotón de Mlodinow y Brun: [n, A_n mín, A_n máx,
    /// A_n rms, ω(k) − ω(−k), Δx/ℓ_P con el rms, Δx/ℓ_P en la peor dirección] (tiempo de vuelo, LHAASO),
    /// seguidos de [Δx/ℓ_P mediana, Δx/ℓ_P en la peor orientación, norma de k_(V)jm^(5) por unidad de Δx]
    /// por birrefringencia del fotón autoconjugado (Kislat y Krawczynski); NaN si n = 2.
    pub fn lorentz_violation_coefficients(&mut self, walk: u32) -> *const f64 {
        self.an.out = lorentz_coefficients(walk.min(5), 2000);
        self.an.out.as_ptr()
    }

    /// §2.30: campo real (F = E + iB) del fotón de Mlodinow y Brun en el par k, −k con polarización inicial
    /// lineal: [rotación de la polarización medida (rad/paso), (ω(k) − ω(−k))/2, helicidad del modo ω(k),
    /// helicidad del modo ω(−k)].
    pub fn photon_polarization_rotation(&mut self, kx: f32, ky: f32, kz: f32, steps: u32) -> *const f64 {
        self.an.out = photon_rotation([kx as f64, ky as f64, kz as f64], steps.clamp(100, 100_000) as usize).to_vec();
        self.an.out.as_ptr()
    }

    /// §2.30: velocidad radial de la banda simétrica de la caminata de Grover con |k| = k_over_m·m en la
    /// dirección (nx, ny, nz). Con k ≫ m tiende a ‖n̂‖∞.
    pub fn grover_relativistic_speed(&self, dim: u32, m: f32, k_over_m: f32, nx: f32, ny: f32, nz: f32) -> f64 {
        grover_speed(dim, (m as f64).clamp(1e-6, 1.5), (k_over_m as f64).clamp(0.01, 1e4), [nx as f64, ny as f64, nz as f64])
    }
}
