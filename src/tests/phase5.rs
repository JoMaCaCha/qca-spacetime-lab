//! Fase 5 (Yang–Mills SU(2)/SU(3) y quarks de Dirac, §2.11, §2.14, §3.8 y §3.11):
//! tests 48–55, 72, 73 y 114. La materia es f32 y los enlaces f64 (umbrales "en f32" de §5).

use crate::sun::Mat;
use crate::util::{fit_linear, min_image};
use crate::QuantumSpacetimeUniverse as U;
use std::f64::consts::PI;

fn colour_matter(w: usize, h: usize, d: usize, n: u32) -> U {
    let mut u = U::new(w, h, d, 0.5, 0.1, 1.0);
    u.set_neighbor_coupling(0.0);
    u.set_capacity(0.0);
    u.set_gauge_group(n);
    u
}

fn densities(u: &U) -> Vec<f64> {
    (0..u.n).map(|i| u.matter_density_f64(i)).collect()
}

/// Test 48: Gauss e invariancia gauge no abelianas con materia de color neutra.
#[test]
fn t48_gauss_gauge_sun() {
    for &n in &[2u32, 3] {
        // 2D, 48², 300 pasos
        let mut u = colour_matter(48, 48, 1, n);
        for c in 0..n {
            let ang = 2.0 * PI * c as f64 / n as f64;
            u.inject_colour_wavepacket((24.0 + 10.0 * ang.cos()) as i32, (24.0 + 10.0 * ang.sin()) as i32, 0, 6.0, 100.0, (0.2 * ang.sin()) as f32, (-0.2 * ang.cos()) as f32, 0.0, c);
        }
        u.set_em(2, 0.6, 0.25);
        let mut worst = 0.0f64;
        for s in 0..300 {
            u.step();
            if s % 50 == 0 {
                worst = worst.max(u.gauss_residual());
            }
        }
        worst = worst.max(u.gauss_residual());
        assert!(worst < 1e-4, "SU({n}) 2D: Gauss {worst:e}");
        // 3D, 16³, 150 pasos, con la copia transformada
        let build = |gauge: bool| -> U {
            let mut u = colour_matter(16, 16, 16, n);
            for c in 0..n {
                let ang = 2.0 * PI * c as f64 / n as f64;
                u.inject_colour_wavepacket((8.0 + 4.0 * ang.cos()) as i32, (8.0 + 4.0 * ang.sin()) as i32, 8, 6.0, 100.0, 0.2, 0.0, 0.1, c);
            }
            u.set_em(2, 0.6, 0.25);
            if gauge {
                u.apply_random_gauge_transform(3.0, 48);
            }
            u
        };
        let mut a = build(false);
        let mut b = build(true);
        a.step_n(150);
        b.step_n(150);
        assert!(a.gauss_residual() < 1e-4, "SU({n}) 3D: Gauss {:e}", a.gauss_residual());
        assert!(b.gauss_residual() < 1e-4, "SU({n}) 3D transformada: Gauss {:e}", b.gauss_residual());
        let d = densities(&a).iter().zip(densities(&b)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(d < 1e-4, "SU({n}) 3D: diferencia de densidades {d:e}");
    }
}

fn wilson_energy(u: &U) -> f64 {
    let mut w = 0.0;
    for i in 0..u.n {
        for a in 0..u.dim {
            for b in (a + 1)..u.dim {
                w += u.ym.n as f64 - u.ym_plaq_p(i, a, b).trace().re;
            }
        }
    }
    2.0 * u.em.kappa * w
}

/// Test 49: la fuerza es −∂W/∂θ (diferencias finitas centradas, f64).
#[test]
fn t49_force_gradient() {
    for &(w, h, d) in &[(6usize, 6usize, 1usize), (4, 4, 4)] {
        for &n in &[2u32, 3] {
            let mut u = colour_matter(w, h, d, n);
            u.em.kappa = 0.25;
            let mut rng = crate::rng::SplitMix64::new(49 + n as u64);
            let ng = u.ym.ng;
            for l in 0..u.dim * u.n {
                let th: Vec<f64> = (0..ng).map(|_| 0.8 * rng.gauss()).collect();
                let m = u.ym.alg.exp_minus_i(&th);
                u.ym.set_link(l, &m);
            }
            u.ym_compute_force();
            let eps = 1e-6;
            let mut worst = 0.0f64;
            for l in [0usize, 3, u.n + 2, u.dim * u.n - 1] {
                for c in 0..ng {
                    let base: Mat = u.ym.link(l);
                    let mut th = vec![0.0; ng];
                    th[c] = eps;
                    let up = base.mul(&u.ym.alg.exp_minus_i(&th));
                    th[c] = -eps;
                    let dn = base.mul(&u.ym.alg.exp_minus_i(&th));
                    u.ym.set_link(l, &up);
                    let wp = wilson_energy(&u);
                    u.ym.set_link(l, &dn);
                    let wm = wilson_energy(&u);
                    u.ym.set_link(l, &base);
                    let num = -(wp - wm) / (2.0 * eps);
                    let f = u.ym.force[l * ng + c];
                    let rel = (num - f).abs() / f.abs().max(1e-3);
                    worst = worst.max(rel);
                }
            }
            assert!(worst < 1e-5, "SU({n}) en {w}×{h}×{d}: error relativo {worst:e}");
        }
    }
}

/// Centro de la energía del campo de color en una ventana de x (pico de energía, interpolado).
fn field_peak(u: &mut U, x0: usize, x1: usize) -> f64 {
    u.compute_colour_field_densities();
    let w = u.width();
    let h = u.height();
    let col = |x: usize| (0..h).map(|y| u.ym.energy_density[y * w + x] as f64).sum::<f64>();
    let mut best = (f64::NEG_INFINITY, x0);
    for x in x0..x1 {
        let v = col(x);
        if v > best.0 {
            best = (v, x);
        }
    }
    let xm = best.1;
    let (a, b, c) = (col(xm - 1), best.0, col(xm + 1));
    let den = a - 2.0 * b + c;
    xm as f64 + if den.abs() > 1e-300 { 0.5 * (a - c) / den } else { 0.0 }
}

/// Test 50: los gluones de cada color viajan a √κ.
#[test]
fn t50_gluon_speed() {
    for &n in &[2u32, 3] {
        let ng = (n * n - 1) as usize;
        for comp in 0..ng {
            for &amp in &[0.05f32, 1.0] {
                let mut u = colour_matter(256, 4, 1, n);
                u.set_em(2, 1.0, 0.25);
                u.inject_gluon_pulse(64, comp as u32, amp, 8.0);
                let (mut ts, mut xs) = (Vec::new(), Vec::new());
                for t in 0..=160 {
                    if t >= 40 {
                        ts.push(t as f64);
                        xs.push(field_peak(&mut u, 70, 190));
                    }
                    u.step();
                }
                let v = fit_linear(&ts, &xs).1;
                assert!((v / 0.5 - 1.0).abs() < 0.02, "SU({n}) color {comp}, a = {amp}: v = {v}");
            }
        }
    }
}

/// Desviación de la superposición de dos pulsos de gluones (máximo en el tiempo).
fn superposition_deviation(n: u32, ca: u32, cb: u32, a: f32) -> f64 {
    let run = |pa: bool, pb: bool| -> Vec<Vec<f64>> {
        let mut u = colour_matter(128, 4, 1, n);
        u.set_em(2, 1.0, 0.25);
        if pa {
            u.inject_gluon_pulse(40, ca, a, 6.0);
        }
        if pb {
            u.inject_gluon_pulse(88, cb, a, 6.0);
        }
        let mut out = Vec::new();
        for t in 0..=160 {
            if t % 4 == 0 {
                out.push(u.ym.e.clone());
            }
            u.step();
        }
        out
    };
    let ea = run(true, false);
    let eb = run(false, true);
    let eab = run(true, true);
    let mut worst = 0.0f64;
    for k in 0..ea.len() {
        let (mut num, mut den) = (0.0, 0.0);
        for l in 0..ea[k].len() {
            num += (eab[k][l] - ea[k][l] - eb[k][l]).powi(2);
            den += (ea[k][l] + eb[k][l]).powi(2);
        }
        if den > 0.0 {
            worst = worst.max((num / den).sqrt());
        }
    }
    worst
}

/// Test 51: los gluones interactúan entre sí (∝ a si los colores no conmutan, ∝ a² si conmutan).
#[test]
fn t51_gluon_self_interaction() {
    let amps = [0.001f32, 0.002, 0.004];
    let slope = |vals: &[f64]| fit_linear(&amps.iter().map(|a| (*a as f64).ln()).collect::<Vec<_>>(), &vals.iter().map(|v| v.ln()).collect::<Vec<_>>()).1;
    // conmutan: SU(2) T³,T³ ; SU(3) T³,T⁸
    for &(n, ca, cb) in &[(2u32, 2u32, 2u32), (3, 2, 7)] {
        let v: Vec<f64> = amps.iter().map(|&a| superposition_deviation(n, ca, cb, a)).collect();
        assert!(v.iter().all(|x| *x < 1e-6), "SU({n}) conmutan: {v:?}");
        let s = slope(&v);
        assert!((s - 2.0).abs() < 0.2, "SU({n}) conmutan: pendiente {s}");
    }
    // no conmutan: SU(2) T¹,T² ; SU(3) T¹,T⁴
    let mut at4 = Vec::new();
    for &(n, ca, cb) in &[(2u32, 0u32, 1u32), (3, 0, 3)] {
        let v: Vec<f64> = amps.iter().map(|&a| superposition_deviation(n, ca, cb, a)).collect();
        assert!(v[2] > 1e-3, "SU({n}) no conmutan, a = 0,004: {}", v[2]);
        let s = slope(&v);
        assert!((s - 1.0).abs() < 0.15, "SU({n}) no conmutan: pendiente {s}");
        at4.push(v[2]);
    }
    let r = at4[1] / at4[0];
    assert!((r - 0.5).abs() < 0.05, "razón SU(3)/SU(2) = {r}");
}

/// Aceleración del solitón de prueba de color `col` frente a una carga estática roja (§3.8).
fn colour_force_run(n: u32, col: u32, q: f32) -> (f64, f64) {
    let mut u = colour_matter(128, 128, 1, n);
    u.inject_colour_wavepacket(79, 64, 0, 6.0, 300.0, 0.0, 0.0, 0.0, col);
    u.step_n(300);
    u.set_em(2, q, 0.25);
    u.add_static_colour_charge_sigma(49, 64, 0, 600.0, 0, 2.0);
    let mut tr = super::common::Tracker::new(79.0, 64.0, 0.0, 12.0);
    tr.update(&u);
    // predicción: w_test·Ẽ de la carga sola en la posición de la prueba
    let mut s = colour_matter(128, 128, 1, n);
    s.set_em(2, q, 0.25);
    s.add_static_colour_charge_sigma(49, 64, 0, 600.0, 0, 2.0);
    let x = tr.pos[0].round() as i32;
    let v = crate::sun::basis_colour(n as usize, col as usize);
    let mut w = [0.0f64; 8];
    s.ym.alg.charge(&v, &mut w);
    let ng = s.ym.ng;
    let mut f = 0.0;
    for c in 0..ng {
        let e = 0.5 * (s.colour_link_e(x - 1, 64, 0, 0, c as u32) + s.colour_link_e(x, 64, 0, 0, c as u32));
        f += w[c] * e;
    }
    let (mut ts, mut xs) = (vec![0.0], vec![tr.unwrapped[0]]);
    for t in 1..=600 {
        u.step();
        tr.update(&u);
        if t % 4 == 0 {
            ts.push(t as f64);
            xs.push(tr.unwrapped[0]);
        }
    }
    (2.0 * crate::util::fit_quadratic(&ts, &xs).2, f / (2.0 * (0.5f64).tan()))
}

/// Test 52: factores de color. Rojo-rojo repele; rojo-verde y rojo-azul atraen con −½.
#[test]
fn t52_colour_factors() {
    let q = 0.0035;
    let acc = |n: u32, col: u32| -> (f64, f64) {
        let (a, pred) = colour_force_run(n, col, q);
        let (c, _) = colour_force_run(n, col, 0.0);
        (a - c, pred)
    };
    let (ar, pr) = acc(3, 0);
    let (ag, pg) = acc(3, 1);
    let (ab, _) = acc(3, 2);
    assert!(ar > 0.0 && ag < 0.0, "rojo {ar:e}, verde {ag:e}");
    assert!((ab / ag - 1.0).abs() < 0.02, "azul {ab:e} frente a verde {ag:e}");
    assert!((ag / ar + 0.5).abs() < 0.08, "verde/rojo = {}", ag / ar);
    for (a, p) in [(ar, pr), (ag, pg)] {
        let r = a / p;
        assert!(r > 0.85 && r < 1.05, "a/predicción = {r}");
    }
    let (au, _) = acc(2, 0);
    let (ad, _) = acc(2, 1);
    assert!((ad / au + 1.0).abs() < 0.1, "SU(2): ↓/↑ = {}", ad / au);
}

/// λ de la distancia entre copias en la ventana D ∈ [1e-10, 1e-4] (pendiente de ln D).
fn lyapunov(n: u32, amp: f32, seed: u32, dims: (usize, usize, usize), steps: usize) -> (f64, f64) {
    let mut u = colour_matter(dims.0, dims.1, dims.2, n);
    u.set_em(2, 1.0, 0.25);
    u.seed_gluon_bath(amp, seed);
    u.step_n(300);
    let eps_link = u.em_energy() / (u.dim * u.n) as f64;
    let mut v = colour_matter(dims.0, dims.1, dims.2, n);
    v.set_em(2, 1.0, 0.25);
    v.copy_gauge_from(&u);
    v.perturb_gauge_field(1e-9);
    let (mut ts, mut ls) = (Vec::new(), Vec::new());
    for t in 0..steps {
        u.step();
        v.step();
        let d = u.plaquette_distance(&v);
        if d > 1e-10 && d < 1e-4 {
            ts.push(t as f64);
            ls.push(d.ln());
        }
    }
    let lam = if ts.len() > 5 { fit_linear(&ts, &ls).1 } else { 0.0 };
    (lam, eps_link)
}

/// Test 53: caos con λ ∝ √ε en 2D, U(1) sin caos, y el integrador con s = 2 conserva la energía.
#[test]
fn t53_chaos_integrator() {
    let (l1, e1) = lyapunov(2, 0.2, 7, (32, 32, 1), 400);
    let (l2, e2) = lyapunov(2, 0.6, 7, (32, 32, 1), 400);
    assert!(l1 > 0.0 && l2 > 0.0, "λ = {l1}, {l2}");
    let expo = (l2 / l1).ln() / (e2 / e1).ln();
    assert!((expo - 0.5).abs() < 0.12, "exponente {expo} (ε = {e1:.3}, {e2:.3}; λ = {l1:.4}, {l2:.4})");
    // U(1) con un baño equivalente: la distancia no crece
    let mut u = U::new(32, 32, 1, 0.5, 0.1, 1.0);
    u.set_em(2, 1.0, 0.25);
    u.seed_gluon_bath(0.6, 7);
    u.step_n(300);
    let mut v = U::new(32, 32, 1, 0.5, 0.1, 1.0);
    v.set_em(2, 1.0, 0.25);
    v.copy_gauge_from(&u);
    v.perturb_gauge_field(1e-9);
    // la perturbación toca E: la distancia entre plaquetas aparece tras el primer paso
    u.step();
    v.step();
    let d0 = u.plaquette_distance(&v);
    u.step_n(400);
    v.step_n(400);
    let d1 = u.plaquette_distance(&v);
    assert!(d1 < 10.0 * d0, "U(1): D {d0:e} → {d1:e}");
    // 3D 10³, SU(2), ε ≈ 0,045, s = 2: la energía se conserva a 1e-3 en 700 pasos
    let mut u = colour_matter(10, 10, 10, 2);
    u.set_em(2, 1.0, 0.25);
    u.seed_gluon_bath(0.3, 53);
    // energía del leapfrog (½Σ Ẽ(t−½)·Ẽ(t+½) + W), medida desde el paso 10: el baño inicial
    // (todo eléctrico, U = 1) tiene un transitorio de unos pocos pasos
    u.step_n(10);
    let h0 = u.em_energy();
    let mut worst = 0.0f64;
    for _ in 0..70 {
        u.step_n(10);
        worst = worst.max((u.em_energy() / h0 - 1.0).abs());
    }
    assert!(worst < 1e-3, "energía con s = 2: variación {worst:e}");
}

/// Test 54: no hay cuerda: energía ∝ ln d en 2D y anchura del campo ∝ d.
#[test]
fn t54_no_string() {
    let energy_width = |d: i32| -> (f64, f64) {
        let mut u = colour_matter(256, 256, 1, 2);
        u.set_em(2, 1.0, 0.25);
        let (x1, x2) = (128 - d / 2, 128 + d / 2);
        u.add_static_colour_charge_sigma(x1, 128, 0, 20.0, 0, 1.0);
        u.add_static_colour_charge_sigma(x2, 128, 0, 20.0, 1, 1.0);
        let e = u.em_energy();
        // anchura a media altura de |Ẽ|² en el plano medio
        let xm = 128;
        let prof: Vec<f64> = (0..256).map(|y| (0..2).map(|a| (0..3).map(|c| u.colour_link_e(xm, y, 0, a, c).powi(2)).sum::<f64>()).sum::<f64>()).collect();
        let pk = prof[128];
        let mut y = 128;
        while y < 255 && prof[y] > 0.5 * pk {
            y += 1;
        }
        let width = 2.0 * (y as f64 - 128.0);
        (e, width)
    };
    let (e8, w8) = energy_width(8);
    let (e16, w16) = energy_width(16);
    let (e32, w32) = energy_width(32);
    let pred = 100.0 * 2f64.ln() / (2.0 * PI);
    for (de, lab) in [(e16 - e8, "8→16"), (e32 - e16, "16→32")] {
        assert!((de / pred - 1.0).abs() < 0.1, "{lab}: ΔE = {de} frente a {pred}");
    }
    for (w, d) in [(w8, 8.0), (w16, 16.0), (w32, 32.0)] {
        assert!(w >= 0.5 * d && w <= 0.8 * d, "anchura {w} con d = {d}");
    }
    // la solución estática no cambia con la dinámica no abeliana completa
    let mut u = colour_matter(256, 256, 1, 2);
    u.set_em(2, 1.0, 0.25);
    u.add_static_colour_charge_sigma(112, 128, 0, 20.0, 0, 1.0);
    u.add_static_colour_charge_sigma(144, 128, 0, 20.0, 1, 1.0);
    let e0 = u.ym.e.clone();
    u.step_n(100);
    let dmax = u.ym.e.iter().zip(&e0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(dmax < 1e-10, "el campo cambió {dmax:e}");
}

/// ⟨|Ẽ|²⟩ por celda promediado en el tiempo (pasos 200–1000).
fn bath_average(n: u32, charges: bool) -> Vec<f64> {
    let mut u = if n > 1 { colour_matter(96, 96, 1, n) } else { U::new(96, 96, 1, 0.5, 0.1, 1.0) };
    u.set_em(2, 1.0, 0.25);
    if n > 1 {
        u.set_field_substeps(2);
    }
    if charges {
        if n > 1 {
            u.add_static_colour_charge_sigma(24, 48, 0, 20.0, 0, 1.0);
            u.add_static_colour_charge_sigma(72, 48, 0, 20.0, 1, 1.0);
        } else {
            u.add_static_charge(24, 48, 0, 10.0, 1.0);
            u.add_static_charge(72, 48, 0, -10.0, 1.0);
        }
    }
    u.seed_gluon_bath(0.37, 55);
    let mut acc = vec![0.0f64; u.n];
    let mut cnt = 0.0;
    for t in 0..1000 {
        u.step();
        if t >= 200 {
            for i in 0..u.n {
                let (x, y, _) = u.coords(i);
                let mut s = 0.0;
                for a in 0..2u32 {
                    for c in 0..(if n > 1 { 3 } else { 1 }) {
                        s += u.colour_link_e(x as i32, y as i32, 0, a, c).powi(2);
                    }
                }
                acc[i] += s;
            }
            cnt += 1.0;
        }
    }
    acc.iter().map(|v| v / cnt).collect()
}

fn coulomb_bare(n: u32) -> Vec<f64> {
    let mut u = if n > 1 { colour_matter(96, 96, 1, n) } else { U::new(96, 96, 1, 0.5, 0.1, 1.0) };
    u.set_em(2, 1.0, 0.25);
    if n > 1 {
        u.add_static_colour_charge_sigma(24, 48, 0, 20.0, 0, 1.0);
        u.add_static_colour_charge_sigma(72, 48, 0, 20.0, 1, 1.0);
    } else {
        u.add_static_charge(24, 48, 0, 10.0, 1.0);
        u.add_static_charge(72, 48, 0, -10.0, 1.0);
    }
    (0..u.n)
        .map(|i| {
            let (x, y, _) = u.coords(i);
            let mut s = 0.0;
            for a in 0..2u32 {
                for c in 0..(if n > 1 { 3 } else { 1 }) {
                    s += u.colour_link_e(x as i32, y as i32, 0, a, c).powi(2);
                }
            }
            s
        })
        .collect()
}

fn shell_mean(f: &[f64], r0: f64, r1: f64) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for i in 0..f.len() {
        let (x, y) = ((i % 96) as f64, (i / 96) as f64);
        let r = min_image(x - 24.0, 96.0).hypot(y - 48.0);
        if r >= r0 && r < r1 {
            s += f[i];
            c += 1.0;
        }
    }
    s / c
}

/// Test 55: apantallamiento de color por un baño de gluones (y U(1) como control).
#[test]
fn t55_colour_screening() {
    // SU(2)
    let with = bath_average(2, true);
    let without = bath_average(2, false);
    let bare = coulomb_bare(2);
    let excess: Vec<f64> = with.iter().zip(&without).map(|(a, b)| a - b).collect();
    let near = shell_mean(&excess, 2.0, 6.0) - shell_mean(&excess, 18.0, 24.0);
    let r = near / shell_mean(&bare, 2.0, 6.0);
    assert!(r < 0.05, "SU(2): exceso/Coulomb = {r}");
    // U(1): el exceso coincide con Coulomb en todas las capas
    let with = bath_average(1, true);
    let without = bath_average(1, false);
    let bare = coulomb_bare(1);
    for (r0, r1) in [(2.0, 6.0), (6.0, 12.0), (12.0, 18.0)] {
        let ex = shell_mean(&with, r0, r1) - shell_mean(&without, r0, r1);
        let q = ex / shell_mean(&bare, r0, r1);
        assert!((q - 1.0).abs() < 0.01, "U(1), capa [{r0}, {r1}): exceso/Coulomb = {q}");
    }
}

fn dirac_quarks(w: usize, h: usize, n: u32, m: f32) -> U {
    let mut u = U::new(w, h, 1, 0.5, 0.0, 1.0);
    u.set_scenario(2);
    u.set_mass(m);
    u.set_dirac_colour(n);
    u
}

/// Test 72: Gauss e invariancia gauge de los quarks de Dirac (64², q = 0,5, κ = 1, s = 2).
#[test]
fn t72_quark_gauss() {
    for &n in &[2u32, 3] {
        let build = |gauge: bool| -> U {
            let mut u = dirac_quarks(64, 64, n, 0.1);
            for c in 0..n {
                let ang = 2.0 * PI * c as f64 / n as f64;
                u.inject_dirac_colour_packet((32.0 + 10.0 * ang.cos()) as f32, (32.0 + 10.0 * ang.sin()) as f32, 5.0, (0.3 * ang.sin()) as f32, (-0.3 * ang.cos()) as f32, c);
            }
            u.set_em(2, 0.5, 1.0);
            if gauge {
                u.apply_random_gauge_transform(3.0, 72);
            }
            u
        };
        let mut a = build(false);
        let mut b = build(true);
        let n0 = a.total_norm();
        a.step_n(400);
        b.step_n(400);
        assert!(a.gauss_residual() < 1e-4, "SU({n}): Gauss {:e}", a.gauss_residual());
        let d = densities(&a).iter().zip(densities(&b)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(d < 1e-4, "SU({n}): densidades {d:e}");
        assert!((a.total_norm() / n0 - 1.0).abs() < 1e-5, "norma");
    }
}

/// Test 73: Aharonov–Bohm no abeliano con h = 1, iσ₃ y −1.
#[test]
fn t73_nonabelian_ab() {
    let ratio = |theta: f32, col: [f32; 4]| -> f64 {
        let mut u = dirac_quarks(256, 256, 2, 0.1);
        u.add_wall(120, 124, 0, 8, 8, 1);
        u.add_wall(120, 252, 0, 8, 8, 1);
        u.set_em(1, 1.0, 1.0);
        if theta != 0.0 {
            u.set_colour_flux_tube(123, 127, theta, 2);
        }
        // k_y = atan(tan m·tan k_x) anula la deriva transversal de §2.8: el haz viaja recto
        let ky = ((0.1f64).tan() * (0.3f64).tan()).atan() as f32;
        u.inject_dirac_colour_vector_packet(40.0, 128.0, 25.0, 0.3, ky, col[0], col[1], col[2], col[3], 0.0, 0.0);
        let mut acc = vec![0.0f64; 256];
        for _ in 0..400 {
            u.step();
            for y in 0..256 {
                acc[y] += u.density_at(200, y as i32, 0) as f64;
            }
        }
        let mean = |a: usize, b: usize| acc[a..b].iter().sum::<f64>() / (b - a) as f64;
        mean(124, 132) / (0.5 * (mean(96, 104) + mean(152, 160)))
    };
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let cols = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [s, 0.0, s, 0.0]];
    let r1: Vec<f64> = cols.iter().map(|c| ratio(0.0, *c)).collect();
    let rm: Vec<f64> = cols.iter().map(|c| ratio(2.0 * PI as f32, *c)).collect();
    let ri: Vec<f64> = cols.iter().map(|c| ratio(PI as f32, *c)).collect();
    for k in 0..3 {
        assert!(rm[k] < 0.4 * r1[k], "h = −1, color {k}: {} frente a {}", rm[k], r1[k]);
    }
    assert!((rm[1] / rm[0] - 1.0).abs() < 0.02 && (rm[2] / rm[0] - 1.0).abs() < 0.02, "h = −1 debe ser ciego al color: {rm:?}");
    for k in 0..2 {
        assert!(ri[k] > rm[k] && ri[k] < r1[k], "h = iσ₃, color {k}: {} (entre {} y {})", ri[k], rm[k], r1[k]);
    }
}

/// Test 114: trayectoria de Wong (256×512, b = 0,04): (y↑ − y↓)/2 en los pasos 75 y 225.
#[test]
fn t114_wong() {
    let run = |col: u32| -> Vec<f64> {
        let mut u = dirac_quarks(256, 512, 2, 0.2);
        u.set_em(1, 1.0, 1.0);
        u.set_uniform_colour_links(0.05, 0, 0.04, 1);
        u.inject_dirac_colour_packet(64.0, 256.0, WONG_SIGMA, 0.3, 0.0, col);
        let mut ys = Vec::new();
        for _ in 0..=225 {
            ys.push(u.center_of_mass(1));
            u.step();
        }
        ys
    };
    let up = run(0);
    let dn = run(1);
    let half = |t: usize| 0.5 * min_image(up[t] - dn[t], 512.0);
    assert!((half(75) + 2.38).abs() < 0.3, "paso 75: {}", half(75));
    assert!((half(225) + 2.19).abs() < 0.3, "paso 225: {}", half(225));
}

/// Anchura del paquete de quark del montaje de Wong (§3.8 no la fija; ver ANALISIS.md).
const WONG_SIGMA: f32 = 12.0;
