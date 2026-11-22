//! Fase 1 (núcleo): tests 1–18 y 116 de §5.

use super::common::*;
use crate::rng::SplitMix64;
use crate::util::fit_sinusoid;
use crate::QuantumSpacetimeUniverse as U;

/// Test 1: la moneda es unitaria para espinores aleatorios con n_c = 4 y 6.
#[test]
fn t01_coin_unitary() {
    let mut rng = SplitMix64::new(11);
    for &nc in &[4usize, 6] {
        for trial in 0..200 {
            let m = (trial as f32) * 0.0078;
            let mut psi: Vec<f32> = (0..2 * nc).map(|_| rng.gauss() as f32).collect();
            let n0: f64 = psi.iter().map(|v| (*v as f64).powi(2)).sum();
            crate::apply_coin(&mut psi, nc, m.cos(), m.sin());
            let n1: f64 = psi.iter().map(|v| (*v as f64).powi(2)).sum();
            assert!(((n1 - n0) / n0).abs() < 1e-6, "nc={nc} m={m}: {n0} → {n1}");
        }
    }
}

fn set_amp(u: &mut U, x: usize, y: usize, z: usize, ch: usize, re: f32) {
    let stride = u.stride();
    let i = u.idx(x, y, z);
    u.buffer_read[i * stride + 2 * ch] = re;
}

fn amp(u: &U, x: usize, y: usize, z: usize, ch: usize) -> f32 {
    let stride = u.stride();
    let i = u.idx(x, y, z);
    u.buffer_read[i * stride + 2 * ch]
}

/// Test 2: transporte libre (m = 0, g = 0, V = 0): cada canal avanza una celda por paso
/// y vuelve a su origen tras recorrer el toro.
#[test]
fn t02_free_transport() {
    // 2D
    let (w, h) = (16usize, 12usize);
    // canal → (dx, dy) por paso
    let moves2 = [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)];
    for ch in 0..4 {
        let mut u = U::new(w, h, 1, 0.0, 0.0, 1.0);
        u.set_capacity(0.0);
        set_amp(&mut u, 5, 7, 0, ch, 1.0);
        u.step();
        let (dx, dy) = moves2[ch];
        let x1 = (5 + dx).rem_euclid(w as i32) as usize;
        let y1 = (7 + dy).rem_euclid(h as i32) as usize;
        assert_eq!(amp(&u, x1, y1, 0, ch), 1.0, "canal {ch} tras 1 paso");
        let period = if dx != 0 { w } else { h };
        for _ in 1..period {
            u.step();
        }
        assert_eq!(amp(&u, 5, 7, 0, ch), 1.0, "canal {ch} vuelve tras {period} pasos");
    }
    // 3D
    let (w, h, d) = (8usize, 6usize, 10usize);
    let moves3 = [(0i32, -1i32, 0i32), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)];
    for ch in 0..6 {
        let mut u = U::new(w, h, d, 0.0, 0.0, 1.0);
        u.set_capacity(0.0);
        set_amp(&mut u, 3, 2, 4, ch, 1.0);
        u.step();
        let (dx, dy, dz) = moves3[ch];
        let x1 = (3 + dx).rem_euclid(w as i32) as usize;
        let y1 = (2 + dy).rem_euclid(h as i32) as usize;
        let z1 = (4 + dz).rem_euclid(d as i32) as usize;
        assert_eq!(amp(&u, x1, y1, z1, ch), 1.0, "3D canal {ch}");
        let period = if dx != 0 { w } else if dy != 0 { h } else { d };
        for _ in 1..period {
            u.step();
        }
        assert_eq!(amp(&u, 3, 2, 4, ch), 1.0, "3D canal {ch} vuelve");
    }
}

/// Test 3: conservación de la norma con g ≠ 0, V ≠ 0 y muros.
#[test]
fn t03_norm_conservation() {
    let mut u = U::new(64, 64, 1, 0.5, 0.1, 1.0);
    u.set_capacity(0.0);
    u.add_barrier(10, 10, 0, 8, 20, 1, 0.8);
    u.add_barrier(40, 30, 0, 6, 6, 1, -0.5);
    u.add_wall(25, 0, 0, 2, 40, 1);
    u.seed_random_soup(0.5, 3);
    let n0 = u.total_norm();
    u.step_n(1000);
    let n1 = u.total_norm();
    assert!(((n1 - n0) / n0).abs() < 1e-4, "2D: {n0} → {n1}");

    let mut u = U::new(16, 16, 16, 0.5, 0.1, 1.0);
    u.set_capacity(0.0);
    u.add_barrier(2, 2, 2, 4, 4, 4, 0.7);
    u.add_wall(10, 0, 0, 1, 10, 16);
    u.seed_random_soup(0.5, 4);
    let n0 = u.total_norm();
    u.step_n(300);
    let n1 = u.total_norm();
    assert!(((n1 - n0) / n0).abs() < 1e-4, "3D: {n0} → {n1}");
}

/// Test 4: un paquete que viaja al este contra un muro vuelve hacia el oeste.
#[test]
fn t04_wall_reflection() {
    let mut u = U::new(128, 64, 1, 0.5, 0.0, 1.0);
    u.set_capacity(0.0);
    u.add_wall(80, 0, 0, 4, 64, 1);
    u.inject_wavepacket(40, 32, 0, 12.0, 10.0, 0.4, 0.0, 0.0, 0.0);
    let n0 = u.total_norm();
    let p0 = u.total_momentum(0);
    let x0 = u.center_of_mass(0);
    assert!(p0 > 0.0);
    u.step_n(240);
    let p1 = u.total_momentum(0);
    let x1 = u.center_of_mass(0);
    let n1 = u.total_norm();
    assert!(p1 < -0.5 * p0, "momento tras el rebote: {p0} → {p1}");
    assert!(x1 < 80.0 && x1 > 0.0, "sigue al oeste del muro: x = {x1} (x0 = {x0})");
    assert!(((n1 - n0) / n0).abs() < 1e-5);
}

/// Test 5: la sopa tiene ⟨E⟩ = energy_level ± 1 % y es reproducible bit a bit.
#[test]
fn t05_soup() {
    for &(w, h, d) in &[(128usize, 96usize, 1usize), (24, 20, 16)] {
        let mut a = U::new(w, h, d, 0.5, 0.1, 1.0);
        let mut b = U::new(w, h, d, 0.5, 0.1, 1.0);
        a.seed_random_soup(0.5, 42);
        b.seed_random_soup(0.5, 42);
        let mean = a.total_norm() / (w * h * d) as f64;
        assert!((mean - 0.5).abs() < 0.005, "⟨E⟩ = {mean}");
        assert!(a.buffer_read.iter().zip(&b.buffer_read).all(|(x, y)| x.to_bits() == y.to_bits()));
        b.seed_random_soup(0.5, 43);
        assert!(a.buffer_read.iter().zip(&b.buffer_read).any(|(x, y)| x != y));
    }
}

/// Test 6: sobre un campo vacío, `inject_energy_blob` añade `amount` ± 1 %.
#[test]
fn t06_injection() {
    let mut u = U::new(64, 64, 1, 0.5, 0.1, 1.0);
    u.inject_energy_blob(2, 3, 0, 6.0, 123.0);
    assert!((u.total_norm() - 123.0).abs() < 1.23);
    let mut u = U::new(24, 24, 24, 0.5, 0.1, 1.0);
    u.inject_energy_blob(-1, 30, 5, 5.0, 77.0);
    assert!((u.total_norm() - 77.0).abs() < 0.77);
}

/// Tests 7 y 11 (2D): emergencia desde la sopa con el modelo por defecto completo.
#[test]
fn t07_emergence_2d() {
    for seed in 1..=3u32 {
        let mut u = U::new(128, 128, 1, 0.5, 0.1, 1.0);
        u.seed_random_soup(0.5, seed);
        u.step_n(4000);
        let ipr = u.ipr();
        assert!(ipr > 5.0, "semilla {seed}: IPR = {ipr}");
        assert!(u.analyze_solitons() >= 1, "test 11: sin solitones (semilla {seed})");
        let mut c = U::new(128, 128, 1, 0.5, 0.0, 1.0);
        c.seed_random_soup(0.5, seed);
        c.step_n(4000);
        assert!(c.ipr() < 3.0, "control g = 0: IPR = {}", c.ipr());
    }
}

/// Tests 8 y 11 (3D): emergencia en 48³ con los valores por defecto de §3.1.
#[test]
fn t08_emergence_3d() {
    for seed in 1..=3u32 {
        let mut u = U::new(48, 48, 48, 0.5, 0.1, 1.0);
        u.seed_random_soup(0.5, seed);
        u.step_n(3000);
        let ipr = u.ipr();
        assert!(ipr > 5.0, "semilla {seed}: IPR = {ipr}");
        assert!(u.analyze_solitons() >= 1, "test 11: sin solitones 3D (semilla {seed})");
    }
    let mut c = U::new(48, 48, 48, 0.5, 0.0, 1.0);
    c.seed_random_soup(0.5, 1);
    c.step_n(3000);
    assert!(c.ipr() < 3.0, "control g = 0: IPR = {}", c.ipr());
}

/// Test 9: `mip_depth` devuelve la profundidad del paquete inyectado.
#[test]
fn t09_mip_depth() {
    let mut u = U::new(32, 32, 32, 0.5, 0.1, 1.0);
    u.inject_energy_blob(10, 12, 23, 4.0, 50.0);
    assert_eq!(u.mip_depth(10, 12), 23);
    assert_eq!(u.mip_depth(0, 30), 16, "columna vacía → P/2");
}

/// Test 10: la inestabilidad modulacional del fondo uniforme sigue a Bogoliubov.
#[test]
fn t10_modulational_instability() {
    let (w, h) = (256usize, 256usize);
    let mut u = base_2d(w, h);
    let rho = 0.5f64;
    let mut rng = SplitMix64::new(7);
    let stride = u.stride();
    for i in 0..w * h {
        let re = rho.sqrt() + 1e-5 * rng.gauss();
        let im = 1e-5 * rng.gauss();
        for k in 0..4 {
            u.buffer_read[i * stride + 2 * k] = (re / 2.0) as f32;
            u.buffer_read[i * stride + 2 * k + 1] = (im / 2.0) as f32;
        }
    }
    u.step_n(150);
    u.compute_power_spectrum();
    let p150 = u.spectrum.clone();
    u.step_n(200);
    u.compute_power_spectrum();
    let p350 = u.spectrum.clone();
    let dk = u.spectrum_dk();
    let mut best = (f64::NEG_INFINITY, 0usize);
    for b in 1..p150.len() {
        if p150[b] > 0.0 && p350[b] > 0.0 {
            let s = 0.5 * ((p350[b] as f64) / (p150[b] as f64)).ln() / 200.0;
            if s > best.0 {
                best = (s, b);
            }
        }
    }
    let sigma_pred = rho * 0.1 / (1.0 + rho).powi(2);
    let m_star = 1.09;
    let k_pred = (2.0 * m_star * rho * 0.1).sqrt() / (1.0 + rho);
    let k_meas = best.1 as f64 * dk;
    assert!((best.0 / sigma_pred - 1.0).abs() < 0.15, "σ_max = {} frente a {sigma_pred}", best.0);
    assert!((k_meas / k_pred - 1.0).abs() < 0.20, "k* = {k_meas} frente a {k_pred}");
}

/// Test 12: umbral de existencia (norma 50 se dispersa; 300 forma un solitón).
#[test]
fn t12_existence_threshold() {
    let mut u = base_2d(128, 128);
    u.inject_energy_blob(64, 64, 0, 6.0, 50.0);
    u.step_n(1000);
    assert!(peak(&u) < 0.1, "norma 50: pico {}", peak(&u));
    let mut u = base_2d(128, 128);
    u.inject_energy_blob(64, 64, 0, 6.0, 300.0);
    u.step_n(1000);
    assert!(peak(&u) > 1.5, "norma 300: pico {}", peak(&u));
    assert!(u.analyze_solitons() >= 1);
}

/// Test 13: momento → velocidad, 0,85 < v·m*/k < 1,05.
#[test]
fn t13_momentum_velocity() {
    let m_star = 2.0 * (0.5f64).tan();
    for &k in &[0.1f32, 0.2, 0.3] {
        let mut u = base_2d(128, 128);
        u.inject_wavepacket(30, 64, 0, 6.0, 300.0, k, 0.0, 0.0, 0.0);
        let mut tr = Tracker::new(30.0, 64.0, 0.0, 12.0);
        let mut xs = Vec::new();
        let mut ts = Vec::new();
        for s in 0..=800 {
            if s >= 200 && s % 10 == 0 {
                ts.push(s as f64);
                xs.push(tr.unwrapped[0]);
            }
            if s < 800 {
                u.step();
                tr.update(&u);
            }
        }
        let (_, v) = crate::util::fit_linear(&ts, &xs);
        let r = v * m_star / k as f64;
        assert!(r > 0.85 && r < 1.05, "k = {k}: v·m*/k = {r}");
    }
}

/// Test 14: inercia en una trampa armónica, Ω = √(κ/m*) ± 6 %.
#[test]
fn t14_inertia_trap() {
    let m_star = 2.0 * (0.5f64).tan();
    let kappa = 1e-4f32;
    let omega_pred = (kappa as f64 / m_star).sqrt();
    for &norm in &[150.0f32, 300.0] {
        let mut u = base_2d(128, 128);
        u.set_harmonic_potential(kappa, 64.0, 64.0, 0.0);
        u.inject_energy_blob(64, 64, 0, 6.0, norm);
        u.step_n(300);
        // desplazar 10 celdas el centro de la trampa equivale a desplazar el solitón
        u.set_harmonic_potential(kappa, 74.0, 64.0, 0.0);
        let mut tr = Tracker::new(64.0, 64.0, 0.0, 12.0);
        tr.update(&u);
        let mut ts = Vec::new();
        let mut xs = Vec::new();
        for s in 0..3000 {
            u.step();
            tr.update(&u);
            if s % 5 == 0 {
                ts.push(s as f64);
                xs.push(tr.unwrapped[0]);
            }
        }
        let (om, amp) = fit_sinusoid(&ts, &xs, 0.5 * omega_pred, 1.5 * omega_pred);
        assert!(amp > 3.0, "amplitud {amp}");
        assert!((om / omega_pred - 1.0).abs() < 0.06, "norma {norm}: Ω = {om} frente a {omega_pred}");
    }
}

/// Test 15: conservación del momento en un choque, |ΔP/P| < 1 % a los 300 pasos.
#[test]
fn t15_momentum_conservation() {
    let mut u = base_2d(128, 128);
    u.inject_wavepacket(34, 64, 0, 6.0, 300.0, 0.2, 0.0, 0.0, 0.0);
    u.inject_wavepacket(94, 64, 0, 6.0, 300.0, 0.0, 0.0, 0.0, 0.0);
    let p0 = u.total_momentum(0);
    u.step_n(300);
    let p1 = u.total_momentum(0);
    assert!(((p1 - p0) / p0).abs() < 0.01, "P: {p0} → {p1}");
}

/// Test 16: dos solitones en contrafase nunca se fusionan.
#[test]
fn t16_antiphase() {
    let mut u = base_2d(128, 128);
    u.inject_wavepacket(49, 64, 0, 6.0, 300.0, 0.15, 0.0, 0.0, 0.0);
    u.inject_wavepacket(79, 64, 0, 6.0, 300.0, -0.15, 0.0, 0.0, std::f32::consts::PI);
    for s in 0..=1200 {
        if s % 20 == 0 {
            let n = u.analyze_solitons();
            assert!(n >= 2, "paso {s}: {n} solitones");
        }
        u.step();
    }
}

/// Test 17: tope absoluto. Ninguna celda supera E_cap y la norma se conserva.
#[test]
fn t17_capacity() {
    let mut u = U::new(128, 128, 1, 0.5, 0.1, 1.0);
    u.set_capacity(20.0);
    u.seed_random_soup(0.5, 5);
    u.inject_energy_blob(64, 64, 0, 4.0, 600.0); // fuerza que el tope actúe desde el principio
    let n0 = u.total_norm();
    let mut acted = 0;
    for _ in 0..1000 {
        u.step();
        assert!(u.max_density() <= 20.0 * (1.0 + 1e-6), "E máx = {}", u.max_density());
        if u.capacity_last_iterations() > 0 {
            acted += 1;
        }
    }
    let n1 = u.total_norm();
    assert!(((n1 - n0) / n0).abs() < 1e-4, "{n0} → {n1}");
    assert!(acted > 0 && u.capacity_activity() > 0.0, "el tope no llegó a actuar");
}

/// Test 18: acoplamiento con los vecinos en la función interna de fase.
#[test]
fn t18_neighbor_coupling() {
    for &(alpha, expect_rot) in &[(0.0f32, true), (1.0f32, false)] {
        let mut u = U::new(16, 16, 1, 0.0, 0.1, 1.0);
        u.set_neighbor_coupling(alpha);
        u.set_capacity(0.0);
        let stride = u.stride();
        let i = u.idx(8, 8, 0);
        for k in 0..4 {
            u.buffer_read[i * stride + 2 * k] = 0.5; // E = 1
        }
        u.compute_e_eff::<4>();
        u.coin_phase_pass::<4>();
        let re = u.buffer_read[i * stride] as f64;
        let im = u.buffer_read[i * stride + 1] as f64;
        let ph = im.atan2(re);
        let e_eff = 1.0 / (1.0 + 1.0);
        let expected = if expect_rot { 0.1 * e_eff } else { 0.0 };
        assert!((ph - expected).abs() < 1e-6, "α = {alpha}: fase {ph}, esperada {expected}");
    }
}

/// Test 116: masa y ligadura. ω < m en los tres y m − ω crece con la norma.
#[test]
fn t116_binding() {
    let mut prev = 0.0;
    for &norm in &[150.0f32, 300.0, 600.0] {
        let mut u = base_2d(128, 128);
        u.inject_energy_blob(64, 64, 0, 6.0, norm);
        u.step_n(1500);
        let om = measure_omega(&mut u, 512);
        let bind = 0.5 - om;
        assert!(om < 0.5, "norma {norm}: ω = {om}");
        assert!(bind > prev, "norma {norm}: m − ω = {bind} no crece (anterior {prev})");
        prev = bind;
    }
}
