//! Fase 3 (vórtices y Dirac 2D, §2.7, §2.8 y §3.5): tests 26–33 y 104.

use crate::dirac2d::{dirac2d_berry, dirac2d_matrix, dirac2d_omega, su2_band};
use crate::rng::SplitMix64;
use crate::util::{fit_linear, fit_sinusoid, min_image};
use crate::QuantumSpacetimeUniverse as U;
use std::f64::consts::PI;

/// Escenario de vórtices por defecto: 256², m = 0,5, g = −0,03, E_max = 1, ρ̄ = 0,5, α = 0, sin tope.
fn vortex_universe(w: usize, h: usize) -> U {
    let mut u = U::new(w, h, 1, 0.5, 0.1, 1.0);
    u.set_scenario(1);
    u.seed_uniform_background(0.5, 1e-3, 11);
    u
}

fn vortices(u: &mut U) -> Vec<(f64, f64, i32)> {
    let n = u.analyze_vortices() as usize;
    (0..n).map(|k| (u.vort.data[3 * k] as f64, u.vort.data[3 * k + 1] as f64, u.vort.data[3 * k + 2] as i32)).collect()
}

/// Test 26: carga cuantizada y conservada; siempre exactamente 2 vórtices.
#[test]
fn t26_quantized_charge() {
    let mut u = vortex_universe(256, 256);
    u.imprint_vortex(108.0, 128.0, 1);
    u.imprint_vortex(148.0, 128.0, -1);
    for s in 0..=1500 {
        if s % 20 == 0 {
            let v = vortices(&mut u);
            assert_eq!(v.len(), 2, "paso {s}: {} vórtices", v.len());
            assert_eq!(u.total_topological_charge(), 0, "paso {s}: carga total");
            assert!(u.winding_max_error() < 1e-3, "paso {s}: winding no entero");
        }
        u.step();
    }
}

/// Test 27: ley de deriva, 0,9 < v·m*·d̄ < 1,25 con d = 32, 40 y 48.
#[test]
fn t27_drift_law() {
    let m_star = 2.0 * (0.5f64).tan();
    for &d in &[32.0f64, 40.0, 48.0] {
        let mut u = vortex_universe(256, 256);
        let x1 = 128.0 - d / 2.0;
        u.imprint_vortex(x1 as f32, 128.0, 1);
        u.imprint_vortex((x1 + d) as f32, 128.0, -1);
        let (mut ts, mut ys, mut seps) = (Vec::new(), Vec::new(), Vec::new());
        let mut y_un = 128.0f64;
        let mut last_y = 128.0f64;
        for s in 0..=1500 {
            if s % 10 == 0 {
                let v = vortices(&mut u);
                assert_eq!(v.len(), 2, "d = {d}, paso {s}");
                let my = {
                    // punto medio en y (media circular de los dos)
                    let a = v[0].1;
                    a + 0.5 * min_image(v[1].1 - a, 256.0)
                };
                y_un += min_image(my - last_y, 256.0);
                last_y = my;
                let sep = min_image(v[1].0 - v[0].0, 256.0).abs().hypot(min_image(v[1].1 - v[0].1, 256.0));
                if s >= 200 {
                    ts.push(s as f64);
                    ys.push(y_un);
                    seps.push(sep);
                }
            }
            u.step();
        }
        let v = fit_linear(&ts, &ys).1.abs();
        let dbar = seps.iter().sum::<f64>() / seps.len() as f64;
        let r = v * m_star * dbar;
        assert!(r > 0.9 && r < 1.25, "d = {d}: v·m*·d̄ = {r} (v = {v}, d̄ = {dbar})");
    }
}

/// Test 28: un par a 6 celdas se aniquila antes del paso 100; uno a 40 sobrevive 1500 pasos.
#[test]
fn t28_annihilation() {
    let mut u = vortex_universe(256, 256);
    u.imprint_vortex(125.0, 128.0, 1);
    u.imprint_vortex(131.0, 128.0, -1);
    let mut gone = None;
    for s in 0..100 {
        u.step();
        if u.analyze_vortices() == 0 {
            gone = Some(s);
            break;
        }
    }
    assert!(gone.is_some(), "el par a 6 celdas no se aniquiló antes del paso 100");
    let mut u = vortex_universe(256, 256);
    u.imprint_vortex(108.0, 128.0, 1);
    u.imprint_vortex(148.0, 128.0, -1);
    u.step_n(1500);
    assert_eq!(u.analyze_vortices(), 2, "el par a 40 celdas debe sobrevivir");
}

/// Test 29: dispersión de Bogoliubov del sonido en 256×4.
#[test]
fn t29_bogoliubov() {
    let m_star = 2.0 * (0.5f64).tan();
    let rho = 0.5f64;
    let mu = rho * 0.03 / (1.0 + rho).powi(2);
    for &mode in &[2u32, 4, 6, 10] {
        let mut u = U::new(256, 4, 1, 0.5, 0.1, 1.0);
        u.set_scenario(1);
        u.seed_density_mode(rho as f32, 0.005, mode, 0.0);
        let k = 2.0 * PI * mode as f64 / 256.0;
        let (mut ts, mut cs) = (Vec::new(), Vec::new());
        for s in 0..3000 {
            if s % 2 == 0 {
                // componente cos(kx) de la densidad relativa
                let (mut c, mut tot) = (0.0, 0.0);
                for i in 0..u.n {
                    let x = (i % 256) as f64;
                    let e = u.density(i) as f64;
                    c += e * (k * x).cos();
                    tot += e;
                }
                ts.push(s as f64);
                cs.push(2.0 * c / tot);
            }
            u.step();
        }
        let ek = k * k / (2.0 * m_star);
        let pred = (ek * (ek + 2.0 * mu)).sqrt();
        let (w, _) = fit_sinusoid(&ts, &cs, 0.5 * pred, 1.5 * pred);
        assert!((w / pred - 1.0).abs() < 0.05, "k = {k:.4}: ω = {w} frente a {pred}");
    }
}

/// Test 30: |ω(k)| = √(m² + k²) ± 0,5 % para k ≤ 0,3 sobre el eje (m = 0,1).
#[test]
fn t30_dirac_dispersion() {
    let m = 0.1;
    for j in 0..=30 {
        let k = 0.01 * j as f64;
        for &(kx, ky) in &[(k, 0.0), (0.0, k)] {
            let u = dirac2d_matrix(m, kx, ky);
            let (w, _) = su2_band(&u, 1);
            let pred = (m * m + k * k).sqrt();
            assert!((w / pred - 1.0).abs() < 0.005, "k = {k}: ω = {w} frente a {pred}");
        }
    }
}

/// Test 31: fase de Berry en |k| = 0,2: π con m = 0 y π(1 − m/E) con m = 0,1.
#[test]
fn t31_berry_phase() {
    let b0 = dirac2d_berry(0.0, 0.2, 400);
    assert!((b0 - PI).abs() < 0.01, "m = 0: {b0}");
    let b1 = dirac2d_berry(0.1, 0.2, 400);
    let pred = PI * (1.0 - 0.1 / (0.01f64 + 0.04).sqrt());
    assert!((b1 / pred - 1.0).abs() < 0.03, "m = 0,1: {b1} frente a {pred}");
}

/// Test 32: la velocidad de grupo máxima en la zona de Brillouin es ≤ 1 + 10⁻³.
#[test]
fn t32_causality() {
    for &m in &[0.0f64, 0.1] {
        let n = 241;
        let dk = 2.0 * PI / (n - 1) as f64;
        let h = 1e-5;
        let mut vmax = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let kx = -PI + i as f64 * dk;
                let ky = -PI + j as f64 * dk;
                let vx = (dirac2d_omega(m, kx + h, ky) - dirac2d_omega(m, kx - h, ky)) / (2.0 * h);
                let vy = (dirac2d_omega(m, kx, ky + h) - dirac2d_omega(m, kx, ky - h)) / (2.0 * h);
                let v = vx.hypot(vy);
                if v.is_finite() {
                    vmax = vmax.max(v);
                }
            }
        }
        assert!(vmax <= 1.0 + 1e-3, "m = {m}: v máx = {vmax}");
    }
}

/// Test 33: Zitterbewegung (frecuencia 2m, amplitud 1/(2m)) y velocidad k/√(k² + m²) en 1D.
#[test]
fn t33_zitterbewegung() {
    let m = 0.1f64;
    let mut u = U::new(2048, 1, 1, 0.5, 0.0, 1.0);
    u.set_scenario(2);
    u.set_mass(m as f32);
    u.inject_dirac_packet(1024.0, 0.0, 40.0, 0.0, 0.0, 0);
    let x0 = u.center_of_mass(0);
    let (mut ts, mut xs) = (Vec::new(), Vec::new());
    for s in 0..160 {
        ts.push(s as f64);
        xs.push(u.center_of_mass(0) - x0);
        u.step();
    }
    // el paquete ↑ finito deriva a ⟨k²/E²⟩ ≈ 1/(2σ²m²): se ajusta junto con el temblor
    let (w, amp, drift) = crate::util::fit_sinusoid_drift(&ts, &xs, 0.1, 0.35);
    assert!((w / (2.0 * m) - 1.0).abs() < 0.10, "frecuencia {w} frente a 2m = {}", 2.0 * m);
    assert!((amp / (1.0 / (2.0 * m)) - 1.0).abs() < 0.15, "amplitud {amp} frente a 1/(2m) = {}", 0.5 / m);
    let drift_pred = 1.0 / (2.0 * 40.0 * 40.0 * m * m);
    assert!((drift / drift_pred - 1.0).abs() < 0.25, "deriva {drift} frente a {drift_pred}");
    for &k in &[0.3f64, 0.6] {
        let mut u = U::new(2048, 1, 1, 0.5, 0.0, 1.0);
        u.set_scenario(2);
        u.set_mass(m as f32);
        u.inject_dirac_packet(600.0, 0.0, 40.0, k as f32, 0.0, 1);
        let (mut ts, mut xs) = (Vec::new(), Vec::new());
        let mut un = 600.0;
        let mut last = u.center_of_mass(0);
        for s in 0..300 {
            u.step();
            let c = u.center_of_mass(0);
            un += min_image(c - last, 2048.0);
            last = c;
            if s >= 50 {
                ts.push(s as f64);
                xs.push(un);
            }
        }
        let v = fit_linear(&ts, &xs).1;
        let pred = k / (k * k + m * m).sqrt();
        assert!((v / pred - 1.0).abs() < 0.03, "k = {k}: v = {v} frente a {pred}");
    }
}

/// Test 104: dispersión exacta del Dirac 2D, deriva transversal y mínimo de la banda con m > π/4.
#[test]
fn t104_exact_dispersion() {
    let mut rng = SplitMix64::new(104);
    for &m in &[0.1f64, 0.5, 0.8] {
        for _ in 0..100 {
            let kx = (2.0 * rng.next_f64() - 1.0) * PI;
            let ky = (2.0 * rng.next_f64() - 1.0) * PI;
            let (w, _) = su2_band(&dirac2d_matrix(m, kx, ky), 1);
            let pred = m.cos() * kx.cos() * ky.cos() + m.sin() * kx.sin() * ky.sin();
            assert!((w.cos() - pred).abs() < 1e-6);
        }
    }
    // v_y = −sin m·sin k_x/sin ω con m = 0,2, k = (0,3, 0)
    let h = 1e-6;
    let vy = (dirac2d_omega(0.2, 0.3, h) - dirac2d_omega(0.2, 0.3, -h)) / (2.0 * h);
    assert!((vy + 0.1672).abs() < 1e-4, "v_y = {vy}");
    // con m = 0,8 el mínimo de la banda +1 vale π/2 − m y está en ±(π/2, π/2)
    let n = 241;
    let mut best = (f64::INFINITY, 0.0, 0.0);
    for i in 0..n {
        for j in 0..n {
            let kx = -PI + 2.0 * PI * i as f64 / (n - 1) as f64;
            let ky = -PI + 2.0 * PI * j as f64 / (n - 1) as f64;
            let w = dirac2d_omega(0.8, kx, ky);
            if w < best.0 {
                best = (w, kx, ky);
            }
        }
    }
    assert!((best.0 - (PI / 2.0 - 0.8)).abs() < 1e-3, "mínimo {}", best.0);
    assert!((best.1.abs() - PI / 2.0).abs() < 0.03 && (best.2.abs() - PI / 2.0).abs() < 0.03 && best.1 * best.2 > 0.0, "en ({}, {})", best.1, best.2);
}
