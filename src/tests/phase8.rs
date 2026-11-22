//! Fase 8 (el modelo estándar clásico, §2.21, §2.22, §3.19 y §3.20): tests 125–130.

use crate::sm::FIELDS;
use crate::util::fit_quadratic;
use crate::QuantumSpacetimeUniverse as U;
use std::f64::consts::PI;

/// Escenario 6 en w × h × d con los valores por defecto de §3.19.
fn sm_universe(w: usize, h: usize, d: usize, generations: u32, quarks: bool) -> U {
    let mut u = U::new(w, h, d, 0.5, 0.0, 1.0);
    u.set_scenario(6);
    if generations != 1 || !quarks {
        u.set_scenario_sm(generations, quarks, true, 0.6, 0.4, 0.8);
    }
    assert_eq!(u.scenario(), 6, "{}", u.last_message());
    u
}

/// Test 125: masas y·v en k = 0 (dobles por espín, por color en los quarks) y un neutrino sin masa
/// con una sola helicidad.
#[test]
fn t125_sm_masses() {
    let u = sm_universe(16, 16, 16, 1, true);
    let ev = u.sm_spectrum(0.0, 0.0, 0.0);
    let mut expect = Vec::new();
    for (m, mult) in [(0.5, 6), (0.4, 2), (0.3, 6)] {
        for _ in 0..mult {
            expect.push(m);
            expect.push(-m);
        }
    }
    expect.extend([0.0, 0.0]);
    expect.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(ev.len(), 30);
    let err = ev.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("k = 0: {ev:?}, error {err:.2e}");
    assert!(err < 1e-6, "error de las masas {err:e}");
    // k = (π/4, 0, 0): estados con todo el peso en ν_L
    let k = PI / 4.0;
    let e = u.sm_two_step_eigen([k, 0.0, 0.0]);
    let nu = [FIELDS[0].off, FIELDS[0].off + 2];
    let mut nu_eps = Vec::new();
    for (j, l) in e.values.iter().enumerate() {
        let v = &e.vectors[j];
        let tot: f64 = v.iter().map(|z| z.norm_sqr()).sum();
        let w: f64 = nu.iter().map(|&c| v[c].norm_sqr()).sum();
        if w / tot > 0.99 {
            nu_eps.push(-l.arg() / 2.0);
        }
    }
    nu_eps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("neutrino en k = (π/4, 0, 0): {nu_eps:?}");
    assert_eq!(nu_eps.len(), 2, "el neutrino debe tener dos estados");
    assert!((nu_eps[0] + k).abs() < 1e-4 && (nu_eps[1] - k).abs() < 1e-4);
}

/// Estado de prueba de Gauss: Higgs perturbado y paquetes de las siete especies en movimiento.
fn gauss_state(u: &mut U, seed: u32) {
    u.perturb_sm_higgs(0.1, seed);
    u.set_packet_norm(2.0);
    let n = u.width() as f32;
    for sp in 0..7u32 {
        let a = sp as f32 * 0.9;
        let (x, y, z) = (0.2 * n + 0.6 * n * (a.cos() * 0.5 + 0.5), 0.5 * n + 0.3 * n * a.sin(), 0.3 * n + 0.1 * n * sp as f32 % (0.5 * n));
        u.inject_sm_packet(sp, x, y, z, 1.5, 0.4 * a.cos(), 0.3 * a.sin(), 0.2, sp % 3);
    }
    u.sm_store_gauss_reference();
}

/// Test 126: Gauss de SU(3)×SU(2)×U(1)_Y con todo dinámico, invariancia gauge del grupo completo y la
/// necesidad de la reacción del Yukawa sobre Π.
#[test]
fn t126_sm_gauss() {
    let mut u = sm_universe(10, 10, 10, 1, true);
    gauss_state(&mut u, 3);
    let mut v = sm_universe(10, 10, 10, 1, true);
    gauss_state(&mut v, 3);
    v.sm_random_gauge(0.6, 11);
    let tot0: f64 = (0..7).map(|s| u.species_norm(s)).sum();
    let mut worst = [0.0f64; 3];
    let mut dist = 0.0f64;
    for t in 0..100 {
        u.step();
        v.step();
        let r = u.sm_gauss_residuals();
        for c in 0..3 {
            worst[c] = worst[c].max(r[c]);
        }
        if t % 10 == 9 {
            dist = dist.max(u.sm_density_distance(&v));
        }
    }
    let tot1: f64 = (0..7).map(|s| u.species_norm(s)).sum();
    println!("Gauss [SU(3), SU(2), U(1)_Y] = {worst:?}; copia gauge {dist:.2e}; norma {tot0} → {tot1}");
    assert!(worst.iter().all(|w| *w < 1e-11), "deriva de Gauss {worst:?}");
    assert!(dist < 1e-10, "copia gauge {dist:e}");
    assert!((tot1 - tot0).abs() < 1e-9 * tot0);
    // sin la reacción sobre Π, la ley de Gauss de SU(2) deriva
    let mut w = sm_universe(8, 8, 8, 1, false);
    w.perturb_sm_higgs(0.1, 5);
    w.seed_sm_random(0.3, 7);
    w.sm_store_gauss_reference();
    w.set_sm_higgs_reaction(false);
    let mut drift = [0.0f64; 3];
    for _ in 0..50 {
        w.step();
        let r = w.sm_gauss_residuals();
        for c in 0..3 {
            drift[c] = drift[c].max(r[c]);
        }
    }
    println!("sin la reacción: {drift:?} (referencia 0,13 y 0,16)");
    assert!(drift[1] > 1e-2, "sin la reacción, SU(2) deriva solo {:e}", drift[1]);
}

/// Aceleración transversal (en y) de cada especie en un campo uniforme que crece como θ = 2·10⁻⁴·t,
/// con el enlace T³ de ángulo θ y A_Y = ay(θ). Hasta el paso 160: después, la difracción del paquete
/// sin masa (k₀ = 0,3, σ = 12) lo extiende por todo el toro transversal y el centroide pierde sentido.
pub(super) fn transverse_accels(z_boson: bool) -> Vec<f64> {
    let mut u = sm_universe(128, 128, 4, 1, true);
    u.set_yukawa_sm(0.0, 0.0, 0.0);
    u.set_sm_bosons_dynamic(false);
    let gp = 0.4f64;
    let tan2 = (0.4f64 / 0.6).powi(2);
    for sp in 0..7u32 {
        u.inject_sm_packet(sp, 20.0, 64.0, 2.0, 12.0, 0.3, 0.0, 0.0, 0);
    }
    let (mut ts, mut ys) = (Vec::new(), vec![Vec::new(); 7]);
    let y0: Vec<f64> = (0..7).map(|s| u.species_centroid(s, 1)).collect();
    for t in 0..=160u32 {
        if t % 20 == 0 {
            ts.push(t as f64);
            for s in 0..7 {
                let mut dy = u.species_centroid(s as u32, 1) - y0[s];
                dy -= 128.0 * (dy / 128.0).round();
                ys[s].push(dy);
            }
        }
        let th = 2e-4 * t as f64;
        let ay = if z_boson { tan2 * th / gp } else { -th / gp };
        u.set_sm_link_field(1, 3, th as f32, ay as f32);
        u.step();
    }
    (0..7).map(|s| 2.0 * fit_quadratic(&ts, &ys[s]).2).collect()
}

/// Test 127: Q = T³ + Y. Con el fotón, la aceleración relativa a la de e_L es Q/(−1) para las siete
/// especies; L y R la sienten igual.
#[test]
fn t127_electric_charges() {
    let a = transverse_accels(false);
    let rel: Vec<f64> = a.iter().map(|x| x / a[1]).collect();
    println!("aceleraciones {a:?}\nrelativas a e_L (ν, e_L, e_R, u_L, d_L, u_R, d_R): {rel:?}");
    let q = [0.0, 1.0, 1.0, -2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0];
    for s in 0..7 {
        assert!((rel[s] - q[s]).abs() < 0.02, "especie {s}: {} frente a {}", rel[s], q[s]);
    }
    // el Z (§3.19): (T³ − sin²θ_W·Q)/(−½ + sin²θ_W), distinto para L y R
    let z = transverse_accels(true);
    let zrel: Vec<f64> = z.iter().map(|x| x / z[1]).collect();
    let s2 = 0.16 / 0.52;
    let t3 = [0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0];
    let pred: Vec<f64> = (0..7).map(|s| (t3[s] - s2 * -q[s]) / (-0.5 + s2)).collect();
    println!("Z, relativas a e_L: {zrel:?}\npredicción: {pred:?}");
    for s in 0..7 {
        assert!((zrel[s] - pred[s]).abs() < 0.02 * pred[s].abs().max(1.0), "Z, especie {s}: {} frente a {}", zrel[s], pred[s]);
    }
}

/// Test 128: el W¹ estático solo convierte a los levógiros: sin²(φt/2) para ν_L → e_L y u_L → d_L,
/// y nada para e_R.
#[test]
fn t128_charged_current() {
    let phi = 0.05f64;
    for (sp, partner) in [(0u32, 1u32), (3, 4), (2, 1)] {
        let mut u = sm_universe(512, 4, 4, 1, true);
        u.set_yukawa_sm(0.0, 0.0, 0.0);
        u.set_sm_bosons_dynamic(false);
        u.set_sm_link_field(0, 1, phi as f32, 0.0);
        u.inject_sm_packet(sp, 100.0, 1.5, 1.5, 40.0, 0.5, 0.0, 0.0, 0);
        for t in 1..=60 {
            u.step();
            if t == 30 || t == 60 {
                let f = u.species_norm(partner) / (u.species_norm(partner) + u.species_norm(sp));
                if sp == 2 {
                    let other = u.species_norm(0) + u.species_norm(1);
                    println!("e_R, paso {t}: e_L + ν_L = {other:.2e}");
                    assert!(other < 1e-10);
                } else {
                    let ex = (phi * t as f64 / 2.0).sin().powi(2);
                    println!("especie {sp}, paso {t}: {f:.5} frente a {ex:.5}");
                    assert!((f - ex).abs() < 1e-3);
                }
            }
        }
    }
}

/// Test 129: oscilaciones de dos sabores frente a la suma exacta y violación de T con tres sabores.
#[test]
fn t129_flavour_oscillations() {
    let mk = |masses: &[f32], angles: &[f32], delta: f32| {
        let mut u = U::new(4096, 1, 1, 0.5, 0.0, 1.0);
        u.set_scenario(2);
        u.set_flavour_mixing(masses, angles, delta);
        u
    };
    let mut u = mk(&[0.10, 0.16], &[0.6], 0.0);
    u.inject_flavour_packet(1000.0, 300.0, 0.5, 0);
    let mut worst = 0.0f64;
    for t in 1..=600 {
        u.step();
        if t % 100 == 0 {
            let p = u.flavour_norms()[1];
            let ex = u.flavour_transition_exact(0, 1, t as f32, 300.0, 0.5);
            println!("paso {t}: P(e → μ) = {p:.4}, exacta {ex:.4}");
            worst = worst.max((p - ex).abs());
        }
    }
    assert!(worst < 0.005, "error {worst}");
    // tres sabores: asimetría P(e → μ) − P(μ → e)
    let asym = |delta: f32, steps: u32| {
        let ang = [0.58f32, 0.78, 0.40];
        let ms = [0.10f32, 0.13, 0.20];
        let mut a = mk(&ms, &ang, delta);
        a.inject_flavour_packet(1000.0, 300.0, 0.5, 0);
        let mut b = mk(&ms, &ang, delta);
        b.inject_flavour_packet(1000.0, 300.0, 0.5, 1);
        for _ in 0..steps {
            a.step();
            b.step();
        }
        (a.flavour_norms()[1] - b.flavour_norms()[0], a.flavour_jarlskog())
    };
    let (d1, j) = asym(PI as f32 / 2.0, 400);
    let (d0, _) = asym(0.0, 400);
    println!("δ = π/2: asimetría {d1:.4} (J = {j:.4}); δ = 0: {d0:.2e}");
    assert!(d1.abs() > 0.5);
    assert!(d0.abs() < 1e-6);
}

/// Test 130: tres generaciones con matrices de Yukawa (masas, CKM de los autovectores, Jarlskog) y
/// Gauss con todo dinámico en 8³.
#[test]
fn t130_three_generations() {
    let mut u = sm_universe(4, 4, 4, 3, true);
    let m = u.random_yukawa_matrices(2026, false);
    u.set_yukawa_matrices_sm(&m[0..18], &m[18..36], &m[36..54]);
    let ev = u.sm_spectrum(0.0, 0.0, 0.0);
    let mut pos: Vec<f64> = ev.iter().copied().filter(|e| *e > 1e-6).collect();
    pos.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    let nulls = ev.iter().filter(|e| e.abs() < 1e-6).count();
    let sv = [0.05, 0.08, 0.11, 0.17, 0.21, 0.26, 0.37, 0.43, 0.52];
    println!("cuasi-energías positivas {pos:?}; nulas {nulls}");
    assert_eq!(pos.len(), 9);
    assert_eq!(nulls, 6);
    for (a, b) in pos.iter().zip(sv) {
        assert!((a - b).abs() < 1e-6, "{a} frente a {b}");
    }
    let walk = u.sm_walk_ckm();
    let ckm = u.ckm();
    let err = (0..9).map(|k| (walk[k] - ckm[k]).abs()).fold(0.0, f64::max);
    println!("|V| de la caminata {walk:?}\n|W_u†·W_d| {:?}; J = {:.6}", &ckm[..9], ckm[9]);
    assert!(err < 1e-4, "CKM {err}");
    let jc = u.jarlskog_check();
    let rel = (jc[0] - jc[1]).abs() / jc[1].abs();
    println!("det[Y_u·Y_u†, Y_d·Y_d†] = {:.6e}·i (real {:.1e}), 2J·Π = {:.6e}, error relativo {rel:.1e}", jc[0], jc[2], jc[1]);
    assert!(rel < 1e-9 && jc[2].abs() < 1e-9 * jc[0].abs());
    // matrices reales: J = 0
    let mr = u.random_yukawa_matrices(2026, true);
    u.set_yukawa_matrices_sm(&mr[0..18], &mr[18..36], &mr[36..54]);
    assert!(u.ckm()[9].abs() < 1e-7, "J con matrices reales {}", u.ckm()[9]);
    // Gauss con tres generaciones y todo dinámico
    let mut g = sm_universe(8, 8, 8, 3, true);
    g.set_yukawa_matrices_sm(&m[0..18], &m[18..36], &m[36..54]);
    g.perturb_sm_higgs(0.1, 9);
    g.set_packet_norm(2.0);
    for sp in 0..7u32 {
        g.inject_sm_packet(sp, 1.0 + sp as f32, 4.0, 3.0 + 0.5 * sp as f32, 1.5, 0.3, -0.2, 0.1 * sp as f32, sp % 3);
    }
    g.sm_store_gauss_reference();
    let mut worst = [0.0f64; 3];
    for _ in 0..100 {
        g.step();
        let r = g.sm_gauss_residuals();
        for c in 0..3 {
            worst[c] = worst[c].max(r[c]);
        }
    }
    println!("Gauss con tres generaciones: {worst:?}");
    assert!(worst.iter().all(|w| *w < 1e-10));
}
