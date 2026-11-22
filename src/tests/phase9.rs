//! Fase 9 (creación de partículas, §2.23, §2.24, §3.21 y §3.22): tests 131–143.

use crate::QuantumSpacetimeUniverse as U;
use std::f64::consts::PI;

fn lab() -> U {
    U::new(8, 8, 1, 0.5, 0.0, 1.0)
}

fn schwinger_pred(m: f64, qe: f64, c: f64) -> f64 {
    qe / (2.0 * PI) * (-PI * m * m / (c * qe)).exp()
}

/// Test 131: Schwinger con el mar lleno (caminata convencional): el doble de la tasa del continuo
/// (el hueco de Floquet en k = π es un pseudo-doblador) y ocupaciones ≤ 1.
#[test]
fn t131_schwinger_full_sea() {
    let mut u = lab();
    for qe in [0.01, 0.04] {
        let r = u.pair_rate_schwinger(0.1, qe as f32, 4096, 0.0);
        let p = 2.0 * schwinger_pred(0.1, qe, 1.0);
        let occ = u.pair_max_occupation();
        println!("q·E = {qe}: {r:.4e} frente a 2·(qE/2π)·e^(−πm²/qE) = {p:.4e} (cociente {:.4}); ocupación máxima {occ:.3}", r / p);
        assert!((r / p - 1.0).abs() < 0.03);
        assert!(occ <= 1.0 + 1e-12);
    }
}

/// Test 132: Higgs → f f̄ y W¹ → ℓν por nodo (3D, 2000 pasos).
#[test]
fn t132_decays_per_node() {
    let mut u = lab();
    let (mf, mh, dm) = (0.05f64, 0.2f64, 1e-4f64);
    let beta = (1.0 - 4.0 * mf * mf / (mh * mh)).sqrt();
    let ph = mh * mh * dm * dm * beta.powi(3) / (16.0 * PI);
    let h0 = u.pair_rate_decay(0, 0, mf as f32, mh as f32, dm as f32, 2000) / ph;
    let h1 = u.pair_rate_decay(0, 1, mf as f32, mh as f32, dm as f32, 2000) / ph;
    println!("Higgs: nodo 0 {h0:.4}, nodo (0, 0, π/2) {h1:.4} (veces m_H²δm²β³/16π = {ph:.4e})");
    assert!(h0 > 0.97 && h0 < 1.05);
    assert!(h1 < 0.01);
    let pw = 0.2f64.powi(2) * 1e-8 / (96.0 * PI);
    let w0 = u.pair_rate_decay(1, 0, 0.0, 0.2, 1e-4, 2000) / pw;
    let w21 = u.pair_rate_decay(1, 21, 0.0, 0.2, 1e-4, 2000) / pw;
    println!("W: nodo 0 {w0:.4}, nodo (π/2, π/2, π/2) {w21:.4} (veces M²θ₀²/96π = {pw:.4e})");
    assert!(w0 > 0.97 && w0 < 1.05);
    assert!(w21 > 0.97 && w21 < 1.05);
}

/// Test 133: el mar de la caminata convencional en 1D es inerte con cualquier campo.
#[test]
fn t133_inert_sea() {
    let mut u = lab();
    let r = u.sea_inertness(128, 400, 7);
    println!("max |n − 1|, |j| = {r:.2e}");
    assert!(r < 1e-11);
}

/// Test 134: Schwinger de una sola especie con la caminata de Gupta y Short.
#[test]
fn t134_schwinger_one_species() {
    let mut u = lab();
    let c = u.lazy_walk_c_eff(0.1, 0.3);
    println!("c_ef = {c:.5}");
    assert!((c - 0.9388).abs() < 0.001);
    for qe in [0.01, 0.04] {
        let r = u.pair_rate_schwinger(0.1, qe as f32, 4096, 0.3);
        let p = schwinger_pred(0.1, qe, c);
        println!("q·E = {qe}: {r:.4e} frente a {p:.4e} (cociente {:.4})", r / p);
        assert!((r / p - 1.0).abs() < 0.02);
        let r0 = u.pair_rate_schwinger(0.1, qe as f32, 4096, 0.0);
        let c0 = u.lazy_walk_c_eff(0.1, 0.0);
        let p0 = 2.0 * schwinger_pred(0.1, qe, c0);
        println!("  θ = 0: {r0:.4e} frente a 2× {p0:.4e} (cociente {:.4})", r0 / p0);
        assert!((r0 / p0 - 1.0).abs() < 0.03);
    }
}

/// Test 135: la rotura de la cuerda con la caminata de Gupta y Short.
#[test]
fn t135_string_breaking() {
    let mut u = lab();
    let obs = |u: &U| -> Vec<f64> { unsafe { std::slice::from_raw_parts(u.pairs.out.as_ptr(), 5).to_vec() } };
    u.string_breaking_start(128, 0.3, 0.5, 0.4, 48);
    let (mut emax, mut gauss) = (0.0f64, 0.0f64);
    let mut qp = Vec::new();
    for t in 1..=600 {
        u.string_breaking_step(1);
        let o = obs(&u);
        gauss = gauss.max(o[3]);
        if t >= 300 {
            emax = emax.max(o[0]);
        }
        if t >= 300 && t % 100 == 0 {
            qp.push(o[1]);
        }
        if t % 100 == 0 {
            println!("paso {t}: E dentro {:.4}, carga junto a +q {:.4}, junto a −q {:.4}, Gauss {:.1e}, energía del campo {:.3}", o[0], o[1], o[2], o[3], o[4]);
        }
    }
    let pe = u.string_breaking_pairs();
    println!("pares {:.3}, energía de los fermiones {:.4}", pe[0], pe[1]);
    assert!(emax < 0.15, "campo dentro {emax}");
    // la carga junto a +q en los pasos 300, 400, 500 y 600 (la referencia la da muestreada: −0,22 a −0,30)
    assert!(qp.iter().all(|q| *q > -0.35 && *q < -0.15), "carga inducida {qp:?}");
    assert!(gauss < 1e-11);
    // d = 4: solo polarización
    u.string_breaking_start(128, 0.3, 0.5, 0.4, 4);
    let mut emin = f64::INFINITY;
    for t in 1..=600 {
        u.string_breaking_step(1);
        if t >= 300 {
            emin = emin.min(obs(&u)[0]);
        }
    }
    println!("d = 4: campo mínimo en la cuerda {emin:.4}");
    assert!(emin > 0.32);
    // θ = 0: la caminata convencional no rompe la cuerda
    u.string_breaking_start(128, 0.3, 0.0, 0.4, 48);
    let mut dev = 0.0f64;
    for _ in 0..600 {
        u.string_breaking_step(1);
        // q llega en f32: 0,4f32 = 0,40000000596
        dev = dev.max((obs(&u)[0] - 0.4f32 as f64).abs());
    }
    println!("θ = 0: |E − 0,4| ≤ {dev:.1e}");
    assert!(dev < 1e-9);
}

/// Test 136: Z → f f̄ por nodo y quiralidad, y las fracciones de ramificación.
#[test]
fn t136_z_decay() {
    let mut u = lab();
    let p = 0.04 * 1e-8 / (48.0 * PI);
    for node in [0u32, 1] {
        for right in [false, true] {
            let r = u.pair_rate_z(node, right, 0.2, 1e-4, 2000) / p;
            println!("nodo {node}, {}: {r:.4}", if right { "R" } else { "L" });
            assert!(r > 0.97 && r < 1.05);
        }
    }
    let b = unsafe { std::slice::from_raw_parts(u.z_branching_ptr(0.2312), 4).to_vec() };
    println!("fracciones del Z: {b:?}");
    for (x, y) in b.iter().zip([0.2054, 0.6913, 0.1033]) {
        assert!((x - y).abs() < 1e-3);
    }
}

/// Test 137: quarks en un campo de color: la tasa es la suma por autovalores de n·T.
#[test]
fn t137_colour_schwinger() {
    let mut u = lab();
    let c = u.lazy_walk_c_eff(0.1, 0.3);
    let mut rates = Vec::new();
    for gen in [2usize, 7] {
        let mut n = [0.0f32; 8];
        n[gen] = 1.0;
        let lam = u.colour_channel_charges(&n);
        let pred: f64 = lam.iter().filter(|l| l.abs() > 1e-9).map(|l| schwinger_pred(0.1, l.abs() * 0.06, c)).sum();
        let r = u.pair_rate_colour_schwinger(&n, 0.06, 0.1, 0.3, 2048);
        println!("T{}: λ = {lam:?}, tasa {r:.4e} frente a {pred:.4e} (cociente {:.4})", if gen == 2 { "³" } else { "⁸" }, r / pred);
        assert!((r / pred - 1.0).abs() < 0.01);
        rates.push(r);
    }
    assert!(rates[0] > 1.05 * rates[1]);
}

fn sea_run(u: &mut U, m: f32, theta: f32, q: f32, e0: f32, back: bool, steps: u32) -> (Vec<f64>, Vec<f64>) {
    let p = u.homogeneous_sea_run(m, theta, q, e0, back, steps);
    let v = unsafe { std::slice::from_raw_parts(p, 2 * steps as usize).to_vec() };
    (v[..steps as usize].to_vec(), v[steps as usize..].to_vec())
}

/// Test 141: la anomalía quiral de 1+1D con la caminata sin dobladores (dj/dt = c·q·E/π) y su
/// cancelación con la convencional.
#[test]
fn t141_anomaly() {
    let mut u = lab();
    let (j, _) = sea_run(&mut u, 0.0, 0.3, 1.0, 0.002, false, 120);
    let ts: Vec<f64> = (20..120).map(|t| t as f64).collect();
    let (_, slope) = crate::util::fit_linear(&ts, &j[20..120]);
    let pred = 0.3f64.cos() * 0.002 / PI;
    println!("dj/dt = {slope:.6e} frente a c·q·E/π = {pred:.6e} (cociente {:.4})", slope / pred);
    assert!((slope / pred - 1.0).abs() < 0.01);
    let (j0, _) = sea_run(&mut u, 0.0, 0.0, 1.0, 0.002, false, 120);
    let (_, s0) = crate::util::fit_linear(&ts, &j0[20..120]);
    println!("θ = 0: dj/dt = {s0:.2e}");
    assert!(s0.abs() < 1e-12);
}

/// Test 142: el bosón de Schwinger (ω = q·√(c/π), con la corrección del salto de rana) y la
/// polarización del vacío (ε − 1 = q²·c/(6π·m²)).
#[test]
fn t142_schwinger_boson_polarisation() {
    let mut u = lab();
    let (_, e) = sea_run(&mut u, 0.0, 0.3, 0.1, 1e-3, true, 1500);
    let c = 0.3f64.cos();
    let pred = 2.0 * (0.1 * (c / PI).sqrt() / 2.0).asin();
    let w = super::common::peak_frequency(&e, 0.2 * pred, 3.0 * pred);
    println!("ω = {w:.5} frente a 2·arcsen(q·√(c/π)/2) = {pred:.5} (cociente {:.4})", w / pred);
    assert!((w / pred - 1.0).abs() < 0.01);
    let (_, e) = sea_run(&mut u, 0.1, 0.3, 0.1, 1e-4, true, 3000);
    let ce = u.lazy_walk_c_eff(0.1, 0.3);
    let mean = e[500..3000].iter().sum::<f64>() / 2500.0 / 1e-4;
    let pred = 1.0 / (1.0 + 0.01 * ce / (6.0 * PI * 0.01));
    println!("⟨E⟩/E₀ = {mean:.5} frente a 1/(1 + q²c/(6πm²)) = {pred:.5} (cociente {:.4})", mean / pred);
    assert!((mean / pred - 1.0).abs() < 0.01);
}

/// Escenario electrodébil (g = 0,6, g′ = 0,4, κ = 0,25, s = 2) con v y λ dados.
fn ew(w: usize, h: usize, d: usize, v: f32, lambda: f32) -> U {
    let mut u = U::new(w, h, d, 0.5, 0.1, 1.0);
    u.set_capacity(0.0);
    u.set_field_substeps(2);
    u.set_em(2, 0.2, 0.25);
    u.set_electroweak_couplings(0.6, 0.4);
    u.set_higgs(2, v, lambda);
    assert_eq!(u.higgs_mode(), 2);
    u
}

/// Test 138: un modo de W crece con el exponente de Floquet de su capa; un fotón no crece.
#[test]
fn t138_w_mode_floquet() {
    let run = |species: u32| -> f64 {
        let mut u = ew(32, 2, 2, 1.0, 0.09);
        u.set_higgs_oscillation(0.2, 0.0, 1);
        u.seed_gauge_mode(species, 2, 1e-7);
        let mut amp = vec![u.gauge_mode_amplitude(species, 2)];
        for _ in 0..600 {
            u.step();
            amp.push(u.gauge_mode_amplitude(species, 2));
        }
        crate::bosons::envelope_rate(&amp, 300, 600)
    };
    let mut u = ew(32, 2, 2, 1.0, 0.09);
    u.set_higgs_oscillation(0.2, 0.0, 1);
    let fl = u.floquet_exponent(0.15224, 0);
    let w = run(0);
    println!("W¹ (k̂² = 0,1522): {w:.5} frente a Floquet {fl:.5} (cociente {:.4})", w / fl);
    assert!((w / fl - 1.0).abs() < 0.03);
    let g = run(2);
    println!("fotón: {g:.2e}");
    assert!(g.abs() < 1e-3);
}

/// Test 139: los gluones cargados de Nielsen–Olesen crecen con √(κB).
#[test]
fn t139_nielsen_olesen() {
    let mut u = U::new(32, 32, 8, 0.5, 0.1, 1.0);
    u.set_capacity(0.0);
    u.set_gauge_group(2);
    u.set_uniform_colour_b(1, 1e-6);
    let mut en = vec![u.charged_gluon_energy()];
    for _ in 0..400 {
        u.step();
        en.push(u.charged_gluon_energy());
    }
    // Ẽ = 0 al empezar: el suelo es el del ruido tras los primeros pasos
    let floor = en[5..40].iter().cloned().fold(f64::INFINITY, f64::min);
    let max = en.iter().cloned().fold(0.0, f64::max);
    let (mut ts, mut ls) = (Vec::new(), Vec::new());
    for (t, e) in en.iter().enumerate() {
        if *e > 1e3 * floor && *e < 1e-3 * max {
            ts.push(t as f64);
            ls.push(e.ln());
        }
    }
    let rate = 0.5 * crate::util::fit_linear(&ts, &ls).1;
    let pred = (0.25f64 * 4.0 * PI / 1024.0).sqrt();
    let gauss = u.gauss_residual();
    println!("crecimiento {rate:.5} frente a √(κB) = {pred:.5} (cociente {:.4}); {} puntos; Gauss {gauss:.1e}; energía {floor:.1e} → {max:.3}", rate / pred, ts.len());
    assert!((rate / pred - 1.0).abs() < 0.02);
    assert!(gauss < 1e-12);
}

/// Test 140: el W de Ambjørn–Olesen crece con √(κB − M_W²) por debajo de v_c y no crece por encima.
#[test]
fn t140_ambjorn_olesen() {
    let b = 4.0 * PI / 1024.0;
    let mut u = ew(32, 32, 8, 0.15, 0.5);
    u.set_uniform_em_b(1, 0.0);
    u.step();
    let f = u.field_max_abs();
    println!("fuerza del fondo (U(1)_Y compacta): {f:.1e}");
    assert!(f < 1e-12);
    u.set_uniform_em_b(1, 1e-6);
    let mut en = vec![u.charged_w_energy()];
    for _ in 0..500 {
        u.step();
        en.push(u.charged_w_energy());
    }
    // Ẽ = 0 al empezar: el suelo es el del ruido tras los primeros pasos
    let floor = en[5..40].iter().cloned().fold(f64::INFINITY, f64::min);
    let max = en.iter().cloned().fold(0.0, f64::max);
    let (mut ts, mut ls) = (Vec::new(), Vec::new());
    for (t, e) in en.iter().enumerate() {
        if *e > 1e3 * floor && *e < 1e-3 * max.max(1e3 * floor * 1e3) {
            ts.push(t as f64);
            ls.push(e.ln());
        }
    }
    let rate = 0.5 * crate::util::fit_linear(&ts, &ls).1;
    let mw2 = 0.25 * 0.36 * 0.0225 / 2.0;
    let pred = (0.25 * b - mw2).sqrt();
    println!("v = 0,15: {rate:.5} frente a √(κB − M_W²) = {pred:.5} (cociente {:.4}); {} puntos; Φ₁ máx {:.2e}", rate / pred, ts.len(), u.charged_higgs_max());
    assert!((rate / pred - 1.0).abs() < 0.03);
    let mut u = ew(32, 32, 8, 0.3, 0.5);
    u.set_uniform_em_b(1, 1e-6);
    // Ẽ = 0 al empezar: la referencia es la energía media del ruido en los pasos 10–30
    let mut en = Vec::new();
    for _ in 0..500 {
        u.step();
        en.push(u.charged_w_energy());
    }
    let e0 = en[10..30].iter().sum::<f64>() / 20.0;
    let emax = en[30..].iter().cloned().fold(0.0, f64::max);
    println!("v = 0,3: energía máxima / la de los pasos 10–30 = {:.3}", emax / e0);
    assert!(emax < 2.0 * e0);
}

/// Test 143: el condensado de W produce Higgs en la resonancia ω_k = M_W.
#[test]
fn t143_higgs_from_w() {
    let mut u = ew(64, 2, 2, 1.0, 0.15f32 * 0.15 / 4.0);
    u.set_w_condensate(0.15, 3, 1e-7);
    u.step_n(4000);
    let r = u.w_condensate_rates(200, 3800);
    println!("m_H = 0,15: red {:.5}, integración lineal {:.5} (cociente {:.4})", r[0], r[1], r[0] / r[1]);
    assert!((r[0] / r[1] - 1.0).abs() < 0.10);
    let mut u = ew(64, 2, 2, 1.0, 0.25f32 * 0.25 / 4.0);
    u.set_w_condensate(0.15, 3, 1e-7);
    u.step_n(4000);
    let s = u.w_condensate_series();
    let steps = s[0] as usize;
    let amp = &s[1..steps + 2];
    let growth = amp.iter().cloned().fold(0.0, f64::max) / amp[0];
    println!("m_H = 0,25: la amplitud crece {growth:.3} veces");
    assert!(growth < 2.0);
}
