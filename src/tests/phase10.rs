//! Fase 10 (más allá de la caminata de tipo luz, §2.25–§2.27 y §3.23–§3.25): tests 144–152.

use crate::QuantumSpacetimeUniverse as U;
use std::f64::consts::PI;

fn twd(w: usize, m: f32, dt: f32, q: f32, dynamic: bool) -> U {
    let mut u = U::new(w, w, w, 0.5, 0.0, 1.0);
    u.set_scenario(7);
    u.set_scenario_twd(m, 1.0, dt, q, dynamic);
    assert_eq!(u.scenario(), 7, "{}", u.last_message());
    u
}

/// Test 144: un solo Dirac en K = 0 (±dt·m), sin pseudo-dobladores (nada cerca de ±π).
#[test]
fn t144_twd_spectrum() {
    let u = twd(4, 0.1, 0.3, 0.0, false);
    let (mut best, mut best_k, mut far) = (f64::INFINITY, [0usize; 3], f64::INFINITY);
    for ix in 0..16 {
        for iy in 0..16 {
            for iz in 0..16 {
                let k = [ix, iy, iz].map(|i| (2.0 * PI * i as f64 / 16.0) as f32);
                let ev = u.twd_cell_spectrum(k[0], k[1], k[2], false);
                for e in ev {
                    if e.abs() < best - 1e-9 {
                        best = e.abs();
                        best_k = [ix, iy, iz];
                    }
                    far = far.min(PI - e.abs());
                }
            }
        }
    }
    println!("nivel más cercano a 0: {best:.8} en K = {best_k:?} (dt·m = 0,03); distancia mínima a ±π: {far:.4}");
    assert!((best - 0.03).abs() < 1e-6 && best_k == [0, 0, 0]);
    assert!(far > 1.0);
    let e0 = u.twd_cell_spectrum(0.0, 0.0, 0.0, false);
    println!("K = 0: {:?}", e0.iter().map(|e| (e * 1e4).round() / 1e4).collect::<Vec<_>>());
}

/// Test 145: Gauss, norma e invariancia gauge con el U(1) dinámico.
#[test]
fn t145_twd_gauss() {
    let build = || {
        let mut u = twd(8, 0.1, 0.3, 0.5, true);
        for (j, &(x, y, z)) in [(2.0f32, 2.0f32, 2.0f32), (5.0, 4.0, 3.0), (3.0, 6.0, 5.0)].iter().enumerate() {
            u.inject_twd_packet(x, y, z, 1.5, 0.3 * j as f32, -0.2, 0.1, 3.0);
        }
        u.twd_random_field(1.0, 45);
        u
    };
    let mut u = build();
    let mut v = build();
    v.twd_random_gauge(1.0, 7);
    let n0 = u.twd_norm();
    let mut g = 0.0f64;
    for _ in 0..60 {
        u.step();
        v.step();
        g = g.max(u.twd_gauss_residual());
    }
    let n1 = u.twd_norm();
    let dist = u.twd_density_distance(&v);
    println!("norma {n0:.9} → {n1:.9}; Gauss {g:.1e}; copia gauge {dist:.1e}");
    assert!((n1 - n0).abs() < 1e-12 * n0.max(1.0) * 10.0);
    assert!(g < 1e-12);
    assert!(dist < 1e-12);
}

/// Test 146: Higgs → f f̄ con el motor: una especie, 0,79 veces la tasa del continuo.
#[test]
fn t146_twd_higgs_rate() {
    let mut u = twd(4, 0.1, 0.1, 0.0, false);
    let (mf, mh, dm) = (0.1f64, 0.4f64, 1e-3f64);
    let beta = (1.0 - 4.0 * mf * mf / (mh * mh)).sqrt();
    let pred = mh * mh * dm * dm * beta.powi(3) / (16.0 * PI);
    let r = u.twd_pair_rate_higgs(0.1, 0.4, 1e-3, 0.1, 500.0) / pred;
    println!("tasa / continuo = {r:.4} (referencia 0,793; regla de oro de la red 0,781)");
    assert!((r / 0.79 - 1.0).abs() < 0.05);
}

/// Test 147: probabilidad de par por espín de un modo (k⊥ = 0).
#[test]
fn t147_twd_landau_zener() {
    let mut u = twd(4, 0.1, 0.05, 0.0, false);
    let p = u.twd_landau_zener(0.1, 0.01, 0.0, 0.05);
    println!("P = {p:.5} (Wilson–Dirac continuo 0,0503; referencia 0,04962)");
    assert!((p / 0.0503 - 1.0).abs() < 0.03);
}

/// Test 148: reacción en 3D con el mar lleno: el campo débil oscila sin crecer y la energía se conserva.
#[test]
fn t148_twd_backreaction() {
    let mut u = twd(4, 0.3, 0.2, 0.0, false);
    let p = u.twd_homogeneous_run(0.3, 0.3, 0.01, 6, 200.0, 0.2);
    let n = unsafe { *p } as usize;
    let v = unsafe { std::slice::from_raw_parts(p, 1 + 2 * n) };
    let es = &v[1..n + 1];
    let en: Vec<f64> = v[n + 1..].iter().cloned().filter(|x| x.is_finite()).collect();
    let (lo, hi) = es.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(x / 0.01), b.max(x / 0.01)));
    let de = en.iter().map(|e| (e - en[0]).abs()).fold(0.0, f64::max);
    println!("E/E₀ entre {lo:.4} y {hi:.4}; energía {:.9}, deriva {de:.1e}", en[0]);
    assert!(lo > 0.93 && hi < 1.001);
    assert!(de < 1e-6);
}

/// Test 149: la anomalía en 3+1D con E ∥ B, con y sin la masa que cancela la de Wilson del nivel de
/// Landau más bajo.
#[test]
fn t149_twd_anomaly() {
    let mut u = twd(4, 0.0, 0.2, 0.0, false);
    let r = u.twd_anomaly_run(6, 1, 0.02, -0.0873, 0.2, 16, 30.0).abs();
    println!("m = −r·qB/2: |dj/dt| = {r:.4} × (qB/2π)(qE/π) (referencia 0,882)");
    assert!(r > 0.8 && r < 0.95);
    let r0 = u.twd_anomaly_run(6, 1, 0.02, 0.0, 0.2, 16, 30.0).abs();
    println!("m = 0: {r0:.4} (referencia 0,255)");
    assert!(r0 > 0.2 && r0 < 0.35);
}

/// Test 150: el erizo de Skyrme, radial (Adkins, Nappi y Witten) y en la red.
#[test]
fn t150_skyrme_baryon() {
    let mut u = U::new(8, 8, 8, 0.5, 0.0, 1.0);
    u.set_scenario(8);
    let q = unsafe { std::slice::from_raw_parts(u.skyrme_radial_quantization(), 7).to_vec() };
    println!("radial: E/12π² = {:.5}, Λ = {:.3}, e = {:.4}, F_π = {:.2} MeV, M = {:.1} MeV, radio {:.4} fm, C = {:.4}", q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
    assert!((q[0] - 1.2315).abs() < 0.001);
    assert!((q[1] - 51.0).abs() < 0.3);
    assert!((q[2] - 5.45).abs() < 0.03);
    u.skyrme_init(1, 0.15, 5.0);
    let e = u.skyrme_relax(500);
    let b = u.skyrme_charge();
    println!("red (h = 0,15): E/12π² = {e:.5}, B = {b:.4} (referencia 1,22179 y −0,999)");
    assert!((e - 1.222).abs() < 0.005);
    assert!(b.abs() > 0.99);
}

/// Test 151: dos bariones se ligan (energía por barión menor que la del barión solo).
#[test]
fn t151_skyrme_two_baryons() {
    let mut u = U::new(8, 8, 8, 0.5, 0.0, 1.0);
    u.set_scenario(8);
    u.skyrme_init(1, 0.15, 5.5);
    let e1 = u.skyrme_relax(500);
    u.skyrme_init(2, 0.15, 5.5);
    let e2 = u.skyrme_relax(1200);
    let b = u.skyrme_charge();
    let m = u.skyrme_moments();
    println!("B = 2: {e2:.5} por barión frente a {e1:.5} (razón {:.4}; referencia 0,957); B = {b:.4}; momentos {m:?}", e2 / e1);
    assert!(e2 < 0.965 * e1);
    assert!(b.abs() > 1.99);
}

/// Test 152: Floquet anómalo (borde quiral, también con desorden; interior quieto) e índice nulo.
#[test]
fn t152_floquet_anomalous() {
    let mut u = U::new(32, 32, 1, 0.5, 0.0, 1.0);
    u.set_scenario(9);
    for dis in [0.0f32, 1.0] {
        u.floquet_anomalous_init(0, 20, dis);
        u.floquet_inject(0, 11);
        let mut path = Vec::new();
        for _ in 0..4 {
            u.floquet_periods(3);
            let s = u.floquet_max_site();
            path.push((s[0], s[1], s[2]));
        }
        println!("desorden {dis}: {path:?}");
        let last = path[3];
        assert!(last.0 == 12.0 && last.1 == 0.0 && last.2 > 0.999);
        u.floquet_inject(10, 10);
        u.floquet_periods(7);
        let s = u.floquet_max_site();
        println!("  interior: {s:?}");
        assert!(s[0] == 10.0 && s[1] == 10.0 && s[2] > 0.999);
    }
    let c = u.chiral_mode_counts(1, 3, 0.0);
    println!("alfombra, 3 cuantos: modos casi nulos {c:?}");
    assert!(c[2].abs() < 1e-9);
    assert!(u.chiral_index(1, 3, 0.0).abs() < 1e-9);
}
