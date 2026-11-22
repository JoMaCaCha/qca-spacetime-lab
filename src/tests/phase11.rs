//! Fase 11 (gravedad análoga, §2.28 y §3.26, con los añadidos de §2.6 y §2.7): tests 153–161.

use crate::QuantumSpacetimeUniverse as U;

fn analog(w: usize, h: usize, d: usize) -> U {
    let mut u = U::new(w, h, d, 0.5, -0.03, 1.0);
    u.set_scenario(10);
    assert_eq!(u.scenario(), 10, "{}", u.last_message());
    u
}

fn fit_slope(xs: &[f64], ys: &[f64]) -> f64 {
    crate::util::fit_linear(xs, ys).1
}

/// Test 153: la resonancia paramétrica del fondo (bloques de Floquet–Bogoliubov) y su crecimiento
/// en una tira de 4096 celdas con ruido de 10⁻⁹.
#[test]
fn t153_parametric_resonance() {
    let u = analog(64, 64, 1);
    let r1 = u.parametric_rate(0.5, -0.03, 0.5, 1, 4096, 0.0);
    let r2 = u.parametric_rate(1.4, -0.03, 0.5, 1, 4096, 0.0);
    let r3 = u.parametric_rate(0.5, -0.01, 0.5, 2, 1024, 0.0);
    let r4 = u.parametric_rate(1.35, -0.01, 0.5, 1, 16384, 0.0859);
    println!("tasas: {r1:.5e} (2,84e-3), {r2:.3e} (<1e-10), {r3:.5e} (9,45e-4), {r4:.3e} (<1e-10)");
    assert!((r1 / 2.84e-3 - 1.0).abs() < 0.03, "tira m = 0,5: {r1}");
    assert!(r2 < 1e-10, "tira m = 1,4: {r2}");
    assert!((r3 / 9.45e-4 - 1.0).abs() < 0.03, "2D m = 0,5: {r3}");
    assert!(r4 < 1e-10, "tira m = 1,35 con flujo: {r4}");
    // simulación: fondo uniforme de 4096 celdas con ruido de 10⁻⁹
    let mut s = analog(64, 64, 1);
    s.acoustic_strip_init(4096, 0, -0.03, -0.03, 0.0, 0.5);
    s.strip_add_noise(1e-9, 7);
    let (mut ts, mut ls, mut qs) = (Vec::new(), Vec::new(), Vec::new());
    for t in 1..=4000u32 {
        s.step();
        if t >= 1500 && t % 50 == 0 {
            let m = s.strip_dominant_mode();
            ts.push(t as f64);
            ls.push(m[1].ln());
            qs.push(m[0]);
        }
    }
    // ventana de crecimiento lineal: mientras la amplitud relativa sea < 10⁻²
    let end = ls.iter().position(|&l| l > (1e-2f64).ln()).unwrap_or(ls.len());
    let rate = fit_slope(&ts[..end], &ls[..end]);
    let q = qs[end.saturating_sub(1)];
    println!("tira: q = {q:.4} (1,167), tasa = {rate:.4e} (2,84e-3; ref. 2,72e-3), {end} puntos");
    assert!((q - 1.167).abs() < 0.01, "q dominante {q}");
    assert!((rate / 2.84e-3 - 1.0).abs() < 0.08, "tasa {rate}");
}

/// Test 154: láser de agujero negro en la tira (fondo estacionario, BdG por potencia y simulación).
#[test]
fn t154_black_hole_laser() {
    let mut u = analog(64, 64, 1);
    u.acoustic_strip_init(512, 16, -0.02, -0.0025, 10.0, 1.4);
    let info = u.strip_info();
    println!("v = {:.5}, c_sub = {:.5}, c_super = {:.5}, Mach {:.3} y {:.3}, ξ = {:.3}", info[1], info[2], info[3], info[4], info[5], info[6]);
    let norm0 = info[11].sqrt();
    for _ in 0..1000 {
        u.step();
    }
    let dev = u.background_deviation();
    println!("desviación tras 1000 pasos: {dev:.3e} (‖ψ₀‖ = {norm0:.2})");
    assert!(dev < 1e-4 * norm0);
    let rate = u.bdg_power_rate(60000);
    println!("bdg_power_rate = {rate:.5e} (2,851e-4)");
    assert!((rate / 2.85e-4 - 1.0).abs() < 0.02);
    let mut s = analog(64, 64, 1);
    s.acoustic_strip_init(512, 16, -0.02, -0.0025, 10.0, 1.4);
    s.strip_add_noise(1e-7, 3);
    let (mut ts, mut ls) = (Vec::new(), Vec::new());
    for t in 1..=45000u32 {
        s.step();
        if t % 100 == 0 {
            let d = s.background_deviation();
            if (1e-4..1e-2).contains(&d) {
                ts.push(t as f64);
                ls.push(d.ln());
            }
            if d > 1e-2 {
                break;
            }
        }
    }
    let slope = fit_slope(&ts, &ls);
    println!("pendiente de ln(desviación) = {slope:.5e} ({} puntos; ref. 2,861e-4)", ts.len());
    assert!(ts.len() > 20);
    assert!((slope / rate - 1.0).abs() < 0.05);
}

/// Test 155: Hawking espontáneo por lotes (vacío de Wigner truncado propagado sin ruido).
#[test]
fn t155_spontaneous_hawking() {
    let mut u = analog(64, 64, 1);
    u.acoustic_strip_init(512, 7, -0.01, -0.01 / 16.0, 240.0, 1.35);
    let info = u.strip_info();
    let (xh, x2) = (info[8], info[9]);
    println!("v = {:.5}, c_sub = {:.5}, c_super = {:.5}, ξ_sub = {:.3}, horizontes en {xh:.2} y {x2:.2}", info[1], info[2], info[3], info[6]);
    let map = |u: &mut U, engine: u32, t: u32| -> Vec<f64> {
        let p = u.hawking_correlations(engine, 6000, t);
        unsafe { std::slice::from_raw_parts(p, 512 * 512) }.to_vec()
    };
    for t in [12000u32, 16000] {
        let gw = map(&mut u, 0, t);
        let vac = u.hawking_vacuum_info();
        let lobe = u.hawking_lobe(0.0061, 0.0015, 0.0050, 0.0001);
        let gc = map(&mut u, 1, t);
        let (mut num, mut den) = (0.0, 0.0);
        for x in 0..512usize {
            for xp in 0..512usize {
                if (x as f64) < xh && (xp as f64) > xh && (xp as f64) < x2 {
                    num += (gw[x * 512 + xp] - gc[x * 512 + xp]).powi(2);
                    den += gc[x * 512 + xp].powi(2);
                }
            }
        }
        let rel = (num / den).sqrt();
        println!("t = {t}: Tr Σ₀/2N = {:.4} (0,513), caminata frente a continuo en el cuadrante: {:.2} % (3–6 %)", vac[0], 100.0 * rel);
        println!("        lóbulo: mínimo en v_d = {:.4} (0,0030) con ξG/ρ̄ = {:.4e} (t_h = {:.0})", lobe[0], lobe[1], lobe[3]);
        assert!(rel < 0.08);
        if t == 16000 {
            assert!((lobe[0] - 0.0030).abs() <= 0.0005 + 1e-9);
            assert!((lobe[1] / -2.05e-2 - 1.0).abs() < 0.10);
        }
    }
}

/// Test 156: ergorregión y ventanas de la división (BdG radial).
#[test]
fn t156_ergoregion_windows() {
    let mut u = analog(64, 64, 1);
    let a = unsafe { std::slice::from_raw_parts(u.vortex_radial_bdg(2, 9.5, 2), 2) }.to_vec();
    let b = unsafe { std::slice::from_raw_parts(u.vortex_radial_bdg(1, 12.1, 2), 2) }.to_vec();
    let c = unsafe { std::slice::from_raw_parts(u.vortex_radial_bdg(2, 12.1, 2), 2) }.to_vec();
    let d = unsafe { std::slice::from_raw_parts(u.vortex_radial_bdg(3, 14.5, 2), 2) }.to_vec();
    println!("κ = 2, R = 9,5ξ: r_e = {:.4} (3,17), tasa = {:.5} (0,0413)", a[0], a[1]);
    println!("κ = 1, R = 12,1ξ: r_e = {:.4} (1,60), tasa = {:.5}", b[0], b[1]);
    println!("κ = 2, R = 12,1ξ: tasa = {:.5} (< 1e-3)", c[1]);
    println!("κ = 3, R = 14,5ξ: r_e = {:.4} (4,68), tasa = {:.5} (0,030)", d[0], d[1]);
    assert!((a[0] - 3.17).abs() < 0.03);
    assert!((a[1] - 0.0413).abs() < 0.002);
    assert!((b[0] - 1.60).abs() < 0.03);
    assert!(c[1] < 1e-3);
    assert!((d[1] - 0.030).abs() < 0.002);
}

/// Test 157: división del vórtice de carga 2 en la caminata (m = 1,4, ξ = 8, disco de R = 76).
#[test]
fn t157_vortex_splitting_walk() {
    let mut u = analog(160, 160, 1);
    u.set_analog_setup(2);
    assert_eq!(u.analog_setup(), 2, "{}", u.last_message());
    let fi = u.analog_fluid_info(1024);
    println!("m* = {:.3}, ξ = {:.3} (8), c_s = {:.5}, tasa paramétrica 2D = {:.2e}", fi[1], fi[4], fi[5], fi[7]);
    let total = 200_000u32;
    let (mut ts, mut q2, mut worst_dip) = (Vec::new(), Vec::new(), 0.0f64);
    let t0 = std::time::Instant::now();
    for t in 1..=total {
        u.step();
        if t % 2000 == 0 {
            let a2 = u.vortex_multipole(2, 24.0);
            let a1 = u.vortex_multipole(1, 24.0);
            ts.push(t as f64);
            q2.push(a2.ln());
            worst_dip = worst_dip.max(a1 / a2);
            if t % 20000 == 0 {
                println!("paso {t}: cuadrupolo {a2:.4e}, dipolo/cuadrupolo {:.2e} ({:?})", a1 / a2, t0.elapsed());
            }
        }
    }
    let from = ts.iter().position(|&t| t >= 0.7 * total as f64).unwrap();
    let slope = fit_slope(&ts[from..], &q2[from..]);
    let early = fit_slope(&ts[ts.len() / 5..ts.len() / 2], &q2[ts.len() / 5..ts.len() / 2]);
    println!("pendiente del último 30 %: {slope:.4e} (2,79e-5; ref. 2,67e-5); tramo 0,2–0,5: {early:.4e}; dipolo/cuadrupolo máx. {worst_dip:.2e}");
    assert!((slope / 2.79e-5 - 1.0).abs() < 0.06);
    assert!(worst_dip < 1e-3);
}

/// Test 158: reconexión de dos anillos de vórtice en 96³ (m = 0,5, g = −0,03).
#[test]
fn t158_ring_reconnection() {
    let mut u = analog(96, 96, 96);
    assert_eq!(u.analog_setup(), 3, "{}", u.last_message());
    let kappa = 2.0 * std::f64::consts::PI / u.analog_fluid_info(0)[1];
    let mut rec: Vec<(f64, f64)> = Vec::new();
    let mut extra_before = 0.0f64;
    let t0 = std::time::Instant::now();
    let mut zero_steps = Vec::new();
    for t in 1..=760u32 {
        u.step();
        if t % 20 == 0 || (540..=760).contains(&t) {
            let dd = u.vortex_line_distance();
            let info = u.vortex_line_info();
            if dd == 0.0 {
                zero_steps.push(t);
            } else if zero_steps.is_empty() {
                extra_before = extra_before.max(info[2]);
            }
            rec.push((t as f64, dd));
            if t % 100 == 0 || (600..=700).contains(&t) {
                println!("paso {t}: δ = {dd:.3}, componentes {}, puntos {}, fuera {} ({:?})", info[0], info[1], info[2], t0.elapsed());
            }
        }
    }
    assert!(!zero_steps.is_empty(), "no reconectan");
    let tr = 0.5 * (zero_steps[0] as f64 + *zero_steps.last().unwrap() as f64);
    println!("reconexión: pasos con δ = 0 {:?}, t_r = {tr}", zero_steps);
    let fit = |pts: Vec<(f64, f64)>| -> f64 {
        let (mut a, mut b) = (0.0, 0.0);
        for (dt, d) in pts {
            a += d * d * dt;
            b += dt * dt;
        }
        (a / (kappa * b)).sqrt()
    };
    let before: Vec<(f64, f64)> = rec.iter().filter(|(t, d)| *t < tr && *d >= 2.5 && *d <= 10.0).map(|(t, d)| (tr - t, *d)).collect();
    let after: Vec<(f64, f64)> = rec.iter().filter(|(t, d)| *t > tr && *d >= 2.5 && *d <= 10.0).map(|(t, d)| (t - tr, *d)).collect();
    let (nb, na) = (before.len(), after.len());
    let (am, ap) = (fit(before), fit(after));
    println!("A₋ = {am:.3} ({nb} puntos; ref. 0,92), A₊ = {ap:.3} ({na} puntos; ref. 1,6), vórtices fuera antes: {extra_before}");
    assert!(zero_steps[0] >= 660 && zero_steps[0] <= 690);
    assert_eq!(extra_before, 0.0);
    assert!(ap > 1.3 * am);
}

/// Test 159: Unruh de fermiones con el mar lleno (T = 1/2π) y con el mar de Floquet.
#[test]
fn t159_unruh() {
    let mut u = analog(16, 16, 1);
    let lv = unsafe { std::slice::from_raw_parts(u.unruh_occupations(100, 0.0), 20) }.to_vec();
    for k in 0..10 {
        let (x, n) = (lv[2 * k], lv[2 * k + 1]);
        let t = x / ((1.0 - n) / n).ln();
        println!("nivel {k}: ω/T = {x:.4}, n = {n:.5}, T/(1/2π) = {t:.5}");
        assert!((t - 1.0).abs() < 0.005);
    }
    let lv = unsafe { std::slice::from_raw_parts(u.unruh_occupations(100, 0.5), 20) }.to_vec();
    for k in 0..8 {
        let (x, n) = (lv[2 * k], lv[2 * k + 1]);
        let t = x / ((1.0 - n) / n).ln();
        println!("Floquet dt = 0,5, nivel {k}: T/(1/2π) = {t:.5} (0,984)");
        assert!((t - 0.984).abs() < 0.01);
    }
}

/// Test 160: problema radial de Thomas–Fermi con saturación frente a GP–Poisson.
#[test]
fn t160_thomas_fermi_radial() {
    let mut u = analog(16, 16, 1);
    for (rho0, n_ref, ratio_ref) in [(0.15f32, 4087.0, 1.019), (0.3, 6732.0, 1.037), (0.6, 9727.0, 1.083)] {
        let r = unsafe { std::slice::from_raw_parts(u.thomas_fermi_radial(-0.0074, 6.5e-6, 42.3043, rho0), 5) }.to_vec();
        let ratio = r[3] / r[2];
        println!("ρ₀ = {rho0}: N = {:.1} ({n_ref}), borde = {:.2}, R½ TF = {:.3}, R½ GP = {:.3}, cociente = {ratio:.4} ({ratio_ref}), ρ₀ GP = {:.4}", r[0], r[1], r[2], r[3], r[4]);
        assert!((r[0] / n_ref - 1.0).abs() < 0.01);
        assert!((ratio - ratio_ref).abs() < 0.01);
    }
}

/// Test 161 (lento, `--ignored`): núcleo de Thomas–Fermi de la materia oscura autointeractuante en la
/// caminata 3D de 128³ (m = 1,5, g = −0,0074, γ = 6,5·10⁻⁶, ρ₀ de Thomas–Fermi = 0,3).
#[test]
#[ignore]
fn t161_thomas_fermi_core_walk() {
    let mut u = U::new(128, 128, 128, 1.5, -0.0074, 1.0);
    u.set_mass(1.5);
    u.set_g(-0.0074);
    u.set_neighbor_coupling(0.0);
    u.set_capacity(0.0);
    u.set_gravity(1, 6.5e-6);
    let tf = u.seed_thomas_fermi(0.3);
    println!("TF de la caja: N = {:.0} (6925), borde = {:.2} (28,07), R½ = {:.2} (14,96)", tf[0], tf[1], tf[2]);
    let t0 = std::time::Instant::now();
    let mut prev = [0.0f64; 2];
    for k in 0..40 {
        u.relax_imaginary_time(100, 20.0);
        let c = u.core_half_radius();
        println!("tiempo imaginario {}: R½ = {:.3}, ρ₀ = {:.4} ({:?})", 100 * (k + 1), c[0], c[1], t0.elapsed());
        if (c[0] - prev[0]).abs() < 2e-3 && (c[1] - prev[1]).abs() < 1e-4 {
            break;
        }
        prev = [c[0], c[1]];
    }
    let c0 = u.core_half_radius();
    println!("estado fundamental: R½ = {:.3} (15,41), ρ₀ = {:.4} (0,251)", c0[0], c0[1]);
    assert!((c0[0] - 15.4).abs() < 0.3);
    assert!((c0[1] - 0.251).abs() < 0.01);
    for _ in 0..500 {
        u.step();
    }
    let c1 = u.core_half_radius();
    println!("tras 500 pasos: R½ = {:.3}, ρ₀ = {:.4} ({:?})", c1[0], c1[1], t0.elapsed());
    assert!((c1[0] / c0[0] - 1.0).abs() < 0.01);
    assert!((c1[1] / c0[1] - 1.0).abs() < 0.01);
}
