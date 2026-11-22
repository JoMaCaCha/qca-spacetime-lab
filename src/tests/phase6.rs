//! Fase 6 (mecanismo de Higgs, paredes de dominio y Yukawa, §2.12, §2.13, §3.9, §3.10 y §3.16):
//! tests 56–71, 78–80, 103, 108–110, 121 y 124. El Higgs vive en f64 (umbrales "en f64" de §5).

use super::common::{amplitudes_at, leapfrog_omega, peak_frequency, spectrum_peaks};
use crate::util::min_image;
use crate::QuantumSpacetimeUniverse as U;
use std::f64::consts::PI;

/// Universo con el U(1) dinámico (q, κ = 0,25 y s subpasos), listo para `set_higgs`.
fn gauge_universe(w: usize, h: usize, d: usize, q: f32, s: u32) -> U {
    let mut u = U::new(w, h, d, 0.5, 0.1, 1.0);
    u.set_capacity(0.0);
    u.set_field_substeps(s);
    u.set_em(2, q, 0.25);
    u
}

fn abs_phi(u: &U) -> Vec<f64> {
    (0..u.n).map(|i| u.higgs.phi[2 * i].hypot(u.higgs.phi[2 * i + 1])).collect()
}

/// Test 57: Gauss e invariancia gauge del Higgs abeliano.
#[test]
fn t57_higgs_gauss_gauge() {
    for &(w, h, d, s) in &[(48usize, 48usize, 1usize, 1u32), (16, 16, 16, 2)] {
        let build = |gauge: bool| -> U {
            let mut u = gauge_universe(w, h, d, 0.3, s);
            u.set_higgs(1, 1.0, 0.05);
            u.seed_higgs_random(0.3, 0.1, 57);
            if gauge {
                u.apply_random_gauge_transform(3.0, 5);
            }
            u
        };
        let mut a = build(false);
        let mut b = build(true);
        a.step_n(1000);
        b.step_n(1000);
        let ga = a.gauss_residual();
        let gb = b.gauss_residual();
        assert!(ga < 1e-12 && gb < 1e-12, "{w}×{h}×{d}: deriva de Gauss {ga:e}, {gb:e}");
        let d_abs = abs_phi(&a).iter().zip(abs_phi(&b)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(d_abs < 1e-9, "{w}×{h}×{d}: |φ| difiere {d_abs:e}");
    }
}

/// Frecuencia medida y predicha de una onda estacionaria (256×4, s = 1).
fn standing_wave(kind: u32, k_index: u32, q: f32, lambda: f32) -> (f64, f64) {
    let mut u = gauge_universe(256, 4, 1, q, 1);
    u.set_higgs(1, 1.0, lambda);
    u.seed_higgs_wave(kind, k_index, 1e-3);
    let mut ys = Vec::with_capacity(3000);
    for _ in 0..3000 {
        ys.push(match kind {
            0 => u.link_a(0, 0, 0, 1),
            1 => u.higgs_phi_at(0, 0, 0, 0) - 1.0,
            _ => u.higgs_phi_at(0, 0, 0, 1).atan2(u.higgs_phi_at(0, 0, 0, 0)),
        });
        u.step();
    }
    let m = u.higgs_masses();
    let mass = match kind {
        1 => m[1],
        _ => m[0],
    };
    let k = 2.0 * PI * k_index as f64 / 256.0;
    let pred = leapfrog_omega(0.25, k, mass, 1.0);
    (peak_frequency(&ys, 0.3 * pred, 2.0 * pred), pred)
}

/// Test 58: masas abelianas (fotón, Higgs y fase; con q = 0, Goldstone sin masa).
#[test]
fn t58_abelian_masses() {
    for &ki in &[1u32, 4, 16] {
        for kind in 0..3 {
            let (w, p) = standing_wave(kind, ki, 0.1, 0.005);
            println!("t58: k = {ki}, tipo {kind}: {w:.5} frente a {p:.5}");
            assert!((w / p - 1.0).abs() < 5e-3, "q = 0,1, tipo {kind}, k = {ki}: ω = {w}, predicción {p}");
        }
        let (w, p) = standing_wave(2, ki, 0.0, 0.005);
        assert!((w / p - 1.0).abs() < 5e-3, "q = 0, fase, k = {ki}: ω = {w}, predicción {p}");
    }
}

/// Relaja n cuantos de flujo uniforme en 128² (q = 0,1, energía compacta, η = 0,02).
fn relaxed_flux(lambda: f32, n_flux: i32, steps: u32) -> U {
    let mut u = gauge_universe(128, 128, 1, 0.1, 1);
    u.set_compact_magnetic_energy(true);
    u.set_higgs(1, 1.0, lambda);
    u.set_uniform_flux(n_flux);
    u.add_higgs_noise(0.01, 59);
    u.set_relaxation(0.02);
    u.step_n(steps);
    u
}

/// Vórtices [(x, y, devanado)].
fn vortices(u: &mut U) -> Vec<(f64, f64, i32)> {
    let n = u.higgs_vortex_count() as usize;
    (0..n).map(|k| (u.higgs.vortices[3 * k] as f64, u.higgs.vortices[3 * k + 1] as f64, u.higgs.vortices[3 * k + 2] as i32)).collect()
}

/// Flujo q·Σ B/(2π) en las plaquetas a menos de r de (x, y).
fn flux_within(u: &U, x: f64, y: f64, r: f64) -> f64 {
    let mut s = 0.0;
    for py in 0..u.height() {
        for px in 0..u.width() {
            let dx = min_image(px as f64 + 0.5 - x, u.width() as f64);
            let dy = min_image(py as f64 + 0.5 - y, u.height() as f64);
            if dx * dx + dy * dy < r * r {
                s += u.higgs_plaquette_b(px as i32, py as i32, 0, 0, 1);
            }
        }
    }
    s * u.em_q() / (2.0 * PI)
}

const BPS_UNIT: f64 = 2.0 * PI * 0.25; // 2π·c²·v² con c² = 0,25 y v = 1

/// Test 59: Meissner, un vórtice con un cuanto de flujo, y la cota de Bogomolny.
#[test]
fn t59_meissner_bogomolny() {
    let mut u = relaxed_flux(0.00125, 1, 6000);
    let vs = vortices(&mut u);
    assert_eq!(vs.len(), 1, "vórtices: {vs:?}");
    assert_eq!(vs[0].2, 1);
    let fl = flux_within(&u, vs[0].0, vs[0].1, 63.0);
    assert!((fl - 1.0).abs() < 5e-3, "flujo {fl}");
    let e = u.higgs_energy() / BPS_UNIT;
    assert!((e - 1.0).abs() < 0.01, "Bogomolny: E = {e}");
    let mut u2 = relaxed_flux(0.005, 1, 6000);
    let e2 = u2.higgs_energy() / BPS_UNIT;
    assert!(e2 > 1.2, "tipo II: E = {e2}");
    let mut u1 = relaxed_flux(0.0003, 1, 6000);
    let e1 = u1.higgs_energy() / BPS_UNIT;
    assert!(e1 < 0.85, "tipo I: E = {e1}");
    println!("t59: flujo {fl:.5}, E/(2πc²v²) = {e:.5} (Bogomolny), {e2:.4} (tipo II), {e1:.4} (tipo I)");
}

/// Test 60: tipo I y tipo II con 4 cuantos.
#[test]
fn t60_type_one_two() {
    let iso = |lam: f32| relaxed_flux(lam, 1, 6000).higgs_energy();
    let mut u = relaxed_flux(0.005, 4, 12000);
    let vs = vortices(&mut u);
    assert!(vs.len() == 4 && vs.iter().all(|v| v.2 == 1), "tipo II: {vs:?}");
    let e2 = u.higgs_energy() / 4.0;
    let i2 = iso(0.005);
    assert!(e2 >= i2 * (1.0 - 1e-3), "tipo II: {e2} frente a {i2}");
    let mut u1 = relaxed_flux(0.0003, 4, 12000);
    let e1 = u1.higgs_energy() / 4.0;
    let i1 = iso(0.0003);
    assert!(e1 < 0.95 * i1, "tipo I: {e1} frente a {i1}");
    let mut ub = relaxed_flux(0.00125, 4, 12000);
    let eb = ub.higgs_energy() / 4.0;
    let ib = iso(0.00125);
    assert!((eb / ib - 1.0).abs() < 2e-3, "Bogomolny: {eb} frente a {ib}");
    println!("t60: por cuanto/aislado: tipo II {:.4} ({:.4}/{:.4}), tipo I {:.4}, Bogomolny {:.5}", e2 / i2, e2 / BPS_UNIT, i2 / BPS_UNIT, e1 / i1, eb / ib);
}

/// Energía relajada de un par de monopolos externos a distancia d (48³, q = 0,2, λ = 0,005, s = 2).
fn monopole_pair_energy(v: f32, d: u32) -> (f64, f64) {
    let mut u = gauge_universe(48, 48, 48, 0.2, 2);
    u.set_higgs(1, v, 0.005);
    u.add_monopole_pair(24, 24, 24 - d as i32 / 2, d);
    u.set_relaxation(0.03);
    u.step_n(3000);
    let flux = u.higgs_plane_flux(24);
    (u.higgs_energy(), flux)
}

/// Residuos rms de los ajustes lineal (a + b·d) y de Coulomb (a + b/d).
fn fit_residuals(ds: &[f64], es: &[f64]) -> (f64, f64) {
    let res = |xs: &[f64]| -> f64 {
        let (a, b) = crate::util::fit_linear(xs, es);
        (xs.iter().zip(es).map(|(x, e)| (e - a - b * x).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
    };
    let inv: Vec<f64> = ds.iter().map(|d| 1.0 / d).collect();
    (res(ds), res(&inv))
}

/// Test 61: monopolos confinados por un tubo de flujo en la fase de Higgs; Coulomb sin Higgs.
#[test]
fn t61_confined_monopoles() {
    // tensión del vórtice 2D con los mismos parámetros (q = 0,2, λ = 0,005: punto de Bogomolny)
    let mut v2 = gauge_universe(96, 96, 1, 0.2, 2);
    v2.set_compact_magnetic_energy(true);
    v2.set_higgs(1, 1.0, 0.005);
    v2.set_uniform_flux(1);
    v2.add_higgs_noise(0.01, 61);
    v2.set_relaxation(0.03);
    v2.step_n(3000);
    let tension = v2.higgs_energy();
    let ds = [8.0, 16.0, 24.0];
    let mut eh = Vec::new();
    let mut ec = Vec::new();
    for &d in &ds {
        let (e, f) = monopole_pair_energy(1.0, d as u32);
        assert!((f - 1.0).abs() < 1e-6, "flujo por el plano medio {f}");
        eh.push(e);
        let (e0, f0) = monopole_pair_energy(0.0, d as u32);
        assert!((f0 - 1.0).abs() < 1e-6, "flujo por el plano medio (v = 0) {f0}");
        ec.push(e0);
    }
    let slope = (eh[2] - eh[1]) / 8.0;
    println!("t61: Higgs E = {eh:?}, pendiente {slope:.4} frente a T = {tension:.4}; Coulomb E = {ec:?}");
    assert!((slope / tension - 1.0).abs() < 0.05, "pendiente {slope} frente a la tensión {tension}");
    let (lin_h, cou_h) = fit_residuals(&ds, &eh);
    assert!(lin_h < cou_h, "Higgs: residuo lineal {lin_h} frente a Coulomb {cou_h}");
    assert!(ec[2] - ec[1] < 0.6 * (ec[1] - ec[0]), "v = 0: {ec:?}");
    let (lin_c, cou_c) = fit_residuals(&ds, &ec);
    assert!(cou_c < lin_c, "v = 0: residuo de Coulomb {cou_c} frente al lineal {lin_c}");
}

/// Universo con el doblete electrodébil (κ = 0,25, s subpasos).
fn electroweak(w: usize, h: usize, d: usize, g: f32, gp: f32, lambda: f32, s: u32) -> U {
    let mut u = gauge_universe(w, h, d, 0.0, s);
    u.set_electroweak_couplings(g, gp);
    u.set_higgs(2, 1.0, lambda);
    u
}

/// Test 62: las dos leyes de Gauss del doblete se conservan.
#[test]
fn t62_electroweak_gauss() {
    for &(w, h, d, s) in &[(32usize, 32usize, 1usize, 1u32), (12, 12, 12, 2)] {
        let mut u = electroweak(w, h, d, 0.6, 0.4, 0.02, s);
        u.seed_higgs_random(0.3, 0.1, 62);
        u.step_n(300);
        let r = u.higgs_gauss_residuals();
        println!("t62: {w}×{h}×{d}: deriva SU(2) {:e}, U(1) {:e}", r[0], r[1]);
        assert!(r[0] < 1e-11 && r[1] < 1e-11, "{w}×{h}×{d}: Gauss {r:?}");
    }
}

/// Serie temporal de una sonda en x = 0 tras sembrar una onda electrodébil (256×4, 4000 pasos).
/// probe: 0 = Ẽ¹_y, 1 = Ẽ³_y, 2 = E_B,y, 3 = Re Φ₂ − v.
fn ew_series(kind: u32, k_index: u32, probe: u32) -> (Vec<f64>, Vec<f64>) {
    let mut u = electroweak(256, 4, 1, 0.3, 0.18, 0.0056, 1);
    u.seed_higgs_wave(kind, k_index, 1e-3);
    let mut ys = Vec::with_capacity(4000);
    for _ in 0..4000 {
        ys.push(match probe {
            0 => u.colour_link_e(0, 0, 0, 1, 0),
            1 => u.colour_link_e(0, 0, 0, 1, 2),
            2 => u.link_e(0, 0, 0, 1),
            _ => u.higgs_phi_at(0, 0, 0, 2) - 1.0,
        });
        u.step();
    }
    (ys, u.higgs_masses())
}

/// Test 63: masas de W, Z, fotón y Higgs, y la mezcla de Weinberg.
#[test]
fn t63_electroweak_masses() {
    for &ki in &[1u32, 4] {
        let k = 2.0 * PI * ki as f64 / 256.0;
        let (ys, m) = ew_series(0, ki, 0);
        let pw = leapfrog_omega(0.25, k, m[0], 1.0);
        let w = peak_frequency(&ys, 0.5 * pw, 1.5 * pw);
        let (ys, _) = ew_series(3, ki, 3);
        let ph = leapfrog_omega(0.25, k, m[3], 1.0);
        let hgs = peak_frequency(&ys, 0.5 * ph, 1.5 * ph);
        let (yb, _) = ew_series(2, ki, 2);
        let pg = leapfrog_omega(0.25, k, 0.0, 1.0);
        let pz = leapfrog_omega(0.25, k, m[1], 1.0);
        let wg = peak_frequency(&yb, 0.5 * pg, 1.5 * pg);
        let wz = peak_frequency(&yb, 0.8 * pz, 1.2 * pz);
        println!("t63: k = {ki}: W {w:.5}/{pw:.5}, H {hgs:.5}/{ph:.5}, γ {wg:.5}/{pg:.5}, Z {wz:.5}/{pz:.5}, cos θ_W = {:.4}", m[2]);
        for (a, b, lab) in [(w, pw, "W"), (hgs, ph, "Higgs"), (wg, pg, "fotón"), (wz, pz, "Z")] {
            assert!((a / b - 1.0).abs() < 5e-3, "k = {ki}, {lab}: {a} frente a {b}");
        }
        let c2 = m[2] * m[2];
        let s2 = 1.0 - c2;
        let amp = amplitudes_at(&yb, &[wg, wz]);
        let rb = amp[0] / amp[1];
        let pb = c2 / s2 * wg / wz;
        let (y3, _) = ew_series(1, ki, 1);
        let amp3 = amplitudes_at(&y3, &[wg, wz]);
        let r3 = amp3[0] / amp3[1];
        let p3 = s2 / c2 * wg / wz;
        println!("t63: mezcla: B {rb:.4} frente a {pb:.4}; W³ {r3:.4} frente a {p3:.4}");
        assert!((rb / pb - 1.0).abs() < 0.05, "onda de B: {rb} frente a {pb}");
        assert!((r3 / p3 - 1.0).abs() < 0.05, "onda de W³: {r3} frente a {p3}");
    }
}

/// Escenario Dirac 2D sin campo gauge (m global 0,3).
fn dirac(w: usize, h: usize) -> U {
    let mut u = U::new(w, h, 1, 0.5, 0.0, 1.0);
    u.set_scenario(2);
    u.set_mass(0.3);
    u
}

/// Estados [ω, x, decaimiento, fracción cerca] de la salida plana de `dirac_*_modes`.
fn modes(v: &[f64]) -> Vec<[f64; 4]> {
    v.chunks(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
}

/// Test 64: un modo quiral por pared, con velocidad cos m₀ y decaimiento atanh(sin m₀).
#[test]
fn t64_wall_spectrum() {
    let u = dirac(8, 8);
    for &m0 in &[0.1f32, 0.3, 0.5] {
        let near = 2.5 / m0;
        let kys: Vec<f64> = (0..7).map(|j| 0.6 * m0 as f64 * (j as f64 - 3.0) / 3.0).collect();
        let mut branch = [Vec::new(), Vec::new()];
        let mut decay = [Vec::new(), Vec::new()];
        for &ky in &kys {
            let st = modes(&u.dirac_strip_modes(96, m0, -m0, 48.0, ky as f32, 0.9 * m0, near));
            for (k, wall) in [24.0f64, 72.0].iter().enumerate() {
                let at: Vec<&[f64; 4]> = st.iter().filter(|s| min_image(s[1] - wall, 96.0).abs() < near as f64).collect();
                assert_eq!(at.len(), 1, "m₀ = {m0}, k_y = {ky}, pared {wall}: {st:?}");
                branch[k].push(at[0][0]);
                decay[k].push(at[0][2]);
            }
            assert_eq!(st.len(), 2, "m₀ = {m0}, k_y = {ky}: estados en el hueco {st:?}");
        }
        let v: Vec<f64> = (0..2).map(|k| crate::util::fit_linear(&kys, &branch[k]).1).collect();
        let dec: Vec<f64> = (0..2).map(|k| decay[k].iter().sum::<f64>() / decay[k].len() as f64).collect();
        let (vc, dc) = ((m0 as f64).cos(), (m0 as f64).sin().atanh());
        println!("t64: m₀ = {m0}: velocidades {v:?} (cos m₀ = {vc:.4}), decaimientos {dec:?} (atanh sin m₀ = {dc:.4})");
        assert!(v[0] * v[1] < 0.0, "las dos paredes deben ir en sentidos opuestos");
        for k in 0..2 {
            assert!((v[k].abs() / vc - 1.0).abs() < 0.01, "m₀ = {m0}: velocidad {}", v[k]);
            assert!((dec[k] / dc - 1.0).abs() < 0.03, "m₀ = {m0}: decaimiento {}", dec[k]);
        }
    }
    // control: canal de m = 0 dentro de m = 0,3 (mismo signo), sin estados con |ω| < 0,1
    let st = modes(&u.dirac_strip_modes(96, 0.3, 0.0, 10.0, 0.0, 0.1, 5.0));
    let low: Vec<f64> = u.dirac_strip_spectrum(96, 0.3, 0.0, 10.0, 0.0).into_iter().filter(|w| w.abs() < 0.2).collect();
    println!("t64: control, estados con |ω| < 0,2: {low:?}");
    assert!(st.is_empty(), "control: {st:?}");
}

/// Centroide circular en y y desviación (en y) de la densidad a menos de `half` celdas de x.
fn wall_moments(u: &U, x: i32, half: i32) -> (f64, f64) {
    let (w, h) = (u.width() as i32, u.height() as f64);
    let (mut sc, mut ss, mut tot) = (0.0, 0.0, 0.0);
    let mut cells = Vec::new();
    for y in 0..u.height() as i32 {
        for dx in -half..=half {
            let e = u.density_at((x + dx).rem_euclid(w), y, 0) as f64;
            let th = 2.0 * PI * y as f64 / h;
            sc += e * th.cos();
            ss += e * th.sin();
            tot += e;
            cells.push((y as f64, e));
        }
    }
    let yc = (ss.atan2(sc) / (2.0 * PI) * h).rem_euclid(h);
    let var = cells.iter().map(|(y, e)| e * min_image(y - yc, h).powi(2)).sum::<f64>() / tot.max(1e-300);
    (yc, var.sqrt())
}

/// Velocidad (en y) de la parte en la pared, fracción en la pared y anchuras inicial y final.
fn run_wall_packet(u: &mut U, x: i32, steps: usize) -> (f64, f64, f64, f64) {
    let (y0, s0) = wall_moments(u, x, 8);
    let (mut ts, mut ys) = (vec![0.0], vec![y0]);
    let mut acc = y0;
    let mut last = y0;
    for t in 1..=steps {
        u.step();
        if t % 4 == 0 {
            let (yc, _) = wall_moments(u, x, 8);
            acc += min_image(yc - last, u.height() as f64);
            last = yc;
            ts.push(t as f64);
            ys.push(acc);
        }
    }
    let (_, s1) = wall_moments(u, x, 8);
    (crate::util::fit_linear(&ts, &ys).1, u.wall_fraction(x, 8), s0, s1)
}

/// Test 65: propagación en un solo sentido.
#[test]
fn t65_one_way() {
    let mut vs = Vec::new();
    for &x in &[64i32, 0] {
        let mut u = dirac(128, 128);
        u.set_domain_walls(0.3);
        u.inject_wall_packet(x, 64.0, 8.0, 0.0);
        let (v, frac, s0, s1) = run_wall_packet(&mut u, x, 160);
        println!("t65: pared en x = {x}: v = {v:.4}, en la pared {frac:.3}, anchura {s0:.2} → {s1:.2}");
        assert!((v.abs() / (0.3f64).cos() - 1.0).abs() < 0.01, "velocidad {v}");
        assert!(s1 < 1.2 * s0, "anchura {s0} → {s1}");
        assert!(frac > 0.9, "fracción en la pared {frac}");
        vs.push(v);
    }
    assert!(vs[0] * vs[1] < 0.0, "sentidos {vs:?}");
}

/// Configura la línea: pared quiral (paredes ±0,3) o canal no quiral (m = 0 en 10 celdas, 0,5 fuera).
fn line_setup(chiral: bool) -> U {
    let mut u = dirac(128, 256);
    if chiral {
        u.set_domain_walls(0.3);
    } else {
        u.set_dirac_mass_shape(3, 0.5, 10.0, 1.0);
    }
    u
}

/// Fracciones reflejada y transmitida por un obstáculo (σ = 3) a 40 celdas por delante del paquete
/// (128×256, 120 pasos), contando la norma a menos de 10 celdas de la línea.
fn obstacle(chiral: bool, v: f32) -> (f64, f64) {
    let ky = if chiral { 0.0 } else { 0.3 };
    // sentido de avance: unos pasos de prueba en una copia sin obstáculo
    let mut probe = line_setup(chiral);
    probe.inject_wall_packet(64, 128.0, 8.0, ky);
    let (y0, _) = wall_moments(&probe, 64, 10);
    probe.step_n(20);
    let (y1, _) = wall_moments(&probe, 64, 10);
    let dir = min_image(y1 - y0, 256.0).signum();
    let yo = 128.0 + 40.0 * dir;
    let mut u = line_setup(chiral);
    u.inject_wall_packet(64, 128.0, 8.0, ky);
    u.add_gaussian_potential(64.0, yo as f32, 0.0, 3.0, v);
    u.step_n(120);
    let (mut refl, mut tran, mut tot) = (0.0, 0.0, 0.0);
    for y in 0..256 {
        for x in 0..128 {
            let e = u.density_at(x, y, 0) as f64;
            tot += e;
            if (x - 64).abs() > 10 {
                continue;
            }
            let ahead = min_image(y as f64 - yo, 256.0) * dir;
            let behind_start = min_image(y as f64 - 128.0, 256.0) * dir;
            if ahead > 3.0 {
                tran += e;
            } else if behind_start < -5.0 {
                refl += e;
            }
        }
    }
    (refl / tot, tran / tot)
}

/// Test 66: sin retrodispersión en la pared quiral; el canal no quiral sí refleja.
#[test]
fn t66_no_backscattering() {
    for &v in &[0.3f32, 0.6] {
        let (r, t) = obstacle(true, v);
        println!("t66: pared quiral, V = {v}: reflejado {r:.5}, transmitido {t:.4}");
        assert!(r < 1e-3 && t > 0.9, "pared quiral, V = {v}: {r}, {t}");
    }
    let (r, t) = obstacle(false, 0.6);
    println!("t66: canal no quiral, V = 0,6: reflejado {r:.4}, transmitido {t:.4}");
    assert!(r > 0.3, "canal no quiral: reflejado {r}");
}

/// Kink (x = 32) y antikink (x = 96) del Higgs de Yukawa relajados (128², y = 0,3, v = 1,
/// m_H = 0,3, η = 0,05, 3000 pasos).
fn relaxed_kinks() -> U {
    let mut u = dirac(128, 128);
    u.set_yukawa(2, 0.3, 1.0, 0.3);
    u.seed_higgs_kinks(32, 96, 0.0);
    u.set_relaxation(0.05);
    u.step_n(3000);
    u.set_relaxation(0.0);
    u
}

/// Velocidad de la rama quiral de cada kink (pendiente dω/dk_y del estado en el hueco más cercano).
fn kink_branch_velocity(u: &U, x: f64) -> f64 {
    let at = |ky: f32| -> f64 {
        let st = modes(&u.dirac_row_modes(64, ky, 0.25, 8.0));
        st.iter().filter(|s| min_image(s[1] - x, 128.0).abs() < 10.0).min_by(|a, b| a[0].abs().partial_cmp(&b[0].abs()).unwrap()).map(|s| s[0]).unwrap_or(f64::NAN)
    };
    (at(0.05) - at(-0.05)) / 0.1
}

/// Test 67: masa de Yukawa estática, anchura del kink y una rama quiral por kink.
#[test]
fn t67_yukawa_static() {
    let u = relaxed_kinks();
    let w1 = u.yukawa_kink_width(32.0);
    let w2 = u.yukawa_kink_width(96.0);
    println!("t67: anchuras {w1:.4}, {w2:.4} (2c/m_H = 6,6667)");
    for w in [w1, w2] {
        assert!((w / (2.0 / 0.3) - 1.0).abs() < 0.02, "anchura {w}");
    }
    let gap = u.dirac_strip_spectrum(96, 0.3, 0.3, 0.0, 0.0).iter().map(|w| w.abs()).fold(f64::INFINITY, f64::min);
    assert!((gap / 0.3 - 1.0).abs() < 0.01, "hueco {gap}");
    let v1 = kink_branch_velocity(&u, 32.0);
    let v2 = kink_branch_velocity(&u, 96.0);
    println!("t67: hueco {gap:.5}; velocidades de las ramas {v1:.4}, {v2:.4}");
    assert!(v1 * v2 < 0.0, "sentidos opuestos: {v1}, {v2}");
    for v in [v1, v2] {
        assert!((v.abs() - 0.99).abs() < 0.02, "velocidad {v}");
    }
}

/// Test 68: un fermión sobre el kink dinámico (con la reacción sobre el Higgs).
#[test]
fn t68_dynamic_kink() {
    let mut u = relaxed_kinks();
    // el kink cuya rama va hacia −y
    let x = if kink_branch_velocity(&u, 32.0) < 0.0 { 32 } else { 96 };
    let phi0: Vec<f64> = (0..128).map(|xx| u.yukawa_phi_at(xx, 64)).collect();
    u.inject_wall_packet(x, 64.0, 8.0, 0.0);
    let (v, frac, _, _) = run_wall_packet(&mut u, x, 300);
    let deform = (0..128).map(|xx| (u.yukawa_phi_at(xx, 64) - phi0[xx as usize]).abs()).fold(0.0, f64::max);
    println!("t68: v = {v:.4}, deformación {deform:e}, ligado {frac:.3}");
    assert!((v + 0.99).abs() < 0.02, "velocidad {v}");
    assert!(deform < 0.01, "deformación {deform}");
    assert!(frac > 0.85, "ligado {frac}");
}

/// Test 69: el kink en movimiento arrastra al fermión (composición relativista de velocidades).
#[test]
fn t69_moving_kink() {
    let rest = relaxed_kinks();
    let x2 = 96.0;
    let v_rest = kink_branch_velocity(&rest, x2);
    let mut u = dirac(128, 128);
    u.set_yukawa(2, 0.3, 1.0, 0.3);
    u.seed_higgs_kinks(32, 96, 0.2);
    u.inject_wall_packet(96, 64.0, 8.0, 0.0);
    let (mut ts, mut xs, mut fx, mut fy) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut xk = x2;
    for t in 0..=150 {
        if t % 5 == 0 {
            xk = u.yukawa_kink_position(xk as f32);
            // centroide del fermión a menos de 10 celdas del kink
            let (mut sx, mut sy_c, mut sy_s, mut tot) = (0.0, 0.0, 0.0, 0.0);
            for y in 0..128 {
                for dx in -10i32..=10 {
                    let xx = (xk.round() as i32 + dx).rem_euclid(128);
                    let e = u.density_at(xx, y, 0) as f64;
                    let th = 2.0 * PI * y as f64 / 128.0;
                    sx += e * (xk.round() + dx as f64);
                    sy_c += e * th.cos();
                    sy_s += e * th.sin();
                    tot += e;
                }
            }
            ts.push(t as f64);
            xs.push(xk);
            fx.push(sx / tot);
            fy.push((sy_s.atan2(sy_c) / (2.0 * PI) * 128.0).rem_euclid(128.0));
        }
        u.step();
    }
    let unwrap = |v: &[f64]| -> Vec<f64> {
        let mut out = vec![v[0]];
        for k in 1..v.len() {
            let last = *out.last().unwrap();
            out.push(last + min_image(v[k] - v[k - 1], 128.0));
        }
        out
    };
    let uk = crate::util::fit_linear(&ts, &unwrap(&xs)).1;
    let ux = crate::util::fit_linear(&ts, &unwrap(&fx)).1;
    let uy = crate::util::fit_linear(&ts, &unwrap(&fy)).1;
    let gam = 1.0 / (1.0f64 - 0.04).sqrt();
    println!("t69: kink {uk:.4}; fermión ({ux:.4}, {uy:.4}) frente a (0,2, {:.4}) con v' = {v_rest:.4}", v_rest / gam);
    assert!((uk / 0.2 - 1.0).abs() < 0.02, "kink {uk}");
    assert!((ux / 0.2 - 1.0).abs() < 0.1, "fermión en x {ux}");
    assert!((uy / (v_rest / gam) - 1.0).abs() < 0.02, "fermión en y {uy} frente a {}", v_rest / gam);
}

/// Test 70: un campo eléctrico a lo largo de la pared saca al paquete hacia el volumen pasado k_y ≈ π/2.
#[test]
fn t70_field_along_wall() {
    let mut u = dirac(128, 128);
    u.set_domain_walls(0.3);
    u.set_field_substeps(2);
    u.set_em(1, 1.0, 1.0);
    u.inject_wall_packet(64, 64.0, 8.0, 0.0);
    u.set_uniform_electric_field(0.0, 4e-3, 0.0);
    let mut early_min = 1.0f64;
    for t in 1..=750 {
        u.step();
        if t % 25 == 0 && 4e-3 * (t as f64) < 1.5 {
            early_min = early_min.min(u.wall_fraction(64, 8));
        }
    }
    let late = u.wall_fraction(64, 8);
    println!("t70: fracción mínima con qEt < 1,5: {early_min:.3}; con qEt = 3: {late:.3}");
    assert!(early_min > 0.9, "antes de π/2: {early_min}");
    assert!(late < 0.3, "con qEt ≈ 3: {late}");
}

/// Picos locales del espectro de autocorrelación por encima de una fracción del máximo.

/// Test 71: borde Hall (m = ±0,3, 32 cuantos de flujo).
#[test]
fn t71_hall_edge() {
    let setup = || -> U {
        let mut u = dirac(128, 128);
        u.set_domain_walls(0.3);
        u.set_field_substeps(2);
        u.set_em(1, 1.0, 1.0);
        u.set_uniform_magnetic_field(32);
        u
    };
    // el paquete de la pared avanza; los del volumen solo giran
    let mut u = setup();
    u.inject_wall_packet(64, 64.0, 8.0, 0.0);
    let (v, frac, _, _) = run_wall_packet(&mut u, 64, 400);
    println!("t71: pared: v = {v:.4}, en la pared {frac:.3}");
    assert!((v.abs() / (0.3f64).cos() - 1.0).abs() < 0.03 && frac > 0.9, "pared: {v}, {frac}");
    for &xb in &[32.0f32, 96.0] {
        let mut u = setup();
        u.inject_dirac_packet(xb, 64.0, 6.0, 0.0, 0.0, 1);
        let (x0, y0) = (u.center_of_mass(0), u.center_of_mass(1));
        u.step_n(400);
        let dr = min_image(u.center_of_mass(0) - x0, 128.0).hypot(min_image(u.center_of_mass(1) - y0, 128.0)) / 400.0;
        println!("t71: volumen en x = {xb}: deriva {dr:.5} celdas/paso");
        assert!(dr < 0.01, "deriva {dr}");
    }
    // nivel n = 0 a cada lado, por autocorrelación de un estado aleatorio en ese lado
    let mut lvls = Vec::new();
    for &xb in &[32.0f32, 96.0] {
        let mut u = setup();
        u.start_autocorrelation(71);
        u.mask_outside(xb, 64.0, 0.0, 24.0);
        u.start_autocorrelation_from_current();
        u.step_n(4095);
        let p = spectrum_peaks(&mut u, 0.05);
        let near = p.iter().cloned().filter(|w| (w.abs() - 0.3).abs() < 0.05).min_by(|a, b| (a.abs() - 0.3).abs().partial_cmp(&(b.abs() - 0.3).abs()).unwrap()).unwrap_or(f64::NAN);
        println!("t71: lado x = {xb}: nivel cero {near:.4}");
        lvls.push(near);
    }
    assert!(lvls[0] * lvls[1] < 0.0, "signos opuestos {lvls:?}");
    for l in &lvls {
        assert!((l.abs() / 0.3 - 1.0).abs() < 0.01, "nivel cero {l}");
    }
}

/// Monopolo de 't Hooft–Polyakov relajado en L³ (g, v = 1, κ = c² = 0,25, s = 2, η = 0,05), centrado
/// entre celdas.
fn thooft(l: usize, g: f32, lambda: f32, steps: u32) -> (U, f32) {
    let mut u = gauge_universe(l, l, l, 0.0, 2);
    u.set_adjoint_higgs(g, 1.0, lambda);
    let c = l as f32 / 2.0 - 0.5;
    u.seed_thooft_polyakov(c, c, c);
    u.set_relaxation(0.05);
    u.step_n(steps);
    (u, c)
}

/// Tests 78 y 79: carga magnética cuantizada y masa fuera del límite BPS (32³, λ = 0,05).
#[test]
fn t78_t79_thooft_charge_mass() {
    let (mut u, c) = thooft(32, 0.5, 0.05, 1500);
    let flux = u.monopole_flux(c, c, c, 12);
    let fl8 = u.monopole_flux(c, c, c, 8);
    let unit = 4.0 * PI * 0.25 * 1.0 / 0.5;
    let mass = |u: &mut U, r: f32| (u.higgs_energy_in_sphere(c, c, c, r) + 2.0 * PI * 0.25 / (0.25 * r as f64)) / unit;
    let (m8, m12) = (mass(&mut u, 8.0), mass(&mut u, 12.0));
    println!("t78: flujo {flux:.5} (R = 12), {fl8:.5} (R = 8); t79: masa {m8:.4} (R = 8), {m12:.4} (R = 12) frente a f(1,265) = 1,272");
    assert!((flux.abs() - 1.0).abs() < 1e-3, "flujo {flux}");
    assert!((m12 / 1.272 - 1.0).abs() < 0.03, "masa {m12}");
    assert!((m12 / m8 - 1.0).abs() < 5e-3, "masa con R = 8 y 12: {m8}, {m12}");
}

/// Test 80: límite BPS con la solución exacta de Prasad–Sommerfield como semilla y borde.
#[test]
fn t80_bps_limit() {
    let (mut u, c) = thooft(40, 0.25, 0.0, 400);
    let (g, v, kap) = (0.25f64, 1.0f64, 0.25f64);
    let unit = 4.0 * PI * kap * v / g;
    for &r in &[6.0f64, 8.0, 10.0, 16.0] {
        let xi = g * v * r;
        let h = 1.0 / xi.tanh() - 1.0 / xi;
        let k = xi / xi.sinh();
        let e = u.higgs_energy_in_sphere(c, c, c, r as f32);
        let m = e / (h * (1.0 - k * k)) / unit;
        println!("t80: R = {r}: masa {m:.4} × 4πκv/g");
        assert!((m - 1.0).abs() < 0.01, "R = {r}: {m}");
    }
    let ratio = u.higgs_bogomolny_ratio(c, c, c, 2.0, 8.0);
    println!("t80: |DΦ|²/|B|² en 2 < r < 8: {ratio:.4}");
    assert!((ratio - 1.0).abs() < 0.01, "Bogomolny: {ratio}");
}

/// Par monopolo-antimonopolo (48³, g = 0,5, v = 1, κ = 0,25, d = 20 en z), relajado 150 pasos con
/// los núcleos fijados y después libre. Devuelve el paso de la aniquilación (d < 3) y la serie
/// [paso, energía total, energía fuera de r = 16].
fn monopole_pair(lambda: f32, steps: u32) -> (Option<u32>, Vec<(u32, f64, f64)>) {
    let mut u = gauge_universe(48, 48, 48, 0.0, 2);
    u.set_adjoint_higgs(0.5, 1.0, lambda);
    let c = 23.5f32;
    u.seed_monopole_pair(c, c, c, 20.0);
    u.set_relaxation(0.1);
    u.step_n(150);
    u.set_relaxation(0.0);
    u.clear_higgs_pins();
    let mut p1 = [c as f64, c as f64, c as f64 + 10.0];
    let mut p2 = [c as f64, c as f64, c as f64 - 10.0];
    let mut annihilated = None;
    let mut series = Vec::new();
    for t in 1..=steps {
        u.step();
        if t % 2 == 0 && annihilated.is_none() {
            let a = u.higgs_min_abs_near(p1[0] as f32, p1[1] as f32, p1[2] as f32, 4.0);
            let b = u.higgs_min_abs_near(p2[0] as f32, p2[1] as f32, p2[2] as f32, 4.0);
            p1 = [a[0], a[1], a[2]];
            p2 = [b[0], b[1], b[2]];
            let d = min_image(p1[2] - p2[2], 48.0).abs();
            if d < 3.0 {
                annihilated = Some(t);
            }
        }
        if t % 10 == 0 {
            let tot = u.higgs_energy();
            let inside = u.higgs_energy_in_sphere(c, c, c, 16.0);
            series.push((t, tot, tot - inside));
        }
    }
    (annihilated, series)
}

/// Test 110: pares de monopolos que se aniquilan (antes con λ = 0, por la atracción del Higgs sin masa).
#[test]
fn t110_monopole_pairs() {
    let (a0, _) = monopole_pair(0.0, 140);
    let (a5, _) = monopole_pair(0.5, 160);
    println!("t110: aniquilación en el paso {a0:?} (λ = 0) y {a5:?} (λ = 0,5)");
    assert!(a0.map_or(false, |t| t < 90), "λ = 0: {a0:?}");
    assert!(a5.map_or(false, |t| t > 90 && t < 140), "λ = 0,5: {a5:?}");
}

/// Test 124: la radiación de la aniquilación (λ = 0, 220 pasos).
#[test]
fn t124_annihilation_radiation() {
    let (a0, series) = monopole_pair(0.0, 220);
    let ta = a0.expect("no se aniquilan");
    let e0 = series[0].1;
    let drift = series.iter().map(|s| (s.1 / e0 - 1.0).abs()).fold(0.0, f64::max);
    let after = series.iter().find(|s| s.0 >= ta + 60).expect("serie corta");
    let frac = after.2 / after.1;
    println!("t124: aniquilación en {ta}; energía {e0:.3} (deriva {drift:.4}); fuera de r = 16 a los {} pasos: {frac:.3}", after.0);
    assert!(drift < 0.02, "energía: {drift}");
    assert!(frac > 0.7, "fracción fuera {frac}");
}

/// Condensado cargado de la caminata: escenario de vórtices (m = 0,5, g = −0,03, ρ̄ = 0,5) en 64²,
/// ruido de 10⁻³ y el U(1) dinámico.
fn charged_condensate(q: f32, kappa: f32) -> U {
    let mut u = U::new(64, 64, 1, 0.5, 0.1, 1.0);
    u.set_scenario(1);
    u.seed_uniform_background(0.5, 1e-3, 56);
    u.set_em(2, q, kappa);
    u
}

/// Serie de ln(E rms) durante `steps` pasos.
fn log_erms(u: &mut U, steps: usize) -> Vec<f64> {
    (0..steps)
        .map(|_| {
            u.step();
            u.em_e_rms().max(1e-300).ln()
        })
        .collect()
}

/// Tasa de crecimiento: pendiente de ln(E rms) entre ln E₀ + 2 y ln E_max − 2, y el paso final de
/// esa ventana.
fn growth_window(series: &[f64]) -> (f64, usize, usize) {
    let lo = series[..20].iter().cloned().fold(f64::INFINITY, f64::min) + 2.0;
    let hi = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - 2.0;
    let start = series.iter().position(|&v| v > lo).unwrap_or(0);
    let end = series.iter().position(|&v| v > hi).unwrap_or(series.len() - 1);
    if end <= start + 5 {
        return (0.0, start, end);
    }
    let ts: Vec<f64> = (start..end).map(|t| t as f64).collect();
    let (_, slope) = crate::util::fit_linear(&ts, &series[start..end]);
    (slope, start, end)
}

/// ω_p = q·√(ρ̄/m*) con ρ̄ = 0,5 y m* = 2·tan(0,5).
fn plasma_frequency(q: f64) -> f64 {
    q * (0.5 / (2.0 * (0.5f64).tan())).sqrt()
}

/// Test 56: el condensado cargado de la caminata es inestable, con γ ≈ ½·ω_p; sin carga no crece.
#[test]
fn t56_charged_condensate_instability() {
    let mut u = charged_condensate(0.05, 0.25);
    let s = log_erms(&mut u, 2000);
    let (gamma, a, b) = growth_window(&s);
    let wp = plasma_frequency(0.05);
    println!("t56: γ = {gamma:.5} (pasos {a}–{b}), ω_p = {wp:.5}, γ/ω_p = {:.3}", gamma / wp);
    assert!(gamma / wp > 0.4 && gamma / wp < 0.6, "γ/ω_p = {}", gamma / wp);
    let mut u0 = charged_condensate(0.0, 0.25);
    let s0 = log_erms(&mut u0, 2000);
    let grew = s0.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - s0[0];
    println!("t56: sin carga, ln(E máx/E₀) = {grew:.3}");
    assert!(grew < 1.0, "sin carga crece: {grew}");
}

/// Test 103: con fotones lentos (κ = 0,05) sigue creciendo, y el modo que crece cumple la resonancia
/// ε₀ − ε_b(±k) = ω_fotón(k) con las bandas de Grover.
#[test]
fn t103_grover_no_fix() {
    let mut u = charged_condensate(0.05, 0.05);
    let s = log_erms(&mut u, 2500);
    let (gamma, _, end) = growth_window(&s);
    let wp = plasma_frequency(0.05);
    // segunda corrida idéntica: 128 instantáneas que acaban al final de la ventana exponencial
    let mut v = charged_condensate(0.05, 0.05);
    let t0 = end.saturating_sub(128);
    v.step_n(t0 as u32);
    for _ in t0..end {
        v.step();
        v.capture_field_snapshot();
    }
    let m = v.analyze_growing_mode(0.5);
    println!("t103: γ/ω_p = {:.3}; modo k = ({:.3}, {:.3}), ω = {:.4}, fotón {:.4}, longitudinal {:.3}, desajuste {:.4}", gamma / wp, m[0], m[1], m[2], m[4], m[3], m[5]);
    assert!(gamma / wp > 0.3, "γ/ω_p = {}", gamma / wp);
    assert!(m[5] < 0.02, "resonancia: desajuste {}", m[5]);
}

/// |ψ₆| local medio: para cada vórtice, |Σ e^{6iθ}|/6 sobre sus 6 vecinos más cercanos (toro).
fn psi6(vs: &[(f64, f64, i32)], w: f64, h: f64) -> f64 {
    let mut acc = 0.0;
    for (k, a) in vs.iter().enumerate() {
        let mut nb: Vec<(f64, f64, f64)> = vs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != k)
            .map(|(_, b)| {
                let dx = min_image(b.0 - a.0, w);
                let dy = min_image(b.1 - a.1, h);
                (dx.hypot(dy), dx, dy)
            })
            .collect();
        nb.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        let (mut re, mut im) = (0.0, 0.0);
        for n in nb.iter().take(6) {
            let th = 6.0 * n.2.atan2(n.1);
            re += th.cos();
            im += th.sin();
        }
        acc += re.hypot(im) / 6.0;
    }
    acc / vs.len().max(1) as f64
}

/// Test 108: red de Abrikosov (96², q = 0,15, 24 cuantos, tipo II con m_H/m_γ = 2).
#[test]
fn t108_abrikosov_lattice() {
    let mut u = gauge_universe(96, 96, 1, 0.15, 1);
    u.set_compact_magnetic_energy(true);
    // m_γ² = 2c²q²v² = 0,01125; m_H = 2·m_γ ⇒ 4λ = 0,045
    u.set_higgs(1, 1.0, 0.01125);
    u.set_uniform_flux(24);
    // ruido 0,3 y 2000 pasos sin amortiguar antes de relajar: con poco ruido y amortiguamiento desde
    // el principio, a veces la red queda en un estado metaestable con defectos (|ψ₆| ≈ 0,83)
    u.add_higgs_noise(0.3, 108);
    u.step_n(2000);
    u.set_relaxation(0.02);
    u.step_n(18000);
    let vs = vortices(&mut u);
    let p6 = psi6(&vs, 96.0, 96.0);
    println!("t108: {} vórtices (devanados {:?}), |ψ₆| medio {p6:.3}", vs.len(), vs.iter().map(|v| v.2).collect::<Vec<_>>());
    assert!(vs.len() == 24 && vs.iter().all(|v| v.2 == 1), "vórtices: {vs:?}");
    assert!(p6 > 0.85, "|ψ₆| = {p6}");
}

/// Perfil radial |φ|(r) de un vórtice aislado relajado (q = 0,3, 96², su antivórtice a 48 celdas).
fn isolated_profile(lambda: f32) -> Vec<(f64, f64)> {
    let mut u = gauge_universe(96, 96, 1, 0.3, 1);
    u.set_higgs(1, 1.0, lambda);
    u.imprint_higgs_vortex(24.5, 48.5, 1, 1.5);
    u.imprint_higgs_vortex(72.5, 48.5, -1, 1.5);
    u.set_relaxation(0.05);
    u.step_n(3000);
    u.higgs_radial_profile(24.5, 48.5, 20.0).chunks(2).map(|c| (c[0], c[1])).collect()
}

fn profile_at(prof: &[(f64, f64)], r: f64) -> f64 {
    if r <= prof[0].0 {
        return prof[0].1;
    }
    for w in prof.windows(2) {
        if r <= w[1].0 {
            let t = (r - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 * (1.0 - t) + w[1].1 * t;
        }
    }
    1.0
}

/// Imprime vórtices en centros de plaqueta y fija los 4 sitios de cada núcleo con |φ| igual al
/// producto de los perfiles aislados de todos los vórtices (la fase, la de la semilla).
fn pinned_vortices(u: &mut U, cores: &[(f64, f64, i32)], prof: &[(f64, f64)]) {
    let (w, h) = (u.width() as f64, u.height() as f64);
    for &(x, y, wd) in cores {
        u.imprint_higgs_vortex(x as f32, y as f32, wd, 1.5);
    }
    for &(x, y, _) in cores {
        for (dx, dy) in [(-0.5f64, -0.5f64), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
            let (cx, cy) = (x + dx, y + dy);
            let a: f64 = cores.iter().map(|&(vx, vy, _)| profile_at(prof, min_image(cx - vx, w).hypot(min_image(cy - vy, h)))).product();
            u.set_higgs_abs(cx as i32, cy as i32, 0, a as f32);
        }
        u.pin_higgs_cells(x as f32, y as f32, 0.0, 0.75);
    }
}

/// Energía (en unidades de 2πc²v²) de dos vórtices +1 a distancia d (y dos −1 lejos) relajados
/// 3000 pasos (η = 0,05) con los núcleos fijados; con `release`, además 200 pasos libres, y las
/// posiciones finales de los +1.
fn pinned_pair_energy(lambda: f32, d: f64, prof: &[(f64, f64)], release: bool) -> (f64, Vec<f64>) {
    let mut u = gauge_universe(96, 96, 1, 0.3, 1);
    u.set_higgs(1, 1.0, lambda);
    let cores = [(48.5 - d / 2.0, 24.5, 1), (48.5 + d / 2.0, 24.5, 1), (24.5, 72.5, -1), (72.5, 72.5, -1)];
    pinned_vortices(&mut u, &cores, prof);
    u.set_relaxation(0.05);
    u.step_n(3000);
    if release {
        u.clear_higgs_pins();
        u.step_n(200);
    }
    let xs: Vec<f64> = vortices(&mut u).into_iter().filter(|v| v.2 > 0).map(|v| v.0).collect();
    (u.higgs_energy() / BPS_UNIT, xs)
}

/// Test 109: en el punto de Bogomolny no hay fuerza entre vórtices; en tipo II se repelen y en tipo I
/// se atraen. Los núcleos fijados cuestan un ~0,5 % de energía que depende algo de d: en el punto
/// crítico se sueltan (no se mueven, porque no hay fuerza) y se mide sin ese sesgo.
#[test]
fn t109_bogomolny_pairs() {
    // q = 0,3: m_γ² = 0,045; λ = (m_H/m_γ)²·m_γ²/4
    let lam = |ratio: f64| (ratio * ratio * 0.045 / 4.0) as f32;
    let pb = isolated_profile(lam(1.0));
    let (e4, x4) = pinned_pair_energy(lam(1.0), 4.0, &pb, true);
    let (e16, x16) = pinned_pair_energy(lam(1.0), 16.0, &pb, true);
    let (p4, _) = pinned_pair_energy(lam(1.0), 4.0, &pb, false);
    let (p16, _) = pinned_pair_energy(lam(1.0), 16.0, &pb, false);
    println!("t109: Bogomolny: E(4) − E(16) = {:.4} sueltos (posiciones {x4:?}, {x16:?}); {:.4} fijados", e4 - e16, p4 - p16);
    assert_eq!(x4, vec![46.5, 50.5], "Bogomolny: los vórtices se movieron");
    assert!((e4 - e16).abs() < 0.005, "Bogomolny: {}", e4 - e16);
    let p2 = isolated_profile(lam(2.0));
    let d2 = pinned_pair_energy(lam(2.0), 4.0, &p2, false).0 - pinned_pair_energy(lam(2.0), 16.0, &p2, false).0;
    let p1 = isolated_profile(lam(0.7));
    let d1 = pinned_pair_energy(lam(0.7), 4.0, &p1, false).0 - pinned_pair_energy(lam(0.7), 16.0, &p1, false).0;
    println!("t109: tipo II {d2:.4}, tipo I {d1:.4}");
    assert!(d2 > 0.1, "tipo II: {d2}");
    assert!(d1 < -0.03, "tipo I: {d1}");
}

/// Posiciones de los vórtices +1 (x, y), en orden.
fn plus_vortices(u: &mut U) -> Vec<(f64, f64)> {
    vortices(u).into_iter().filter(|v| v.2 > 0).map(|v| (v.0, v.1)).collect()
}

/// Ángulo (grados) entre las velocidades relativas de entrada y de salida de dos vórtices +1
/// lanzados uno contra otro (192², q = 0,3, d = 24, parámetro de impacto b, velocidades ±v).
fn scattering_angle(lambda: f32, b: f64, v: f32) -> f64 {
    let mut u = gauge_universe(192, 192, 1, 0.3, 1);
    u.set_higgs(1, 1.0, lambda);
    let (xa, ya, xb, yb) = (84.5, 96.5 - b / 2.0, 108.5, 96.5 + b / 2.0);
    let cores = [(xa, ya, 1), (xb, yb, 1), (84.5, 0.5, -1), (108.5, 0.5, -1)];
    let prof = isolated_profile(lambda);
    pinned_vortices(&mut u, &cores, &prof);
    u.set_relaxation(0.05);
    u.step_n(1500);
    u.set_relaxation(0.0);
    u.clear_higgs_pins();
    // regiones simétricas respecto de (96,5; 96,5), el punto medio de los dos vórtices
    u.boost_higgs_region(49, 57, 97, 137, v, 0.0);
    u.boost_higgs_region(97, 57, 145, 137, -v, 0.0);
    let steps = (48.0 / v as f64) as usize + 50;
    let mut rel = Vec::new();
    let mut last: Option<[(f64, f64); 2]> = None;
    for t in 0..=steps {
        if t % 2 == 0 {
            let p = plus_vortices(&mut u);
            if p.len() == 2 {
                // emparejar con las posiciones anteriores
                let pair = match last {
                    Some(l) => {
                        let d0 = min_image(p[0].0 - l[0].0, 192.0).hypot(min_image(p[0].1 - l[0].1, 192.0));
                        let d1 = min_image(p[1].0 - l[0].0, 192.0).hypot(min_image(p[1].1 - l[0].1, 192.0));
                        if d0 <= d1 { [p[0], p[1]] } else { [p[1], p[0]] }
                    }
                    None => {
                        if p[0].0 < p[1].0 { [p[0], p[1]] } else { [p[1], p[0]] }
                    }
                };
                rel.push((t as f64, min_image(pair[1].0 - pair[0].0, 192.0), min_image(pair[1].1 - pair[0].1, 192.0)));
                last = Some(pair);
            }
        }
        u.step();
    }
    // velocidad relativa de entrada (primer cuarto) y de salida (último cuarto)
    // ventanas de entrada y de salida ancladas al momento de máximo acercamiento
    let tc = rel.iter().min_by(|a, b| a.1.hypot(a.2).partial_cmp(&b.1.hypot(b.2)).unwrap()).map(|r| r.0).unwrap_or(0.0);
    let vel = |s: Vec<&(f64, f64, f64)>| -> (f64, f64) {
        let ts: Vec<f64> = s.iter().map(|r| r.0).collect();
        let xs: Vec<f64> = s.iter().map(|r| r.1).collect();
        let ys: Vec<f64> = s.iter().map(|r| r.2).collect();
        (crate::util::fit_linear(&ts, &xs).1, crate::util::fit_linear(&ts, &ys).1)
    };
    let vin = vel(rel.iter().filter(|r| r.0 < tc - 20.0).collect());
    let vout = vel(rel.iter().filter(|r| r.0 > tc + 40.0).collect());
    let c = (vin.0 * vout.0 + vin.1 * vout.1) / (vin.0.hypot(vin.1) * vout.0.hypot(vout.1));
    c.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Test 121: dispersión a 90° de dos vórtices BPS, independiente de la velocidad; tipo II rebota.
#[test]
fn t121_ninety_degrees() {
    let bps = 0.01125f32;
    let head = scattering_angle(bps, 0.0, 0.1);
    let b8 = scattering_angle(bps, 8.0, 0.1);
    let b8f = scattering_angle(bps, 8.0, 0.2);
    let t2 = scattering_angle(0.045, 0.0, 0.1);
    println!("t121: de frente {head:.1}°, b = 8: {b8:.1}° (v = 0,1) y {b8f:.1}° (v = 0,2); tipo II {t2:.1}°");
    assert!((head - 90.0).abs() < 3.0, "de frente: {head}");
    assert!(b8 > 15.0 && b8 < 26.0, "b = 8: {b8}");
    assert!((b8f - b8).abs() < 3.0, "v = 0,2: {b8f} frente a {b8}");
    assert!(t2 > 170.0, "tipo II: {t2}");
}

/// Test 77: las fuerzas del Higgs adjunto (sobre φ y sobre los enlaces) son −∂H/∂φ y −g²·∂H/∂θ
/// (diferencias finitas en 4³ con U y φ aleatorios), y su ley de Gauss se conserva.
#[test]
fn t77_adjoint_forces() {
    let (g, lam) = (0.7f32, 0.1f32);
    let build = || -> U {
        let mut u = gauge_universe(4, 4, 4, 0.0, 1);
        u.set_adjoint_higgs(g, 1.0, lam);
        let mut rng = crate::rng::SplitMix64::new(77);
        for v in u.higgs.phi.iter_mut() {
            *v = 0.8 * rng.gauss();
        }
        for l in 0..3 * u.n {
            let th: Vec<f64> = (0..3).map(|_| 0.7 * rng.gauss()).collect();
            let m = u.ym.alg.exp_minus_i(&th);
            u.ym.set_link(l, &m);
        }
        u
    };
    // fuerzas del integrador: con π = Ẽ = 0 y Δt = 1, un paso deja π = F_φ y Ẽ = F − g²·J
    let mut stepped = build();
    stepped.step();
    let base = build();
    let energy = |u: &mut U| u.higgs_energy();
    let eps = 1e-6;
    let mut worst = 0.0f64;
    for &i in &[0usize, 5, 21, 63] {
        for c in 0..3 {
            let mut a = build();
            a.higgs.phi[3 * i + c] += eps;
            let mut b = build();
            b.higgs.phi[3 * i + c] -= eps;
            let num = -(energy(&mut a) - energy(&mut b)) / (2.0 * eps);
            let f = stepped.higgs.pi[3 * i + c];
            worst = worst.max((num - f).abs() / f.abs().max(1e-3));
        }
    }
    let g2 = (g as f64).powi(2);
    for &l in &[0usize, 17, 70, 191] {
        for c in 0..3 {
            let mut th = [0.0f64; 3];
            th[c] = eps;
            let up = base.ym.link(l).mul(&base.ym.alg.exp_minus_i(&th));
            th[c] = -eps;
            let dn = base.ym.link(l).mul(&base.ym.alg.exp_minus_i(&th));
            let mut a = build();
            a.ym.set_link(l, &up);
            let mut b = build();
            b.ym.set_link(l, &dn);
            let num = -g2 * (energy(&mut a) - energy(&mut b)) / (2.0 * eps);
            let f = stepped.ym.e[l * 3 + c];
            worst = worst.max((num - f).abs() / f.abs().max(1e-3));
        }
    }
    println!("t77: error relativo máximo de las fuerzas {worst:e}");
    assert!(worst < 1e-5, "fuerzas: {worst:e}");
    // ley de Gauss con π y Ẽ aleatorios
    let mut u = build();
    let mut rng = crate::rng::SplitMix64::new(78);
    for v in u.higgs.pi.iter_mut() {
        *v = 0.2 * rng.gauss();
    }
    u.higgs_store_gauss_ref();
    u.step_n(100);
    let r = u.higgs_gauss_residuals()[0];
    println!("t77: deriva de Gauss del Higgs adjunto {r:e}");
    assert!(r < 1e-10, "Gauss adjunto: {r:e}");
}
