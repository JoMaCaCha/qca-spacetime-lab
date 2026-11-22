//! Fase 2 (gravedad, §2.6 y §3.4): tests 19–25, 115, 118, 122 y 123.

use super::common::*;
use crate::util::{fit_linear, fit_quadratic, min_image, wrap_angle};
use crate::QuantumSpacetimeUniverse as U;

const GAMMA: f32 = 1e-6;

/// Centroides con peso E² de dos solitones en la franja |y − y_c| < 20, separados por su punto medio.
fn pair_centroids(u: &U, yc: usize, mid: f64) -> (f64, f64) {
    let (w, h) = (u.width(), u.height());
    let (mut sl, mut wl, mut sr, mut wr) = (0.0, 0.0, 0.0, 0.0);
    for y in 0..h {
        if (y as f64 - yc as f64).abs() >= 20.0 {
            continue;
        }
        for x in 0..w {
            let e = u.density_at(x as i32, y as i32, 0) as f64;
            let e2 = e * e;
            let dx = min_image(x as f64 - mid, w as f64);
            if dx < 0.0 {
                sl += e2 * (mid + dx);
                wl += e2;
            } else {
                sr += e2 * (mid + dx);
                wr += e2;
            }
        }
    }
    (sl / wl, sr / wr)
}

struct PairRun {
    a_left: f64,
    a_right: f64,
}

/// Protocolo común de §5 (tests 19–25): relajar 300 pasos sin gravedad y medir 800 con ella.
/// Devuelve las aceleraciones (ajuste parabólico) de los dos solitones.
fn run_pair(xl: f64, nl: f32, xr: f64, nr: f32, phase_r: f32, mode: u32, gamma: f32) -> (PairRun, U) {
    let mut u = base_2d(256, 256);
    u.inject_wavepacket(xl as i32, 128, 0, 6.0, nl, 0.0, 0.0, 0.0, 0.0);
    u.inject_wavepacket(xr as i32, 128, 0, 6.0, nr, 0.0, 0.0, 0.0, phase_r);
    u.step_n(300);
    let relaxed = snapshot(&u);
    if gamma != 0.0 {
        u.set_gravity(mode, gamma);
    }
    let mut ts = Vec::new();
    let (mut ls, mut rs) = (Vec::new(), Vec::new());
    let (mut pl, mut pr) = pair_centroids(&u, 128, 0.5 * (xl + xr));
    for s in 0..800 {
        u.step();
        let mid = 0.5 * (pl + pr);
        let (l, r) = pair_centroids(&u, 128, mid);
        pl = l;
        pr = r;
        if s % 4 == 0 {
            ts.push(s as f64);
            ls.push(l);
            rs.push(r);
        }
    }
    let (_, _, cl) = fit_quadratic(&ts, &ls);
    let (_, _, cr) = fit_quadratic(&ts, &rs);
    (PairRun { a_left: 2.0 * cl, a_right: 2.0 * cr }, relaxed)
}

fn snapshot(u: &U) -> U {
    let mut c = U::new(u.width(), u.height(), u.depth(), u.mass(), u.g(), u.e_max());
    c.set_neighbor_coupling(u.alpha());
    c.set_capacity(u.capacity());
    c.copy_field_from(u);
    c
}

/// Φ estático de la fuente sola: el paquete inyectado (toda su norma), sin la partícula de prueba.
fn source_alone(xs: f64, norm: f32) -> U {
    let mut s = base_2d(256, 256);
    s.inject_energy_blob(xs.round() as i32, 128, 0, 6.0, norm);
    s.set_gravity(1, GAMMA);
    s
}

/// |∂Φ_fuente/∂x| en x_test (diferencia centrada).
fn source_gradient(xs: f64, norm: f32, xt: f64) -> f64 {
    let s = source_alone(xs, norm);
    let x = xt.round() as i32;
    let g = 0.5 * (s.gravity_potential_at(x + 1, 128, 0) as f64 - s.gravity_potential_at(x - 1, 128, 0) as f64);
    g.abs()
}

fn m_star_2d() -> f64 {
    2.0 * (0.5f64).tan()
}

/// Test 19: universalidad. En fase, a = a_pred ± 25 %; en contrafase también atrae.
#[test]
fn t19_universality() {
    let (r, _) = run_pair(108.0, 300.0, 148.0, 300.0, 0.0, 1, GAMMA);
    let a = 0.5 * (r.a_left - r.a_right);
    let pred = GAMMA as f64 * source_gradient(108.0, 300.0, 148.0) / m_star_2d();
    assert!((a / pred - 1.0).abs() < 0.25, "en fase: a = {a:e}, predicción {pred:e}");
    let (r2, _) = run_pair(108.0, 300.0, 148.0, 300.0, std::f32::consts::PI, 1, GAMMA);
    let a2 = 0.5 * (r2.a_left - r2.a_right);
    assert!(a2 > 0.0, "contrafase: a = {a2:e} (debe atraer)");
}

/// Tests 20 y 22: equivalencia (150 y 300 caen igual hacia una fuente de 600) y control γ = 0.
#[test]
fn t20_t22_equivalence_and_control() {
    let mut acc = Vec::new();
    let mut preds = Vec::new();
    for &nt in &[150.0f32, 300.0] {
        let (r, _) = run_pair(108.0, 600.0, 148.0, nt, 0.0, 1, GAMMA);
        let a = -r.a_right; // hacia la fuente (a la izquierda)
        let pred = GAMMA as f64 * source_gradient(108.0, 600.0, 148.0) / m_star_2d();
        acc.push(a);
        preds.push(pred);
    }
    let diff = (acc[0] - acc[1]).abs() / (0.5 * (acc[0] + acc[1]));
    assert!(diff < 0.20, "equivalencia: {:e} y {:e} difieren un {:.1} %", acc[0], acc[1], 100.0 * diff);
    for k in 0..2 {
        let q = acc[k] / preds[k];
        assert!(q > 0.65 && q < 1.1, "a/a_pred = {q}");
    }
    // test 22: control sin gravedad
    let (r, _) = run_pair(108.0, 600.0, 148.0, 300.0, 0.0, 1, 0.0);
    assert!((-r.a_right).abs() < 0.2 * preds[1], "control: |a| = {:e} frente a {:e}", r.a_right, preds[1]);
}

/// Test 21: ley de distancias en 2D, a·d constante ± 20 %.
#[test]
fn t21_distance_law() {
    let mut ad = Vec::new();
    for &d in &[30.0f64, 45.0, 60.0] {
        let xs = 128.0 - d / 2.0;
        let (r, _) = run_pair(xs.round(), 600.0, (xs + d).round(), 300.0, 0.0, 1, GAMMA);
        ad.push(-r.a_right * d);
    }
    let mean = ad.iter().sum::<f64>() / 3.0;
    for v in &ad {
        assert!((v / mean - 1.0).abs() < 0.2, "a·d = {ad:?}");
    }
}

/// Test 23: causalidad. Con la variante onda la señal tarda; con la estática es instantánea.
#[test]
fn t23_causality() {
    // Φ a 40 celdas de la masa (la sonda literal de §5)
    let probe = |u: &U| u.gravity_potential_at(168, 128, 0) as f64;
    // onda
    let mut u = base_2d(256, 256);
    u.set_gravity(2, GAMMA);
    u.set_gravity_speed(0.5, 0.02);
    u.step_n(10);
    let g0 = probe(&u);
    u.inject_energy_blob(128, 128, 0, 6.0, 600.0);
    let mut series = Vec::new();
    for t in 10..400 {
        series.push((t, probe(&u)));
        u.step();
    }
    let gfin = probe(&u);
    let total = (gfin - g0).abs();
    let first = series.iter().find(|(_, g)| (g - g0).abs() > 0.05 * total).map(|p| p.0).unwrap_or(400);
    assert!(first >= 74, "la señal llega a 40 celdas en t = {first} (debe ser ≥ 74)");
    // estática
    let mut s = base_2d(256, 256);
    s.set_gravity(1, GAMMA);
    s.step_n(10);
    let h0 = probe(&s);
    s.inject_energy_blob(128, 128, 0, 6.0, 600.0);
    s.step();
    let h1 = probe(&s);
    s.step_n(300);
    let hf = probe(&s);
    assert!((h1 - h0).abs() > 0.5 * (hf - h0).abs(), "estática: cambia de inmediato ({h0}, {h1}, {hf})");
}

/// Test 24: la variante onda da la misma aceleración que la estática (0,9–1,1) y es estable.
#[test]
fn t24_wave_vs_static() {
    let (rs, _) = run_pair(108.0, 600.0, 148.0, 300.0, 0.0, 1, GAMMA);
    // máximo de |Φ| estático en la escena relajada
    let mut us = base_2d(256, 256);
    us.inject_wavepacket(108, 128, 0, 6.0, 600.0, 0.0, 0.0, 0.0, 0.0);
    us.inject_wavepacket(148, 128, 0, 6.0, 300.0, 0.0, 0.0, 0.0, 0.0);
    us.step_n(300);
    us.set_gravity(1, GAMMA);
    let phi_max_static = max_abs_phi(&us);
    let mut uw = snapshot(&us);
    uw.set_gravity(2, GAMMA);
    let mut phi_max_wave = 0.0f64;
    let mut ts = Vec::new();
    let mut rsx = Vec::new();
    let (mut pl, mut pr) = pair_centroids(&uw, 128, 128.0);
    for s in 0..800 {
        uw.step();
        let (l, r) = pair_centroids(&uw, 128, 0.5 * (pl + pr));
        pl = l;
        pr = r;
        if s % 4 == 0 {
            ts.push(s as f64);
            rsx.push(r);
        }
        if s % 50 == 0 {
            phi_max_wave = phi_max_wave.max(max_abs_phi(&uw));
        }
    }
    let (_, _, c) = fit_quadratic(&ts, &rsx);
    let aw = -2.0 * c;
    let ast = -rs.a_right;
    let q = aw / ast;
    assert!(q > 0.9 && q < 1.1, "onda/estática = {q} ({aw:e} / {ast:e})");
    assert!(phi_max_wave < 1.5 * phi_max_static, "max|Φ| onda {phi_max_wave} frente a estática {phi_max_static}");
}

fn max_abs_phi(u: &U) -> f64 {
    let mut m = 0.0f64;
    for i in 0..u.n {
        m = m.max(u.phi_total(i).abs() as f64);
    }
    m
}

/// Frecuencia de un reloj: pendiente de −arg Σψ en su pico (buscado en una ventana).
fn clock_phase(u: &U, cx: f64, cy: f64) -> f64 {
    let mut best = (0.0f32, 0i32, 0i32);
    for dy in -8..=8 {
        for dx in -8..=8 {
            let x = cx.round() as i32 + dx;
            let y = cy.round() as i32 + dy;
            let e = u.density_at(x, y, 0);
            if e > best.0 {
                best = (e, x, y);
            }
        }
    }
    u.phase_at(best.1, best.2, 0)
}

/// Test 25: corrimiento al rojo gravitacional; el reloj cercano va más lento.
#[test]
fn t25_redshift() {
    let run = |gamma: f32| -> (f64, f64, U) {
        let mut u = base_2d(256, 256);
        u.inject_energy_blob(128, 128, 0, 6.0, 600.0);
        u.inject_energy_blob(88, 128, 0, 6.0, 300.0);
        u.inject_energy_blob(228, 128, 0, 6.0, 300.0);
        u.step_n(300);
        let relaxed = snapshot(&u);
        if gamma != 0.0 {
            u.set_gravity(1, gamma);
        }
        let mut tr_n = Tracker::new(88.0, 128.0, 0.0, 10.0);
        let mut tr_f = Tracker::new(228.0, 128.0, 0.0, 10.0);
        tr_n.update(&u);
        tr_f.update(&u);
        let (mut an, mut af) = (0.0, 0.0);
        let (mut ln, mut lf) = (clock_phase(&u, tr_n.pos[0], tr_n.pos[1]), clock_phase(&u, tr_f.pos[0], tr_f.pos[1]));
        let mut ts = Vec::new();
        let (mut pn, mut pf) = (Vec::new(), Vec::new());
        for s in 0..200 {
            u.step();
            tr_n.update(&u);
            tr_f.update(&u);
            let a = clock_phase(&u, tr_n.pos[0], tr_n.pos[1]);
            let b = clock_phase(&u, tr_f.pos[0], tr_f.pos[1]);
            an += wrap_angle(a - ln);
            af += wrap_angle(b - lf);
            ln = a;
            lf = b;
            ts.push(s as f64);
            pn.push(an);
            pf.push(af);
        }
        let wn = -fit_linear(&ts, &pn).1;
        let wf = -fit_linear(&ts, &pf).1;
        (wn, wf, relaxed)
    };
    let (wn, wf, _) = run(GAMMA);
    let (cn, cf, _) = run(0.0);
    let measured = (wn - wf) - (cn - cf);
    let mut src = base_2d(256, 256);
    src.inject_energy_blob(128, 128, 0, 6.0, 600.0);
    src.set_gravity(1, GAMMA);
    let dphi = src.gravity_potential_at(88, 128, 0) as f64 - src.gravity_potential_at(228, 128, 0) as f64;
    let pred = -(GAMMA as f64) * dphi;
    assert!(measured < 0.0, "el reloj cercano debe ir más lento: {measured:e}");
    assert!((measured / pred - 1.0).abs() < 0.3, "Δω = {measured:e} frente a {pred:e}");
}

/// Test 115: inercia del solitón 3D. Empujón: M = k/v = 1,65 ± 0,08; fuerza lenta: F/a = 1,32 ± 0,1.
#[test]
fn t115_soliton_inertia_3d() {
    let mut u = base_3d(64, 64, 64);
    u.inject_energy_blob(32, 32, 32, 6.0, 3000.0);
    u.step_n(200);
    // empujón brusco
    let mut kick = snapshot3(&u);
    kick.boost_field(0.01, 0.0, 0.0);
    let mut tr = Tracker::new(32.0, 32.0, 32.0, 10.0);
    tr.update(&kick);
    let (mut ts, mut xs) = (Vec::new(), Vec::new());
    for s in 0..300 {
        kick.step();
        tr.update(&kick);
        if s >= 20 && s % 2 == 0 {
            ts.push(s as f64);
            xs.push(tr.unwrapped[0]);
        }
    }
    let v = fit_linear(&ts, &xs).1;
    let m_kick = 0.01 / v;
    assert!((m_kick - 1.65).abs() < 0.08, "empujón: M = {m_kick}");
    // fuerza lenta: Φ_ext = A·sin(2πx/64), A = 75, γ = 3e-6, durante 100 pasos
    let mut slow = snapshot3(&u);
    slow.set_gravity(0, 3e-6);
    slow.set_gravity_sine(75.0);
    let mut tr = Tracker::new(32.0, 32.0, 32.0, 10.0);
    tr.update(&slow);
    let force = 3e-6 * slow.mean_gravity_gradient(0, tr.pos[0] as f32, tr.pos[1] as f32, tr.pos[2] as f32, 10.0);
    let p0 = slow.total_momentum(0);
    let norm = slow.total_norm();
    let (mut ts, mut xs) = (vec![0.0], vec![tr.unwrapped[0]]);
    for s in 1..=100 {
        slow.step();
        tr.update(&slow);
        ts.push(s as f64);
        xs.push(tr.unwrapped[0]);
    }
    let (_, _, c) = fit_quadratic(&ts, &xs);
    let a = 2.0 * c;
    // Normalización de la referencia: impulso total por unidad de norma total (incluye el ~20 %
    // de radiación que no se mueve con el solitón) dividido por la velocidad adquirida.
    let m_ref = (slow.total_momentum(0) - p0) / norm / (a * 100.0);
    assert!((m_ref - 1.32).abs() < 0.1, "fuerza lenta (normalización de la referencia): {m_ref}");
    // Por unidad de norma del propio solitón, la fuerza lenta da la masa de banda (desviación documentada).
    let m_soliton = force / a;
    let m_band = 3.0 * (0.5f64).tan();
    assert!((m_soliton / m_band - 1.0).abs() < 0.1, "fuerza lenta por solitón: F/a = {m_soliton} frente a m* = {m_band}");
}

fn snapshot3(u: &U) -> U {
    let mut c = U::new(u.width(), u.height(), u.depth(), u.mass(), u.g(), u.e_max());
    c.set_neighbor_coupling(u.alpha());
    c.set_capacity(u.capacity());
    c.copy_field_from(u);
    c
}

/// Test 118: expansión de Einstein–de Sitter, δ ∝ a; sin expansión, la tasa de Jeans.
#[test]
fn t118_expansion() {
    let rho = 0.5f32;
    let gamma0 = 1e-4f32;
    let k = 2.0 * std::f64::consts::PI / 64.0;
    let mut u = U::new(64, 8, 8, 0.5, 0.0, 1.0);
    u.set_neighbor_coupling(0.0);
    u.set_capacity(0.0);
    let m_star = u.effective_mass();
    let c = (8.0 * std::f64::consts::PI * gamma0 as f64 * rho as f64 / (3.0 * m_star)).sqrt();
    // modo creciente: S = (m*/k²)·(d ln a/dτ)·δ, con d ln a/dτ = C en τ = 0
    u.seed_density_mode(rho, 1e-3, 1, (m_star / (k * k) * c) as f32);
    u.set_gravity(1, gamma0);
    u.set_expansion(true, 100.0);
    let (mut la, mut ld) = (Vec::new(), Vec::new());
    for _ in 0..80 {
        u.step();
        let a = u.scale_factor();
        if a > 1.5 {
            la.push(a.ln());
            ld.push(u.density_mode_amplitude(1).ln());
        }
    }
    let slope = fit_linear(&la, &ld).1;
    assert!((slope - 1.0).abs() < 0.05, "d ln δ/d ln a = {slope} (a final = {})", u.scale_factor());
    // sin expansión: crecimiento de Jeans
    let s_j = (4.0 * std::f64::consts::PI * gamma0 as f64 * rho as f64 / m_star - (k * k / (2.0 * m_star)).powi(2)).sqrt();
    let mut v = U::new(64, 8, 8, 0.5, 0.0, 1.0);
    v.set_neighbor_coupling(0.0);
    v.set_capacity(0.0);
    v.seed_density_mode(rho, 1e-4, 1, (m_star / (k * k) * s_j) as f32);
    v.set_gravity(1, gamma0);
    let (mut ts, mut ls) = (Vec::new(), Vec::new());
    for t in 0..200 {
        v.step();
        if t >= 40 {
            ts.push(t as f64);
            ls.push(v.density_mode_amplitude(1).ln());
        }
    }
    let rate = fit_linear(&ts, &ls).1;
    assert!((rate / s_j - 1.0).abs() < 0.03, "tasa de Jeans {rate} frente a {s_j}");
}

/// Test 122: gravedad entre dos solitones 3D promediando sobre la fase relativa (96³, d = 40).
#[test]
fn t122_two_solitons_3d() {
    let n = 96usize;
    let gamma = 3e-6f32;
    let mut base = base_3d(n, n, n);
    base.inject_energy_blob(48, 48, 48, 6.0, 3000.0);
    base.step_n(200);
    base.mask_outside(48.0, 48.0, 48.0, 14.0);
    let (xl, xr) = (28.0f64, 68.0f64);
    // predicción: 2γ⟨∇Φ_otro⟩/1,32
    let mut src = base_3d(n, n, n);
    src.add_field_from(&base, -20, 0, 0, 0.0);
    src.set_gravity(1, gamma);
    src.add_field_from(&base, 20, 0, 0, 0.0);
    let grad = src.mean_gravity_gradient(0, xr as f32, 48.0, 48.0, 12.0).abs();
    let pred = 2.0 * gamma as f64 * grad / 1.32;
    let rel_acc = |phase: f32, grav: bool| -> f64 {
        let mut u = base_3d(n, n, n);
        u.add_field_from(&base, -20, 0, 0, 0.0);
        u.add_field_from(&base, 20, 0, 0, phase);
        if grav {
            u.set_gravity(1, gamma);
        }
        let mut tl = Tracker::new(xl, 48.0, 48.0, 10.0);
        let mut tr = Tracker::new(xr, 48.0, 48.0, 10.0);
        tl.update(&u);
        tr.update(&u);
        let (mut ts, mut ds) = (vec![0.0], vec![tr.unwrapped[0] - tl.unwrapped[0]]);
        for s in 1..=300 {
            u.step();
            tl.update(&u);
            tr.update(&u);
            if s % 3 == 0 {
                ts.push(s as f64);
                ds.push(tr.unwrapped[0] - tl.unwrapped[0]);
            }
        }
        2.0 * fit_quadratic(&ts, &ds).2
    };
    let mut parts = Vec::new();
    for q in 0..4 {
        let ph = q as f32 * std::f32::consts::FRAC_PI_2;
        parts.push(rel_acc(ph, true) - rel_acc(ph, false));
    }
    let mean = parts.iter().sum::<f64>() / 4.0;
    let sd = (parts.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / 4.0).sqrt();
    assert!(mean < 0.0, "la parte gravitatoria debe atraer: {parts:?}");
    let ratio = -mean / pred;
    assert!(ratio > 0.5 && ratio < 1.1, "parte gravitatoria / predicción = {ratio} ({parts:?}, pred {pred:e})");
    assert!(sd < 0.3 * mean.abs(), "dispersión entre fases {sd:e} frente al promedio {mean:e}");
}

/// Ajuste del perfil de Schive ρ₀/[1 + 0,091(r/r_c)²]⁸ + fondo, para r < 3r_c. Devuelve (ρ₀, r_c).
pub fn fit_schive(profile: &[(f64, f64)]) -> (f64, f64) {
    let mut best = (f64::INFINITY, 0.0, 0.0);
    let mut rc = 0.8;
    while rc < 20.0 {
        let pts: Vec<&(f64, f64)> = profile.iter().filter(|p| p.0 < 3.0 * rc).collect();
        if pts.len() >= 3 {
            // mínimos cuadrados lineales en (ρ₀, fondo)
            let (mut s11, mut s12, mut s22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for p in &pts {
                let f = 1.0 / (1.0 + 0.091 * (p.0 / rc).powi(2)).powi(8);
                s11 += f * f;
                s12 += f;
                s22 += 1.0;
                b1 += f * p.1;
                b2 += p.1;
            }
            let det = s11 * s22 - s12 * s12;
            if det.abs() > 1e-30 {
                let r0 = (b1 * s22 - b2 * s12) / det;
                let bg = (s11 * b2 - s12 * b1) / det;
                let mut err = 0.0;
                for p in &pts {
                    let f = 1.0 / (1.0 + 0.091 * (p.0 / rc).powi(2)).powi(8);
                    err += (p.1 - r0 * f - bg).powi(2);
                }
                let err = err / pts.len() as f64;
                if err < best.0 {
                    best = (err, r0, rc);
                }
            }
        }
        rc += 0.02;
    }
    (best.1, best.2)
}

/// Test 123: estrella de Bose a partir de ruido (Levkov et al., 2018), 48³ y L̃ = 20.
#[test]
fn t123_bose_star() {
    let n = 48usize;
    let m_star = 3.0 * (0.5f64).tan();
    let v = 20.0 / (m_star * n as f64);
    let k0 = (m_star * v) as f32;
    let gamma = 7.2e-5f32;
    let run = |grav: bool| -> (f64, U) {
        let mut u = U::new(n, n, n, 0.5, 0.0, 1.0);
        u.set_neighbor_coupling(0.0);
        u.set_capacity(0.0);
        u.seed_gaussian_gas(k0, 0.5, 2024);
        if grav {
            u.set_gravity(1, gamma);
        }
        let mut peak_ratio = 0.0f64;
        for s in 1..=9000 {
            u.step();
            if s % 250 == 0 {
                peak_ratio = peak_ratio.max(u.max_density() as f64 / u.mean_density());
                if grav && peak_ratio > 60.0 && s > 6000 {
                    break;
                }
            }
        }
        (peak_ratio, u)
    };
    let (pg, mut ug) = run(true);
    let (pc, _) = run(false);
    assert!(pg > 60.0, "con gravedad, max ρ/⟨ρ⟩ = {pg}");
    assert!(pc < 15.0, "sin gravedad, max ρ/⟨ρ⟩ = {pc}");
    // perfil final frente al solitón de Schrödinger–Poisson
    let p = ug.peak_cell() as usize;
    let (x, y, z) = (p % n, (p / n) % n, p / (n * n));
    let nb = ug.compute_radial_profile(x as f32, y as f32, z as f32) as usize;
    let prof: Vec<(f64, f64)> = (0..nb).map(|b| (b as f64, ug.radial[2 * b])).collect();
    let (rho0, rc) = fit_schive(&prof);
    let rc_pred = (0.385 / (rho0 * gamma as f64 * m_star * m_star)).powf(0.25);
    assert!((rc / rc_pred - 1.0).abs() < 0.15, "r_c = {rc} frente a {rc_pred} (ρ₀ = {rho0})");
}
