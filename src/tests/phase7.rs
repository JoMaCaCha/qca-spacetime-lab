//! Fase 7 (Dirac 3D, Weyl, cuerda de Higgs, condensado de Dirac, Nordström y Schwarzschild,
//! escenario luz y Kaplan; §2.15–§2.20 y §3.12–§3.18): tests 74–76, 81–102, 111–113, 117, 119 y
//! 120 (el 77, del Higgs adjunto, está con la fase 6). El motor de espinores vive en f64.

use super::common::spectrum_peaks;
use crate::rng::SplitMix64;
use crate::spinor::{band_projector_s, m4_dag, m4_mul, M4};
use crate::util::{fit_linear, fit_quadratic, wrap_angle};
use crate::QuantumSpacetimeUniverse as U;
use std::f64::consts::PI;

/// Escenario Dirac 3D (4 componentes, orden alternado) con masa m.
pub(super) fn dirac3d(w: usize, h: usize, d: usize, m: f32) -> U {
    let mut u = U::new(w, h, d, 0.5, 0.0, 1.0);
    u.set_scenario(2);
    u.set_mass(m);
    u
}

fn m4_err(a: &M4, b: &M4) -> f64 {
    let mut e = 0.0f64;
    for i in 0..4 {
        for j in 0..4 {
            e = e.max((a[i][j] - b[i][j]).norm());
        }
    }
    e
}

/// Test 83: dispersión 3D (operador de dos pasos), degeneración de espín, velocidad máxima y el
/// desdoblamiento del orden fijo.
#[test]
fn t83_dirac3d_dispersion() {
    let mut u = dirac3d(4, 4, 4, 0.1);
    let (mut worst_deg, mut worst_om) = (0.0f64, 0.0f64);
    let s3 = 1.0 / 3f64.sqrt();
    for &kk in &[0.1, 0.2, 0.3] {
        for dir in [[1.0, 0.0, 0.0], [s3, s3, s3]] {
            let k = [kk * dir[0], kk * dir[1], kk * dir[2]];
            let lv = u.sw_band_levels(k);
            assert_eq!(lv.len(), 2, "niveles: {lv:?}");
            worst_deg = worst_deg.max((lv[0] - lv[1]).abs());
            let om = 0.5 * (lv[0] + lv[1]);
            let rel = (om / (0.01f64 + kk * kk).sqrt() - 1.0).abs();
            println!("t83: |k| = {kk}, dir {:?}: ω = {om:.4} (√(m² + k²) = {:.4})", dir, (0.01f64 + kk * kk).sqrt());
            worst_om = worst_om.max(rel);
        }
    }
    println!("t83: degeneración {worst_deg:e}, error de ω {worst_om:.4}");
    assert!(worst_deg < 1e-9 && worst_om < 0.03);
    let mut rng = SplitMix64::new(83);
    let h = 1e-6;
    let mut vmax = 0.0f64;
    for _ in 0..4000 {
        let k = [PI * (2.0 * rng.next_f64() - 1.0), PI * (2.0 * rng.next_f64() - 1.0), PI * (2.0 * rng.next_f64() - 1.0)];
        let mut g = [0.0f64; 3];
        for a in 0..3 {
            let mut kp = k;
            kp[a] += h;
            let mut km = k;
            km[a] -= h;
            g[a] = (u.sw_omega(kp) - u.sw_omega(km)) / (2.0 * h);
        }
        vmax = vmax.max((g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt());
    }
    println!("t83: velocidad de grupo máxima {vmax:.4}");
    assert!(vmax <= 1.0 + 1e-6, "v_max = {vmax}");
    u.set_dirac_order(true);
    let lv = u.sw_band_levels([0.1 * s3, 0.1 * s3, 0.1 * s3]);
    let split = (lv[1] - lv[0]) / (0.5 * (lv[0] + lv[1]));
    println!("t83: orden fijo, diagonal |k| = 0,1: {:.4} y {:.4} ({:.1} %)", lv[0], lv[1], 100.0 * split);
    assert!(split > 0.03);
}

/// Test 84: ley de Gauss e invariancia gauge del Dirac 3D con el U(1) dinámico.
#[test]
fn t84_dirac3d_gauss_gauge() {
    let build = |gauge: bool| -> U {
        let mut u = dirac3d(24, 24, 24, 0.1);
        u.inject_dirac_packet_3d(8.0, 12.0, 12.0, 3.0, 0.4, 0.0, 0.0, 1, 0);
        u.inject_dirac_packet_3d(16.0, 12.0, 13.0, 3.0, -0.3, 0.2, 0.0, 1, 0);
        u.set_field_substeps(2);
        u.set_em(2, 1.0, 1.0);
        if gauge {
            u.apply_random_gauge_transform(3.0, 84);
        }
        u
    };
    let mut a = build(false);
    let mut b = build(true);
    let n0 = a.spinor_norm();
    for _ in 0..300 {
        a.step();
        b.step();
    }
    let (ra, rb) = (a.gauss_residual(), b.gauss_residual());
    let mut dd = 0.0f64;
    for i in 0..a.n {
        dd = dd.max((a.sw_density(i) - b.sw_density(i)).abs());
    }
    let de = a.em.e.iter().zip(b.em.e.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    println!("t84: Gauss {ra:e} y {rb:e}; densidad {dd:e}; E {de:e}; norma {:e}", (a.spinor_norm() - n0).abs());
    assert!(ra < 1e-12 && rb < 1e-12 && dd < 1e-12 && de < 1e-12);
}

/// Test 87: el proyector de banda P₊ = ½(1 − S/s) del operador de dos pasos.
#[test]
fn t87_band_projector() {
    let mut u = dirac3d(24, 24, 24, 0.8);
    let mut rng = SplitMix64::new(87);
    let (mut e1, mut e2, mut e3) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..1000 {
        let k = [PI * (2.0 * rng.next_f64() - 1.0), PI * (2.0 * rng.next_f64() - 1.0), PI * (2.0 * rng.next_f64() - 1.0)];
        u.set_mass((1.2 * rng.next_f64()) as f32);
        let w = u.sw_period_matrix(0, k);
        let wd = m4_dag(&w);
        let mut s = [[crate::spinor::cx(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                s[i][j] = (w[i][j] - wd[i][j]) / crate::spinor::cx(0.0, 2.0);
            }
        }
        let s2 = m4_mul(&s, &s);
        let tr: f64 = (0..4).map(|i| s2[i][i].re).sum::<f64>() / 4.0;
        let mut id = [[crate::spinor::cx(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            id[i][i] = crate::spinor::cx(tr, 0.0);
        }
        e1 = e1.max(m4_err(&s2, &id));
        if tr < 1e-12 {
            continue;
        }
        let p = band_projector_s(&w, 4, 1).expect("S² ∝ 1");
        e2 = e2.max(m4_err(&m4_mul(&p, &p), &p));
        e3 = e3.max(m4_err(&m4_mul(&p, &w), &m4_mul(&w, &p)));
    }
    println!("t87: |S² − s²| = {e1:e}, |P² − P| = {e2:e}, |PW − WP| = {e3:e}");
    assert!(e1 < 1e-12 && e2 < 1e-12 && e3 < 1e-12);
    u.set_mass(0.8);
    u.inject_dirac_packet_3d(12.0, 12.0, 12.0, 3.0, 0.0, 0.0, 0.0, 1, 0);
    let neg = 1.0 - u.spinor_band_population();
    u.clear();
    u.inject_dirac_packet_3d_unprojected(12.0, 12.0, 12.0, 3.0, 0.0, 0.0, 0.0, 1, 0);
    let neg_raw = 1.0 - u.spinor_band_population();
    println!("t87: banda −1: proyectado {neg:e}, sin proyectar {neg_raw:.4} (ref. 0,064)");
    assert!(neg < 1e-10 && neg_raw > 0.03);
}

/// Test 88: respuesta a un campo uniforme (A_x −= E en cada paso).
#[test]
fn t88_uniform_field() {
    let mut u = dirac3d(48, 48, 48, 0.8);
    u.inject_dirac_packet_3d(24.0, 24.0, 24.0, 4.0, 0.0, 0.0, 0.0, 1, 0);
    let curv = u.spinor_mean_curvature(0);
    let e = 2e-3;
    u.set_em(1, 1.0, 0.3);
    u.set_uniform_electric_field(e as f32, 0.0, 0.0);
    let (mut ts, mut xs) = (Vec::new(), Vec::new());
    for t in 0..=40 {
        ts.push(t as f64);
        xs.push(u.part_centroid(u32::MAX, 0));
        if t < 40 {
            u.step();
        }
    }
    let (_, _, c2) = fit_quadratic(&ts, &xs);
    let a = 2.0 * c2;
    let ratio = a / (e * curv);
    println!("t88: ⟨∂²ω/∂k²⟩ = {curv:.4} (ref. 0,820), a = {a:e}, razón {ratio:.4} (ref. 0,987)");
    assert!((ratio - 1.0).abs() < 0.03, "razón {ratio}");
}

/// Test 85: niveles de Landau en 3D (k_z = 0): n = 0 en +m y en −m.
#[test]
fn t85_landau_3d() {
    let m = 0.1f64;
    let mut u = dirac3d(32, 32, 4, m as f32);
    u.set_em(1, 1.0, 0.3);
    u.set_uniform_magnetic_field(8);
    let qb = u.uniform_field_b(8);
    u.start_spinor_autocorrelation(85, 0.0, 0.0, 0.0, 0.0, true);
    u.step_n(4095);
    let peaks = spectrum_peaks(&mut u, 0.003);
    let near = |t: f64| peaks.iter().map(|p| (p - t).abs()).fold(f64::INFINITY, f64::min);
    println!("t85: qB = {qb:.4}; picos {:?}", peaks.iter().filter(|p| p.abs() < 0.7).map(|p| (p * 1e4).round() / 1e4).collect::<Vec<_>>());
    assert!(near(m) < 0.01 * m && near(-m) < 0.01 * m, "n = 0 en ±m");
    for nn in 1..=3 {
        let e = (m * m + 2.0 * nn as f64 * qb).sqrt();
        println!("t85: n = {nn}: predicción ±{e:.4}, distancia {:.4} y {:.4}", near(e), near(-e));
        assert!(near(e) < 0.05 * e && near(-e) < 0.05 * e, "n = {nn}");
    }
}

/// Test 86: g = 2, la helicidad se conserva: el espín gira con la velocidad. La velocidad es la
/// exacta (el flujo de norma de los subpasos), que no depende de las vueltas al toro.
#[test]
fn t86_helicity() {
    let mut u = dirac3d(96, 96, 8, 0.1);
    u.set_em(1, 1.0, 0.3);
    u.set_uniform_magnetic_field(24);
    u.set_packet_spin(1.0, 0.0, 0.0);
    u.set_flux_tracking(true);
    u.inject_dirac_packet_3d(48.0, 48.0, 4.0, 8.0, 0.3, 0.0, 0.0, 1, 0);
    let (mut ts, mut av, mut asp) = (Vec::new(), Vec::new(), Vec::new());
    let (mut uv, mut us) = (0.0f64, 0.0f64);
    let (mut lv, mut ls) = (None::<f64>, None::<f64>);
    let mut cmin = 1.0f64;
    for t in 0..140 {
        u.step();
        let v = [u.spinor_axis_flux(0), u.spinor_axis_flux(1)];
        let s = [u.spinor_spin(0), u.spinor_spin(1), u.spinor_spin(2)];
        let (pv, ps) = (v[1].atan2(v[0]), s[1].atan2(s[0]));
        if let (Some(a), Some(b)) = (lv, ls) {
            uv += wrap_angle(pv - a);
            us += wrap_angle(ps - b);
        }
        lv = Some(pv);
        ls = Some(ps);
        ts.push(t as f64);
        av.push(uv);
        asp.push(us);
        let vn = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let sn = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        cmin = cmin.min((v[0] * s[0] + v[1] * s[1]) / (vn * sn).max(1e-300));
    }
    let (_, wv) = fit_linear(&ts, &av);
    let (_, ws) = fit_linear(&ts, &asp);
    println!("t86: giro del espín {ws:.5} rad/paso, de la velocidad {wv:.5} (razón {:.4}, ref. 0,987); coseno mínimo {cmin:.3}", ws / wv);
    assert!((ws / wv - 1.0).abs() < 0.03 && cmin > 0.9);
}

/// Test 90: Gauss no abeliana e invariancia gauge de los quarks 3D (SU(2) y SU(3)).
#[test]
fn t90_quarks3d_gauss_gauge() {
    for nc in [2u32, 3] {
        let build = |gauge: bool| -> U {
            let mut u = dirac3d(16, 16, 16, 0.2);
            u.set_dirac_colour(nc);
            let ks = [[0.4f32, 0.0, 0.0], [-0.3, 0.3, 0.0], [0.0, -0.2, 0.4]];
            for c in 0..nc as usize {
                u.inject_dirac_packet_3d(5.0 + 3.0 * c as f32, 8.0, 8.0 + c as f32, 2.5, ks[c][0], ks[c][1], ks[c][2], 1, c as u32);
            }
            u.set_field_substeps(2);
            u.set_em(2, 0.7, 1.0);
            if gauge {
                u.apply_random_gauge_transform(3.0, 90);
            }
            u
        };
        let mut a = build(false);
        let mut b = build(true);
        let n0 = a.spinor_norm();
        for _ in 0..200 {
            a.step();
            b.step();
        }
        let (ra, rb) = (a.gauss_residual(), b.gauss_residual());
        let mut dd = 0.0f64;
        for i in 0..a.n {
            dd = dd.max((a.sw_density(i) - b.sw_density(i)).abs());
        }
        let mut dp = 0.0f64;
        for i in 0..a.n {
            for (p, q) in [(0usize, 1usize), (1, 2), (2, 0)] {
                dp = dp.max((a.ym_plaq_p(i, p, q).trace().re - b.ym_plaq_p(i, p, q).trace().re).abs());
            }
        }
        let dn = (a.spinor_norm() - n0).abs();
        println!("t90: SU({nc}): Gauss {ra:e} y {rb:e}; densidad {dd:e}; Tr P {dp:e}; norma {dn:e}");
        assert!(ra < 1e-12 && rb < 1e-12 && dd < 1e-12 && dp < 1e-12 && dn < 1e-10);
    }
}

/// Test 94: evolución por partes (la suma de las partes es la evolución completa).
#[test]
fn t94_parts() {
    for nc in [1u32, 3] {
        let mut u = dirac3d(12, 12, 12, 0.3);
        if nc > 1 {
            u.set_dirac_colour(nc);
        }
        u.set_packet_parts(true);
        u.inject_dirac_packet_3d(3.0, 6.0, 6.0, 2.0, 0.3, 0.0, 0.0, 1, 0);
        u.inject_dirac_packet_3d(9.0, 6.0, 6.0, 2.0, -0.3, 0.1, 0.0, 1, if nc > 1 { 1 } else { 0 });
        u.set_field_substeps(2);
        u.set_em(2, if nc > 1 { 0.7 } else { 1.0 }, 1.0);
        u.step_n(30);
        let mut worst = 0.0f64;
        for k in 0..u.sw.psi.len() {
            let s = u.sw.parts[0][k] + u.sw.parts[1][k];
            worst = worst.max((s - u.sw.psi[k]).abs());
        }
        println!("t94: N = {nc}: |Σ partes − completa| = {worst:e}");
        assert!(worst < 1e-12);
    }
}

/// ⟨E_x⟩ del campo de Poisson de `src` (media nula, en el toro) promediado sobre la densidad `rho`.
fn poisson_ex_over(u: &mut U, src: &[f64], rho: &[f64]) -> f64 {
    for v in u.em.e.iter_mut() {
        *v = 0.0;
    }
    u.em_add_gradient_field(src);
    let n = u.n;
    let tot: f64 = rho.iter().sum();
    let mut acc = 0.0;
    for i in 0..n {
        let im = u.nb(i, 0, -1);
        acc += rho[i] * 0.5 * (u.em.e[i] + u.em.e[im]);
    }
    acc / tot
}

/// Aceleración (ajuste cuadrático) de una serie de posiciones tomadas en cada paso.
fn accel(xs: &[f64]) -> f64 {
    let ts: Vec<f64> = (0..xs.len()).map(|t| t as f64).collect();
    2.0 * fit_quadratic(&ts, xs).2
}

/// Test 89: fuerza de Coulomb sobre un paquete proyectado en 3D (80³), con el control q = 0.
#[test]
fn t89_coulomb_3d() {
    let (xq, xp, yc) = (28.0f32, 52.0f32, 40.0f32);
    let mut pred = 0.0;
    let mut res = Vec::new();
    for q in [1.0f32, 0.0] {
        let mut u = dirac3d(80, 80, 80, 0.8);
        u.inject_dirac_packet_3d(xp, yc, yc, 4.0, 0.0, 0.0, 0.0, 1, 0);
        if q > 0.0 {
            let curv = u.spinor_mean_curvature(0);
            let rho: Vec<f64> = (0..u.n).map(|i| u.sw_density(i)).collect();
            let mut aux = dirac3d(80, 80, 80, 0.8);
            let mut src = vec![0.0f64; aux.n];
            let c = aux.wrap_cell(xq as i32, yc as i32, yc as i32);
            let (cx_, cy_, cz_) = aux.coords(c);
            let mut tot = 0.0;
            for i in 0..aux.n {
                let (x, y, z) = aux.coords(i);
                let dx = crate::util::min_image(x as f64 - cx_ as f64, 80.0);
                let dy = crate::util::min_image(y as f64 - cy_ as f64, 80.0);
                let dz = crate::util::min_image(z as f64 - cz_ as f64, 80.0);
                src[i] = (-(dx * dx + dy * dy + dz * dz) / 8.0).exp();
                tot += src[i];
            }
            for v in src.iter_mut() {
                *v *= 5.0 / tot;
            }
            let ex = poisson_ex_over(&mut aux, &src, &rho);
            pred = q as f64 * ex * curv;
            println!("t89: ⟨E_x⟩ = {ex:e}, ⟨∂²ω/∂k²⟩ = {curv:.4}");
        }
        u.set_field_substeps(2);
        u.set_em(2, q, 1.0);
        u.add_static_charge(xq as i32, yc as i32, yc as i32, 5.0, 2.0);
        let mut xs = Vec::new();
        for t in 0..=40 {
            xs.push(u.part_centroid(u32::MAX, 0));
            if t < 40 {
                u.step();
            }
        }
        res.push(accel(&xs));
    }
    println!("t89: a = {:e} frente a {pred:e} (razón {:.4}, ref. 0,972); control {:e}", res[0], res[0] / pred, res[1]);
    assert!((res[0] / pred - 1.0).abs() < 0.06 && res[1].abs() < 1e-6);
}

/// Test 95: dos paquetes dinámicos (normas 0,7 y 0,3, q = 2, d = 28) y la tercera ley, cada uno
/// evolucionado como parte propia.
#[test]
fn t95_third_law() {
    let (x1, x2, yc) = (26.0f32, 54.0f32, 40.0f32);
    let q = 2.0f64;
    let mut u = dirac3d(80, 80, 80, 0.8);
    u.set_packet_parts(true);
    u.set_packet_norm(0.7);
    u.inject_dirac_packet_3d(x1, yc, yc, 4.0, 0.0, 0.0, 0.0, 1, 0);
    u.set_packet_norm(0.3);
    u.inject_dirac_packet_3d(x2, yc, yc, 4.0, 0.0, 0.0, 0.0, 1, 0);
    let st = 2 * u.sw_namp();
    let dens = |b: &[f64]| -> Vec<f64> { (0..b.len() / st).map(|i| b[i * st..(i + 1) * st].iter().map(|v| v * v).sum()).collect() };
    let r1 = dens(&u.sw.parts[0]);
    let r2 = dens(&u.sw.parts[1]);
    let c1 = u.sw_mean_curvature(&u.sw.parts[0].clone(), 0);
    let c2 = u.sw_mean_curvature(&u.sw.parts[1].clone(), 0);
    let mut aux = dirac3d(80, 80, 80, 0.8);
    let unit = |r: &[f64]| -> Vec<f64> {
        let t: f64 = r.iter().sum();
        r.iter().map(|v| v / t).collect()
    };
    let g1 = poisson_ex_over(&mut aux, &unit(&r2), &r1);
    let g2 = poisson_ex_over(&mut aux, &unit(&r1), &r2);
    let p1 = q * q * 0.3 * g1 * c1;
    let p2 = q * q * 0.7 * g2 * c2;
    drop(aux);
    u.set_field_substeps(2);
    u.set_em(2, q as f32, 1.0);
    let (mut xa, mut xb) = (Vec::new(), Vec::new());
    for t in 0..=40 {
        xa.push(u.part_centroid(0, 0));
        xb.push(u.part_centroid(1, 0));
        if t < 40 {
            u.step();
        }
    }
    let (a1, a2) = (accel(&xa), accel(&xb));
    let third = (0.7 * a1 + 0.3 * a2).abs() / (0.7 * a1).abs();
    println!("t95: a1 = {a1:e} (pred. {p1:e}, razón {:.3}; ref. 0,930), a2 = {a2:e} (pred. {p2:e}, razón {:.3}; ref. 0,966); tercera ley {third:.3} (ref. 0,038)", a1 / p1, a2 / p2);
    assert!((a1 / p1 - 1.0).abs() < 0.1 && (a2 / p2 - 1.0).abs() < 0.1 && third < 0.1);
}

/// Test 98: factores de color entre dos quarks SU(3) en reposo (rojo-rojo frente a rojo-verde),
/// cada uno como parte propia. El desplazamiento de cada quark se integra con su flujo exacto: en
/// 48³ las colas rápidas de los paquetes, que se ensanchan, dan la vuelta al toro hacia el paso 30
/// y falsean los centroides con imagen mínima.
#[test]
fn t98_colour_factors_3d() {
    let mut rel = Vec::new();
    for c2 in [0u32, 1] {
        let mut u = dirac3d(48, 48, 48, 0.8);
        u.set_dirac_colour(3);
        u.set_packet_parts(true);
        u.set_packet_norm(0.5);
        u.set_flux_tracking(true);
        u.inject_dirac_packet_3d(16.0, 24.0, 24.0, 3.0, 0.0, 0.0, 0.0, 1, 0);
        u.inject_dirac_packet_3d(32.0, 24.0, 24.0, 3.0, 0.0, 0.0, 0.0, 1, c2);
        u.set_field_substeps(2);
        u.set_em(2, 2.0, 1.0);
        let (mut xa, mut xb) = (vec![0.0f64], vec![0.0f64]);
        for _ in 0..40 {
            u.step();
            xa.push(xa.last().unwrap() + u.part_axis_flux(0, 0) / 0.5);
            xb.push(xb.last().unwrap() + u.part_axis_flux(1, 0) / 0.5);
        }
        let r = accel(&xb) - accel(&xa);
        println!("t98: colores 0 y {c2}: aceleración relativa {r:e}; Gauss {:e}", u.gauss_residual());
        rel.push(r);
    }
    let ratio = rel[1] / rel[0];
    println!("t98: razón rojo-verde / rojo-rojo {ratio:.4} (ref. −0,501)");
    assert!(rel[0] > 0.0 && rel[1] < 0.0 && (ratio + 0.5).abs() < 0.03);
}

/// Universo de plasma de Dirac (§3.13): caja nx × ny × nz, m = 0,8, q = 1, s = 2 y un condensado
/// con ω_p dado (ρ̄ = ω_p²·m\*).
fn dirac_plasma(nx: usize, ny: usize, nz: usize, wp: f64, kappa: f32, modulation: f32, noise: f32) -> (U, f64) {
    let mut u = dirac3d(nx, ny, nz, 0.8);
    let mstar = 1.0 / u.dirac_band_curvature(0);
    let rho = wp * wp * mstar;
    u.seed_dirac_condensate(rho as f32, modulation, 1, noise, 7);
    u.set_field_substeps(2);
    u.set_em(2, 1.0, kappa);
    (u, rho)
}

/// Test 96: oscilación de Langmuir del condensado de Dirac (y el estado sin modular no se mueve).
#[test]
fn t96_langmuir() {
    let wp = 0.1;
    let (mut u, rho) = dirac_plasma(32, 4, 4, wp, 1.0, 0.05, 0.0);
    let k = 2.0 * PI / 32.0;
    let eps = u.sw_omega([k, 0.0, 0.0]) - u.sw_omega([0.0; 3]);
    let k2 = 2.0 - 2.0 * k.cos();
    let pred = (eps * eps + 2.0 * eps * rho / k2).sqrt();
    let steps = (6.0 * 2.0 * PI / pred).ceil() as usize;
    let mut ys = Vec::new();
    for _ in 0..steps {
        ys.push(u.density_mode_x(1));
        u.step();
    }
    let om = super::common::peak_frequency(&ys, 0.5 * pred, 1.5 * pred);
    println!("t96: ω = {om:.5} frente a {pred:.5} (razón {:.4}, ref. 0,994)", om / pred);
    assert!((om / pred - 1.0).abs() < 0.02);
    let (mut u0, _) = dirac_plasma(32, 4, 4, wp, 1.0, 0.0, 0.0);
    let mut emax = 0.0f64;
    for _ in 0..500 {
        u0.step();
        emax = emax.max(u0.em_energy().abs());
    }
    println!("t96: sin modular, energía del campo máxima {emax:e}");
    assert!(emax < 1e-20);
}

/// Test 101: la resonancia que desestabiliza el plasma de Dirac (κ = 1).
#[test]
fn t101_plasma_resonance() {
    let (mut u, _) = dirac_plasma(64, 4, 4, 0.2, 1.0, 0.0, 1e-7);
    let root = u.dirac_photon_resonance_x();
    let e0 = u.em_energy();
    let mut neg_max = 0.0f64;
    let mut neg_step = 0usize;
    let mut analysed: Option<Vec<f64>> = None;
    u.clear_field_snapshots();
    for t in 1..=300 {
        u.step();
        if analysed.is_none() {
            u.capture_field_snapshot();
            if u.em_energy() > 1e6 * e0.abs().max(1e-30) {
                analysed = Some(u.analyze_growing_mode_x(1));
            }
        }
        if t % 10 == 0 {
            let neg = 1.0 - u.spinor_band_population();
            if neg > neg_max {
                neg_max = neg;
                neg_step = t;
            }
        }
    }
    let g = analysed.unwrap_or_else(|| u.analyze_growing_mode_x(1));
    println!("t101: modo k_x = {:.3} (resonancia {root:.3}; ref. 2,258 y 2,286), ω = {:.4}, fotón {:.4} (ref. 1,880 y 1,876); banda −1 máx. {neg_max:.3} en el paso {neg_step}", g[0], g[1], g[2]);
    assert!((g[0] - root).abs() < 0.1 && (g[1] / g[2] - 1.0).abs() < 0.02 && neg_max > 0.3);
}

/// Test 102: con fotones lentos (κ = 0,1) el condensado es estable; con κ = 1 explota. La
/// referencia es la energía tras el primer paso: el E inicial de la ley de Gauss solo tiene la
/// parte longitudinal, y las corrientes del ruido llenan la transversal en ese paso.
#[test]
fn t102_slow_photons() {
    let (mut u, _) = dirac_plasma(12, 12, 12, 0.2, 0.1, 0.0, 1e-7);
    u.step();
    let e0 = u.em_energy();
    let mut emax = e0;
    let mut e_end = e0;
    for _ in 1..3000 {
        u.step();
        e_end = u.em_energy();
        emax = emax.max(e_end);
    }
    println!("t102: κ = 0,1: energía del campo {e0:e} → {e_end:e} (máx. {emax:e}; ref. 3,4e-13 → 2,7e-13)");
    let (mut v, _) = dirac_plasma(16, 16, 16, 0.2, 1.0, 0.0, 1e-7);
    v.step();
    let f0 = v.em_energy();
    let mut when = 0;
    for t in 2..=100 {
        v.step();
        if v.em_energy() > 1e4 * f0 {
            when = t;
            break;
        }
    }
    println!("t102: κ = 1: ×10⁴ en el paso {when} (ref. 59)");
    assert!(emax < 10.0 * e0 && when > 0);
}

/// Picos del espectro de Hann de una serie en [lo, hi] por encima de `frac` del máximo.
fn series_peaks(ys: &[f64], lo: f64, hi: f64, frac: f64) -> Vec<f64> {
    let n = 2000;
    let ws: Vec<f64> = (0..=n).map(|j| lo + (hi - lo) * j as f64 / n as f64).collect();
    let ps: Vec<f64> = ws.iter().map(|w| super::common::hann_power(ys, *w)).collect();
    let pmax = ps.iter().cloned().fold(0.0, f64::max);
    let mut out = Vec::new();
    for j in 1..n {
        if ps[j] > frac * pmax && ps[j] >= ps[j - 1] && ps[j] > ps[j + 1] {
            out.push(ws[j]);
        }
    }
    out
}

/// Test 112: efecto Hall anómalo del condensado polarizado (κ = 0,1, E_y uniforme).
#[test]
fn t112_anomalous_hall() {
    let mut res = Vec::new();
    for spin in [[0.0f32, 0.0, 1.0], [0.0, 1.0, 0.0]] {
        let mut u = dirac3d(16, 4, 4, 0.8);
        u.set_packet_spin(spin[0], spin[1], spin[2]);
        let mstar = 1.0 / u.dirac_band_curvature(0);
        u.seed_dirac_condensate((0.04 * mstar) as f32, 0.0, 0, 0.0, 0);
        u.set_field_substeps(2);
        u.set_em(2, 1.0, 0.1);
        u.set_uniform_electric_field(0.0, 1e-4, 0.0);
        let mut ys = Vec::new();
        for _ in 0..2048 {
            u.step();
            ys.push(u.field_mean(1));
        }
        let p = series_peaks(&ys, 0.1, 0.35, 0.05);
        println!("t112: espín {spin:?}: picos {p:?}");
        res.push(p);
    }
    assert!(res[0].len() == 2 && ((res[0][1] - res[0][0]) - 0.042).abs() < 0.005, "espín z: {:?}", res[0]);
    assert!(res[1].len() == 1 && (res[1][0] - 0.2025).abs() < 0.003, "espín y: {:?}", res[1]);
}



/// Escenario Weyl (2 componentes) con la variante dada; en 2D, con k_z fijo.
fn weyl(w: usize, h: usize, d: usize, variant: u32) -> U {
    let mut u = U::new(w, h, d, 0.5, 0.0, 1.0);
    u.set_scenario(3);
    u.set_weyl_variant(variant);
    u
}

/// Test 91: nodos de Weyl y su quiralidad (orden fijo y bombeos de Higashikawa).
#[test]
fn t91_weyl_nodes() {
    let u = weyl(4, 4, 4, 0);
    let h = 0.5 * PI;
    let cases = [([0.0, 0.0, 0.0], 1.0, 1.0), ([h, h, h], 1.0, -1.0), ([PI, 0.0, 0.0], -1.0, -1.0), ([h, h, -h], -1.0, 1.0)];
    for (k, f, det) in cases {
        let a = u.weyl_a(k);
        let err = (a[0] - f).abs().max((a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt());
        let dd = u.weyl_det_da(k);
        println!("t91: orden fijo, k = {k:?}: F = {f} (error {err:e}), det = {dd:.7} (quiralidad {})", if f > 0.0 { dd.signum() } else { -dd.signum() });
        assert!(err < 1e-12 && (dd - det).abs() < 1e-6);
    }
    let v = weyl(4, 4, 4, 2);
    for (k, det) in [([0.0, 0.0, 0.0], 2.0), ([0.0, 0.0, PI], -2.0)] {
        let a = v.weyl_a(k);
        let err = (a[0] - 1.0).abs().max((a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt());
        let dd = v.weyl_det_da(k);
        println!("t91: bombeos, k = {k:?}: F = +1 (error {err:e}), det = {dd:.7}");
        assert!(err < 1e-12 && (dd - det).abs() < 1e-6);
    }
    let mut rng = SplitMix64::new(91);
    let mut worst = 0.0f64;
    for _ in 0..20 {
        let k = [PI * (2.0 * rng.next_f64() - 1.0), PI * (2.0 * rng.next_f64() - 1.0), h];
        let a = v.weyl_a(k);
        worst = worst.max((a[0] + 1.0).abs().max((a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt()));
    }
    println!("t91: bombeos, plano k_z = π/2: |F + 1| ≤ {worst:e}");
    assert!(worst < 1e-12);
}

/// Test 92: el número de arrollamiento W₃ (malla de 48³).
#[test]
fn t92_weyl_winding() {
    let mut out = Vec::new();
    for (variant, half) in [(0u32, false), (1, false), (2, true), (2, false)] {
        let u = weyl(4, 4, 4, variant);
        let w3 = u.weyl_winding(48, half);
        println!("t92: variante {variant}, {}: W₃ = {w3:.5}", if half { "media zona" } else { "zona completa" });
        out.push(w3);
    }
    assert!(out[0].abs() < 0.01 && out[1].abs() < 0.01 && (out[2] - 1.0).abs() < 0.01 && out[3].abs() < 0.01);
}

/// Pico del espectro por autocorrelación más cercano a 0 (Weyl 2D con k_z fijo y B uniforme).
fn chiral_peak(variant: u32, kz: f32) -> f64 {
    let mut u = weyl(16, 16, 1, variant);
    u.set_weyl_kz(kz);
    u.set_em(1, 1.0, 0.5);
    u.set_uniform_magnetic_field(2);
    u.start_spinor_autocorrelation(93, 0.0, 0.0, 0.0, 0.0, false);
    u.step_n(4095);
    let peaks = spectrum_peaks(&mut u, 0.002);
    *peaks.iter().min_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap()).unwrap()
}

/// Test 93: la rama quiral (pendientes opuestas junto a un nodo y junto a su doblador).
#[test]
fn t93_chiral_branch() {
    let h = 0.5 * PI as f32;
    let fixed = [(0.1f32, 0.099), (-0.1, -0.099), (h + 0.1, -0.099), (h - 0.1, 0.099)];
    for (kz, want) in fixed {
        let p = chiral_peak(0, kz);
        println!("t93: orden fijo, k_z = {kz:.4}: pico {p:.4} (esperado {want})");
        assert!((p - want).abs() < 0.003);
    }
    let pi = PI as f32;
    let pumps = [(0.05f32, 0.112), (-0.05, -0.087), (pi + 0.05, -0.087), (pi - 0.05, 0.112)];
    for (kz, want) in pumps {
        let p = chiral_peak(2, kz);
        println!("t93: bombeos, k_z = {kz:.4}: pico {p:.4} (esperado {want})");
        assert!((p - want).abs() < 0.003);
    }
}

/// Test 97: la rama quiral en movimiento (efecto magnético quiral de una partícula). Sin filtrar,
/// el gaussiano es el de la referencia: tal cual en el gauge de Landau (sin la fase e^{iq·A(r₀)·r}),
/// así que la fracción que se lleva la rama depende de dónde está (es una cantidad de gauge). Se
/// comprueba la relación invariante: la deriva del espín arriba es esa fracción (medida con el
/// mismo filtro) por la velocidad de la rama, y el espín abajo casi no se mueve.
#[test]
fn t97_chiral_magnetic() {
    let build = || -> U {
        let mut u = weyl(24, 24, 128, 0);
        u.set_em(1, 1.0, 0.3);
        u.set_uniform_magnetic_field(4);
        u.set_flux_tracking(true);
        u
    };
    let h = 0.5 * PI as f32;
    let mut vbranch = 0.0;
    for (k, chi, want) in [([0.0f32, 0.0, 0.0], 1, 0.989), ([h, h, h], -1, -0.989)] {
        let mut u = build();
        u.inject_weyl_chiral_packet(12.0, 12.0, 64.0, 10.0, k[0], k[1], k[2], chi, 0.08, 300);
        let n0 = u.spinor_norm();
        let w0 = u.spinor_width(2);
        let mut v = 0.0;
        for _ in 0..60 {
            u.step();
            v += u.spinor_axis_flux(2) / n0;
        }
        v /= 60.0;
        if chi > 0 {
            vbranch = v;
        }
        let w1 = u.spinor_width(2);
        println!("t97: nodo {k:?} (χ = {chi}): v_z = {v:.4} (ref. {want}), anchura {w0:.3} → {w1:.3}");
        assert!((v - want).abs() < 0.005 && (w1 / w0 - 1.0).abs() < 0.02);
    }
    let mut drift = Vec::new();
    for sz in [1.0f32, -1.0] {
        let mut u = build();
        u.set_packet_gauge_compensation(false);
        u.set_packet_spin(0.0, 0.0, sz);
        let f = u.weyl_chiral_fraction(12.0, 12.0, 64.0, 10.0, 0.0, 0.0, 0.0, 1, 0.08, 300);
        u.inject_dirac_packet_3d_unprojected(12.0, 12.0, 64.0, 10.0, 0.0, 0.0, 0.0, 1, 0);
        let mut v = 0.0;
        for _ in 0..300 {
            u.step();
            v += u.spinor_axis_flux(2);
        }
        v /= 300.0;
        println!("t97: sin filtrar, espín {sz:+}: deriva {v:.4}; la rama se lleva {f:.4} (× {vbranch:.4} = {:.4}; ref. 0,1109 frente a 0,114 × 0,9889)", f * vbranch);
        drift.push((v, f));
    }
    assert!((drift[0].0 / (drift[0].1 * vbranch) - 1.0).abs() < 0.1 && drift[1].0.abs() < 0.04);
}

/// Test 99: la masa compleja de la cuerda (B₂ anticonmuta con β y las α; el paso de masa es
/// unitario) y la fase del par vórtice-antivórtice (devanado ±n y continuidad en las costuras).
#[test]
fn t99_complex_mass() {
    use crate::spinor::{alpha4, b2_matrix, beta4, cx, m4_add};
    let b2 = b2_matrix();
    let mut worst = 0.0f64;
    for x in [beta4(), alpha4(0), alpha4(1), alpha4(2)] {
        let ac = m4_add(&m4_mul(&b2, &x), &m4_mul(&x, &b2), cx(1.0, 0.0));
        worst = worst.max(m4_err(&ac, &[[cx(0.0, 0.0); 4]; 4]));
    }
    let mut rng = SplitMix64::new(99);
    let mut unit = 0.0f64;
    for _ in 0..200 {
        let (mu, th) = (2.0 * rng.next_f64(), 2.0 * PI * rng.next_f64());
        let mut m = m4_add(&beta4(), &b2, cx(0.0, 0.0));
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = cx(0.0, -mu.sin()) * (beta4()[i][j] * th.cos() + b2[i][j] * th.sin());
            }
            m[i][i] += cx(mu.cos(), 0.0);
        }
        let mm = m4_mul(&m, &m4_dag(&m));
        let mut id = [[cx(0.0, 0.0); 4]; 4];
        for i in 0..4 {
            id[i][i] = cx(1.0, 0.0);
        }
        unit = unit.max(m4_err(&mm, &id));
    }
    println!("t99: anticonmutadores {worst:e}, unitariedad {unit:e}");
    assert!(worst < 1e-15 && unit < 1e-12);
    for n in [1i32, 2] {
        let mut u = U::new(24, 24, 1, 0.6, 0.0, 1.0);
        u.set_scenario(2);
        u.set_dirac_slice(true);
        u.set_higgs_string(n, 0.6, 1.5);
        let th = |x: i32, y: i32| u.higgs_string_phase(x, y);
        let mut windings = Vec::new();
        for c in 0..2u32 {
            let p = u.higgs_string_core(c);
            let (cx0, cy0) = (p[0].round() as i32, p[1].round() as i32);
            let mut loop_pts = Vec::new();
            for x in cx0 - 3..cx0 + 3 {
                loop_pts.push((x, cy0 - 3));
            }
            for y in cy0 - 3..cy0 + 3 {
                loop_pts.push((cx0 + 3, y));
            }
            for x in (cx0 - 2..=cx0 + 3).rev() {
                loop_pts.push((x, cy0 + 3));
            }
            for y in (cy0 - 2..=cy0 + 3).rev() {
                loop_pts.push((cx0 - 3, y));
            }
            let mut s = 0.0;
            for k in 0..loop_pts.len() {
                let (a, b) = (loop_pts[k], loop_pts[(k + 1) % loop_pts.len()]);
                s += wrap_angle(th(b.0, b.1) - th(a.0, a.1));
            }
            windings.push((s / (2.0 * PI)).round() as i32);
        }
        let mut seam = 0.0f64;
        for x in 0..24 {
            seam = seam.max((wrap_angle(th(x, 0) - th(x, 23)) - wrap_angle(th(x, 1) - th(x, 0))).abs());
        }
        for y in 0..24 {
            seam = seam.max((wrap_angle(th(0, y) - th(23, y)) - wrap_angle(th(1, y) - th(0, y))).abs());
        }
        println!("t99: devanado {n}: alrededor de los núcleos {windings:?}; salto de costura frente al interior {seam:.4} (ref. ≤ 0,035 y 0,069)");
        assert!(windings == vec![n, -n] && seam < 0.1);
    }
}

/// Test 100: la cuerda quiral (orden fijo): espectro del corte 2D a k_z fijo en cada núcleo y el
/// paquete sencillo en 3D, que viaja en sentidos opuestos en los dos núcleos.
#[test]
fn t100_chiral_string() {
    for (kz, signs) in [(0.1f32, [1.0, -1.0]), (-0.1, [-1.0, 1.0])] {
        for core in 0..2u32 {
            let mut u = U::new(24, 24, 1, 0.6, 0.0, 1.0);
            u.set_scenario(2);
            u.set_dirac_slice(true);
            u.set_dirac_order(true);
            u.set_weyl_kz(kz);
            u.set_higgs_string(1, 0.6, 1.5);
            let p = u.higgs_string_core(core);
            u.start_spinor_autocorrelation(100 + core, p[0] as f32, p[1] as f32, 0.0, 4.0, false);
            u.step_n(4095);
            let nb = u.autocorrelation_spectrum() as usize;
            let sp = u.autocorr.spec.clone();
            let smax = (0..nb).filter(|&j| (sp[2 * j] as f64).abs() < 0.4).map(|j| sp[2 * j + 1] as f64).fold(0.0, f64::max);
            let mut peaks = Vec::new();
            for j in 1..nb - 1 {
                let (om, s) = (sp[2 * j] as f64, sp[2 * j + 1] as f64);
                if om.abs() < 0.4 && s > 0.1 * smax && s >= sp[2 * j - 1] as f64 && s > sp[2 * j + 3] as f64 {
                    peaks.push(om);
                }
            }
            let want = 0.085 * signs[core as usize];
            println!("t100: k_z = {kz}, núcleo {core}: picos con |ε| < 0,4 {peaks:?} (esperado {want})");
            assert!(peaks.len() == 1 && (peaks[0] - want).abs() < 0.003);
        }
    }
    for (core, want) in [(0u32, 0.83), (1, -0.83)] {
        let mut u = U::new(24, 24, 96, 0.6, 0.0, 1.0);
        u.set_scenario(2);
        u.set_dirac_order(true);
        u.set_higgs_string(1, 0.6, 1.5);
        u.inject_string_packet(core, 48.0, 8.0, 2.5);
        let p = u.higgs_string_core(core);
        let (mut ts, mut zs) = (Vec::new(), Vec::new());
        let mut last = 48.0;
        let mut unwrapped = 48.0;
        let mut frac = 0.0;
        for t in 0..=50 {
            let s = u.spinor_tube_stats(p[0] as f32, p[1] as f32, 5.0, 2);
            unwrapped += crate::util::min_image(s[1] - last, 96.0);
            last = s[1];
            if t >= 20 {
                ts.push(t as f64);
                zs.push(unwrapped);
            }
            frac = s[0] / u.spinor_norm();
            u.step();
        }
        let (_, v) = fit_linear(&ts, &zs);
        println!("t100: 3D, núcleo {core}: la parte del núcleo ({:.0} %) viaja a {v:.4} (ref. ±0,833)", 100.0 * frac);
        assert!((v - want).abs() < 0.03);
    }
}


/// Test 111: modos de Jackiw–Rebbi de un quark isoespinor en el fondo de un par monopolo-
/// antimonopolo (24³, G·v = 0,6, orden fijo). El peso espectral es la fracción de ∫S(ω) del
/// espectro por autocorrelación con |ε| < 0,1, para un estado aleatorio en r < 2.
#[test]
fn t111_jackiw_rebbi() {
    let mut w = Vec::new();
    for (k, p) in [(12.0f32, 12.0f32, 18.0f32), (12.0, 12.0, 6.0), (0.0, 0.0, 12.0)].into_iter().enumerate() {
        let mut u = dirac3d(24, 24, 24, 0.0);
        u.set_dirac_colour(2);
        u.set_dirac_order(true);
        u.set_monopole_background(12.0, 0.6, 2.0);
        u.start_spinor_autocorrelation(111 + k as u32, p.0, p.1, p.2, 2.0, false);
        u.step_n(4095);
        let wt = u.autocorrelation_spectrum_fraction(-0.1, 0.1);
        let norm_w = u.autocorrelation_weight(-0.1, 0.1);
        let peaks: Vec<f64> = spectrum_peaks(&mut u, 0.02).into_iter().filter(|q| q.abs() < 0.12).map(|q| (q * 1e4).round() / 1e4).collect();
        println!("t111: estado en {p:?}: peso espectral con |ε| < 0,1 = {:.3} % (fracción de la norma {:.3} %); picos {peaks:?}", 100.0 * wt, 100.0 * norm_w);
        w.push(wt);
    }
    assert!(w[0] > 0.01 && w[1] > 0.01 && w[2] < 0.005);
}


/// Test 117: Kaplan en 4+1D (L_s = 64, m₀ = 0,3, orden fijo): un Weyl por pared, de quiralidades
/// opuestas.
#[test]
fn t117_kaplan() {
    for k in [[0.1f32, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]] {
        let mut u = U::new(8, 8, 1, 0.5, 0.0, 1.0);
        u.set_scenario(5);
        u.set_scenario_kaplan(64, 0.3, k[0], k[1], k[2], true);
        let modes = u.kaplan_modes(0.2);
        let want = (0.3f64).cos() * 0.1;
        for wall in [0.0, 1.0] {
            let ws: Vec<&[f64; 5]> = modes.iter().filter(|m| m[1] == wall).collect();
            let pos = ws.iter().find(|m| m[0] > 0.0).expect("estado de energía positiva");
            println!("t117: k = {k:?}, pared {wall}: {:?}", ws.iter().map(|m| format!("ε {:.4} γ⁵ {:.3} hel {:.3} frac {:.3}", m[0], m[2], m[3], m[4])).collect::<Vec<_>>());
            assert_eq!(ws.len(), 2);
            for m in &ws {
                assert!((m[0].abs() - want).abs() < 0.002 && m[4] > 0.9);
            }
            let g = pos[2];
            assert!((g.abs() - 0.955).abs() < 0.01);
            assert!((pos[3] - g.signum()).abs() < 0.01, "helicidad {} con γ⁵ {}", pos[3], g);
        }
        let g0 = modes.iter().find(|m| m[1] == 0.0).unwrap()[2];
        let g1 = modes.iter().find(|m| m[1] == 1.0).unwrap()[2];
        assert!(g0 * g1 < 0.0);
    }
}

fn light(w: usize, h: usize) -> U {
    let mut u = U::new(w, h, 1, 0.0, 0.0, 1.0);
    u.set_scenario(4);
    u
}

/// Trayectoria (centroide sin envolver por ronda) de un paquete de luz en 700×400.
fn light_run(gm: f32, y0: f32, ym: f32, rounds: usize) -> Vec<[f64; 2]> {
    let mut u = light(700, 400);
    u.set_light_index(gm, 250.0, ym, 4.0);
    u.inject_light_packet(40.0, y0, 10.0, 20.0, 0.4);
    let mut tr = vec![[u.light_centroid(0), u.light_centroid(1)]];
    for _ in 0..rounds {
        u.step();
        tr.push([u.light_centroid(0), u.light_centroid(1)]);
    }
    tr
}

fn heading(tr: &[[f64; 2]], a: usize, b: usize) -> f64 {
    (tr[b][1] - tr[a][1]).atan2(tr[b][0] - tr[a][0])
}

/// Test 113: el cono isótropo (autovalores) y la desviación de la luz hacia la masa.
#[test]
fn t113_light_deflection() {
    let c = (0.3f64).cos() / 2.0;
    for (dx, dy) in [(1.0f32, 0.0f32), (0.0, 1.0), (1.0, 1.0)] {
        let u = light(8, 8);
        let v = u.light_cone_speed(0.3, dx, dy);
        println!("t113: cono en ({dx}, {dy}): {v:.4} frente a cos θ/2 = {c:.4}");
        assert!((v / c - 1.0).abs() < 0.01);
    }
    let (gm, b, ym) = (2.0f32, 70.0f32, 130.0f32);
    let rounds = 230;
    let tr = light_run(gm, ym + b, ym, rounds);
    let ctl = light_run(0.0, ym + b, ym, rounds);
    let defl = (heading(&tr, rounds - 10, rounds) - heading(&tr, 0, 10)) - (heading(&ctl, rounds - 10, rounds) - heading(&ctl, 0, 10));
    // integral eikonal de ∂_y ln n sobre el tramo recorrido, sin la deriva transversal propia del
    // cono (el desplazamiento del control en el medio uniforme, −0,24 celdas por ronda con q = 0,4):
    // es un artefacto de la red, no una desviación
    let mut eik = 0.0;
    let path: Vec<[f64; 2]> = (0..=rounds).map(|r| [tr[r][0], tr[r][1] - (ctl[r][1] - ctl[0][1])]).collect();
    for r in 0..rounds {
        let (x, y) = (0.5 * (path[r][0] + path[r + 1][0]), 0.5 * (path[r][1] + path[r + 1][1]));
        let ds = ((path[r + 1][0] - path[r][0]).powi(2) + (path[r + 1][1] - path[r][1]).powi(2)).sqrt();
        let (dx, dy) = (x - 250.0, y - ym as f64);
        let rho = (dx * dx + dy * dy + 16.0).sqrt();
        let n = 1.0 + 2.0 * gm as f64 / rho;
        eik += (-2.0 * gm as f64 * dy / rho.powi(3)) / n * ds;
    }
    println!("t113: desviación − control = {defl:.4} rad, integral eikonal {eik:.4} (razón {:.3}; ref. −0,115 frente a −0,102)", defl / eik);
    assert!(defl < 0.0 && defl / eik > 0.7 && defl / eik < 1.3);
}

/// Test 119: corrimiento al rojo cosmológico (a de 1 a 2 en 200 rondas). La frecuencia es la de la
/// componente de Fourier central (un k de la red), proyectada sobre el autovector instantáneo.
#[test]
fn t119_cosmological_redshift() {
    let mut u = light(512, 512);
    u.inject_light_packet(256.0, 256.0, 16.0, 16.0, 0.3);
    let t1 = 200.0 / (2f64.powf(1.5) - 1.0);
    u.set_light_metric_t(1, 2.0, t1 as f32);
    // la componente de Fourier de la red más cercana a la portadora π/2 + 0,3 (un modo propio)
    let dk = 2.0 * PI / 512.0;
    let k = [0.5 * PI + (0.3 / dk).round() * dk, 0.5 * PI];
    let mut last = u.light_mode_amplitude(k);
    let mut om = 0.0;
    for r in 0..200 {
        u.step();
        let a = u.light_mode_amplitude(k);
        if r == 199 {
            om = -wrap_angle((a / last).arg());
        }
        last = a;
    }
    let pred = u.light_band_energy(k[0] as f32, k[1] as f32, 199.5);
    let up = u.light_upper_weight();
    println!("t119: ω = {om:.5} frente a la energía de banda instantánea {pred:.5} (razón {:.4}); peso en la banda superior {up:.6}", om / pred);
    assert!((om / pred - 1.0).abs() < 0.01 && up > 0.999);
}

/// Velocidad de grupo de la banda superior (por ronda) en la dirección del eje con los θ dados.
fn light_group_velocity(thx: f64, thy: f64, k: [f64; 2], axis: usize) -> f64 {
    let h = 1e-5;
    let mut kp = k;
    kp[axis] += h;
    let mut km = k;
    km[axis] -= h;
    (crate::light::light_upper(thx, thy, kp).0 - crate::light::light_upper(thx, thy, km).0) / (2.0 * h)
}

/// Test 120: onda gravitacional (h₀ = 0,05, período de 120 rondas): el interferómetro.
#[test]
fn t120_gravitational_wave() {
    let run = |along_y: bool, h0: f32| -> f64 {
        let mut u = light(256, 256);
        if along_y {
            u.inject_light_packet_y(128.0, 128.0, 16.0, 0.3);
        } else {
            u.inject_light_packet(128.0, 128.0, 16.0, 16.0, 0.3);
        }
        u.set_light_metric_t(2, h0, 120.0);
        let a = if along_y { 1 } else { 0 };
        let p0 = u.light_centroid(a);
        u.step_n(60);
        u.light_centroid(a) - p0
    };
    let d = (run(false, 0.05) - run(false, 0.0)) - (run(true, 0.05) - run(true, 0.0));
    let c0 = (0.3f64).cos();
    let mut pred = 0.0;
    let (kx, ky) = ([0.5 * PI + 0.3, 0.5 * PI], [0.5 * PI, 0.5 * PI + 0.3]);
    let (vx0, vy0) = (light_group_velocity(0.3, 0.3, kx, 0), light_group_velocity(0.3, 0.3, ky, 1));
    for r in 0..60 {
        let h = 0.05 * (2.0 * PI * (r as f64 + 0.5) / 120.0).sin();
        let (tx, ty) = ((c0 / (1.0 + h).sqrt()).acos(), (c0 / (1.0 - h).sqrt()).acos());
        pred += (light_group_velocity(tx, ty, kx, 0) - vx0) - (light_group_velocity(tx, ty, ky, 1) - vy0);
    }
    println!("t120: diferencia de recorridos en la ronda 60: {d:.3} celdas frente a {pred:.3} (razón {:.3}; ref. −4,12 frente a −3,95)", d / pred);
    assert!((d / pred - 1.0).abs() < 0.1);
}


/// Escenario Dirac 2D (2 componentes) con masa m.
fn dirac2(w: usize, h: usize, m: f32) -> U {
    let mut u = U::new(w, h, 1, 0.5, 0.0, 1.0);
    u.set_scenario(2);
    u.set_mass(m);
    u
}

/// Aceleración en x (ajuste cuadrático del centroide) durante `steps` pasos.
fn accel_x(u: &mut U, steps: usize) -> f64 {
    let mut xs = Vec::new();
    for t in 0..=steps {
        xs.push(u.center_of_mass(0));
        if t < steps {
            u.step();
        }
    }
    accel(&xs)
}

/// Test 74: principio de equivalencia de Nordström (m₀ = 0,1, 0,2 y 0,3 caen igual).
#[test]
fn t74_nordstrom_equivalence() {
    let grad = 0.05 * (2.0 * PI / 1024.0) * (2.0 * PI * 256.0 / 1024.0).sin();
    let mut accs = Vec::new();
    for m0 in [0.1f32, 0.2, 0.3] {
        let mut u = dirac2(1024, 4, m0);
        u.set_conformal_potential(1, -0.05, 0.0, 0.0, 1.0);
        u.inject_dirac_packet(256.0, 2.0, 60.0, 0.0, 0.0, 1);
        let a = accel_x(&mut u, 300);
        println!("t74: m₀ = {m0}: a = {a:e} ({:.3} de −∇Φ = {:e})", a / -grad, -grad);
        accs.push(a);
    }
    let mean = accs.iter().sum::<f64>() / 3.0;
    for a in &accs {
        assert!((a / mean - 1.0).abs() < 0.03 && (a / -grad - 1.0).abs() < 0.1);
    }
}

/// Fase de ψ↑ en una celda desenrollada en el tiempo, y su pendiente en [t0, t1].
fn clock_rate(u: &mut U, x: usize, y: usize, t0: usize, t1: usize) -> f64 {
    let (mut ts, mut ph) = (Vec::new(), Vec::new());
    let (mut acc, mut last) = (0.0, None::<f64>);
    for t in 0..=t1 {
        let i = y * u.width() + x;
        let a = (u.buffer_read[4 * i + 1] as f64).atan2(u.buffer_read[4 * i] as f64);
        if let Some(l) = last {
            acc += wrap_angle(a - l);
        }
        last = Some(a);
        if t >= t0 {
            ts.push(t as f64);
            ph.push(acc);
        }
        u.step();
    }
    -fit_linear(&ts, &ph).1
}

/// Test 75: corrimiento al rojo de Nordström: relojes de m₀ = 0,2 en Φ = −0,05, 0 y +0,05.
#[test]
fn t75_nordstrom_redshift() {
    let mut f = Vec::new();
    for (kind, x) in [(1u32, 0usize), (0, 256), (1, 512)] {
        let mut u = dirac2(1024, 4, 0.2);
        u.set_conformal_potential(kind, -0.05, 0.0, 0.0, 1.0);
        u.inject_dirac_packet(x as f32, 2.0, 6.0, 0.0, 0.0, 1);
        let om = clock_rate(&mut u, x, 2, 100, 600);
        println!("t75: Φ = {:+.2}: ω = {om:.5}", if kind == 0 { 0.0 } else { -0.05 * (2.0 * PI * x as f64 / 1024.0).cos() });
        f.push(om);
    }
    let (d1, d2) = (f[1] - f[0], f[2] - f[1]);
    println!("t75: diferencias {d1:.5} y {d2:.5} frente a m₀·ΔΦ = 0,0100 (ref. 0,00994 y 0,00987)");
    assert!((d1 / 0.01 - 1.0).abs() < 0.03 && (d2 / 0.01 - 1.0).abs() < 0.03);
}

/// Test 76: una partícula rápida pesa menos (Nordström frente a la relatividad general).
#[test]
fn t76_fast_particle_lighter() {
    let mut accs = Vec::new();
    for n in [0u32, 2, 4] {
        let ky = 2.0 * PI * n as f64 / 64.0;
        let mut u = dirac2(1024, 64, 0.2);
        u.set_conformal_potential(1, -0.05, 0.0, 0.0, 1.0);
        u.inject_dirac_plane_wave_y(256.0, 60.0, ky as f32, 1);
        let a = accel_x(&mut u, 300);
        let e = u.dirac_dispersion(0.0, ky as f32);
        println!("t76: k_y = {ky:.3}: a = {a:e}, (m/E)² = {:.3}", (0.2 / e).powi(2));
        accs.push((a, (0.2 / e).powi(2)));
    }
    let r = accs[1].0 / accs[0].0;
    println!("t76: razón {r:.3} frente a (m/E)² = {:.3} (ref. 0,474 frente a 0,509)", accs[1].1);
    assert!(accs[0].0.abs() > accs[1].0.abs() && accs[1].0.abs() > accs[2].0.abs());
    assert!((r / accs[1].1 - 1.0).abs() < 0.15);
}

/// Pendiente de −arg c(t) (autocorrelación del estado inicial) en [0, t1].
fn autocorr_rate(u: &mut U, t1: usize) -> f64 {
    u.start_autocorrelation_from_current();
    u.step_n(t1 as u32);
    let c = &u.autocorr.c[..u.autocorr.count];
    let (mut ts, mut ph) = (Vec::new(), Vec::new());
    let mut acc = 0.0;
    for t in 0..c.len() {
        if t > 0 {
            acc += wrap_angle(c[t].arg() - c[t - 1].arg());
        }
        ts.push(t as f64);
        ph.push(acc);
    }
    -fit_linear(&ts, &ph).1
}

/// Test 81: corrimiento al rojo de Schwarzschild en 1+1D (M = 40, m₀ = 0,3).
#[test]
fn t81_schwarzschild_redshift() {
    let (m, x0) = (40.0f64, 1500.0f64);
    for rr in [2.0f64, 4.0, 10.0] {
        let r = rr * 2.0 * m;
        let mut u = dirac2(4096, 4, 0.3);
        u.set_conformal_potential(3, m as f32, x0 as f32, 0.0, 1.0);
        let xs = u.tortoise_coordinate(r as f32, m as f32);
        u.inject_dirac_packet((x0 + xs) as f32, 2.0, 40.0, 0.0, 0.0, 1);
        // ventana corta: el reloj cae (a ≈ 10⁻³ junto a r = 4M) y con 200 pasos se desplaza ~30
        // celdas, lo que mete en la fase de c(t) el momento que gana
        let om = autocorr_rate(&mut u, 40);
        let pred = 0.3 * (1.0 - 2.0 * m / r).sqrt();
        println!("t81: r/2M = {rr}: ω = {om:.5} frente a m₀·√(1 − 2M/r) = {pred:.5} ({:+.2} %)", 100.0 * (om / pred - 1.0));
        assert!((om / pred - 1.0).abs() < 0.005);
    }
}

/// Test 82: caída hacia el horizonte desde r₀ = 240 frente a la geodésica.
#[test]
fn t82_falling_to_horizon() {
    let (m, x0) = (40.0f32, 1500.0f64);
    let mut u = dirac2(4096, 4, 0.3);
    u.set_conformal_potential(3, m, x0 as f32, 0.0, 1.0);
    let xs0 = u.tortoise_coordinate(240.0, m);
    u.inject_dirac_packet((x0 + xs0) as f32, 2.0, 40.0, 0.0, 0.0, 1);
    let geo = u.schwarzschild_geodesic(xs0 as f32, m, 1500);
    let mut traj = Vec::new();
    for t in 0..=1500 {
        traj.push(u.center_of_mass(0) - x0);
        if t < 1500 {
            u.step();
        }
    }
    let (mut e1, mut e2) = (0.0f64, 0.0f64);
    for t in 0..=1500 {
        let err = (traj[t] - geo[t]).abs();
        if t <= 600 {
            e1 = e1.max(err);
        } else {
            e2 = e2.max(err / (xs0 - geo[t]).abs());
        }
    }
    let v = (traj[1500] - traj[1450]) / 50.0;
    let rf = u.tortoise_radius(traj[1500] as f32, m);
    println!("t82: error ≤ {e1:.2} celdas hasta t = 600 y ≤ {:.2} % del recorrido después; dx*/dt final {v:.4}; r final {rf:.2}", 100.0 * e2);
    assert!(e1 < 2.0 && e2 < 0.05 && (v + 1.0).abs() < 0.02 && rf > 2.0 * m as f64);
}



