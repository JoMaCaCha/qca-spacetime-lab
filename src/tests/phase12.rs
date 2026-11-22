//! Fase 12 (la ecuación efectiva, aprendida de los datos, §2.29 y §3.27): test 162.

use crate::QuantumSpacetimeUniverse as U;

const NAMES: [&str; 11] = ["φ", "−i∂φ", "∂²φ", "i∂³φ", "∂⁴φ", "ρφ", "ρ²φ", "ρ³φ", "(∂²ρ)φ", "iρφ", "i∂²φ"];

fn discover(u: &mut U, m: f32, g: f32, alpha: f32, seed: u32) -> Vec<f64> {
    let p = u.discover_envelope_equation(m, g, alpha, seed);
    unsafe { std::slice::from_raw_parts(p, 14) }.to_vec()
}

/// Test 162: la regresión dispersa recupera m*, g y E_max de la caminata, descarta la deriva y la
/// disipación, y encuentra la no linealidad con derivadas del espinor de la banda.
#[test]
fn t162_learned_envelope_equation() {
    let mut u = U::new(16, 16, 1, 0.5, 0.1, 1.0);
    let c = discover(&mut u, 0.5, 0.1, 0.0, 1);
    for (n, v) in NAMES.iter().zip(&c) {
        println!("   {n:8} {v:+.6}");
    }
    let (res, mstar, emax) = (c[11], c[12], c[13]);
    let pred_mstar = 2.0 * 0.5f64.tan();
    let pred_c = -0.1 / (8.0 * 0.5f64.sin().powi(2));
    println!("residuo {res:.2e}; m* = {mstar:.4} (2·tan 0,5 = {pred_mstar:.4}; ref. 1,0901), g = {:.4} (ref. 0,1000), E_max = {emax:.3} (ref. 0,95), (∂²ρ)φ = {:.4} (−g/(8 sen²m) = {pred_c:.4}; ref. −0,0418)", -c[5], c[8]);
    assert!((mstar / pred_mstar - 1.0).abs() < 0.01);
    assert!((-c[5] / 0.1 - 1.0).abs() < 0.02);
    assert!(emax > 0.8 && emax < 1.2);
    assert_eq!(c[1], 0.0, "deriva −i∂φ");
    assert_eq!(c[3], 0.0, "deriva i∂³φ");
    assert_eq!(c[10], 0.0, "disipación i∂²φ");
    let ratio = c[8] / pred_c;
    assert!(ratio > 0.6 && ratio < 1.0, "(∂²ρ)φ / predicción = {ratio}");
    // repulsiva: el signo de g sale de los datos
    let r = discover(&mut u, 0.5, -0.05, 0.0, 1);
    println!("g = −0,05: c_ρφ = {:+.5} (+0,0498)", r[5]);
    assert!((r[5] / 0.05 - 1.0).abs() < 0.02);
}
