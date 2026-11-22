//! Fase 13 (contraste con la física real, §2.30 y §3.28): tests 163 y 164.

use crate::QuantumSpacetimeUniverse as U;

fn coeffs(u: &mut U, w: u32) -> Vec<f64> {
    let p = u.lorentz_violation_coefficients(w);
    unsafe { std::slice::from_raw_parts(p, 10) }.to_vec()
}

/// Test 163: orden y coeficientes de la violación de Lorentz de cada caminata, cotas de LHAASO y la
/// velocidad de la caminata de Grover con k ≫ m.
#[test]
fn t163_planck_kinematics() {
    let mut u = U::new(8, 8, 1, 0.5, 0.0, 1.0);
    let names = ["Weyl de orden fijo", "Weyl alternado", "Higashikawa", "fotón de red", "Wilson–Dirac", "fotón de Mlodinow y Brun"];
    let mut all = Vec::new();
    for w in 0..6u32 {
        let c = coeffs(&mut u, w);
        println!("{}: n = {}, A mín {:+.4}, máx {:+.4}, rms {:.4}, ω(k) − ω(−k) = {:+.2e}, Δx/ℓ_P < {:.3e} (rms) y {:.3e} (peor)", names[w as usize], c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
        all.push(c);
    }
    // n = 1: orden fijo (A₁ = −2·nx·ny·nz) y el fotón de Mlodinow y Brun (A₁ = −nx·ny·nz)
    assert_eq!(all[0][0], 1.0);
    assert!((all[0][2] / 0.3849 - 1.0).abs() < 0.01);
    assert!((all[0][3] / 0.1952 - 1.0).abs() < 0.02);
    assert_eq!(all[5][0], 1.0);
    assert!((all[5][2] / 0.19245 - 1.0).abs() < 0.01);
    assert!((all[5][3] / 0.0976 - 1.0).abs() < 0.02);
    assert!((all[5][5] - 1.25).abs() < 0.05, "cota del fotón de Mlodinow y Brun con el rms: {}", all[5][5]);
    // n = 2: las caminatas de la app con ω par
    for w in 1..5 {
        assert_eq!(all[w][0], 2.0, "{}", names[w]);
        assert!(all[w][4].abs() < 1e-12, "{}: ω(k) − ω(−k) = {}", names[w], all[w][4]);
    }
    // el fotón de red es subluminal en todas las direcciones y su cota supera 10⁷·ℓ_P
    assert!((all[3][1] / -0.0937 - 1.0).abs() < 0.05 && (all[3][2] / -0.0104 - 1.0).abs() < 0.05);
    assert!(all[3][6] > 1e7);
    // Grover con k ≫ m: v → ‖n̂‖∞
    let v111 = u.grover_relativistic_speed(3, 0.001, 100.0, 1.0, 1.0, 1.0);
    let v110 = u.grover_relativistic_speed(3, 0.001, 100.0, 1.0, 1.0, 0.0);
    let v100 = u.grover_relativistic_speed(3, 0.001, 100.0, 1.0, 0.0, 0.0);
    println!("Grover 3D, k = 100·m: v = {v100:.4} (1,0,0), {v110:.4} (1,1,0) y {v111:.4} (1,1,1); ‖n̂‖∞ = 1, 0,7071 y 0,5774");
    assert!((v111 - 0.5773).abs() < 0.001);
    assert!((v110 - 0.7070).abs() < 0.001);
}

/// Test 164: el fotón autoconjugado de una regla con ω(k) ≠ ω(−k) es birrefringente (sus dos helicidades van
/// con ω(k) y ω(−k)), y las cotas de polarización de Kislat y Krawczynski acotan Δx muy por debajo de ℓ_P.
#[test]
fn t164_self_conjugate_photon_birefringence() {
    let mut u = U::new(8, 8, 1, 0.5, 0.0, 1.0);
    let p = u.photon_polarization_rotation(0.21, 0.13, 0.07, 4000);
    let r = unsafe { std::slice::from_raw_parts(p, 4) }.to_vec();
    println!("fotón de Mlodinow y Brun como campo real: rotación {:+.5e} rad/paso, (ω(k) − ω(−k))/2 = {:+.5e}; helicidades {:+.4} y {:+.4}", r[0], r[1], r[2], r[3]);
    assert!((r[0] / r[1] - 1.0).abs() < 1e-3);
    assert!(r[2] > 0.99 && r[3] < -0.99);
    let names = ["Weyl de orden fijo como fotón", "fotón de Mlodinow y Brun"];
    for (i, w) in [0u32, 5].iter().enumerate() {
        let c = coeffs(&mut u, *w);
        println!("{}: Δx/ℓ_P < {:.3e} (mediana de orientaciones) y {:.3e} (peor); norma de k_(V)^(5) = {:.4}·Δx", names[i], c[7], c[8], c[9]);
    }
    let c0 = coeffs(&mut u, 0);
    let c5 = coeffs(&mut u, 5);
    // norma de j = 3: c·√(4π/105)/2 con c = 2 y 1 (Σ Y·k = A₁·Δx/2)
    assert!((c0[9] / (4.0 * std::f64::consts::PI / 105.0).sqrt() - 1.0).abs() < 0.01);
    assert!((c5[9] / (0.5 * (4.0 * std::f64::consts::PI / 105.0).sqrt()) - 1.0).abs() < 0.01);
    assert!((c5[7] / 1.38e-6 - 1.0).abs() < 0.1 && c5[8] > 2.0e-6 && c5[8] < 2.45e-6, "{} {}", c5[7], c5[8]);
    assert!((c0[8] / c5[8] - 0.5).abs() < 0.01);
    // las reglas de ω par no tienen este mecanismo
    for w in 1..5 {
        assert!(coeffs(&mut u, w)[7].is_nan());
    }
}
