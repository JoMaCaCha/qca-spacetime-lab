//! Tests de la fase 16 (177–182): plataforma de investigación en física fundamental —
//! DMRG variacional de dos sitios (Bethe + Calabrese–Cardy CFT c = 1), restauración infrarroja
//! de Lorentz en dos especies acopladas (Collins et al. vs Chadha–Nielsen), Monte Carlo dinámico
//! de conjuntos causales 2D con acción de Benincasa–Dowker (Surya 2012), Monte Carlo cuántico
//! Yang–Mills SU(2) en 3+1D euclídeo (Creutz 1980), gravedad de espín 2 frente a modelos escalares
//! (Weinberg–Witten), redes 3D no cúbicas (SC/BCC/FCC/isotrópica), motor de verosimilitud sobre
//! catálogos astrofísicos reales y fermiones quirales de frontera finita (Kaplan y Sen 2024).

use crate::QuantumSpacetimeUniverse;

#[test]
fn t177_dmrg_cft_scaling_and_lorentz_rg() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    // 1. DMRG variacional en cadena crítica interactuante XXZ (Δ = 1/2, K = 3/4): L = 16, 24, 32, 48
    let p = u.dmrg_xxz_cft_scaling(32, 5);
    let r = unsafe { std::slice::from_raw_parts(p, 10) }.to_vec();
    println!(
        "t177 DMRG XXZ(Δ=0,5): e_inf={:.6} (Bethe {:.6}, err={:.2e}), c_CFT={:.4} (exacto 1), S_mid={:.4}, e(L)=[{:.5}, {:.5}, {:.5}, {:.5}]",
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]
    );
    assert!(r[2] < 3e-3, "e_inf extrapolado debe coincidir con Bethe −3/8 al 0,3%: err = {}", r[2]);
    assert!((r[3] - 1.0).abs() < 0.08, "carga central CFT de Calabrese–Cardy debe ser c = 1,0 ± 0,08: {}", r[3]);
    assert!(r[8] < r[5], "la energía por sitio debe bajar monótonamente con L: {} vs {}", r[8], r[5]);

    // 2. Dos especies acopladas en DMRG + flujo RG exacto de Chadha–Nielsen (M_P → m_e)
    let p2 = u.two_species_lorentz_rg_study(10, 1.0, 0.5, 0.8, 24);
    let rg = unsafe { std::slice::from_raw_parts(p2, 10) }.to_vec();
    println!(
        "t177 2-species RG: dE_bind={:.4}, zz_free={:.4} → zz_coup={:.4}, S_free={:.3} → S_coup={:.3}; QED ratio={:.4} (exact {:.4}); strong IR dc={:.2e}; gamma_min={:.3}",
        rg[0], rg[1], rg[2], rg[3], rg[4], rg[5], rg[6], rg[7], rg[8]
    );
    assert!(rg[0] > 0.15, "el acoplamiento entre especies debe ligar energía por peldaño: {}", rg[0]);
    assert!(rg[1].abs() < 0.03, "control sin acoplamiento: correlación entre especies ≈ 0: {}", rg[1]);
    assert!(rg[2] < -0.08, "el acoplamiento antiferromagnético g_AB > 0 debe correlacionar las dos especies: {}", rg[2]);
    assert!((rg[5] - rg[6]).abs() < 1e-3, "flujo RG QED RK4 debe coincidir con (α_IR/α_UV)⁴: {} vs {}", rg[5], rg[6]);
    // Resultado científico clave: QED débil solo suprime δc un ~30% (falla por >18 órdenes!), mientras γ_*=0,90 baja de 10⁻²⁰
    assert!(rg[5] > 0.50, "QED débil NO cura Collins et al.: ratio = {}", rg[5]);
    assert!(rg[7] < 1e-20, "un sector fuerte con γ_* = 0,90 sí suprime δc por debajo de 10⁻²⁰: {:.2e}", rg[7]);
    assert!(rg[8] > 0.75 && rg[8] < 0.90, "dimensión anómala crítica γ_min ≈ 0,82: {}", rg[8]);
}

#[test]
fn t178_dynamical_causal_set_bd_phase_transition() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.causal_set_bd_mc(28, 1200, 16);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    println!(
        "t178 Causal Set BD MC: β=0 (S/N={:.3}, d_MM={:.3}, R={:.3}) | β=2,0 (S/N={:.3}, d_MM={:.3}, R={:.3}) | β=-0,25 (S/N={:.3}, N1/N={:.2})",
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]
    );
    // β = 0: variedad 2D de Minkowski (d_MM ≈ 2, R ≈ 0,5, acción plana S_BD/N ≈ 0)
    assert!((out[1] - 2.0).abs() < 0.12, "en β = 0 debe recuperar d_MM ≈ 2: {}", out[1]);
    assert!((out[2] - 0.5).abs() < 0.05, "en β = 0 la fracción de ordenamiento R ≈ 0,5: {}", out[2]);
    assert!(out[0].abs() < 0.5, "en β = 0 la acción S_BD/N ≈ 0 (espacio-tiempo plano): {}", out[0]);
    // β = +2,0: transición de fase de Surya (2012) a fase cristalina de capas (N₂ ~ O(N²), S_BD/N ≫ 1)
    assert!(out[3] > 3.0, "en β = 2,0 la acción S_BD/N debe saltar a la fase cristalina de capas (≫ 1): {}", out[3]);
    // β = −0,25: colapso a fase bipartita cristalina (N₁ ~ O(N²), S_BD/N ≪ 0)
    assert!(out[6] < -3.0, "en β = −0,25 colapsa a fase bipartita cristalina (S_BD/N ≪ 0): {}", out[6]);
    assert!(out[7] > 2.5, "en fase bipartita N₁/N crece muy por encima de la fase de variedad: {}", out[7]);
}

#[test]
fn t179_lattice_su2_3p1d_quantum_confinement() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.lattice_su2_4d_mc(4, 30, 50, 16);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    println!(
        "t179 4D SU(2) MC (4⁴): β=1,0 P={:.4} (fuerte {:.4}); β=3,0 P={:.4} (débil {:.4}); χ(2,2) β=1,5: {:.3}, β=2,0: {:.3}",
        out[0], out[1], out[2], out[3], out[4], out[5]
    );
    // Acoplamiento fuerte β = 1,0: ⟨P⟩ ≈ β/4 − β³/96 ≈ 0,240
    assert!((out[0] - out[1]).abs() < 0.025, "acoplamiento fuerte β = 1,0: {} vs {}", out[0], out[1]);
    // Acoplamiento débil β = 3,0: ⟨P⟩ cercano a 1 − 3/(4β) = 0,75 (exacto 4⁴ ≈ 0,722)
    assert!(out[2] > 0.68 && out[2] < 0.77, "acoplamiento débil β = 3,0: {} (pred 1-lazo {})", out[2], out[3]);
    // Confinamiento cuántico 3+1D: tensión de cuerda positiva (χ(2,2) > 0) y decrece con β
    assert!(out[4] > 0.45, "tensión de cuerda positiva en β = 1,5: χ(2,2) = {}", out[4]);
    assert!(out[5] > 0.15 && out[5] < out[4], "χ(2,2) en β = 2,0 positiva y menor que en β = 1,5: {} vs {}", out[5], out[4]);
}

#[test]
fn t180_spin2_vs_scalar_and_weinberg_witten() {
    let u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let r = u.spin2_emergent_gravity_check();
    println!(
        "t180 spin-2 vs scalar: err_TT={:.1e}, err_rot45={:.1e}, light_ratio(Einstein/Nordstrom/Refract)=({:.1}, {:.1}, {:.1}), WW_proj_h2={:.1e}",
        r[0], r[1], r[2], r[3], r[4], r[6]
    );
    assert!(r[0] < 1e-14, "polarizaciones h₊, h_× deben ser exactamente transversales y sin traza");
    assert!(r[1] < 1e-14, "bajo rotación de 45°, h₊ debe transformarse exactamente en h_× (espín 2)");
    assert_eq!(r[2], 2.0, "Einstein espín 2 da factor 2 de deflexión de la luz");
    assert_eq!(r[3], 0.0, "Nordström escalar da 0 deflexión de la luz");
    assert!(r[6] < 1e-14, "obstrucción de Weinberg–Witten: proyección de T^μν sobre |h| = 2 es exactamente 0");
}

#[test]
fn t181_noncubic_lattices_and_real_catalog_likelihood() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    // 1. Mapa de redes 3D no cúbicas
    let m = u.noncubic_lattice_lv_map();
    println!(
        "t181 noncubic 3D: asym_SC_fix={:.4e}, asym_BCC_fix={:.4e} | spread_A2: SC={:.4}, BCC={:.4}, FCC={:.4}, ISO(SC+FCC)={:.2e} (O(k⁴)={:.4})",
        m[0], m[5], m[1], m[2], m[3], m[4], m[10]
    );
    assert!(m[0] > 1e-3, "SC de orden fijo tiene asimetría CPT-impar: {}", m[0]);
    assert!(m[5] > 1e-4, "BCC de orden fijo también rompe ω(k) = ω(−k): {}", m[5]);
    // Valores analíticos exactos: SC = 1/36 = 0,0278, BCC = 1/18 = 0,0556, FCC = 1/72 = 0,0139
    assert!(m[1] > 0.025 && m[2] > 0.050 && m[3] > 0.012, "SC, BCC y FCC puras tienen anisotropía direccional O(k²Δx²)");
    // Esténcil isotrópico SC+FCC: cancela exactamente la anisotropía direccional de orden k²Δx² (>10⁴× menor!)
    assert!(m[4] < 1e-5, "esténcil isotrópico SC+FCC debe cancelar la anisotropía direccional en k²: {}", m[4]);

    // 2. Motor de verosimilitud de perfil χ² sobre catálogos reales
    let p = u.real_catalog_lv_likelihood(600);
    let r = unsafe { std::slice::from_raw_parts(p, 10) }.to_vec();
    println!(
        "t181 catalog χ²: MB polarim 95%CL < {:.3e}·ℓ_P, 5σ < {:.3e}·ℓ_P (med 95% {:.3e}); KM3NeT cons(110 PeV) < {:.3e}·ℓ_P; K1(221009A)={:.3e} s",
        r[0], r[1], r[2], r[3], r[7]
    );
    assert!(r[0] < 3.0e-6, "cota 95% CL perfil χ² en la peor orientación debe ser < 3e-6 ℓ_P: {:.3e}", r[0]);
    assert!(r[1] < 8.0e-6, "exclusión 5σ en TODA orientación debe ser < 8e-6 ℓ_P: {:.3e}", r[1]);
    assert!(r[2] < r[0], "mediana debe ser más estricta que la peor orientación");
    assert!(r[3] < 1e-10, "KM3NeT conservador (110 PeV) excluye Weyl orden fijo en >10¹⁰: {:.3e}", r[3]);
    // Integral cosmológica K_1(z=0,151) ≈ 0,15·(1/H0) ≈ 7e16 s
    assert!(r[7] > 5e16 && r[7] < 1e17, "integral cosmológica K₁(GRB 221009A) ≈ 7·10¹⁶ s: {:.3e}", r[7]);
}

#[test]
fn t182_kaplan_sen_boundary_chiral_and_4comp_qca() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.chiral_4comp_kaplan_sen_test(14, 0.45);
    let r = unsafe { std::slice::from_raw_parts(p, 11) }.to_vec();
    println!(
        "t182 Kaplan–Sen (L=14, m=0,45): ω_edge={:.5} (sin kx={:.5}), χ_L={:.4}, χ_R={:.4}, suma={:.2e}, decay={:.4} (pred {:.4}), gap_dbl={:.3}, QCA4(r=0/1)=({:.2}/{:.2})",
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]
    );
    assert!((r[0] - r[1]).abs() < 1e-3, "dispersión del modo Weyl de frontera debe ser sin(k_x): {} vs {}", r[0], r[1]);
    assert!(r[2] > 0.99 && r[3] < -0.99, "frontera izquierda (+1) y derecha (−1) puramente quirales: {} y {}", r[2], r[3]);
    assert!(r[4].abs() < 1e-3, "suma de quiralidades del contorno cerrado = 0 (Golterman–Shamir): {}", r[4]);
    assert!((r[5] - r[6]).abs() < 0.02, "localización exponencial (1−m)² en la frontera: {} vs {}", r[5], r[6]);
    assert!(r[7] > 1.2, "los 7 dobladores en k=(π,0,0) desaparecen con hueco grande: {}", r[7]);
    assert!(r[8] < 1e-12 && r[9] > 1.9, "QCA 4-comp Wilson: r=0 tiene dobladores (0) y r=1 abre hueco = 2");
}
