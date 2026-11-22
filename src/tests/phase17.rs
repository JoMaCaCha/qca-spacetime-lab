//! Tests de la fase 17 (183–185): los tres estudios numéricos de preguntas abiertas —
//! (1) diagrama de fases no perturbativo de restauración de Lorentz vs hueco de masa en dos
//! especies acopladas (DMRG + SWAP-TEBD), (2) clasificación exhaustiva de 8 familias QCA 3+1D y
//! construcción del QCA isotrópico SC+FCC de orden n = 4 sin dobladores, y (3) propagador de
//! Johnston + retroacción de materia cuántica en gravedad de conjuntos causales 2D con barras de error.

use crate::QuantumSpacetimeUniverse;

#[test]
fn t183_study1_two_species_lorentz_phases() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.study1_two_species_lorentz_phases(24, 16);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let v_harm = out[1];
    let names = ["Control (g=0)", "1A Dens-Dens (g_z=0.6)", "1B Heisenberg (g=1.4)", "1C Mezcla (g_xy=0.75)"];
    println!("t183 Study 1 (v_harm_pred = {:.4}):", v_harm);
    let mut rows = [[0.0; 8]; 4];
    for i in 0..4 {
        rows[i].copy_from_slice(&out[(2 + 8 * i)..(2 + 8 * (i + 1))]);
        let r = rows[i];
        println!(
            "  {:<23}: Δ_∞={:.4}, S(8)={:.3}→S(16)={:.3} (ΔS={:+.3}), vA={:.3}, vB={:.3}, mismatch={:.3}, tr_AB={:.4}",
            names[i], r[0], r[1], r[2], r[7], r[3], r[4], r[5], r[6]
        );
    }
    // 0. Control: sin hueco (Δ_∞ < 0,12, S crece con N_r), velocidades distintas (mismatch > 0,40), transferencia EXACTA = 0
    assert!(rows[0][0] < 0.12 && rows[0][7] > 0.06, "control debe ser crítico sin hueco: Δ_∞={}, ΔS={}", rows[0][0], rows[0][7]);
    assert!(rows[0][5] > 0.40, "control debe preservar v_A ≠ v_B: mismatch = {}", rows[0][5]);
    assert!(rows[0][6] < 1e-10, "control sin acoplamiento tiene transferencia EXACTA = 0: {}", rows[0][6]);

    // 1A. Densidad-densidad pura (U(1)_A × U(1)_B conservado): sigue sin hueco, transferencia EXACTA = 0 y NO iguala velocidades
    assert!(rows[1][0] < 0.12 && rows[1][7] > 0.05, "canal 1A debe seguir crítico sin hueco: Δ_∞={}, ΔS={}", rows[1][0], rows[1][7]);
    assert!(rows[1][6] < 1e-10, "canal 1A conserva U(1)_A × U(1)_B (transferencia EXACTA = 0): {}", rows[1][6]);
    assert!(rows[1][5] > 0.35, "canal 1A NO iguala las velocidades de dos especies conservadas: {}", rows[1][5]);

    // 1B. Intercambio isótropo de Heisenberg: abre un hueco de masa real (Δ_∞ > 0,35) y S(N_r) satura (ley de área, ΔS ≈ 0)
    assert!(rows[2][0] > 0.35, "canal 1B debe abrir hueco de masa real (Δ_∞ > 0,35): {}", rows[2][0]);
    assert!(rows[2][7].abs() < 0.02, "canal 1B obedece ley de área (S(16) − S(8) ≈ 0): {}", rows[2][7]);
    assert!(rows[2][2] < rows[0][2] - 0.5, "S(16) en fase con hueco cae muy por debajo del control: {} vs {}", rows[2][2], rows[0][2]);

    // 1C. Mezcla de sabor sin hueco: permanece sin hueco (Δ_∞ < 0,10, S crece), mezcla especies Y colapsa la diferencia de velocidades!
    assert!(rows[3][0] < 0.10 && rows[3][7] > 0.03, "canal 1C debe permanecer sin hueco: Δ_∞={}, ΔS={}", rows[3][0], rows[3][7]);
    assert!(rows[3][6] > 0.40, "canal 1C mezcla activamente las dos especies: max|ΔN_B| = {}", rows[3][6]);
    assert!(rows[3][5] < 0.15, "canal 1C debe colapsar la diferencia de velocidades infrarrojas (mismatch < 0,15): {} vs {}", rows[3][5], rows[0][5]);
}

#[test]
fn t184_study2_3d_qca_triple_classification() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.study2_3d_qca_triple_classification(18);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let names = [
        "0: SC-Weyl-Fijo (2c)",
        "1: SC-Weyl-Pal (2c)",
        "2: BCC-Weyl-Fijo (2c)",
        "3: BCC-Weyl-Pal (2c)",
        "4: FCC-Weyl-Pal (2c)",
        "5: SC-Foton-MB (6c)",
        "6: SC-Wilson-Dirac (4c)",
        "7: Iso-SC+FCC-WD n=4 (4c)",
    ];
    println!("t184 Study 2 (8 familias QCA 3+1D):");
    let mut rows = [[0.0; 6]; 8];
    for i in 0..8 {
        rows[i].copy_from_slice(&out[(1 + 6 * i)..(1 + 6 * (i + 1))]);
        let r = rows[i];
        println!(
            "  {:<26}: asym={:.2e}, gap={:.4}, A2_spr={:.2e}, A2_mean={:+.2e}, A4_max={:.4}, pass3={:.0}",
            names[i], r[0], r[1], r[2], r[3], r[4], r[5]
        );
    }
    // Ninguna caminata local de 2 componentes (0..4) pasa la triple restricción (todas tienen dobladores o CPT-impar)
    for f in 0..5 {
        assert_eq!(rows[f][5], 0.0, "familia 2c {} no puede pasar las 3 restricciones (Nielsen–Ninomiya)", names[f]);
        assert!(rows[f][1] < 0.05, "familia 2c {} debe tener dobladores casi nulos: gap = {}", names[f], rows[f][1]);
    }
    // Orden fijo (0, 2, 5) rompen ω(k) = ω(−k)
    assert!(rows[0][0] > 1e-3 && rows[2][0] > 1e-3 && rows[5][0] > 1e-3, "reglas de orden fijo deben ser CPT-impares");
    // SC Wilson–Dirac (6) pasa (i) y (ii) pero falla (iii) por anisotropía cúbica en O(k²)
    assert!(rows[6][0] < 1e-12 && rows[6][1] > 0.30 && rows[6][2] > 0.02, "SC Wilson–Dirac tiene hueco pero anisotropía cúbica O(k²)");
    // NUEVA CONSTRUCCIÓN (7): Iso-SC+FCC Wilson–Dirac (r = 2/√3, dt = 0,25) PASA LAS TRES Y CANCELA A₂ ENTERO (orden n = 4 exacto!)
    assert_eq!(rows[7][5], 1.0, "Iso-SC+FCC Wilson–Dirac (r=2/√3) debe pasar la triple restricción");
    assert!(rows[7][0] < 1e-12, "asimetría CPT exacta = 0: {}", rows[7][0]);
    assert!(rows[7][1] > 0.30, "sin dobladores fuera de |k| < 0,35: gap = {}", rows[7][1]);
    assert!(rows[7][2] < 1e-4, "anisotropía direccional A₂ cancelada (< 10⁻⁴): {}", rows[7][2]);
    assert!(rows[7][3].abs() < 2e-3, "incluso el término isótropo A₂ medio se anula con r = 2/√3 (orden n = 4!): {}", rows[7][3]);
}

#[test]
fn t185_study3_causal_set_matter_backreaction() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.study3_causal_set_matter_backreaction(16, 300, 4);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    println!(
        "t185 Study 3 Johnston G_R: rms_err={:.4} vs ½J₀(mτ); σ_G(cono)={:.4}, σ_G(bulk)={:.4}",
        out[0], out[1], out[2]
    );
    println!(
        "t185 Study 3 Backreaction (N=16, 4 seeds): β=0 dMM(Nf=0→2)=({:.3}→{:.3}); β=1,9 S/N=({:.2}±{:.2} → {:.2}±{:.2}); β=-0,25 S/N=({:.2} → {:.2})",
        out[3], out[4], out[5], out[9], out[6], out[10], out[7], out[8]
    );
    // Parte A: el propagador retardado discreto de Johnston reproduce ½ J₀(m τ) con error RMS pequeño
    assert!(out[0] < 0.06, "⟨G_R(τ)⟩ de Johnston debe coincidir con ½ J₀(m τ): rms_err = {}", out[0]);
    // Firma no local de la discretización causal: fluctuaciones intrínsecas σ_G > 0,05 en el propagador
    assert!(out[1] > 0.02 && out[2] > 0.05, "varianza no local intrínseca medible en G_R: cono={}, bulk={}", out[1], out[2]);
    // Parte B: en β = 0 sin materia (N_f = 0) recupera d_MM ≈ 2; al encender N_f = 2 campos cuánticos
    // el término −N_f S_mat desplaza cuantitativamente el diagrama de fases (empuja hacia menor densidad causal / mayor d_MM o menor S_mat)!
    assert!((out[3] - 2.0).abs() < 0.20, "vacío puro β = 0 da d_MM ≈ 2: {}", out[3]);
    assert!((out[4] - out[3]).abs() > 0.15 || (out[6] - out[5]).abs() > 0.4, "la retroacción de N_f = 2 campos cuánticos debe producir un desplazamiento medible en el diagrama de fases");
}
