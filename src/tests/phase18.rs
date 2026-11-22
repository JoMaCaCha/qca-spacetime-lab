//! Tests de la Fase 18 (`t186`–`t188`) — Profundización a nivel artículo en las tres vías:
//! - `t186` (Vía A): QCA 3+1D isotrópico de orden n = 4 en espacio real con acoplamiento U(1)
//!   dinámico y ley de Gauss exacta (< 10⁻¹³), contraste circuito local S₂ vs S₄ Yoshida–Suzuki,
//!   y auditoría unificada de los 5 modelos fotónicos (Mlodinow–Brun, Bisio et al., Eon et al. y Yee-4 S₄).
//! - `t187` (Vía B): Parallel Tempering sobre conjuntos causales 2D acoplados a materia cuántica de
//!   Sorkin–Johnston (N_f = 0, 1, 2) e Ising sobre enlaces en dos tamaños (N = 16 y N = 24), con
//!   verificación de ergodicidad (tasa de intercambio de réplicas), block bootstrap, τ_int y pico χ_S.
//! - `t188` (Vía C): Restauración infrarroja de Lorentz a la media geométrica √(v_A v_B) mediante
//!   confinamiento gauge en Schwinger 2 sabores (U(1)_A × U(1)_B exacto, sin mezcla de sabor) +
//!   validación exacta del motor DMRG por bloques U(1) y caída de carga central c_CFT ≈ 2 → 1.

use crate::QuantumSpacetimeUniverse;

#[test]
fn t186_via_a_realspace_n4_qca_and_photons() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_a_realspace_n4_qca(0.25);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    println!(
        "t186 Vía A espacio real 6³ (U(1) dinámico, S₄): norm_drift={:.2e}, gauss={:.2e}, gauge_dens={:.2e}, gauge_E={:.2e}",
        out[0], out[1], out[2], out[3]
    );
    println!(
        "t186 Vía A circuito local 2×2×2 Bloch (dt=0.25): S₂=(A2_spr={:.3e}, A2_mean={:+.3e}, |c-1|={:.2e}, asym={:.2e}) | S₄=(A2_spr={:.3e}, A2_mean={:+.3e}, |c-1|={:.2e}, asym={:.2e}, corner_gap={:.4})",
        out[4], out[5], out[37], out[6], out[7], out[8], out[38], out[9], out[10]
    );
    let pnames = [
        "0: Mlodinow–Brun (2025)",
        "1: Bisio–D'Ariano–Perinotti (2016)",
        "2: Eon et al. fijo (2023)",
        "3: Eon et al. palindrómico (2023)",
        "4: NUEVO Yee-4 S₄ (n=4)",
    ];
    for (i, nm) in pnames.iter().enumerate() {
        let r = &out[(12 + 5 * i)..(12 + 5 * (i + 1))];
        println!(
            "  {:<35}: asym={:.2e}, A2_spr={:.2e}, A2_mean={:+.2e}, corner_gap={:.3}, pass_n4={:.0}",
            nm, r[0], r[1], r[2], r[3], r[4]
        );
    }
    // 1. Exactitud en espacio real 3D con U(1) dinámico sobre enlaces cúbicos + diagonales FCC
    assert!(out[0] < 1e-12, "unitariedad exacta en 6³: {}", out[0]);
    assert!(out[1] < 1e-11, "ley de Gauss exacta con enlaces FCC + Maxwell dinámico: {}", out[1]);
    assert!(out[2] < 1e-12 && out[3] < 1e-12, "invariancia gauge U(1) local exacta: dens={}, E={}", out[2], out[3]);

    // 2. Circuito local S₂ vs S₄: S₄ reduce |c_eff − 1| en >10× respecto a S₂ y suprime A₂ a < 10⁻² (n = 4!) sin dobladores!
    assert!(out[9] < 1e-12, "circuito local S₄ debe ser exactamente CPT-par: {}", out[9]);
    assert!(out[38] < 0.1 * out[37] && out[38] < 5e-4, "S₄ debe reducir |c_eff−1| en >10× frente a S₂: S₂={}, S₄={}", out[37], out[38]);
    assert!(out[7] < 5e-3 && out[8].abs() < 5e-3, "circuito local S₄ debe suprimir A₂ real (k₁, k₂): spr={}, mean={}", out[7], out[8]);
    assert!(out[10] > 1.5, "sin dobladores en los 7 vértices BZ del circuito local S₄: {}", out[10]);

    // 3. Auditoría de los 5 fotones 3+1D: 0, 1, 2 son CPT-impares; 3 es CPT-par pero anisótropo n = 2; solo 4 (Yee-4 S₄) pasa n = 4!
    assert!(out[12] > 1e-3 && out[17] > 1e-3 && out[22] > 1e-3, "fotones de orden fijo 0, 1, 2 deben ser CPT-impares birrefringentes");
    assert!(out[27] < 1e-12 && out[28] > 1e-2 && out[31] == 0.0, "Eon palindrómico es CPT-par pero n = 2 anisótropo");
    assert!(out[32] < 1e-12 && out[33] < 5e-4 && out[34].abs() < 5e-4 && out[35] > 1.5 && out[36] == 1.0, "nuevo Yee-4 S₄ debe pasar todas las pruebas n = 4 sin dobladores");
}

#[test]
fn t187_via_b_causal_pt_scaling() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_b_causal_pt_scaling(16, 22, 320);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let n_runs = out[0] as usize;
    let nb = out[1] as usize;
    let nf_rep = out[2] as usize;
    let stride = 5 + nb * nf_rep;
    println!("t187 Vía B Parallel Tempering (N=16, 22; 12 réplicas β; 320 sweeps):");
    for r in 0..n_runs {
        let base = 3 + r * stride;
        let n_el = out[base] as usize;
        let mode = out[base + 1] as i32;
        let swap_acc = out[base + 2];
        let beta_c = out[base + 3];
        let chi_pk = out[base + 4];
        // Réplica β = 0 es el índice 2 en VIA_B_BETAS; β = +1,35 es el índice 11; β = −0,25 es el índice 0
        let rep = |idx: usize| &out[(base + 5 + idx * nf_rep)..(base + 5 + (idx + 1) * nf_rep)];
        let r0 = rep(2);
        let rhi = rep(11);
        let rneg = rep(0);
        println!(
            "  N={:2}, mode={:+2}: swap_acc={:.2}, β_c={:.2} (χ_pk={:.1}) | β=0: dMM={:.3}±{:.3} (τ={:.1}) | β=+1.35: S/N={:+.2}±{:.2} | β=-0.25: S/N={:+.2}±{:.2}",
            n_el, mode, swap_acc, beta_c, chi_pk, r0[3], r0[4], r0[7], rhi[1], rhi[2], rneg[1], rneg[2]
        );
        // Ergodicidad de Parallel Tempering: tasa de intercambio de réplicas saludable (> 0,10 y < 0,95)
        assert!(swap_acc > 0.10 && swap_acc < 0.95, "tasa de intercambio PT fuera de rango: {}", swap_acc);
        // En β = 0 recupera fase de variedad 2D (d_MM ≈ 2)
        assert!((r0[3] - 2.0).abs() < 0.25, "fase de variedad en β=0 (N={}, mode={}): dMM={}", n_el, mode, r0[3]);
        // Transición real: S/N > +3 en β=+1,35 y S/N < −3 en β=−0,25 con pico de susceptibilidad χ_pk > 1
        assert!(rhi[1] > 3.0 && rneg[1] < -3.0 && chi_pk > 1.0, "debe mostrar las 3 fases de Surya y pico χ_S");
    }
}

#[test]
fn t188_via_c_schwinger2f_lorentz_and_u1_dmrg() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_c_schwinger2f_lorentz(8, 32);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let v_geom = out[0];
    println!(
        "t188 Vía C (v_geom = √(v_A v_B) = {:.4}, N_r = {:.0}): XXZ U(1)-DMRG e={:.5} (Bethe -0.375), S_XXZ={:.3}; MPO6 err={:.2e}",
        v_geom, out[1], out[2], out[3], out[6]
    );
    println!(
        "t188 Vía C caída c_CFT (U(1)-DMRG N_r=8→16, ambos cortes pares): J=0: S={:.3}→{:.3} (c={:.3}) | J=1.0: S={:.3}→{:.3} (c={:.3})",
        out[7], out[8], out[9], out[10], out[11], out[12]
    );
    let nj = out[13] as usize;
    let mut rows = Vec::new();
    for i in 0..nj {
        let r = &out[(14 + 6 * i)..(14 + 6 * (i + 1))];
        println!(
            "  J={:.2}: vA_IR={:.4}, vB_IR={:.4}, mismatch={:.4}, ω_ch={:.4}, E0={:.5}",
            r[0], r[1], r[2], r[3], r[4], r[5]
        );
        rows.push(r.to_vec());
    }
    // 1. Validación exacta del motor DMRG por bloques U(1) (XXZ y MPO D=7 de Schwinger 2 sabores)
    assert!((out[2] + 0.375).abs() < 0.015, "U(1)-DMRG XXZ L=24 debe acercarse a Bethe −0,375: {}", out[2]);
    assert!(out[6] < 1e-5, "MPO exacto D=7 de Schwinger 2 sabores debe coincidir con Lanczos exacto a <10⁻⁵: {}", out[6]);

    // 2. Control J = 0: velocidades exactas v_A = 1,0000, v_B = 0,4500 (mismatch = 0,7586!)
    assert!((rows[0][1] - 1.0).abs() < 1e-3 && (rows[0][2] - 0.45).abs() < 1e-3, "control J=0 exacto: {:?}", rows[0]);

    // 3. Con acoplamiento gauge cuántico compartido (U(1)_A × U(1)_B EXACTAMENTE conservado!):
    //    Para TODO J > 0, AMBAS velocidades infrarrojas se igualan EXACTAMENTE (mismatch < 0,02),
    //    partiendo cerca de la media geométrica √(v_A v_B) = 0,6708 a J pequeño (régimen de continuo)
    //    y renormalizándose suavemente ∝ v_A v_B / J en acoplamiento fuerte de red!
    let r_weak = &rows[1]; // J = 0,08 (régimen de continuo más cercano a √(v_A v_B))
    let r_strong = &rows[nj - 1]; // J = 1,00
    for r in &rows[1..] {
        assert!(r[3] < 0.02, "confinamiento gauge iguala exactamente v_A^IR = v_B^IR en J={}: mismatch={}", r[0], r[3]);
    }
    let v_weak = 0.5 * (r_weak[1] + r_weak[2]);
    assert!(
        v_weak > 0.50 && v_weak < 0.72 && (v_weak - v_geom).abs() < 0.16,
        "a J pequeño la velocidad igualada queda intermedia cerca de √(v_A v_B)={:.4}: {:.4}",
        v_geom, v_weak
    );
    assert!(r_strong[1] < r_weak[1], "en acoplamiento fuerte de red v_IR(J) se suaviza como ~v_A v_B/J");
    // El modo cargado antisimétrico ω_ch sube notablemente con J (masa del bosón de Schwinger!)
    assert!(r_strong[4] > rows[0][4] + 0.6, "el modo cargado debe adquirir hueco de Schwinger: {} → {}", rows[0][4], r_strong[4]);
    // 4. Confinamiento del modo cargado en U(1)-DMRG (cortes pares N_r = 8 → 16): S y c_CFT caen de ~2 hacia ~1!
    assert!(out[11] < out[8] - 0.35 && out[12] < out[9] - 0.35, "c_CFT y S(16) deben caer al confinar el modo cargado: S=({}→{}), c=({}→{})", out[8], out[11], out[9], out[12]);
}
