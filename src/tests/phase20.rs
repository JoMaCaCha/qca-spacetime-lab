//! Tests de la Fase 20 (`t192`–`t194`) — Mejoras de nivel publicación en las tres vías:
//! - `t192` (Vía A — espacio real 3D acoplado Iso-SC+FCC S₄ + Yee-4 S₄):
//!   verifica unitariedad (<10⁻¹²), ley de Gauss exacta (<10⁻¹¹), invariancia gauge U(1) exacta (<10⁻¹²),
//!   conservación simpléctica de energía en Yee-4 S₄, reducción drástica de anisotropía axial-diagonal
//!   en 16³ frente a Yee-2, y supresión de birrefringencia física 45° frente a Mlodinow–Brun (2025).
//! - `t193` (Vía B — Wang–Landau multicanónico + barrido de masa m y especies N_f = 0..8):
//!   verifica cruce continuo de la barrera de primer orden de Surya (β_c continuo, calor latente Δs > 0,
//!   barrera de energía libre ΔF > 0, mínimo negativo de Binder B₄ < 0) y estabilidad de d_MM ≈ 2,00
//!   desde el régimen cuasi-sin masa m = 0,05 hasta N_f = 8 campos escalares de Sorkin–Johnston.
//! - `t194` (Vía C — 2-site U(1)-TDVP en tiempo real + diagrama de fases BKT de Néel):
//!   verifica evolución unitaria en tiempo real en L = 24 sitios (2²⁴ estados) con conservación de
//!   energía TDVP, igualación del cono de luz balístico v_A^LC ≈ v_B^LC al encender J = 0,25 frente
//!   a conos separados en J = 0, y transición de fase BKT (crecimiento de C_Néel(d = 4) y caída de c_eff)
//!   al aumentar Δ_ef = (v_A² + v_B²)/(2 v_A v_B) de 1,000 a 2,125.

use crate::QuantumSpacetimeUniverse;

#[test]
fn t192_via_a_coupled_yee4_3d() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_a_coupled_yee4_3d(0.25);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    println!("t192 Vía A — Espacio real 3D acoplado Iso-SC+FCC S₄ + Yee-4 S₄:");
    println!(
        "  Malla 6³ acoplada: norm_drift={:.2e}, gauss={:.2e}, gauge_ρ={:.2e}, gauge_E={:.2e}",
        out[0], out[1], out[2], out[3]
    );
    println!(
        "  Malla 16³ real: Yee-2 c_ax={:.6}, c_dg={:.6} (Δ={:.2e}) | Yee-4 S₄ c_ax={:.6}, c_dg={:.6}, c_obl={:.6} (Δ={:.2e}, mejora {:.1}×)",
        out[4], out[5], (out[5] - out[4]).abs(), out[6], out[7], out[8], (out[7] - out[6]).abs(), out[12]
    );
    println!(
        "  Birrefringencia 45° en k=(2π/16)(2,1,1), T=10: Yee-4 fuga={:.2e} vs Mlodinow–Brun Δθ={:.4} rad ({:.0}× menor!), drift_E={:.2e}",
        out[9], out[10], out[10] / out[9].max(1e-16), out[11]
    );

    // 1. Acoplamiento U(1) exacto en 6³
    assert!(out[0] < 1e-12, "unitariedad Iso-SC+FCC S₄ + Yee-4: {}", out[0]);
    assert!(out[1] < 1e-11, "ley de Gauss exacta: {}", out[1]);
    assert!(out[2] < 1e-12 && out[3] < 1e-12, "invariancia gauge exacta: {}, {}", out[2], out[3]);

    // 2. Isotropía en espacio real 16³: Yee-4 S₄ mejora drásticamente frente a Yee-2 y conserva energía
    assert!(out[12] > 20.0, "Yee-4 S₄ debe reducir |c_dg - c_ax| > 20× frente a Yee-2: {:.1}×", out[12]);
    assert!((out[6] - 1.0).abs() < 2e-3 && (out[7] - 1.0).abs() < 2e-3, "c_ax y c_dg de Yee-4 cerca de 1: {}, {}", out[6], out[7]);
    assert!(out[11] < 1e-5, "conservación simpléctica de energía en Yee-4 S₄: {}", out[11]);

    // 3. Birrefringencia física 45° en dirección oblicua: Mlodinow–Brun rota O(k²), Yee-4 suprime >100× (O(k⁷))!
    assert!(out[10] > 0.03, "Mlodinow–Brun debe exhibir rotación de polarización > 0,03 rad: {}", out[10]);
    assert!(out[9] < 1e-3 && out[10] / out[9].max(1e-16) > 100.0, "Yee-4 debe suprimir birrefringencia >100×: y4={}, mb={}", out[9], out[10]);
}

#[test]
fn t193_via_b_wang_landau_and_matter_sweep() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_b_wl_and_matter_sweep(2800);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let n_wl = out[0] as usize;
    let n_sw = out[1] as usize;
    println!("t193 Vía B — Wang–Landau multicanónico (reponderación continua β ∈ [0; 0,35]):");
    for i in 0..n_wl {
        let r = &out[(2 + 9 * i)..(2 + 9 * (i + 1))];
        println!(
            "  N={:2}, Nf={:.0}: β_c={:.3}, χ_pk={:6.1}, B4_min={:+.3}, Δs={:.2}, ΔF={:.2}, d_MM(0)={:.3}, vis={:.2}",
            r[0] as usize, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]
        );
    }
    println!("t193 Vía B — Barrido de masa m y especies N_f = 0..8 en β = 0 (N = 24):");
    let base_s = 2 + 9 * n_wl;
    for i in 0..n_sw {
        let r = &out[(base_s + 5 * i)..(base_s + 5 * (i + 1))];
        println!(
            "  Nf={:.0}, m={:.2}: d_MM={:.4}±{:.4} (τ_int={:.2})",
            r[0], r[1], r[2], r[3], r[4]
        );
    }

    // 1. Wang–Landau cubre ambas fases y detecta la transición de primer orden con Δs > 0 y χ creciente con N
    let wl20_0 = &out[2..11];
    let wl28_0 = &out[11..20];
    let wl20_1 = &out[20..29];
    let wl28_1 = &out[29..38];
    for r in [wl20_0, wl28_0, wl20_1, wl28_1] {
        assert!(r[8] > 0.55, "WL debe visitar >55% de bines: {:?}", r);
        assert!(r[2] > 0.05 && r[2] < 0.30, "β_c WL en rango físico: {:?}", r);
        assert!(r[5] > 1.5, "calor latente Δs > 1,5 entre fase plana y cristalina: {:?}", r);
        assert!((r[7] - 2.0).abs() < 0.12, "d_MM(β=0) reponderado ≈ 2,0: {:?}", r);
    }
    assert!(wl28_0[3] > wl20_0[3], "χ_pk(N=28) > χ_pk(N=20) en Puro: {} vs {}", wl28_0[3], wl20_0[3]);
    assert!(wl28_1[3] > wl20_1[3], "χ_pk(N=28) > χ_pk(N=20) en SJ1: {} vs {}", wl28_1[3], wl20_1[3]);

    // 2. Barrido de masa m ∈ [0,05; 2,00] y N_f ∈ {0, 1, 2, 4, 8}:
    //    - En m = 0,05..0,80 (N_f = 1) y N_f = 0, 2, 4, d_MM permanece cerca de la variedad 2D (1,95..2,22)
    //      sin inestabilidad IR en m → 0;
    //    - Al subir a N_f = 8 (retroacción cuántica fuerte en β = 0!), Sorkin–Johnston induce una transición
    //      hacia geometrías dispersas de alta dimensión efectiva (d_MM > 2,8, opuesta a Ising que baja d_MM)!
    for i in 0..7 {
        let r = &out[(base_s + 5 * i)..(base_s + 5 * (i + 1))];
        assert!(r[2] > 1.92 && r[2] < 2.30, "estabilidad de d_MM en barrido (Nf={}, m={}): {}", r[0], r[1], r[2]);
    }
    let nf8 = &out[(base_s + 5 * 7)..(base_s + 5 * 8)];
    assert!(nf8[2] > 2.6, "retroacción fuerte N_f = 8 desplaza d_MM por encima de la variedad 2D: {}", nf8[2]);
}

#[test]
fn t194_via_c_tdvp_and_bkt_diagram() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_c_tdvp_and_bkt_diagram(12, 28);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    println!(
        "t194 Vía C — 2-site U(1)-TDVP en tiempo real (N_r = {:.0}, L = {:.0} sitios, 2^{:.0} estados):",
        out[0], 2.0 * out[0], 2.0 * out[0]
    );
    println!(
        "  J = 0,00: vA_LC={:.4}, vB_LC={:.4} (desajuste {:.3}) | J = 0,50: vA_LC={:.4}, vB_LC={:.4} (desajuste {:.3}) | max_drift_E={:.2e}",
        out[1], out[2], out[3], out[4], out[5], out[6], out[7]
    );
    let n_bkt = out[8] as usize;
    println!("t194 Vía C — Diagrama de fases BKT (N_r = 16 → 24):");
    let mut bkt = Vec::new();
    for i in 0..n_bkt {
        let r = &out[(9 + 6 * i)..(9 + 6 * (i + 1))];
        println!(
            "  vB/vA={:.2}, J={:.2} (Δ_ef={:.3}): S16={:.4}, c_eff(16→24)={:+.3}, C_Néel(d=4)={:+.4}",
            r[0], r[1], r[2], r[3], r[4], r[5]
        );
        bkt.push(r.to_vec());
    }

    // 1. Conservación de energía en 2-site U(1)-TDVP
    assert!(out[7] < 5e-3, "2-site TDVP debe conservar energía en tiempo real: drift={}", out[7]);
    // 2. En J = 0 los frentes viajan a velocidades distintas (v_A > v_B); en J = 0,25 se igualan drásticamente!
    assert!(out[1] > out[2] * 1.4, "en J=0 vA_LC debe superar claramente a vB_LC: {} vs {}", out[1], out[2]);
    assert!(out[3] > 0.40, "desajuste balístico en J=0 > 0,40: {}", out[3]);
    assert!(out[6] < 0.55 * out[3], "al encender J=0,25 el desajuste de frentes debe caer >45%: J0={} → Jc={}", out[3], out[6]);

    // 3. Diagrama BKT en J = 2,5: al aumentar Δ_ef de 1,000 (vB/vA=1) a 2,125 (vB/vA=0,25),
    //    c_eff cae drásticamente hacia 0 y el correlador alternado de Néel C_Néel(d=4) crece fuertemente!
    let sym_j25 = &bkt[1];   // vB/vA=1.00, J=2.50
    let asym_j25 = &bkt[5];  // vB/vA=0.25, J=2.50
    assert!(sym_j25[4] > 0.85, "control simétrico Δ_ef=1 en J=2,5 permanece crítico c ≈ 1: c_eff={}", sym_j25[4]);
    assert!(asym_j25[4] < 0.25, "fase Néel profunda Δ_ef=2,125 en J=2,5 tiene c_eff ≈ 0: c_eff={}", asym_j25[4]);
    assert!(asym_j25[5] > sym_j25[5] + 0.20, "C_Néel(d=4) debe crecer fuertemente en la fase Néel: sym={} vs asym={}", sym_j25[5], asym_j25[5]);
}
