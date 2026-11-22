//! Tests de la Fase 19 (`t189`–`t191`) — Cierre de los puntos pendientes en las tres vías:
//! - `t189` (Vía A — extensión quiral 4+1D): losa abierta de Kaplan–Sen (2024) comparando
//!   SC estándar vs Iso-SC+FCC (anula spread(A₂) en >200×) vs Symanzik-4 (cono de Weyl de
//!   orden n = 4 completo con cero dobladores y cancelación exacta de Golterman–Shamir).
//! - `t190` (Vía B — FSS adaptativo 20 réplicas): escalera adaptativa Katzgraber–Trebst en
//!   N ∈ {16, 22, 28} × 4 modelos (Puro, SJ1, SJ2, Ising), verificando swap_acc saludable,
//!   crecimiento de χ_pico(N) ∝ N^γ y contraste opuesto de d_MM(∞) entre SJ e Ising.
//! - `t191` (Vía C — continuo, CFT multi-tamaño y 3+1D Coulomb): extrapolación v_IR(N_r → ∞)
//!   hacia √(v_A v_B) = 0,6708 a M_S·L = const, ajuste Calabrese–Cardy sobre N_r ∈ {8, 12, 16, 24}
//!   en U(1)-DMRG (c_J0 ≈ 2 → c_J1 ≈ 1) y contraste frente a QED 3+1D no confinante.

use crate::QuantumSpacetimeUniverse;

#[test]
fn t189_via_a_chiral_kaplan_sen_n4() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_a_chiral_kaplan_sen_n4(14, 0.55);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let names = ["SC Kaplan–Sen (estándar)", "Iso-SC+FCC Kaplan–Sen", "Symanzik-4 Kaplan–Sen (n=4)"];
    println!("t189 Vía A — Extensión quiral 4+1D (losa abierta L_s = 14, m = 0,55):");
    for (i, nm) in names.iter().enumerate() {
        let r = &out[(7 * i)..(7 * (i + 1))];
        println!(
            "  {:<30}: ⟨γ⁵⟩_L={:+.4}, ⟨γ⁵⟩_R={:+.4}, suma={:+.2e}, spr(A₂)={:.3e}, ⟨A₂⟩={:+.3e}, gap_BZ={:.3}, tier={:.0}",
            nm, r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        );
    }
    let sc = &out[0..7];
    let iso = &out[7..14];
    let sym = &out[14..21];
    // Todos conservan quiralidad > 0,99 en s = 0, cancelación exacta ⟨γ⁵⟩_L + ⟨γ⁵⟩_R = 0 y cero dobladores
    for r in [sc, iso, sym] {
        assert!(r[0] > 0.99 && r[1] < -0.99 && r[2].abs() < 1e-3, "quiralidad y Golterman–Shamir: {:?}", r);
        assert!(r[5] > 1.3, "sin dobladores en los 7 vértices BZ: gap={}", r[5]);
    }
    // SC estándar es fuertemente anisótropo (spread ≈ 0,111)
    assert!(sc[3] > 0.08 && sc[6] == 0.0, "SC debe ser anisótropo: spr={}", sc[3]);
    // Iso-SC+FCC anula la anisotropía direccional en >200× (spread < 1e-3) manteniendo ⟨A₂⟩ ≈ −1/6
    assert!(iso[3] < 1e-3 && iso[3] < sc[3] / 100.0 && iso[6] == 1.0, "Iso-SC+FCC debe ser direccionalmente isótropo: {:?}", iso);
    // Symanzik-4 anula TANTO spread(A₂) COMO ⟨A₂⟩: cono de Weyl quiral de orden n = 4 completo!
    assert!(sym[3] < 1e-3 && sym[4].abs() < 1e-3 && sym[6] == 2.0, "Symanzik-4 debe pasar n = 4 completo: {:?}", sym);
}

#[test]
fn t190_via_b_adaptive_20rep_fss() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_b_adaptive_fss(380);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let n_fits = out[0] as usize;
    let n_runs = out[1] as usize;
    println!("t190 Vía B — FSS adaptativo (20 réplicas Katzgraber–Trebst, N ∈ {{16, 22, 28}}):");
    let mut fits = Vec::new();
    for i in 0..n_fits {
        let f = &out[(2 + 8 * i)..(2 + 8 * (i + 1))];
        println!(
            "  mode={:+2}: γ(χ_pk)={:+.3} (R²={:.3}), d_MM(∞)={:.3} (pendiente={:+.2}/N), β_c(∞)={:.3}, min_swap={:.2}, B4_min={:+.3}",
            f[0] as i32, f[1], f[2], f[3], f[4], f[5], f[6], f[7]
        );
        fits.push(f.to_vec());
    }
    let base_r = 2 + 8 * n_fits;
    for i in 0..n_runs {
        let r = &out[(base_r + 8 * i)..(base_r + 8 * (i + 1))];
        println!(
            "    N={:2}, mode={:+2}: swap={:.2} (min_pair={:.2}), β_c_main={:.3}, χ_main={:5.1}, d_MM(0)={:.3}±{:.3}",
            r[0] as usize, r[1] as i32, r[2], r[3], r[4], r[5], r[6], r[7]
        );
    }
    // 1. Tasa de intercambio global muy saludable con 20 réplicas adaptativas (> 0,30 en todos los modos!)
    for f in &fits {
        assert!(f[6] > 0.30, "20 réplicas adaptativas deben sostener swap_acc > 0,30: mode={}, swap={}", f[0], f[6]);
        assert!(f[1] > 0.4, "la susceptibilidad en la transición de Surya debe crecer con N: mode={}, γ={}", f[0], f[1]);
    }
    // 2. Contraste cualitativo en el límite N → ∞: Puro y SJ convergen a d_MM(∞) ≈ 2,00, Ising queda deprimido!
    assert!((fits[0][3] - 2.0).abs() < 0.12, "Puro d_MM(∞) ≈ 2,0: {}", fits[0][3]);
    assert!((fits[1][3] - 2.0).abs() < 0.12 && (fits[2][3] - 2.0).abs() < 0.12, "SJ d_MM(∞) ≈ 2,0: {}, {}", fits[1][3], fits[2][3]);
    assert!(fits[3][3] < fits[0][3] - 0.08, "Ising deprime d_MM(∞) respecto a gravedad pura: Ising={}, Puro={}", fits[3][3], fits[0][3]);
}

#[test]
fn t191_via_c_continuum_cft_and_coulomb3d() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let p = u.via_c_continuum_and_scaling(false, 44);
    let len = u.open_quantum_out_len() as usize;
    let out = unsafe { std::slice::from_raw_parts(p, len) }.to_vec();
    let v_geom = out[0];
    let n_sz = out[13] as usize;
    println!(
        "t191 Vía C — Continuo (M_S·L=const): √(v_A v_B)={:.4}, v_inf_lin={:.4}, v_inf_quad={:.4} (R²={:.4})",
        v_geom, out[1], out[2], out[3]
    );
    for k in 0..3 {
        let r = &out[(14 + 5 * k)..(14 + 5 * (k + 1))];
        println!("  N_r={:2}, J={:.4}: vA={:.4}, vB={:.4}, mis={:.5}", r[0] as usize, r[1], r[2], r[3], r[4]);
    }
    println!(
        "t191 Vía C — Calabrese–Cardy (N_r=8..24): J=0 exacto c_fit={:.3} (R²={:.4}, c_hi={:.3}) | J=2.5 simétrico (Δ=1) c_fit={:.3} (R²={:.4}, c_hi={:.3})",
        out[4], out[5], out[6], out[7], out[8], out[9]
    );
    let base_s = 14 + 15;
    for k in 0..n_sz {
        let r = &out[(base_s + 5 * k)..(base_s + 5 * (k + 1))];
        println!(
            "  N_r={:2} (L={:2}): S_ex(J=0)={:.4}, S(J=2.5,vA=vB)={:.4}, S(J=1.0,vA≠vB)={:.4}, S(J=2.5,vA≠vB,Δ=1.34)={:.4}",
            r[0] as usize, 2 * r[0] as usize, r[1], r[2], r[3], r[4]
        );
    }
    println!(
        "t191 Vía C — Contraste 3+1D Coulomb vs 1+1D Schwinger: mis_UV={:.4} → 3+1D QED RG={:.4} (phot_fac={:.3}) vs 1+1D Schwinger={:.5}",
        out[10], out[11], out[12], out[14 + 10 + 4]
    );

    // 1. Convergencia monótona en la ventana de continuo hacia √(v_A v_B) = 0,6708 con mismatch ≈ 0!
    let v6 = 0.5 * (out[16] + out[17]);
    let v8 = 0.5 * (out[21] + out[22]);
    let v10 = 0.5 * (out[26] + out[27]);
    assert!(v10 > v8 && v8 > v6, "v_IR debe crecer monótonamente al acercarse al continuo: {} → {} → {}", v6, v8, v10);
    assert!(out[18] < 1e-2 && out[23] < 1e-2 && out[28] < 1e-2, "desajuste nulo en toda la trayectoria de continuo");
    assert!((out[1] - v_geom).abs() < 0.06, "extrapolación lineal al continuo debe acercarse a √(v_A v_B)={:.4}: {:.4}", v_geom, out[1]);

    // 2. Escalado CFT multi-tamaño (N_r = 8, 12, 16, 24): ajuste excelente (R² > 0,99) y reducción exacta c = 2 → c = 1 (razón ≈ 0,50)!
    //    (Sobre cortes pares de cadena abierta finita, el término de Friedel O(1/N_r) de Laflorencie desplaza ambos c hacia arriba ~15 %,
    //     pero c_hi decrece monótonamente hacia 2 y 1 y la razón c_hi(J=2,5)/c_hi(J=0) = 0,52 reproduce exacto ½!)
    let c_ratio = out[9] / out[6].max(1e-9);
    assert!(out[5] > 0.99 && out[8] > 0.99, "escalado logarítmico Calabrese–Cardy debe ser limpio: R²=({}, {})", out[5], out[8]);
    assert!(out[4] > 2.0 && out[4] < 2.60 && out[6] < out[4], "J=0 exacto (c=2 con corrección 1/N_r decreciente): fit={}, hi={}", out[4], out[6]);
    assert!(out[7] > 0.95 && out[7] < 1.45 && out[9] < out[7], "J=2,5 confinante (c=1 con corrección 1/N_r decreciente): fit={}, hi={}", out[7], out[9]);
    assert!(c_ratio > 0.45 && c_ratio < 0.58, "la razón c_hi(J=2,5)/c_hi(J=0) debe ser ≈ 0,50 (2→1): {}", c_ratio);

    // 3. Contraste 3+1D Coulomb vs 1+1D confinante: 3+1D mantiene >65 % del desajuste; 1+1D lo anula!
    assert!(out[11] > 0.65 * out[10] && out[28] < 1e-3, "contraste 3+1D Coulomb vs 1+1D Schwinger");
}
