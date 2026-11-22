//! Tests de la fase 15 (171–176): QFT de muchos cuerpos 1+1D en MPS (modelo de Schwinger
//! cuántico con confinamiento y rotura de cuerda en tiempo real), violación de Lorentz radiativa a
//! un lazo (Collins et al., 2004) + renormalización de velocidad en MPS XXZ, espacio-tiempo
//! discreto covariante (conjuntos causales + d'Alembertiano de Benincasa–Dowker + grafo dinámico),
//! obstrucción de dobladores en caminatas 3D de tipo luz (§8.6), aprendizaje del mapa discreto de
//! Floquet (§8.10) y fermiones CPT-impares con umbral de Cohen–Glashow (§8.11).

use crate::QuantumSpacetimeUniverse;

#[test]
fn t171_mps_schwinger_confinement_and_string_breaking() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    // 1. Confinamiento lineal en el régimen de masa moderada (m = 0,6, J = 0,5, w = 0,5):
    //    V(d) = E0(d) − E0(0) crece linealmente con d mientras la energía de la cuerda < 2m_eff
    let conf = u.mps_schwinger_confinement(12, 0.5, 0.6, 0.5, 16);
    println!(
        "t171 conf: E0={:.4}, V(2)={:.4}, V(4)={:.4}, V(6)={:.4}, S0={:.4}, S4={:.4}, Ec(4)={:.4}",
        conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6]
    );
    assert!(conf[1] > 0.3, "V(2) debe ser positivo (confinamiento): {}", conf[1]);
    assert!(conf[2] > conf[1] + 0.3, "V(4) > V(2): {} vs {}", conf[2], conf[1]);
    assert!(conf[3] > conf[2] + 0.2, "V(6) > V(4): {} vs {}", conf[3], conf[2]);
    // El campo eléctrico en el centro de la cuerda sin romper sigue cerca de +1
    assert!(conf[6] > 0.5, "cuerda confinante mantiene flujo eléctrico: E_c(4) = {}", conf[6]);

    // 2. Rotura cuántica de la cuerda en tiempo real (m pequeña = 0,15, J grande = 0,9, d = 6):
    //    el flujo eléctrico central cae drásticamente al crearse pares e⁺e⁻ y crece S_vN con energía conservada
    let ptr = u.mps_schwinger_string_breaking(12, 0.8, 0.15, 0.9, 6, 16, 0.08, 45);
    let len = u.open_quantum_out_len() as usize;
    let sb = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    println!(
        "t171 sb: E_c {:.3} → min {:.3} → fin {:.3}; S_vN max={:.4}; E_tot {:.4} → {:.4}",
        sb[0], sb[1], sb[2], sb[3], sb[5], sb[6]
    );
    assert!((sb[0] - 1.0).abs() < 1e-6, "campo inicial de la cuerda = +1: {}", sb[0]);
    assert!(sb[1] < 0.35, "rotura cuántica de cuerda debe apantallar E_centro: min = {}", sb[1]);
    assert!(sb[3] > 0.35, "la creación de pares debe generar entrelazamiento cuántico: S_vN max = {}", sb[3]);
    let de_rel = ((sb[6] - sb[5]) / sb[5].abs().max(1.0)).abs();
    assert!(de_rel < 0.05, "energía total conservada en TEBD: {} → {} (rel={})", sb[5], sb[6], de_rel);
}

#[test]
fn t172_collins_one_loop_lv_and_mps_velocity() {
    let u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    // 1. Collins et al. (PRL 2004): un lazo Yukawa en 1+1D
    let lv = u.collins_one_loop_lv(0.25, 1.2, 0.5, 1.0);
    println!(
        "t172 Collins: dc2_cont(120)={:.3e}, dc2_cont(20)={:.3e}, dc2_lat(r=1)={:.4e}, dc2_lat(r=0)={:.4e}, d_c={:.4e}",
        lv[0], lv[1], lv[2], lv[3], lv[4]
    );
    // En el continuo, δc² → 0 conforme sube Λ (control nulo covariante de Lorentz)
    assert!(lv[0].abs() < 5e-4, "regulador continuo debe dar δc² → 0: {}", lv[0]);
    assert!(lv[0].abs() < lv[1].abs(), "al subir Λ de 20 a 120, |δc²_cont| debe decrecer: {} vs {}", lv[0], lv[1]);
    // En la red discreta, los modos UV generan δc² = O(g_Y²) ≠ 0 de dimensión 4 (sin supresión E/E_P!)
    assert!(lv[2].abs() > 5e-3, "regulador de red debe generar δc² O(g²) ≠ 0: {}", lv[2]);
    assert!(lv[2].abs() > 20.0 * lv[0].abs(), "red debe superar al control continuo por >20×: {} vs {}", lv[2], lv[0]);
    // Escala cuadrática en g_Y: doblar g_Y cuadruplica δc²_red
    let lv2 = u.collins_one_loop_lv(0.25, 1.2, 1.0, 1.0);
    let ratio = lv2[2] / lv[2];
    assert!((ratio - 4.0).abs() < 1e-6, "δc² a un lazo debe escalar exactamente como g_Y²: ratio = {}", ratio);

    // 2. Velocidad del cono de luz en cadena interactuante XXZ en MPS (Δ = 0 libre vs Δ = 0,22 repulsivo)
    let v0 = u.mps_interacting_velocity(16, 0.5, 0.0, 16, 0.08, 30);
    let v1 = u.mps_interacting_velocity(16, 0.5, 0.22, 16, 0.08, 30);
    println!(
        "t172 MPS XXZ: Δ=0 v={:.4} (Bethe {:.4}, S={:.3}); Δ=0,22 v={:.4} (Bethe {:.4}, S={:.3})",
        v0[0], v0[1], v0[2], v1[0], v1[1], v1[2]
    );
    assert!((v0[1] - 1.0).abs() < 1e-4, "Bethe libre = 2w = 1,0: {}", v0[1]);
    assert!(v1[1] > 1.45, "Bethe repulsivo (Δ_XXZ=0,88) > 1,45: {}", v1[1]);
    assert!(v1[0] > v0[0] * 1.15, "la interacción repulsiva en MPS debe acelerar el cono de luz: {} vs {}", v1[0], v0[0]);
}

#[test]
fn t173_causal_set_lorentz_and_dynamic_graph() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    // 1. Conjunto causal 1+1D bajo boost fuerte η = 0,8 (factor de distorsión cúbica e^{1,6} ≈ 4,95×)
    let ptr = u.causal_set_lorentz_test(200, 0.8, 24, 7);
    let cs = unsafe { std::slice::from_raw_parts(ptr, 7) }.to_vec();
    println!(
        "t173 causal set: d_MM={:.4}, diff_links={:.0}, deg={:.2}, BD_rest={:.3}, BD_boost={:.3}, cubic_dist={:.2}×",
        cs[0], cs[1], cs[2], cs[3], cs[4], cs[5]
    );
    assert_eq!(cs[1], 0.0, "el grafo causal debe ser EXACTAMENTE invariante bajo boost de Lorentz");
    assert!((cs[0] - 2.0).abs() < 0.10, "dimensión de Myrheim–Meyer debe recuperar d = 2: {}", cs[0]);
    assert!(cs[2] > 3.5, "precio en no localidad: grado medio de enlaces > 3,5 (crece ~ ln N): {}", cs[2]);
    assert!((cs[3] - cs[4]).abs() < 1e-10, "□_BD debe ser exactamente igual en reposo y tras boost: {} vs {}", cs[3], cs[4]);
    assert!((cs[3] - 2.0).abs() < 0.6, "⟨□_BD φ⟩ debe aproximar +2,0 sobre φ=½(t²−x²): {}", cs[3]);
    assert!(cs[5] > 4.5, "red cúbica fija sufriría distorsión e^{{2η}} ≈ 4,95×: {}", cs[5]);

    // 2. Caminata cuántica unitaria sobre grafo dinámico con retroacción materia-geometría
    let ptr_g = u.dynamic_graph_walk_test(96, 0.25, 0.85, 0.35, 1.0, 55);
    let dg = unsafe { std::slice::from_raw_parts(ptr_g, 7) }.to_vec();
    println!(
        "t173 dyn graph: x_flat={:.2}, x_curved={:.2}, delay={:.2}, c_flat={:.3}, c_center={:.3}, norm_err={:.2e}",
        dg[0], dg[1], dg[2], dg[3], dg[4], dg[5]
    );
    assert!(dg[5] < 1e-12, "unitariedad exacta sobre grafo dinámico: err_norma = {}", dg[5]);
    assert!(dg[4] < dg[3] - 0.15, "la masa central debe reducir c(x) local: {} vs {}", dg[4], dg[3]);
    assert!(dg[2] > 2.0, "retraso de Shapiro emergente positivo: {} celdas", dg[2]);
}

#[test]
fn t174_3d_lightlike_doubler_obstruction() {
    let u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    // 1. Caminata de Weyl 3D estándar (θ = 0): tiene dobladores exactos (hueco = 0)
    let w0 = u.scan_3d_lightlike_walk(0.0, 0.0, 0.0, false, 24);
    // 2. Caminata simétrica de Gupta–Short 3D (θ = 0,4): mantiene modos espurios cerca de 0
    let gs = u.scan_3d_lightlike_walk(0.4, 0.4, 0.4, false, 24);
    // 3. Barrido sistemático multihilo en (θ_x, θ_y, θ_z)
    let sr = u.search_3d_lightlike_doublers(24);
    println!(
        "t174 3D lightlike: Weyl(0) eps_min={:.4} (c={:.3}, n_low={:.0}); GS(0.4) eps_min={:.4} (c={:.3}, n_low={:.0}); best={:.4} at ({:.2},{:.2},{:.2}) c={:.3}",
        w0[0], w0[4], w0[5], gs[0], gs[4], gs[5], sr[0], sr[1], sr[2], sr[3], sr[4]
    );
    assert!(w0[0] < 1e-3, "Weyl 3D estándar debe tener dobladores exactos: eps_min = {}", w0[0]);
    assert!((w0[4] - 1.0).abs() < 0.02, "velocidad de cono Weyl estándar c = 1: {}", w0[4]);
    assert!(sr[0] >= gs[0] - 1e-9, "el barrido debe encontrar un hueco ≥ GS simétrico");
    assert!(sr[4] > 0.25, "el mejor candidato debe conservar cono físico no degenerado: c = {}", sr[4]);
}

#[test]
fn t175_discrete_floquet_map_learning() {
    let u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let fm = u.discover_discrete_floquet_map(0.5, -0.05, 0.5, 1);
    println!(
        "t175 Floquet map: learned_rate={:.4e}, exact_rate={:.4e} (ratio={:.4}), q={:.3}/{:.3}, coin_err={:.2e}, thp_err={:.2e}, cont_pde={:.1}",
        fm[0], fm[1], fm[0] / fm[1].max(1e-30), fm[2], fm[3], fm[4], fm[5], fm[6]
    );
    assert!(fm[4] < 1e-4, "la matriz de moneda 4×4 debe aprenderse con precisión < 1e-4: {}", fm[4]);
    assert!(fm[5] < 2e-2, "θ′(ρ̄) debe aprenderse con error < 2%: {}", fm[5]);
    assert!(fm[1] > 1e-3, "la caminata exacta con g < 0 debe ser paramétricamente inestable: {}", fm[1]);
    let ratio = fm[0] / fm[1];
    assert!((ratio - 1.0).abs() < 0.05, "el mapa discreto aprendido debe predecir la tasa de Floquet al 5%: ratio = {}", ratio);
    assert!((fm[2] - fm[3]).abs() < 0.05, "debe acertar el número de onda inestable q_*: {} vs {}", fm[2], fm[3]);
    assert_eq!(fm[6], 0.0, "referencia: la EDP continua aprendida da tasa 0 para g < 0");
}

#[test]
fn t176_fermion_cpt_cohen_glashow() {
    let u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let flv = u.fermion_cpt_lorentz_bounds();
    println!(
        "t176 fermion LV: A1_rms={:.5}, E_th(l_P)={:.3e} GeV, dx/l_P < {:.3e} (IceCube 6 PeV), < {:.3e} (KM3NeT 220 PeV), asym_pal={:.2e}",
        flv[0], flv[1], flv[2], flv[3], flv[4]
    );
    assert!((flv[0] - 2.0 / 105.0f64.sqrt()).abs() < 1e-12, "A1_rms exacto = 2/√105");
    // A escala de Planck, el umbral de Cohen–Glashow para Weyl de orden fijo cae en ~40,3 TeV (4,03e4 GeV),
    // muy por debajo de IceCube (6 PeV) y KM3NeT (220 PeV) ⇒ ambos excluyen Δx = ℓ_P por órdenes de magnitud!
    assert!(flv[1] > 3.0e4 && flv[1] < 5.0e4, "umbral E_th(ℓ_P) debe estar en ~40 TeV: {:.3e} GeV", flv[1]);
    assert!(flv[2] < 1e-6, "IceCube 6 PeV debe excluir Δx = ℓ_P con margen > 10⁶: {:.3e}", flv[2]);
    assert!(flv[3] < 1e-11, "KM3NeT 220 PeV debe excluir Δx = ℓ_P con margen > 10¹¹: {:.3e}", flv[3]);
    assert!(flv[4] < 1e-14, "la caminata palindrómica debe tener asimetría CPT exacta = 0: {:.2e}", flv[4]);
}
