//! Tests de la fase 14 (165–170): sistemas cuánticos abiertos de pocos modos (bloqueo del falso
//! vacío, función de Wigner y negatividad no gaussiana, des-compresión y pureza frente a
//! entrelazamiento), complejidad efectiva EFT e índice de ensamblaje de la sopa, relación de
//! dispersión de Kramers–Kronig de la polarización del vacío + fotón oscuro U(1)_D, y mecanismo
//! GIM con extensión BSM Z′ en el modelo estándar de tres generaciones.

use super::common::base_2d;
use crate::QuantumSpacetimeUniverse;

#[test]
fn t165_false_vacuum_cosmic_lockdown() {
    let mut u = QuantumSpacetimeUniverse::new(64, 64, 1, 0.5, 0.1, 1.0);
    // Pozo doble asimétrico: energía inicial estrictamente por debajo de la barrera
    let run = |u: &mut QuantumSpacetimeUniverse, gamma: f32, kind: u32| -> Vec<f64> {
        let p = u.false_vacuum_lockdown(0.5, 1.45, 0.025, 1.0, gamma, kind, 14.0, 350);
        let n = u.open_quantum_out_len() as usize;
        unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
    };
    let closed = run(&mut u, 0.0, 1);
    let mid = run(&mut u, 1.0, 1);
    let locked = run(&mut u, 8.0, 1);
    println!(
        "t165: E_init={:.4} < V_B={:.4}; p_max cerrado={:.4}, γ=1: {:.4}, γ=8: {:.4}; pureza γ=8: {:.4}",
        closed[5], closed[6], closed[1], mid[1], locked[1], locked[3]
    );
    // Control clásico: energía inicial < barrera ⇒ partícula clásica atrapada (0)
    assert!(closed[5] < closed[6], "E_init = {} debe estar por debajo de V_barrera = {}", closed[5], closed[6]);
    assert_eq!(closed[7], 0.0, "la partícula clásica con E_init no puede cruzar la barrera");
    // Traza exacta en los tres casos
    for (name, r) in [("cerrado", &closed), ("medio", &mid), ("bloqueado", &locked)] {
        assert!(r[4] < 1e-11, "{name}: deriva de traza {}", r[4]);
    }
    // Túnel unitario (γ = 0): pureza = 1 y paso claro al vacío verdadero
    assert!((closed[3] - 1.0).abs() < 1e-11, "pureza unitaria = {}", closed[3]);
    assert!(closed[1] > 0.50, "sin decoherencia debe tunelar al vacío verdadero: p_max = {}", closed[1]);
    // Cosmic Lockdown (Zenón cuántico): la decoherencia suprime el túnel drásticamente
    assert!(mid[1] < closed[1], "γ = 1 debe reducir p_max: {} vs {}", mid[1], closed[1]);
    assert!(locked[1] < 0.25 * closed[1], "γ = 8 debe bloquear el falso vacío: p_max = {} vs {}", locked[1], closed[1]);
    assert!(locked[3] < 0.98, "la decoherencia mezcla el estado: pureza = {}", locked[3]);

    // Comprobación complementaria con monitor lineal en posición (kind = 0) a γ moderada
    let lin_lock = run(&mut u, 0.25, 0);
    assert!(lin_lock[4] < 1e-11 && lin_lock[1] < closed[1], "monitor lineal debe suprimir el túnel inicial: {} vs {}", lin_lock[1], closed[1]);

    // Dinámica clásica espacial en el campo de Higgs abeliano (q = 0, sesgo ε > 0):
    // radio crítico 2D R_c = 4·v²·√(λ c²) / (3ε) = 4·1·√(0,04·0,25)/(3·0,03) ≈ 4,44 celdas
    let mut uh = QuantumSpacetimeUniverse::new(64, 64, 1, 0.5, 0.0, 1.0);
    uh.set_field_substeps(4);
    uh.set_em(2, 0.0, 0.25);
    uh.set_relaxation(0.0);
    uh.set_higgs(1, 1.0, 0.04);
    uh.set_higgs_bias(0.03);
    // Burbuja subcrítica R = 2,5 < R_c: colapsa
    uh.seed_vacuum_bubble(32.0, 32.0, 0.0, 2.5, 1.5);
    let f0_sub = uh.higgs_true_vacuum_fraction();
    let e0_sub = uh.higgs_energy();
    uh.step_n(40);
    let f1_sub = uh.higgs_true_vacuum_fraction();
    let e1_sub = uh.higgs_energy();
    assert!(f0_sub > 0.002 && f1_sub < 0.001, "burbuja subcrítica debe colapsar: {f0_sub} → {f1_sub}");
    assert!(((e1_sub - e0_sub) / e0_sub.abs().max(1.0)).abs() < 2e-3, "energía conservada en colapso: {e0_sub} → {e1_sub}");

    // Burbuja supercrítica R = 9,0 > R_c: se expande
    uh.seed_vacuum_bubble(32.0, 32.0, 0.0, 9.0, 1.8);
    let f0_sup = uh.higgs_true_vacuum_fraction();
    uh.step_n(30);
    let f1_sup = uh.higgs_true_vacuum_fraction();
    assert!(f1_sup > 2.2 * f0_sup, "burbuja supercrítica debe expandirse: {f0_sup} → {f1_sup}");
}

#[test]
fn t166_wigner_negativity_and_decoherence() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let run = |u: &mut QuantumSpacetimeUniverse, om: f32, xi: f32, g3: f32, g4: f32, gphi: f32, gth: f32, nth: f32, init: u32, a0: f32, tm: f32| -> Vec<f64> {
        let p = u.wigner_open_oscillator(om, xi, g3, g4, gphi, gth, nth, init, a0, tm, 200, 48, 4.5);
        unsafe { std::slice::from_raw_parts(p, 9) }.to_vec()
    };
    // 1. Teorema de Hudson: compresión gaussiana pura (g3 = g4 = 0, γ = 0) ⇒ N_W ≈ 0 pese a compresión fuerte
    let gauss = run(&mut u, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1.4);
    assert!(gauss[6] < 1e-11 && (gauss[5] - 1.0).abs() < 5e-3, "norma/traza Wigner gaussiana: {:?}", gauss);
    assert!(gauss[7] < 1e-3, "truncación de Fock limpia: edge = {}", gauss[7]);
    assert!(gauss[2] > 0.999, "estado gaussiano puro: pureza = {}", gauss[2]);
    assert!(gauss[3] > 2.0 * gauss[4] || gauss[4] > 2.0 * gauss[3], "debe estar comprimido: Vq={}, Vp={}", gauss[3], gauss[4]);
    assert!(gauss[0] < 1e-3, "teorema de Hudson (TWA válido): N_W = {}", gauss[0]);

    // 2. Ireland & Vennin (2026): añadir no gaussianidad (g3, g4) con modo congelado/cuasi-resonante
    //    genera fuerte negatividad de Wigner (W < 0) donde TWA clásica falla
    let nongauss = run(&mut u, 0.15, 0.45, 0.14, 0.08, 0.0, 0.0, 0.0, 0, 0.0, 2.2);
    assert!(nongauss[6] < 1e-11 && nongauss[7] < 1e-2, "norma/borde no gaussiano: {:?}", nongauss);
    assert!(nongauss[0] > 0.10 && nongauss[1] < -0.02, "no gaussianidad debe generar W < 0: N_W={}, W_min={}", nongauss[0], nongauss[1]);

    // 3. Decoherencia ambiental sobre el estado no gaussiano: suprime N_W
    let dec_ng = run(&mut u, 0.15, 0.45, 0.14, 0.08, 0.45, 0.25, 0.5, 0, 0.0, 2.2);
    assert!(dec_ng[0] < 0.25 * nongauss[0], "la decoherencia debe suprimir N_W: {} vs {}", dec_ng[0], nongauss[0]);

    // 4. Control frente a superposiciones macroscópicas en entornos cálidos (Video 2):
    // gato de Schrödinger |α⟩ + |−α⟩ en baño térmico (n_th = 3) pierde toda su negatividad de Wigner
    let cat_pure = run(&mut u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, 1.8, 0.0);
    let cat_warm = run(&mut u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 3.0, 2, 1.8, 1.0);
    assert!(cat_pure[0] > 0.25, "gato puro debe tener fuerte N_W = {}", cat_pure[0]);
    assert!(cat_warm[0] < 0.02, "baño térmico borra las franjas cuánticas: N_W = {}", cat_warm[0]);
    assert!(cat_warm[2] < 0.4, "el baño térmico deja un estado mezcla (no un colapso): pureza = {}", cat_warm[2]);
}

#[test]
fn t167_gaussian_desqueezing_and_entanglement() {
    let mut u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    // 1. Evolución libre vs canal abierto (des-compresión de Micheli, Oshima y Takahashi, PRD 2026)
    let p_free = u.gaussian_desqueezing_run(0.0, 0.5, 0.0, 0.0, 0.0, 2.0, 400);
    let free = unsafe { std::slice::from_raw_parts(p_free, 7) }.to_vec();
    let p_open = u.gaussian_desqueezing_run(0.0, 0.5, 0.8, 0.3, 0.0, 2.0, 400);
    let open = unsafe { std::slice::from_raw_parts(p_open, 7) }.to_vec();
    // Libre: puro, r = λ·t = 1,0, E_N = 2r / ln 2
    assert!((free[0] - 1.0).abs() < 1e-10, "libre puro: {}", free[0]);
    assert!((free[1] - 1.0).abs() < 1e-8, "r libre = λt = 1: {}", free[1]);
    assert!((free[5] - 2.0 / 2.0f64.ln()).abs() < 1e-6, "E_N libre = 2r/ln2: {}", free[5]);
    // Abierto: cae la pureza Y hay des-compresión (r_eff < r_libre)
    assert!(open[0] < 0.6, "pureza abierta debe caer: {}", open[0]);
    assert!(open[1] < 0.75 * free[1], "debe haber des-compresión: r_eff={} vs r_libre={}", open[1], free[1]);
    assert!(open[3] >= 0.5 - 1e-9, "Heisenberg ν_− ≥ ½: {}", open[3]);

    // 2. Resultado central de Micheli et al. (2026): dos estados con la MISMA pureza γ,
    // uno entrelazado (ν̃_− < 0,5, E_N > 0) y otro separable (ν̃_− ≥ 0,5, E_N = 0)
    let cmp = u.gaussian_same_purity_comparison(0.75, 0.10, 0.35, 0.2, 2.0);
    let (pur_a, r_a, nut_a, en_a) = (cmp[0], cmp[1], cmp[2], cmp[3]);
    let (pur_b, r_b, nut_b, en_b) = (cmp[4], cmp[5], cmp[6], cmp[7]);
    assert!((pur_a - pur_b).abs() < 1e-7, "misma pureza: γ_A={pur_a} vs γ_B={pur_b}");
    assert!(r_a > 2.0 * r_b, "A más comprimido que B: r_A={r_a} vs r_B={r_b}");
    assert!(nut_a < 0.45 && en_a > 0.2, "A debe seguir entrelazado: ν̃_−A={nut_a}, E_NA={en_a}");
    assert!(nut_b >= 0.5 && en_b == 0.0, "B debe ser separable pese a tener igual pureza: ν̃_−B={nut_b}, E_NB={en_b}");

    // 3. Pareja de Bogoliubov de Hawking con ruido térmico: pérdida de entrelazamiento al bajar η
    let hw_pure = u.analog_bogoliubov_entanglement(0.8, 1.0, 0.0);
    let hw_noisy = u.analog_bogoliubov_entanglement(0.8, 0.35, 1.0);
    assert!((hw_pure[0] - 1.0).abs() < 1e-10 && hw_pure[4] > 2.0);
    assert!(hw_noisy[4] == 0.0 && hw_noisy[3] >= 0.5, "ruido fuerte rompe el entrelazamiento de Hawking: {:?}", hw_noisy);
}

#[test]
fn t168_eft_compression_and_soup_assembly() {
    let mut u = QuantumSpacetimeUniverse::new(64, 64, 1, 0.5, 0.1, 1.0);
    // 1. Complejidad efectiva EFT (envelope.rs): suave y diluida vs k_c alto y denso vs Floquet repulsivo
    let get_eft = |u: &mut QuantumSpacetimeUniverse, g: f32, rho: f32, kc: f32| -> Vec<f64> {
        let p = u.envelope_eft_complexity(0.5, g, 0.0, rho, kc, 1);
        unsafe { std::slice::from_raw_parts(p, 7) }.to_vec()
    };
    let smooth = get_eft(&mut u, 0.1, 0.02, 0.20);
    let rough = get_eft(&mut u, 0.1, 0.25, 0.70);
    let rep_flq = get_eft(&mut u, -0.05, 0.50, 0.20);
    assert!(smooth[1] < 0.01 && smooth[3] > 0.3, "régimen EFT suave: res={}, comp={}", smooth[1], smooth[3]);
    assert!(rough[1] > 3.5 * smooth[1] && rough[3] < 0.7 * smooth[3] && rough[2] > smooth[2],
        "régimen alto-k/denso rompe compresión EFT: rough={:?} vs smooth={:?}", rough, smooth);
    assert!(rep_flq[6] > 1e-4, "g < 0 con ρ alto tiene tasa de Floquet positiva: {}", rep_flq[6]);

    // 2. Proxy de ensamblaje y multiplicidad de la sopa (analysis.rs):
    //    separa limpiamente ruido, picos de 1 celda (alto IPR, A_proxy = 0) y sopa de solitones multicelda
    let mut u_noise = base_2d(64, 64);
    u_noise.seed_random_soup(0.15, 7);
    let c_noise = u_noise.analyze_soup_complexity();

    let mut u_spikes = base_2d(64, 64);
    u_spikes.seed_uniform_background(0.02, 0.0, 1);
    for k in 0..12 {
        let x = 5 + (k % 4) * 14;
        let y = 8 + (k / 4) * 18;
        u_spikes.inject_energy_blob(x as i32, y as i32, 0, 0.4, 25.0);
    }
    let c_spikes = u_spikes.analyze_soup_complexity();

    let mut u_soup = base_2d(64, 64);
    // Sembramos 9 solitones idénticos (misma clase estructural, múltiples copias n_c > 1)
    for ix in 0..3 {
        for iy in 0..3 {
            u_soup.inject_energy_blob((12 + ix * 20) as i32, (12 + iy * 20) as i32, 0, 4.0, 160.0);
        }
    }
    u_soup.step_n(120);
    let c_soup = u_soup.analyze_soup_complexity();

    assert!(c_noise[0] < 0.5 && c_noise[6] < 3.0, "ruido: A={}, IPR={}", c_noise[0], c_noise[6]);
    assert!(c_spikes[6] > 20.0 && c_spikes[3] >= 10.0 && c_spikes[0] == 0.0,
        "picos de 1 celda tienen IPR alto ({}) pero A_proxy = 0 ({}), picos = {}", c_spikes[6], c_spikes[0], c_spikes[3]);
    assert!(c_soup[2] >= 6.0 && c_soup[5] >= 3.0 && c_soup[0] > 3.0,
        "sopa de solitones multicelda debe dar A_proxy alto y copias repetidas: {:?}", c_soup);
}

#[test]
fn t169_vacuum_polarization_kramers_kronig_and_dark_photon() {
    let u = QuantumSpacetimeUniverse::new(64, 64, 1, 0.5, 0.0, 1.0);
    // 1. Polarización del vacío dependiente de ω en la caminata 1D de Gupta y Short (m = 0,25, θ = 0,4)
    let m = 0.25f32;
    let theta = 0.4f32;
    let q = 0.3f32;
    let below = u.vacuum_polarization_spectrum(m, theta, q, 0.22, 2e-3, 256, 320);
    let above = u.vacuum_polarization_spectrum(m, theta, q, 0.72, 2e-3, 256, 320);
    // Por debajo del umbral 2m = 0,5: apantallamiento χ′ > 0 y absorción prácticamente nula
    assert!(below[0] > 0.02, "χ′(ω < 2m) debe ser positivo: {}", below[0]);
    assert!(below[1].abs() < 0.05 * below[0], "χ″(ω < 2m) debe ser casi nulo: χ″={}, χ′={}", below[1], below[0]);
    // Por encima del umbral 2m = 0,5: absorción clara χ″ > 0 y coincide con 2·Γ_par/E₀² (teorema óptico)
    assert!(above[1] > 0.01 && above[2] > 0.0, "χ″(ω > 2m) y Γ_par deben ser positivos: {:?}", above);
    let opt_ratio = above[3] / above[1];
    assert!(opt_ratio > 0.7 && opt_ratio < 1.35, "teorema óptico χ″ frente a 2Γ/E₀²: ratio = {opt_ratio} ({:?})", above);

    // 2. Relación de dispersión de Kramers–Kronig:
    //    a) en el límite continuo (m = 0,05, θ = 0,3), la integral dispersiva de χ″(ω)/ω
    //       reproduce q²·c_ef/(6πm²) a < 3%
    let kk_cont = u.kramers_kronig_vp_check(0.05, 0.3, 0.2);
    //    b) en m = 0,25, θ = 0,4, la integral dispersiva de red (que incluye el aporte UV de red)
    //       coincide con la respuesta directa en tiempo real χ′(ω = 0,22) (~5% de mejora dinámica)
    let kk_lat = u.kramers_kronig_vp_check(m, theta, q);
    //    c) contraste directo con el apantallamiento estático de `homogeneous_sea_run` (m = 0,10)
    let mut u_sea = QuantumSpacetimeUniverse::new(8, 8, 1, 0.5, 0.0, 1.0);
    let kk_m10 = u_sea.kramers_kronig_vp_check(0.10, 0.3, 0.1);
    let p_sea = u_sea.homogeneous_sea_run(0.10, 0.3, 0.1, 1e-4, true, 2400);
    let sea_e = unsafe { std::slice::from_raw_parts(p_sea, 4800) }[2400..].to_vec();
    let e_mean = sea_e[400..2400].iter().sum::<f64>() / 2000.0 / 1e-4;
    let e_pred_kk = 1.0 / (1.0 + kk_m10[0]);
    println!(
        "t169: KK(m=0,05)={:.5} vs cont={:.5} (razón {:.4}); ⟨E⟩/E₀(m=0,10)={:.5} vs 1/(1+χ_KK)={:.5}; KK(m=0,25)={:.5} vs RT(ω=0,22)={:.5}; frac_umbral={:.3}, Δ4%={:.4}",
        kk_cont[0], kk_cont[1], kk_cont[0] / kk_cont[1], e_mean, e_pred_kk, kk_lat[0], below[0], kk_cont[2], kk_cont[3]
    );
    assert!((kk_cont[0] / kk_cont[1] - 1.0).abs() < 0.03, "Kramers–Kronig m=0,05: {} vs continuo {}", kk_cont[0], kk_cont[1]);
    assert!((e_mean / e_pred_kk - 1.0).abs() < 0.005, "estático ⟨E⟩/E₀={} frente a 1/(1+χ_KK)={}", e_mean, e_pred_kk);
    assert!((below[0] / kk_lat[0] - 1.05).abs() < 0.08, "RT χ′(ω=0,22)={} frente a KK de red {}", below[0], kk_lat[0]);
    assert!(kk_cont[2] > 0.75, "la ventana de umbral [2m, 4m] domina la integral dispersiva: frac = {}", kk_cont[2]);
    assert!(kk_cont[3] > 0.03, "un sesgo de +4% en el umbral desplaza χ′(0) en > 3%: {}", kk_cont[3]);

    // 3. Fotón oscuro U(1)_D con mezcla cinética ε y masa m_A' en Maxwell 2D
    let mut ue = QuantumSpacetimeUniverse::new(32, 32, 1, 0.5, 0.0, 1.0);
    ue.set_field_substeps(4);
    ue.set_em(2, 0.05, 0.25);
    ue.set_relaxation(0.0);
    ue.set_dark_photon(true, 0.25, 0.20);
    // Modo transversal puro visible (k_x = 2π/32, A_y(x) = A₀ sin(kx)) y Gauss residual exacto
    let n = ue.width();
    for x in 0..n {
        let val = 0.2 * (2.0 * std::f64::consts::PI * x as f64 / n as f64).sin();
        for y in 0..n {
            ue.set_link_a(x as i32, y as i32, 0, 1, val);
        }
    }
    let e0 = ue.dark_photon_energy();
    let mut max_dark = 0.0f64;
    for _ in 0..80 {
        ue.step();
        let en = ue.dark_photon_energy();
        max_dark = max_dark.max(en[2]);
    }
    let e1 = ue.dark_photon_energy();
    assert!(ue.gauss_residual() < 1e-13, "ley de Gauss visible debe seguir exacta: {}", ue.gauss_residual());
    assert!(((e1[0] - e0[0]) / e0[0]).abs() < 5e-3, "energía total acoplada conservada: {} → {}", e0[0], e1[0]);
    assert!(max_dark > 0.01 * e0[0], "la mezcla cinética debe transferir energía al fotón oscuro: {max_dark} vs {}", e0[0]);

    // Resonancia de antena visible a ω_A' = m_A'/√(1 − ε²) frente a fuera de resonancia
    let probe_res = |omega_drive: f32| -> f64 {
        let mut ur = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.0, 1.0);
        ur.set_field_substeps(4);
        ur.set_em(2, 0.05, 0.25);
        ur.set_relaxation(0.0);
        ur.set_dark_photon(true, 0.20, 0.25);
        for i in 0..16 {
            for j in 0..16 {
                ur.add_static_current(i, j, 0, 1, 1e-3);
            }
        }
        ur.set_current_oscillation(omega_drive);
        ur.step_n(120);
        ur.dark_photon_energy()[2]
    };
    let w_ap = 0.25f32 / (1.0 - 0.04f32).sqrt();
    let e_on = probe_res(w_ap);
    let e_off = probe_res(0.65);
    assert!(e_on > 5.0 * e_off, "resonancia del fotón oscuro en ω = m_A'/√(1−ε²): on={e_on} vs off={e_off}");
}

#[test]
fn t170_sm_gim_and_zprime_fcnc() {
    let mut u = QuantumSpacetimeUniverse::new(12, 12, 12, 0.5, 0.0, 1.0);
    u.set_scenario(6);
    u.set_scenario_sm(3, true, false, 0.6, 0.4, 0.8);
    let ym = u.random_yukawa_matrices(7, false);
    u.set_yukawa_matrices_sm(&ym[0..18], &ym[18..36], &ym[36..54]);
    // 1. Mecanismo GIM exacto en el modelo estándar de 3 generaciones:
    //    Z⁰ no cambia sabor (< 10⁻¹⁴) pese a una CKM no trivial en W¹
    let gim = u.sm_gim_check();
    assert_eq!(gim.len(), 7);
    assert!(gim[0] < 1e-13 && gim[1] < 1e-13, "Z⁰ debe ser estrictamente diagonal en sabor (GIM): d={}, u={}", gim[0], gim[1]);
    assert!(gim[2] > 0.10, "la CKM en W¹ debe tener mezcla fuera de la diagonal: max={}", gim[2]);
    assert!(gim[5] < 1e-12, "unitariedad CKM: {}", gim[5]);
    assert!(gim[6] < 1e-13, "acoplamiento Z⁰ universal entre generaciones: spread={}", gim[6]);

    // 2. Evolución real 3D con k = 0 y bosones fijos: inyectar b_L puro (generación de masa 2)
    //    Usamos las mismas matrices W_d, W_u complejas no triviales escaladas a masas diminutas
    //    (10⁻⁶) para que no haya mezcla L↔R durante los 20 pasos mientras W_d sigue siendo general.
    u.set_sm_bosons_dynamic(false);
    let ym_tiny: Vec<f32> = ym.iter().map(|&x| x * 1e-6).collect();
    u.set_yukawa_matrices_sm(&ym_tiny[0..18], &ym_tiny[18..36], &ym_tiny[36..54]);
    u.clear();
    u.set_sm_link_field(2, 3, 0.25, -0.15);
    u.set_sm_zprime_field(2, 0.0, 0.0);
    // Onda plana homogénea k = 0 (σ grande) para aislar la rotación exacta de sabor sin mezcla de espín
    u.inject_sm_mass_state(4, 2, 6.0, 6.0, 6.0, 100.0, 0.0, 0.0, 0.0, 0);
    u.step_n(20);
    let norms_sm = u.sm_down_mass_norms();
    assert!(norms_sm[1] < 1e-24 && norms_sm[0] < 1e-24, "SM Z⁰ no puede convertir b_L → s_L: {:?}", norms_sm);
    assert!((norms_sm[2] - 1.0).abs() < 1e-8, "b_L se conserva intacto bajo Z⁰: {:?}", norms_sm);

    //    b) con campo externo BSM Z′ (θ_bs = 0,04) → conversión b_L → s_L = sin²(θ_bs·t) exacta
    u.clear();
    u.set_sm_link_field(2, 3, 0.0, 0.0);
    let tbs = 0.04f32;
    u.set_sm_zprime_field(2, tbs, 0.02);
    u.inject_sm_mass_state(4, 2, 6.0, 6.0, 6.0, 100.0, 0.0, 0.0, 0.0, 0);
    let steps = 20u32;
    u.step_n(steps);
    let norms_zp = u.sm_down_mass_norms();
    let pred_s = ((tbs as f64) * (steps as f64)).sin().powi(2);
    let pred_b = ((tbs as f64) * (steps as f64)).cos().powi(2);
    assert!((norms_zp[1] - pred_s).abs() < 1e-6, "Z′ b_L→s_L: {} vs sin²(θt)={pred_s}", norms_zp[1]);
    assert!((norms_zp[2] - pred_b).abs() < 1e-6, "Z′ b_L restante: {} vs cos²(θt)={pred_b}", norms_zp[2]);
    assert!(norms_zp[0] < 1e-24, "d_L no se contamina: {}", norms_zp[0]);

    // 3. Tasas de desintegración Z′ → b_L s̄_L + s_L b̄_L frente a Z′ → μ_L⁺ μ_L⁻ en el nodo 0
    //    (régimen perturbativo de regla de oro de Fermi, g·T ≪ 1 como en los tests 132 y 136)
    let zp_rates = u.pair_rate_zprime(0, 0.2, 1.5e-4, 2.5e-4, 600);
    println!("t170: Γ_bs={:.4e}, Γ_μ={:.4e}, razón={:.4} vs 6·g_bs²/g_μ²={:.4}", zp_rates[0], zp_rates[1], zp_rates[2], zp_rates[3]);
    assert!(zp_rates[0] > 0.0 && zp_rates[1] > 0.0, "tasas Z′ positivas: {:?}", zp_rates);
    assert!((zp_rates[2] / zp_rates[3] - 1.0).abs() < 0.02,
        "razón Γ(Z′→bs̄+sb̄)/Γ(Z′→μμ) = {} frente a 6·g_bs²/g_μ² = {}", zp_rates[2], zp_rates[3]);
}
