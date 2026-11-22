//! Ejecutable nativo por lotes (Fase 15) con soporte multihilo (`std::thread::scope`) para los
//! estudios pesados de muchos cuerpos (MPS), espacio-tiempo covariante y problemas abiertos de
//! frontera (§8), sin las limitaciones de memoria y tiempo de un navegador.
//!
//! Uso:
//!   cargo run --release --bin qsu_cli -- [all|phase15|phase16|phase17|phase18|phase19|phase20|pub-checks|via-a|via-b|via-c|via-a-chiral|via-b-fss|via-c-scaling|via-a-yee4|via-b-wl|via-b-multiseed|via-c-tdvp|via-c-chi-conv|schwinger|collins|causal-set|scan-3d|floquet-map|fermion-lv|dmrg|causal-mc|su2-4d|spin2|noncubic|kaplan-sen|study1|study2|study3]

use qca_spacetime_lab::research_io::write_research_file;
use qca_spacetime_lab::via_b_causal_pt::{
    run_parallel_tempering_2d, run_via_b_fss_suite, run_via_b_large_nf_point, run_via_b_matter_mass_nf_sweep,
    run_via_b_wang_landau_2d, VIA_B_BETAS,
};
use qca_spacetime_lab::via_c_schwinger2f::run_via_c_chi_convergence;
use qca_spacetime_lab::QuantumSpacetimeUniverse;
use std::time::Instant;

fn main() {
    let task = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let u = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
    let run_all = task == "all";
    let run_p15 = run_all || task == "phase15";
    let run_p16 = run_all || task == "phase16";
    let run_p17 = run_all || task == "phase17";
    let run_p18 = run_all || task == "phase18";
    let run_p19 = run_all || task == "phase19";
    let run_p20 = run_all || task == "phase20";

    if run_p15 || task == "schwinger" {
        let t0 = Instant::now();
        println!("=== 1. QFT 1+1D muchos cuerpos (MPS + MPO flujo eléctrico exacto): Modelo de Schwinger ===");
        let conf = u.mps_schwinger_confinement(12, 0.5, 0.6, 0.5, 16);
        println!(
            "  Confinamiento (L=12, m=0,6, J=0,5): E0={:.4}, V(2)={:.4}, V(4)={:.4}, V(6)={:.4}, S_vN(0)={:.4}, S_vN(4)={:.4}, E_centro(4)={:.4}",
            conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], conf[6]
        );
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let ptr = u2.mps_schwinger_string_breaking(12, 0.8, 0.15, 0.9, 6, 16, 0.08, 45);
        let len = u2.open_quantum_out_len() as usize;
        let sb = unsafe { std::slice::from_raw_parts(ptr, len) };
        println!(
            "  Rotura cuántica de cuerda (L=12, d=6, m=0,15, J=0,9): E_c {:.3} → mín {:.3} → fin {:.3}; S_vN máx={:.4} ({:.2} s)",
            sb[0], sb[1], sb[2], sb[3], t0.elapsed().as_secs_f64()
        );
    }

    if run_p15 || task == "collins" {
        let t0 = Instant::now();
        println!("=== 2. Violación de Lorentz radiativa a un lazo (Collins et al., 2004) + MPS interactuante ===");
        let lv = u.collins_one_loop_lv(0.25, 1.2, 0.5, 1.0);
        println!(
            "  Un lazo Yukawa 1+1D: δc²_cont(Λ=120)={:.3e}, δc²_red(r=1)={:.4e}, δc²_red(r=0)={:.4e}, Δc_especies={:.4e}",
            lv[0], lv[2], lv[3], lv[4]
        );
        let v0 = u.mps_interacting_velocity(16, 0.5, 0.0, 16, 0.08, 30);
        let v1 = u.mps_interacting_velocity(16, 0.5, 0.22, 16, 0.08, 30);
        println!(
            "  Cono de luz MPS XXZ: libre (Δ=0) v={:.4} (Bethe {:.4}); interactuante (Δ=0,22) v={:.4} (Bethe {:.4}) ({:.2} s)",
            v0[0], v0[1], v1[0], v1[1], t0.elapsed().as_secs_f64()
        );
    }

    if run_p15 || task == "causal-set" {
        let t0 = Instant::now();
        println!("=== 3. Espacio-tiempo discreto covariante (Conjuntos Causales + Grafo Dinámico) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let ptr = u2.causal_set_lorentz_test(200, 0.8, 24, 7);
        let cs = unsafe { std::slice::from_raw_parts(ptr, 7) };
        println!(
            "  Conjunto causal (N=200, boost η=0,8): d_MM={:.4} (exacto 2), Δenlaces={:.0}, grado_medio={:.2}, □_BD reposo={:.3} / boost={:.3} (exacto 2,0), distorsión cúbica={:.2}×",
            cs[0], cs[1], cs[2], cs[3], cs[4], cs[5]
        );
        let ptr_g = u2.dynamic_graph_walk_test(96, 0.25, 0.85, 0.35, 1.0, 55);
        let dg = unsafe { std::slice::from_raw_parts(ptr_g, 7) };
        println!(
            "  Grafo dinámico: x_plano={:.2}, x_curvo={:.2}, retraso_Shapiro={:.2} celdas, c_centro={:.3} vs c_plano={:.3}, err_norma={:.2e} ({:.2} s)",
            dg[0], dg[1], dg[2], dg[4], dg[3], dg[5], t0.elapsed().as_secs_f64()
        );
    }

    if run_p15 || task == "scan-3d" {
        let t0 = Instant::now();
        println!("=== 4. Búsqueda multihilo de caminatas 3D de tipo luz sin dobladores (§8.6) ===");
        let sr = u.search_3d_lightlike_doublers(24);
        println!(
            "  Weyl θ=0 hueco={:.4}; Gupta–Short simétrico (0,4) hueco={:.4}; mejor barrido 3D hueco={:.4} en θ=({:.2},{:.2},{:.2}), c={:.3} ({:.2} s)",
            sr[5], sr[6], sr[0], sr[1], sr[2], sr[3], sr[4], t0.elapsed().as_secs_f64()
        );
    }

    if run_p15 || task == "floquet-map" {
        let t0 = Instant::now();
        println!("=== 5. Aprendizaje del mapa discreto multicomponente de Floquet (§8.10) ===");
        let fm = u.discover_discrete_floquet_map(0.5, -0.05, 0.5, 1);
        println!(
            "  Mapa aprendido: tasa={:.4e} (exacta {:.4e}, razón {:.4}) en q={:.3} (exacto {:.3}); err_moneda={:.2e}; EDP continua={:.1} ({:.2} s)",
            fm[0], fm[1], fm[0] / fm[1].max(1e-30), fm[2], fm[3], fm[4], fm[6], t0.elapsed().as_secs_f64()
        );
    }

    if run_p15 || task == "fermion-lv" {
        println!("=== 6. Fermiones CPT-impares y umbral de Cohen–Glashow (§8.11) ===");
        let flv = u.fermion_cpt_lorentz_bounds();
        println!(
            "  A1_rms={:.4}; E_th(Δx=ℓ_P)={:.3e} GeV; Δx/ℓ_P < {:.3e} (IceCube 6 PeV), < {:.3e} (KM3NeT 220 PeV); asimetría palindrómica={:.2e}",
            flv[0], flv[1], flv[2], flv[3], flv[4]
        );
    }

    if run_p16 || task == "dmrg" {
        let t0 = Instant::now();
        println!("=== 7. [Fase 16] DMRG variacional (Bethe + CFT c=1) y flujo RG de Lorentz (Chadha–Nielsen) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.dmrg_xxz_cft_scaling(32, 4);
        let r = unsafe { std::slice::from_raw_parts(p, 10) };
        println!(
            "  DMRG XXZ Δ=1/2 (L=16..48): e_∞={:.6} (Bethe {:.6}, err={:.2e}), c_CFT={:.4} (exacto 1,0)",
            r[0], r[1], r[2], r[3]
        );
        let p2 = u2.two_species_lorentz_rg_study(10, 1.0, 0.5, 0.8, 24);
        let rg = unsafe { std::slice::from_raw_parts(p2, 10) };
        println!(
            "  Dos especies DMRG + RG: ΔE_lig={:.4}, ⟨S^z_A S^z_B⟩={:.4}→{:.4}; QED M_P→m_e ratio={:.4} (falla); fuerte γ_*=0,90 δc_IR={:.2e} (γ_min={:.3}) ({:.2} s)",
            rg[0], rg[1], rg[2], rg[5], rg[7], rg[8], t0.elapsed().as_secs_f64()
        );
    }

    if run_p16 || task == "causal-mc" {
        let t0 = Instant::now();
        println!("=== 8. [Fase 16] Monte Carlo dinámico de conjuntos causales 2D (acción Benincasa–Dowker, Surya 2012) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.causal_set_bd_mc(32, 1800, 16);
        let r = unsafe { std::slice::from_raw_parts(p, 9) };
        println!(
            "  β=0 (variedad 2D): S/N={:.3}, d_MM={:.3}, R={:.3} | β=2,0 (fase ordenada): S/N={:.3}, R={:.3} | β=-0,25 (bipartito): S/N={:.2}, N1/N={:.2} ({:.2} s)",
            r[0], r[1], r[2], r[3], r[5], r[6], r[7], t0.elapsed().as_secs_f64()
        );
    }

    if run_p16 || task == "su2-4d" {
        let t0 = Instant::now();
        println!("=== 9. [Fase 16] Yang–Mills cuántico SU(2) 3+1D euclídeo (Creutz 1980, baño térmico) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.lattice_su2_4d_mc(6, 30, 50, 16);
        let r = unsafe { std::slice::from_raw_parts(p, 7) };
        println!(
            "  Malla 6⁴: β=1,0 ⟨P⟩={:.4} (fuerte {:.4}); β=3,0 ⟨P⟩={:.4} (débil {:.4}); confinamiento χ(2,2) β=1,5: {:.3}, β=2,0: {:.3} ({:.2} s)",
            r[0], r[1], r[2], r[3], r[4], r[5], t0.elapsed().as_secs_f64()
        );
    }

    if run_p16 || task == "spin2" {
        println!("=== 10. [Fase 16] Espín 2 frente a modelos escalares y teorema de Weinberg–Witten ===");
        let r = u.spin2_emergent_gravity_check();
        println!(
            "  Polarizaciones TT err={:.1e}, rot45° espín-2 err={:.1e}, deflexión luz (Einstein/Nordström/Refract)=({:.1}, {:.1}, {:.1}), proy_WW(|h|=2)={:.1e}",
            r[0], r[1], r[2], r[3], r[4], r[6]
        );
    }

    if run_p16 || task == "noncubic" {
        let t0 = Instant::now();
        println!("=== 11. [Fase 16] Redes 3D no cúbicas (SC/BCC/FCC/isotrópica) y verosimilitud sobre catálogos reales ===");
        let m = u.noncubic_lattice_lv_map();
        println!(
            "  Anisotropía A₂: SC={:.4}, BCC={:.4}, FCC={:.4}, SC+FCC isotrópica={:.2e} (O(k⁴)={:.4}); asim CPT BCC fija={:.3e}",
            m[1], m[2], m[3], m[4], m[10], m[5]
        );
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.real_catalog_lv_likelihood(1500);
        let r = unsafe { std::slice::from_raw_parts(p, 10) };
        println!(
            "  Perfil χ² catálogos (1500 orient.): 95%CL < {:.3e}·ℓ_P, 5σ < {:.3e}·ℓ_P; KM3NeT conservador(110 PeV) < {:.3e}·ℓ_P ({:.2} s)",
            r[0], r[1], r[3], t0.elapsed().as_secs_f64()
        );
    }

    if run_p16 || task == "kaplan-sen" {
        let t0 = Instant::now();
        println!("=== 12. [Fase 16] Fermiones quirales en frontera finita (Kaplan–Sen 2024 / Golterman–Shamir 2025) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.chiral_4comp_kaplan_sen_test(14, 0.45);
        let r = unsafe { std::slice::from_raw_parts(p, 11) };
        println!(
            "  Losa L=14: ω_borde={:.5} (sin k_x={:.5}), ⟨γ⁵⟩_L={:.4}, ⟨γ⁵⟩_R={:.4} (suma={:.2e}), hueco doblador={:.3}, QCA4(r=1)={:.2} ({:.2} s)",
            r[0], r[1], r[2], r[3], r[4], r[7], r[9], t0.elapsed().as_secs_f64()
        );
    }

    if run_p17 || task == "study1" {
        let t0 = Instant::now();
        println!("=== 13. [Fase 17 — Estudio 1] Restauración de Lorentz vs hueco de masa en dos especies acopladas (Lanczos exacto + S(q,ω) + DMRG) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.study1_two_species_lorentz_phases(24, 16);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        let names = ["Control (g=0)", "1A Dens-Dens (g_z=0.6)", "1B Heisenberg (g=1.4)", "1C Mezcla (g_xy=0.75)"];
        println!("  Media armónica analítica 1C: v_harm = {:.4}", out[1]);
        for (i, name) in names.iter().enumerate() {
            let r = &out[(2 + 8 * i)..(2 + 8 * (i + 1))];
            println!(
                "  {:<23}: Δ_∞={:.4}, S(8)={:.3}→S(16)={:.3} (ΔS={:+.3}), vA_IR={:.3}, vB_IR={:.3}, mismatch={:.3}, max|ΔN_B|={:.4}",
                name, r[0], r[1], r[2], r[7], r[3], r[4], r[5], r[6]
            );
        }
        println!("  Tiempo Estudio 1: {:.2} s", t0.elapsed().as_secs_f64());
    }

    if run_p17 || task == "study2" {
        let t0 = Instant::now();
        println!("=== 14. [Fase 17 — Estudio 2] Clasificación triple de 8 familias QCA 3+1D + QCA isotrópico n = 4 ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.study2_3d_qca_triple_classification(22);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
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
        for (i, name) in names.iter().enumerate() {
            let r = &out[(1 + 6 * i)..(1 + 6 * (i + 1))];
            println!(
                "  {:<26}: asym={:.2e}, gap={:.4}, A2_spr={:.2e}, A2_mean={:+.2e}, A4_max={:.4}, pass3={:.0}",
                name, r[0], r[1], r[2], r[3], r[4], r[5]
            );
        }
        println!("  Tiempo Estudio 2: {:.2} s", t0.elapsed().as_secs_f64());
    }

    if run_p17 || task == "study3" {
        let t0 = Instant::now();
        println!("=== 15. [Fase 17 — Estudio 3] Conjuntos causales dinámicos + materia cuántica (Johnston + Sorkin–Johnston) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.study3_causal_set_matter_backreaction(20, 600, 6);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        println!(
            "  Propagador Johnston vs ½J₀(mτ): err_rms={:.4}; σ_G(cono)={:.4}, σ_G(bulk)={:.4}",
            out[0], out[1], out[2]
        );
        let nb_gr = out[11] as usize;
        let n_betas = out[12] as usize;
        let n_nfs = out[13] as usize;
        let base = 14 + 4 * nb_gr;
        for nf_i in 0..n_nfs {
            let mut line = String::new();
            for b_i in 0..n_betas {
                let row = &out[(base + 7 * (nf_i * n_betas + b_i))..(base + 7 * (nf_i * n_betas + b_i + 1))];
                line.push_str(&format!(
                    "  β={:+.2}: S/N={:+.2}±{:.2} dMM={:.2}±{:.2} |",
                    row[1], row[2], row[3], row[4], row[5]
                ));
            }
            println!("  N_f={:.0}:{line}", out[base + 7 * (nf_i * n_betas)]);
        }
        println!("  Tiempo Estudio 3: {:.2} s", t0.elapsed().as_secs_f64());
    }

    if run_p18 || task == "via-a" {
        let t0 = Instant::now();
        println!("=== 16. [Fase 18 — Vía A] QCA 3+1D isotrópico n = 4 en espacio real (U(1) exacto, S₂ vs S₄) + auditoría fotones ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.via_a_realspace_n4_qca(0.25);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        println!(
            "  Espacio real 6³ (U(1) + Maxwell): norm_drift={:.2e}, gauss={:.2e}, gauge_dens={:.2e}, gauge_E={:.2e}",
            out[0], out[1], out[2], out[3]
        );
        println!(
            "  Circuito local 2×2×2 Bloch: S₂(spr={:.3e}, mean={:+.3e}, |c-1|={:.2e}, asym={:.2e}) | S₄(spr={:.3e}, mean={:+.3e}, |c-1|={:.2e}, asym={:.2e}, corner_gap={:.4})",
            out[4], out[5], out[37], out[6], out[7], out[8], out[38], out[9], out[10]
        );
        let pnames = [
            "Mlodinow-Brun-2025",
            "Bisio-DAriano-Perinotti-2016",
            "Eon-et-al-Fijo-2023",
            "Eon-et-al-Palindromico-2023",
            "Yee4-S4-Nuevo",
        ];
        let mut csv = String::from("modelo,cpt_asym,a2_spread,a2_mean,corner_gap,pass_n4\n");
        csv.push_str(&format!(
            "Fermion-IsoSCFCC-S2,{:.6e},{:.6e},{:.6e},0.0,0\n",
            out[6], out[4], out[5]
        ));
        csv.push_str(&format!(
            "Fermion-IsoSCFCC-S4,{:.6e},{:.6e},{:.6e},{:.6},1\n",
            out[9], out[7], out[8], out[10]
        ));
        for (i, nm) in pnames.iter().enumerate() {
            let r = &out[(12 + 5 * i)..(12 + 5 * (i + 1))];
            println!(
                "  {:<30}: asym={:.2e}, A2_spr={:.2e}, A2_mean={:+.2e}, corner_gap={:.3}, pass_n4={:.0}",
                nm, r[0], r[1], r[2], r[3], r[4]
            );
            csv.push_str(&format!(
                "{},{:.6e},{:.6e},{:.6e},{:.6},{:.0}\n",
                nm, r[0], r[1], r[2], r[3], r[4]
            ));
        }
        let json = format!(
            "{{\n  \"via\": \"A\",\n  \"dt\": 0.25,\n  \"r_wilson\": 1.1547005383792517,\n  \"grid_realspace\": \"6x6x6\",\n  \"norm_drift\": {:.6e},\n  \"gauss_residual\": {:.6e},\n  \"gauge_density_err\": {:.6e},\n  \"gauge_e_err\": {:.6e},\n  \"s2_a2_spread\": {:.6e},\n  \"s2_a2_mean\": {:.6e},\n  \"s4_a2_spread\": {:.6e},\n  \"s4_a2_mean\": {:.6e},\n  \"s4_corner_gap\": {:.6}\n}}\n",
            out[0], out[1], out[2], out[3], out[4], out[5], out[7], out[8], out[10]
        );
        let _ = write_research_file("via_a_qca_n4.csv", &csv);
        let _ = write_research_file("via_a_resumen.json", &json);
        println!("  Exportado datos_investigacion/via_a_qca_n4.csv + via_a_resumen.json ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p18 || task == "via-b" {
        let t0 = Instant::now();
        println!("=== 17. [Fase 18 — Vía B] Parallel Tempering 2D Causal Sets + Sorkin–Johnston + Ising (N = 20, 28, 36, 48) ===");
        let sizes = [20usize, 28, 36, 48];
        let modes = [0i32, 1, 2, -1]; // 0=puro, 1=SJ1, 2=SJ2, -1=Ising
        let mut tasks = Vec::new();
        for &n in &sizes {
            let sw = if n <= 28 { 900 } else if n <= 36 { 650 } else { 420 };
            for &m in &modes {
                tasks.push((n, m, sw));
            }
        }
        let runs: Vec<_> = std::thread::scope(|sc| {
            let handles: Vec<_> = tasks
                .into_iter()
                .enumerate()
                .map(|(idx, (n, m, sw))| {
                    sc.spawn(move || {
                        run_parallel_tempering_2d(n, &VIA_B_BETAS, sw, m, 0x18B0_9001 ^ (idx as u64 * 104729))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut csv = String::from("N,matter_mode,swap_acc,beta,s_mean,s_err,dmm_mean,dmm_err,chi_s,cv,binder,tau_int,mat_mean\n");
        let mut summary = String::from("N,matter_mode,swap_acc,beta_c_est,chi_peak,dmm_b0,dmm_b0_err,s_bhi,s_bhi_err,s_bneg,s_bneg_err\n");
        for run in &runs {
            let mode_lbl = match run.matter_mode {
                0 => "Puro(Nf=0)",
                1 => "SJ(Nf=1)",
                2 => "SJ(Nf=2)",
                _ => "Ising(J=0.45)",
            };
            let r0 = &run.replicas[2];   // β = 0.00
            let rhi = &run.replicas[11]; // β = +1.35
            let rneg = &run.replicas[0]; // β = -0.25
            println!(
                "  N={:2} {:<13}: swap={:.2}, β_c={:.2} (χ_pk={:5.1}) | β=0: dMM={:.3}±{:.3} (τ={:.1}) | β=+1.35: S/N={:+.2}±{:.2} | β=-0.25: S/N={:+.2}±{:.2}",
                run.n_elem, mode_lbl, run.swap_acceptance, run.beta_c_est, run.chi_peak,
                r0.dmm_mean, r0.dmm_err, r0.tau_int, rhi.s_mean, rhi.s_err, rneg.s_mean, rneg.s_err
            );
            summary.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                run.n_elem, mode_lbl, run.swap_acceptance, run.beta_c_est, run.chi_peak,
                r0.dmm_mean, r0.dmm_err, rhi.s_mean, rhi.s_err, rneg.s_mean, rneg.s_err
            ));
            for rep in &run.replicas {
                csv.push_str(&format!(
                    "{},{},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.6}\n",
                    run.n_elem, mode_lbl, run.swap_acceptance, rep.beta, rep.s_mean, rep.s_err,
                    rep.dmm_mean, rep.dmm_err, rep.chi_s, rep.cv, rep.binder, rep.tau_int, rep.mat_mean
                ));
            }
        }
        let _ = write_research_file("via_b_causal_pt_full.csv", &csv);
        let _ = write_research_file("via_b_causal_pt_scaling.csv", &summary);
        println!("  Exportado datos_investigacion/via_b_causal_pt_full.csv + via_b_causal_pt_scaling.csv ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p18 || task == "via-c" {
        let t0 = Instant::now();
        println!("=== 18. [Fase 18 — Vía C] Schwinger 2 sabores (N_r = 10 ⇒ 63 504 estados exactos + DMRG por bloques U(1) χ = 48) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.via_c_schwinger2f_lorentz(10, 48);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        println!(
            "  Media geométrica analítica √(v_A v_B) = {:.4} | U(1)-DMRG XXZ L=24: e={:.5}, S={:.3} | MPO6 exact err={:.2e}",
            out[0], out[2], out[3], out[6]
        );
        println!(
            "  Caída carga central U(1)-DMRG (N_r=8→16, cortes pares): J=0: S={:.3}→{:.3} (c_eff={:.3}) | J=1.0: S={:.3}→{:.3} (c_eff={:.3})",
            out[7], out[8], out[9], out[10], out[11], out[12]
        );
        let nj = out[13] as usize;
        let mut csv = String::from("J,vA_IR,vB_IR,mismatch,omega_sym,E0\n");
        for i in 0..nj {
            let r = &out[(14 + 6 * i)..(14 + 6 * (i + 1))];
            println!(
                "  J={:.2}: vA_IR={:.4}, vB_IR={:.4}, mismatch={:.4}, ω_sym={:.4}, E0={:.5}",
                r[0], r[1], r[2], r[3], r[4], r[5]
            );
            csv.push_str(&format!("{:.4},{:.6},{:.6},{:.6},{:.6},{:.6}\n", r[0], r[1], r[2], r[3], r[4], r[5]));
        }
        let json = format!(
            "{{\n  \"via\": \"C\",\n  \"n_rungs_exact\": {:.0},\n  \"v_a\": 1.0,\n  \"v_b\": 0.45,\n  \"v_geom_pred\": {:.6},\n  \"u1_dmrg_xxz_e\": {:.6},\n  \"mpo6_exact_err\": {:.6e},\n  \"c_eff_j0\": {:.4},\n  \"c_eff_j15\": {:.4}\n}}\n",
            out[1], out[0], out[2], out[6], out[9], out[12]
        );
        let _ = write_research_file("via_c_schwinger2f.csv", &csv);
        let _ = write_research_file("via_c_resumen.json", &json);
        println!("  Exportado datos_investigacion/via_c_schwinger2f.csv + via_c_resumen.json ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p19 || task == "via-a-chiral" {
        let t0 = Instant::now();
        println!("=== 19. [Fase 19 — Vía A quiral 4+1D] Kaplan–Sen en losa abierta (SC vs Iso-SC+FCC vs Symanzik-4 n = 4) ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.via_a_chiral_kaplan_sen_n4(14, 0.55);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        let names = ["SC-Kaplan-Sen", "Iso-SC+FCC-Kaplan-Sen", "Symanzik4-Kaplan-Sen"];
        let mut csv = String::from("modelo,chi_left,chi_right,net_chiral,a2_spread,a2_mean,corner_gap,tier\n");
        for (i, nm) in names.iter().enumerate() {
            let r = &out[(7 * i)..(7 * (i + 1))];
            println!(
                "  {:<24}: ⟨γ⁵⟩_L={:+.4}, ⟨γ⁵⟩_R={:+.4}, suma={:+.2e}, spr(A₂)={:.3e}, ⟨A₂⟩={:+.3e}, gap={:.3}, tier={:.0}",
                nm, r[0], r[1], r[2], r[3], r[4], r[5], r[6]
            );
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6e},{:.6e},{:.6e},{:.6},{:.0}\n",
                nm, r[0], r[1], r[2], r[3], r[4], r[5], r[6]
            ));
        }
        let _ = write_research_file("via_a_chiral_kaplan_sen_n4.csv", &csv);
        println!("  Exportado datos_investigacion/via_a_chiral_kaplan_sen_n4.csv ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p19 || task == "via-b-fss" {
        let t0 = Instant::now();
        println!("=== 20. [Fase 19 — Vía B FSS] Parallel Tempering adaptativo 20 réplicas + checkpoints (N = 20, 28, 36, 48, 64) ===");
        let sizes = [20usize, 28, 36, 48, 64];
        let sweeps = [1600usize, 1400, 1200, 950, 650];
        let (runs, fits) = run_via_b_fss_suite(&sizes, &sweeps, true);
        let mode_name = |m: i32| match m {
            0 => "Puro(Nf=0)",
            1 => "SJ(Nf=1)",
            2 => "SJ(Nf=2)",
            _ => "Ising(J=0.45)",
        };
        let mut csv_full = String::from("N,matter_mode,swap_acc,min_pair_swap,beta,s_mean,s_err,dmm_mean,dmm_err,chi_s,cv,binder,tau_int,mat_mean\n");
        let mut csv_sum = String::from("N,matter_mode,swap_acc,min_pair_swap,beta_c_main,chi_peak_main,binder_min_main,dmm_b0,dmm_b0_err,tau_b0,s_bhi,s_bneg\n");
        for run in &runs {
            let lbl = mode_name(run.matter_mode);
            let r0 = &run.replicas[run.beta_zero_idx];
            let rhi = run.replicas.last().unwrap();
            let rneg = &run.replicas[0];
            println!(
                "  N={:2} {:<13}: swap={:.2} (min={:.2}), β_c={:.3} (χ={:6.1}, B4_min={:+.2}) | β=0: dMM={:.3}±{:.3} (τ={:.1})",
                run.n_elem, lbl, run.swap_acceptance, run.min_pair_swap, run.beta_c_main, run.chi_peak_main,
                run.binder_min_main, r0.dmm_mean, r0.dmm_err, r0.tau_int
            );
            csv_sum.push_str(&format!(
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4}\n",
                run.n_elem, lbl, run.swap_acceptance, run.min_pair_swap, run.beta_c_main, run.chi_peak_main,
                run.binder_min_main, r0.dmm_mean, r0.dmm_err, r0.tau_int, rhi.s_mean, rneg.s_mean
            ));
            for rep in &run.replicas {
                csv_full.push_str(&format!(
                    "{},{},{:.4},{:.4},{:.5},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.6}\n",
                    run.n_elem, lbl, run.swap_acceptance, run.min_pair_swap, rep.beta, rep.s_mean, rep.s_err,
                    rep.dmm_mean, rep.dmm_err, rep.chi_s, rep.cv, rep.binder, rep.tau_int, rep.mat_mean
                ));
            }
        }
        let mut json = String::from("{\n  \"via\": \"B-FSS\",\n  \"replicas\": 20,\n  \"sizes\": [20, 28, 36, 48, 64],\n  \"fits\": [\n");
        println!("  --- Ajustes FSS (N = 20..64) ---");
        for (idx, f) in fits.iter().enumerate() {
            let lbl = mode_name(f.matter_mode);
            println!(
                "  {:<13}: χ_pk ∝ N^{:.3} (R²={:.3}) | d_MM(∞)={:.4} (b={:+.2}/N) | β_c(∞)={:.4} | min_swap={:.2} | B4_min(64)={:+.3}",
                lbl, f.gamma_chi, f.r2_chi, f.dmm_inf, f.dmm_slope, f.beta_c_inf, f.min_swap_all_n, f.binder_min_largest
            );
            json.push_str(&format!(
                "    {{\"mode\": \"{}\", \"gamma_chi\": {:.4}, \"r2_chi\": {:.4}, \"dmm_inf\": {:.4}, \"dmm_slope\": {:.4}, \"beta_c_inf\": {:.4}, \"min_swap\": {:.4}, \"binder_min_64\": {:.4}}}{}\n",
                lbl, f.gamma_chi, f.r2_chi, f.dmm_inf, f.dmm_slope, f.beta_c_inf, f.min_swap_all_n, f.binder_min_largest,
                if idx + 1 < fits.len() { "," } else { "" }
            ));
        }
        json.push_str("  ]\n}\n");
        let _ = write_research_file("via_b_fss_20rep_full.csv", &csv_full);
        let _ = write_research_file("via_b_fss_20rep_summary.csv", &csv_sum);
        let _ = write_research_file("via_b_fss_fits.json", &json);
        println!("  Exportado datos_investigacion/via_b_fss_20rep_*.csv + via_b_fss_fits.json + checkpoints/ ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p19 || task == "via-c-scaling" {
        let t0 = Instant::now();
        println!("=== 21. [Fase 19 — Vía C escalado] Continuo v_IR → √(v_A v_B) + Calabrese–Cardy N_r = 8..32 (L = 64, χ = 56) + 3+1D Coulomb ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.via_c_continuum_and_scaling(true, 56);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        let n_sz = out[13] as usize;
        println!(
            "  Continuo (M_S·L = const, κ = 8,0): √(v_A v_B)={:.4} | v_inf_lin={:.4}, v_inf_quad={:.4} (R²={:.4})",
            out[0], out[1], out[2], out[3]
        );
        let mut csv_cont = String::from("N_r,J,vA_IR,vB_IR,mismatch\n");
        for k in 0..3 {
            let r = &out[(14 + 5 * k)..(14 + 5 * (k + 1))];
            println!("    N_r={:2}, J={:.4}: vA={:.4}, vB={:.4}, mis={:.5}", r[0] as usize, r[1], r[2], r[3], r[4]);
            csv_cont.push_str(&format!("{:.0},{:.6},{:.6},{:.6},{:.6}\n", r[0], r[1], r[2], r[3], r[4]));
        }
        println!(
            "  Calabrese–Cardy U(1)-DMRG (N_r = 8..32, χ = 56): J=0 exacto c_fit={:.3} (R²={:.4}, c_hi={:.3}) | J=2.50 simétrico (Δ_ef=1) c_fit={:.3} (R²={:.4}, c_hi={:.3})",
            out[4], out[5], out[6], out[7], out[8], out[9]
        );
        let mut csv_cft = String::from("N_r,L_sites,S_J0_exact,S_J25_sym_c1,S_J1_asym,S_J25_asym_neel\n");
        let base_s = 14 + 15;
        for k in 0..n_sz {
            let r = &out[(base_s + 5 * k)..(base_s + 5 * (k + 1))];
            println!(
                "    N_r={:2} (L={:2}): S_ex(J=0)={:.4} | S(J=2.5,vA=vB,Δ=1)={:.4} | S(J=1.0,vA≠vB)={:.4} | S(J=2.5,vA≠vB,Δ=1.336)={:.4}",
                r[0] as usize, 2 * r[0] as usize, r[1], r[2], r[3], r[4]
            );
            csv_cft.push_str(&format!("{:.0},{:.0},{:.6},{:.6},{:.6},{:.6}\n", r[0], 2.0 * r[0], r[1], r[2], r[3], r[4]));
        }
        println!(
            "  Contraste 3+1D Coulomb vs 1+1D Schwinger: mis_UV={:.4} → 3+1D QED RG (M_P→m_e)={:.4} (phot_fac={:.4}) vs 1+1D Schwinger={:.5}",
            out[10], out[11], out[12], out[14 + 10 + 4]
        );
        let json = format!(
            "{{\n  \"via\": \"C-scaling\",\n  \"v_geom\": {:.6},\n  \"v_inf_lin\": {:.6},\n  \"v_inf_quad\": {:.6},\n  \"r2_cont\": {:.6},\n  \"c_fit_j0\": {:.4},\n  \"r2_j0\": {:.5},\n  \"c_hi_j0\": {:.4},\n  \"c_fit_j1\": {:.4},\n  \"r2_j1\": {:.5},\n  \"c_hi_j1\": {:.4},\n  \"mis_uv\": {:.4},\n  \"mis_3d_coulomb_ir\": {:.4},\n  \"rg_factor_3d_phot\": {:.4},\n  \"mis_1d_schwinger_ir\": {:.6}\n}}\n",
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[14 + 10 + 4]
        );
        let _ = write_research_file("via_c_continuum_traj.csv", &csv_cont);
        let _ = write_research_file("via_c_cft_scaling.csv", &csv_cft);
        let _ = write_research_file("via_c_scaling_resumen.json", &json);
        println!("  Exportado datos_investigacion/via_c_continuum_traj.csv + via_c_cft_scaling.csv + via_c_scaling_resumen.json ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p20 || task == "via-a-yee4" {
        let t0 = Instant::now();
        println!("=== 22. [Fase 20 — Vía A Yee-4 S₄] Acoplamiento real 6³ (Iso-SC+FCC S₄ + Yee-4 S₄) + propagación polarizada 16³ ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.via_a_coupled_yee4_3d(0.25);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        println!(
            "  Malla 6³ acoplada: norm_drift={:.2e}, gauss={:.2e}, gauge_ρ={:.2e}, gauge_E={:.2e}",
            out[0], out[1], out[2], out[3]
        );
        println!(
            "  Malla 16³ real: Yee-2 c_ax={:.6}, c_dg={:.6} (Δ={:.2e}) | Yee-4 S₄ c_ax={:.6}, c_dg={:.6}, c_obl={:.6} (Δ={:.2e}, mejora {:.1}×)",
            out[4], out[5], (out[5] - out[4]).abs(), out[6], out[7], out[8], (out[7] - out[6]).abs(), out[12]
        );
        println!(
            "  Birrefringencia 45° k=(2π/16)(2,1,1), T=10: Yee-4 fuga={:.2e} vs Mlodinow–Brun Δθ={:.4} rad ({:.0}× menor), drift_E={:.2e}",
            out[9], out[10], out[10] / out[9].max(1e-16), out[11]
        );
        let json = format!(
            "{{\n  \"via\": \"A-Yee4\",\n  \"norm_drift\": {:.6e},\n  \"gauss_residual\": {:.6e},\n  \"gauge_dens_err\": {:.6e},\n  \"gauge_e_err\": {:.6e},\n  \"c_ax_yee2\": {:.8},\n  \"c_dg_yee2\": {:.8},\n  \"c_ax_yee4\": {:.8},\n  \"c_dg_yee4\": {:.8},\n  \"c_obl_yee4\": {:.8},\n  \"pol_leak_yee4_45deg\": {:.6e},\n  \"pol_rot_mlodinow_brun\": {:.6},\n  \"biref_suppression_ratio\": {:.2},\n  \"max_energy_drift_yee4\": {:.6e},\n  \"aniso_improvement_yee2_over_yee4\": {:.2}\n}}\n",
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8],
            out[9], out[10], out[10] / out[9].max(1e-16), out[11], out[12]
        );
        let _ = write_research_file("via_a_coupled_yee4.json", &json);
        println!("  Exportado datos_investigacion/via_a_coupled_yee4.json ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p20 || task == "via-b-wl" {
        let t0 = Instant::now();
        println!("=== 23. [Fase 20 — Vía B Wang–Landau + m/N_f] Reponderación continua sobre barrera 1.er orden + barrido masa/N_f ===");
        let wl_tasks: [(usize, usize, usize); 7] = [
            (20, 0, 5000),
            (28, 0, 5000),
            (36, 0, 4500),
            (48, 0, 4000),
            (20, 1, 4500),
            (28, 1, 4000),
            (36, 1, 3500),
        ];
        let wl_runs: Vec<_> = std::thread::scope(|sc| {
            let hs: Vec<_> = wl_tasks
                .into_iter()
                .enumerate()
                .map(|(i, (n, nf, st))| {
                    sc.spawn(move || run_via_b_wang_landau_2d(n, nf, 0.80, st, 0x20B0_7777 ^ (i as u64 * 104729)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut csv_wl = String::from("N,Nf,m_scalar,beta_c_wl,chi_peak_wl,binder_min_wl,latent_heat,barrier_df,dmm_b0,visited_frac\n");
        for w in &wl_runs {
            println!(
                "  WL N={:2}, Nf={}: β_c={:.3}, χ_pk={:6.1}, B4_min={:+.3}, Δs={:.2}, ΔF={:.2}, d_MM(0)={:.3}, vis={:.2}",
                w.n_elem, w.nf, w.beta_c_wl, w.chi_peak_wl, w.binder_min_wl, w.latent_heat, w.barrier_df, w.dmm_b0, w.visited_frac
            );
            csv_wl.push_str(&format!(
                "{},{},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                w.n_elem, w.nf, w.m_scalar, w.beta_c_wl, w.chi_peak_wl, w.binder_min_wl, w.latent_heat, w.barrier_df, w.dmm_b0, w.visited_frac
            ));
        }
        let sweep = run_via_b_matter_mass_nf_sweep(36, 900, 0x20B0_9999);
        let mut csv_sw = String::from("N,Nf,m_scalar,dmm_mean,dmm_err,tau_int\n");
        println!("  --- Barrido de masa m y especies N_f = 0..8 en β = 0 (N = 36) ---");
        for r in &sweep {
            println!(
                "  Nf={:.0}, m={:.2}: d_MM={:.4}±{:.4} (τ_int={:.2})",
                r[0], r[1], r[2], r[3], r[4]
            );
            csv_sw.push_str(&format!("36,{:.0},{:.2},{:.6},{:.6},{:.4}\n", r[0], r[1], r[2], r[3], r[4]));
        }
        let _ = write_research_file("via_b_wl_multicanonical.csv", &csv_wl);
        let _ = write_research_file("via_b_matter_mass_nf.csv", &csv_sw);
        println!("  Exportado datos_investigacion/via_b_wl_multicanonical.csv + via_b_matter_mass_nf.csv ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if run_p20 || task == "via-c-tdvp" {
        let t0 = Instant::now();
        println!("=== 24. [Fase 20 — Vía C TDVP + BKT] 2-site U(1)-TDVP en tiempo real (L = 32 sitios, 2³² estados) + diagrama BKT ===");
        let mut u2 = QuantumSpacetimeUniverse::new(16, 16, 1, 0.5, 0.1, 1.0);
        let p = u2.via_c_tdvp_and_bkt_diagram(16, 36);
        let len = u2.open_quantum_out_len() as usize;
        let out = unsafe { std::slice::from_raw_parts(p, len) };
        println!(
            "  2-site U(1)-TDVP (N_r = {:.0}, L = {:.0}, χ = 36): J=0 vA={:.4}, vB={:.4} (mis={:.3}) | J=0,50 vA={:.4}, vB={:.4} (mis={:.3}) | drift_E={:.2e}",
            out[0], 2.0 * out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]
        );
        let n_bkt = out[8] as usize;
        let mut csv_bkt = String::from("vB_over_vA,J,Delta_eff,S16,c_eff_16_24,C_neel_d4\n");
        println!("  --- Diagrama de fases BKT (N_r = 16 → 24, χ = 36) ---");
        for i in 0..n_bkt {
            let r = &out[(9 + 6 * i)..(9 + 6 * (i + 1))];
            println!(
                "  vB/vA={:.2}, J={:.2} (Δ_ef={:.3}): S16={:.4}, c_eff(16→24)={:+.3}, C_Néel(d=4)={:+.4}",
                r[0], r[1], r[2], r[3], r[4], r[5]
            );
            csv_bkt.push_str(&format!("{:.4},{:.4},{:.6},{:.6},{:.6},{:.6}\n", r[0], r[1], r[2], r[3], r[4], r[5]));
        }
        let json = format!(
            "{{\n  \"via\": \"C-TDVP-BKT\",\n  \"nr_tdvp\": {:.0},\n  \"l_sites\": {:.0},\n  \"chi_tdvp\": 36,\n  \"vA_LC_J0\": {:.6},\n  \"vB_LC_J0\": {:.6},\n  \"mismatch_LC_J0\": {:.6},\n  \"vA_LC_J050\": {:.6},\n  \"vB_LC_J050\": {:.6},\n  \"mismatch_LC_J050\": {:.6},\n  \"max_energy_drift\": {:.6e}\n}}\n",
            out[0], 2.0 * out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]
        );
        let _ = write_research_file("via_c_tdvp_bkt.csv", &csv_bkt);
        let _ = write_research_file("via_c_tdvp_resumen.json", &json);
        println!("  Exportado datos_investigacion/via_c_tdvp_bkt.csv + via_c_tdvp_resumen.json ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if task == "pub-checks" || task == "via-c-chi-conv" {
        let t0 = Instant::now();
        println!("=== 25. [Auditoría publicación — Vía C] Convergencia en dimensión de enlace χ ∈ {{24, 32, 44, 56}} (J = 2,50, N_r = 16 → 24) ===");
        let rows = run_via_c_chi_convergence();
        let mut csv = String::from("vB_over_vA,J,chi,E24,S16,c_eff_16_24,C_neel_d4\n");
        for r in &rows {
            println!(
                "  vB/vA={:.2}, J={:.2}, χ={:2.0}: E24={:+.6}, S16={:.5}, c_eff(16→24)={:+.4}, C_Néel(d=4)={:+.5}",
                r[0], r[1], r[2], r[3], r[4], r[5], r[6]
            );
            csv.push_str(&format!("{:.4},{:.4},{:.0},{:.6},{:.6},{:.6},{:.6}\n", r[0], r[1], r[2], r[3], r[4], r[5], r[6]));
        }
        let _ = write_research_file("via_c_chi_convergence.csv", &csv);
        println!("  Exportado datos_investigacion/via_c_chi_convergence.csv ({:.2} s)", t0.elapsed().as_secs_f64());
    }

    if task == "pub-checks" || task == "via-b-multiseed" {
        let t0 = Instant::now();
        println!("=== 26. [Auditoría publicación — Vía B] Wang–Landau multi-semilla (5 semillas × 6 puntos) + escalado gran N_f (N = 24, 36, 48) ===");
        let wl_specs: [(usize, usize, usize); 6] = [
            (20, 0, 4200),
            (28, 0, 4200),
            (36, 0, 3800),
            (20, 1, 3800),
            (28, 1, 3500),
            (36, 1, 3000),
        ];
        let n_seeds = 5usize;
        let mut wl_jobs = Vec::new();
        for (gi, &(n, nf, st)) in wl_specs.iter().enumerate() {
            for s in 0..n_seeds {
                let seed = 0x20B5_0001u64 ^ ((gi as u64 + 1) * 104729) ^ ((s as u64 + 1) * 0x9E37_79B9_7F4A_7C15);
                wl_jobs.push((gi, n, nf, st, seed));
            }
        }
        let wl_all: Vec<_> = std::thread::scope(|sc| {
            let hs: Vec<_> = wl_jobs
                .into_iter()
                .map(|(gi, n, nf, st, sd)| sc.spawn(move || (gi, run_via_b_wang_landau_2d(n, nf, 0.80, st, sd))))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut csv_ms = String::from("N,Nf,n_seeds,beta_c_mean,beta_c_sem,chi_mean,chi_sem,ds_mean,ds_sem,df_mean,df_sem,dmm0_mean,dmm0_sem,vis_mean\n");
        let mean_sem = |xs: &[f64]| -> (f64, f64) {
            let k = xs.len() as f64;
            let m = xs.iter().sum::<f64>() / k;
            let var = if xs.len() > 1 {
                xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (k - 1.0)
            } else {
                0.0
            };
            (m, (var / k).sqrt())
        };
        for (gi, &(n, nf, _)) in wl_specs.iter().enumerate() {
            let sub: Vec<_> = wl_all.iter().filter(|(i, _)| *i == gi).map(|(_, w)| w).collect();
            let (bc_m, bc_e) = mean_sem(&sub.iter().map(|w| w.beta_c_wl).collect::<Vec<_>>());
            let (ch_m, ch_e) = mean_sem(&sub.iter().map(|w| w.chi_peak_wl).collect::<Vec<_>>());
            let (ds_m, ds_e) = mean_sem(&sub.iter().map(|w| w.latent_heat).collect::<Vec<_>>());
            let (df_m, df_e) = mean_sem(&sub.iter().map(|w| w.barrier_df).collect::<Vec<_>>());
            let (d0_m, d0_e) = mean_sem(&sub.iter().map(|w| w.dmm_b0).collect::<Vec<_>>());
            let (vs_m, _) = mean_sem(&sub.iter().map(|w| w.visited_frac).collect::<Vec<_>>());
            println!(
                "  WL 5-seed N={:2}, Nf={}: β_c={:.4}±{:.4}, χ={:6.1}±{:4.1}, Δs={:.2}±{:.2}, ΔF={:.2}±{:.2}, d_MM(0)={:.3}±{:.3}, vis={:.0}%",
                n, nf, bc_m, bc_e, ch_m, ch_e, ds_m, ds_e, df_m, df_e, d0_m, d0_e, 100.0 * vs_m
            );
            csv_ms.push_str(&format!(
                "{},{},{},{:.5},{:.5},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                n, nf, n_seeds, bc_m, bc_e, ch_m, ch_e, ds_m, ds_e, df_m, df_e, d0_m, d0_e, vs_m
            ));
        }

        // Escalado de tamaño finito de la transición a gran N_f ∈ {0, 2, 4, 6, 8} en N ∈ {24, 36, 48} (β = 0, m = 0,80)
        let mut nf_jobs = Vec::new();
        for &n in &[24usize, 36, 48] {
            let sw = if n <= 24 { 1000usize } else if n <= 36 { 800 } else { 550 };
            for &nf in &[0usize, 2, 4, 6, 8] {
                nf_jobs.push((n, nf, sw));
            }
        }
        let nf_rows: Vec<[f64; 5]> = std::thread::scope(|sc| {
            let hs: Vec<_> = nf_jobs
                .into_iter()
                .enumerate()
                .map(|(i, (n, nf, sw))| {
                    sc.spawn(move || run_via_b_large_nf_point(n, nf, 0.80, sw, 0x20B6_1111 ^ (i as u64 * 7919)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut csv_nf = String::from("N,Nf,m_scalar,dmm_mean,dmm_err,tau_int\n");
        println!("  --- Escalado finito de la transición a gran N_f (β = 0, m = 0,80) ---");
        for r in &nf_rows {
            println!("  N={:2.0}, Nf={:.0}: d_MM={:.4}±{:.4} (τ_int={:.2})", r[0], r[1], r[2], r[3], r[4]);
            csv_nf.push_str(&format!("{:.0},{:.0},0.80,{:.6},{:.6},{:.4}\n", r[0], r[1], r[2], r[3], r[4]));
        }
        let _ = write_research_file("via_b_wl_multiseed.csv", &csv_ms);
        let _ = write_research_file("via_b_large_nf_fss.csv", &csv_nf);
        println!("  Exportado datos_investigacion/via_b_wl_multiseed.csv + via_b_large_nf_fss.csv ({:.2} s)", t0.elapsed().as_secs_f64());
    }
}
