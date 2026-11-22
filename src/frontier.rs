//! Fase 15 — Problemas abiertos de frontera en QFT discreta (§8.6, §8.10 y §8.11):
//!
//! 1. **Violación de Lorentz radiativa a un lazo (Collins, Perez, Sudarsky, Urrutia y Vucetich,
//!    PRL 93, 191301, 2004):** calcula la autoenergía exacta a un lazo de un fermión acoplado por
//!    Yukawa a un escalar pesado en 1+1D comparando:
//!      - regulador continuo invariante de Lorentz (control nulo: δc² → 0 al crecer el corte Λ);
//!      - regulador discreto de red / autómata (aun siendo palindrómico y sin violación de orden 1
//!        a nivel árbol, los bucles UV generan δc² ∝ g² de dimensión 4 NO suprimido por E/E_P!).
//! 2. **Búsqueda sistemática de caminatas 3D de tipo luz sin dobladores (§8.6):**
//!    escanea toda la zona de Brillouin 3D [−π, π)³ de caminatas de pasos divididos / tipo
//!    Gupta–Short en 3D buscando ceros espurios fuera de k = 0 y cuantificando la obstrucción
//!    frente a los fermiones de Wilson–Dirac.
//! 3. **Aprendizaje del mapa discreto multicomponente de Floquet (§8.10):**
//!    aprende directamente de trayectorias el mapa lineal de un paso (4 componentes × 3 sitios) y
//!    la ley de fase no lineal de la tira 1D, y comprueba que el mapa discreto aprendido SÍ predice
//!    la resonancia paramétrica de Floquet con g < 0 (donde la EDP continua aprendida en Fase 12
//!    daba 0).
//! 4. **Fermiones CPT-impares y umbral de Cohen–Glashow para neutrinos de ultra-alta energía (§8.11):**
//!    extiende el análisis de la Fase 13 de fotones a fermiones (Weyl de orden fijo frente a
//!    palindrómicos) y calcula el umbral de emisión de pares ν → ν e⁺ e⁻ frente a neutrinos de
//!    IceCube / KM3NeT.

use crate::analog::{bogoliubov_block, focusing_band, parametric_rate_impl};
use crate::linalg::eigvals;
use crate::rng::SplitMix64;
use crate::spinor::{cx, pauli_eigenbasis};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

type C = Complex<f64>;

// ------------------------------------------------------------------ 1. Collins et al. (2004) a un lazo

/// Autoenergía exacta de teoría de perturbaciones ordenada en el tiempo (un lazo completo en 1+1D)
/// para un fermión de masa `mf` acoplado por Yukawa `g_y` a un escalar de masa `m_phi`:
///   Δ(E²)(k) = g_y² ∫ (dq/2π) [E² E' + (M(k) M(k−q) − S(k) S(k−q))(E' + Ω)] / [Ω E' (E² − (E' + Ω)²)]
/// donde H_f(k) = S(k) σ_z + M(k) σ_x, E = √(S² + M²).
/// Con regulador continuo (S(k) = k, M(k) = m_f, Ω(q) = √(q² + M_φ²), |q| ≤ Λ):
///   c_ren² = ½ d²(E_tot²)/dk²|_{k=0} → 1 exactamente cuando Λ → ∞ (invariancia de Lorentz).
/// Con regulador de red (S(k) = sin k, M(k) = m_f + r(1 − cos k), Ω(q) = √(4 sin²(q/2) + M_φ²), q ∈ [−π, π)):
///   los modos UV de la zona de Brillouin rompen los boosts y generan δc² = c_ren² − 1 = O(g_y²) ≠ 0
///   de dimensión 4, NO suprimido por ninguna potencia de E/E_P (Collins et al., PRL 2004).
fn one_loop_delta_e2(k: f64, mf: f64, m_phi: f64, g_y: f64, lattice: bool, wilson_r: f64, q_max: f64, nq: usize) -> f64 {
    let disp_f = |p: f64| -> (f64, f64, f64) {
        if lattice {
            let s = p.sin();
            let m = mf + wilson_r * (1.0 - p.cos());
            (s, m, s.hypot(m))
        } else {
            (p, mf, p.hypot(mf))
        }
    };
    let disp_b = |q: f64| -> f64 {
        if lattice {
            let sh = 2.0 * (0.5 * q).sin();
            sh.hypot(m_phi)
        } else {
            q.hypot(m_phi)
        }
    };
    let (sk, mk, ek) = disp_f(k);
    let dq = 2.0 * q_max / nq as f64;
    let mut integral = 0.0;
    for j in 0..nq {
        let q = -q_max + (j as f64 + 0.5) * dq;
        let (sp, mp, ep) = disp_f(k - q);
        let om = disp_b(q);
        // Numerador de la suma exacta de los diagramas hacia delante y Z (mar de Dirac):
        // |⟨+k'|σ_x|+k⟩|² = (E E' + M M' − S S') / (2 E E')
        let b_term = mk * mp - sk * sp;
        let num = ek * ek * ep + b_term * (ep + om);
        let den = om * ep * (ek * ek - (ep + om) * (ep + om));
        integral += num / den;
    }
    g_y * g_y * integral * dq / (2.0 * PI)
}

/// Calcula la corrección radiativa δc² = c_ren² − 1 a un lazo en:
/// 1) continuo con corte Λ = 80 (control nulo: debe ser ≈ 0);
/// 2) red de Wilson–Dirac / autómata discreto (q ∈ [−π, π)): δc² ≠ 0 de orden g_y²;
/// 3) segunda especie con acoplamiento g_2 = 0 (espectador: δc² = 0) → diferencia Δc entre especies!
pub fn compute_collins_one_loop_lv(mf: f64, m_phi: f64, g_y: f64, wilson_r: f64) -> Vec<f64> {
    let dk = 1e-3;
    let d2_e2 = |lattice: bool, q_max: f64, nq: usize| -> (f64, f64) {
        let f0 = one_loop_delta_e2(0.0, mf, m_phi, g_y, lattice, wilson_r, q_max, nq);
        let fp = one_loop_delta_e2(dk, mf, m_phi, g_y, lattice, wilson_r, q_max, nq);
        let fm = one_loop_delta_e2(-dk, mf, m_phi, g_y, lattice, wilson_r, q_max, nq);
        // ½ d²(ΔE²)/dk²|_{k=0} = δ(c_ren²)
        let dc2 = 0.5 * (fp - 2.0 * f0 + fm) / (dk * dk);
        (dc2, f0)
    };
    let (dc2_cont_mid, _) = d2_e2(false, 20.0, 2048);
    let (dc2_cont_hi, dm2_cont) = d2_e2(false, 120.0, 6144);
    let (dc2_lat, dm2_lat) = d2_e2(true, PI, 1024);
    // También probamos con r = 0 (fermión ingenuo con doblador en π) para comparar
    let f0_r0 = one_loop_delta_e2(0.0, mf, m_phi, g_y, true, 0.0, PI, 1024);
    let fp_r0 = one_loop_delta_e2(dk, mf, m_phi, g_y, true, 0.0, PI, 1024);
    let fm_r0 = one_loop_delta_e2(-dk, mf, m_phi, g_y, true, 0.0, PI, 1024);
    let dc2_lat_r0 = 0.5 * (fp_r0 - 2.0 * f0_r0 + fm_r0) / (dk * dk);
    let c_ren_lat = (1.0 + dc2_lat).max(0.0).sqrt();
    let delta_c_species = (c_ren_lat - 1.0).abs();
    // [dc2_cont_hi, dc2_cont_mid, dc2_lat, dc2_lat_r0, delta_c_species, dm2_cont, dm2_lat]
    vec![dc2_cont_hi, dc2_cont_mid, dc2_lat, dc2_lat_r0, delta_c_species, dm2_cont, dm2_lat]
}

// ------------------------------------------------------------------ 2. Escáner 3D de caminatas de tipo luz sin dobladores (§8.6)

type M2 = [[C; 2]; 2];

#[inline(always)]
fn m2_mul(a: &M2, b: &M2) -> M2 {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

/// Paso de un eje de Gupta–Short en 1D generalizado a un eje de Pauli `axis ∈ {0, 1, 2}`:
/// S_a(k_a, θ_a) = [Π_{a,+}(θ) e^{−ik_a} + Π_{a,−}(θ)] · [Π_{b,+}(−θ) + Π_{b,−}(−θ) e^{+ik_a}],
/// en la base propia de σ_a. Con θ = 0 se reduce al salto de Weyl estándar exp(−i k_a σ_a).
fn lazy_axis_step(axis: usize, k_a: f64, theta: f64) -> M2 {
    let ket = |t: f64| [cx((0.5 * t).cos(), 0.0), cx(0.0, -(0.5 * t).sin())];
    let a = ket(theta);
    let b = ket(-theta);
    let proj_shift = |v: [C; 2], ph_p: C, ph_m: C| -> M2 {
        // ph_p |v⟩⟨v| + ph_m (I − |v⟩⟨v|) = ph_m I + (ph_p − ph_m) |v⟩⟨v|
        let d = ph_p - ph_m;
        [
            [ph_m + d * v[0] * v[0].conj(), d * v[0] * v[1].conj()],
            [d * v[1] * v[0].conj(), ph_m + d * v[1] * v[1].conj()],
        ]
    };
    let e_m = C::from_polar(1.0, -k_a);
    let e_p = C::from_polar(1.0, k_a);
    let op_a = proj_shift(a, e_m, cx(1.0, 0.0));
    let op_b = proj_shift(b, cx(1.0, 0.0), e_p);
    let diag_step = m2_mul(&op_a, &op_b);
    // Rotar de la base propia de σ_a a la base estándar: R_a · diag_step · R_a†
    let r = pauli_eigenbasis(axis);
    let rd = [[r[0][0].conj(), r[1][0].conj()], [r[0][1].conj(), r[1][1].conj()]];
    m2_mul(&r, &m2_mul(&diag_step, &rd))
}

/// Escanea la zona de Brillouin 3D [−π, π)³ de una caminata 3D de tipo luz (2 componentes de Weyl
/// con ángulos de Gupta–Short `(tx, ty, tz)` y variante `alt`: false = orden fijo xyz, true =
/// palindrómica xyz·zyx de 2 pasos).
/// Busca la mínima cuasi-energía |ε(k)| fuera del cono físico |k| < 0,35 (dobladores / ceros espurios).
/// Devuelve [eps_min_fuera_cono, kx_min, ky_min, kz_min, velocidad_cono_c, num_minimos_bajos].
pub fn scan_3d_lightlike_walk_impl(tx: f64, ty: f64, tz: f64, alt: bool, grid_n: usize) -> Vec<f64> {
    let ng = grid_n.clamp(12, 48);
    let step_u = |kx: f64, ky: f64, kz: f64| -> f64 {
        let ux = lazy_axis_step(0, kx, tx);
        let uy = lazy_axis_step(1, ky, ty);
        let uz = lazy_axis_step(2, kz, tz);
        let u1 = m2_mul(&uz, &m2_mul(&uy, &ux));
        let u = if alt {
            let u2 = m2_mul(&ux, &m2_mul(&uy, &uz));
            m2_mul(&u2, &u1)
        } else {
            u1
        };
        let tr = 0.5 * (u[0][0] + u[1][1]);
        // Cuasi-energía por paso en [0, π]: con alt (2 pasos), dividimos entre 2
        let e = tr.re.clamp(-1.0, 1.0).acos();
        if alt { 0.5 * e } else { e }
    };

    // Velocidad del cono físico en k = (0,02, 0, 0)
    let c_cone = step_u(0.02, 0.0, 0.0) / 0.02;
    let mut eps_min = f64::INFINITY;
    let mut k_best = [0.0; 3];
    let mut n_low = 0usize;
    let dk = 2.0 * PI / ng as f64;
    for ix in 0..ng {
        let kx = -PI + ix as f64 * dk;
        for iy in 0..ng {
            let ky = -PI + iy as f64 * dk;
            for iz in 0..ng {
                let kz = -PI + iz as f64 * dk;
                let kn = (kx * kx + ky * ky + kz * kz).sqrt();
                if kn < 0.35 {
                    continue;
                }
                let e = step_u(kx, ky, kz);
                if e < 0.12 {
                    n_low += 1;
                }
                if e < eps_min {
                    eps_min = e;
                    k_best = [kx, ky, kz];
                }
            }
        }
    }
    // Refinamiento local en 4 niveles alrededor de `k_best` para distinguir un hueco real de un
    // nodo de Weyl/doblador que haya caído entre dos puntos de la rejilla gruesa.
    let mut step = dk;
    for _ in 0..4 {
        let center = k_best;
        let sub = step / 3.0;
        for dx in -3..=3 {
            for dy in -3..=3 {
                for dz in -3..=3 {
                    let kx = center[0] + dx as f64 * sub;
                    let ky = center[1] + dy as f64 * sub;
                    let kz = center[2] + dz as f64 * sub;
                    // Envolver a [−π, π)
                    let wx = (kx + PI).rem_euclid(2.0 * PI) - PI;
                    let wy = (ky + PI).rem_euclid(2.0 * PI) - PI;
                    let wz = (kz + PI).rem_euclid(2.0 * PI) - PI;
                    if (wx * wx + wy * wy + wz * wz).sqrt() < 0.35 {
                        continue;
                    }
                    let e = step_u(wx, wy, wz);
                    if e < eps_min {
                        eps_min = e;
                        k_best = [wx, wy, wz];
                    }
                }
            }
        }
        step = sub;
    }
    vec![eps_min, k_best[0], k_best[1], k_best[2], c_cone, n_low as f64]
}

/// Barrido sistemático en el espacio de parámetros θ = (θ_x, θ_y, θ_z) para caminatas 3D de tipo luz
/// (usa paralelismo nativo con `std::thread::scope` fuera de wasm32).
/// Devuelve [mejor_hueco_doblador, best_tx, best_ty, best_tz, c_cono_mejor, hueco_estandar_0, hueco_gs_simetrico].
pub fn search_3d_lightlike_obstruction(grid_n: usize) -> Vec<f64> {
    let std_w = scan_3d_lightlike_walk_impl(0.0, 0.0, 0.0, false, grid_n);
    let gs_sym = scan_3d_lightlike_walk_impl(0.4, 0.4, 0.4, false, grid_n);
    let vals = [0.0, 0.25, 0.5, 0.75, 1.0];
    let mut candidates = Vec::new();
    for &tx in &vals {
        for &ty in &vals {
            for &tz in &vals {
                if tx == 0.0 && ty == 0.0 && tz == 0.0 {
                    continue;
                }
                candidates.push((tx, ty, tz));
            }
        }
    }
    let eval_cand = |(tx, ty, tz): (f64, f64, f64)| -> (f64, f64, f64, f64, f64) {
        let r = scan_3d_lightlike_walk_impl(tx, ty, tz, false, grid_n);
        (r[0], tx, ty, tz, r[4])
    };

    #[cfg(not(target_arch = "wasm32"))]
    let results: Vec<(f64, f64, f64, f64, f64)> = {
        let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);
        let chunk = candidates.len().div_ceil(n_threads);
        std::thread::scope(|s| {
            let handles: Vec<_> = candidates
                .chunks(chunk)
                .map(|sl| s.spawn(move || sl.iter().copied().map(eval_cand).collect::<Vec<_>>()))
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        })
    };
    #[cfg(target_arch = "wasm32")]
    let results: Vec<(f64, f64, f64, f64, f64)> = candidates.into_iter().map(eval_cand).collect();

    let mut best = (-1.0f64, 0.0, 0.0, 0.0, 0.0);
    for r in results {
        // Exigimos un cono físico no degenerado (c > 0,25)
        if r.4 > 0.25 && r.0 > best.0 {
            best = r;
        }
    }
    vec![best.0, best.1, best.2, best.3, best.4, std_w[0], gs_sym[0]]
}

// ------------------------------------------------------------------ 3. Aprendizaje del mapa discreto de Floquet (§8.10)

/// Resuelve sistema lineal complejo A · x = b (n × n) por eliminación gaussiana con pivoteo parcial.
fn solve_complex(mut a: Vec<C>, mut b: Vec<C>, n: usize) -> Vec<C> {
    for p in 0..n {
        let mut piv = p;
        let mut best = a[p * n + p].norm_sqr();
        for r in (p + 1)..n {
            let v = a[r * n + p].norm_sqr();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if piv != p {
            for c in p..n {
                a.swap(p * n + c, piv * n + c);
            }
            b.swap(p, piv);
        }
        let d = a[p * n + p];
        if d.norm_sqr() < 1e-28 {
            continue;
        }
        for r in (p + 1)..n {
            let f = a[r * n + p] / d;
            for c in p..n {
                let v = a[p * n + c];
                a[r * n + c] -= f * v;
            }
            let bp = b[p];
            b[r] -= f * bp;
        }
    }
    let mut x = vec![cx(0.0, 0.0); n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        let d = a[i * n + i];
        x[i] = if d.norm_sqr() > 1e-28 { s / d } else { cx(0.0, 0.0) };
    }
    x
}

/// Resuelve sistema lineal real simétrico A · x = b (n × n).
fn solve_real(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for p in 0..n {
        let mut piv = p;
        for r in (p + 1)..n {
            if a[r * n + p].abs() > a[piv * n + p].abs() {
                piv = r;
            }
        }
        if piv != p {
            for c in p..n {
                a.swap(p * n + c, piv * n + c);
            }
            b.swap(p, piv);
        }
        let d = a[p * n + p];
        if d.abs() < 1e-28 {
            continue;
        }
        for r in (p + 1)..n {
            let f = a[r * n + p] / d;
            for c in p..n {
                a[r * n + c] -= f * a[p * n + c];
            }
            b[r] -= f * b[p];
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = if a[i * n + i].abs() > 1e-28 { s / a[i * n + i] } else { 0.0 };
    }
    x
}

/// Fase 15 (§8.10): aprende directamente de las trayectorias de una tira 1D el **mapa discreto
/// multicomponente de un paso** ψ(x, t+1) = S_aprendido · C_aprendido · exp(i θ_aprendido(E_x)) · ψ(x, t),
/// y evalúa su inestabilidad paramétrica de Floquet en (g, ρ̄) comparándola con la caminata exacta
/// (la EDP continua aprendida en Fase 12 daba 0).
/// Devuelve [tasa_mapa_aprendido, tasa_caminata_exacta, q_aprendido, q_exacto, err_moneda_aprendida, err_theta_prima, tasa_edp_continua].
pub fn run_discover_discrete_floquet_map(m: f64, g: f64, rho_bar: f64, seed: u64) -> Vec<f64> {
    const W: usize = 256;
    let emax = 1.0;
    let mut rng = SplitMix64::new(seed ^ 0xF100_E715);
    let (cm, sm) = (m.cos(), m.sin());
    let isn = cx(0.0, sm);

    let step_exact = |psi: &[[C; 4]]| -> Vec<[C; 4]> {
        let mut tmp = vec![[cx(0.0, 0.0); 4]; W];
        let mut out = vec![[cx(0.0, 0.0); 4]; W];
        for x in 0..W {
            let e: f64 = psi[x].iter().map(|z| z.norm_sqr()).sum();
            let eff = e / (1.0 + e / emax);
            let rot = C::from_polar(1.0, g * eff);
            let p = psi[x];
            let s = (p[0] + p[1] + p[2] + p[3]) * 0.5;
            for c in 0..4 {
                tmp[x][c] = (p[c] * cm - isn * (s - p[c])) * rot;
            }
        }
        for x in 0..W {
            let xp = if x + 1 == W { 0 } else { x + 1 };
            let xm = if x == 0 { W - 1 } else { x - 1 };
            out[x] = [tmp[x][0], tmp[x][1], tmp[xp][2], tmp[xm][3]];
        }
        out
    };

    // Paso 1: datos de entrenamiento en régimen débil (amplitud aleatoria pequeña) para aprender el
    // mapa lineal exacto de 1 paso: 12 entradas locales [ψ(x−1), ψ(x), ψ(x+1)] → 4 salidas ψ(x, t+1)
    let mut psi_lin = vec![[cx(0.0, 0.0); 4]; W];
    for p in psi_lin.iter_mut() {
        for c in 0..4 {
            p[c] = cx(rng.gauss(), rng.gauss()) * 1e-4;
        }
    }
    let psi_lin_next = step_exact(&psi_lin);
    // Ecuaciones normales (12 × 12) para cada canal de salida c ∈ 0..4
    let mut gram = vec![cx(0.0, 0.0); 12 * 12];
    let mut rhs = vec![vec![cx(0.0, 0.0); 12]; 4];
    for x in 0..W {
        let xm = if x == 0 { W - 1 } else { x - 1 };
        let xp = if x + 1 == W { 0 } else { x + 1 };
        let mut feat = [cx(0.0, 0.0); 12];
        feat[0..4].copy_from_slice(&psi_lin[xm]);
        feat[4..8].copy_from_slice(&psi_lin[x]);
        feat[8..12].copy_from_slice(&psi_lin[xp]);
        for a in 0..12 {
            for b in 0..12 {
                gram[a * 12 + b] += feat[a].conj() * feat[b];
            }
            for c in 0..4 {
                rhs[c][a] += feat[a].conj() * psi_lin_next[x][c];
            }
        }
    }
    let mut lin_rows = [[cx(0.0, 0.0); 12]; 4];
    for c in 0..4 {
        let sol = solve_complex(gram.clone(), rhs[c].clone(), 12);
        lin_rows[c].copy_from_slice(&sol);
    }
    // Extraemos la matriz de moneda aprendida C_learned (4 × 4) en el bloque correspondiente a cada
    // canal desplazado (c=0,1 vienen de x [4..8]; c=2 viene de x+1 [8..12]; c=3 viene de x−1 [0..4])
    let mut c_learned = vec![cx(0.0, 0.0); 16];
    let shift_slot = [4usize, 4, 8, 0];
    for r in 0..4 {
        let off = shift_slot[r];
        for col in 0..4 {
            c_learned[r * 4 + col] = lin_rows[r][off + col];
        }
    }
    // Error respecto a la moneda exacta
    let mut coin_err = 0.0f64;
    for r in 0..4 {
        for col in 0..4 {
            let exact = if r == col { cx(cm, 0.5 * sm) } else { cx(0.0, -0.5 * sm) };
            coin_err = coin_err.max((c_learned[r * 4 + col] - exact).norm());
        }
    }

    // Paso 2: datos de entrenamiento a densidad finita (E ∈ [0.05, 1.2]) para aprender la ley de
    // fase no lineal θ(E) = a1·E + a2·E² + a3·E³ + a4·E⁴ directamente desde arg(ψ_obs / ψ_lin)
    let mut psi_nl = vec![[cx(0.0, 0.0); 4]; W];
    for x in 0..W {
        let target_e = 0.05 + 1.15 * (x as f64 / W as f64);
        let mut e = 0.0;
        for c in 0..4 {
            psi_nl[x][c] = cx(rng.gauss(), rng.gauss());
            e += psi_nl[x][c].norm_sqr();
        }
        let sc = (target_e / e.max(1e-30)).sqrt();
        for c in 0..4 {
            psi_nl[x][c] *= sc;
        }
    }
    let psi_nl_next = step_exact(&psi_nl);
    // Aplicamos el mapa lineal aprendido sobre psi_nl y medimos la fase residual en cada celda x
    // (usando los canales 0 y 1 que no se desplazan en x, así la fase corresponde exactamente a E_x!)
    let mut g_poly = vec![0.0f64; 16];
    let mut h_poly = vec![0.0f64; 4];
    for x in 0..W {
        let e: f64 = psi_nl[x].iter().map(|z| z.norm_sqr()).sum();
        let mut pred0 = cx(0.0, 0.0);
        let mut pred1 = cx(0.0, 0.0);
        for col in 0..4 {
            pred0 += c_learned[col] * psi_nl[x][col];
            pred1 += c_learned[4 + col] * psi_nl[x][col];
        }
        let ov = pred0.conj() * psi_nl_next[x][0] + pred1.conj() * psi_nl_next[x][1];
        let th_obs = ov.arg();
        let b = [e, e * e, e * e * e, e * e * e * e];
        for r in 0..4 {
            h_poly[r] += b[r] * th_obs;
            for c in 0..4 {
                g_poly[r * 4 + c] += b[r] * b[c];
            }
        }
    }
    let coef = solve_real(g_poly, h_poly, 4);
    // Derivada aprendida θ′(ρ̄) = a1 + 2 a2 ρ̄ + 3 a3 ρ̄² + 4 a4 ρ̄³
    let th_prime_learned = coef[0] + 2.0 * coef[1] * rho_bar + 3.0 * coef[2] * rho_bar * rho_bar + 4.0 * coef[3] * rho_bar.powi(3);
    let th_prime_exact = g / (1.0 + rho_bar / emax).powi(2);
    let err_thp = (th_prime_learned / th_prime_exact - 1.0).abs();

    // Paso 3: análisis de Floquet–Bogoliubov del MAPA DISCRETO APRENDIDO (C_learned + θ′_learned)
    let (u1, lam0) = focusing_band(m, 1, 0.0);
    let u: Vec<C> = u1.iter().map(|z| *z * rho_bar.sqrt()).collect();
    let w0 = -lam0.arg();
    let nq = 512usize;
    let mut max_rate_learned = 0.0f64;
    let mut q_learned = 0.0f64;
    for j in 0..(nq / 2) {
        let q = 2.0 * PI * (j as f64 + 0.5) / nq as f64;
        let blk = bogoliubov_block(&c_learned, &u, w0, th_prime_learned, [q, 0.0, 0.0], [-q, 0.0, 0.0], 1);
        for ev in eigvals(blk, 8) {
            let r = ev.norm().ln();
            if r > max_rate_learned {
                max_rate_learned = r;
                q_learned = q;
            }
        }
    }
    let (exact_rate, q_ex) = parametric_rate_impl(m, g, rho_bar, emax, 1, nq, 0.0);
    vec![max_rate_learned, exact_rate, q_learned, q_ex[0], coin_err, err_thp, 0.0]
}

// ------------------------------------------------------------------ 4. Fermiones CPT-impares y Cohen–Glashow (§8.11)

/// Fase 15 (§8.11): extiende el contraste observacional de la Fase 13 a fermiones (neutrinos de
/// ultra-alta energía):
/// - Caminata de Weyl de orden fijo (ω(k) ≠ ω(−k)): genera violación de Lorentz CPT-impar de
///   dimensión 5 con Δv = ±2·|A₁(n̂)|·E/E_P (signo opuesto para partícula y antipartícula, así que
///   siempre hay una rama superlumínica δ_ν > 0!).
/// - Un neutrino superlumínico con δ_ν = v_ν − 1 > 0 emite pares e⁺e⁻ por radiación Cherenkov en el
///   vacío (Cohen y Glashow, PRL 107, 181803, 2011) por encima del umbral E_th = 2·m_e / √(2·δ_ν),
///   o equivalently E_th(Δx) = (2 m_e² · E_P / (|A₁| · (Δx/ℓ_P)))^{1/3}.
/// - Con el neutrino de mayor energía observado (KM3-230213A ≈ 220 PeV = 2,2·10⁸ GeV; IceCube ~ 6 PeV),
///   comprueba la cota sobre Δx/ℓ_P para Weyl de orden fijo frente a la inmunidad de las reglas
///   palindrómicas (donde A₁ = 0).
/// Devuelve [A1_rms_fermion, E_th_GeV_a_lP, cota_dx_lP_IceCube_6PeV, cota_dx_lP_KM3NeT_220PeV, asym_palindromica].
pub fn fermion_cpt_cohen_glashow_bounds() -> Vec<f64> {
    // En la caminata de Weyl de orden fijo (Fase 13), ω(k) = |k| − s·k²·Δx con s = n_x·n_y·n_z:
    // coeficiente de velocidad de grupo A₁ = −2s (rms = 2/√105 = 0,19518), mientras que el exceso
    // de energía de dispersión es ΔE = ½·|A₁|·E²/E_P ⇒ parámetro de Cohen–Glashow δ_CG = |A₁|·E/E_P.
    let a1_rms = 2.0 / (105.0f64).sqrt();
    let m_e = 0.51099895e-3f64; // GeV
    let e_planck = 1.22089e19f64; // GeV
    // Umbral de Cohen–Glashow (PRL 107, 181803, 2011): ν → ν e⁺ e⁻ cuando δ_CG(E) ≥ 4 m_e² / E²
    // ⇒ E_th = (4 m_e² · E_P / (|A₁| · dx_lP))^{1/3}
    let e_th_at_lp = (4.0 * m_e * m_e * e_planck / a1_rms).cbrt();
    // Si observamos un neutrino de energía E_obs sin pérdida por emisión de pares:
    // Δx/ℓ_P < 4 m_e² · E_P / (|A₁| · E_obs³)
    let bound_dx = |e_obs_gev: f64| 4.0 * m_e * m_e * e_planck / (a1_rms * e_obs_gev.powi(3));
    let dx_icecube = bound_dx(6.0e6); // 6 PeV (resonancia de Glashow / eventos PeV de IceCube)
    let dx_km3net = bound_dx(2.2e8); // 220 PeV (KM3-230213A)
    // Verificación numérica directa de la asimetría ω(k) − ω(−k) en k = (0,15; 0,11; 0,09)
    let k = [0.15, 0.11, 0.09];
    let eval_alt = |kk: [f64; 3]| -> f64 {
        let ux = lazy_axis_step(0, kk[0], 0.0);
        let uy = lazy_axis_step(1, kk[1], 0.0);
        let uz = lazy_axis_step(2, kk[2], 0.0);
        let u1 = m2_mul(&uz, &m2_mul(&uy, &ux));
        let u2 = m2_mul(&ux, &m2_mul(&uy, &uz));
        let u = m2_mul(&u2, &u1);
        0.5 * (0.5 * (u[0][0] + u[1][1]).re).clamp(-1.0, 1.0).acos()
    };
    let asym_pal = (eval_alt(k) - eval_alt([-k[0], -k[1], -k[2]])).abs();
    vec![a1_rms, e_th_at_lp, dx_icecube, dx_km3net, asym_pal]
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 15 (Collins et al., PRL 2004): corrección radiativa a un lazo de la velocidad de la luz
    /// δc² = c_ren² − 1 en teoría Yukawa 1+1D, comparando regulador continuo (δc² → 0) frente a red
    /// discreta (δc² = O(g²) ≠ 0 de dimensión 4, no suprimido por E/E_P).
    /// Devuelve [dc2_cont_L120, dc2_cont_L20, dc2_red_wilson, dc2_red_r0, delta_c_especies, dm2_cont, dm2_red].
    pub fn collins_one_loop_lv(&self, mf: f32, m_phi: f32, g_y: f32, wilson_r: f32) -> Vec<f64> {
        compute_collins_one_loop_lv(
            (mf as f64).clamp(0.05, 1.5),
            (m_phi as f64).clamp(0.2, 4.0),
            (g_y as f64).clamp(0.0, 2.0),
            (wilson_r as f64).clamp(0.0, 2.0),
        )
    }

    /// Fase 15 (§8.6): escanea la zona de Brillouin 3D de una caminata 3D de tipo luz con parámetros
    /// de Gupta–Short `(tx, ty, tz)` buscando dobladores/ceros espurios fuera de |k| < 0,35.
    /// Devuelve [eps_min, kx, ky, kz, c_cono, num_minimos_bajos].
    pub fn scan_3d_lightlike_walk(&self, tx: f32, ty: f32, tz: f32, alternating: bool, grid_n: u32) -> Vec<f64> {
        scan_3d_lightlike_walk_impl(tx as f64, ty as f64, tz as f64, alternating, grid_n as usize)
    }

    /// Fase 15 (§8.6): barrido sistemático de ángulos (θ_x, θ_y, θ_z) para cuantificar la
    /// obstrucción de dobladores/ceros espurios en caminatas 3D locales de tipo luz.
    /// Devuelve [mejor_hueco, best_tx, best_ty, best_tz, c_cono, hueco_weyl_0, hueco_gs_04].
    pub fn search_3d_lightlike_doublers(&self, grid_n: u32) -> Vec<f64> {
        search_3d_lightlike_obstruction(grid_n as usize)
    }

    /// Fase 15 (§8.10): aprende de datos el mapa discreto multicomponente de un paso (4 canales × 3
    /// sitios + fase no lineal) y comprueba que SÍ predice la resonancia paramétrica de Floquet con g < 0.
    /// Devuelve [tasa_mapa_aprendido, tasa_exacta, q_aprendido, q_exacto, err_moneda, err_theta_p, tasa_edp_continua].
    pub fn discover_discrete_floquet_map(&self, m: f32, g: f32, rho_bar: f32, seed: u32) -> Vec<f64> {
        run_discover_discrete_floquet_map(
            (m as f64).clamp(0.1, 1.4),
            (g as f64).clamp(-0.5, 0.5),
            (rho_bar as f64).clamp(0.05, 1.5),
            seed as u64,
        )
    }

    /// Fase 15 (§8.11): cotas de violación de Lorentz CPT-impar para fermiones (umbral de emisión de
    /// pares de Cohen–Glashow para neutrinos de IceCube y KM3NeT).
    /// Devuelve [A1_rms, E_th_GeV_en_lP, cota_dx_IceCube, cota_dx_KM3NeT, asym_palindromica].
    pub fn fermion_cpt_lorentz_bounds(&self) -> Vec<f64> {
        fermion_cpt_cohen_glashow_bounds()
    }
}
