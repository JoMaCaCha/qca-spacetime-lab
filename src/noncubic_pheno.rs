//! Fase 16 — Fenomenología de redes 3D no cúbicas (BCC, FCC, isotrópica SC+FCC), motor de
//! verosimilitud estadística sobre catálogos astrofísicos reales (GRB, Kislat–Krawczynski,
//! IceCube/KM3NeT) y fermiones quirales de frontera finita (Kaplan y Sen, PRL 132, 141604, 2024;
//! Golterman y Shamir, 2025) + escáner QCA de 4 componentes.

use crate::rng::SplitMix64;
use crate::util::{clamp_finite, fibonacci_sphere};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

type C = Complex<f64>;

const E_PLANCK: f64 = 1.22089e19; // GeV

// ------------------------------------------------------------------ 1. Redes 3D no cúbicas (SC, BCC, FCC, isotrópica SC+FCC)

/// Evalúa la relación de dispersión y su anisotropía direccional A₂(n̂) en 600 direcciones de
/// Fibonacci para cinco clases de redes 3D:
///   0: Cúbica simple (SC) orden fijo (n = 1, birrefringente)
///   1: Cúbica simple (SC) palindrómica (n = 2, anisotropía ~ I₄ = n_x⁴ + n_y⁴ + n_z⁴)
///   2: BCC (cúbica centrada en el cuerpo, 4 diagonales (±1,±1,±1)) palindrómica (n = 2)
///   3: FCC (cúbica centrada en caras, 6 pares de diagonales de cara) palindrómica (n = 2)
///   4: Esténcil isotrópico palindrómico SC+FCC (⅓ SC + ⅔ FCC: cancela exactamente I₄ en O(k²Δx²)!)
///   5: BCC de orden fijo (comprueba si BCC sin palíndromo también tiene asimetría CPT-impar n = 1)
pub fn run_noncubic_lattice_lv_map() -> Vec<f64> {
    let ndirs = 400usize;
    let dirs = fibonacci_sphere(ndirs);

    // Funciones de dispersión sin masa ω(k) = √(2 F(k)) para cada red palindrómica (con F(k) = k²/2 + O(k⁴))
    // y el producto de ejes para SC / BCC de orden fijo.
    let omega_sc_fixed = |k: [f64; 3]| -> f64 {
        let a0 = k[0].cos() * k[1].cos() * k[2].cos() + k[0].sin() * k[1].sin() * k[2].sin();
        a0.clamp(-1.0, 1.0).acos()
    };
    let omega_sc_pal = |k: [f64; 3]| -> f64 {
        let f = (1.0 - k[0].cos()) + (1.0 - k[1].cos()) + (1.0 - k[2].cos());
        (2.0 * f).max(0.0).sqrt()
    };
    let omega_bcc_pal = |k: [f64; 3]| -> f64 {
        // 4 diagonales (±1,±1,±1): F_BCC = 1 − cos kx cos ky cos kz = k²/2 − (3k⁴ − 2Σk_a⁴)/24
        let f = 1.0 - k[0].cos() * k[1].cos() * k[2].cos();
        (2.0 * f).max(0.0).sqrt()
    };
    let omega_fcc_pal = |k: [f64; 3]| -> f64 {
        // 6 pares (±1,±1,0): F_FCC = ½ Σ_{a<b}(1 − cos k_a cos k_b) = k²/2 − (3k⁴ − Σk_a⁴)/48
        let (cx, cy, cz) = (k[0].cos(), k[1].cos(), k[2].cos());
        let f = 0.5 * ((1.0 - cx * cy) + (1.0 - cy * cz) + (1.0 - cz * cx));
        (2.0 * f).max(0.0).sqrt()
    };
    let omega_iso_sc_fcc = |k: [f64; 3]| -> f64 {
        // Combinación isotrópica exacta: ⅓ F_SC + ⅔ F_FCC cancela Σ k_a⁴ exactamente a O(k⁴)!
        let (cx, cy, cz) = (k[0].cos(), k[1].cos(), k[2].cos());
        let f_sc = (1.0 - cx) + (1.0 - cy) + (1.0 - cz);
        let f_fcc = 0.5 * ((1.0 - cx * cy) + (1.0 - cy * cz) + (1.0 - cz * cx));
        let f = (f_sc + 2.0 * f_fcc) / 3.0;
        (2.0 * f).max(0.0).sqrt()
    };
    // Caminata unitaria BCC de orden fijo en SU(2): producto de 4 saltos a lo largo de d_m = (±1,±1,±1)/√3
    let omega_bcc_fixed = |k: [f64; 3]| -> f64 {
        let s3 = 1.0 / 3.0f64.sqrt();
        let axes = [
            [s3, s3, s3],
            [s3, -s3, -s3],
            [-s3, s3, -s3],
            [-s3, -s3, s3],
        ];
        // Multiplicamos cuaterniones unitarios q_m = (cos(θ_m), −sin(θ_m) d_m) con θ_m = (√3/4)(k·d_m)
        // Nota: Σ_m (d_m ⊗ d_m) = (4/3) I ⇒ con ángulo θ_m = (3/4)(k·d_m) el límite k→0 da ω = |k| exacto!
        let mut q = [1.0, 0.0, 0.0, 0.0];
        for ax in axes {
            let kd = k[0] * ax[0] + k[1] * ax[1] + k[2] * ax[2];
            let th = 0.75 * kd;
            let (st, ct) = th.sin_cos();
            let qm = [ct, -st * ax[0], -st * ax[1], -st * ax[2]];
            q = [
                q[0] * qm[0] - q[1] * qm[1] - q[2] * qm[2] - q[3] * qm[3],
                q[0] * qm[1] + q[1] * qm[0] + q[2] * qm[3] - q[3] * qm[2],
                q[0] * qm[2] - q[1] * qm[3] + q[2] * qm[0] + q[3] * qm[1],
                q[0] * qm[3] + q[1] * qm[2] - q[2] * qm[1] + q[3] * qm[0],
            ];
        }
        q[0].clamp(-1.0, 1.0).acos()
    };

    // Mide el coeficiente A₂(n̂) = (ω(k n̂)/k − 1) / k² en k = 0,04 y su dispersión direccional (max − min)
    // además de la asimetría CPT-impar |ω(k) − ω(−k)| en k = (0,18; 0,13; 0,09).
    let measure_aniso = |f: &dyn Fn([f64; 3]) -> f64| -> (f64, f64, f64, f64) {
        let kt = [0.18, 0.13, 0.09];
        let asym = (f(kt) - f([-kt[0], -kt[1], -kt[2]])).abs();
        let k0 = 0.04;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for &d in &dirs {
            let om = f([k0 * d[0], k0 * d[1], k0 * d[2]]);
            let a2 = (om / k0 - 1.0) / (k0 * k0);
            mn = mn.min(a2);
            mx = mx.max(a2);
            sum += a2;
            sum2 += a2 * a2;
        }
        let mean = sum / ndirs as f64;
        let std = (sum2 / ndirs as f64 - mean * mean).max(0.0).sqrt();
        (asym, mean, mx - mn, std)
    };

    let r_sc_fix = measure_aniso(&omega_sc_fixed);
    let r_sc_pal = measure_aniso(&omega_sc_pal);
    let r_bcc_pal = measure_aniso(&omega_bcc_pal);
    let r_fcc_pal = measure_aniso(&omega_fcc_pal);
    let r_iso = measure_aniso(&omega_iso_sc_fcc);
    let r_bcc_fix = measure_aniso(&omega_bcc_fixed);

    // También comprobamos la anisotropía en O(k⁴) del esténcil SC+FCC (en k = 0,12):
    let mut mn4 = f64::INFINITY;
    let mut mx4 = f64::NEG_INFINITY;
    for &d in &dirs {
        let k1 = 0.04;
        let k2 = 0.12;
        let a2_1 = (omega_iso_sc_fcc([k1 * d[0], k1 * d[1], k1 * d[2]]) / k1 - 1.0) / (k1 * k1);
        let a2_2 = (omega_iso_sc_fcc([k2 * d[0], k2 * d[1], k2 * d[2]]) / k2 - 1.0) / (k2 * k2);
        let a4 = (a2_2 - a2_1) / (k2 * k2 - k1 * k1);
        mn4 = mn4.min(a4);
        mx4 = mx4.max(a4);
    }

    // [asym_sc_fix, spread_sc_pal, spread_bcc_pal, spread_fcc_pal, spread_iso_sc_fcc, asym_bcc_fix, mean_sc, mean_bcc, mean_fcc, mean_iso, spread_a4_iso]
    vec![
        r_sc_fix.0,
        r_sc_pal.2,
        r_bcc_pal.2,
        r_fcc_pal.2,
        r_iso.2,
        r_bcc_fix.0,
        r_sc_pal.1,
        r_bcc_pal.1,
        r_fcc_pal.1,
        r_iso.1,
        mx4 - mn4,
    ]
}

// ------------------------------------------------------------------ 2. Motor de verosimilitud estadística sobre catálogos reales

/// Cotas al 95 % de Kislat y Krawczynski (PRD 95, 083013, 2017, tabla IV) sobre los 16 coeficientes
/// k_(V)jm^(5) en GeV⁻¹: dividimos entre 1,96 para obtener 1σ.
const KK_TABLE_IV: [(usize, usize, bool, f64); 16] = [
    (0, 0, true, 5.0e-26),
    (1, 0, true, 6.5e-26),
    (1, 1, true, 7.8e-27),
    (1, 1, false, 2.9e-26),
    (2, 0, true, 3.0e-26),
    (2, 1, true, 7.5e-27),
    (2, 1, false, 2.4e-26),
    (2, 2, true, 2.1e-26),
    (2, 2, false, 1.7e-26),
    (3, 0, true, 2.1e-26),
    (3, 1, true, 1.4e-26),
    (3, 1, false, 1.0e-26),
    (3, 2, true, 8.3e-27),
    (3, 2, false, 1.4e-26),
    (3, 3, true, 9.7e-27),
    (3, 3, false, 5.8e-27),
];

/// Integral cosmológica de distancia de Jacob–Piran (2008) K_n(z) en segundos:
///   K_n(z) = ((1 + n) / (2 H₀)) ∫_0^z (1 + z')^n / √(Ω_m (1+z')³ + Ω_Λ) dz'
/// con parámetros Planck 2018: H₀ = 67,4 km/s/Mpc (1/H₀ = 4,578e17 s), Ω_m = 0,315, Ω_Λ = 0,685.
fn cosmic_k_n(z: f64, n: i32) -> f64 {
    let inv_h0 = 4.578e17f64; // segundos
    let om = 0.315;
    let ol = 0.685;
    let steps = 400usize;
    let dz = z / steps as f64;
    let mut sum = 0.0;
    for i in 0..steps {
        let zp = (i as f64 + 0.5) * dz;
        let ez = (om * (1.0 + zp).powi(3) + ol).sqrt();
        sum += (1.0 + zp).powi(n) / ez;
    }
    0.5 * (1 + n) as f64 * inv_h0 * sum * dz
}

/// Armónicos esféricos Y_3m(n̂) para m = 0, 1, 2, 3 (patrón puro j = 3 de n_x·n_y·n_z).
fn y3_all(n: [f64; 3]) -> [C; 4] {
    let (x, y, z) = (n[0], n[1], n[2]);
    let w = C::new(x, y);
    let w2 = w * w;
    let k = |num: f64| (num / PI).sqrt();
    [
        C::new(k(7.0 / 16.0) * (5.0 * z * z * z - 3.0 * z), 0.0),
        w * (-k(21.0 / 64.0) * (5.0 * z * z - 1.0)),
        w2 * (k(105.0 / 32.0) * z),
        w2 * w * (-k(35.0 / 64.0)),
    ]
}

/// Motor de verosimilitud estadística sobre catálogos astrofísicos reales:
/// 1) Catálogo de 3 GRB (GRB 090510 Fermi-LAT, GRB 190114C MAGIC, GRB 221009A LHAASO) con
///    integrales cosmológicas exactas K₁(z), K₂(z).
/// 2) Perfil χ²(Δx/ℓ_P) minimizado sobre `norients` orientaciones SO(3) usando los 16 coeficientes
///    de polarimetría óptica de Kislat y Krawczynski (PRD 95, 083013, 2017, Tabla IV), dando las
///    cotas exactas al 95 % CL (Δχ² = 3,84) y 5σ (Δχ² = 25,0) para el fotón de Mlodinow–Brun.
/// 3) Catálogo de neutrinos UHE (IceCube 1,04 / 2,0 / 6,05 PeV + KM3NeT 220 PeV, cota inferior al
///    90 % CL = 110 PeV) con cota conservadora de Cohen–Glashow para Weyl de orden fijo.
pub fn run_real_catalog_lv_likelihood(norients: usize) -> Vec<f64> {
    let n_or = norients.clamp(200, 3000);
    // 1. Integrales cosmológicas K_1(z) y K_2(z) para GRB 090510, GRB 190114C y GRB 221009A
    let k1_090510 = cosmic_k_n(0.903, 1);
    let k1_190114c = cosmic_k_n(0.4245, 1);
    let k1_221009a = cosmic_k_n(0.151, 1);
    let k2_221009a = cosmic_k_n(0.151, 2);

    // 2. Perfil χ²(Δx/ℓ_P) de polarimetría óptica (72 fuentes, Kislat & Krawczynski 2017):
    //    Para Mlodinow–Brun, A₁(n̂) = −n_x n_y n_z ⇒ Σ Y_jm k_jm = −½ n_x n_y n_z Δx.
    //    En cada orientación R ∈ SO(3), proyectamos −½ (R^T s)_x (R^T s)_y (R^T s)_z sobre Y_{3,m} (j=3 puro)
    //    y acumulamos S(R) = Σ_{a ∈ j=3} (c_a(R) / σ_a)² (en unidades de (Δx/ℓ_P)⁻²),
    //    de modo que χ²_perfil(x) = x² · min_R S(R) exactamente!
    let ndirs = 400usize;
    let dirs = fibonacci_sphere(ndirs);
    let wq = 4.0 * PI / ndirs as f64;
    // Extraemos las 7 entradas j = 3 de KK_TABLE_IV (1σ = lim_95 / 1,96)
    let j3_lims: Vec<(usize, bool, f64)> = KK_TABLE_IV
        .iter()
        .filter(|&&(j, _, _, _)| j == 3)
        .map(|&(_, m, re, lim)| (m, re, lim / 1.959964))
        .collect();

    let eval_chunk = |start: usize, end: usize| -> (f64, Vec<f64>) {
        let mut s_min = f64::INFINITY;
        let mut s_vals = Vec::with_capacity(end - start);
        for idx in start..end {
            let mut rng = SplitMix64::new(0xC47A_1060u64 ^ (idx as u64 * 0x9E37_79B9_7F4A_7C15));
            let u0 = rng.next_f64();
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            let (a, b) = ((1.0 - u0).sqrt(), u0.sqrt());
            let (q0, q1, q2, q3) = (
                a * (2.0 * PI * u1).sin(),
                a * (2.0 * PI * u1).cos(),
                b * (2.0 * PI * u2).sin(),
                b * (2.0 * PI * u2).cos(),
            );
            let rot = [
                [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
                [2.0 * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q2 * q3 - q0 * q1)],
                [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
            ];
            let mut k3 = [C::new(0.0, 0.0); 4];
            for &d in &dirs {
                let s_vec = [0, 1, 2].map(|r| rot[r][0] * d[0] + rot[r][1] * d[1] + rot[r][2] * d[2]);
                let fval = -0.5 * d[0] * d[1] * d[2] * wq;
                let y3 = y3_all(s_vec);
                for m in 0..4 {
                    k3[m] += y3[m].conj() * fval;
                }
            }
            // χ² por unidad de (Δx / ℓ_P)² = Σ ((|k_{3m}| / E_P) / σ_{3m})²
            let mut s_chi2 = 0.0;
            for &(m, re, sig) in &j3_lims {
                let val = (if re { k3[m].re } else { k3[m].im }) / E_PLANCK;
                let z_score = val / sig;
                s_chi2 += z_score * z_score;
            }
            if s_chi2 < s_min {
                s_min = s_chi2;
            }
            s_vals.push(s_chi2);
        }
        (s_min, s_vals)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let (s_min, mut all_s) = {
        let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);
        let chunk = n_or.div_ceil(nt);
        std::thread::scope(|sc| {
            let handles: Vec<_> = (0..nt)
                .map(|t| {
                    let st = t * chunk;
                    let en = (st + chunk).min(n_or);
                    sc.spawn(move || if st < en { eval_chunk(st, en) } else { (f64::INFINITY, Vec::new()) })
                })
                .collect();
            let mut gmin = f64::INFINITY;
            let mut merged = Vec::with_capacity(n_or);
            for h in handles {
                let (m, v) = h.join().unwrap();
                if m < gmin {
                    gmin = m;
                }
                merged.extend(v);
            }
            (gmin, merged)
        })
    };
    #[cfg(target_arch = "wasm32")]
    let (s_min, mut all_s) = eval_chunk(0, n_or);

    all_s.sort_by(f64::total_cmp);
    let s_med = all_s[all_s.len() / 2].max(1e-30);
    let s_min = s_min.max(1e-30);

    // Cotas de exclusión de perfil χ² en la PEOR orientación de la red (conservadoras):
    //   95 % CL (1 dof, Δχ² = 3,8415) y 5σ (Δχ² = 25,0)
    let dx_95_worst = (3.8414588 / s_min).sqrt();
    let dx_5sig_worst = (25.0 / s_min).sqrt();
    let dx_95_med = (3.8414588 / s_med).sqrt();

    // 3. Catálogo de neutrinos UHE (Cohen–Glashow conservador usando el extremo inferior al 90 % CL
    //    de KM3-230213A: E_lo = 110 PeV = 1,1e8 GeV, frente al nominal 220 PeV y IceCube 6,05 PeV):
    let a1_nu = 2.0 / 105.0f64.sqrt();
    let me = 0.51099895e-3;
    let dx_nu_km3_cons = 4.0 * me * me * E_PLANCK / (a1_nu * (1.1e8f64).powi(3));
    let dx_nu_icecube = 4.0 * me * me * E_PLANCK / (a1_nu * (6.05e6f64).powi(3));

    // Curva de perfil χ²(log10(Δx/ℓ_P)) para graficar (log10 x ∈ [−7,5, −4,5])
    let n_grid = 25usize;
    let mut out = vec![
        dx_95_worst,
        dx_5sig_worst,
        dx_95_med,
        dx_nu_km3_cons,
        dx_nu_icecube,
        k1_090510,
        k1_190114c,
        k1_221009a,
        k2_221009a,
        n_grid as f64,
    ];
    for i in 0..n_grid {
        let log_x = -7.5 + 3.0 * i as f64 / (n_grid - 1) as f64;
        let x = 10.0f64.powf(log_x);
        let chi2_worst = s_min * x * x;
        let chi2_med = s_med * x * x;
        out.push(log_x);
        out.push(chi2_worst.min(200.0));
        out.push(chi2_med.min(200.0));
    }
    out
}

// ------------------------------------------------------------------ 3. Kaplan–Sen (2024) frontera finita 4+1D y QCA de 4 componentes

/// Diagonalización hermítica completa por Jacobi complejo (para matrices pequeñas n ≤ 80).
pub(crate) fn hermitian_eig(mut a: Vec<C>, n: usize) -> (Vec<f64>, Vec<C>) {
    let mut v = vec![C::new(0.0, 0.0); n * n];
    for i in 0..n {
        v[i * n + i] = C::new(1.0, 0.0);
    }
    for _ in 0..50 {
        let mut max_off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                let r = apq.norm();
                if r > max_off {
                    max_off = r;
                }
                if r < 1e-14 {
                    continue;
                }
                let e_ip = apq / r;
                let e_im = e_ip.conj();
                let app = a[p * n + p].re;
                let aqq = a[q * n + q].re;
                let tau = (aqq - app) / (2.0 * r);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let se_ip = e_ip * s;
                let se_im = e_im * s;
                a[p * n + p] = C::new(app - t * r, 0.0);
                a[q * n + q] = C::new(aqq + t * r, 0.0);
                a[p * n + q] = C::new(0.0, 0.0);
                a[q * n + p] = C::new(0.0, 0.0);
                for k in 0..n {
                    if k != p && k != q {
                        let akp = a[k * n + p];
                        let akq = a[k * n + q];
                        let nkp = akp * c - akq * se_im;
                        let nkq = akp * se_ip + akq * c;
                        a[k * n + p] = nkp;
                        a[p * n + k] = nkp.conj();
                        a[k * n + q] = nkq;
                        a[q * n + k] = nkq.conj();
                    }
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = vkp * c - vkq * se_im;
                    v[k * n + q] = vkp * se_ip + vkq * c;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }
    }
    let evals: Vec<f64> = (0..n).map(|i| a[i * n + i].re).collect();
    (evals, v)
}

/// Construye y diagonaliza el hamiltoniano de frontera abierta de **Kaplan y Sen (PRL 132, 141604, 2024)**
/// en una losa finita de `l_slab` sitios en la 4.ª dimensión espacial (0 < m_bulk < 1):
///   H(k) = Σ_a sin(k_a) (τ_z ⊗ σ_a) + M₀(k) (τ_x ⊗ I₂) + salto_s (−τ_− en s → s+1, −τ_+ en s+1 → s)
/// donde M₀(k) = (1 − m_bulk) + Σ_a (1 − cos k_a).
/// Verifica:
/// 1) Localización exponencial (1 − m_bulk)^s del modo Weyl levógiro (⟨γ⁵⟩ ≈ +1) en la frontera s = 0
///    y su compañero dextrógiro (⟨γ⁵⟩ ≈ −1) en s = L − 1 (quiralidad total del contorno = 0, Golterman–Shamir 2025).
/// 2) Dispersión sin masa ω(k_x) = sin k_x ≈ k_x en el borde y desaparición total de los 7 dobladores
///    en k = (π, 0, 0) (donde M₀ = 3 − m > 2 sale de la fase topológica).
/// 3) Compara además el hueco de dobladores de un QCA local de 4 componentes (Wilson–Dirac) en r = 0 frente a r = 1.
pub fn run_kaplan_sen_and_4comp_qca(l_slab: usize, m_bulk: f64) -> Vec<f64> {
    let l = l_slab.clamp(8, 20);
    let m = m_bulk.clamp(0.15, 0.85);
    let dim = 4 * l;

    let build_and_diag = |kx: f64, ky: f64, kz: f64| -> (Vec<f64>, Vec<C>) {
        let mut h = vec![C::new(0.0, 0.0); dim * dim];
        let m0 = (1.0 - m) + (1.0 - kx.cos()) + (1.0 - ky.cos()) + (1.0 - kz.cos());
        let (sx, sy, sz) = (kx.sin(), ky.sin(), kz.sin());
        // σ·s (2×2): [[sz, sx - i sy], [sx + i sy, -sz]]
        let s2 = [
            [C::new(sz, 0.0), C::new(sx, -sy)],
            [C::new(sx, sy), C::new(-sz, 0.0)],
        ];
        for s in 0..l {
            let b = 4 * s;
            // Bloque τ_z ⊗ (σ·s): +s2 en (0..2), −s2 en (2..4)
            for r in 0..2 {
                for c in 0..2 {
                    h[(b + r) * dim + (b + c)] += s2[r][c];
                    h[(b + 2 + r) * dim + (b + 2 + c)] -= s2[r][c];
                }
            }
            // Masa en sitio: M₀ (τ_x ⊗ I₂) acopla (0,1) ↔ (2,3)
            for r in 0..2 {
                h[(b + r) * dim + (b + 2 + r)] += C::new(m0, 0.0);
                h[(b + 2 + r) * dim + (b + r)] += C::new(m0, 0.0);
            }
            // Salto abierto s → s+1: −τ_− ⊗ I₂ conecta |L, s+1⟩ con ⟨R, s| (y h.c. |R, s⟩ con ⟨L, s+1|)
            if s + 1 < l {
                let bn = 4 * (s + 1);
                for r in 0..2 {
                    h[(b + 2 + r) * dim + (bn + r)] -= C::new(1.0, 0.0);
                    h[(bn + r) * dim + (b + 2 + r)] -= C::new(1.0, 0.0);
                }
            }
        }
        hermitian_eig(h, dim)
    };

    // 1. Espectro y perfil en k = (0,15; 0; 0): los 4 estados más cercanos a 0 son los dos Weyl (2 bordes × 2 signos)
    let kx0 = 0.15;
    let (ev0, vec0) = build_and_diag(kx0, 0.0, 0.0);
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&i, &j| ev0[i].abs().total_cmp(&ev0[j].abs()));
    // Buscamos entre los 4 modos de borde el que vive en s = 0 (peso mayor en la primera mitad) con ω > 0
    let mut left_idx = order[0];
    let mut right_idx = order[0];
    let mut best_left_w = -1.0;
    let mut best_right_w = -1.0;
    for &idx in order.iter().take(4) {
        let mut wl = 0.0;
        let mut wr = 0.0;
        for s in 0..(l / 2) {
            for c in 0..4 {
                wl += vec0[(4 * s + c) * dim + idx].norm_sqr();
            }
        }
        for s in (l / 2)..l {
            for c in 0..4 {
                wr += vec0[(4 * s + c) * dim + idx].norm_sqr();
            }
        }
        if ev0[idx] > 0.0 && wl > best_left_w {
            best_left_w = wl;
            left_idx = idx;
        }
        if ev0[idx] > 0.0 && wr > best_right_w {
            best_right_w = wr;
            right_idx = idx;
        }
    }
    let omega_edge = ev0[left_idx].abs();
    let omega_exact = kx0.sin();
    // Quiralidad ⟨γ⁵⟩ = Σ_s (|ψ_L|² − |ψ_R|²) en el modo izquierdo y en el derecho
    let calc_chiral = |idx: usize| -> (f64, Vec<f64>) {
        let mut g5 = 0.0;
        let mut prof = vec![0.0; l];
        for s in 0..l {
            let p_l = vec0[(4 * s) * dim + idx].norm_sqr() + vec0[(4 * s + 1) * dim + idx].norm_sqr();
            let p_r = vec0[(4 * s + 2) * dim + idx].norm_sqr() + vec0[(4 * s + 3) * dim + idx].norm_sqr();
            g5 += p_l - p_r;
            prof[s] = p_l + p_r;
        }
        (g5, prof)
    };
    let (chi_left, prof_left) = calc_chiral(left_idx);
    let (chi_right, prof_right) = calc_chiral(right_idx);
    let net_chirality = chi_left + chi_right;
    // Razón de decaimiento exponencial prof[1]/prof[0] frente a (1 − m)²
    let decay_ratio = prof_left[1] / prof_left[0].max(1e-30);
    let decay_pred = (1.0 - m) * (1.0 - m);

    // 2. En el doblador k = (π, 0, 0), TODOS los modos deben tener hueco grande (sin doblador en el borde!)
    let (ev_dbl, _) = build_and_diag(PI, 0.0, 0.0);
    let gap_doubler = ev_dbl.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);

    // 3. Control de QCA local 3D de 4 componentes (Wilson–Dirac) en r = 0 (con dobladores) vs r = 1 (hueco limpio):
    let qca4_gap = |r_w: f64| -> f64 {
        let mut mn = f64::INFINITY;
        let pts = [
            [PI, 0.0, 0.0],
            [PI, PI, 0.0],
            [PI, PI, PI],
            [0.0, PI, 0.0],
            [0.0, 0.0, PI],
        ];
        for k in pts {
            let w_mass = r_w * ((1.0 - k[0].cos()) + (1.0 - k[1].cos()) + (1.0 - k[2].cos()));
            let e = (k[0].sin().powi(2) + k[1].sin().powi(2) + k[2].sin().powi(2) + w_mass * w_mass).sqrt();
            mn = mn.min(e);
        }
        mn
    };
    let gap_qca4_r0 = qca4_gap(0.0);
    let gap_qca4_r1 = qca4_gap(1.0);

    // Cabecera (10): [omega_edge, omega_exact, chi_left, chi_right, net_chirality, decay_ratio, decay_pred, gap_doubler, gap_qca4_r0, gap_qca4_r1, L, ...prof_left, ...prof_right]
    let mut out = vec![
        omega_edge,
        omega_exact,
        chi_left,
        chi_right,
        net_chirality,
        decay_ratio,
        decay_pred,
        gap_doubler,
        gap_qca4_r0,
        gap_qca4_r1,
        l as f64,
    ];
    out.extend_from_slice(&prof_left);
    out.extend_from_slice(&prof_right);
    out
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 16: mapa de anisotropía y asimetría CPT en redes 3D no cúbicas (SC, BCC, FCC e
    /// isotrópica SC+FCC que cancela exactamente la anisotropía direccional de orden k²Δx²).
    pub fn noncubic_lattice_lv_map(&self) -> Vec<f64> {
        run_noncubic_lattice_lv_map()
    }

    /// Fase 16: motor de verosimilitud estadística (perfil χ² minimizado sobre SO(3)) sobre
    /// catálogos reales (GRB Fermi-LAT/MAGIC/LHAASO, Kislat–Krawczynski 2017, IceCube y KM3NeT).
    /// Devuelve puntero a [dx_95_peor, dx_5sig_peor, dx_95_med, dx_KM3_cons, dx_IceCube, K1_090510, K1_190114C, K1_221009A, K2_221009A, n_pts, log10_x, chi2_peor, chi2_med, …].
    pub fn real_catalog_lv_likelihood(&mut self, norients: u32) -> *const f64 {
        self.oq.out = run_real_catalog_lv_likelihood(norients as usize);
        self.oq.out.as_ptr()
    }

    /// Fase 16 (Kaplan y Sen, PRL 2024; Golterman y Shamir, 2025): fermión quiral de Weyl aislado
    /// en la frontera abierta de una red finita 4+1D (localización exponencial, desaparición de los
    /// 7 dobladores y suma de quiralidad total = 0 en el contorno cerrado) + QCA 3D de 4 componentes.
    pub fn chiral_4comp_kaplan_sen_test(&mut self, l_slab: u32, m_bulk: f32) -> *const f64 {
        let m = clamp_finite(m_bulk as f64, 0.15, 0.85, 0.45);
        self.oq.out = run_kaplan_sen_and_4comp_qca(l_slab as usize, m);
        self.oq.out.as_ptr()
    }
}
