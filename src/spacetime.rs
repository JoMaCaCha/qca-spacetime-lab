//! Fase 15 — Espacio-tiempo discreto covariante y geometría dinámica:
//!
//! 1. **Conjuntos causales (Causal Sets; Bombelli, Henson y Sorkin, 2009):**
//!    espolvoreado de Poisson en un diamante causal de Minkowski 1+1D. Comprueba la invariancia
//!    exacta de Lorentz del orden causal y de la dimensión fractal de Myrheim–Meyer (d_MM ≈ 2)
//!    bajo cualquier transformación de Lorentz Λ(η), frente a la anisotropía intrínseca de una red
//!    cúbica fija, y mide el precio en no localidad (grado medio de enlaces causales ~ ln N).
//! 2. **D'Alembertiano discreto de Benincasa–Dowker □_ρ en 1+1D:**
//!    operador de onda retardado covariante sobre el conjunto causal que converge en promedio al
//!    d'Alembertiano continuo (∂²_t − ∂²_x)φ sin escoger sistema de referencia.
//! 3. **Caminata cuántica sobre un grafo dinámico con retroacción materia-geometría (modelo de
//!    juguete):** los ángulos de enlace θ_i(t) (velocidad local de la luz c_i = cos θ_i) reaccionan
//!    a la densidad de materia conservando exactamente la unitariedad de la caminata y produciendo
//!    refracción / retraso de Shapiro emergente (modelo escalar, no relatividad general de espín 2).

use crate::rng::SplitMix64;
use crate::util::{clamp_finite, mm_dim_2d};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use wasm_bindgen::prelude::*;

type C = Complex<f64>;

/// Genera N puntos de un espolvoreado de Poisson uniforme en el diamante causal centrado en el origen
/// (en coordenadas nulas u = (t + x)/√2, v = (t − x)/√2 ∈ [−S/2, +S/2], volumen V = S²).
fn sprinkle_diamond(n: usize, s_side: f64, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = SplitMix64::new(seed ^ 0xCA05_A15E);
    let inv_s2 = 1.0 / 2.0f64.sqrt();
    let half = 0.5 * s_side;
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.next_f64() * s_side - half;
        let v = rng.next_f64() * s_side - half;
        let t = (u + v) * inv_s2;
        let x = (u - v) * inv_s2;
        pts.push((t, x));
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    pts
}

/// Aplica un boost de Lorentz de rapidez `eta` a (t, x).
#[inline]
fn lorentz_boost(t: f64, x: f64, eta: f64) -> (f64, f64) {
    let ch = eta.cosh();
    let sh = eta.sinh();
    (ch * t - sh * x, -sh * t + ch * x)
}

/// Calcula el grafo causal de `pts` (ordenados en t): matriz booleana `prec[i * n + j]` (i ≺ j)
/// y matriz de cardinalidad de intervalos `between[i * n + j]` = |{k : i ≺ k ≺ j}|.
fn causal_structure(pts: &[(f64, f64)]) -> (Vec<bool>, Vec<u32>, usize, usize) {
    let n = pts.len();
    let mut prec = vec![false; n * n];
    let mut n_rel = 0usize;
    for i in 0..n {
        let (ti, xi) = pts[i];
        for j in (i + 1)..n {
            let dt = pts[j].0 - ti;
            let dx = (pts[j].1 - xi).abs();
            if dt > dx {
                prec[i * n + j] = true;
                n_rel += 1;
            }
        }
    }
    let mut between = vec![0u32; n * n];
    let mut n_links = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if !prec[i * n + j] {
                continue;
            }
            let mut cnt = 0u32;
            for k in (i + 1)..j {
                if prec[i * n + k] && prec[k * n + j] {
                    cnt += 1;
                }
            }
            between[i * n + j] = cnt;
            if cnt == 0 {
                n_links += 1;
            }
        }
    }
    (prec, between, n_rel, n_links)
}

/// Análisis completo de un conjunto causal 1+1D bajo boost de Lorentz `eta`:
/// 1. Invariancia exacta del grafo causal bajo boost (diferencia de enlaces = 0).
/// 2. Estimador de dimensión de Myrheim–Meyer d_MM frente a d = 2.
/// 3. Grado medio de enlaces causales (no localidad ~ ln N).
/// 4. Operador de onda discreto de Benincasa–Dowker □_ρ evaluado sobre φ(t, x) = ½(t² − x²)
///    (cuyo d'Alembertiano exacto es (∂²_t − ∂²_x)φ = 1 − (−1) = 2 en todo punto y en cualquier boost!).
/// 5. Control de anisotropía direccional en red cúbica fija bajo el mismo boost.
pub fn run_causal_set_analysis(n_pts: usize, eta_boost: f64, trials: usize, seed: u64) -> Vec<f64> {
    let n = n_pts.clamp(40, 400);
    let eta_boost = clamp_finite(eta_boost, -4.0, 4.0, 0.0);
    let n_trials = trials.clamp(1, 200);
    let s_side = (n as f64).sqrt(); // densidad ρ = N / V = 1 ⇒ ℓ = 1
    let mut d_mm_sum = 0.0;
    let mut deg_sum = 0.0;
    let mut link_diff_max = 0usize;
    let mut bd_rest_sum = 0.0;
    let mut bd_boost_sum = 0.0;
    let mut bd_cnt = 0usize;
    let mut pts_sample = Vec::new();

    for tr in 0..n_trials {
        let pts = sprinkle_diamond(n, s_side, seed.wrapping_add(tr as u64 * 9973));
        if tr == 0 {
            pts_sample = pts.clone();
        }
        let (prec, between, n_rel, n_links) = causal_structure(&pts);
        let r_frac = 2.0 * n_rel as f64 / (n * (n - 1)) as f64;
        d_mm_sum += mm_dim_2d(r_frac);
        deg_sum += 2.0 * n_links as f64 / n as f64;

        // Aplicamos el boost Λ(η) a las coordenadas y reconstruimos el orden causal
        let mut pts_b: Vec<(f64, f64, usize)> = pts
            .iter()
            .enumerate()
            .map(|(idx, &(t, x))| {
                let (tb, xb) = lorentz_boost(t, x, eta_boost);
                (tb, xb, idx)
            })
            .collect();
        pts_b.sort_by(|a, b| a.0.total_cmp(&b.0));
        let pts_b_xy: Vec<(f64, f64)> = pts_b.iter().map(|&(t, x, _)| (t, x)).collect();
        let (prec_b, between_b, n_rel_b, n_links_b) = causal_structure(&pts_b_xy);
        link_diff_max = link_diff_max.max(n_rel.abs_diff(n_rel_b)).max(n_links.abs_diff(n_links_b));

        // D'Alembertiano discreto no local de Sorkin–Benincasa–Dowker B^(2)_ε en 1+1D (PRL 104, 181301, 2010):
        //   B^(2)_ε φ(x) = −(4ε/ℓ²) [ −½ φ(x) + ε Σ_{y ≺ x} f₂(n(y, x), ε) φ(y) ]
        //   f₂(n, ε) = (1 − ε)^n · [1 − 2εn/(1−ε) + ε²n(n−1)/(2(1−ε)²)]
        // El suavizado covariante ε ∈ (0, 1) (escala no local ℓ_k = ℓ/√ε) suprime las fluctuaciones
        // de Poisson conservando la invariancia de Lorentz exacta y dando ⟨B^(2)_ε φ⟩ = □φ = +2
        // sobre el campo invariante de Lorentz φ(t, x) = ½(t² − x²) = u·v.
        let eps = 0.35f64;
        let om_e = 1.0 - eps;
        let inv_s2 = 1.0 / 2.0f64.sqrt();
        let half = 0.5 * s_side;
        let f2_kernel = |k: u32| -> f64 {
            let kf = k as f64;
            let base = om_e.powi(k as i32);
            base * (1.0 - 2.0 * eps * kf / om_e + 0.5 * eps * eps * kf * (kf - 1.0) / (om_e * om_e))
        };
        let eval_bd_at = |coords: &[(f64, f64)], pr: &[bool], btw: &[u32], idx_list: &[usize]| -> (f64, usize) {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            for &x_idx in idx_list {
                let phi = |idx: usize| {
                    let (t, x) = coords[idx];
                    0.5 * (t * t - x * x)
                };
                let mut acc = -0.5 * phi(x_idx);
                for y_idx in 0..x_idx {
                    if !pr[y_idx * n + x_idx] {
                        continue;
                    }
                    let k = btw[y_idx * n + x_idx];
                    if k < 48 {
                        acc += eps * f2_kernel(k) * phi(y_idx);
                    }
                }
                sum += -4.0 * eps * acc;
                cnt += 1;
            }
            (sum, cnt)
        };
        // Evaluamos en la región interior lejos de la frontera pasada (u, v ∈ [0, 0.35·S], con
        // profundidad pasada ≥ 0.5·S ⇒ ε·V_pasado ≥ 0.35·(S/2)² ≫ 1) y donde |φ| = |u·v| es pequeño.
        let mut rest_ids = Vec::new();
        let mut rest_to_boost = vec![0usize; n];
        for (b_idx, &(_, _, orig_idx)) in pts_b.iter().enumerate() {
            rest_to_boost[orig_idx] = b_idx;
        }
        let mut boost_ids = Vec::new();
        for x_idx in 0..n {
            let (tx, xx) = pts[x_idx];
            let u = (tx + xx) * inv_s2;
            let v = (tx - xx) * inv_s2;
            if u >= -0.05 * half && u <= 0.65 * half && v >= -0.05 * half && v <= 0.65 * half {
                rest_ids.push(x_idx);
                boost_ids.push(rest_to_boost[x_idx]);
            }
        }
        let (s_r, c_r) = eval_bd_at(&pts, &prec, &between, &rest_ids);
        let (s_b, _) = eval_bd_at(&pts_b_xy, &prec_b, &between_b, &boost_ids);
        bd_rest_sum += s_r;
        bd_boost_sum += s_b;
        bd_cnt += c_r;
    }

    let t_f = n_trials as f64;
    let d_mm = d_mm_sum / t_f;
    let mean_deg = deg_sum / t_f;
    let bd_rest = if bd_cnt > 0 { bd_rest_sum / bd_cnt as f64 } else { f64::NAN };
    let bd_boost = if bd_cnt > 0 { bd_boost_sum / bd_cnt as f64 } else { f64::NAN };
    // Factor de distorsión de paso de una red cúbica fija bajo el mismo boost: e^{2|η|}
    let cubic_distortion = (2.0 * eta_boost.abs()).exp();
    // Cabecera de 7 valores + hasta 120 pares (x, t) del primer espolvoreado para graficar el diamante
    let keep = pts_sample.len().min(120);
    let mut out = vec![d_mm, link_diff_max as f64, mean_deg, bd_rest, bd_boost, cubic_distortion, keep as f64];
    for &(t, x) in pts_sample.iter().take(keep) {
        out.push(x);
        out.push(t);
    }
    out
}

/// Caminata cuántica 1D de Dirac/split-step sobre un grafo dinámico con retroacción materia-geometría:
/// en cada sitio x el ángulo de moneda efectivo θ(x, t) (velocidad local de la luz c(x) = cos θ(x))
/// evoluciona dinámicamente según la densidad de materia local ρ(x):
///   θ(x, t+1) = (1 − γ_g)·θ(x, t) + γ_g·[θ₀ + λ_g·ρ_suav(x)]
/// Esto garantiza unitariedad exacta de la caminata (norma conservada a 10⁻¹⁵) mientras una masa
/// central curva el cono de luz local y retrasa un paquete de prueba (análogo escalar del retraso
/// de Shapiro sobre un grafo dinámico, sin ser gravedad de espín 2 de Einstein).
#[allow(clippy::too_many_arguments)]
pub fn run_dynamic_graph_walk(l: usize, theta0: f64, lam_geom: f64, gamma_geom: f64, mass_amp: f64, steps: usize) -> Vec<f64> {
    let n = l.clamp(32, 256);
    let run_one = |lam: f64| -> (f64, f64, f64, Vec<f64>, Vec<f64>) {
        // Dos campos independientes en el mismo grafo dinámico:
        // 1) masa fuente estacionaria (perfil ρ_src centrado en n/2)
        // 2) paquete de prueba relativista (componente + que viaja a la derecha desde x = n/6)
        let mut psi = vec![[C::new(0.0, 0.0); 2]; n];
        let mut tmp = vec![[C::new(0.0, 0.0); 2]; n];
        let x0 = n as f64 / 6.0;
        let sig = 3.5;
        let mut nrm2 = 0.0;
        for x in 0..n {
            let dx = x as f64 - x0;
            let amp = (-(dx * dx) / (2.0 * sig * sig)).exp();
            // Estado que viaja a la derecha (k = 0, componente 0)
            psi[x][0] = C::new(amp, 0.0);
            nrm2 += amp * amp;
        }
        let inv = 1.0 / nrm2.sqrt();
        for p in psi.iter_mut() {
            p[0] *= inv;
        }
        let n0: f64 = psi.iter().map(|p| p[0].norm_sqr() + p[1].norm_sqr()).sum();

        // Densidad de fuente localizada en el centro n/2
        let xc = n as f64 / 2.0;
        let src: Vec<f64> = (0..n)
            .map(|x| {
                let dx = x as f64 - xc;
                mass_amp * (-(dx * dx) / (2.0 * 4.5 * 4.5)).exp()
            })
            .collect();
        let mut theta = vec![theta0; n];
        let mut next_th = vec![0.0; n];
        let mut c_prof = vec![theta0.cos(); n];
        let mut x_hist = Vec::with_capacity(steps);

        for _ in 0..steps {
            // 1. Dinámica del grafo: θ(x) reacciona a ρ_src + |ψ|² con suavizado difusivo
            for x in 0..n {
                let xm = if x == 0 { n - 1 } else { x - 1 };
                let xp = if x + 1 == n { 0 } else { x + 1 };
                let rho_tot = src[x] + psi[x][0].norm_sqr() + psi[x][1].norm_sqr();
                let target = (theta0 + lam * rho_tot).clamp(0.0, 1.35);
                let lap = 0.25 * (theta[xm] + theta[xp] - 2.0 * theta[x]);
                next_th[x] = ((1.0 - gamma_geom) * (theta[x] + lap) + gamma_geom * target).clamp(0.0, 1.35);
            }
            std::mem::swap(&mut theta, &mut next_th);

            // 2. Paso unitario exacto sobre el grafo dinámico: moneda local C(θ(x)) + desplazamiento
            for x in 0..n {
                let (sn, cs) = theta[x].sin_cos();
                let (a, b) = (psi[x][0], psi[x][1]);
                tmp[x][0] = a * cs - b * sn;
                tmp[x][1] = a * sn + b * cs;
            }
            for x in 0..n {
                let xm = if x == 0 { n - 1 } else { x - 1 };
                let xp = if x + 1 == n { 0 } else { x + 1 };
                psi[x][0] = tmp[xm][0];
                psi[x][1] = tmp[xp][1];
            }
            // Posición media del paquete
            let mut x_mean = 0.0;
            let mut p_tot = 0.0;
            for (x, p) in psi.iter().enumerate() {
                let e = p[0].norm_sqr() + p[1].norm_sqr();
                x_mean += x as f64 * e;
                p_tot += e;
            }
            x_hist.push(x_mean / p_tot.max(1e-30));
        }
        let n1: f64 = psi.iter().map(|p| p[0].norm_sqr() + p[1].norm_sqr()).sum();
        for x in 0..n {
            c_prof[x] = theta[x].cos();
        }
        (*x_hist.last().unwrap_or(&x0), (n1 - n0).abs(), c_prof[n / 2], c_prof, x_hist)
    };

    let (x_flat, norm_err_flat, c_flat, _, _) = run_one(0.0);
    let (x_curved, norm_err_curved, c_center, c_prof, _) = run_one(lam_geom);
    let shapiro_delay = x_flat - x_curved;
    // [x_plano, x_curvo, retraso_shapiro, c_plano, c_centro_curvo, err_norma_max, L, ...c_prof(x)]
    let mut out = vec![x_flat, x_curved, shapiro_delay, c_flat, c_center, norm_err_flat.max(norm_err_curved), n as f64];
    out.extend_from_slice(&c_prof);
    out
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 15 (Bombelli, Henson y Sorkin, 2009; Benincasa y Dowker, 2010): conjunto causal 1+1D
    /// en un diamante de Minkowski, invariancia de Lorentz bajo boost `eta_boost`, dimensión fractal
    /// de Myrheim–Meyer y operador de onda discreto □_ρ sobre φ = ½(t² − x²) (valor exacto = 2).
    /// Devuelve puntero a [d_MM, diff_enlaces_boost, grado_medio, BD_reposo, BD_boost, dist_cubica, n_pts, x0, t0, …].
    pub fn causal_set_lorentz_test(&mut self, n_pts: u32, eta_boost: f32, trials: u32, seed: u32) -> *const f64 {
        let eta = clamp_finite(eta_boost as f64, -4.0, 4.0, 0.0);
        self.oq.out = run_causal_set_analysis(n_pts as usize, eta, (trials as usize).clamp(1, 200), seed as u64);
        self.oq.out.as_ptr()
    }

    /// Fase 15: caminata cuántica unitaria sobre un grafo dinámico con retroacción materia-geometría
    /// (modelo de juguete de métrica emergente: retraso de Shapiro y conservación exacta de norma).
    /// Devuelve puntero a [x_plano, x_curvo, retraso, c_plano, c_centro, err_norma, L, c(0)…c(L−1)].
    pub fn dynamic_graph_walk_test(&mut self, l: u32, theta0: f32, lam_geom: f32, gamma_geom: f32, mass_amp: f32, steps: u32) -> *const f64 {
        self.oq.out = run_dynamic_graph_walk(
            l as usize,
            clamp_finite(theta0 as f64, 0.0, 1.0, 0.25),
            clamp_finite(lam_geom as f64, 0.0, 2.0, 0.5),
            clamp_finite(gamma_geom as f64, 0.01, 1.0, 0.25),
            clamp_finite(mass_amp as f64, 0.0, 5.0, 1.0),
            (steps as usize).clamp(10, 200),
        );
        self.oq.out.as_ptr()
    }
}
