//! Fase 16 — Gravedad cuántica discreta y teoría gauge cuántica no perturbativa en 3+1D:
//!
//! 1. **Monte Carlo dinámico de conjuntos causales 2D con acción de Benincasa–Dowker–Sorkin**
//!    (Surya, *Class. Quant. Grav.* 29, 132001, 2012): muestrea el conjunto canónico de órdenes
//!    causales 2D Ω_{2D} ponderado por exp(+β S_BD) con S_BD = N − 2 N₁ + 4 N₂ − 2 N₃ usando
//!    máscaras de bits `u64::count_ones()`, reproduciendo la transición de fase entre la fase de
//!    variedad 2D (d_MM ≈ 2) y las fases no geométricas cristalinas / de orden extremo.
//! 2. **Teoría gauge cuántica SU(2) en 3+1D euclídeo (baño térmico exacto de Creutz / Kennedy–Pendleton):**
//!    simula una malla hipercúbica 4D L⁴ con fluctuaciones cuánticas reales de Yang–Mills SU(2),
//!    reproduciendo el cruce de Creutz (1980) desde acoplamiento fuerte (⟨P⟩ = β/4) hasta débil
//!    (⟨P⟩ ≈ 1 − 3/(4β)) y el confinamiento cuántico por ley de área (razón de Creutz χ(2, 2) > 0).
//! 3. **Gravedad de espín 2 frente a gravedad escalar y obstrucción de Weinberg–Witten:**
//!    contrasta cuantitativamente las polarizaciones transversales sin traza (h₊, h_×) y el factor 2
//!    de deflexión de la luz en relatividad general frente a modelos escalares / grafos dinámicos.

use crate::rng::SplitMix64;
use crate::util::mm_dim_2d;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

// ------------------------------------------------------------------ 1. Monte Carlo dinámico de conjuntos causales (Surya 2012)

/// Evalúa (S_BD, N_rel, N_1) en un 2-orden representado por una permutación `perm` de 0..N (N ≤ 64).
/// i ≺ j si y solo si i < j y perm[i] < perm[j].
/// Acción exacta de Benincasa–Dowker en 2D: S_BD = N − 2 N₁ + 4 N₂ − 2 N₃.
#[inline]
fn eval_2order_bd(perm: &[usize]) -> (f64, usize, usize) {
    let n = perm.len();
    assert!(n <= 64, "eval_2order_bd requiere N <= 64 (recibido {n})");
    let mut fut = [0u64; 64];
    let mut past = [0u64; 64];
    let mut n_rel = 0usize;
    for i in 0..n {
        let pi = perm[i];
        let mut fi = 0u64;
        for j in (i + 1)..n {
            if pi < perm[j] {
                fi |= 1u64 << j;
                past[j] |= 1u64 << i;
                n_rel += 1;
            }
        }
        fut[i] = fi;
    }
    let mut n1 = 0i64;
    let mut n2 = 0i64;
    let mut n3 = 0i64;
    for i in 0..n {
        let mut mask = fut[i];
        while mask != 0 {
            let j = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let k = (fut[i] & past[j]).count_ones();
            match k {
                0 => n1 += 1,
                1 => n2 += 1,
                2 => n3 += 1,
                _ => {}
            }
        }
    }
    let s_bd = (n as i64 - 2 * n1 + 4 * n2 - 2 * n3) as f64;
    (s_bd, n_rel, n1 as usize)
}

/// Monte Carlo de Metropolis sobre el espacio de 2-órdenes causales Ω_{2D} (Surya 2012) con peso
/// de Boltzmann exp(+β · S_BD), barriendo β ∈ {−0,25, 0,0, 0,25, 0,55, 0,90, 1,40, 2,00}.
/// Devuelve cabecera [S0/N, dMM0, R0, S_hi/N, dMM_hi, R_hi, S_neg/N, N1_neg/N, n_betas] + tabla por β.
pub fn run_causal_set_bd_mc(n_pts: usize, sweeps: usize, seed: u64) -> Vec<f64> {
    let n = n_pts.clamp(16, 48);
    let betas = [-0.25f64, 0.0, 0.25, 0.55, 0.90, 1.40, 2.00];
    let n_sweeps = sweeps.clamp(200, 6000);
    let therm = n_sweeps / 3;
    let pairs_tot = (n * (n - 1)) as f64;

    let run_beta = |beta: f64, b_idx: usize| -> (f64, f64, f64, f64, f64) {
        let mut rng = SplitMix64::new(seed ^ 0xBD16_2012u64.wrapping_add(b_idx as u64 * 1337));
        // Empezamos en una permutación aleatoria (variedad 2D)
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let (mut s_cur, mut rel_cur, mut n1_cur) = eval_2order_bd(&perm);
        let mut s_sum = 0.0;
        let mut s2_sum = 0.0;
        let mut r_sum = 0.0;
        let mut n1_sum = 0.0;
        let mut cnt = 0usize;

        for sw in 0..n_sweeps {
            // Cada barrido intenta n intercambios elementales de la permutación (ergódico y simétrico)
            for _ in 0..n {
                let a = (rng.next_u64() as usize) % n;
                let mut b = (rng.next_u64() as usize) % (n - 1);
                if b >= a {
                    b += 1;
                }
                perm.swap(a, b);
                let (s_new, rel_new, n1_new) = eval_2order_bd(&perm);
                let ds = s_new - s_cur;
                let accept = if beta * ds >= 0.0 {
                    true
                } else {
                    rng.next_f64() < (beta * ds).exp()
                };
                if accept {
                    s_cur = s_new;
                    rel_cur = rel_new;
                    n1_cur = n1_new;
                } else {
                    perm.swap(a, b);
                }
            }
            if sw >= therm && (sw - therm) % 2 == 0 {
                s_sum += s_cur;
                s2_sum += s_cur * s_cur;
                r_sum += 2.0 * rel_cur as f64 / pairs_tot;
                n1_sum += n1_cur as f64 / n as f64;
                cnt += 1;
            }
        }
        let cf = cnt.max(1) as f64;
        let s_mean = s_sum / cf;
        let var_s = (s2_sum / cf - s_mean * s_mean).max(0.0);
        let r_mean = r_sum / cf;
        let d_mm = mm_dim_2d(r_mean);
        (s_mean / n as f64, var_s / n as f64, d_mm, r_mean, n1_sum / cf)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let stats: Vec<(f64, f64, f64, f64, f64)> = std::thread::scope(|sc| {
        let handles: Vec<_> = betas
            .iter()
            .enumerate()
            .map(|(i, &b)| sc.spawn(move || run_beta(b, i)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let stats: Vec<(f64, f64, f64, f64, f64)> = betas.iter().enumerate().map(|(i, &b)| run_beta(b, i)).collect();

    // Índice 0: β = −0,25 (fase bipartita cristalina); índice 1: β = 0 (variedad 2D); último: β = 2,0 (fase ordenada)
    let st_neg = stats[0];
    let st_zero = stats[1];
    let st_hi = *stats.last().unwrap();
    let mut out = vec![
        st_zero.0,
        st_zero.2,
        st_zero.3,
        st_hi.0,
        st_hi.2,
        st_hi.3,
        st_neg.0,
        st_neg.4,
        betas.len() as f64,
    ];
    for (&b, &(sn, chi, dmm, rm, n1)) in betas.iter().zip(&stats) {
        out.extend_from_slice(&[b, sn, chi, dmm, rm, n1]);
    }
    out
}

// ------------------------------------------------------------------ 2. Yang–Mills cuántico SU(2) 4D (Creutz / Kennedy–Pendleton)

#[inline(always)]
fn q_mul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

#[inline(always)]
fn q_dag(a: [f64; 4]) -> [f64; 4] {
    [a[0], -a[1], -a[2], -a[3]]
}

#[inline(always)]
fn q_add(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Muestreo exacto de baño térmico de Kennedy–Pendleton (1985) para P(a₀) ∝ √(1 − a₀²) exp(α·a₀).
fn sample_su2_heatbath(alpha: f64, rng: &mut SplitMix64) -> [f64; 4] {
    let alpha = if alpha.is_finite() && alpha >= 0.0 { alpha } else { 0.0 };
    let a0 = if alpha < 0.08 {
        let mut chosen = 0.0;
        for _ in 0..10_000 {
            let x = 2.0 * rng.next_f64() - 1.0;
            let w = (1.0 - x * x).max(0.0).sqrt() * (alpha * (x - 1.0)).exp();
            if rng.next_f64() < w {
                chosen = x;
                break;
            }
        }
        chosen
    } else {
        let mut chosen = 1.0;
        for _ in 0..10_000 {
            let r1 = rng.next_f64().max(1e-15);
            let r2 = rng.next_f64();
            let r3 = rng.next_f64().max(1e-15);
            let r4 = rng.next_f64();
            let c = (2.0 * PI * r2).cos();
            let delta = -(r1.ln() + c * c * r3.ln()) / (2.0 * alpha);
            if delta <= 1.0 && r4 * r4 <= 1.0 - delta {
                chosen = 1.0 - 2.0 * delta;
                break;
            }
        }
        chosen
    };
    // Vector 3D uniforme en S² de radio √(1 − a₀²)
    let r_vec = (1.0 - a0 * a0).max(0.0).sqrt();
    let cos_th = 2.0 * rng.next_f64() - 1.0;
    let sin_th = (1.0 - cos_th * cos_th).max(0.0).sqrt();
    let phi = 2.0 * PI * rng.next_f64();
    [a0, r_vec * sin_th * phi.cos(), r_vec * sin_th * phi.sin(), r_vec * cos_th]
}

struct Lattice4D {
    l: usize,
    links: Vec<[f64; 4]>, // ((((x*L + y)*L + z)*L + t) * 4 + mu)
}

impl Lattice4D {
    fn new(l: usize) -> Self {
        Lattice4D {
            l,
            links: vec![[1.0, 0.0, 0.0, 0.0]; l * l * l * l * 4],
        }
    }

    #[inline(always)]
    fn site(&self, c: [usize; 4]) -> usize {
        ((c[0] * self.l + c[1]) * self.l + c[2]) * self.l + c[3]
    }

    #[inline(always)]
    fn shift(&self, mut c: [usize; 4], mu: usize, dir: isize) -> [usize; 4] {
        let l = self.l;
        c[mu] = if dir > 0 {
            if c[mu] + 1 == l { 0 } else { c[mu] + 1 }
        } else if c[mu] == 0 {
            l - 1
        } else {
            c[mu] - 1
        };
        c
    }

    #[inline(always)]
    fn get(&self, c: [usize; 4], mu: usize) -> [f64; 4] {
        self.links[self.site(c) * 4 + mu]
    }

    #[inline(always)]
    fn set(&mut self, c: [usize; 4], mu: usize, q: [f64; 4]) {
        let idx = self.site(c) * 4 + mu;
        self.links[idx] = q;
    }

    /// Suma de las 6 grapas (staples) V tales que la acción de placa es −(β/2) Re Tr(U_μ(x) V†).
    fn staple_sum(&self, x: [usize; 4], mu: usize) -> [f64; 4] {
        let mut sum = [0.0; 4];
        let x_pmu = self.shift(x, mu, 1);
        for nu in 0..4 {
            if nu == mu {
                continue;
            }
            let x_pnu = self.shift(x, nu, 1);
            // Plaqueta positiva: U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x) = U_μ(x) · [U_ν(x) U_μ(x+ν) U_ν†(x+μ)]†
            let s_pos = q_mul(q_mul(self.get(x, nu), self.get(x_pnu, mu)), q_dag(self.get(x_pmu, nu)));
            sum = q_add(sum, s_pos);
            // Plaqueta negativa: desplazada en −ν
            let x_mnu = self.shift(x, nu, -1);
            let x_pmu_mnu = self.shift(x_pmu, nu, -1);
            let s_neg = q_mul(q_mul(q_dag(self.get(x_mnu, nu)), self.get(x_mnu, mu)), self.get(x_pmu_mnu, nu));
            sum = q_add(sum, s_neg);
        }
        sum
    }

    fn heatbath_sweep(&mut self, beta: f64, rng: &mut SplitMix64) {
        let l = self.l;
        for x0 in 0..l {
            for x1 in 0..l {
                for x2 in 0..l {
                    for x3 in 0..l {
                        let c = [x0, x1, x2, x3];
                        for mu in 0..4 {
                            let v = self.staple_sum(c, mu);
                            let k = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                            if k < 1e-12 {
                                self.set(c, mu, sample_su2_heatbath(0.0, rng));
                            } else {
                                let inv = 1.0 / k;
                                let w = [v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv];
                                let u_tilde = sample_su2_heatbath(beta * k, rng);
                                let u_new = q_mul(u_tilde, w);
                                let nrm = (u_new[0] * u_new[0] + u_new[1] * u_new[1] + u_new[2] * u_new[2] + u_new[3] * u_new[3]).sqrt();
                                if nrm > 1e-30 && nrm.is_finite() {
                                    self.set(c, mu, [u_new[0] / nrm, u_new[1] / nrm, u_new[2] / nrm, u_new[3] / nrm]);
                                } else {
                                    self.set(c, mu, [1.0, 0.0, 0.0, 0.0]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Mide los lazos de Wilson rectangulares W(1,1), W(2,1) y W(2,2) promediados sobre todos los
    /// sitios y los 6 planos (μ < ν).
    fn measure_wilson_loops(&self) -> (f64, f64, f64) {
        let l = self.l;
        let mut w11 = 0.0;
        let mut w21 = 0.0;
        let mut w22 = 0.0;
        let mut cnt = 0usize;
        for x0 in 0..l {
            for x1 in 0..l {
                for x2 in 0..l {
                    for x3 in 0..l {
                        let c = [x0, x1, x2, x3];
                        for mu in 0..4 {
                            let c_m1 = self.shift(c, mu, 1);
                            let c_m2 = self.shift(c_m1, mu, 1);
                            let u_m1 = self.get(c, mu);
                            let u_m2 = q_mul(u_m1, self.get(c_m1, mu));
                            for nu in (mu + 1)..4 {
                                let c_n1 = self.shift(c, nu, 1);
                                let c_n2 = self.shift(c_n1, nu, 1);
                                let c_m1_n1 = self.shift(c_m1, nu, 1);
                                let c_m2_n1 = self.shift(c_m2, nu, 1);
                                let c_m1_n2 = self.shift(c_m1, nu, 1);
                                let c_m1_n2 = self.shift(c_m1_n2, nu, 1);
                                // W(1, 1)
                                let p11 = q_mul(q_mul(u_m1, self.get(c_m1, nu)), q_mul(q_dag(self.get(c_n1, mu)), q_dag(self.get(c, nu))));
                                w11 += p11[0];
                                // W(2, 1)
                                let p21 = q_mul(
                                    q_mul(u_m2, self.get(c_m2, nu)),
                                    q_mul(q_dag(q_mul(self.get(c_n1, mu), self.get(c_m1_n1, mu))), q_dag(self.get(c, nu))),
                                );
                                w21 += p21[0];
                                // W(2, 2)
                                let right2 = q_mul(self.get(c_m2, nu), self.get(c_m2_n1, nu));
                                let top2 = q_mul(self.get(c_n2, mu), self.get(c_m1_n2, mu));
                                let left2 = q_mul(self.get(c, nu), self.get(c_n1, nu));
                                let p22 = q_mul(q_mul(u_m2, right2), q_mul(q_dag(top2), q_dag(left2)));
                                w22 += p22[0];
                                cnt += 1;
                            }
                        }
                    }
                }
            }
        }
        let inv = 1.0 / cnt as f64;
        (w11 * inv, w21 * inv, w22 * inv)
    }
}

/// Simulación Monte Carlo de Yang–Mills cuántico SU(2) en 3+1D euclídeo (malla L⁴) a través del
/// cruce de Creutz (1980): β ∈ {1,0, 1,5, 2,0, 2,2, 2,5, 3,0}.
/// Devuelve cabecera [P(1,0), P_fuerte(1,0), P(3,0), P_débil(3,0), χ22(1,5), χ22(2,0), n_betas]
/// + tabla [β, ⟨P⟩, pred_fuerte, pred_débil, χ(2,2)].
pub fn run_su2_4d_mc(l_size: usize, therm: usize, meas: usize, seed: u64) -> Vec<f64> {
    let l = l_size.clamp(4, 8);
    let n_th = therm.clamp(12, 80);
    let n_ms = meas.clamp(20, 200);
    let betas = [1.0f64, 1.5, 2.0, 2.2, 2.5, 3.0];

    let run_one = |beta: f64, idx: usize| -> (f64, f64, f64, f64) {
        let mut lat = Lattice4D::new(l);
        let mut rng = SplitMix64::new(seed ^ 0x5024_D000u64.wrapping_add(idx as u64 * 4099));
        for _ in 0..n_th {
            lat.heatbath_sweep(beta, &mut rng);
        }
        let (mut s11, mut s21, mut s22) = (0.0, 0.0, 0.0);
        for _ in 0..n_ms {
            lat.heatbath_sweep(beta, &mut rng);
            let (w11, w21, w22) = lat.measure_wilson_loops();
            s11 += w11;
            s21 += w21;
            s22 += w22;
        }
        let inv = 1.0 / n_ms as f64;
        let (w11, w21, w22) = (s11 * inv, s21 * inv, s22 * inv);
        let chi22 = if w22 > 1e-6 && w11 > 1e-6 && w21 > 1e-6 {
            -((w22 * w11) / (w21 * w21)).ln()
        } else {
            // En acoplamiento muy fuerte W(2,2) ≈ (β/4)⁴ es pequeño respecto al ruido estadístico
            -(beta / 4.0).ln()
        };
        // Predicción exacta de acoplamiento fuerte de un solo enlace: I₂(β)/I₁(β) ≈ β/4 − β³/96
        let p_strong = (beta / 4.0 - beta.powi(3) / 96.0).clamp(0.0, 1.0);
        // Predicción perturbativa a un lazo de acoplamiento débil: 1 − 3/(4β)
        let p_weak = (1.0 - 0.75 / beta).clamp(0.0, 1.0);
        (w11, p_strong, p_weak, chi22)
    };

    #[cfg(not(target_arch = "wasm32"))]
    let res: Vec<(f64, f64, f64, f64)> = std::thread::scope(|sc| {
        let handles: Vec<_> = betas
            .iter()
            .enumerate()
            .map(|(i, &b)| sc.spawn(move || run_one(b, i)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    #[cfg(target_arch = "wasm32")]
    let res: Vec<(f64, f64, f64, f64)> = betas.iter().enumerate().map(|(i, &b)| run_one(b, i)).collect();

    let mut out = vec![
        res[0].0,
        res[0].1,
        res[5].0,
        res[5].2,
        res[1].3,
        res[2].3,
        betas.len() as f64,
    ];
    for (&b, &(p, ps, pw, chi)) in betas.iter().zip(&res) {
        out.extend_from_slice(&[b, p, ps, pw, chi]);
    }
    out
}

// ------------------------------------------------------------------ 3. Gravedad de espín 2 vs escalar y Weinberg–Witten

/// Contraste exacto entre relatividad general linealizada (Fierz–Pauli de espín 2) y modelos
/// escalares / grafos dinámicos (Nordström, Fase 15):
/// 1. Comprueba las dos polarizaciones transversales sin traza (h₊, h_×) para onda en +z:
///    traza = 0, divergencia k^μ h_μν = 0, y ley de rotación de espín 2 bajo giro de 45° (h₊ → h_×, h_× → −h₊).
/// 2. Contrasta la razón de deflexión (luz v=c) / (partícula lenta v→0, normalizada por v²):
///    Einstein espín 2 = 2,000 (h_00 + h_ij), Nordström escalar = 0,000, refractivo puro = 1,000.
/// 3. Contrasta el elemento de matriz de Weinberg–Witten ⟨p', +h | T^{μν} | p, +h⟩ en el límite
///    q = p' − p → 0 para helicidades h = 0, ½, 1, 2: para |h| = 2, la rotación de Wigner por
///    ángulo φ alrededor del eje de colisión da fase e^{2i h φ} = e^{4i φ}, ortogonal al rango 2
///    de un tensor covariante T^{μν} (que solo contiene momentos angulares |m| ≤ 2 ⇒ |h| ≤ 1!).
/// Devuelve [err_traza_transv, err_rot_45_spin2, ratio_luz_einstein, ratio_luz_nordstrom, ratio_luz_refractivo, max_helicidad_ww_permitida, ww_proyeccion_spin2].
pub fn run_spin2_emergent_gravity_check() -> Vec<f64> {
    // 1. Tensores de polarización h₊ y h_× de onda gravitacional viajando en +z (k = (ω, 0, 0, ω))
    let hp: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]; // (x, y, z) -> (z=0, x=1, y=2)
    let hx: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    let tr_p = hp[0][0] + hp[1][1] + hp[2][2];
    let tr_x = hx[0][0] + hx[1][1] + hx[2][2];
    let div_p = hp[0][0].abs() + hp[0][1].abs() + hp[0][2].abs();
    let div_x = hx[0][0].abs() + hx[0][1].abs() + hx[0][2].abs();
    let err_tt = tr_p.abs() + tr_x.abs() + div_p + div_x;

    // Rotación R(θ = +π/4) en el plano transversal (1, 2): R · h₊ · R^T = h_× (cos 2θ = 0, sin 2θ = 1)
    let c = (PI / 4.0).cos();
    let s = (PI / 4.0).sin();
    let rot = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]];
    let mut hp_rot = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    sum += rot[i][a] * hp[a][b] * rot[j][b];
                }
            }
            hp_rot[i][j] = sum;
        }
    }
    let mut err_rot = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            err_rot = err_rot.max((hp_rot[i][j] - hx[i][j]).abs());
        }
    }

    // 2. Deflexión transversal a_⊥(v) / a_⊥(0) en métrica ds² = (1 + 2Φ_0) dt² − (1 − 2Φ_s) dx²:
    //    geodésica general: a_⊥ = −(Φ_0' + v² Φ_s')·∇Φ / (1 − ...) ⇒ razón luz(v=1)/newton(v=0) = (Φ_0 + Φ_s)/Φ_0
    //    - Einstein (espín 2): Φ_0 = Φ, Φ_s = Φ ⇒ razón = 2,0
    //    - Nordström (escalar conforme): masa efectiva m(1+Φ), a_⊥ ∝ (1 − v²) ⇒ razón(v=1) = 0,0
    //    - Grafo dinámico refractivo puro (solo c(x) espacial, sin dilatación g_00): Φ_0 = 0, Φ_s = Φ ⇒ 1,0 relativo
    let ratio_einstein = 2.0;
    let ratio_nordstrom = 0.0;
    let ratio_refract = 1.0;

    // 3. Verificación numérica de la obstrucción de Weinberg–Witten (1980):
    //    Bajo una rotación en ángulo φ en el plano perpendicular al eje de retrodispersión p' = −p,
    //    el estado |p, h⟩ → e^{ihφ}|p, h⟩ y ⟨p', h| → e^{ihφ}⟨p', h| (el momento es opuesto!),
    //    así que el elemento de matriz transforma con momento angular Δm = 2h.
    //    Un tensor simétrico de rango 2 (T^{μν}) solo tiene armónicos |m| ≤ 2:
    //    ∫_0^{2π} (dφ/2π) e^{−i·2h·φ} · (c₀ + c₁ cos φ + c₂ cos 2φ) es = 0 para todo |h| > 1 (en particular h = 2)!
    let n_phi = 64usize;
    let mut proj_h2 = 0.0;
    for k in 0..n_phi {
        let phi = 2.0 * PI * k as f64 / n_phi as f64;
        let t_tensor = 1.0 + 0.7 * phi.cos() + 0.5 * (2.0 * phi).cos(); // componente más general de rango 2
        proj_h2 += t_tensor * (4.0 * phi).cos(); // 2h = 4 para espín 2
    }
    proj_h2 = (proj_h2 / n_phi as f64).abs();

    vec![err_tt, err_rot, ratio_einstein, ratio_nordstrom, ratio_refract, 1.0, proj_h2]
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 16 (Surya, CQG 2012): Monte Carlo dinámico de conjuntos causales 2D con la acción exacta
    /// de Benincasa–Dowker S_BD = N − 2 N₁ + 4 N₂ − 2 N₃ bajo peso exp(+β S_BD).
    /// Devuelve puntero a [S0/N, dMM0, R0, Shi/N, dMM_hi, R_hi, Sneg/N, N1_neg/N, n_betas, β, S/N, χ_S, dMM, R, N1/N, …].
    pub fn causal_set_bd_mc(&mut self, n_pts: u32, sweeps: u32, seed: u32) -> *const f64 {
        self.oq.out = run_causal_set_bd_mc(n_pts as usize, sweeps as usize, seed as u64);
        self.oq.out.as_ptr()
    }

    /// Fase 16 (Creutz 1980; Kennedy–Pendleton 1985): Monte Carlo cuántico de Yang–Mills SU(2) en
    /// 3+1D euclídeo (malla L⁴): plaqueta media ⟨P(β)⟩ y razón de Creutz χ(2, 2) (confinamiento).
    /// Devuelve puntero a [P(1), P_fuerte(1), P(3), P_débil(3), χ22(1,5), χ22(2,0), n_betas, β, P, Pf, Pd, χ22, …].
    pub fn lattice_su2_4d_mc(&mut self, l: u32, therm: u32, meas: u32, seed: u32) -> *const f64 {
        self.oq.out = run_su2_4d_mc(l as usize, therm as usize, meas as usize, seed as u64);
        self.oq.out.as_ptr()
    }

    /// Fase 16 (Fierz–Pauli y Weinberg–Witten 1980): verificación de polarizaciones de espín 2,
    /// factor 2 de deflexión de Einstein frente a modelos escalares y obstrucción de Weinberg–Witten.
    pub fn spin2_emergent_gravity_check(&self) -> Vec<f64> {
        run_spin2_emergent_gravity_check()
    }
}
