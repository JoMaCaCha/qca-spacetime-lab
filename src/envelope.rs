//! La ecuación efectiva, aprendida de los datos (§2.29): regresión dispersa (SINDy/PDE-FIND con
//! STRidge) de la envolvente de la banda focalizante en una tira W × 1 de la caminata.
//!
//! - Campo inicial: ruido gaussiano complejo filtrado en k (∝ e^{−(k/k_c)²}), proyectado en el espinor
//!   u(k) de la banda en cada k, con ⟨|φ|²⟩ = ρ̄.
//! - La caminata de la tira con g, E_max y α (el promedio de los 4 vecinos de una tira de altura 1 es
//!   E + ∂²E/4), sin tope.
//! - Envolvente φ(x, t) = Σ_k u(k)†ψ_k e^{ikx}, demodulada con e^{imt}; ∂_tφ por diferencias centradas
//!   de 5 puntos y derivadas espaciales en Fourier.
//! - Biblioteca {φ, −i∂φ, ∂²φ, i∂³φ, ∂⁴φ, ρφ, ρ²φ, ρ³φ, (∂²ρ)φ, iρφ, i∂²φ} con coeficientes reales (se
//!   ajustan a la vez la parte real y la imaginaria), por ecuaciones normales con las columnas
//!   normalizadas, y STRidge con un umbral de 10⁻³·‖i∂_tφ‖.

use crate::analog::track_band;
use crate::linalg::C;
use crate::rng::SplitMix64;
use crate::QuantumSpacetimeUniverse;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub const N_TERMS: usize = 11;
const W: usize = 2048;
const RHO: f64 = 0.02;
const K_C: f64 = 0.2;
const T0: usize = 40;
const NT: usize = 60;

/// Resuelve G·x = h (densa, pivoteo parcial). Devuelve None si es singular.
fn solve(mut g: Vec<f64>, mut h: Vec<f64>) -> Option<Vec<f64>> {
    let n = h.len();
    for c in 0..n {
        let p = (c..n).max_by(|&a, &b| g[a * n + c].abs().partial_cmp(&g[b * n + c].abs()).unwrap())?;
        if g[p * n + c].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for k in 0..n {
                g.swap(c * n + k, p * n + k);
            }
            h.swap(c, p);
        }
        for r in c + 1..n {
            let f = g[r * n + c] / g[c * n + c];
            if f != 0.0 {
                for k in c..n {
                    g[r * n + k] -= f * g[c * n + k];
                }
                h[r] -= f * h[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = h[r];
        for k in r + 1..n {
            s -= g[r * n + k] * x[k];
        }
        x[r] = s / g[r * n + r];
    }
    Some(x)
}

/// Mínimos cuadrados restringidos a `active` sobre las ecuaciones normales (g, h).
fn lstsq_subset(g: &[f64], h: &[f64], active: &[bool]) -> Vec<f64> {
    let idx: Vec<usize> = (0..N_TERMS).filter(|&i| active[i]).collect();
    let n = idx.len();
    let mut x = vec![0.0; N_TERMS];
    if n == 0 {
        return x;
    }
    let gs: Vec<f64> = idx.iter().flat_map(|&i| idx.iter().map(move |&j| g[i * N_TERMS + j])).collect();
    let hs: Vec<f64> = idx.iter().map(|&i| h[i]).collect();
    if let Some(sol) = solve(gs, hs) {
        for (k, &i) in idx.iter().enumerate() {
            x[i] = sol[k];
        }
    }
    x
}

/// Coeficientes de la biblioteca, residuo relativo, m* y E_max (14 valores).
pub(crate) fn discover(m: f64, g: f64, emax: f64, alpha: f64, seed: u64) -> Vec<f64> {
    discover_cfg(m, g, emax, alpha, RHO, K_C, seed)
}

/// Versión parametrizada de `discover` con densidad media `rho_bar` y ancho espectral `k_c`
/// configurables (Fase 14: diagnóstico de complejidad efectiva EFT).
pub(crate) fn discover_cfg(m: f64, g: f64, emax: f64, alpha: f64, rho_bar: f64, k_c: f64, seed: u64) -> Vec<f64> {
    // espinores de la banda focalizante en la rejilla de la tira
    let half = W / 2;
    let pos: Vec<f64> = (0..=half).map(|j| 2.0 * PI * j as f64 / W as f64).collect();
    let neg: Vec<f64> = (1..(W - half)).map(|j| -2.0 * PI * j as f64 / W as f64).collect();
    let mut us = vec![[C::new(0.0, 0.0); 4]; W];
    for (j, (u, _)) in track_band(m, 1, &pos).into_iter().enumerate() {
        us[j] = [u[0], u[1], u[2], u[3]];
    }
    for (j, (u, _)) in track_band(m, 1, &neg).into_iter().enumerate() {
        us[W - 1 - j] = [u[0], u[1], u[2], u[3]];
    }
    let kv: Vec<f64> = (0..W).map(|j| 2.0 * PI * (if j < half { j as f64 } else { j as f64 - W as f64 }) / W as f64).collect();
    let mut planner = FftPlanner::new();
    let fwd = planner.plan_fft_forward(W);
    let inv = planner.plan_fft_inverse(W);
    let inv_w = 1.0 / W as f64;
    // campo inicial de la banda
    let mut rng = SplitMix64::new(seed ^ 0x5EED_E17E);
    let mut ak: Vec<C> = (0..W).map(|j| C::new(rng.gauss(), rng.gauss()) * (-(kv[j] / k_c).powi(2)).exp()).collect();
    let mut phi0 = ak.clone();
    inv.process(&mut phi0);
    let mean: f64 = phi0.iter().map(|z| (z * inv_w).norm_sqr()).sum::<f64>() * inv_w;
    let scale = (rho_bar / mean.max(1e-30)).sqrt();
    for z in ak.iter_mut() {
        *z *= scale; // amplitudes en k del φ normalizado (la ifft lleva 1/W)
    }
    let mut psi = vec![[C::new(0.0, 0.0); 4]; W];
    let mut chan = vec![C::new(0.0, 0.0); W];
    for c in 0..4 {
        for j in 0..W {
            chan[j] = ak[j] * us[j][c];
        }
        inv.process(&mut chan);
        for x in 0..W {
            psi[x][c] = chan[x] * inv_w;
        }
    }
    // evolución y envolventes demoduladas en t = T0 − 2 … T0 + NT + 1
    let (cm, sm) = (m.cos(), m.sin());
    let isn = C::new(0.0, sm);
    let mut frames: Vec<Vec<C>> = Vec::with_capacity(NT + 4);
    let mut ee = vec![0.0; W];
    let mut tmp = vec![[C::new(0.0, 0.0); 4]; W];
    let mut chans = vec![vec![C::new(0.0, 0.0); W]; 4];
    for t in 0..T0 + NT + 2 {
        if t + 2 >= T0 {
            for c in 0..4 {
                for x in 0..W {
                    chans[c][x] = psi[x][c];
                }
                fwd.process(&mut chans[c]);
            }
            let mut ph: Vec<C> = (0..W).map(|j| (0..4).map(|c| us[j][c].conj() * chans[c][j]).sum::<C>()).collect();
            inv.process(&mut ph);
            let dm = C::new((m * t as f64).cos(), (m * t as f64).sin()) * inv_w;
            frames.push(ph.iter().map(|z| z * dm).collect());
        }
        for x in 0..W {
            let e: f64 = psi[x].iter().map(|z| z.norm_sqr()).sum();
            ee[x] = e / (1.0 + e / emax);
        }
        for x in 0..W {
            let xp = if x + 1 == W { 0 } else { x + 1 };
            let xm = if x == 0 { W - 1 } else { x - 1 };
            let wnl = if alpha != 0.0 { (1.0 - alpha) * ee[x] + alpha * (ee[xm] + ee[xp] + 2.0 * ee[x]) / 4.0 } else { ee[x] };
            let rot = C::new((g * wnl).cos(), (g * wnl).sin()); // e^{−iθ}, θ = −g·w
            let p = psi[x];
            let s = (p[0] + p[1] + p[2] + p[3]) * 0.5;
            for c in 0..4 {
                tmp[x][c] = (p[c] * cm - isn * (s - p[c])) * rot;
            }
        }
        for x in 0..W {
            let xp = if x + 1 == W { 0 } else { x + 1 };
            let xm = if x == 0 { W - 1 } else { x - 1 };
            psi[x] = [tmp[x][0], tmp[x][1], tmp[xp][2], tmp[xm][3]];
        }
    }
    // ecuaciones normales de la regresión
    let mut gm = vec![0.0f64; N_TERMS * N_TERMS];
    let mut hv = vec![0.0f64; N_TERMS];
    let mut bb = 0.0f64;
    let deriv = |f: &[C], n: i32| -> Vec<C> {
        let mut a = f.to_vec();
        fwd.process(&mut a);
        let ik = |j: usize| C::new(0.0, kv[j]).powi(n);
        for (j, z) in a.iter_mut().enumerate() {
            *z *= ik(j) * inv_w;
        }
        inv.process(&mut a);
        a
    };
    let i1 = C::new(0.0, 1.0);
    for i in 2..2 + NT {
        let f = &frames[i];
        let d1 = deriv(f, 1);
        let d2 = deriv(f, 2);
        let d3 = deriv(f, 3);
        let d4 = deriv(f, 4);
        let r: Vec<C> = f.iter().map(|z| C::new(z.norm_sqr(), 0.0)).collect();
        let r2 = deriv(&r, 2);
        for x in 0..W {
            let dt = (-frames[i + 2][x] + frames[i + 1][x] * 8.0 - frames[i - 1][x] * 8.0 + frames[i - 2][x]) / 12.0;
            let b = i1 * dt;
            let rx = r[x].re;
            let lib = [
                f[x],
                -i1 * d1[x],
                d2[x],
                i1 * d3[x],
                d4[x],
                f[x] * rx,
                f[x] * (rx * rx),
                f[x] * (rx * rx * rx),
                f[x] * r2[x].re,
                i1 * f[x] * rx,
                i1 * d2[x],
            ];
            for part in 0..2 {
                let col = |z: C| if part == 0 { z.re } else { z.im };
                let row: [f64; N_TERMS] = std::array::from_fn(|k| col(lib[k]));
                let bv = col(b);
                for a in 0..N_TERMS {
                    hv[a] += row[a] * bv;
                    for c in 0..N_TERMS {
                        gm[a * N_TERMS + c] += row[a] * row[c];
                    }
                }
                bb += bv * bv;
            }
        }
    }
    // STRidge con las columnas normalizadas
    let nrm: Vec<f64> = (0..N_TERMS).map(|i| gm[i * N_TERMS + i].sqrt().max(1e-300)).collect();
    let gn: Vec<f64> = (0..N_TERMS * N_TERMS).map(|k| gm[k] / (nrm[k / N_TERMS] * nrm[k % N_TERMS])).collect();
    let hn: Vec<f64> = (0..N_TERMS).map(|i| hv[i] / nrm[i]).collect();
    let lam = 1e-3 * bb.sqrt();
    let mut active = [true; N_TERMS];
    let mut xn = lstsq_subset(&gn, &hn, &active);
    for _ in 0..10 {
        let mut changed = false;
        for k in 0..N_TERMS {
            if active[k] && xn[k].abs() < lam {
                active[k] = false;
                changed = true;
            }
        }
        xn = lstsq_subset(&gn, &hn, &active);
        if !changed {
            break;
        }
    }
    let x: Vec<f64> = (0..N_TERMS).map(|k| if active[k] { xn[k] / nrm[k] } else { 0.0 }).collect();
    let mut quad = 0.0;
    for a in 0..N_TERMS {
        for c in 0..N_TERMS {
            quad += x[a] * gm[a * N_TERMS + c] * x[c];
        }
    }
    let lin: f64 = (0..N_TERMS).map(|k| x[k] * hv[k]).sum();
    let res = ((quad - 2.0 * lin + bb).max(0.0) / bb).sqrt();
    let mstar = if x[2] != 0.0 { -1.0 / (2.0 * x[2]) } else { f64::NAN };
    let emax_fit = if x[6] != 0.0 { -x[5] / x[6] } else { f64::NAN };
    let mut out = x;
    out.extend_from_slice(&[res, mstar, emax_fit]);
    out
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.29: regresión dispersa en la tira de 2048 celdas (ρ̄ = 0,02, k_c = 0,2, pasos 40–99): los 11
    /// coeficientes de {φ, −i∂φ, ∂²φ, i∂³φ, ∂⁴φ, ρφ, ρ²φ, ρ³φ, (∂²ρ)φ, iρφ, i∂²φ} (0 = descartado por
    /// STRidge), el residuo relativo, m* = −1/(2·c_∂²) y E_max = −c_ρφ/c_ρ²φ (E_max = 1 en la caminata).
    pub fn discover_envelope_equation(&mut self, m: f32, g: f32, alpha: f32, seed: u32) -> *const f64 {
        let m = if m.is_finite() { (m as f64).clamp(0.05, 1.5) } else { 0.5 };
        let g = if g.is_finite() { (g as f64).clamp(-1.0, 1.0) } else { 0.1 };
        let alpha = if alpha.is_finite() { (alpha as f64).clamp(0.0, 1.0) } else { 0.0 };
        self.an.out = discover(m, g, 1.0, alpha, seed as u64);
        self.an.out.as_ptr()
    }

    /// Fase 14: diagnóstico de complejidad efectiva (compresión EFT frente a la microdinámica)
    /// variando densidad `rho_bar` y corte espectral `k_c`.
    /// Devuelve puntero a [terminos_activos, residuo_rel, bic, compresion_eft, err_m_estrella, err_g, tasa_floquet].
    pub fn envelope_eft_complexity(&mut self, m: f32, g: f32, alpha: f32, rho_bar: f32, k_c: f32, seed: u32) -> *const f64 {
        let mf = if m.is_finite() { (m as f64).clamp(0.05, 1.5) } else { 0.5 };
        let gf = if g.is_finite() { (g as f64).clamp(-1.0, 1.0) } else { 0.1 };
        let af = if alpha.is_finite() { (alpha as f64).clamp(0.0, 1.0) } else { 0.0 };
        let rh = if rho_bar.is_finite() { (rho_bar as f64).clamp(1e-4, 2.0) } else { RHO };
        let kc = if k_c.is_finite() { (k_c as f64).clamp(0.03, 1.5) } else { K_C };
        let res = discover_cfg(mf, gf, 1.0, af, rh, kc, seed as u64);
        let active = res[..N_TERMS].iter().filter(|&&v| v != 0.0).count() as f64;
        let rel = res[11];
        let n_obs = (2 * NT * W) as f64;
        let bic = n_obs * (rel * rel + 1e-30).ln() + active * n_obs.ln();
        let compression = -(rel + 1e-15).log10() / active.max(1.0);
        let m_pred = 2.0 * mf.tan();
        let err_m = if res[12].is_finite() { (res[12] / m_pred - 1.0).abs() } else { f64::NAN };
        let err_g = if gf.abs() > 1e-12 { (-res[5] / gf - 1.0).abs() } else { res[5].abs() };
        let floquet = crate::analog::parametric_rate_impl(mf, gf, rh, 1.0, 1, 256, 0.0).0;
        self.an.out = vec![active, rel, bic, compression, err_m, err_g, floquet];
        self.an.out.as_ptr()
    }
}
