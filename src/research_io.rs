//! Fase 18 — Infraestructura común de investigación reproducible:
//! - Estadística rigurosa de series temporales de Monte Carlo: tiempo de autocorrelación integrado
//!   (ventana automática de Madras–Sokal), *block bootstrap* móvil, susceptibilidad / calor específico
//!   y cumulante de Binder.
//! - Exportación reproducible (solo destino nativo) a CSV y JSON en `datos_investigacion/` con
//!   metadatos de versión, parámetros y semilla exacta, más soporte de *checkpoints* reanudables.

use crate::rng::SplitMix64;

/// Media y error estándar ingenuo (sin corregir autocorrelación).
pub fn mean_and_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let mu = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mu) * (x - mu)).sum::<f64>() / (n - 1.0).max(1.0);
    (mu, var.sqrt())
}

/// Tiempo de autocorrelación integrado τ_int = ½ + Σ_{t=1}^W ρ(t) mediante ventana automática
/// de Madras–Sokal (corta cuando t ≥ 6·τ_int o ρ(t) ≤ 0).
pub fn integrated_autocorr_time(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 8 {
        return 0.5;
    }
    let mu = xs.iter().sum::<f64>() / n as f64;
    let c0: f64 = xs.iter().map(|&x| (x - mu) * (x - mu)).sum::<f64>() / n as f64;
    if c0 <= 1e-30 {
        return 0.5;
    }
    let max_lag = (n / 3).min(200);
    let mut tau = 0.5;
    for lag in 1..=max_lag {
        let mut num = 0.0;
        let cnt = n - lag;
        for i in 0..cnt {
            num += (xs[i] - mu) * (xs[i + lag] - mu);
        }
        let rho = (num / cnt as f64) / c0;
        if rho <= 0.0 {
            break;
        }
        tau += rho;
        if (lag as f64) >= 6.0 * tau {
            break;
        }
    }
    tau.max(0.5)
}

/// *Moving-block bootstrap* para media e incertidumbre robusta ante autocorrelación temporal.
pub fn block_bootstrap_mean_err(xs: &[f64], block_len: usize, n_boot: usize, seed: u64) -> (f64, f64) {
    let n = xs.len();
    if n < 4 {
        let (m, s) = mean_and_std(xs);
        return (m, s / (n.max(1) as f64).sqrt());
    }
    let mu = xs.iter().sum::<f64>() / n as f64;
    let b = block_len.clamp(1, (n / 2).max(1));
    let n_blocks = n.div_ceil(b);
    let max_start = n - b + 1;
    let mut rng = SplitMix64::new(seed ^ 0xB007_574A);
    let mut boots = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        let mut acc = 0.0;
        let mut taken = 0usize;
        for _ in 0..n_blocks {
            let st = (rng.next_u64() as usize) % max_start;
            let len = b.min(n - taken);
            for k in 0..len {
                acc += xs[st + k];
            }
            taken += len;
        }
        boots.push(acc / taken as f64);
    }
    let (_, err) = mean_and_std(&boots);
    (mu, err)
}

/// Susceptibilidad χ_S = Var(S)/N, calor específico C_v = β²·χ_S y cumulante de Binder
/// B_4 = 1 − ⟨(S − S̄)⁴⟩ / (3 ⟨(S − S̄)²⟩²) (cero para gaussiana pura, → 2/3 en coexistencia bimodal).
pub fn susceptibility_cv_binder(s_total: &[f64], beta: f64, n_elem: usize) -> (f64, f64, f64) {
    let n = s_total.len();
    if n < 4 {
        return (0.0, 0.0, 0.0);
    }
    let mu = s_total.iter().sum::<f64>() / n as f64;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &x in s_total {
        let d = x - mu;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n as f64;
    m4 /= n as f64;
    let chi = m2 / n_elem.max(1) as f64;
    let cv = beta * beta * chi;
    let binder = if m2 > 1e-24 { 1.0 - m4 / (3.0 * m2 * m2) } else { 0.0 };
    (chi, cv, binder)
}

/// Regresión lineal ordinaria y = a + b·x devolviendo (intercepto a, pendiente b, R²).
pub fn lin_fit_r2(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let nf = n as f64;
    let mx = xs[..n].iter().sum::<f64>() / nf;
    let my = ys[..n].iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let b = if sxx > 1e-30 { sxy / sxx } else { 0.0 };
    let a = my - b * mx;
    let r2 = if sxx > 1e-30 && syy > 1e-30 {
        (sxy * sxy) / (sxx * syy)
    } else {
        1.0
    };
    (a, b, r2.clamp(0.0, 1.0))
}

/// Exportación reproducible de tablas CSV y resumen JSON en `datos_investigacion/` (solo nativo).
#[cfg(not(target_arch = "wasm32"))]
pub fn write_research_file(rel_path: &str, content: &str) -> std::io::Result<()> {
    let p = std::path::Path::new("datos_investigacion").join(rel_path);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(p, content)
}

/// Lectura de archivo o checkpoint en `datos_investigacion/` (solo nativo).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_research_file(rel_path: &str) -> Option<String> {
    let p = std::path::Path::new("datos_investigacion").join(rel_path);
    std::fs::read_to_string(p).ok()
}
