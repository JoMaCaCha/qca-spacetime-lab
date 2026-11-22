//! Utilidades numéricas compartidas: seno/coseno rápidos renormalizados, color HSV,
#![allow(dead_code)]
//! envolturas toroidales y ajustes sencillos usados por el laboratorio y los tests.

use std::f32::consts::PI as PI32;
use std::f64::consts::PI;

/// (cos x, sin x) con reducción de rango a [−π/4, π/4] y polinomios de Taylor de grado 8/9.
/// El par se renormaliza para que |e^{ix}| = 1 a precisión de f32: así la fase no lineal
/// conserva la norma exactamente (§4.2).
#[inline(always)]
pub fn fast_cos_sin(x: f32) -> (f32, f32) {
    const FRAC_2_PI: f32 = 0.636_619_77;
    const PI_2_HI: f32 = 1.570_796_4;
    const PI_2_LO: f32 = -4.371_139e-8;
    let q = (x * FRAC_2_PI).round();
    let r = (x - q * PI_2_HI) - q * PI_2_LO;
    let r2 = r * r;
    // sin r ≈ r − r³/6 + r⁵/120 − r⁷/5040 + r⁹/362880
    let s = r * (1.0 + r2 * (-1.666_666_7e-1 + r2 * (8.333_333e-3 + r2 * (-1.984_127e-4 + r2 * 2.755_731_9e-6))));
    // cos r ≈ 1 − r²/2 + r⁴/24 − r⁶/720 + r⁸/40320
    let c = 1.0 + r2 * (-0.5 + r2 * (4.166_666_8e-2 + r2 * (-1.388_888_9e-3 + r2 * 2.480_158_8e-5)));
    let (c, s) = match (q as i32) & 3 {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    };
    let inv = 1.0 / (c * c + s * s).sqrt();
    (c * inv, s * inv)
}

/// Versión SIMD 4-ancho (f32x4, sin ramas) de `fast_cos_sin` para wasm32 +simd128.
/// Calcula 4 pares (cos x_i, sin x_i) renormalizados en paralelo.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(crate) unsafe fn fast_cos_sin_x4(x: core::arch::wasm32::v128) -> (core::arch::wasm32::v128, core::arch::wasm32::v128) {
    use core::arch::wasm32::*;
    let frac_2_pi = f32x4_splat(0.636_619_77);
    let pi_2_hi = f32x4_splat(1.570_796_4);
    let pi_2_lo = f32x4_splat(-4.371_139e-8);
    let q = f32x4_nearest(f32x4_mul(x, frac_2_pi));
    let r = f32x4_sub(f32x4_sub(x, f32x4_mul(q, pi_2_hi)), f32x4_mul(q, pi_2_lo));
    let r2 = f32x4_mul(r, r);
    // Polinomios de Taylor grado 9 (sin) y grado 8 (cos)
    let mut s = f32x4_splat(2.755_731_9e-6);
    s = f32x4_add(f32x4_mul(s, r2), f32x4_splat(-1.984_127e-4));
    s = f32x4_add(f32x4_mul(s, r2), f32x4_splat(8.333_333e-3));
    s = f32x4_add(f32x4_mul(s, r2), f32x4_splat(-1.666_666_7e-1));
    s = f32x4_mul(r, f32x4_add(f32x4_mul(s, r2), f32x4_splat(1.0)));

    let mut c = f32x4_splat(2.480_158_8e-5);
    c = f32x4_add(f32x4_mul(c, r2), f32x4_splat(-1.388_888_9e-3));
    c = f32x4_add(f32x4_mul(c, r2), f32x4_splat(4.166_666_8e-2));
    c = f32x4_add(f32x4_mul(c, r2), f32x4_splat(-0.5));
    c = f32x4_add(f32x4_mul(c, r2), f32x4_splat(1.0));

    // Reducción de cuadrante sin ramas:
    // q&3: 0→(c,s), 1→(−s,c), 2→(−c,−s), 3→(s,−c)
    let qi = i32x4_trunc_sat_f32x4(q);
    let swap_mask = i32x4_ne(v128_and(qi, i32x4_splat(1)), i32x4_splat(0));
    let c1 = v128_bitselect(s, c, swap_mask);
    let s1 = v128_bitselect(c, s, swap_mask);
    let sign_c = i32x4_shl(v128_and(i32x4_add(qi, i32x4_splat(1)), i32x4_splat(2)), 30);
    let sign_s = i32x4_shl(v128_and(qi, i32x4_splat(2)), 30);
    let c2 = v128_xor(c1, sign_c);
    let s2 = v128_xor(s1, sign_s);

    let inv = f32x4_div(f32x4_splat(1.0), f32x4_sqrt(f32x4_add(f32x4_mul(c2, c2), f32x4_mul(s2, s2))));
    (f32x4_mul(c2, inv), f32x4_mul(s2, inv))
}

/// Envuelve un ángulo a (−π, π].
#[inline]
pub fn wrap_angle(a: f64) -> f64 {
    let mut x = a % (2.0 * PI);
    if x <= -PI {
        x += 2.0 * PI;
    } else if x > PI {
        x -= 2.0 * PI;
    }
    x
}

/// Envuelve un ángulo en f32 a (−π, π].
#[inline]
pub fn wrap_angle32(a: f32) -> f32 {
    let mut x = a % (2.0 * PI32);
    if x <= -PI32 {
        x += 2.0 * PI32;
    } else if x > PI32 {
        x -= 2.0 * PI32;
    }
    x
}

/// Desplazamiento de imagen mínima en un anillo de longitud `l`.
#[inline]
pub fn min_image(dx: f64, l: f64) -> f64 {
    let mut d = dx % l;
    if d > 0.5 * l {
        d -= l;
    } else if d < -0.5 * l {
        d += l;
    }
    d
}

/// Coordenada entera envuelta en [0, l).
#[inline]
pub fn wrap_i(x: i32, l: usize) -> usize {
    x.rem_euclid(l as i32) as usize
}

/// HSV → RGB con h en radianes (cualquier rango), s y v en [0, 1].
#[inline]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let mut hh = h / (2.0 * PI32);
    hh -= hh.floor();
    let h6 = hh * 6.0;
    let i = h6.floor();
    let f = h6 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// Ajuste por mínimos cuadrados de y = a + b·x + c·x²; devuelve (a, b, c).
pub fn fit_quadratic(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let (mut sx, mut sx2, mut sx3, mut sx4, mut sy, mut sxy, mut sx2y) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys) {
        let x2 = x * x;
        sx += x;
        sx2 += x2;
        sx3 += x2 * x;
        sx4 += x2 * x2;
        sy += y;
        sxy += x * y;
        sx2y += x2 * y;
    }
    let m = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]];
    let r = [sy, sxy, sx2y];
    let sol = solve3(m, r);
    (sol[0], sol[1], sol[2])
}

/// Ajuste lineal y = a + b·x; devuelve (a, b).
pub fn fit_linear(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        sxx += (x - mx) * (x - mx);
        sxy += (x - mx) * (y - my);
    }
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (my - b * mx, b)
}

/// Resuelve un sistema 3×3 por eliminación de Gauss con pivoteo parcial.
pub fn solve3(mut m: [[f64; 3]; 3], mut r: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let mut piv = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        m.swap(col, piv);
        r.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-300 {
            continue;
        }
        for row in col + 1..3 {
            let f = m[row][col] / d;
            for k in col..3 {
                m[row][k] -= f * m[col][k];
            }
            r[row] -= f * r[col];
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        let mut s = r[i];
        for k in i + 1..3 {
            s -= m[i][k] * x[k];
        }
        x[i] = if m[i][i].abs() > 1e-300 { s / m[i][i] } else { 0.0 };
    }
    x
}

/// Ajuste de una sinusoide y ≈ c + a·cos(Ωt) + b·sin(Ωt): barrido de Ω en [w_lo, w_hi]
/// y refinamiento por sección dorada del residuo. Devuelve (Ω, amplitud).
pub fn fit_sinusoid(ts: &[f64], ys: &[f64], w_lo: f64, w_hi: f64) -> (f64, f64) {
    let resid = |w: f64| -> (f64, f64) {
        // mínimos cuadrados lineales en (c, a, b)
        let mut m = [[0.0; 3]; 3];
        let mut r = [0.0; 3];
        for (&t, &y) in ts.iter().zip(ys) {
            let f = [1.0, (w * t).cos(), (w * t).sin()];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += f[i] * f[j];
                }
                r[i] += f[i] * y;
            }
        }
        let p = solve3(m, r);
        let mut s = 0.0;
        for (&t, &y) in ts.iter().zip(ys) {
            let e = y - (p[0] + p[1] * (w * t).cos() + p[2] * (w * t).sin());
            s += e * e;
        }
        (s, (p[1] * p[1] + p[2] * p[2]).sqrt())
    };
    let n_scan = 400;
    let mut best = (f64::INFINITY, w_lo);
    for i in 0..=n_scan {
        let w = w_lo + (w_hi - w_lo) * i as f64 / n_scan as f64;
        let (s, _) = resid(w);
        if s < best.0 {
            best = (s, w);
        }
    }
    let dw = (w_hi - w_lo) / n_scan as f64;
    let (mut a, mut b) = ((best.1 - dw).max(w_lo * 0.5), best.1 + dw);
    let gr = 0.618_033_988_75;
    for _ in 0..60 {
        let c = b - gr * (b - a);
        let d = a + gr * (b - a);
        if resid(c).0 < resid(d).0 {
            b = d;
        } else {
            a = c;
        }
    }
    let w = 0.5 * (a + b);
    (w, resid(w).1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fast_trig_matches_std() {
        let mut x = -12.0f32;
        while x < 12.0 {
            let (c, s) = fast_cos_sin(x);
            assert!((c as f64 - (x as f64).cos()).abs() < 5e-7, "cos {x}");
            assert!((s as f64 - (x as f64).sin()).abs() < 5e-7, "sin {x}");
            assert!(((c as f64 * c as f64 + s as f64 * s as f64) - 1.0).abs() < 3e-7);
            x += 0.0137;
        }
    }
}

/// Ajuste y ≈ c + d·t + a·cos(Ωt) + b·sin(Ωt) (sinusoide con deriva lineal).
/// Devuelve (Ω, amplitud, deriva d).
pub fn fit_sinusoid_drift(ts: &[f64], ys: &[f64], w_lo: f64, w_hi: f64) -> (f64, f64, f64) {
    let fit = |w: f64| -> (f64, [f64; 4]) {
        let mut m = [[0.0f64; 4]; 4];
        let mut r = [0.0f64; 4];
        for (&t, &y) in ts.iter().zip(ys) {
            let f = [1.0, t, (w * t).cos(), (w * t).sin()];
            for i in 0..4 {
                for j in 0..4 {
                    m[i][j] += f[i] * f[j];
                }
                r[i] += f[i] * y;
            }
        }
        let p = solve_n::<4>(m, r);
        let mut s = 0.0;
        for (&t, &y) in ts.iter().zip(ys) {
            let e = y - (p[0] + p[1] * t + p[2] * (w * t).cos() + p[3] * (w * t).sin());
            s += e * e;
        }
        (s, p)
    };
    let n_scan = 400;
    let mut best = (f64::INFINITY, w_lo);
    for i in 0..=n_scan {
        let w = w_lo + (w_hi - w_lo) * i as f64 / n_scan as f64;
        let (s, _) = fit(w);
        if s < best.0 {
            best = (s, w);
        }
    }
    let dw = (w_hi - w_lo) / n_scan as f64;
    let (mut a, mut b) = (best.1 - dw, best.1 + dw);
    let gr = 0.618_033_988_75;
    for _ in 0..60 {
        let c = b - gr * (b - a);
        let d = a + gr * (b - a);
        if fit(c).0 < fit(d).0 {
            b = d;
        } else {
            a = c;
        }
    }
    let w = 0.5 * (a + b);
    let p = fit(w).1;
    (w, (p[2] * p[2] + p[3] * p[3]).sqrt(), p[1])
}

/// Eliminación de Gauss con pivoteo parcial para un sistema N×N.
pub fn solve_n<const N: usize>(mut m: [[f64; N]; N], mut r: [f64; N]) -> [f64; N] {
    for col in 0..N {
        let mut piv = col;
        for row in col + 1..N {
            if m[row][col].abs() > m[piv][col].abs() {
                piv = row;
            }
        }
        m.swap(col, piv);
        r.swap(col, piv);
        let d = m[col][col];
        if d.abs() < 1e-300 {
            continue;
        }
        for row in col + 1..N {
            let f = m[row][col] / d;
            for k in col..N {
                m[row][k] -= f * m[col][k];
            }
            r[row] -= f * r[col];
        }
    }
    let mut x = [0.0; N];
    for i in (0..N).rev() {
        let mut s = r[i];
        for k in i + 1..N {
            s -= m[i][k] * x[k];
        }
        x[i] = if m[i][i].abs() > 1e-300 { s / m[i][i] } else { 0.0 };
    }
    x
}
