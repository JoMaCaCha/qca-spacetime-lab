//! Álgebra lineal densa compleja para los espectros exactos del paso (§2.13 y §3.10): reducción
//! de Hessenberg con reflexiones de Householder, iteración QR con desplazamiento de Wilkinson hasta
//! la forma de Schur A = Z·T·Z†, y autovectores por sustitución hacia atrás en T.

use rustfft::num_complex::Complex;

pub type C = Complex<f64>;

/// Autovalores y autovectores (normalizados) de una matriz n×n (fila mayor).
pub struct Eigen {
    pub values: Vec<C>,
    pub vectors: Vec<Vec<C>>,
}

#[inline]
fn cz() -> C {
    Complex::new(0.0, 0.0)
}

/// Autovalores y autovectores de `a` (n×n, fila mayor). `a` se destruye.
pub fn eig(a: Vec<C>, n: usize) -> Eigen {
    schur_eig(a, n, true)
}

/// Solo los autovalores de `a` (n×n, fila mayor): la misma iteración QR sin acumular Z.
pub fn eigvals(a: Vec<C>, n: usize) -> Vec<C> {
    schur_eig(a, n, false).values
}

fn schur_eig(mut a: Vec<C>, n: usize, want_vectors: bool) -> Eigen {
    let idx = |i: usize, j: usize| i * n + j;
    let mut z = vec![cz(); if want_vectors { n * n } else { 0 }];
    for i in 0..z.len().min(n) {
        z[idx(i, i)] = Complex::new(1.0, 0.0);
    }
    // 1) Hessenberg: A ← P·A·P con P = I − 2vv†, acumulando Z ← Z·P
    for k in 0..n.saturating_sub(2) {
        let norm: f64 = (k + 1..n).map(|i| a[idx(i, k)].norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-300 {
            continue;
        }
        let x0 = a[idx(k + 1, k)];
        let ph = if x0.norm() > 0.0 { x0 / x0.norm() } else { Complex::new(1.0, 0.0) };
        let alpha = -ph * norm;
        let mut v: Vec<C> = (k + 1..n).map(|i| a[idx(i, k)]).collect();
        v[0] -= alpha;
        let vn: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if vn < 1e-300 {
            continue;
        }
        for c in v.iter_mut() {
            *c /= vn;
        }
        // filas k+1..n: A ← A − 2v(v†A)
        for j in 0..n {
            let mut s = cz();
            for (t, i) in (k + 1..n).enumerate() {
                s += v[t].conj() * a[idx(i, j)];
            }
            for (t, i) in (k + 1..n).enumerate() {
                a[idx(i, j)] -= v[t] * s * 2.0;
            }
        }
        // columnas k+1..n: A ← A − 2(Av)v†, y lo mismo en Z
        for m in [&mut a, &mut z] {
            if m.is_empty() {
                continue;
            }
            for i in 0..n {
                let mut s = cz();
                for (t, j) in (k + 1..n).enumerate() {
                    s += m[idx(i, j)] * v[t];
                }
                for (t, j) in (k + 1..n).enumerate() {
                    m[idx(i, j)] -= s * v[t].conj() * 2.0;
                }
            }
        }
    }
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            a[idx(i, j)] = cz();
        }
    }
    // 2) QR con desplazamiento sobre el bloque activo [lo, hi]
    let eps = 1e-15;
    if n > 1 {
        let mut hi = n - 1;
        let mut iter = 0usize;
        let mut total = 0usize;
        let mut rots: Vec<(C, C, f64)> = Vec::with_capacity(n);
        while hi > 0 && total < 100 * n {
            let mut lo = hi;
            while lo > 0 {
                let s = a[idx(lo - 1, lo - 1)].norm() + a[idx(lo, lo)].norm();
                if a[idx(lo, lo - 1)].norm() <= eps * s.max(1e-300) {
                    a[idx(lo, lo - 1)] = cz();
                    break;
                }
                lo -= 1;
            }
            if lo == hi {
                hi -= 1;
                iter = 0;
                continue;
            }
            iter += 1;
            total += 1;
            // desplazamiento de Wilkinson (excepcional cada 10 iteraciones sin converger)
            let (p, q, r, s) = (a[idx(hi - 1, hi - 1)], a[idx(hi - 1, hi)], a[idx(hi, hi - 1)], a[idx(hi, hi)]);
            let mu = if iter % 10 == 0 {
                s + Complex::new(a[idx(hi, hi - 1)].norm(), 0.0)
            } else {
                let half = (p - s) * 0.5;
                let disc = (half * half + q * r).sqrt();
                let m1 = (p + s) * 0.5 + disc;
                let m2 = (p + s) * 0.5 - disc;
                if (m1 - s).norm() < (m2 - s).norm() {
                    m1
                } else {
                    m2
                }
            };
            for i in lo..=hi {
                a[idx(i, i)] -= mu;
            }
            rots.clear();
            for k in lo..hi {
                let x = a[idx(k, k)];
                let y = a[idx(k + 1, k)];
                let rr = (x.norm_sqr() + y.norm_sqr()).sqrt();
                if rr < 1e-300 {
                    rots.push((Complex::new(1.0, 0.0), cz(), 1.0));
                    continue;
                }
                rots.push((x, y, rr));
                // filas k, k+1 (columnas k..n): (u, w) → ((x̄u + ȳw)/r, (−yu + xw)/r)
                for j in k..n {
                    let (u, w) = (a[idx(k, j)], a[idx(k + 1, j)]);
                    a[idx(k, j)] = (x.conj() * u + y.conj() * w) / rr;
                    a[idx(k + 1, j)] = (-y * u + x * w) / rr;
                }
            }
            for (t, k) in (lo..hi).enumerate() {
                let (x, y, rr) = rots[t];
                // columnas k, k+1 por G†: (u, w) → ((xu + yw)/r, (−ȳu + x̄w)/r)
                let top = (k + 2).min(hi);
                for i in 0..=top {
                    let (u, w) = (a[idx(i, k)], a[idx(i, k + 1)]);
                    a[idx(i, k)] = (x * u + y * w) / rr;
                    a[idx(i, k + 1)] = (-y.conj() * u + x.conj() * w) / rr;
                }
                if want_vectors {
                    for i in 0..n {
                        let (u, w) = (z[idx(i, k)], z[idx(i, k + 1)]);
                        z[idx(i, k)] = (x * u + y * w) / rr;
                        z[idx(i, k + 1)] = (-y.conj() * u + x.conj() * w) / rr;
                    }
                }
            }
            for i in lo..=hi {
                a[idx(i, i)] += mu;
            }
        }
    }
    // 3) autovectores de T (triangular superior) y vuelta a la base original
    let values: Vec<C> = (0..n).map(|i| a[idx(i, i)]).collect();
    if !want_vectors {
        return Eigen { values, vectors: Vec::new() };
    }
    let scale = values.iter().map(|v| v.norm()).fold(1e-300, f64::max);
    let mut vectors = Vec::with_capacity(n);
    for k in 0..n {
        let mut y = vec![cz(); n];
        y[k] = Complex::new(1.0, 0.0);
        for i in (0..k).rev() {
            let mut s = cz();
            for j in (i + 1)..=k {
                s += a[idx(i, j)] * y[j];
            }
            let mut den = a[idx(i, i)] - values[k];
            if den.norm() < 1e-14 * scale {
                den = Complex::new(1e-14 * scale, 0.0);
            }
            y[i] = -s / den;
        }
        let mut v = vec![cz(); n];
        for i in 0..n {
            let mut s = cz();
            for j in 0..=k {
                s += z[idx(i, j)] * y[j];
            }
            v[i] = s;
        }
        let nn: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
        for c in v.iter_mut() {
            *c /= nn;
        }
        vectors.push(v);
    }
    Eigen { values, vectors }
}

/// Autovalores (ordenados) de una matriz hermítica n×n (fila mayor): reducción de Householder a una
/// tridiagonal hermítica, fases para hacerla real y QL implícito. O(n³), sin autovectores.
pub fn hermitian_eigvals(mut a: Vec<C>, n: usize) -> Vec<f64> {
    let idx = |i: usize, j: usize| i * n + j;
    for k in 0..n.saturating_sub(2) {
        let norm: f64 = (k + 1..n).map(|i| a[idx(i, k)].norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-300 {
            continue;
        }
        let x0 = a[idx(k + 1, k)];
        let phase = if x0.norm() > 0.0 { x0 / x0.norm() } else { Complex::new(1.0, 0.0) };
        let mut v = vec![cz(); n];
        v[k + 1] = x0 + phase * norm;
        for i in k + 2..n {
            v[i] = a[idx(i, k)];
        }
        let vn: f64 = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if vn < 1e-300 {
            continue;
        }
        for z in v.iter_mut() {
            *z /= vn;
        }
        // A ← P·A·P con P = I − 2vv†: p = A·v, K = v†·p, w = p − K·v; A ← A − 2(v·w† + w·v†)
        let mut p = vec![cz(); n];
        for i in 0..n {
            let mut s = cz();
            for j in k + 1..n {
                s += a[idx(i, j)] * v[j];
            }
            p[i] = s;
        }
        let kk: C = (k + 1..n).map(|i| v[i].conj() * p[i]).sum();
        let w: Vec<C> = (0..n).map(|i| p[i] - kk * v[i]).collect();
        for i in 0..n {
            for j in 0..n {
                a[idx(i, j)] -= (v[i] * w[j].conj() + w[i] * v[j].conj()) * 2.0;
            }
        }
    }
    // tridiagonal: diagonal real, subdiagonal en módulo (las fases se quitan con una matriz diagonal)
    let mut d: Vec<f64> = (0..n).map(|i| a[idx(i, i)].re).collect();
    let mut e: Vec<f64> = (0..n).map(|i| if i + 1 < n { a[idx(i + 1, i)].norm() } else { 0.0 }).collect();
    // QL implícito (tqli)
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-15 * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut i = m;
            let mut early = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            if early {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_residuals() {
        let mut rng = crate::rng::SplitMix64::new(11);
        for &n in &[2usize, 7, 40] {
            let a: Vec<C> = (0..n * n).map(|_| Complex::new(rng.gauss(), rng.gauss())).collect();
            let e = eig(a.clone(), n);
            for k in 0..n {
                let v = &e.vectors[k];
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut s = Complex::new(0.0, 0.0);
                    for j in 0..n {
                        s += a[i * n + j] * v[j];
                    }
                    worst = worst.max((s - e.values[k] * v[i]).norm());
                }
                assert!(worst < 1e-8, "n = {n}, k = {k}: residuo {worst:e}");
            }
        }
    }

    #[test]
    fn hermitian_eigenvalues_match() {
        let mut rng = crate::rng::SplitMix64::new(5);
        for &n in &[3usize, 17, 60] {
            let mut a = vec![Complex::new(0.0, 0.0); n * n];
            for i in 0..n {
                a[i * n + i] = Complex::new(rng.gauss(), 0.0);
                for j in i + 1..n {
                    let z = Complex::new(rng.gauss(), rng.gauss());
                    a[i * n + j] = z;
                    a[j * n + i] = z.conj();
                }
            }
            let h = hermitian_eigvals(a.clone(), n);
            let mut g: Vec<f64> = eig(a, n).values.iter().map(|v| v.re).collect();
            g.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let err = h.iter().zip(&g).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(err < 1e-9, "n = {n}: {err:e}");
        }
    }
}
