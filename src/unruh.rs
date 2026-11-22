//! Dos problemas sin malla de la gravedad análoga (§2.28) y de la materia oscura autointeractuante
//! (§2.6):
//! - **Unruh de fermiones con el mar lleno.** El mar de Wilson–Dirac 1D (r = 1, m = 0) de la cadena
//!   infinita, con la matriz de correlación por una integral en k, restringido a ℓ sitios. El generador
//!   de Casini–Huerta de red pesa sitios y enlaces con x(ℓ − x)/ℓ (sitios en x = j + ½ del intervalo
//!   [0, ℓ]); sus modos están ocupados según Fermi–Dirac con T = 1/2π (Bisognano–Wichmann). Con
//!   dt > 0 el mar es el de Floquet del Trotter simétrico de §2.25 (celda de dos sitios).
//! - **Núcleo de Thomas–Fermi.** Lane–Emden con saturación, h″ + 2h′/r = −4πγ·ρ(h) con h = |g|·f(ρ),
//!   frente al estado fundamental de Gross–Pitaevskii–Poisson radial con la misma N (tiempo
//!   imaginario con la cinética implícita), los dos en el espacio infinito y sin restar ⟨ρ⟩.

use crate::linalg::{self, C};
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

#[inline]
fn cz() -> C {
    C::new(0.0, 0.0)
}

/// exp(A) de una matriz n×n por escalado y cuadrados con Taylor de 20 términos.
fn expm(a: &[C], n: usize) -> Vec<C> {
    let nrm = (0..n).map(|i| (0..n).map(|j| a[i * n + j].norm()).sum::<f64>()).fold(0.0, f64::max);
    let s = if nrm > 0.5 { (nrm / 0.5).log2().ceil() as i32 } else { 0 };
    let sc = 0.5f64.powi(s);
    let b: Vec<C> = a.iter().map(|z| z * sc).collect();
    let mul = |x: &[C], y: &[C]| -> Vec<C> {
        let mut r = vec![cz(); n * n];
        for i in 0..n {
            for k in 0..n {
                let xv = x[i * n + k];
                for j in 0..n {
                    r[i * n + j] += xv * y[k * n + j];
                }
            }
        }
        r
    };
    let mut res = vec![cz(); n * n];
    let mut term = vec![cz(); n * n];
    for i in 0..n {
        res[i * n + i] = C::new(1.0, 0.0);
        term[i * n + i] = C::new(1.0, 0.0);
    }
    for k in 1..=20 {
        term = mul(&term, &b);
        for z in term.iter_mut() {
            *z /= k as f64;
        }
        for (r, t) in res.iter_mut().zip(&term) {
            *r += t;
        }
    }
    for _ in 0..s {
        res = mul(&res, &res);
    }
    res
}

/// Hopping de Wilson–Dirac c_x†·A·c_{x+1} con A = −(r/2)σ_z − (i/2)σ_x (base de espín (↑, ↓)).
fn hop(r: f64) -> [C; 4] {
    [C::new(-0.5 * r, 0.0), C::new(0.0, -0.5), C::new(0.0, -0.5), C::new(0.5 * r, 0.0)]
}

/// Proyector del mar en la cadena infinita, P(d) = (1/N_k)·Σ_k e^{ikd}·P₋(k), para |d| < ℓ.
/// dt = 0: P₋ del hamiltoniano H(k) = sen k·σ_x + (m + r(1 − cos k))·σ_z.
/// dt > 0: celda de dos sitios y U(K) = E(dt/2)·O(dt/2)·L(dt)·O(dt/2)·E(dt/2), con las bandas de
/// cuasi-energía negativa llenas. Devuelve la matriz 2ℓ × 2ℓ del intervalo (sitio, espín).
fn sea_correlation(ell: usize, m: f64, r: f64, dt: f64, nk: usize) -> Vec<C> {
    let n2 = 2 * ell;
    let mut cmat = vec![cz(); n2 * n2];
    if dt <= 0.0 {
        let maxd = ell as i64;
        let mut p = vec![[cz(); 4]; 2 * ell + 1];
        for j in 0..nk {
            let k = 2.0 * PI * (j as f64 + 0.5) / nk as f64;
            let (hx, hz) = (k.sin(), m + r * (1.0 - k.cos()));
            let e = (hx * hx + hz * hz).sqrt().max(1e-300);
            // P₋ = (1 − (hx σx + hz σz)/e)/2
            let pk = [C::new(0.5 * (1.0 - hz / e), 0.0), C::new(-0.5 * hx / e, 0.0), C::new(-0.5 * hx / e, 0.0), C::new(0.5 * (1.0 + hz / e), 0.0)];
            for d in -maxd..=maxd {
                let ph = C::new((k * d as f64).cos(), (k * d as f64).sin()) / nk as f64;
                let slot = &mut p[(d + maxd) as usize];
                for a in 0..4 {
                    slot[a] += pk[a] * ph;
                }
            }
        }
        for x in 0..ell {
            for y in 0..ell {
                let d = x as i64 - y as i64 + maxd;
                let pd = p[d as usize];
                for a in 0..2 {
                    for b in 0..2 {
                        cmat[(2 * x + a) * n2 + 2 * y + b] = pd[2 * a + b];
                    }
                }
            }
        }
        return cmat;
    }
    // Floquet: celda (a, b) = sitios (2j, 2j + 1); índices 0..4 = (a↑, a↓, b↑, b↓)
    let a = hop(r);
    let ad = [a[0].conj(), a[2].conj(), a[1].conj(), a[3].conj()];
    let ncell = ell / 2 + 2;
    let maxd = ncell as i64;
    let mut p = vec![vec![cz(); 16]; 2 * ncell + 1];
    let mi = C::new(0.0, -1.0);
    // E: enlace dentro de la celda (b ← a por A†, a ← b por A)
    let mut he = vec![cz(); 16];
    for s in 0..2 {
        for t in 0..2 {
            he[s * 4 + 2 + t] = a[s * 2 + t];
            he[(2 + s) * 4 + t] = ad[s * 2 + t];
        }
    }
    let ue = expm(&he.iter().map(|z| z * mi * (0.5 * dt)).collect::<Vec<_>>(), 4);
    let mut ul = vec![cz(); 16];
    let w = dt * (m + r);
    for (i, sgn) in [1.0, -1.0, 1.0, -1.0].iter().enumerate() {
        ul[i * 4 + i] = C::new((sgn * w).cos(), -(sgn * w).sin());
    }
    let mul4 = |x: &[C], y: &[C]| -> Vec<C> {
        let mut rr = vec![cz(); 16];
        for i in 0..4 {
            for k in 0..4 {
                for j in 0..4 {
                    rr[i * 4 + j] += x[i * 4 + k] * y[k * 4 + j];
                }
            }
        }
        rr
    };
    for j in 0..nk {
        let kk = 2.0 * PI * (j as f64 + 0.5) / nk as f64;
        let (ep, em) = (C::new(kk.cos(), kk.sin()), C::new(kk.cos(), -kk.sin()));
        // O: enlace b(j) — a(j + 1): (b, a) = A·e^{iK}, (a, b) = A†·e^{−iK}
        let mut ho = vec![cz(); 16];
        for s in 0..2 {
            for t in 0..2 {
                ho[(2 + s) * 4 + t] = a[s * 2 + t] * ep;
                ho[s * 4 + 2 + t] = ad[s * 2 + t] * em;
            }
        }
        let uo = expm(&ho.iter().map(|z| z * mi * (0.5 * dt)).collect::<Vec<_>>(), 4);
        let u = mul4(&mul4(&mul4(&mul4(&ue, &uo), &ul), &uo), &ue);
        let e = linalg::eig(u, 4);
        // las dos bandas de cuasi-energía negativa (λ = e^{−iε·dt}, ε < 0 ⇔ arg λ > 0)
        let mut sel: Vec<Vec<C>> = Vec::new();
        for (l, v) in e.values.iter().zip(e.vectors.iter()) {
            if l.arg() > 0.0 {
                sel.push(v.clone());
            }
        }
        // ortonormaliza (Gram–Schmidt) por si hay casi degeneración
        for i in 0..sel.len() {
            for jj in 0..i {
                let pr: C = sel[jj].iter().zip(&sel[i]).map(|(x, y)| x.conj() * y).sum();
                let (h, t) = sel.split_at_mut(i);
                for c in 0..4 {
                    t[0][c] -= h[jj][c] * pr;
                }
            }
            let nn: f64 = sel[i].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            for z in sel[i].iter_mut() {
                *z /= nn;
            }
        }
        let mut pk = vec![cz(); 16];
        for v in &sel {
            for x in 0..4 {
                for y in 0..4 {
                    pk[x * 4 + y] += v[x] * v[y].conj();
                }
            }
        }
        for d in -maxd..=maxd {
            let ph = C::new((kk * d as f64).cos(), (kk * d as f64).sin()) / nk as f64;
            let slot = &mut p[(d + maxd) as usize];
            for q in 0..16 {
                slot[q] += pk[q] * ph;
            }
        }
    }
    for x in 0..ell {
        for y in 0..ell {
            let (cx, sx) = (x / 2, x % 2);
            let (cy, sy) = (y / 2, y % 2);
            let pd = &p[(cx as i64 - cy as i64 + maxd) as usize];
            for a2 in 0..2 {
                for b2 in 0..2 {
                    cmat[(2 * x + a2) * n2 + 2 * y + b2] = pd[(2 * sx + a2) * 4 + 2 * sy + b2];
                }
            }
        }
    }
    cmat
}

/// Generador de Casini–Huerta de red: Σ β(x)·(m + r)σ_z en los sitios y β(x + ½)·(A + h.c.) en los
/// enlaces, con β(x) = x(ℓ − x)/ℓ y los sitios en x = j + ½.
fn ch_generator(ell: usize, m: f64, r: f64) -> Vec<C> {
    let n2 = 2 * ell;
    let mut k = vec![cz(); n2 * n2];
    let a = hop(r);
    let l = ell as f64;
    let beta = |x: f64| x * (l - x) / l;
    for j in 0..ell {
        let xs = j as f64 + 0.5;
        let b = beta(xs) * (m + r);
        k[(2 * j) * n2 + 2 * j] = C::new(b, 0.0);
        k[(2 * j + 1) * n2 + 2 * j + 1] = C::new(-b, 0.0);
        if j + 1 < ell {
            let bb = beta(xs + 0.5);
            for s in 0..2 {
                for t in 0..2 {
                    k[(2 * j + s) * n2 + 2 * (j + 1) + t] = a[s * 2 + t] * bb;
                    k[(2 * (j + 1) + t) * n2 + 2 * j + s] = a[s * 2 + t].conj() * bb;
                }
            }
        }
    }
    k
}

/// Los `count` niveles de menor |ω| del generador: [ω/T, n] con T = 1/2π.
pub(crate) fn unruh_levels(ell: usize, dt: f64, count: usize) -> Vec<f64> {
    let ell = ell.clamp(4, 400);
    let (m, r) = (0.0, 1.0);
    let cmat = sea_correlation(ell, m, r, dt, 8192);
    let n2 = 2 * ell;
    let e = linalg::eig(ch_generator(ell, m, r), n2);
    let mut idx: Vec<usize> = (0..n2).collect();
    idx.sort_by(|&a, &b| e.values[a].re.abs().partial_cmp(&e.values[b].re.abs()).unwrap());
    let mut out = Vec::with_capacity(2 * count);
    for &j in idx.iter().take(count) {
        let v = &e.vectors[j];
        let mut occ = cz();
        for x in 0..n2 {
            let mut s = cz();
            for y in 0..n2 {
                s += cmat[x * n2 + y] * v[y];
            }
            occ += v[x].conj() * s;
        }
        out.push(2.0 * PI * e.values[j].re);
        out.push(occ.re);
    }
    out
}

/// Lane–Emden con saturación, h″ + 2h′/r = −4πγ·(ρ(h) − ⟨ρ⟩), desde h(0) = |g|·f(ρ₀) hasta h = 0
/// (RK4, paso 0,002). `mean` = ⟨ρ⟩ restada (0 en el espacio infinito). Devuelve (r, ρ, borde).
fn lane_emden(ga: f64, gamma: f64, rho0: f64, emax: f64, mean: f64) -> (Vec<f64>, Vec<f64>, f64) {
    let f = |rho: f64| rho / (1.0 + rho / emax);
    let rho_of_h = |h: f64| {
        let s = (h / ga).max(0.0);
        if s >= emax {
            f64::INFINITY
        } else {
            s / (1.0 - s / emax)
        }
    };
    let h0 = ga * f(rho0);
    let dr = 0.002;
    let mut r = 1e-6;
    let mut y = [h0, -4.0 * PI * gamma * (rho0 - mean) * r / 3.0];
    let rhs = |r: f64, y: [f64; 2]| -> [f64; 2] { [y[1], -4.0 * PI * gamma * (rho_of_h(y[0]) - mean) - 2.0 * y[1] / r] };
    let mut tf_r = vec![0.0];
    let mut tf_rho = vec![rho0];
    let mut edge = f64::NAN;
    while r < 1e4 {
        let k1 = rhs(r, y);
        let k2 = rhs(r + 0.5 * dr, [y[0] + 0.5 * dr * k1[0], y[1] + 0.5 * dr * k1[1]]);
        let k3 = rhs(r + 0.5 * dr, [y[0] + 0.5 * dr * k2[0], y[1] + 0.5 * dr * k2[1]]);
        let k4 = rhs(r + dr, [y[0] + dr * k3[0], y[1] + dr * k3[1]]);
        let yn = [y[0] + dr / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]), y[1] + dr / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])];
        if yn[0] <= 0.0 {
            edge = r + dr * y[0] / (y[0] - yn[0]);
            tf_r.push(edge);
            tf_rho.push(0.0);
            break;
        }
        y = yn;
        r += dr;
        tf_r.push(r);
        tf_rho.push(rho_of_h(y[0]));
    }
    (tf_r, tf_rho, edge)
}

fn shell_mass(tf_r: &[f64], tf_rho: &[f64]) -> f64 {
    let mut n = 0.0;
    for i in 1..tf_r.len() {
        let (a, b) = (tf_r[i - 1], tf_r[i]);
        n += 0.5 * (4.0 * PI * a * a * tf_rho[i - 1] + 4.0 * PI * b * b * tf_rho[i]) * (b - a);
    }
    n
}

/// Thomas–Fermi con saturación y GP–Poisson radial: [N, borde TF, R½ TF, R½ GP, ρ₀ GP], con R½ el
/// radio en que la densidad baja a la mitad de la central.
pub(crate) fn thomas_fermi_profiles(g: f64, gamma: f64, mass_eff: f64, rho0: f64, emax: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ga = g.abs().max(1e-12);
    let f = |rho: f64| rho / (1.0 + rho / emax);
    let (tf_r, tf_rho, edge) = lane_emden(ga, gamma, rho0, emax, 0.0);
    let n_tf = shell_mass(&tf_r, &tf_rho);
    let half_radius = |rs: &[f64], rho: &[f64]| -> f64 {
        let target = 0.5 * rho[0];
        for i in 1..rs.len() {
            if rho[i] <= target {
                let t = (rho[i - 1] - target) / (rho[i - 1] - rho[i]);
                return rs[i - 1] + t * (rs[i] - rs[i - 1]);
            }
        }
        f64::NAN
    };
    let rh_tf = half_radius(&tf_r, &tf_rho);
    // GP radial en una malla escalonada r_i = (i + ½)h hasta 3·borde, Dirichlet fuera
    let hh = 0.05;
    let rmax = 3.0 * if edge.is_finite() { edge } else { 30.0 };
    let n = (rmax / hh) as usize;
    let rg: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) * hh).collect();
    let interp_tf = |x: f64| -> f64 {
        if x >= *tf_r.last().unwrap() {
            return 0.0;
        }
        let i = tf_r.partition_point(|&v| v < x).max(1);
        let t = (x - tf_r[i - 1]) / (tf_r[i] - tf_r[i - 1]);
        tf_rho[i - 1] + t * (tf_rho[i] - tf_rho[i - 1])
    };
    let mut u: Vec<f64> = rg.iter().map(|&x| (interp_tf(x) + 1e-10).sqrt()).collect();
    let dt = 5.0;
    let k2m = 1.0 / (2.0 * mass_eff);
    let mut lo = vec![0.0; n];
    let mut di = vec![0.0; n];
    let mut up = vec![0.0; n];
    for i in 0..n {
        let rp = (i as f64 + 1.0) * hh;
        let rm = i as f64 * hh;
        let den = rg[i] * rg[i] * hh * hh;
        di[i] = 1.0 + dt * k2m * (rp * rp + rm * rm) / den;
        if i + 1 < n {
            up[i] = -dt * k2m * rp * rp / den;
        } else {
            di[i] += dt * k2m * rp * rp / den;
        }
        if i > 0 {
            lo[i] = -dt * k2m * rm * rm / den;
        }
    }
    let wv: Vec<f64> = rg.iter().map(|&x| 4.0 * PI * x * x * hh).collect();
    let mut phi = vec![0.0; n];
    let mut rhs_v = vec![0.0; n];
    let mut cp = vec![0.0; n];
    for it in 0..80000 {
        // Φ(r) = M(<r)/r + ∫_r^∞ 4πρr′dr′
        let mut inner = 0.0;
        for i in 0..n {
            let rho = u[i] * u[i];
            phi[i] = (inner + 0.5 * rho * wv[i]) / rg[i];
            inner += rho * wv[i];
        }
        let mut outer = 0.0;
        for i in (0..n).rev() {
            let rho = u[i] * u[i];
            let own = rho * 4.0 * PI * rg[i] * hh;
            phi[i] += outer + 0.5 * own;
            outer += own;
        }
        for i in 0..n {
            let rho = u[i] * u[i];
            rhs_v[i] = u[i] * (-dt * (ga * f(rho) - gamma * phi[i])).exp();
        }
        // Thomas (tridiagonal)
        cp[0] = up[0] / di[0];
        rhs_v[0] /= di[0];
        for i in 1..n {
            let den = di[i] - lo[i] * cp[i - 1];
            cp[i] = up[i] / den;
            rhs_v[i] = (rhs_v[i] - lo[i] * rhs_v[i - 1]) / den;
        }
        for i in (0..n - 1).rev() {
            rhs_v[i] -= cp[i] * rhs_v[i + 1];
        }
        let nn: f64 = rhs_v.iter().zip(&wv).map(|(a, w)| a * a * w).sum();
        let s = (n_tf / nn).sqrt();
        let mut diff = 0.0f64;
        for i in 0..n {
            let v = rhs_v[i] * s;
            diff = diff.max((v - u[i]).abs());
            u[i] = v;
        }
        if it > 200 && diff < 1e-12 {
            break;
        }
    }
    let gp_rho: Vec<f64> = u.iter().map(|v| v * v).collect();
    let rh_gp = half_radius(&rg, &gp_rho);
    (vec![n_tf, edge, rh_tf, rh_gp, gp_rho[0]], tf_r, tf_rho, rg, gp_rho)
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.28: los 10 niveles más bajos [ω/T, n] del generador de Casini–Huerta (T = 1/2π); dt = 0:
    /// mar del hamiltoniano, dt > 0: mar de Floquet del Trotter simétrico (celda de dos sitios).
    pub fn unruh_occupations(&mut self, ell: u32, dt: f32) -> *const f64 {
        self.an.out = unruh_levels(ell as usize, dt.max(0.0) as f64, 10);
        self.an.out.as_ptr()
    }

    /// Espectro de entrelazamiento ε = ln((1 − ν)/ν) de la correlación del intervalo (los `count`
    /// de menor |ε|), para compararlo con 2π·ω del generador.
    pub fn unruh_entanglement_spectrum(&self, ell: u32, count: u32) -> Vec<f64> {
        let ell = (ell as usize).clamp(4, 400);
        let c = sea_correlation(ell, 0.0, 1.0, 0.0, 8192);
        let mut nu = linalg::hermitian_eigvals(c, 2 * ell);
        let mut eps: Vec<f64> = nu.iter_mut().map(|v| {
            let v = v.clamp(1e-15, 1.0 - 1e-15);
            ((1.0 - v) / v).ln()
        }).collect();
        eps.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        eps.truncate(count as usize);
        eps
    }

    /// §2.6: problema radial en el espacio infinito: [N, borde TF, R½ TF, R½ GP, ρ₀ GP].
    pub fn thomas_fermi_radial(&mut self, g: f32, gamma: f32, mass_eff: f32, rho0: f32) -> *const f64 {
        let (res, ..) = thomas_fermi_profiles(g as f64, gamma as f64, mass_eff as f64, rho0 as f64, self.e_max as f64);
        self.an.out = res;
        self.an.out.as_ptr()
    }

    /// Perfiles del último cálculo radial: [n_TF, r_TF…, ρ_TF…, n_GP, r_GP…, ρ_GP…] (para dibujar).
    pub fn thomas_fermi_profiles(&self, g: f32, gamma: f32, mass_eff: f32, rho0: f32) -> Vec<f64> {
        let (_, tr, trho, gr, grho) = thomas_fermi_profiles(g as f64, gamma as f64, mass_eff as f64, rho0 as f64, self.e_max as f64);
        let step_t = (tr.len() / 400).max(1);
        let step_g = (gr.len() / 400).max(1);
        let tsel: Vec<usize> = (0..tr.len()).step_by(step_t).collect();
        let gsel: Vec<usize> = (0..gr.len()).step_by(step_g).collect();
        let mut out = vec![tsel.len() as f64];
        out.extend(tsel.iter().map(|&i| tr[i]));
        out.extend(tsel.iter().map(|&i| trho[i]));
        out.push(gsel.len() as f64);
        out.extend(gsel.iter().map(|&i| gr[i]));
        out.extend(gsel.iter().map(|&i| grho[i]));
        out
    }

    /// Carga el perfil de Thomas–Fermi con saturación de ρ₀ (espinor simétrico) centrado en la malla:
    /// el de la caja periódica, con ⟨ρ⟩ = N/V restada en la fuente (autoconsistente), como el Poisson
    /// de red. Devuelve [N, borde, R½].
    pub fn seed_thomas_fermi(&mut self, rho0: f32) -> Vec<f64> {
        let ga = (self.g as f64).abs().max(1e-12);
        let gamma = self.grav.gamma_eff() as f64;
        let vol = self.n as f64;
        let mut mean = 0.0;
        let (mut tr, mut trho, mut edge) = lane_emden(ga, gamma, rho0 as f64, self.e_max as f64, mean);
        for _ in 0..6 {
            mean = shell_mass(&tr, &trho) / vol;
            (tr, trho, edge) = lane_emden(ga, gamma, rho0 as f64, self.e_max as f64, mean);
        }
        let n_tf = shell_mass(&tr, &trho);
        let rh = {
            let target = 0.5 * trho[0];
            let i = trho.iter().position(|&v| v <= target).unwrap_or(1).max(1);
            let t = (trho[i - 1] - target) / (trho[i - 1] - trho[i]);
            tr[i - 1] + t * (tr[i] - tr[i - 1])
        };
        let nc = self.nc;
        let stride = self.stride();
        let (cx, cy, cz) = (self.w as f64 / 2.0 - 0.5, self.h as f64 / 2.0 - 0.5, self.d as f64 / 2.0 - 0.5);
        let rmax = *tr.last().unwrap_or(&0.0);
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let (dx, dy, dz) = (x as f64 - cx, y as f64 - cy, if self.dim == 3 { z as f64 - cz } else { 0.0 });
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let rho = if r >= rmax {
                0.0
            } else {
                let j = tr.partition_point(|&v| v < r).max(1);
                let t = (r - tr[j - 1]) / (tr[j] - tr[j - 1]);
                trho[j - 1] + t * (trho[j] - trho[j - 1])
            };
            let a = (rho / nc as f64).sqrt() as f32;
            for k in 0..nc {
                self.buffer_read[i * stride + 2 * k] = a;
                self.buffer_read[i * stride + 2 * k + 1] = 0.0;
            }
        }
        self.t = 0;
        vec![n_tf, edge, rh]
    }

    /// [R½, ρ₀] del núcleo centrado en la malla: ρ₀ media de las 8 celdas centrales y R½ el radio en
    /// que el promedio por capas esféricas (de ancho 1) baja a ρ₀/2.
    pub fn core_half_radius(&self) -> Vec<f64> {
        let (cx, cy, cz) = (self.w as f64 / 2.0 - 0.5, self.h as f64 / 2.0 - 0.5, self.d as f64 / 2.0 - 0.5);
        let nb = self.w.max(self.h).max(self.d);
        let mut sum = vec![0.0f64; nb];
        let mut cnt = vec![0.0f64; nb];
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let (dx, dy, dz) = (x as f64 - cx, y as f64 - cy, if self.dim == 3 { z as f64 - cz } else { 0.0 });
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let b = r.floor() as usize;
            if b < nb {
                sum[b] += self.density(i) as f64;
                cnt[b] += 1.0;
            }
        }
        let prof: Vec<f64> = sum.iter().zip(&cnt).map(|(s, c)| if *c > 0.0 { s / c } else { 0.0 }).collect();
        let rho0 = prof[0];
        let mut rh = f64::NAN;
        for b in 1..nb {
            if prof[b] <= 0.5 * rho0 {
                let (r0, r1) = (b as f64 - 0.5, b as f64 + 0.5);
                let t = (prof[b - 1] - 0.5 * rho0) / (prof[b - 1] - prof[b]);
                rh = r0 + t * (r1 - r0);
                break;
            }
        }
        vec![rh, rho0]
    }
}
