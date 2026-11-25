//! Sistemas cuánticos abiertos de pocos modos (Fase 14): ecuaciones maestras GKLS/Lindblad
//! exactas en f64 para contrastar los límites del campo medio clásico del universo principal.
//!
//! 1. **Falso vacío y bloqueo cuántico (Cosmic Lockdown)** — Christie, Joo, Kaplanek, Vennin y
//!    Wands (*JCAP* 09, 2026, 125): pozo doble asimétrico $V(\phi) = \lambda(\phi^2 - v^2)^2 - \epsilon\phi$
//!    en base de autoestados de energía truncada, comparando partícula clásica (atrapada),
//!    túnel unitario ($\gamma = 0$) y supresión de Zenón cuántico bajo decoherencia Lindblad.
//! 2. **Función de Wigner $W(q,p)$ y negatividad no gaussiana** — Ireland y Vennin (*JCAP* 07,
//!    2026, 042): evolución Lindblad en base de Fock con compresión paramétrica, términos cúbicos
//!    y cuárticos, desfase y baño térmico; verifica el teorema de Hudson ($\mathcal{N}_W = 0$ para
//!    estados puros gaussianos, régimen de TWA), el crecimiento de $\mathcal{N}_W > 0$ con no
//!    gaussianidad y la supresión exponencial de la coherencia por decoherencia térmica.

use crate::linalg::jacobi_symmetric;
use crate::util::clamp_finite;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

type C = Complex<f64>;

#[inline(always)]
fn cx(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

pub struct OpenQuantum {
    pub out: Vec<f64>,
    pub wigner: Vec<f64>,
    pub wigner_n: usize,
    pub wigner_lim: f64,
}

impl OpenQuantum {
    pub fn new() -> Self {
        OpenQuantum { out: Vec::new(), wigner: Vec::new(), wigner_n: 0, wigner_lim: 4.0 }
    }
}

/// Producto de matrices cuadradas complejas m × m (fila mayor) escribiendo en `out`.
fn mat_mul_into(a: &[C], b: &[C], m: usize, out: &mut [C]) {
    let n2 = m * m;
    assert!(a.len() >= n2 && b.len() >= n2 && out.len() >= n2);
    out[..n2].fill(cx(0.0, 0.0));
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        for i in 0..m {
            let out_row = out.as_mut_ptr().add(i * m);
            for k in 0..m {
                let aik = *a.get_unchecked(i * m + k);
                if aik.re == 0.0 && aik.im == 0.0 {
                    continue;
                }
                let a_re = f64x2_splat(aik.re);
                let a_im = f64x2(-aik.im, aik.im);
                let b_row = b.as_ptr().add(k * m);
                for j in 0..m {
                    let bj = v128_load(b_row.add(j) as *const v128);
                    let bj_sw = i64x2_shuffle::<1, 0>(bj, bj);
                    let dst = out_row.add(j) as *mut v128;
                    let acc = v128_load(dst);
                    let prod = f64x2_add(f64x2_mul(a_re, bj), f64x2_mul(a_im, bj_sw));
                    v128_store(dst, f64x2_add(acc, prod));
                }
            }
        }
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        for i in 0..m {
            for k in 0..m {
                let aik = a[i * m + k];
                if aik.re == 0.0 && aik.im == 0.0 {
                    continue;
                }
                for j in 0..m {
                    out[i * m + j] += aik * b[k * m + j];
                }
            }
        }
    }
}

/// Producto de matrices cuadradas complejas m × m (fila mayor).
fn mat_mul(a: &[C], b: &[C], m: usize) -> Vec<C> {
    let mut out = vec![cx(0.0, 0.0); m * m];
    mat_mul_into(a, b, m, &mut out);
    out
}

/// Adjunta A† de una matriz compleja m × m.
fn mat_dag(a: &[C], m: usize) -> Vec<C> {
    let mut out = vec![cx(0.0, 0.0); m * m];
    for i in 0..m {
        for j in 0..m {
            out[j * m + i] = a[i * m + j].conj();
        }
    }
    out
}

/// Buffers temporales reutilizables para evitar asignaciones de heap en cada subpaso RK4.
struct LindbladScratch {
    t1: Vec<C>,
    t2: Vec<C>,
}

impl LindbladScratch {
    fn new(m: usize) -> Self {
        Self {
            t1: vec![cx(0.0, 0.0); m * m],
            t2: vec![cx(0.0, 0.0); m * m],
        }
    }
}

/// Aplica el disipador de Lindblad γ · (L ρ L† − ½ L†L ρ − ½ ρ L†L) y lo suma a `drho`.
#[allow(clippy::too_many_arguments)]
fn add_lindblad_term(
    rho: &[C],
    l: &[C],
    ldag: &[C],
    ldag_l: &[C],
    rate: f64,
    m: usize,
    drho: &mut [C],
    sc: &mut LindbladScratch,
) {
    if !rate.is_finite() || rate <= 0.0 {
        return;
    }
    let n2 = m * m;
    mat_mul_into(l, rho, m, &mut sc.t1);
    mat_mul_into(&sc.t1, ldag, m, &mut sc.t2);
    for k in 0..n2 {
        drho[k] += sc.t2[k] * rate;
    }
    mat_mul_into(ldag_l, rho, m, &mut sc.t1);
    mat_mul_into(rho, ldag_l, m, &mut sc.t2);
    let half_rate = 0.5 * rate;
    for k in 0..n2 {
        drho[k] -= (sc.t1[k] + sc.t2[k]) * half_rate;
    }
}

/// Busca el máximo de barrera y los dos mínimos de V(φ) = λ(φ² − v²)² − ε·φ.
fn double_well_extrema(lambda: f64, v: f64, eps: f64) -> (f64, f64, f64) {
    let newton = |mut x: f64| -> f64 {
        for _ in 0..30 {
            let f = 4.0 * lambda * x * (x * x - v * v) - eps;
            let fp = 4.0 * lambda * (3.0 * x * x - v * v);
            if fp.abs() < 1e-12 {
                break;
            }
            x -= f / fp;
        }
        x
    };
    let phi_f = newton(-v);
    let phi_b = newton(0.0);
    let phi_t = newton(v);
    (phi_f, phi_b, phi_t)
}

#[inline]
fn double_well_v(phi: f64, lambda: f64, v: f64, eps: f64) -> f64 {
    let d = phi * phi - v * v;
    lambda * d * d - eps * phi
}

/// Integrador clásico RK4 en el pozo doble con energía inicial `e0` (en el falso vacío).
/// Devuelve 1.0 si la trayectoria llega a cruzar la barrera `phi_b`, 0.0 si queda atrapada.
fn classical_double_well_crosses(lambda: f64, v: f64, eps: f64, phi_f: f64, phi_b: f64, e_init: f64, t_max: f64, steps: usize) -> f64 {
    let vf = double_well_v(phi_f, lambda, v, eps);
    let p0 = (2.0 * (e_init - vf).max(0.0)).sqrt();
    let mut phi = phi_f;
    let mut p = p0;
    let dt = t_max / steps.max(1) as f64;
    let force = |x: f64| -(4.0 * lambda * x * (x * x - v * v) - eps);
    let mut crossed = false;
    for _ in 0..steps {
        let k1x = p;
        let k1p = force(phi);
        let k2x = p + 0.5 * dt * k1p;
        let k2p = force(phi + 0.5 * dt * k1x);
        let k3x = p + 0.5 * dt * k2p;
        let k3p = force(phi + 0.5 * dt * k2x);
        let k4x = p + dt * k3p;
        let k4p = force(phi + dt * k3x);
        phi += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0;
        p += dt * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0;
        if phi > phi_b {
            crossed = true;
        }
    }
    if crossed { 1.0 } else { 0.0 }
}

/// Evolución Lindblad del pozo doble asimétrico en la base truncada de M autoestados de H.
/// `monitor_kind`: 0 = desfase en posición L = √(2γ)·φ; 1 = monitor de pozo L = √(2γ)·tanh(φ/(0,5v)).
#[allow(clippy::too_many_arguments)]
pub fn run_false_vacuum_lockdown(lambda: f64, v: f64, eps: f64, hbar: f64, gamma_dec: f64, monitor_kind: u32, t_max: f64, steps: usize) -> Vec<f64> {
    let n_grid = 64usize;
    let m_trunc = 20usize;
    let l_box = (2.4 * v).max(2.5);
    let dx = 2.0 * l_box / (n_grid + 1) as f64;
    let xs: Vec<f64> = (0..n_grid).map(|i| -l_box + (i + 1) as f64 * dx).collect();
    let (phi_f, phi_b, _phi_t) = double_well_extrema(lambda, v, eps);
    let v_barrier = double_well_v(phi_b, lambda, v, eps);

    // Hamiltoniano 1D simétrico (diferencias finitas de 2º orden con Dirichlet en ±L)
    let kin = hbar * hbar / (2.0 * dx * dx);
    let mut h_full = vec![0.0f64; n_grid * n_grid];
    for i in 0..n_grid {
        h_full[i * n_grid + i] = 2.0 * kin + double_well_v(xs[i], lambda, v, eps);
        if i + 1 < n_grid {
            h_full[i * n_grid + (i + 1)] = -kin;
            h_full[(i + 1) * n_grid + i] = -kin;
        }
    }
    let (evals, evecs) = jacobi_symmetric(&h_full, n_grid);
    let m = m_trunc.min(n_grid);

    // Estado inicial: estado fundamental del pozo falso aislado (φ <= φ_B)
    let n_left = xs.iter().take_while(|&&x| x <= phi_b).count().max(4);
    let mut h_left = vec![0.0f64; n_left * n_left];
    for i in 0..n_left {
        h_left[i * n_left + i] = 2.0 * kin + double_well_v(xs[i], lambda, v, eps);
        if i + 1 < n_left {
            h_left[i * n_left + (i + 1)] = -kin;
            h_left[(i + 1) * n_left + i] = -kin;
        }
    }
    let (_, left_vecs) = jacobi_symmetric(&h_left, n_left);
    let psi0_grid = &left_vecs[0];

    // Proyección sobre los M primeros autoestados globales
    let mut c0 = vec![0.0f64; m];
    let mut norm2 = 0.0;
    for a in 0..m {
        let mut s = 0.0;
        for i in 0..n_left {
            s += evecs[a][i] * psi0_grid[i];
        }
        c0[a] = s;
        norm2 += s * s;
    }
    let inv_n = 1.0 / norm2.sqrt().max(1e-30);
    for c in c0.iter_mut() {
        *c *= inv_n;
    }
    let e_init: f64 = (0..m).map(|a| c0[a] * c0[a] * evals[a]).sum();
    let cl_cross = classical_double_well_crosses(lambda, v, eps, phi_f, phi_b, e_init, t_max, steps.max(400));

    // Operadores en la base truncada M × M: proyector de vacío verdadero P_true y operador de salto L.
    // monitor_kind = 0: baño de ruido blanco sin corte (L = φ, incluye calentamiento a γ muy alto).
    // monitor_kind = 1: entorno cosmológico frío con corte IR (L = Π_sub·φ·Π_sub proyectado sobre
    //                   los autoestados por debajo de la barrera V_B, sin inyección de energía por
    //                   encima de la barrera: régimen puro de bloqueo de Zenón de Christie et al. 2026).
    let mut p_true = vec![0.0f64; m * m];
    let mut l_mat = vec![cx(0.0, 0.0); m * m];
    for a in 0..m {
        for b in 0..m {
            let mut sp = 0.0;
            let mut sl = 0.0;
            for i in 0..n_grid {
                let prod = evecs[a][i] * evecs[b][i];
                if xs[i] > phi_b {
                    sp += prod;
                }
                sl += prod * xs[i];
            }
            p_true[a * m + b] = sp;
            let keep = monitor_kind != 1 || (evals[a] < v_barrier && evals[b] < v_barrier);
            l_mat[a * m + b] = if keep { cx(sl, 0.0) } else { cx(0.0, 0.0) };
        }
    }
    let ldag = mat_dag(&l_mat, m);
    let ldag_l = mat_mul(&ldag, &l_mat, m);

    // Matriz densidad inicial ρ(0) = |ψ₀⟩⟨ψ₀|
    let mut rho = vec![cx(0.0, 0.0); m * m];
    for a in 0..m {
        for b in 0..m {
            rho[a * m + b] = cx(c0[a] * c0[b], 0.0);
        }
    }

    // Estabilidad: paso de Strang exacto para la evolución unitaria (H es diagonal)
    // y RK4 solo para el disipador de Lindblad cuando γ_dec > 0.
    let l_norm2 = if monitor_kind == 1 { 1.0 } else { l_box * l_box };
    let rate_eff = 2.0 * gamma_dec.max(0.0);
    let min_steps = ((t_max * (4.0 * rate_eff * l_norm2 + 2.0)).ceil() as usize).max(steps).clamp(100, 12000);
    let dt = t_max / min_steps as f64;
    let half_phase: Vec<C> = (0..(m * m))
        .map(|k| {
            let a = k / m;
            let b = k % m;
            C::from_polar(1.0, -(evals[a] - evals[b]) * 0.5 * dt / hbar)
        })
        .collect();

    let mut sc = LindbladScratch::new(m);
    let mut k1 = vec![cx(0.0, 0.0); m * m];
    let mut k2 = vec![cx(0.0, 0.0); m * m];
    let mut k3 = vec![cx(0.0, 0.0); m * m];
    let mut k4 = vec![cx(0.0, 0.0); m * m];

    let measure = |r: &[C]| -> (f64, f64, f64) {
        let mut pt = 0.0;
        let mut tr = 0.0;
        let mut pur = 0.0;
        for a in 0..m {
            tr += r[a * m + a].re;
            for b in 0..m {
                pt += p_true[a * m + b] * r[b * m + a].re;
                pur += r[a * m + b].norm_sqr();
            }
        }
        (pt.clamp(0.0, 1.0), pur, (tr - 1.0).abs())
    };

    let sample_n = 80usize.min(min_steps);
    let sample_every = (min_steps / sample_n).max(1);
    let mut series = Vec::with_capacity(3 * (sample_n + 2));
    let (pt0, pur0, _) = measure(&rho);
    series.extend_from_slice(&[0.0, pt0, pur0]);
    let mut pt_max = pt0;
    let mut pt_int = 0.0;
    let mut max_tr_err = 0.0f64;

    let mut tmp = vec![cx(0.0, 0.0); m * m];
    for step in 1..=min_steps {
        for k in 0..(m * m) {
            rho[k] *= half_phase[k];
        }
        if rate_eff > 0.0 {
            k1.fill(cx(0.0, 0.0));
            add_lindblad_term(&rho, &l_mat, &ldag, &ldag_l, rate_eff, m, &mut k1, &mut sc);
            for k in 0..(m * m) {
                tmp[k] = rho[k] + k1[k] * (0.5 * dt);
            }
            k2.fill(cx(0.0, 0.0));
            add_lindblad_term(&tmp, &l_mat, &ldag, &ldag_l, rate_eff, m, &mut k2, &mut sc);
            for k in 0..(m * m) {
                tmp[k] = rho[k] + k2[k] * (0.5 * dt);
            }
            k3.fill(cx(0.0, 0.0));
            add_lindblad_term(&tmp, &l_mat, &ldag, &ldag_l, rate_eff, m, &mut k3, &mut sc);
            for k in 0..(m * m) {
                tmp[k] = rho[k] + k3[k] * dt;
            }
            k4.fill(cx(0.0, 0.0));
            add_lindblad_term(&tmp, &l_mat, &ldag, &ldag_l, rate_eff, m, &mut k4, &mut sc);
            for k in 0..(m * m) {
                rho[k] += (k1[k] + k2[k] * 2.0 + k3[k] * 2.0 + k4[k]) * (dt / 6.0);
            }
        }
        for k in 0..(m * m) {
            rho[k] *= half_phase[k];
        }
        // Simetrización hermítica para eliminar deriva numérica de redondeo
        for a in 0..m {
            rho[a * m + a].im = 0.0;
            for b in (a + 1)..m {
                let avg = (rho[a * m + b] + rho[b * m + a].conj()) * 0.5;
                rho[a * m + b] = avg;
                rho[b * m + a] = avg.conj();
            }
        }
        let (pt, pur, terr) = measure(&rho);
        pt_max = pt_max.max(pt);
        pt_int += pt * dt;
        max_tr_err = max_tr_err.max(terr);
        if step % sample_every == 0 || step == min_steps {
            series.extend_from_slice(&[step as f64 * dt, pt, pur]);
        }
    }

    let (pt_final, pur_final, _) = measure(&rho);
    let e_final: f64 = (0..m).map(|a| rho[a * m + a].re * evals[a]).sum();
    let p_mean = pt_int / t_max.max(1e-12);
    let n_pts = (series.len() / 3) as f64;
    // Cabecera de 10 valores:
    // [0] p_true_final, [1] p_true_max, [2] p_true_mean, [3] purity_final,
    // [4] max_trace_err, [5] e_init, [6] v_barrier, [7] classical_crosses,
    // [8] e_final, [9] n_pts, seguido de n_pts ternas (t, p_true, purity).
    let mut out = vec![pt_final, pt_max, p_mean, pur_final, max_tr_err, e_init, v_barrier, cl_cross, e_final, n_pts];
    out.extend_from_slice(&series);
    out
}

/// Evalúa la función de Wigner exacta W(q, p) de una matriz densidad ρ en base de Fock (n × n)
/// mediante la recurrencia escalada sin desbordamiento de polinomios de Laguerre asociados.
fn eval_wigner_grid(rho: &[C], n_fock: usize, grid_n: usize, lim: f64) -> (Vec<f64>, f64, f64, f64) {
    let grid_n = grid_n.max(2);
    let mut log_fact = vec![0.0f64; n_fock + 1];
    for k in 1..=n_fock {
        log_fact[k] = log_fact[k - 1] + (k as f64).ln();
    }
    let dq = 2.0 * lim / (grid_n - 1) as f64;
    let mut grid = vec![0.0f64; grid_n * grid_n];
    let mut w_min = f64::INFINITY;
    let mut int_w = 0.0;
    let mut int_abs = 0.0;
    let mut cos_k = vec![0.0f64; n_fock];
    let mut sin_k = vec![0.0f64; n_fock];

    for iy in 0..grid_n {
        let p = -lim + iy as f64 * dq;
        for ix in 0..grid_n {
            let q = -lim + ix as f64 * dq;
            let r2 = q * q + p * p;
            let x = 2.0 * r2;
            let u = x.sqrt();
            let (ct, st) = if r2 > 1e-30 {
                let r = r2.sqrt();
                (q / r, p / r)
            } else {
                (1.0, 0.0)
            };
            cos_k[0] = 1.0;
            sin_k[0] = 0.0;
            for k in 1..n_fock {
                cos_k[k] = cos_k[k - 1] * ct - sin_k[k - 1] * st;
                sin_k[k] = sin_k[k - 1] * ct + cos_k[k - 1] * st;
            }

            let mut w_val = 0.0;
            let ex = (-0.5 * x).exp();
            // k = n - m >= 0
            for k in 0..n_fock {
                let f0 = if k == 0 {
                    ex
                } else if u < 1e-30 {
                    0.0
                } else {
                    (-0.5 * x + (k as f64) * u.ln() - 0.5 * log_fact[k]).exp()
                };
                let mut f_prev2 = 0.0;
                let mut f_prev1 = f0;
                for m in 0..(n_fock - k) {
                    let fm = if m == 0 {
                        f0
                    } else if m == 1 {
                        (1.0 + k as f64 - x) / ((1 + k) as f64).sqrt() * f0
                    } else {
                        let mf = m as f64;
                        let kf = k as f64;
                        let num = (2.0 * mf - 1.0 + kf - x) * f_prev1 - ((mf - 1.0) * (mf - 1.0 + kf)).sqrt() * f_prev2;
                        num / (mf * (mf + kf)).sqrt()
                    };
                    if m >= 1 {
                        f_prev2 = f_prev1;
                        f_prev1 = fm;
                    }
                    let sign = if m & 1 == 0 { 1.0 } else { -1.0 };
                    let n_idx = m + k;
                    let r_mn = rho[m * n_fock + n_idx];
                    if k == 0 {
                        w_val += sign * fm * r_mn.re;
                    } else {
                        // 2 * Re(ρ_{m, m+k} * e^{i k θ})
                        let re_term = r_mn.re * cos_k[k] - r_mn.im * sin_k[k];
                        w_val += 2.0 * sign * fm * re_term;
                    }
                }
            }
            w_val /= PI;
            grid[iy * grid_n + ix] = w_val;
            w_min = w_min.min(w_val);
            let wx = if ix == 0 || ix + 1 == grid_n { 0.5 } else { 1.0 };
            let wy = if iy == 0 || iy + 1 == grid_n { 0.5 } else { 1.0 };
            let wt = wx * wy * dq * dq;
            int_w += wt * w_val;
            int_abs += wt * w_val.abs();
        }
    }
    let neg = (int_abs - 1.0).max(0.0);
    (grid, neg, w_min, int_w)
}

/// Evolución Lindblad en base de Fock (N_F = 24) con compresión ξ, no linealidad g₃ q³ + g₄ q⁴,
/// desfase γ_φ D[q] y baño térmico γ_th [(n_th+1) D[a] + n_th D[a†]].
/// `init_kind`: 0 = vacío |0⟩, 1 = coherente |α₀⟩, 2 = gato de Schrödinger (|α₀⟩ + |−α₀⟩)/√N.
#[allow(clippy::too_many_arguments)]
pub fn run_wigner_open_oscillator(
    omega: f64,
    xi: f64,
    g3: f64,
    g4: f64,
    gamma_phi: f64,
    gamma_th: f64,
    n_th: f64,
    init_kind: u32,
    alpha0: f64,
    t_max: f64,
    steps: usize,
    grid_n: usize,
    lim: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nf = 24usize;
    let inv_s2 = 1.0 / 2.0f64.sqrt();
    let mut a_op = vec![cx(0.0, 0.0); nf * nf];
    for n in 0..(nf - 1) {
        a_op[n * nf + (n + 1)] = cx(((n + 1) as f64).sqrt(), 0.0);
    }
    let adag_op = mat_dag(&a_op, nf);
    let mut q_op = vec![cx(0.0, 0.0); nf * nf];
    let mut p_op = vec![cx(0.0, 0.0); nf * nf];
    for k in 0..(nf * nf) {
        q_op[k] = (a_op[k] + adag_op[k]) * inv_s2;
        p_op[k] = cx(0.0, inv_s2) * (adag_op[k] - a_op[k]);
    }
    let q2 = mat_mul(&q_op, &q_op, nf);
    let q3 = mat_mul(&q2, &q_op, nf);
    let q4 = mat_mul(&q2, &q2, nf);
    let p2 = mat_mul(&p_op, &p_op, nf);
    let a2 = mat_mul(&a_op, &a_op, nf);
    let adag2 = mat_mul(&adag_op, &adag_op, nf);
    let n_op = mat_mul(&adag_op, &a_op, nf);
    let a_adag = mat_mul(&a_op, &adag_op, nf);

    // Hamiltoniano hermítico H = ω a†a + (iξ/2)(a†² − a²) + g₃ q³ + g₄ q⁴
    let mut h_op = vec![cx(0.0, 0.0); nf * nf];
    for n in 0..nf {
        h_op[n * nf + n] += cx(omega * n as f64, 0.0);
    }
    for k in 0..(nf * nf) {
        h_op[k] += cx(0.0, 0.5 * xi) * (adag2[k] - a2[k]) + q3[k] * g3 + q4[k] * g4;
    }

    // Estado inicial |ψ₀⟩ en base de Fock
    let mut psi0 = vec![cx(0.0, 0.0); nf];
    match init_kind {
        1 => {
            // Coherente |α₀⟩
            let mut coef = (-0.5 * alpha0 * alpha0).exp();
            psi0[0] = cx(coef, 0.0);
            for n in 1..nf {
                coef *= alpha0 / (n as f64).sqrt();
                psi0[n] = cx(coef, 0.0);
            }
        }
        2 => {
            // Gato par (|α₀⟩ + |−α₀⟩)/√N: solo Fock pares
            let mut coef = (-0.5 * alpha0 * alpha0).exp();
            psi0[0] = cx(2.0 * coef, 0.0);
            for n in 1..nf {
                coef *= alpha0 / (n as f64).sqrt();
                if n % 2 == 0 {
                    psi0[n] = cx(2.0 * coef, 0.0);
                }
            }
        }
        _ => {
            psi0[0] = cx(1.0, 0.0);
        }
    }
    let nrm: f64 = psi0.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(1e-30);
    for z in psi0.iter_mut() {
        *z /= nrm;
    }
    let mut rho = vec![cx(0.0, 0.0); nf * nf];
    for i in 0..nf {
        for j in 0..nf {
            rho[i * nf + j] = psi0[i] * psi0[j].conj();
        }
    }

    // Estimación conservadora de escala espectral para estabilidad de RK4
    let h_diag_max = (0..nf).map(|n| h_op[n * nf + n].re.abs()).fold(0.0f64, f64::max);
    let rate_scale = h_diag_max + 2.0 * (nf as f64) * (xi.abs() + gamma_phi.abs() + gamma_th.abs() * (2.0 * n_th.abs() + 1.0)) + 1.0;
    let n_steps = if t_max <= 0.0 {
        0
    } else {
        ((t_max * rate_scale * 0.8).ceil() as usize).max(steps).clamp(40, 8000)
    };
    let dt = if n_steps > 0 { t_max / n_steps as f64 } else { 0.0 };

    let mut sc = LindbladScratch::new(nf);
    let mut k1 = vec![cx(0.0, 0.0); nf * nf];
    let mut k2 = vec![cx(0.0, 0.0); nf * nf];
    let mut k3 = vec![cx(0.0, 0.0); nf * nf];
    let mut k4 = vec![cx(0.0, 0.0); nf * nf];
    let mut rhs_into = |r: &[C], dr: &mut [C]| {
        mat_mul_into(&h_op, r, nf, &mut sc.t1);
        mat_mul_into(r, &h_op, nf, &mut sc.t2);
        for k in 0..(nf * nf) {
            let comm = sc.t1[k] - sc.t2[k];
            dr[k] = cx(comm.im, -comm.re);
        }
        add_lindblad_term(r, &q_op, &q_op, &q2, gamma_phi, nf, dr, &mut sc);
        add_lindblad_term(r, &a_op, &adag_op, &n_op, gamma_th * (n_th + 1.0), nf, dr, &mut sc);
        add_lindblad_term(r, &adag_op, &a_op, &a_adag, gamma_th * n_th, nf, dr, &mut sc);
    };

    let mut tmp = vec![cx(0.0, 0.0); nf * nf];
    for _ in 0..n_steps {
        rhs_into(&rho, &mut k1);
        for k in 0..(nf * nf) {
            tmp[k] = rho[k] + k1[k] * (0.5 * dt);
        }
        rhs_into(&tmp, &mut k2);
        for k in 0..(nf * nf) {
            tmp[k] = rho[k] + k2[k] * (0.5 * dt);
        }
        rhs_into(&tmp, &mut k3);
        for k in 0..(nf * nf) {
            tmp[k] = rho[k] + k3[k] * dt;
        }
        rhs_into(&tmp, &mut k4);
        for k in 0..(nf * nf) {
            rho[k] += (k1[k] + k2[k] * 2.0 + k3[k] * 2.0 + k4[k]) * (dt / 6.0);
        }
        for i in 0..nf {
            rho[i * nf + i].im = 0.0;
            for j in (i + 1)..nf {
                let avg = (rho[i * nf + j] + rho[j * nf + i].conj()) * 0.5;
                rho[i * nf + j] = avg;
                rho[j * nf + i] = avg.conj();
            }
        }
    }

    // Observables finales
    let expect_real = |op: &[C]| -> f64 {
        let mut s = 0.0;
        for i in 0..nf {
            for j in 0..nf {
                s += (op[i * nf + j] * rho[j * nf + i]).re;
            }
        }
        s
    };
    let tr: f64 = (0..nf).map(|i| rho[i * nf + i].re).sum();
    let purity: f64 = rho.iter().map(|z| z.norm_sqr()).sum();
    let mean_q = expect_real(&q_op);
    let mean_p = expect_real(&p_op);
    let var_q = (expect_real(&q2) - mean_q * mean_q).max(0.0);
    let var_p = (expect_real(&p2) - mean_p * mean_p).max(0.0);
    let edge_prob = rho[(nf - 1) * nf + (nf - 1)].re.max(0.0);
    let mean_n = expect_real(&n_op);
    let (grid, w_neg, w_min, w_norm) = eval_wigner_grid(&rho, nf, grid_n, lim);
    // [0] w_neg, [1] w_min, [2] purity, [3] var_q, [4] var_p, [5] w_norm, [6] trace_err, [7] edge_prob, [8] mean_n
    let summary = vec![w_neg, w_min, purity, var_q, var_p, w_norm, (tr - 1.0).abs(), edge_prob, mean_n];
    (summary, grid)
}

impl QuantumSpacetimeUniverse {
    /// Acceso seguro en Rust nativo al último vector de salida de los módulos de investigación
    /// (evita `std::slice::from_raw_parts` en `qsu_cli` y pruebas nativas).
    pub fn open_quantum_out_slice(&self) -> &[f64] {
        &self.oq.out
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 14 (Christie et al., JCAP 2026): túnel de falso vacío en pozo doble asimétrico
    /// V(φ) = λ(φ² − v²)² − ε·φ bajo decoherencia de Lindblad.
    /// `monitor_kind`: 0 = desfase lineal sin corte IR (L = √(2γ)·φ, incluye calentamiento térmico a γ alto);
    /// 1 = entorno frío con corte IR por debajo de la barrera (L = √(2γ)·Π_sub φ Π_sub, bloqueo de Zenón puro).
    /// Devuelve puntero a [p_true_final, p_true_max, p_true_mean, purity_final, max_trace_err,
    /// e_init, v_barrier, classical_crosses, e_final, n_pts, t0, p0, pur0, …].
    #[allow(clippy::too_many_arguments)]
    pub fn false_vacuum_lockdown(&mut self, lambda: f32, v: f32, eps: f32, hbar: f32, gamma_dec: f32, monitor_kind: u32, t_max: f32, steps: u32) -> *const f64 {
        let lam = clamp_finite(lambda as f64, 0.05, 5.0, 0.5);
        let vv = clamp_finite(v as f64, 0.5, 3.0, 1.0);
        let ee = clamp_finite(eps as f64, -1.0, 1.0, 0.0);
        let hb = clamp_finite(hbar as f64, 0.1, 2.0, 0.5);
        let gd = clamp_finite(gamma_dec as f64, 0.0, 50.0, 0.0);
        let tm = clamp_finite(t_max as f64, 0.5, 100.0, 10.0);
        let st = (steps as usize).clamp(50, 8000);
        self.oq.out = run_false_vacuum_lockdown(lam, vv, ee, hb, gd, monitor_kind, tm, st);
        self.oq.out.as_ptr()
    }

    /// Longitud del último vector devuelto por `false_vacuum_lockdown` o `wigner_open_oscillator`.
    pub fn open_quantum_out_len(&self) -> u32 {
        self.oq.out.len() as u32
    }

    /// Fase 14 (Ireland y Vennin, JCAP 2026; decoherencia térmica): oscilador abierto en base de
    /// Fock con compresión ξ, no gaussianidad g₃ q³ + g₄ q⁴, desfase γ_φ y baño térmico (γ_th, n_th).
    /// `init_kind`: 0 = vacío, 1 = coherente |α₀⟩, 2 = gato (|α₀⟩ + |−α₀⟩)/√N.
    /// Devuelve puntero a [w_neg, w_min, purity, var_q, var_p, w_norm, trace_err, edge_prob, mean_n]
    /// y guarda la rejilla 2D de W(q, p) en `wigner_grid_ptr`.
    #[allow(clippy::too_many_arguments)]
    pub fn wigner_open_oscillator(
        &mut self,
        omega: f32,
        xi: f32,
        g3: f32,
        g4: f32,
        gamma_phi: f32,
        gamma_th: f32,
        n_th: f32,
        init_kind: u32,
        alpha0: f32,
        t_max: f32,
        steps: u32,
        grid_n: u32,
        lim: f32,
    ) -> *const f64 {
        let gn = (grid_n as usize).clamp(16, 96);
        let lm = clamp_finite(lim as f64, 2.0, 8.0, 4.0);
        let (summary, grid) = run_wigner_open_oscillator(
            clamp_finite(omega as f64, 0.0, 20.0, 1.0),
            clamp_finite(xi as f64, -5.0, 5.0, 0.0),
            clamp_finite(g3 as f64, -2.0, 2.0, 0.0),
            clamp_finite(g4 as f64, -2.0, 2.0, 0.0),
            clamp_finite(gamma_phi as f64, 0.0, 20.0, 0.0),
            clamp_finite(gamma_th as f64, 0.0, 20.0, 0.0),
            clamp_finite(n_th as f64, 0.0, 20.0, 0.0),
            init_kind,
            clamp_finite(alpha0 as f64, -4.0, 4.0, 0.0),
            clamp_finite(t_max as f64, 0.0, 40.0, 1.0),
            (steps as usize).clamp(1, 4000),
            gn,
            lm,
        );
        self.oq.out = summary;
        self.oq.wigner = grid;
        self.oq.wigner_n = gn;
        self.oq.wigner_lim = lm;
        self.oq.out.as_ptr()
    }

    /// Puntero a la última rejilla 2D de W(q, p) (grid_n × grid_n, filas en p y columnas en q).
    pub fn wigner_grid_ptr(&self) -> *const f64 {
        self.oq.wigner.as_ptr()
    }

    /// Lado N de la rejilla N × N de W(q, p).
    pub fn wigner_grid_size(&self) -> u32 {
        self.oq.wigner_n as u32
    }
}
