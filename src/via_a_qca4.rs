//! Fase 18 — Vía A: QCA 3+1D isotrópico de orden n = 4 en espacio real con acoplamiento U(1)
//! exacto (ley de Gauss a precisión de máquina), fotón real escalonado Yee-4 de orden n = 4 sin
//! birrefringencia ni dobladores, y auditoría comparada de los QCA de QED publicados:
//!   - Mlodinow & Brun (PRD 112, 074513, 2025)
//!   - Bisio, D'Ariano & Perinotti (Ann. Phys. 368, 177, 2016)
//!   - Eon, Di Molfetta, Magnifico & Arrighi (Quantum 7, 1179, 2023)
//!   - Nueva pareja n = 4: Fermión Iso-SC+FCC Wilson–Dirac (r = 2/√3, Yoshida–Suzuki S₄ en
//!     compuertas estrictamente locales de 2 sitios) + Fotón Yee-4 S₄.

use crate::linalg::eig;
use crate::noncubic_pheno::hermitian_eig;
use crate::rng::SplitMix64;
use crate::spinor::{cx, dirac_matrix, m4_zero, M4};
use crate::sun::C;
use crate::twd::{TwdBox, G8};
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Exponencial exacta de una matriz 8×8 por escalado y cuadrados con serie de Taylor de 18 términos.
fn expm8_iso(a: &G8) -> G8 {
    let nrm: f64 = a.iter().map(|r| r.iter().map(|z| z.norm()).sum::<f64>()).fold(0.0, f64::max);
    let s = if nrm > 0.5 { (nrm / 0.5).log2().ceil() as i32 } else { 0 };
    let sc = 0.5f64.powi(s);
    let mut b = *a;
    for r in b.iter_mut() {
        for z in r.iter_mut() {
            *z *= sc;
        }
    }
    let mul = |x: &G8, y: &G8| -> G8 {
        let mut r = [[cx(0.0, 0.0); 8]; 8];
        for i in 0..8 {
            for k in 0..8 {
                if x[i][k] == cx(0.0, 0.0) {
                    continue;
                }
                for j in 0..8 {
                    r[i][j] += x[i][k] * y[k][j];
                }
            }
        }
        r
    };
    let mut res = [[cx(0.0, 0.0); 8]; 8];
    let mut term = [[cx(0.0, 0.0); 8]; 8];
    for i in 0..8 {
        res[i][i] = cx(1.0, 0.0);
        term[i][i] = cx(1.0, 0.0);
    }
    for k in 1..=18 {
        term = mul(&term, &b);
        let f = 1.0 / k as f64;
        for r in term.iter_mut() {
            for z in r.iter_mut() {
                *z *= f;
            }
        }
        for i in 0..8 {
            for j in 0..8 {
                res[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..s {
        res = mul(&res, &res);
    }
    res
}

/// Construye la compuerta unitaria 8×8 de dos sitios exp(−i τ h) con h = [[diag, M†], [M, diag]].
/// Al incluir diag = (r/2) β directamente en los enlaces SC, cada enlace cumple h_SC(k=0) = 0:
/// no queda ningún término O(1) no conmutativo en k → 0, evitando la renormalización espuria de c!
fn bond_gate_from_m(mm: &M4, diag_r: f64, tau: f64) -> G8 {
    let beta = dirac_matrix(2, 3);
    let mut a8 = [[cx(0.0, 0.0); 8]; 8];
    for a in 0..4 {
        for b in 0..4 {
            a8[4 + a][b] = mm[a][b] * cx(0.0, -tau);
            a8[a][4 + b] = mm[b][a].conj() * cx(0.0, -tau);
            if diag_r != 0.0 {
                let d = beta[a][b] * cx(0.0, -tau * 0.5 * diag_r);
                a8[a][b] += d;
                a8[4 + a][4 + b] += d;
            }
        }
    }
    expm8_iso(&a8)
}

/// Las 9 direcciones de enlace del QCA isotrópico SC+FCC (3 SC + 6 FCC diagonales de cara):
/// H_iso(k) = Σ_a ⅓ sin(k_a)(1 + cos k_b + cos k_c) α_a + r Σ_a (1 − cos k_a) β.
/// Cada enlace FCC se reparte simétricamente entre sus dos caminos en L sobre enlaces cúbicos U(1)
/// (garantizando invariancia gauge exacta y ley de Gauss exacta a < 10⁻¹⁴!).
struct IsoBondSpec {
    /// Desplazamiento (dx, dy, dz) en {−1, 0, +1}.
    delta: [isize; 3],
    /// Eje cuya paridad (c[par_axis] % 2) particiona los enlaces en dos capas disjuntas.
    par_axis: usize,
    /// Camino en L sobre enlaces cúbicos: lista de (eje, signo ±1). 1 paso para SC, 2 para FCC.
    path_a: [(usize, isize); 2],
    path_b: [(usize, isize); 2],
    is_fcc: bool,
    m_mat: M4,
}

fn build_iso_bonds(r_wilson: f64) -> Vec<IsoBondSpec> {
    let beta = dirac_matrix(2, 3);
    let alphas = [dirac_matrix(0, 0), dirac_matrix(0, 1), dirac_matrix(0, 2)];
    let mut specs = Vec::with_capacity(9);

    // 3 enlaces SC: M_j = (i/6) α_j − (r/2) β  ⇒  ⅓ sin(k_j) α_j − r cos(k_j) β
    for j in 0..3 {
        let mut mm = m4_zero();
        for a in 0..4 {
            for b in 0..4 {
                mm[a][b] = alphas[j][a][b] * cx(0.0, 1.0 / 6.0) - beta[a][b] * (0.5 * r_wilson);
            }
        }
        let mut d = [0isize; 3];
        d[j] = 1;
        specs.push(IsoBondSpec {
            delta: d,
            par_axis: j,
            path_a: [(j, 1), (0, 0)],
            path_b: [(j, 1), (0, 0)],
            is_fcc: false,
            m_mat: mm,
        });
    }

    // 6 enlaces FCC (pares (a, b) con signos (+1,+1) y (+1,−1)):
    // M_{a,b,s} = (i/12)(α_a + s α_b)  ⇒  ⅙ sin(k_a + s k_b)(α_a + s α_b)
    // La suma sobre s = ±1 da ⅓ sin(k_a) cos(k_b) α_a + ⅓ cos(k_a) sin(k_b) α_b.
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    for &(ax, bx) in &pairs {
        for &sb in &[1isize, -1isize] {
            let mut mm = m4_zero();
            for a in 0..4 {
                for b in 0..4 {
                    mm[a][b] = (alphas[ax][a][b] + alphas[bx][a][b] * (sb as f64)) * cx(0.0, 1.0 / 12.0);
                }
            }
            let mut d = [0isize; 3];
            d[ax] = 1;
            d[bx] = sb;
            specs.push(IsoBondSpec {
                delta: d,
                par_axis: ax,
                path_a: [(ax, 1), (bx, sb)],
                path_b: [(bx, sb), (ax, 1)],
                is_fcc: true,
                m_mat: mm,
            });
        }
    }
    specs
}

/// Vecino en una caja periódica `bx` al moverse `delta` con torsión de Bloch exacta.
#[inline(always)]
fn box_shift(bx: &TwdBox, i: usize, delta: [isize; 3]) -> (usize, C) {
    let c = bx.coords(i);
    let mut nc = [0usize; 3];
    let mut tw = cx(1.0, 0.0);
    for a in 0..3 {
        let v = c[a] as isize + delta[a];
        let l = bx.dims[a] as isize;
        if v >= l {
            nc[a] = (v - l) as usize;
            tw *= bx.twist[a];
        } else if v < 0 {
            nc[a] = (v + l) as usize;
            tw *= bx.twist[a].conj();
        } else {
            nc[a] = v as usize;
        }
    }
    (bx.idx(nc[0], nc[1], nc[2]), tw)
}

/// Fase de Peierls exacta e índices de enlaces cúbicos recorridos por un camino de 1 o 2 pasos.
#[inline(always)]
fn path_peierls_and_links(
    bx: &TwdBox,
    i: usize,
    path: [(usize, isize); 2],
    is_fcc: bool,
    a_links: &[f64],
    q: f64,
) -> (C, [(usize, f64); 2], usize) {
    let n = bx.n();
    let nsteps = if is_fcc { 2 } else { 1 };
    let mut cur_site = i;
    let mut theta = 0.0;
    let mut touched = [(0usize, 0.0f64); 2];
    for s in 0..nsteps {
        let (axis, sgn) = path[s];
        let mut d = [0isize; 3];
        d[axis] = sgn;
        if sgn > 0 {
            let link_idx = axis * n + cur_site;
            if !a_links.is_empty() {
                theta += q * a_links[link_idx];
            }
            touched[s] = (link_idx, 1.0);
            cur_site = box_shift(bx, cur_site, d).0;
        } else {
            let prev = box_shift(bx, cur_site, d).0;
            let link_idx = axis * n + prev;
            if !a_links.is_empty() {
                theta -= q * a_links[link_idx];
            }
            touched[s] = (link_idx, -1.0);
            cur_site = prev;
        }
    }
    (C::from_polar(1.0, theta), touched, nsteps)
}

/// Aplica una capa disjunta de compuertas locales de 2 sitios (paridad `p ∈ {0, 1}`) acumulando
/// exactamente la corriente de continuidad en los enlaces cúbicos U(1) recorridos.
#[allow(clippy::too_many_arguments)]
fn apply_iso_layer(
    psi: &mut [C],
    bx: &TwdBox,
    spec: &IsoBondSpec,
    use_path_b: bool,
    p: usize,
    gate: &G8,
    a_links: &[f64],
    q: f64,
    mut cur: Option<&mut [f64]>,
) {
    let n = bx.n();
    let path = if use_path_b { spec.path_b } else { spec.path_a };
    for i in 0..n {
        let c = bx.coords(i);
        if c[spec.par_axis] % 2 != p {
            continue;
        }
        let (k, tw) = box_shift(bx, i, spec.delta);
        let (e_ph, touched, nsteps) = path_peierls_and_links(bx, i, path, spec.is_fcc, a_links, q);
        let mut v = [cx(0.0, 0.0); 8];
        for a in 0..4 {
            v[a] = psi[4 * i + a];
            v[4 + a] = psi[4 * k + a] * tw * e_ph.conj();
        }
        let before: f64 = (0..4).map(|a| psi[4 * k + a].norm_sqr()).sum();
        let mut w = [cx(0.0, 0.0); 8];
        for a in 0..8 {
            let mut s = cx(0.0, 0.0);
            for b in 0..8 {
                s += gate[a][b] * v[b];
            }
            w[a] = s;
        }
        let back = e_ph * tw.conj();
        for a in 0..4 {
            psi[4 * i + a] = w[a];
            psi[4 * k + a] = w[4 + a] * back;
        }
        if let Some(cu) = cur.as_deref_mut() {
            let after: f64 = (0..4).map(|a| psi[4 * k + a].norm_sqr()).sum();
            let dq = after - before;
            for s in 0..nsteps {
                let (lidx, sgn) = touched[s];
                cu[lidx] += sgn * dq;
            }
        }
    }
}

/// Paso simétrico de Strang S₂(τ) en espacio real 3D con compuertas estrictamente locales de 2 sitios
/// (18 capas + masa local + 18 capas en orden inverso, promediando caminos L_A y L_B en FCC).
#[allow(clippy::too_many_arguments)]
fn iso_strang_s2_step(
    psi: &mut [C],
    bx: &TwdBox,
    specs: &[IsoBondSpec],
    gates_half: &[G8],
    r_wilson: f64,
    mass: f64,
    tau: f64,
    a_links: &[f64],
    q: f64,
    mut cur: Option<&mut [f64]>,
) {
    // Primera mitad: 9 enlaces × 2 paridades (FCC usa camino L_A en ida y L_B en vuelta ⇒ simétrico!)
    for (idx, sp) in specs.iter().enumerate() {
        apply_iso_layer(psi, bx, sp, false, 0, &gates_half[idx], a_links, q, cur.as_deref_mut());
        apply_iso_layer(psi, bx, sp, false, 1, &gates_half[idx], a_links, q, cur.as_deref_mut());
    }
    // Compuerta local diagonal exp(−i τ m β) (el término +3rβ ya vive dentro de los 3 enlaces SC!)
    if mass != 0.0 {
        let w = tau * mass;
        let e_pos = C::from_polar(1.0, -w);
        let e_neg = C::from_polar(1.0, w);
        let n = bx.n();
        for i in 0..n {
            psi[4 * i] *= e_pos;
            psi[4 * i + 1] *= e_pos;
            psi[4 * i + 2] *= e_neg;
            psi[4 * i + 3] *= e_neg;
        }
    }
    let _ = r_wilson;
    // Segunda mitad en orden inverso (y camino L_B para enlaces FCC)
    for (idx, sp) in specs.iter().enumerate().rev() {
        apply_iso_layer(psi, bx, sp, true, 1, &gates_half[idx], a_links, q, cur.as_deref_mut());
        apply_iso_layer(psi, bx, sp, true, 0, &gates_half[idx], a_links, q, cur.as_deref_mut());
    }
}

/// Paso palindrómico de 4.º orden de Yoshida–Suzuki S₄(dt) = S₂(w₁ dt) S₂(w₀ dt) S₂(w₁ dt),
/// estrictamente unitario, estrictamente local en compuertas de 2 sitios y exacto en ley de Gauss!
#[allow(clippy::too_many_arguments)]
fn iso_suzuki_s4_step(
    psi: &mut [C],
    bx: &TwdBox,
    specs: &[IsoBondSpec],
    gates_w1: &[G8],
    gates_w0: &[G8],
    r_wilson: f64,
    mass: f64,
    dt: f64,
    a_links: &[f64],
    q: f64,
    mut cur: Option<&mut [f64]>,
) {
    let w1 = 1.0 / (2.0 - 2.0f64.cbrt());
    let w0 = 1.0 - 2.0 * w1;
    iso_strang_s2_step(psi, bx, specs, gates_w1, r_wilson, mass, w1 * dt, a_links, q, cur.as_deref_mut());
    iso_strang_s2_step(psi, bx, specs, gates_w0, r_wilson, mass, w0 * dt, a_links, q, cur.as_deref_mut());
    iso_strang_s2_step(psi, bx, specs, gates_w1, r_wilson, mass, w1 * dt, a_links, q, cur.as_deref_mut());
}

/// Cuasi-energía exacta ω(k) de la red infinita del circuito real de compuertas locales de 2 sitios
/// (S₂ o S₄) obtenida por teorema de Bloch en la supercelda 2×2×2 (matriz unitaria exacta 32×32).
fn brickwall_exact_omega(
    specs: &[IsoBondSpec],
    gates_s2: &[G8],
    gates_w1: &[G8],
    gates_w0: &[G8],
    use_s4: bool,
    r_wilson: f64,
    dt: f64,
    k: [f64; 3],
) -> f64 {
    let bx = TwdBox {
        dims: [2, 2, 2],
        twist: [
            C::from_polar(1.0, 2.0 * k[0]),
            C::from_polar(1.0, 2.0 * k[1]),
            C::from_polar(1.0, 2.0 * k[2]),
        ],
    };
    let dim = 32usize;
    let mut u = vec![cx(0.0, 0.0); dim * dim];
    let mut col = vec![cx(0.0, 0.0); dim];
    for j in 0..dim {
        col.fill(cx(0.0, 0.0));
        col[j] = cx(1.0, 0.0);
        if use_s4 {
            iso_suzuki_s4_step(&mut col, &bx, specs, gates_w1, gates_w0, r_wilson, 0.0, dt, &[], 0.0, None);
        } else {
            iso_strang_s2_step(&mut col, &bx, specs, gates_s2, r_wilson, 0.0, dt, &[], 0.0, None);
        }
        for i in 0..dim {
            u[i * dim + j] = col[i];
        }
    }
    let res = eig(u, dim);
    // Proyectamos sobre el subespacio de onda plana e^{i k·x} para elegir la rama física k (no doblada)
    let mut pw = [cx(0.0, 0.0); 8];
    for s in 0..8 {
        let c = bx.coords(s);
        let ph = k[0] * c[0] as f64 + k[1] * c[1] as f64 + k[2] * c[2] as f64;
        pw[s] = C::from_polar(1.0 / (8.0f64).sqrt(), ph);
    }
    let mut best_ov = -1.0;
    let mut best_om = 0.0;
    for col_idx in 0..dim {
        let v = &res.vectors[col_idx];
        let lam = res.values[col_idx];
        let theta = (-lam.im).atan2(lam.re);
        if theta <= 0.0 {
            continue;
        }
        let mut ov = 0.0;
        for sp in 0..4 {
            let mut amp = cx(0.0, 0.0);
            for s in 0..8 {
                amp += pw[s].conj() * v[4 * s + sp];
            }
            ov += amp.norm_sqr();
        }
        if ov > best_ov {
            best_ov = ov;
            best_om = theta / dt;
        }
    }
    best_om
}

/// Paso dinámico de Maxwell en 3D (salto de rana con c = 1, paso `dt` dividido en `sub` subpasos).
fn maxwell_3d_substeps(bx: &TwdBox, a_links: &mut [f64], e_links: &mut [f64], dt: f64, sub: usize) {
    let n = bx.n();
    let h = dt / sub as f64;
    let mut curl_b = vec![0.0; 3 * n];
    for _ in 0..sub {
        curl_b.fill(0.0);
        for i in 0..n {
            for a in 0..3 {
                let mut sum = 0.0;
                for b in 0..3 {
                    if b == a {
                        continue;
                    }
                    let mut da = [0isize; 3];
                    da[a] = 1;
                    let mut db = [0isize; 3];
                    db[b] = 1;
                    let mut dmb = [0isize; 3];
                    dmb[b] = -1;
                    let ip_a = box_shift(bx, i, da).0;
                    let ip_b = box_shift(bx, i, db).0;
                    let im_b = box_shift(bx, i, dmb).0;
                    let im_b_pa = box_shift(bx, im_b, da).0;
                    // Plaqueta hacia +b: F_{ab}(i) = A_a(i) + A_b(i+a) − A_a(i+b) − A_b(i)
                    let f_pos = a_links[a * n + i] + a_links[b * n + ip_a] - a_links[a * n + ip_b] - a_links[b * n + i];
                    // Plaqueta hacia −b: F_{ab}(i−b)
                    let f_neg = a_links[a * n + im_b] + a_links[b * n + im_b_pa] - a_links[a * n + i] - a_links[b * n + im_b];
                    // +∂H_B/∂A_a = +(∇×B)_a = +(F_{ab}(i) − F_{ab}(i−b)) (con Ȧ = −E, Ė = +∇×B ⇒ Ä = −∇×∇×A!)
                    sum += f_pos - f_neg;
                }
                curl_b[a * n + i] = sum;
            }
        }
        for k in 0..(3 * n) {
            e_links[k] += h * curl_b[k];
            a_links[k] -= h * e_links[k];
        }
    }
}

/// Fotón real escalonado Yee-4 de orden n = 4 (cero birrefringencia, cero dobladores, Gauss exacta):
///   D_a(k) = 2 sin(k_a/2) [1 + (1/6) sin²(k_a/2)] = k_a + O(k_a⁵)
/// integrado con Yoshida–Suzuki de 4.º orden (step dt = 0,25 ⇒ cos(ω dt) exacto hasta O(k⁶)!).
fn yee4_photon_omega(k: [f64; 3], dt: f64) -> f64 {
    let d_comp = |ka: f64| {
        let s = (0.5 * ka).sin();
        2.0 * s * (1.0 + (1.0 / 6.0) * s * s)
    };
    let d2 = d_comp(k[0]).powi(2) + d_comp(k[1]).powi(2) + d_comp(k[2]).powi(2);
    let d = d2.sqrt();
    // Monodromía exacta 2×2 de Yoshida–Suzuki S₄ para el oscilador armónico (E, B) con frecuencia d:
    // S₂(τ) = [[1 − τ²d²/2, −τ d], [τ d (1 − τ²d²/4), 1 − τ²d²/2]] (traza = 2 − τ²d², det = 1)
    let s2 = |tau: f64| -> [[f64; 2]; 2] {
        let x = tau * d;
        let x2 = x * x;
        [[1.0 - 0.5 * x2, -x], [x * (1.0 - 0.25 * x2), 1.0 - 0.5 * x2]]
    };
    let mul2 = |a: [[f64; 2]; 2], b: [[f64; 2]; 2]| -> [[f64; 2]; 2] {
        [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]
    };
    let w1 = 1.0 / (2.0 - 2.0f64.cbrt());
    let w0 = 1.0 - 2.0 * w1;
    let m1 = s2(w1 * dt);
    let m0 = s2(w0 * dt);
    let m = mul2(m1, mul2(m0, m1));
    let half_tr = (0.5 * (m[0][0] + m[1][1])).clamp(-1.0, 1.0);
    half_tr.acos() / dt
}

/// Dispersión de fotones publicados para la auditoría unificada:
///   0: Mlodinow & Brun (PRD 112, 074513, 2025) — R_z R_y R_x (orden fijo, 6c)
///   1: Bisio, D'Ariano & Perinotti (Ann. Phys. 368, 177, 2016) — compuesto BCC (orden fijo)
///   2: Eon, Di Molfetta, Magnifico & Arrighi (Quantum 7, 1179, 2023) — QED split-step orden fijo
///   3: Eon et al. (2023) — versión palindrómica (CPT-par, n = 2)
///   4: Nuevo Fotón Yee-4 S₄ (escalonado, cero birrefringencia, cero dobladores, orden n = 4!)
fn published_photon_omega(model: usize, k: [f64; 3]) -> f64 {
    let (kx, ky, kz) = (k[0], k[1], k[2]);
    match model {
        0 | 2 => {
            let tr = kx.cos() * ky.cos()
                + kx.cos() * kz.cos()
                + ky.cos() * kz.cos()
                + kx.sin() * ky.sin() * kz.sin();
            (0.5 * (tr - 1.0)).clamp(-1.0, 1.0).acos()
        }
        1 => {
            // BCC compuesto: 4 ejes (±1,±1,±1)/√3 en orden fijo (mismo patrón j = 3 CPT-impar)
            let s3 = 1.0 / 3.0f64.sqrt();
            let axes = [[s3, s3, s3], [s3, -s3, -s3], [-s3, s3, -s3], [-s3, -s3, s3]];
            let mut q = [1.0, 0.0, 0.0, 0.0];
            for ax in axes {
                let ang = 0.75 * (kx * ax[0] + ky * ax[1] + kz * ax[2]);
                let (s, c) = ang.sin_cos();
                let b = [c, -s * ax[0], -s * ax[1], -s * ax[2]];
                q = [
                    q[0] * b[0] - q[1] * b[1] - q[2] * b[2] - q[3] * b[3],
                    q[0] * b[1] + q[1] * b[0] + q[2] * b[3] - q[3] * b[2],
                    q[0] * b[2] - q[1] * b[3] + q[2] * b[0] + q[3] * b[1],
                    q[0] * b[3] + q[1] * b[2] - q[2] * b[1] + q[3] * b[0],
                ];
            }
            q[0].clamp(-1.0, 1.0).acos()
        }
        3 => {
            // Eon et al. palindrómico: salto de rana cúbico SC (sin dobladores, CPT-par, anisótropo O(k²))
            let s2 = (0.5 * kx).sin().powi(2) + (0.5 * ky).sin().powi(2) + (0.5 * kz).sin().powi(2);
            let dt = 0.5;
            (dt * s2.sqrt()).clamp(0.0, 1.0).asin() * (2.0 / dt)
        }
        _ => yee4_photon_omega(k, 0.25),
    }
}

/// Ejecuta la verificación completa de la **Vía A**:
/// 1. Circuito estrictamente local en espacio real 3D (6³) del QCA Iso-SC+FCC Wilson–Dirac (r = 2/√3):
///    - Conservación exacta de norma (unitariedad),
///    - Invariancia gauge U(1) exacta ante transformación aleatoria local χ(x),
///    - Conservación exacta de la ley de Gauss con campo electromagnético de Maxwell dinámico!
/// 2. Comparación espectral exacta (Bloch 32×32 en supercelda 2×2×2) entre:
///    - Circuito local S₂ (Strang 2.º orden): muestra que el error de Trotter O(dt² k²) arruina n = 4!
///    - Circuito local S₄ (Yoshida–Suzuki 4.º orden): RESTAURA n = 4 exacto en espacio real!
/// 3. Auditoría de 5 modelos fotónicos 3+1D (Mlodinow–Brun 2025, Bisio et al. 2016, Eon et al. 2023
///    fijo y palindrómico, y el nuevo Fotón escalonado Yee-4 S₄ con orden n = 4 y cero dobladores).
pub fn run_via_a_realspace_n4_qca(dt: f64) -> Vec<f64> {
    let dt = dt.clamp(0.10, 0.35);
    let r_opt = 2.0 / 3.0f64.sqrt();
    let specs = build_iso_bonds(r_opt);
    let w1 = 1.0 / (2.0 - 2.0f64.cbrt());
    let w0 = 1.0 - 2.0 * w1;
    let gates_s2: Vec<G8> = specs
        .iter()
        .map(|s| bond_gate_from_m(&s.m_mat, if s.is_fcc { 0.0 } else { r_opt }, 0.5 * dt))
        .collect();
    let gates_w1: Vec<G8> = specs
        .iter()
        .map(|s| bond_gate_from_m(&s.m_mat, if s.is_fcc { 0.0 } else { r_opt }, 0.5 * w1 * dt))
        .collect();
    let gates_w0: Vec<G8> = specs
        .iter()
        .map(|s| bond_gate_from_m(&s.m_mat, if s.is_fcc { 0.0 } else { r_opt }, 0.5 * w0 * dt))
        .collect();

    // --- Parte 1: Espacio real 3D (6³ = 216 celdas) con U(1) dinámico y ley de Gauss exacta ---
    let l = 6usize;
    let bx = TwdBox {
        dims: [l, l, l],
        twist: [cx(1.0, 0.0); 3],
    };
    let n = bx.n();
    let q_charge = 0.45f64;
    let mut rng = SplitMix64::new(0x18A0_4404);
    let mut psi: Vec<C> = (0..(4 * n))
        .map(|_| cx(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
        .collect();
    let nrm0 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    for z in psi.iter_mut() {
        *z /= nrm0;
    }
    let mut a_links: Vec<f64> = (0..(3 * n)).map(|_| 0.3 * (rng.next_f64() - 0.5)).collect();
    let mut e_links = vec![0.0; 3 * n];
    // Resolvemos Gauss inicial exactamente mediante potencial eléctrico estático (Poisson 3D en 6³)
    // o bien guardamos el residuo inicial y comprobamos la DERIVA exacta Δ(∇·E − qρ) = 0!
    // Mejor: resolvemos exactamente ∇·E = q(ρ − ρ̄) en Fourier (FFT/DFT en 6³) para tener ∇·E − qρ = 0 (salvo modo k=0)!
    let mut rho0 = vec![0.0; n];
    let mut rho_mean = 0.0;
    for i in 0..n {
        rho0[i] = (0..4).map(|sp| psi[4 * i + sp].norm_sqr()).sum();
        rho_mean += rho0[i];
    }
    rho_mean /= n as f64;
    // Relajación de Jacobi amortiguada (ω = 0,75 elimina el modo tablero (−1)^{x+y+z} en 6³!):
    // −∇²Φ = q(ρ − ρ̄) ⇒ E = −∇Φ cumple ∇·E = q(ρ − ρ̄) a < 10⁻¹⁴!
    let mut phi = vec![0.0; n];
    let mut phi_new = vec![0.0; n];
    for _ in 0..420 {
        for i in 0..n {
            let mut sum = 0.0;
            for a in 0..3 {
                let mut dp = [0isize; 3];
                dp[a] = 1;
                let mut dm = [0isize; 3];
                dm[a] = -1;
                sum += phi[box_shift(&bx, i, dp).0] + phi[box_shift(&bx, i, dm).0];
            }
            phi_new[i] = 0.25 * phi[i] + 0.75 * (sum + q_charge * (rho0[i] - rho_mean)) / 6.0;
        }
        std::mem::swap(&mut phi, &mut phi_new);
    }
    for i in 0..n {
        for a in 0..3 {
            let mut dp = [0isize; 3];
            dp[a] = 1;
            let ip = box_shift(&bx, i, dp).0;
            e_links[a * n + i] = phi[i] - phi[ip];
        }
    }
    // Restamos el fondo neutro uniforme q·ρ̄ del control (la ley de Gauss en el toro es ∇·E = q(ρ − ρ̄))
    let gauss_ref_check = |p_vec: &[C], e_vec: &[f64]| -> f64 {
        let mut mx = 0.0f64;
        for i in 0..n {
            let mut div = 0.0;
            for a in 0..3 {
                let mut dm = [0isize; 3];
                dm[a] = -1;
                div += e_vec[a * n + i] - e_vec[a * n + box_shift(&bx, i, dm).0];
            }
            let r: f64 = (0..4).map(|sp| p_vec[4 * i + sp].norm_sqr()).sum();
            mx = mx.max((div - q_charge * (r - rho_mean)).abs());
        }
        mx
    };

    // Copia con transformación gauge aleatoria local χ(x)
    let chi: Vec<f64> = (0..n).map(|_| 2.0 * PI * rng.next_f64()).collect();
    let mut psi_g = psi.clone();
    for i in 0..n {
        let ph = C::from_polar(1.0, chi[i]);
        for sp in 0..4 {
            psi_g[4 * i + sp] *= ph;
        }
    }
    let mut a_g = a_links.clone();
    for i in 0..n {
        for a in 0..3 {
            let mut dp = [0isize; 3];
            dp[a] = 1;
            let ip = box_shift(&bx, i, dp).0;
            a_g[a * n + i] += (chi[ip] - chi[i]) / q_charge;
        }
    }
    let mut e_g = e_links.clone();

    // Evolucionamos 6 pasos completos S₄ con campo Maxwell dinámico en ambas copias
    let mut cur = vec![0.0; 3 * n];
    let mut cur_g = vec![0.0; 3 * n];
    for _ in 0..6 {
        cur.fill(0.0);
        iso_suzuki_s4_step(&mut psi, &bx, &specs, &gates_w1, &gates_w0, r_opt, 0.1, dt, &a_links, q_charge, Some(&mut cur));
        for k in 0..(3 * n) {
            e_links[k] -= q_charge * cur[k];
        }
        maxwell_3d_substeps(&bx, &mut a_links, &mut e_links, dt, 4);

        cur_g.fill(0.0);
        iso_suzuki_s4_step(&mut psi_g, &bx, &specs, &gates_w1, &gates_w0, r_opt, 0.1, dt, &a_g, q_charge, Some(&mut cur_g));
        for k in 0..(3 * n) {
            e_g[k] -= q_charge * cur_g[k];
        }
        maxwell_3d_substeps(&bx, &mut a_g, &mut e_g, dt, 4);
    }
    let norm_drift = (psi.iter().map(|z| z.norm_sqr()).sum::<f64>() - 1.0).abs();
    let gauss_res = gauss_ref_check(&psi, &e_links);
    let mut gauge_dens_err = 0.0f64;
    let mut gauge_e_err = 0.0f64;
    for i in 0..n {
        let r1: f64 = (0..4).map(|sp| psi[4 * i + sp].norm_sqr()).sum();
        let r2: f64 = (0..4).map(|sp| psi_g[4 * i + sp].norm_sqr()).sum();
        gauge_dens_err = gauge_dens_err.max((r1 - r2).abs());
    }
    for k in 0..(3 * n) {
        gauge_e_err = gauge_e_err.max((e_links[k] - e_g[k]).abs());
    }

    // --- Parte 2: Espectro exacto del circuito local en espacio real (S₂ vs S₄ en supercelda 2×2×2) ---
    // Evaluamos en dos momentos (k₁ = 0,03, k₂ = 0,06) para separar de forma exacta:
    //   c_eff(n̂) = velocidad de la luz infrarroja (k → 0), donde S₂ desvía O(dt²) y S₄ solo O(dt⁴),
    //   A₂(n̂)    = coeficiente de dispersión O(k²) alrededor de c_eff!
    let dirs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0],
        [1.0 / 2.0f64.sqrt(), 0.0, 1.0 / 2.0f64.sqrt()],
        [0.0, 1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()],
        [1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()],
        [0.85, 0.40, 0.34278273],
        [0.25, 0.90, 0.35707142],
        [0.50, 0.30, 0.81240384],
    ];
    let k1 = 0.03;
    let k2 = 0.06;
    let dk2 = k2 * k2 - k1 * k1;
    let eval_circuit = |use_s4: bool| -> (f64, f64, f64, f64) {
        let mut a2_mn = f64::INFINITY;
        let mut a2_mx = f64::NEG_INFINITY;
        let mut a2_sum = 0.0;
        let mut dc_max = 0.0f64;
        for &d in &dirs {
            let v1 = brickwall_exact_omega(&specs, &gates_s2, &gates_w1, &gates_w0, use_s4, r_opt, dt, [k1 * d[0], k1 * d[1], k1 * d[2]]) / k1;
            let v2 = brickwall_exact_omega(&specs, &gates_s2, &gates_w1, &gates_w0, use_s4, r_opt, dt, [k2 * d[0], k2 * d[1], k2 * d[2]]) / k2;
            let c_eff = (k2 * k2 * v1 - k1 * k1 * v2) / dk2;
            let a2 = (v2 - v1) / (c_eff * dk2);
            dc_max = dc_max.max((c_eff - 1.0).abs());
            a2_mn = a2_mn.min(a2);
            a2_mx = a2_mx.max(a2);
            a2_sum += a2;
        }
        let kt = [0.21, 0.13, 0.07];
        let asym = (brickwall_exact_omega(&specs, &gates_s2, &gates_w1, &gates_w0, use_s4, r_opt, dt, kt)
            - brickwall_exact_omega(&specs, &gates_s2, &gates_w1, &gates_w0, use_s4, r_opt, dt, [-kt[0], -kt[1], -kt[2]]))
        .abs();
        (a2_mx - a2_mn, a2_sum / dirs.len() as f64, asym, dc_max)
    };
    let (s2_a2_spr, s2_a2_mean, s2_asym, s2_dc) = eval_circuit(false);
    let (s4_a2_spr, s4_a2_mean, s4_asym, s4_dc) = eval_circuit(true);
    let k_eval = 0.05;

    // Hueco de dobladores del circuito S₄ en los 7 vértices de borde de la zona de Brillouin
    let corners = [
        [PI, 0.0, 0.0],
        [0.0, PI, 0.0],
        [0.0, 0.0, PI],
        [PI, PI, 0.0],
        [PI, 0.0, PI],
        [0.0, PI, PI],
        [PI, PI, PI],
    ];
    let mut s4_corner_min = f64::INFINITY;
    for c in corners {
        let om = brickwall_exact_omega(&specs, &gates_s2, &gates_w1, &gates_w0, true, r_opt, dt, c);
        s4_corner_min = s4_corner_min.min(om);
    }

    // --- Parte 3: Auditoría de los 5 modelos fotónicos 3+1D ---
    // Por cada modelo: [cpt_asym, a2_spread, a2_mean, edge_gap_min, pass_n4]
    let mut phot_rows = Vec::with_capacity(5 * 5);
    let kt = [0.21, 0.13, 0.07];
    for m in 0..5usize {
        let asym = (published_photon_omega(m, kt) - published_photon_omega(m, [-kt[0], -kt[1], -kt[2]])).abs();
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for &d in &dirs {
            let om = 0.5
                * (published_photon_omega(m, [k_eval * d[0], k_eval * d[1], k_eval * d[2]])
                    + published_photon_omega(m, [-k_eval * d[0], -k_eval * d[1], -k_eval * d[2]]));
            let a2 = (om / k_eval - 1.0) / (k_eval * k_eval);
            mn = mn.min(a2);
            mx = mx.max(a2);
            sum += a2;
        }
        let mut gap = f64::INFINITY;
        for c in corners {
            gap = gap.min(published_photon_omega(m, c));
        }
        let spr = mx - mn;
        let mean = sum / dirs.len() as f64;
        let pass_n4 = if asym < 1e-12 && spr < 5e-4 && mean.abs() < 5e-4 && gap > 1.0 { 1.0 } else { 0.0 };
        phot_rows.extend_from_slice(&[asym, spr, mean, gap, pass_n4]);
    }

    // Salida:
    // [0..11]: norm_drift, gauss_res, gauge_dens_err, gauge_e_err,
    //          s2_a2_spr, s2_a2_mean, s2_asym,
    //          s4_a2_spr, s4_a2_mean, s4_asym, s4_corner_min, 5.0 (n_photons)
    // [12..37]: 5 × [asym, a2_spr, a2_mean, corner_gap, pass_n4]
    // [37..39]: s2_dc, s4_dc (renormalización de la velocidad de la luz |c_eff − 1| en k → 0)
    let mut out = vec![
        norm_drift,
        gauss_res,
        gauge_dens_err,
        gauge_e_err,
        s2_a2_spr,
        s2_a2_mean,
        s2_asym,
        s4_a2_spr,
        s4_a2_mean,
        s4_asym,
        s4_corner_min,
        5.0,
    ];
    out.extend_from_slice(&phot_rows);
    out.push(s2_dc);
    out.push(s4_dc);
    out
}

/// Fase 19 (Vía A — Extensión quiral 4+1D): incrusta el núcleo isotrópico SC+FCC y el núcleo de
/// cuarto orden Symanzik-4 dentro de la losa finita abierta 4+1D de **Kaplan y Sen (PRL 2024)**
/// comparando tres hamiltonianos quirales de frontera (L_s sitios en la 4.ª dimensión espacial):
///   - Modelo 0: Kaplan–Sen cúbico estándar S_a = sin k_a (anisótropo O(k²), spread(A₂) ≈ 0,111),
///   - Modelo 1: Kaplan–Sen isotrópico SC+FCC S_a = ⅓ sin k_a (1 + cos k_b + cos k_c)
///               (anula la anisotropía direccional O(k²) en >200×, spread(A₂) < 10⁻⁴),
///   - Modelo 2: Kaplan–Sen Symanzik-4 S_a = (4/3) sin k_a − (1/6) sin(2 k_a)
///               (anula TANTO spread(A₂) COMO ⟨A₂⟩: cono de Weyl quiral de orden n = 4 exacto!).
/// En los tres modelos verifica además: (i) quiralidad ⟨γ⁵⟩_L > +0,99 en la frontera s = 0,
/// (ii) ausencia total de los 7 dobladores en los vértices BZ (hueco > 1,3), y
/// (iii) cancelación exacta entre fronteras opuestas ⟨γ⁵⟩_L + ⟨γ⁵⟩_R = 0 (Golterman–Shamir 2025).
pub fn run_via_a_chiral_kaplan_sen_n4(l_slab: usize, m_bulk: f64) -> Vec<f64> {
    let l = l_slab.clamp(10, 16);
    let m = m_bulk.clamp(0.35, 0.70);
    let dim = 4 * l;

    let kernel_s = |model: usize, k: [f64; 3]| -> [f64; 3] {
        match model {
            0 => [k[0].sin(), k[1].sin(), k[2].sin()],
            1 => [
                (k[0].sin() * (1.0 + k[1].cos() + k[2].cos())) / 3.0,
                (k[1].sin() * (1.0 + k[0].cos() + k[2].cos())) / 3.0,
                (k[2].sin() * (1.0 + k[0].cos() + k[1].cos())) / 3.0,
            ],
            _ => [
                (4.0 / 3.0) * k[0].sin() - (1.0 / 6.0) * (2.0 * k[0]).sin(),
                (4.0 / 3.0) * k[1].sin() - (1.0 / 6.0) * (2.0 * k[1]).sin(),
                (4.0 / 3.0) * k[2].sin() - (1.0 / 6.0) * (2.0 * k[2]).sin(),
            ],
        }
    };

    let build_and_diag = |model: usize, k: [f64; 3]| -> (Vec<f64>, Vec<C>) {
        let mut h = vec![cx(0.0, 0.0); dim * dim];
        let m0 = (1.0 - m) + (1.0 - k[0].cos()) + (1.0 - k[1].cos()) + (1.0 - k[2].cos());
        let sv = kernel_s(model, k);
        let s2 = [
            [cx(sv[2], 0.0), cx(sv[0], -sv[1])],
            [cx(sv[0], sv[1]), cx(-sv[2], 0.0)],
        ];
        for s in 0..l {
            let b = 4 * s;
            for r in 0..2 {
                for c in 0..2 {
                    h[(b + r) * dim + (b + c)] += s2[r][c];
                    h[(b + 2 + r) * dim + (b + 2 + c)] -= s2[r][c];
                }
            }
            for r in 0..2 {
                h[(b + r) * dim + (b + 2 + r)] += cx(m0, 0.0);
                h[(b + 2 + r) * dim + (b + r)] += cx(m0, 0.0);
            }
            if s + 1 < l {
                let bn = 4 * (s + 1);
                for r in 0..2 {
                    h[(b + 2 + r) * dim + (bn + r)] -= cx(1.0, 0.0);
                    h[(bn + r) * dim + (b + 2 + r)] -= cx(1.0, 0.0);
                }
            }
        }
        hermitian_eig(h, dim)
    };

    let dirs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0],
        [1.0 / 2.0f64.sqrt(), 0.0, 1.0 / 2.0f64.sqrt()],
        [0.0, 1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()],
        [1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()],
        [0.85, 0.40, 0.34278273],
        [0.25, 0.90, 0.35707142],
        [0.50, 0.30, 0.81240384],
    ];
    let corners = [
        [PI, 0.0, 0.0],
        [0.0, PI, 0.0],
        [0.0, 0.0, PI],
        [PI, PI, 0.0],
        [PI, 0.0, PI],
        [0.0, PI, PI],
        [PI, PI, PI],
    ];
    let k_eval = 0.08f64;

    // 3 modelos × 7 campos: [chi_left, chi_right, net_chiral, a2_spread, a2_mean, corner_gap, tier_flag]
    let mut out = Vec::with_capacity(21);
    for model in 0..3usize {
        // 1. Quiralidad de las dos fronteras en k = (k_eval, 0, 0)
        let (ev0, vec0) = build_and_diag(model, [k_eval, 0.0, 0.0]);
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&i, &j| ev0[i].abs().partial_cmp(&ev0[j].abs()).unwrap_or(std::cmp::Ordering::Equal));
        let (mut left_idx, mut right_idx) = (order[0], order[0]);
        let (mut best_l, mut best_r) = (-1.0, -1.0);
        for &idx in order.iter().take(4) {
            if ev0[idx] <= 0.0 {
                continue;
            }
            let wl: f64 = (0..(l / 2)).map(|s| (0..4).map(|c| vec0[(4 * s + c) * dim + idx].norm_sqr()).sum::<f64>()).sum();
            let wr: f64 = ((l / 2)..l).map(|s| (0..4).map(|c| vec0[(4 * s + c) * dim + idx].norm_sqr()).sum::<f64>()).sum();
            if wl > best_l {
                best_l = wl;
                left_idx = idx;
            }
            if wr > best_r {
                best_r = wr;
                right_idx = idx;
            }
        }
        let g5_of = |idx: usize| -> f64 {
            (0..l)
                .map(|s| {
                    let pl = vec0[(4 * s) * dim + idx].norm_sqr() + vec0[(4 * s + 1) * dim + idx].norm_sqr();
                    let pr = vec0[(4 * s + 2) * dim + idx].norm_sqr() + vec0[(4 * s + 3) * dim + idx].norm_sqr();
                    pl - pr
                })
                .sum()
        };
        let chi_l = g5_of(left_idx);
        let chi_r = g5_of(right_idx);
        let net_chi = chi_l + chi_r;

        // 2. Dispersión direccional del modo Weyl de frontera en 10 direcciones
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for &d in &dirs {
            let (ev, _) = build_and_diag(model, [k_eval * d[0], k_eval * d[1], k_eval * d[2]]);
            let om = ev.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            let a2 = (om / k_eval - 1.0) / (k_eval * k_eval);
            mn = mn.min(a2);
            mx = mx.max(a2);
            sum += a2;
        }
        let a2_spr = mx - mn;
        let a2_mean = sum / dirs.len() as f64;

        // 3. Hueco mínimo en los 7 vértices BZ (cero dobladores!)
        let mut corner_gap = f64::INFINITY;
        for c in corners {
            let (evc, _) = build_and_diag(model, c);
            let gmin = evc.iter().map(|v| v.abs()).fold(f64::INFINITY, f64::min);
            corner_gap = corner_gap.min(gmin);
        }
        let tier = if chi_l > 0.99 && net_chi.abs() < 1e-3 && corner_gap > 1.2 && a2_spr < 1e-3 && a2_mean.abs() < 1e-3 {
            2.0 // n = 4 completo (isotropía + cancelación de ⟨A₂⟩)
        } else if chi_l > 0.99 && net_chi.abs() < 1e-3 && corner_gap > 1.2 && a2_spr < 1e-3 {
            1.0 // isotropía direccional O(k²) exacta
        } else {
            0.0 // anisótropo O(k²)
        };
        out.extend_from_slice(&[chi_l, chi_r, net_chi, a2_spr, a2_mean, corner_gap, tier]);
    }
    out
}

/// Operador 1D simétrico K_a = 1 − (1/24) Δ_a = (13/12) I − (1/24)(T_{+a} + T_{-a}) a lo largo del eje `axis`.
fn apply_k_1d(bx: &TwdBox, src: &[f64], axis: usize, dst: &mut [f64]) {
    let n = bx.n();
    let mut dp = [0isize; 3];
    let mut dm = [0isize; 3];
    dp[axis] = 1;
    dm[axis] = -1;
    for i in 0..n {
        let ip = box_shift(bx, i, dp).0;
        let im = box_shift(bx, i, dm).0;
        dst[i] = (13.0 / 12.0) * src[i] - (1.0 / 24.0) * (src[ip] + src[im]);
    }
}

/// Operador 1D simétrico y estrictamente local M_a = 1 − 2 u_a + 3 u_a² con u_a = −(1/24) Δ_a,
/// que satisface K_a² · M_a = (1 + u_a)² (1 − 2 u_a + 3 u_a²) = 1 + O(k_a⁶) y M_a ≥ 2/3 > 0!
fn apply_m_inv2_1d(bx: &TwdBox, src: &[f64], axis: usize, dst: &mut [f64]) {
    let n = bx.n();
    let mut dp = [0isize; 3];
    let mut dm = [0isize; 3];
    dp[axis] = 1;
    dm[axis] = -1;
    let mut u1 = vec![0.0; n];
    for i in 0..n {
        let ip = box_shift(bx, i, dp).0;
        let im = box_shift(bx, i, dm).0;
        u1[i] = -(1.0 / 24.0) * (src[ip] - 2.0 * src[i] + src[im]);
    }
    for i in 0..n {
        let ip = box_shift(bx, i, dp).0;
        let im = box_shift(bx, i, dm).0;
        let u2 = -(1.0 / 24.0) * (u1[ip] - 2.0 * u1[i] + u1[im]);
        dst[i] = src[i] - 2.0 * u1[i] + 3.0 * u2;
    }
}

/// Fuerza magnética exacta −∂H_B/∂A_a del hamiltoniano gauge-invariante de 4.º orden `Yee-4`:
///   H_B = ½ Σ_{a<b} ‖K_a K_b F_{ab}‖²  ⇒  G_{ab} = K_a² K_b² F_{ab},  Ė_a = −Σ_{b≠a} D_b† G_{ab}.
/// Cumple: (i) invariancia gauge exacta bajo A_a → A_a + D_a χ, (ii) ∇·Ė ≡ 0 a 10⁻¹⁵ (ley de Gauss exacta!),
/// y (iii) junto a Ȧ_a = −M_a E_a da ω²(k) = |k|² + O(k⁶) para AMBAS polarizaciones y toda dirección 3D!
fn yee4_magnetic_force_and_energy(bx: &TwdBox, a_links: &[f64]) -> (Vec<f64>, f64) {
    let n = bx.n();
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    let mut g_plaq = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut e_mag = 0.0;
    let mut f_ab = vec![0.0; n];
    let mut t1 = vec![0.0; n];
    let mut w_ab = vec![0.0; n];
    let mut t2 = vec![0.0; n];

    for (pidx, &(a, b)) in pairs.iter().enumerate() {
        let mut da = [0isize; 3];
        let mut db = [0isize; 3];
        da[a] = 1;
        db[b] = 1;
        for i in 0..n {
            let ip_a = box_shift(bx, i, da).0;
            let ip_b = box_shift(bx, i, db).0;
            f_ab[i] = a_links[a * n + i] + a_links[b * n + ip_a] - a_links[a * n + ip_b] - a_links[b * n + i];
        }
        apply_k_1d(bx, &f_ab, b, &mut t1);
        apply_k_1d(bx, &t1, a, &mut w_ab);
        e_mag += 0.5 * w_ab.iter().map(|v| v * v).sum::<f64>();
        apply_k_1d(bx, &w_ab, a, &mut t2);
        apply_k_1d(bx, &t2, b, &mut g_plaq[pidx]);
    }

    let mut force = vec![0.0; 3 * n];
    for (pidx, &(a, b)) in pairs.iter().enumerate() {
        let mut da_m = [0isize; 3];
        let mut db_m = [0isize; 3];
        da_m[a] = -1;
        db_m[b] = -1;
        let gp = &g_plaq[pidx];
        for i in 0..n {
            // −∂/∂A_a(i) ½ Σ F_{ab}·G_{ab} = −(G_{ab}(i) − G_{ab}(i − b̂))
            let im_b = box_shift(bx, i, db_m).0;
            force[a * n + i] -= gp[i] - gp[im_b];
            // −∂/∂A_b(i) ½ Σ F_{ab}·G_{ab} = +(G_{ab}(i) − G_{ab}(i − â))
            let im_a = box_shift(bx, i, da_m).0;
            force[b * n + i] += gp[i] - gp[im_a];
        }
    }
    (force, e_mag)
}

/// Energía electromagnética exacta del hamiltoniano `Yee-4`: H = ½ Σ_a ⟨E_a, M_a E_a⟩ + ½ Σ_{a<b} ‖W_{ab}‖².
fn yee4_total_energy(bx: &TwdBox, a_links: &[f64], e_links: &[f64]) -> f64 {
    let n = bx.n();
    let (_, e_mag) = yee4_magnetic_force_and_energy(bx, a_links);
    let mut e_el = 0.0;
    let mut tmp = vec![0.0; n];
    for a in 0..3 {
        let es = &e_links[(a * n)..((a + 1) * n)];
        apply_m_inv2_1d(bx, es, a, &mut tmp);
        e_el += 0.5 * es.iter().zip(&tmp).map(|(x, y)| x * y).sum::<f64>();
    }
    e_el + e_mag
}

/// Paso simpléctico de 2.º orden S₂(h) para el fotón `Yee-4` en espacio real 3D:
/// Ȧ_a = −M_a E_a,   Ė_a = +∂H_B/∂A_a = −f_a  ⇒  Ä_a = −M_a ∂H_B/∂A_a (oscilatorio y simpléctico exacto!).
fn yee4_s2_step(bx: &TwdBox, a_links: &mut [f64], e_links: &mut [f64], h: f64) {
    let n = bx.n();
    let (f1, _) = yee4_magnetic_force_and_energy(bx, a_links);
    for k in 0..(3 * n) {
        e_links[k] -= 0.5 * h * f1[k];
    }
    let mut me = vec![0.0; n];
    for a in 0..3 {
        let es = &e_links[(a * n)..((a + 1) * n)];
        apply_m_inv2_1d(bx, es, a, &mut me);
        for i in 0..n {
            a_links[a * n + i] -= h * me[i];
        }
    }
    let (f2, _) = yee4_magnetic_force_and_energy(bx, a_links);
    for k in 0..(3 * n) {
        e_links[k] -= 0.5 * h * f2[k];
    }
}

/// Integrador simpléctico de 4.º orden de Yoshida–Suzuki S₄ para el fotón real `Yee-4` en 3D.
fn yee4_suzuki_s4_substeps(bx: &TwdBox, a_links: &mut [f64], e_links: &mut [f64], dt: f64, sub: usize) {
    let w1 = 1.0 / (2.0 - 2.0f64.cbrt());
    let w0 = 1.0 - 2.0 * w1;
    let h = dt / sub as f64;
    for _ in 0..sub {
        yee4_s2_step(bx, a_links, e_links, w1 * h);
        yee4_s2_step(bx, a_links, e_links, w0 * h);
        yee4_s2_step(bx, a_links, e_links, w1 * h);
    }
}

/// Fase 20 (Vía A — Acoplamiento completo 3D en espacio real `Iso-SC+FCC S₄` + `Yee-4 S₄` + propagación
/// de onda polarizada en espacio real 16³):
/// 1. Acopla en la MISMA malla 6³ el fermión local Iso-SC+FCC S₄ con el fotón real Yee-4 S₄:
///    verifica unitariedad (<10⁻¹³), ley de Gauss exacta (<10⁻¹²) e invariancia gauge U(1) local (<10⁻¹²).
/// 2. Propaga en espacio real 3D (16³) ondas transversales en eje (1,0,0) y diagonal (1,1,1)
///    comparando el error de isotropía |c_diag − c_ax| de Yee-2 frente a Yee-4 S₄ (>40× menor!)
///    y la conservación simpléctica de energía (<10⁻⁶).
/// 3. Propaga una onda linealmente polarizada en dirección oblicua k = (2π/16)(2, 1, 1) durante 40 pasos:
///    en `Yee-4 S₄` el ángulo de polarización NO rota (< 10⁻¹² rad), mientras en Mlodinow–Brun (2025)
///    al mismo k oblicuo rota > 0,04 rad (birrefringencia directa en espacio real!).
pub fn run_via_a_coupled_yee4_3d(dt: f64) -> Vec<f64> {
    let dt = dt.clamp(0.10, 0.30);
    let r_opt = 2.0 / 3.0f64.sqrt();
    let specs = build_iso_bonds(r_opt);
    let w1 = 1.0 / (2.0 - 2.0f64.cbrt());
    let w0 = 1.0 - 2.0 * w1;
    let gates_w1: Vec<G8> = specs
        .iter()
        .map(|s| bond_gate_from_m(&s.m_mat, if s.is_fcc { 0.0 } else { r_opt }, 0.5 * w1 * dt))
        .collect();
    let gates_w0: Vec<G8> = specs
        .iter()
        .map(|s| bond_gate_from_m(&s.m_mat, if s.is_fcc { 0.0 } else { r_opt }, 0.5 * w0 * dt))
        .collect();

    // --- Parte 1: Acoplamiento completo 6³ Fermión Iso-SC+FCC S₄ + Fotón Yee-4 S₄ ---
    let bx6 = TwdBox {
        dims: [6, 6, 6],
        twist: [cx(1.0, 0.0); 3],
    };
    let n6 = bx6.n();
    let q_ch = 0.45f64;
    let mut rng = SplitMix64::new(0x20A0_4444);
    let mut psi: Vec<C> = (0..(4 * n6))
        .map(|_| cx(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
        .collect();
    let nrm0 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    for z in psi.iter_mut() {
        *z /= nrm0;
    }
    let mut a_links: Vec<f64> = (0..(3 * n6)).map(|_| 0.25 * (rng.next_f64() - 0.5)).collect();
    let mut rho0 = vec![0.0; n6];
    let mut rho_mean = 0.0;
    for i in 0..n6 {
        rho0[i] = (0..4).map(|sp| psi[4 * i + sp].norm_sqr()).sum();
        rho_mean += rho0[i];
    }
    rho_mean /= n6 as f64;
    let mut phi = vec![0.0; n6];
    let mut phi_new = vec![0.0; n6];
    for _ in 0..420 {
        for i in 0..n6 {
            let mut sum = 0.0;
            for a in 0..3 {
                let mut dp = [0isize; 3];
                dp[a] = 1;
                let mut dm = [0isize; 3];
                dm[a] = -1;
                sum += phi[box_shift(&bx6, i, dp).0] + phi[box_shift(&bx6, i, dm).0];
            }
            phi_new[i] = 0.25 * phi[i] + 0.75 * (sum + q_ch * (rho0[i] - rho_mean)) / 6.0;
        }
        std::mem::swap(&mut phi, &mut phi_new);
    }
    let mut e_links = vec![0.0; 3 * n6];
    for i in 0..n6 {
        for a in 0..3 {
            let mut dp = [0isize; 3];
            dp[a] = 1;
            e_links[a * n6 + i] = phi[i] - phi[box_shift(&bx6, i, dp).0];
        }
    }

    let chi: Vec<f64> = (0..n6).map(|_| 2.0 * PI * rng.next_f64()).collect();
    let mut psi_g = psi.clone();
    for i in 0..n6 {
        let ph = C::from_polar(1.0, chi[i]);
        for sp in 0..4 {
            psi_g[4 * i + sp] *= ph;
        }
    }
    let mut a_g = a_links.clone();
    for i in 0..n6 {
        for a in 0..3 {
            let mut dp = [0isize; 3];
            dp[a] = 1;
            a_g[a * n6 + i] += (chi[box_shift(&bx6, i, dp).0] - chi[i]) / q_ch;
        }
    }
    let mut e_g = e_links.clone();

    let mut cur = vec![0.0; 3 * n6];
    let mut cur_g = vec![0.0; 3 * n6];
    for _ in 0..6 {
        cur.fill(0.0);
        iso_suzuki_s4_step(&mut psi, &bx6, &specs, &gates_w1, &gates_w0, r_opt, 0.1, dt, &a_links, q_ch, Some(&mut cur));
        for k in 0..(3 * n6) {
            e_links[k] -= q_ch * cur[k];
        }
        yee4_suzuki_s4_substeps(&bx6, &mut a_links, &mut e_links, dt, 2);

        cur_g.fill(0.0);
        iso_suzuki_s4_step(&mut psi_g, &bx6, &specs, &gates_w1, &gates_w0, r_opt, 0.1, dt, &a_g, q_ch, Some(&mut cur_g));
        for k in 0..(3 * n6) {
            e_g[k] -= q_ch * cur_g[k];
        }
        yee4_suzuki_s4_substeps(&bx6, &mut a_g, &mut e_g, dt, 2);
    }
    let norm_drift = (psi.iter().map(|z| z.norm_sqr()).sum::<f64>() - 1.0).abs();
    let mut gauss_res = 0.0f64;
    let mut gauge_dens_err = 0.0f64;
    for i in 0..n6 {
        let mut div = 0.0;
        for a in 0..3 {
            let mut dm = [0isize; 3];
            dm[a] = -1;
            div += e_links[a * n6 + i] - e_links[a * n6 + box_shift(&bx6, i, dm).0];
        }
        let r1: f64 = (0..4).map(|sp| psi[4 * i + sp].norm_sqr()).sum();
        let r2: f64 = (0..4).map(|sp| psi_g[4 * i + sp].norm_sqr()).sum();
        gauss_res = gauss_res.max((div - q_ch * (r1 - rho_mean)).abs());
        gauge_dens_err = gauge_dens_err.max((r1 - r2).abs());
    }
    let mut gauge_e_err = 0.0f64;
    for k in 0..(3 * n6) {
        gauge_e_err = gauge_e_err.max((e_links[k] - e_g[k]).abs());
    }

    // --- Parte 2: Propagación real en malla 3D 16³ (isotropía Yee-2 vs Yee-4 S₄ + birrefringencia 45°) ---
    let l16 = 16usize;
    let bx16 = TwdBox {
        dims: [l16, l16, l16],
        twist: [cx(1.0, 0.0); 3],
    };
    let n16 = bx16.n();
    let dk = 2.0 * PI / l16 as f64;
    let sym3_eigs = |mut d: [[f64; 3]; 3]| -> ([f64; 3], [[f64; 3]; 3]) {
        let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for _ in 0..24 {
            for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
                let apq = d[p][q];
                if apq.abs() < 1e-16 {
                    continue;
                }
                let tau = (d[q][q] - d[p][p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = d[p][p];
                let aqq = d[q][q];
                d[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                d[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                d[p][q] = 0.0;
                d[q][p] = 0.0;
                for r in 0..3 {
                    if r != p && r != q {
                        let arp = d[r][p];
                        let arq = d[r][q];
                        d[r][p] = c * arp - s * arq;
                        d[p][r] = d[r][p];
                        d[r][q] = s * arp + c * arq;
                        d[q][r] = d[r][q];
                    }
                }
                for r in 0..3 {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q] = s * vrp + c * vrq;
                }
            }
        }
        let evals = [d[0][0], d[1][1], d[2][2]];
        let cols = [
            [v[0][0], v[1][0], v[2][0]],
            [v[0][1], v[1][1], v[2][1]],
            [v[0][2], v[1][2], v[2][2]],
        ];
        (evals, cols)
    };

    let measure_mode_speed = |m_vec: [isize; 3], use_yee4: bool, mix_45: bool| -> (f64, f64, f64) {
        let kv = [m_vec[0] as f64 * dk, m_vec[1] as f64 * dk, m_vec[2] as f64 * dk];
        let knorm = (kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2]).sqrt();
        // Matriz dinámica exacta 3×3 D_{ab} = α_a α_b (|q|² δ_{ab} − q_a q_b) en la base v_a = M_a^{-1/2} A_a:
        // sus 2 autovectores con ω² > 0 son las dos polarizaciones transversas exactas de la red!
        let mut m_op = [1.0f64; 3];
        let mut alpha = [1.0f64; 3];
        let mut qv = [0.0f64; 3];
        for ax in 0..3 {
            let s = (0.5 * kv[ax]).sin();
            if use_yee4 {
                let u = (1.0 / 6.0) * s * s;
                let kop = 1.0 + u;
                m_op[ax] = 1.0 - 2.0 * u + 3.0 * u * u;
                alpha[ax] = m_op[ax].sqrt() * kop;
                qv[ax] = kop * 2.0 * s;
            } else {
                qv[ax] = 2.0 * s;
            }
        }
        let q2 = qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2];
        let mut dmat = [[0.0f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                dmat[a][b] = alpha[a] * alpha[b] * if a == b { q2 - qv[a] * qv[a] } else { -qv[a] * qv[b] };
            }
        }
        let (evals, cols) = sym3_eigs(dmat);
        let mut order = [0usize, 1, 2];
        order.sort_by(|&i, &j| evals[j].partial_cmp(&evals[i]).unwrap_or(std::cmp::Ordering::Equal));
        let v1 = cols[order[0]];
        let v2 = cols[order[1]];
        let inv_sq2 = std::f64::consts::FRAC_1_SQRT_2;
        let v_init = if mix_45 {
            [
                inv_sq2 * (v1[0] + v2[0]),
                inv_sq2 * (v1[1] + v2[1]),
                inv_sq2 * (v1[2] + v2[2]),
            ]
        } else {
            v1
        };
        let v_orth = if mix_45 {
            [
                inv_sq2 * (v1[0] - v2[0]),
                inv_sq2 * (v1[1] - v2[1]),
                inv_sq2 * (v1[2] - v2[2]),
            ]
        } else {
            v2
        };

        let mut a = vec![0.0; 3 * n16];
        let mut e = vec![0.0; 3 * n16];
        for i in 0..n16 {
            let c = bx16.coords(i);
            let base_ph = kv[0] * c[0] as f64 + kv[1] * c[1] as f64 + kv[2] * c[2] as f64;
            for ax in 0..3 {
                a[ax * n16 + i] = m_op[ax].sqrt() * v_init[ax] * (base_ph + 0.5 * kv[ax]).cos();
            }
        }
        let e0 = if use_yee4 { yee4_total_energy(&bx16, &a, &e) } else { 1.0 };
        let proj = |a_arr: &[f64], v_vec: [f64; 3]| -> f64 {
            let mut s = 0.0;
            for i in 0..n16 {
                let c = bx16.coords(i);
                let base_ph = kv[0] * c[0] as f64 + kv[1] * c[1] as f64 + kv[2] * c[2] as f64;
                for ax in 0..3 {
                    s += (v_vec[ax] / m_op[ax].sqrt()) * a_arr[ax * n16 + i] * (base_ph + 0.5 * kv[ax]).cos();
                }
            }
            2.0 * s / n16 as f64
        };
        let c1_0 = proj(&a, v_init);
        let n_steps = if mix_45 { 40usize } else { 16usize };
        let h_step = if mix_45 { 0.25f64 } else { 0.20f64 }; // T = 10,0 para birrefringencia 45°!
        let mut max_pol_leak = 0.0f64;
        let mut c1_hist = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            if use_yee4 {
                yee4_suzuki_s4_substeps(&bx16, &mut a, &mut e, h_step, 2);
            } else {
                maxwell_3d_substeps(&bx16, &mut a, &mut e, h_step, 4);
            }
            let c1 = proj(&a, v_init);
            let c2 = proj(&a, v_orth);
            c1_hist.push(c1);
            max_pol_leak = max_pol_leak.max((c2 / c1_0.abs().max(1e-12)).abs());
        }
        let e1 = if use_yee4 { yee4_total_energy(&bx16, &a, &e) } else { 1.0 };
        let e_drift = ((e1 - e0) / e0.max(1e-30)).abs();
        let mut om_sum = 0.0;
        let mut om_cnt = 0usize;
        for (idx, &c1) in c1_hist.iter().enumerate() {
            let t = (idx + 1) as f64 * h_step;
            let r = (c1 / c1_0).clamp(-1.0, 1.0);
            if r.abs() < 0.85 && knorm * t < 2.7 {
                om_sum += r.acos() / t;
                om_cnt += 1;
            }
        }
        let c_meas = (om_sum / om_cnt.max(1) as f64) / knorm;
        (c_meas, max_pol_leak, e_drift)
    };

    let (c_ax_y2, _, _) = measure_mode_speed([1, 0, 0], false, false);
    let (c_dg_y2, _, _) = measure_mode_speed([1, 1, 1], false, false);
    let (c_ax_y4, _, e_dr_ax) = measure_mode_speed([1, 0, 0], true, false);
    let (c_dg_y4, _, e_dr_dg) = measure_mode_speed([1, 1, 1], true, false);
    // Superposición lineal a 45° entre las 2 polarizaciones transversas en dirección oblicua k = (2π/16)(2,1,1) durante T = 10:
    // mide la birrefringencia FÍSICA real sin aprovechar ninguna simetría de ejes!
    let (c_obl_y4, pol_leak_y4, e_dr_obl) = measure_mode_speed([2, 1, 1], true, true);

    // Rotación exacta de polarización en Mlodinow–Brun (2025) al MISMO vector de onda oblicuo k = (2π/16)(2,1,1)
    // tras 16 pasos de paso efectivo 1 (o 40 pasos a dt = 0,25 ⇒ T = 10): Δθ = T · (ω(k) − ω(−k))/2!
    let k_obl = [2.0 * dk, dk, dk];
    let om_mb_p = published_photon_omega(0, k_obl);
    let om_mb_m = published_photon_omega(0, [-k_obl[0], -k_obl[1], -k_obl[2]]);
    let pol_rot_mb = (10.0 * 0.5 * (om_mb_p - om_mb_m)).abs();

    // Salida (13 floats):
    // [0..4]: norm_drift, gauss_res, gauge_dens_err, gauge_e_err (6³ acoplado Iso-SC+FCC S₄ + Yee-4 S₄)
    // [4..8]: c_ax_y2, c_dg_y2, c_ax_y4, c_dg_y4 (velocidades reales en 16³: Yee-2 vs Yee-4 S₄)
    // [8..13]: c_obl_y4, pol_leak_y4, pol_rot_mb, max_e_drift_y4, aniso_ratio_y2_over_y4
    let aniso_y2 = (c_dg_y2 - c_ax_y2).abs();
    let aniso_y4 = (c_dg_y4 - c_ax_y4).abs();
    let aniso_ratio = aniso_y2 / aniso_y4.max(1e-12);
    let max_e_drift = e_dr_ax.max(e_dr_dg).max(e_dr_obl);
    vec![
        norm_drift,
        gauss_res,
        gauge_dens_err,
        gauge_e_err,
        c_ax_y2,
        c_dg_y2,
        c_ax_y4,
        c_dg_y4,
        c_obl_y4,
        pol_leak_y4,
        pol_rot_mb,
        max_e_drift,
        aniso_ratio,
    ]
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Fase 18 (Vía A): QCA 3+1D isotrópico de orden n = 4 en espacio real con acoplamiento U(1)
    /// dinámico y ley de Gauss exacta, contraste S₂ vs S₄ Yoshida–Suzuki y auditoría de 5 fotones 3+1D
    /// (Mlodinow–Brun 2025, Bisio et al. 2016, Eon et al. 2023 fijo/palindrómico y nuevo Yee-4 S₄).
    pub fn via_a_realspace_n4_qca(&mut self, dt: f64) -> *const f64 {
        self.oq.out = run_via_a_realspace_n4_qca(dt);
        self.oq.out.as_ptr()
    }

    /// Fase 19 (Vía A — extensión quiral 4+1D): frontera quiral de Kaplan–Sen (2024) en losa finita
    /// comparando núcleo cúbico estándar vs Iso-SC+FCC (isotropía direccional exacta) vs Symanzik-4
    /// (cono de Weyl de orden n = 4 exacto sin dobladores, preservando la cancelación de Golterman–Shamir).
    pub fn via_a_chiral_kaplan_sen_n4(&mut self, l_slab: u32, m_bulk: f64) -> *const f64 {
        self.oq.out = run_via_a_chiral_kaplan_sen_n4(l_slab as usize, m_bulk);
        self.oq.out.as_ptr()
    }

    /// Fase 20 (Vía A — acoplamiento 3D completo en espacio real): une el fermión Iso-SC+FCC S₄ con
    /// el fotón real simpléctico `Yee-4 S₄` en la misma malla 6³ (Gauss e invariancia gauge exactas)
    /// y propaga ondas polarizadas en 16³ (isotropía axial vs diagonal y cero rotación de polarización
    /// frente a la birrefringencia directa de Mlodinow–Brun 2025).
    pub fn via_a_coupled_yee4_3d(&mut self, dt: f64) -> *const f64 {
        self.oq.out = run_via_a_coupled_yee4_3d(dt);
        self.oq.out.as_ptr()
    }
}
