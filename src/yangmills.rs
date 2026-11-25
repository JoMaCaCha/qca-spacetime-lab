//! Campos de Yang–Mills clásicos SU(2) y SU(3) con materia de color (§2.11) y quarks de Dirac
//! (§2.14). Teoría gauge en red de Kogut–Susskind, clásica:
//! - U_a(r) ∈ SU(N) transporta el color de r a r + â; Ẽ_a(r) ∈ ℝ^{N²−1} vive en el marco de r.
//! - Paso del campo: Ẽ_a ← Ẽ_a − q²·J_a (una vez); s veces: Ẽ^c_a ← Ẽ^c_a + (2κ/s)·Σ_b Im Tr(T^c·[P_ab + Q_ab]),
//!   U_a ← U_a·exp(−iẼ_a·T/s).
//! - Ley de Gauss: G(r) = Σ_a [Ẽ_a(r) − Ad(U_a(r−â))·Ẽ_a(r−â)] − q²·ρ(r), conservada exactamente.
//! Es Yang–Mills clásico real: los gluones tienen color, interactúan entre sí y la dinámica es
//! caótica, pero **no confina** (el confinamiento es cuántico).

use crate::fftn::FftN;
use crate::sun::{basis_colour, Algebra, Mat, C};
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub struct YmField {
    pub n: usize,
    pub ng: usize,
    pub alg: Algebra,
    /// U por enlace: dim·N_celdas matrices N×N (2N² f64 cada una).
    pub u: Vec<f64>,
    pub e: Vec<f64>,
    pub j: Vec<f64>,
    pub force: Vec<f64>,
    pub rho_ext: Vec<f64>,
    pub g_ref: Vec<f64>,
    pub energy_density: Vec<f32>,
    pub elec_density: Vec<f32>,
    pub mag_density: Vec<f32>,
    pub sol_colour: Vec<f32>,
    pub reunit_counter: u32,
}

impl YmField {
    pub fn new(n: usize, dim: usize, cells: usize) -> Self {
        let alg = Algebra::new(n);
        let ng = alg.ng;
        let m2 = 2 * n * n;
        let mut u = vec![0.0; dim * cells * m2];
        for l in 0..dim * cells {
            for i in 0..n {
                u[l * m2 + 2 * (i * n + i)] = 1.0;
            }
        }
        YmField {
            n,
            ng,
            alg,
            u,
            e: vec![0.0; dim * cells * ng],
            j: vec![0.0; dim * cells * ng],
            force: vec![0.0; dim * cells * ng],
            rho_ext: vec![0.0; cells * ng],
            g_ref: vec![0.0; cells * ng],
            energy_density: vec![0.0; cells],
            elec_density: vec![0.0; cells],
            mag_density: vec![0.0; cells],
            sol_colour: Vec::new(),
            reunit_counter: 0,
        }
    }

    #[inline(always)]
    pub fn link(&self, l: usize) -> Mat {
        let m2 = 2 * self.n * self.n;
        Mat::load(self.n, &self.u[l * m2..(l + 1) * m2])
    }

    #[inline(always)]
    pub fn set_link(&mut self, l: usize, m: &Mat) {
        let m2 = 2 * self.n * self.n;
        m.store(&mut self.u[l * m2..(l + 1) * m2]);
    }
}

#[inline(always)]
fn cz() -> C {
    Complex::new(0.0, 0.0)
}

/// SU(2) como cuaternión: U = u₀·1 + i·(u₁σ₁ + u₂σ₂ + u₃σ₃). Las sumas de matrices de SU(2) siguen
/// teniendo esa forma, el producto cuesta 16 multiplicaciones reales y Im Tr(T^c·M) = m_c.
type Quat = [f64; 4];

/// Lee el cuaternión de una matriz 2×2 guardada como [re, im] por entradas (fila mayor):
/// m₀₀ = u₀ + i·u₃ y m₀₁ = u₂ + i·u₁.
#[inline(always)]
fn quat_load(m: &[f64]) -> Quat {
    [m[0], m[3], m[2], m[1]]
}

#[inline(always)]
fn quat_store(q: &Quat, m: &mut [f64]) {
    // [[u₀ + iu₃, u₂ + iu₁], [−u₂ + iu₁, u₀ − iu₃]]
    m[0] = q[0];
    m[1] = q[3];
    m[2] = q[2];
    m[3] = q[1];
    m[4] = -q[2];
    m[5] = q[1];
    m[6] = q[0];
    m[7] = -q[3];
}

/// a·b: (a₀b₀ − a·b, a₀b + b₀a − a×b).
#[inline(always)]
fn quat_mul(a: &Quat, b: &Quat) -> Quat {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let b01 = v128_load(b.as_ptr() as *const v128); // [b0, b1]
        let b23 = v128_load(b.as_ptr().add(2) as *const v128); // [b2, b3]
        let b10 = i64x2_shuffle::<1, 0>(b01, b01); // [b1, b0]
        let b32 = i64x2_shuffle::<1, 0>(b23, b23); // [b3, b2]
        let sign_lo = u64x2(0x8000_0000_0000_0000, 0);
        let sign_hi = u64x2(0, 0x8000_0000_0000_0000);
        let a0 = f64x2_splat(a[0]);
        let a1 = f64x2_splat(a[1]);
        let a2 = f64x2_splat(a[2]);
        let a3 = f64x2_splat(a[3]);
        // lo = a0*[b0,b1] + a1*[-b1,b0] + a2*[-b2,-b3] + a3*[-b3,b2]
        let lo = f64x2_add(
            f64x2_add(f64x2_mul(a0, b01), f64x2_mul(a1, v128_xor(b10, sign_lo))),
            f64x2_sub(f64x2_mul(a3, v128_xor(b32, sign_lo)), f64x2_mul(a2, b23)),
        );
        // hi = a0*[b2,b3] + a1*[b3,-b2] + a2*[b0,b1] + a3*[-b1,b0]
        let hi = f64x2_add(
            f64x2_add(f64x2_mul(a0, b23), f64x2_mul(a1, v128_xor(b32, sign_hi))),
            f64x2_add(f64x2_mul(a2, b01), f64x2_mul(a3, v128_xor(b10, sign_lo))),
        );
        let mut out = [0.0f64; 4];
        v128_store(out.as_mut_ptr() as *mut v128, lo);
        v128_store(out.as_mut_ptr().add(2) as *mut v128, hi);
        return out;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1] - (a[2] * b[3] - a[3] * b[2]),
        a[0] * b[2] + b[0] * a[2] - (a[3] * b[1] - a[1] * b[3]),
        a[0] * b[3] + b[0] * a[3] - (a[1] * b[2] - a[2] * b[1]),
    ]
}

#[inline(always)]
fn quat_dag(a: &Quat) -> Quat {
    [a[0], -a[1], -a[2], -a[3]]
}

/// exp(−i·θ·σ/2) = cos(|θ|/2) − i·sin(|θ|/2)·n·σ.
#[inline(always)]
fn quat_exp_minus_i(th: &[f64]) -> Quat {
    let nrm = (th[0] * th[0] + th[1] * th[1] + th[2] * th[2]).sqrt();
    if nrm < 1e-300 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let (sn, cs) = (0.5 * nrm).sin_cos();
    let f = -sn / nrm;
    [cs, f * th[0], f * th[1], f * th[2]]
}

impl QuantumSpacetimeUniverse {
    /// Vector de color del canal k en la celda i.
    #[inline(always)]
    pub(crate) fn colour_vec(&self, i: usize, k: usize) -> [C; 3] {
        if self.sw.active {
            return self.sw_colour_vec(i, k);
        }
        let nco = self.ncol;
        let b = i * 2 * self.nc * nco + 2 * k * nco;
        let mut v = [cz(); 3];
        for a in 0..nco {
            v[a] = Complex::new(self.buffer_read[b + 2 * a] as f64, self.buffer_read[b + 2 * a + 1] as f64);
        }
        v
    }

    /// ρ^c(r) = Σ_k ψ_k†·T^c·ψ_k.
    pub(crate) fn colour_density(&self, i: usize, out: &mut [f64]) {
        if self.sw.active && self.sw.distinct && !self.sw.parts.is_empty() {
            self.sw_colour_density_parts(i, out);
            return;
        }
        let ng = self.ym.ng;
        for v in out.iter_mut().take(ng) {
            *v = 0.0;
        }
        let mut tmp = [0.0f64; 8];
        for k in 0..self.nc {
            let v = self.colour_vec(i, k);
            self.ym.alg.charge(&v, &mut tmp);
            for c in 0..ng {
                out[c] += tmp[c];
            }
        }
    }

    /// P_ab(r) = U_b(r)†·U_a(r+b̂)†·U_b(r+â)·U_a(r) (lazo +a, +b, −a, −b en el marco de r).
    pub(crate) fn ym_plaq_p(&self, i: usize, a: usize, b: usize) -> Mat {
        let n = self.n;
        let ia = self.nb(i, a, 1);
        let ib = self.nb(i, b, 1);
        let ua = self.ym.link(a * n + i);
        let ub = self.ym.link(b * n + i);
        let ua_b = self.ym.link(a * n + ib);
        let ub_a = self.ym.link(b * n + ia);
        ub.dag_mul(&ua_b.dag_mul(&ub_a.mul(&ua)))
    }

    /// Fuerza de Wilson F^c_a(r) = 2κ·Σ_b Im Tr(T^c·[P_ab + Q_ab]) en `force`. P_ab y Q_ab acaban
    /// en U_a(r): se suman primero sus grapas y se multiplica una sola vez por U_a(r).
    pub(crate) fn ym_compute_force(&mut self) {
        if self.ym.n == 2 {
            self.ym_compute_force_su2();
            return;
        }
        let n = self.n;
        let dim = self.dim;
        let ng = self.ym.ng;
        let kap2 = 2.0 * self.em.kappa;
        let mut force = std::mem::take(&mut self.ym.force);
        let mut tmp = [0.0f64; 8];
        for a in 0..dim {
            for i in 0..n {
                let ia = self.nb(i, a, 1);
                let mut staple = Mat::zero(self.ym.n);
                for b in 0..dim {
                    if b == a {
                        continue;
                    }
                    // P: U_b(r)†·U_a(r+b̂)†·U_b(r+â)
                    let ib = self.nb(i, b, 1);
                    let ub = self.ym.link(b * n + i);
                    let ua_b = self.ym.link(a * n + ib);
                    let ub_a = self.ym.link(b * n + ia);
                    staple = staple.add(&ub.dag_mul(&ua_b.dag_mul(&ub_a)));
                    // Q: U_b(r−b̂)·U_a(r−b̂)†·U_b(r−b̂+â)† = U_b(r−b̂)·[U_b(r−b̂+â)·U_a(r−b̂)]†
                    let imb = self.nb(i, b, -1);
                    let imb_a = self.nb(imb, a, 1);
                    let ub_mb = self.ym.link(b * n + imb);
                    let ua_mb = self.ym.link(a * n + imb);
                    let ub_mba = self.ym.link(b * n + imb_a);
                    staple = staple.add(&ub_mb.mul_dag(&ub_mba.mul(&ua_mb)));
                }
                let acc = staple.mul(&self.ym.link(a * n + i));
                self.ym.alg.im_tr(&acc, &mut tmp);
                let o = (a * n + i) * ng;
                for c in 0..ng {
                    force[o + c] = kap2 * tmp[c];
                }
            }
        }
        self.ym.force = force;
    }

    /// La misma fuerza en SU(2), con cuaterniones.
    fn ym_compute_force_su2(&mut self) {
        let n = self.n;
        let dim = self.dim;
        let kap2 = 2.0 * self.em.kappa;
        let mut force = std::mem::take(&mut self.ym.force);
        let u = &self.ym.u;
        let q = |l: usize| quat_load(&u[8 * l..8 * l + 8]);
        for a in 0..dim {
            for i in 0..n {
                let ia = self.nb(i, a, 1);
                let mut st = [0.0f64; 4];
                for b in 0..dim {
                    if b == a {
                        continue;
                    }
                    let ib = self.nb(i, b, 1);
                    // P: U_b(r)†·U_a(r+b̂)†·U_b(r+â)
                    let p = quat_mul(&quat_dag(&q(b * n + i)), &quat_mul(&quat_dag(&q(a * n + ib)), &q(b * n + ia)));
                    // Q: U_b(r−b̂)·[U_b(r−b̂+â)·U_a(r−b̂)]†
                    let imb = self.nb(i, b, -1);
                    let imb_a = self.nb(imb, a, 1);
                    let qq = quat_mul(&q(b * n + imb), &quat_dag(&quat_mul(&q(b * n + imb_a), &q(a * n + imb))));
                    for k in 0..4 {
                        st[k] += p[k] + qq[k];
                    }
                }
                let m = quat_mul(&st, &q(a * n + i));
                let o = (a * n + i) * 3;
                force[o] = kap2 * m[1];
                force[o + 1] = kap2 * m[2];
                force[o + 2] = kap2 * m[3];
            }
        }
        self.ym.force = force;
    }

    /// Subpasos del campo SU(N) (con el amortiguamiento de `set_relaxation` dentro de cada uno).
    pub(crate) fn ym_substeps(&mut self) {
        let s = self.em.substeps.max(1);
        let inv_s = 1.0 / s as f64;
        let ng = self.ym.ng;
        let links = self.dim * self.n;
        let damp = 1.0 - self.em.eta;
        for _ in 0..s {
            self.ym_compute_force();
            for l in 0..links * ng {
                self.ym.e[l] = (self.ym.e[l] + inv_s * self.ym.force[l]) * damp;
            }
            let mut th = [0.0f64; 8];
            if self.ym.n == 2 {
                for l in 0..links {
                    for c in 0..3 {
                        th[c] = self.ym.e[l * 3 + c] * inv_s;
                    }
                    let m = &mut self.ym.u[8 * l..8 * l + 8];
                    let un = quat_mul(&quat_load(m), &quat_exp_minus_i(&th[..3]));
                    quat_store(&un, m);
                }
            } else {
                for l in 0..links {
                    for c in 0..ng {
                        th[c] = self.ym.e[l * ng + c] * inv_s;
                    }
                    let ex = self.ym.alg.exp_minus_i(&th[..ng]);
                    let u = self.ym.link(l).mul(&ex);
                    self.ym.set_link(l, &u);
                }
            }
        }
        self.ym.reunit_counter = self.ym.reunit_counter.wrapping_add(1);
        if self.ym.reunit_counter % 100 == 0 {
            for l in 0..links {
                let mut u = self.ym.link(l);
                u.reunitarize();
                self.ym.set_link(l, &u);
            }
        }
    }

    /// Paso del campo tras la materia (modo dinámico): Ẽ −= q²·J y los subpasos.
    pub(crate) fn ym_after_step(&mut self) {
        if self.em.mode != 2 {
            return;
        }
        let q2 = self.em.q * self.em.q;
        for l in 0..self.ym.e.len() {
            self.ym.e[l] -= q2 * self.ym.j[l];
        }
        self.ym_substeps();
    }

    /// Paso de la caminata de Grover con color (§2.11): moneda por color, fase con E total,
    /// corrientes de color y desplazamiento con los U. Sin tope.
    pub(crate) fn grover_colour_step<const NC: usize>(&mut self) {
        let nco = self.ncol;
        let namp = NC * nco;
        let stride = 2 * namp;
        let n = self.n;
        // E_eff
        let inv = 1.0 / self.e_max;
        for i in 0..n {
            let mut e = 0.0f32;
            for k in 0..stride {
                let v = self.buffer_read[i * stride + k];
                e += v * v;
            }
            self.e_eff[i] = e / (1.0 + e * inv);
        }
        // moneda + fase
        let (cm, sm) = (self.cos_m, self.sin_m);
        let f = 2.0 / NC as f32;
        let gamma_phi = self.gravity_phase_active();
        let gam = if gamma_phi { self.grav.gamma_eff() } else { 0.0 };
        for z in 0..self.d {
            for y in 0..self.h {
                for x in 0..self.w {
                    let i = (z * self.h + y) * self.w + x;
                    let mut wnl = self.omega_nl_at::<NC>(x, y, z);
                    if gamma_phi {
                        wnl += gam * self.phi_total(i);
                    }
                    let (ct, st) = crate::util::fast_cos_sin(self.potential[i] - wnl);
                    let p = &mut self.buffer_read[i * stride..(i + 1) * stride];
                    for a in 0..nco {
                        let (mut sr, mut si) = (0.0f32, 0.0f32);
                        for k in 0..NC {
                            sr += p[2 * (k * nco + a)];
                            si += p[2 * (k * nco + a) + 1];
                        }
                        let (fsr, fsi) = (f * sr, f * si);
                        for k in 0..NC {
                            let o = 2 * (k * nco + a);
                            let (re, im) = (p[o], p[o + 1]);
                            let ar = cm * re + sm * (fsi - im);
                            let ai = cm * im - sm * (fsr - re);
                            p[o] = ar * ct + ai * st;
                            p[o + 1] = ai * ct - ar * st;
                        }
                    }
                }
            }
        }
        // corrientes de color (modo dinámico): J^c_a(r) = ψ_{+a}†T^cψ_{+a} − w†T^c w, w = U_a(r)†ψ_{−a}(r+â)
        let plus = [3usize, 1, 5];
        let minus = [2usize, 0, 4];
        let ng = self.ym.ng;
        if self.em.mode == 2 {
            let mut ta = [0.0f64; 8];
            let mut tb = [0.0f64; 8];
            for a in 0..self.dim {
                for i in 0..n {
                    let ip = self.nb(i, a, 1);
                    let vp = self.colour_vec(i, plus[a]);
                    let u = self.ym.link(a * n + i);
                    let w = u.apply_dag(&self.colour_vec(ip, minus[a]));
                    self.ym.alg.charge(&vp, &mut ta);
                    self.ym.alg.charge(&w, &mut tb);
                    let o = (a * n + i) * ng;
                    for c in 0..ng {
                        self.ym.j[o + c] = ta[c] - tb[c];
                    }
                }
            }
        }
        // desplazamiento con color: ψ'_{+a}(r) = U_a(r−â)·ψ_{+a}(r−â); ψ'_{−a}(r) = U_a(r)†·ψ_{−a}(r+â)
        {
            let src = std::mem::take(&mut self.buffer_read);
            let mut dst = std::mem::take(&mut self.buffer_write);
            let load = |s: &[f32], i: usize, k: usize| -> [C; 3] {
                let b = i * stride + 2 * k * nco;
                let mut v = [cz(); 3];
                for a in 0..nco {
                    v[a] = Complex::new(s[b + 2 * a] as f64, s[b + 2 * a + 1] as f64);
                }
                v
            };
            for i in 0..n {
                for a in 0..self.dim {
                    let im = self.nb(i, a, -1);
                    let ip = self.nb(i, a, 1);
                    let up = self.ym.link(a * n + im);
                    let vp = up.apply(&load(&src, im, plus[a]));
                    let uc = self.ym.link(a * n + i);
                    let vm = uc.apply_dag(&load(&src, ip, minus[a]));
                    let bp = i * stride + 2 * plus[a] * nco;
                    let bm = i * stride + 2 * minus[a] * nco;
                    for c in 0..nco {
                        dst[bp + 2 * c] = vp[c].re as f32;
                        dst[bp + 2 * c + 1] = vp[c].im as f32;
                        dst[bm + 2 * c] = vm[c].re as f32;
                        dst[bm + 2 * c + 1] = vm[c].im as f32;
                    }
                }
            }
            self.buffer_read = dst;
            self.buffer_write = src;
        }
        self.after_matter_step();
        self.t = self.t.wrapping_add(1);
        self.theta_ref = crate::util::wrap_angle32(self.theta_ref + self.mass);
    }

    /// Paso de los quarks de Dirac 2D (§2.14): la caminata de Dirac con enlaces SU(N).
    pub(crate) fn dirac2d_colour_step(&mut self) {
        let (w, h) = (self.w, self.h);
        let n = self.n;
        let nco = self.ncol;
        let stride = 4 * nco;
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let gauge = self.em.mode != 0;
        let dynamic = self.em.mode == 2;
        let ng = self.ym.ng;
        let ld = |b: &[f32], i: usize, comp: usize| -> [C; 3] {
            let o = i * stride + 2 * comp * nco;
            let mut v = [cz(); 3];
            for a in 0..nco {
                v[a] = Complex::new(b[o + 2 * a] as f64, b[o + 2 * a + 1] as f64);
            }
            v
        };
        let st = |b: &mut [f32], i: usize, comp: usize, v: &[C; 3]| {
            let o = i * stride + 2 * comp * nco;
            for a in 0..nco {
                b[o + 2 * a] = v[a].re as f32;
                b[o + 2 * a + 1] = v[a].im as f32;
            }
        };
        let mut ta = [0.0f64; 8];
        let mut tb = [0.0f64; 8];
        // corrientes en x (antes de T_x)
        if dynamic {
            for i in 0..n {
                let ip = self.nb(i, 0, 1);
                let up = ld(&self.buffer_read, i, 0);
                let w_ = self.ym.link(i).apply_dag(&ld(&self.buffer_read, ip, 1));
                self.ym.alg.charge(&up, &mut ta);
                self.ym.alg.charge(&w_, &mut tb);
                for c in 0..ng {
                    self.ym.j[i * ng + c] = ta[c] - tb[c];
                }
            }
        }
        // T_x y H → buffer_write
        {
            let src = std::mem::take(&mut self.buffer_read);
            let mut dst = std::mem::take(&mut self.buffer_write);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let im = self.nb(i, 0, -1);
                    let ip = self.nb(i, 0, 1);
                    let mut u = ld(&src, im, 0);
                    let mut d = ld(&src, ip, 1);
                    if gauge {
                        u = self.ym.link(im).apply(&u);
                        d = self.ym.link(i).apply_dag(&d);
                    }
                    let mut a = [cz(); 3];
                    let mut b = [cz(); 3];
                    for c in 0..nco {
                        a[c] = (u[c] + d[c]) * s;
                        b[c] = (u[c] - d[c]) * s;
                    }
                    st(&mut dst, i, 0, &a);
                    st(&mut dst, i, 1, &b);
                }
            }
            self.buffer_read = src;
            self.buffer_write = dst;
        }
        // corrientes en y (antes del salto en y)
        if dynamic {
            for i in 0..n {
                let ip = self.nb(i, 1, 1);
                let a = ld(&self.buffer_write, i, 0);
                let w_ = self.ym.link(n + i).apply_dag(&ld(&self.buffer_write, ip, 1));
                self.ym.alg.charge(&a, &mut ta);
                self.ym.alg.charge(&w_, &mut tb);
                for c in 0..ng {
                    self.ym.j[(n + i) * ng + c] = ta[c] - tb[c];
                }
            }
        }
        // salto en y, H, M y e^{−iV} → buffer_read
        let has_v = self.has_potential;
        let mass_field = self.dirac_mass_active();
        {
            let src = std::mem::take(&mut self.buffer_write);
            let mut dst = std::mem::take(&mut self.buffer_read);
            for i in 0..n {
                let im = self.nb(i, 1, -1);
                let ip = self.nb(i, 1, 1);
                let mut a = ld(&src, im, 0);
                let mut b = ld(&src, ip, 1);
                if gauge {
                    a = self.ym.link(n + im).apply(&a);
                    b = self.ym.link(n + i).apply_dag(&b);
                }
                let (cm, sm) = if mass_field { self.dirac_cs(i) } else { (self.cos_m, self.sin_m) };
                let (cm, sm) = (cm as f64, sm as f64);
                let phase = if has_v { C::from_polar(1.0, -(self.potential[i] as f64)) } else { Complex::new(1.0, 0.0) };
                let mut up = [cz(); 3];
                let mut dn = [cz(); 3];
                for c in 0..nco {
                    let u = (a[c] + b[c]) * s;
                    let d = (a[c] - b[c]) * s;
                    up[c] = (u * cm - d * sm) * phase;
                    dn[c] = (u * sm + d * cm) * phase;
                }
                st(&mut dst, i, 0, &up);
                st(&mut dst, i, 1, &dn);
            }
            self.buffer_read = dst;
            self.buffer_write = src;
        }
        if self.has_walls {
            self.absorb_walls();
        }
        self.t = self.t.wrapping_add(1);
    }

    /// En los escenarios de Dirac, los muros absorben: la amplitud que entra se pierde.
    pub(crate) fn absorb_walls(&mut self) {
        let stride = 2 * self.namp();
        for i in 0..self.n {
            if self.wall[i] != 0 {
                for k in 0..stride {
                    self.buffer_read[i * stride + k] = 0.0;
                }
            }
        }
    }

    /// G(r) en `out` (ng componentes).
    pub(crate) fn ym_gauss_at(&self, i: usize, out: &mut [f64]) {
        let n = self.n;
        let ng = self.ym.ng;
        let q2 = self.em.q * self.em.q;
        let mut rho = [0.0f64; 8];
        self.colour_density(i, &mut rho);
        let mut tmp = [0.0f64; 8];
        for c in 0..ng {
            out[c] = -q2 * rho[c];
        }
        for a in 0..self.dim {
            let im = self.nb(i, a, -1);
            for c in 0..ng {
                out[c] += self.ym.e[(a * n + i) * ng + c];
            }
            // Ad(U_a(r−â))·Ẽ_a(r−â)
            let u = self.ym.link(a * n + im);
            let x = self.ym.alg.combine(&self.ym.e[(a * n + im) * ng..(a * n + im + 1) * ng]);
            let ad = u.mul(&x).mul_dag(&u);
            self.ym.alg.components(&ad, &mut tmp);
            for c in 0..ng {
                out[c] -= tmp[c];
            }
        }
    }

    pub(crate) fn ym_store_gauss_ref(&mut self) {
        let ng = self.ym.ng;
        let mut g = std::mem::take(&mut self.ym.g_ref);
        let mut tmp = [0.0f64; 8];
        for i in 0..self.n {
            self.ym_gauss_at(i, &mut tmp);
            g[i * ng..(i + 1) * ng].copy_from_slice(&tmp[..ng]);
        }
        self.ym.g_ref = g;
    }

    pub(crate) fn ym_gauss_residual(&self) -> f64 {
        let ng = self.ym.ng;
        let mut tmp = [0.0f64; 8];
        let mut worst = 0.0f64;
        for i in 0..self.n {
            self.ym_gauss_at(i, &mut tmp);
            for c in 0..ng {
                worst = worst.max((tmp[c] - self.ym.g_ref[i * ng + c]).abs());
            }
        }
        worst
    }

    /// Resuelve −∇²φ^c = src^c (media nula, con U = 1) y suma −∇φ^c a Ẽ.
    pub(crate) fn ym_add_gradient(&mut self, src: &[f64]) {
        let n = self.n;
        let ng = self.ym.ng;
        let mut fft = FftN::<f64>::new(self.w, self.h, self.d);
        let mut buf = vec![Complex::new(0.0, 0.0); n];
        for c in 0..ng {
            let mean = (0..n).map(|i| src[i * ng + c]).sum::<f64>() / n as f64;
            for i in 0..n {
                buf[i] = Complex::new(src[i * ng + c] - mean, 0.0);
            }
            fft.forward(&mut buf);
            for i in 0..n {
                let (x, y, z) = self.coords(i);
                let mut lam = 2.0 * (1.0 - (2.0 * PI * x as f64 / self.w as f64).cos()) + 2.0 * (1.0 - (2.0 * PI * y as f64 / self.h as f64).cos());
                if self.dim == 3 {
                    lam += 2.0 * (1.0 - (2.0 * PI * z as f64 / self.d as f64).cos());
                }
                buf[i] = if lam > 1e-12 { buf[i] / lam } else { Complex::new(0.0, 0.0) };
            }
            fft.inverse(&mut buf);
            for a in 0..self.dim {
                for i in 0..n {
                    let ip = self.nb(i, a, 1);
                    self.ym.e[(a * n + i) * ng + c] -= buf[ip].re - buf[i].re;
                }
            }
        }
    }

    /// U = 1 y Ẽ con la ley de Gauss de la materia y de las cargas externas (§2.11).
    pub(crate) fn ym_init_gauss(&mut self) {
        let n = self.n;
        let ng = self.ym.ng;
        let nn = self.ym.n;
        let m2 = 2 * nn * nn;
        for l in 0..self.dim * n {
            for k in 0..m2 {
                self.ym.u[l * m2 + k] = 0.0;
            }
            for i in 0..nn {
                self.ym.u[l * m2 + 2 * (i * nn + i)] = 1.0;
            }
        }
        for v in self.ym.e.iter_mut() {
            *v = 0.0;
        }
        let q2 = self.em.q * self.em.q;
        let mut src = vec![0.0; n * ng];
        let mut rho = [0.0f64; 8];
        for i in 0..n {
            self.colour_density(i, &mut rho);
            for c in 0..ng {
                src[i * ng + c] = q2 * (rho[c] + self.ym.rho_ext[i * ng + c]);
            }
        }
        self.ym_add_gradient(&src);
        self.ym_store_gauss_ref();
    }

    /// Reasigna los búferes de la materia para n_c·N amplitudes por celda.
    pub(crate) fn resize_colour(&mut self, ncol: usize) {
        self.ncol = ncol;
        let len = self.n * 2 * self.nc_max * ncol.max(1);
        if self.buffer_read.len() != len {
            self.buffer_read = vec![0.0; len];
            self.buffer_write = vec![0.0; len];
        } else {
            self.buffer_read.iter_mut().for_each(|v| *v = 0.0);
            self.buffer_write.iter_mut().for_each(|v| *v = 0.0);
        }
        if ncol > 1 {
            self.ym = YmField::new(ncol, self.dim, self.n);
        } else {
            self.ym = YmField::new(1, self.dim, 0);
        }
        self.em.mode = 0;
        if self.sw.active {
            self.sw_alloc();
        }
    }

    /// Paquete con vector de color: φ·e_colour en todos los canales (Grover) o el espinor de la
    /// banda ⊗ color (Dirac).
    pub(crate) fn add_colour_packet(&mut self, cx: f64, cy: f64, cz_: f64, sigma: f64, amount: f64, k: [f64; 3], colour: [C; 3]) {
        if !(cx.is_finite()
            && cy.is_finite()
            && cz_.is_finite()
            && sigma.is_finite()
            && amount.is_finite()
            && k.iter().all(|v| v.is_finite())
            && colour.iter().all(|c| c.re.is_finite() && c.im.is_finite()))
            || sigma <= 0.0
            || amount <= 0.0
        {
            return;
        }
        let mut sum = 0.0f64;
        self.for_gaussian_cells(cx, cy, cz_, sigma, |_, _, e| sum += e * e);
        if sum <= 0.0 {
            return;
        }
        let amp = (amount / sum).sqrt();
        let nc = self.nc;
        let nco = self.ncol;
        let stride = 2 * nc * nco;
        let inv = 1.0 / (nc as f64).sqrt();
        let cn: f64 = colour.iter().take(nco).map(|v| v.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
        let mut buf = std::mem::take(&mut self.buffer_read);
        self.for_gaussian_cells(cx, cy, cz_, sigma, |cell, r, e| {
            let th = k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
            let ph = C::from_polar(amp * e * inv / cn, th);
            for ch in 0..nc {
                for a in 0..nco {
                    let v = ph * colour[a];
                    let o = cell * stride + 2 * (ch * nco + a);
                    buf[o] += v.re as f32;
                    buf[o + 1] += v.im as f32;
                }
            }
        });
        self.buffer_read = buf;
    }

    /// H = ½Σ Ẽ(t−½)·Ẽ(t+½) + W(t): la energía del leapfrog, que se conserva exactamente en la
    /// parte cuadrática. Con ½Σ|Ẽ|² a secas aparece un desfase O((ω·dt)²) de los modos
    /// ultravioleta que parece una deriva. Ẽ(t+½) = Ẽ(t−½) + F(U(t))/s.
    pub(crate) fn ym_energy(&mut self) -> f64 {
        let n = self.n;
        let s = self.em.substeps.max(1) as f64;
        let sync = self.em.mode == 2;
        if sync {
            self.ym_compute_force();
        }
        let e2: f64 = if sync {
            self.ym.e.iter().zip(self.ym.force.iter()).map(|(e, f)| e * (e + f / s)).sum()
        } else {
            self.ym.e.iter().map(|v| v * v).sum()
        };
        let mut w = 0.0;
        for i in 0..n {
            for a in 0..self.dim {
                for b in (a + 1)..self.dim {
                    w += self.ym.n as f64 - self.ym_plaq_p(i, a, b).trace().re;
                }
            }
        }
        0.5 * e2 + 2.0 * self.em.kappa * w
    }

    /// Carga de color de cada solitón por unidad de norma (la rellena analyze_solitons).
    pub(crate) fn ym_soliton_colours(&mut self) {
        if self.ncol <= 1 {
            return;
        }
        let ng = self.ym.ng;
        let count = self.sol.count;
        if count == 0 || self.sol.labels.len() != self.n {
            return;
        }
        if self.ym.sol_colour.len() < crate::MAX_SOLITONS * 8 {
            self.ym.sol_colour.resize(crate::MAX_SOLITONS * 8, 0.0);
        }
        for v in self.ym.sol_colour.iter_mut().take(count * 8) {
            *v = 0.0;
        }
        let mut rho = [0.0f64; 8];
        let mut acc = vec![0.0f64; count * ng];
        for i in 0..self.n {
            let l = self.sol.labels[i];
            if l < 0 || (l as usize) >= count {
                continue;
            }
            self.colour_density(i, &mut rho);
            for c in 0..ng {
                acc[l as usize * ng + c] += rho[c];
            }
        }
        for s in 0..count {
            let norm = self.sol.stats[s * crate::SOLITON_FIELDS + 1] as f64;
            for c in 0..ng {
                self.ym.sol_colour[s * ng + c] = if norm > 0.0 { (acc[s * ng + c] / norm) as f32 } else { 0.0 };
            }
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.11: 1 = U(1), 2 = SU(2), 3 = SU(3). Solo en el escenario materia; reasigna búferes y
    /// limpia el campo. Con SU(N) el tope se desactiva (§2.5).
    pub fn set_gauge_group(&mut self, n: u32) {
        self.message.clear();
        if self.scenario != 0 {
            self.set_message("El grupo gauge solo se elige en el escenario materia (en Dirac, con set_dirac_colour).");
            return;
        }
        let n = n.clamp(1, 3) as usize;
        self.higgs.mode = 0;
        self.resize_colour(n);
        self.em.substeps = if n > 1 { 2 } else { 1 };
        if n > 1 {
            self.em.q = 7e-3;
            self.e_cap = 0.0;
        }
        self.clear();
    }

    pub fn gauge_group(&self) -> u32 {
        self.ncol as u32
    }

    /// §2.14 (2D): 1 = sin color (U(1)), 2 = SU(2), 3 = SU(3) en el escenario Dirac.
    pub fn set_dirac_colour(&mut self, n: u32) {
        self.message.clear();
        if self.scenario != 2 {
            self.set_message("set_dirac_colour solo actúa en el escenario Dirac.");
            return;
        }
        let n = n.clamp(1, 3) as usize;
        self.resize_colour(n);
        self.em.substeps = 2;
        self.em.q = if n > 1 { 0.5 } else { 1.0 };
        self.em.kappa = 1.0;
        self.clear();
    }

    /// `inject_wavepacket` con el vector de color de la base (SU(2): 0 = ↑, 1 = ↓; SU(3): 0 = rojo,
    /// 1 = verde, 2 = azul). Con U(1) ignora `colour`.
    #[allow(clippy::too_many_arguments)]
    pub fn inject_colour_wavepacket(&mut self, x: i32, y: i32, z: i32, radius: f32, amount: f32, kx: f32, ky: f32, kz: f32, colour: u32) {
        if ![radius, amount, kx, ky, kz].iter().all(|v| v.is_finite()) || radius <= 0.0 || amount <= 0.0 {
            return;
        }
        if self.ncol <= 1 {
            self.inject_wavepacket(x, y, z, radius, amount, kx, ky, kz, 0.0);
            return;
        }
        let c = self.wrap_cell(x, y, z);
        let (cx, cy, czz) = self.coords(c);
        let kz = if self.dim == 3 { kz as f64 } else { 0.0 };
        self.add_colour_packet(cx as f64, cy as f64, czz as f64, radius as f64 / 2.0, amount as f64, [kx as f64, ky as f64, kz], basis_colour(self.ncol, colour as usize));
    }

    /// Paquete de color con un vector de color arbitrario (re/im de hasta 3 componentes).
    #[allow(clippy::too_many_arguments)]
    pub fn inject_colour_vector_packet(&mut self, x: f32, y: f32, z: f32, sigma: f32, amount: f32, kx: f32, ky: f32, c0r: f32, c0i: f32, c1r: f32, c1i: f32, c2r: f32, c2i: f32) {
        if ![x, y, z, sigma, amount, kx, ky, c0r, c0i, c1r, c1i, c2r, c2i].iter().all(|v| v.is_finite()) || sigma <= 0.0 || amount <= 0.0 {
            return;
        }
        let col = [Complex::new(c0r as f64, c0i as f64), Complex::new(c1r as f64, c1i as f64), Complex::new(c2r as f64, c2i as f64)];
        self.add_colour_packet(x as f64, y as f64, z as f64, sigma as f64, amount as f64, [kx as f64, ky as f64, 0.0], col);
    }

    /// Paquete de quark de Dirac 2D (norma 1): espinor de la banda +1 en k ⊗ color de la base.
    pub fn inject_dirac_colour_packet(&mut self, x: f32, y: f32, sigma: f32, kx: f32, ky: f32, colour: u32) {
        if ![x, y, sigma, kx, ky].iter().all(|v| v.is_finite()) || sigma <= 0.0 {
            return;
        }
        let col = basis_colour(self.ncol.max(1), colour as usize);
        self.add_dirac_colour_packet(x as f64, y as f64, sigma as f64, kx as f64, ky as f64, col);
    }

    /// Paquete de quark con un vector de color arbitrario (re/im de hasta 3 componentes).
    #[allow(clippy::too_many_arguments)]
    pub fn inject_dirac_colour_vector_packet(&mut self, x: f32, y: f32, sigma: f32, kx: f32, ky: f32, c0r: f32, c0i: f32, c1r: f32, c1i: f32, c2r: f32, c2i: f32) {
        if ![x, y, sigma, kx, ky, c0r, c0i, c1r, c1i, c2r, c2i].iter().all(|v| v.is_finite()) || sigma <= 0.0 {
            return;
        }
        let col = [Complex::new(c0r as f64, c0i as f64), Complex::new(c1r as f64, c1i as f64), Complex::new(c2r as f64, c2i as f64)];
        self.add_dirac_colour_packet(x as f64, y as f64, sigma as f64, kx as f64, ky as f64, col);
    }

    /// Quark estático (§2.11): ρ_ext = amount·w_colour con perfil gaussiano de σ = 2.
    pub fn add_static_colour_charge(&mut self, x: i32, y: i32, z: i32, amount: f32, colour: u32) {
        self.add_static_colour_charge_sigma(x, y, z, amount, colour, 2.0);
    }

    /// Igual que `add_static_colour_charge`, con la anchura σ elegida.
    #[allow(clippy::too_many_arguments)]
    pub fn add_static_colour_charge_sigma(&mut self, x: i32, y: i32, z: i32, amount: f32, colour: u32, sigma: f32) {
        if !(amount.is_finite() && sigma.is_finite()) {
            return;
        }
        if self.ncol <= 1 {
            self.add_static_charge(x, y, z, amount, sigma);
            return;
        }
        let ng = self.ym.ng;
        let v = basis_colour(self.ncol, colour as usize);
        let mut w = [0.0f64; 8];
        self.ym.alg.charge(&v, &mut w);
        let mut dir = [0.0f64; 8];
        dir[..ng].copy_from_slice(&w[..ng]);
        self.add_static_colour_vector(x, y, z, amount, sigma, &dir[..ng]);
    }

    /// Carga de color externa en una dirección arbitraria del álgebra (componentes sobre T^c).
    #[allow(clippy::too_many_arguments)]
    pub fn add_static_colour_vector(&mut self, x: i32, y: i32, z: i32, amount: f32, sigma: f32, dir: &[f64]) {
        if self.ncol <= 1 || !(amount.is_finite() && sigma.is_finite()) || !dir.iter().all(|v| v.is_finite()) {
            return;
        }
        let n = self.n;
        let ng = self.ym.ng;
        let c0 = self.wrap_cell(x, y, z);
        let (cx, cy, cz_) = self.coords(c0);
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let sg = (sigma as f64).max(0.3);
        let mut prof = vec![0.0f64; n];
        let mut tot = 0.0;
        for i in 0..n {
            let (px, py, pz) = self.coords(i);
            let dx = crate::util::min_image(px as f64 - cx as f64, dims[0]);
            let dy = crate::util::min_image(py as f64 - cy as f64, dims[1]);
            let dz = if self.dim == 3 { crate::util::min_image(pz as f64 - cz_ as f64, dims[2]) } else { 0.0 };
            prof[i] = (-(dx * dx + dy * dy + dz * dz) / (2.0 * sg * sg)).exp();
            tot += prof[i];
        }
        if tot <= 0.0 {
            return;
        }
        let q2 = self.em.q * self.em.q;
        let mut src = vec![0.0f64; n * ng];
        for i in 0..n {
            for c in 0..ng.min(dir.len()) {
                let r = amount as f64 * prof[i] / tot * dir[c];
                self.ym.rho_ext[i * ng + c] += r;
                src[i * ng + c] = q2 * r;
            }
        }
        if self.em.mode != 0 {
            self.ym_add_gradient(&src);
            self.ym_store_gauss_ref();
        }
        if self.em.markers.len() + 3 <= 3 * 4096 {
            self.em.markers.push(cx as f32 + 0.5);
            self.em.markers.push(cy as f32 + 0.5);
            self.em.markers.push(3.0);
        }
    }

    /// Pulso de gluones: U_y(r) ← U_y(r)·exp(i·amplitude·g(x − x₀)·T^component), g gaussiana de
    /// anchura `width`. Polarizado en y, viaja en ±x.
    pub fn inject_gluon_pulse(&mut self, x0: i32, component: u32, amplitude: f32, width: f32) {
        if !(amplitude.is_finite() && width.is_finite()) || width <= 0.0 {
            return;
        }
        if self.ncol <= 1 {
            self.add_photon_pulse(x0 as f32, width, amplitude, 1);
            return;
        }
        let n = self.n;
        let ng = self.ym.ng;
        let comp = (component as usize).min(ng - 1);
        let w = self.w as f64;
        let mut th = [0.0f64; 8];
        for i in 0..n {
            let x = (i % self.w) as f64;
            let dx = crate::util::min_image(x - x0 as f64, w);
            let g = (-(dx * dx) / (2.0 * (width as f64).powi(2))).exp();
            if g < 1e-14 {
                continue;
            }
            th[comp] = -(amplitude as f64) * g;
            let ex = self.ym.alg.exp_minus_i(&th[..ng]);
            let l = n + i;
            let u = self.ym.link(l).mul(&ex);
            self.ym.set_link(l, &u);
        }
        self.ym_store_gauss_ref();
    }

    /// Fija enlaces uniformes U_x = exp(−i·ax·T^gx) y U_y = exp(−i·by·T^gy) (precesión de Wong).
    pub fn set_uniform_colour_links(&mut self, ax: f32, gx: u32, by: f32, gy: u32) {
        if self.ncol <= 1 || !(ax.is_finite() && by.is_finite()) {
            return;
        }
        let n = self.n;
        let ng = self.ym.ng;
        let mut t = [0.0f64; 8];
        t[(gx as usize).min(ng - 1)] = ax as f64;
        let ux = self.ym.alg.exp_minus_i(&t[..ng]);
        let mut t = [0.0f64; 8];
        t[(gy as usize).min(ng - 1)] = by as f64;
        let uy = self.ym.alg.exp_minus_i(&t[..ng]);
        for i in 0..n {
            self.ym.set_link(i, &ux);
            self.ym.set_link(n + i, &uy);
        }
    }

    /// Pone un enlace: U_axis(x, y, z) = exp(−iθ·T^generator). Sirve para montar campos de color
    /// a medida en el laboratorio (por ejemplo, un campo T³ abeliano uniforme en el gauge de Landau).
    #[allow(clippy::too_many_arguments)]
    pub fn set_colour_link(&mut self, x: i32, y: i32, z: i32, axis: u32, generator: u32, theta: f32) {
        if self.ncol <= 1 || axis as usize >= self.dim || !theta.is_finite() {
            return;
        }
        let ng = self.ym.ng;
        let mut t = [0.0f64; 8];
        t[(generator as usize).min(ng - 1)] = theta as f64;
        let m = self.ym.alg.exp_minus_i(&t[..ng]);
        let i = self.wrap_cell(x, y, z);
        self.ym.set_link(axis as usize * self.n + i, &m);
    }

    /// Tubo de flujo de color en la plaqueta (x, y) con holonomía h = exp(iθ·T^generator), con la
    /// cuerda de Dirac hacia y+ llevando h† en los enlaces U_x que cruza (y su compensador).
    pub fn set_colour_flux_tube(&mut self, x: i32, y: i32, theta: f32, generator: u32) {
        if !theta.is_finite() || self.h < 2 {
            return;
        }
        if self.ncol <= 1 {
            self.add_flux_tube(x, y, theta);
            return;
        }
        let ng = self.ym.ng;
        let mut t = [0.0f64; 8];
        t[(generator as usize).min(ng - 1)] = theta as f64; // exp(−iθT) = h†
        let hd = self.ym.alg.exp_minus_i(&t[..ng]);
        let x0 = crate::util::wrap_i(x, self.w);
        let y0 = crate::util::wrap_i(y, self.h);
        for z in 0..self.d {
            for dy in 1..=self.h / 2 {
                let i = self.idx(x0, (y0 + dy) % self.h, z);
                let u = self.ym.link(i).mul(&hd);
                self.ym.set_link(i, &u);
            }
        }
        if self.em.markers.len() + 3 <= 3 * 4096 {
            self.em.markers.push(x0 as f32 + 0.5);
            self.em.markers.push(y0 as f32 + 0.5);
            self.em.markers.push(1.0);
        }
    }

    /// Baño de gluones (§2.11): U = 1, Gauss de la materia y las cargas, y un Ẽ aleatorio
    /// transverso (div Ẽ = 0 con la divergencia hacia atrás) con ⟨Σ_c (Ẽ^c)²⟩ = amplitude² por
    /// enlace. Con U(1), un baño de fotones con A = 0.
    pub fn seed_gluon_bath(&mut self, amplitude: f32, seed: u32) {
        if !amplitude.is_finite() {
            return;
        }
        let n = self.n;
        let dim = self.dim;
        let ng = if self.ncol > 1 { self.ym.ng } else { 1 };
        if self.ncol > 1 {
            self.ym_init_gauss();
        } else {
            self.em_init_gauss();
        }
        let mut rng = crate::rng::SplitMix64::new(seed as u64 ^ 0x6100_B47A);
        let mut fft = FftN::<f64>::new(self.w, self.h, self.d);
        let mut comp: Vec<Vec<Complex<f64>>> = (0..dim).map(|_| vec![Complex::new(0.0, 0.0); n]).collect();
        let mut noise = vec![0.0f64; dim * n * ng];
        for c in 0..ng {
            for a in 0..dim {
                for i in 0..n {
                    comp[a][i] = Complex::new(rng.gauss(), 0.0);
                }
                fft.forward(&mut comp[a]);
            }
            for i in 0..n {
                let (x, y, z) = self.coords(i);
                let ks = [2.0 * PI * x as f64 / self.w as f64, 2.0 * PI * y as f64 / self.h as f64, 2.0 * PI * z as f64 / self.d as f64];
                // d_a = 1 − e^{−ik_a}; transversal: ê ← ê − d*(d·ê)/|d|²
                let mut d = [Complex::new(0.0, 0.0); 3];
                let mut dn = 0.0;
                for a in 0..dim {
                    d[a] = Complex::new(1.0, 0.0) - Complex::from_polar(1.0, -ks[a]);
                    dn += d[a].norm_sqr();
                }
                if dn < 1e-14 {
                    for a in 0..dim {
                        comp[a][i] = Complex::new(0.0, 0.0);
                    }
                    continue;
                }
                let mut dot = Complex::new(0.0, 0.0);
                for a in 0..dim {
                    dot += d[a] * comp[a][i];
                }
                for a in 0..dim {
                    comp[a][i] -= d[a].conj() * dot / dn;
                }
            }
            for a in 0..dim {
                fft.inverse(&mut comp[a]);
                for i in 0..n {
                    noise[(a * n + i) * ng + c] = comp[a][i].re;
                }
            }
        }
        let ms: f64 = noise.iter().map(|v| v * v).sum::<f64>() / (dim * n) as f64;
        let s = if ms > 0.0 { amplitude as f64 / ms.sqrt() } else { 0.0 };
        if self.ncol > 1 {
            for l in 0..dim * n * ng {
                self.ym.e[l] += s * noise[l];
            }
            self.ym_store_gauss_ref();
        } else {
            self.ensure_em_buffers();
            for l in 0..dim * n {
                self.em.e[l] += s * noise[l];
            }
            self.em.ph_valid = false;
        }
    }

    /// Ẽ += ε·F(U)/max|F|: una perturbación que respeta la ley de Gauss exactamente (caos).
    pub fn perturb_gauge_field(&mut self, epsilon: f32) {
        if !epsilon.is_finite() {
            return;
        }
        if self.ncol > 1 {
            self.ym_compute_force();
            let fmax = self.ym.force.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if fmax > 0.0 {
                let s = epsilon as f64 / fmax;
                for l in 0..self.ym.e.len() {
                    self.ym.e[l] += s * self.ym.force[l];
                }
            }
        } else {
            self.ensure_em_buffers();
            let n = self.n;
            let mut f = vec![0.0f64; self.dim * n];
            for a in 0..self.dim {
                for i in 0..n {
                    let mut s = 0.0;
                    for b in 0..self.dim {
                        if b != a {
                            s += self.plaquette_b(i, a, b) - self.plaquette_b(self.nb(i, b, -1), a, b);
                        }
                    }
                    f[a * n + i] = s;
                }
            }
            let fmax = f.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if fmax > 0.0 {
                for l in 0..f.len() {
                    self.em.e[l] += epsilon as f64 * f[l] / fmax;
                }
            }
        }
    }

    /// ⟨|Tr P − Tr P′|⟩/N sobre todas las plaquetas: distancia invariante gauge entre dos campos
    /// (Müller y Trayanov). Exige la misma malla y el mismo grupo. Con U(1), con e^{iB}.
    pub fn plaquette_distance(&self, other: &QuantumSpacetimeUniverse) -> f64 {
        if (other.w, other.h, other.d, other.ncol) != (self.w, self.h, self.d, self.ncol) {
            return f64::NAN;
        }
        let mut s = 0.0;
        let mut cnt = 0usize;
        for i in 0..self.n {
            for a in 0..self.dim {
                for b in (a + 1)..self.dim {
                    let d = if self.ncol > 1 {
                        (self.ym_plaq_p(i, a, b).trace() - other.ym_plaq_p(i, a, b).trace()).norm() / self.ym.n as f64
                    } else {
                        (C::from_polar(1.0, self.plaquette_b(i, a, b)) - C::from_polar(1.0, other.plaquette_b(i, a, b))).norm()
                    };
                    s += d;
                    cnt += 1;
                }
            }
        }
        s / cnt.max(1) as f64
    }

    /// Copia el campo gauge (U(1) o SU(N)) y la materia de otro universo con la misma malla.
    pub fn copy_gauge_from(&mut self, other: &QuantumSpacetimeUniverse) {
        if (other.w, other.h, other.d, other.ncol) != (self.w, self.h, self.d, self.ncol)
            || other.buffer_read.len() != self.buffer_read.len()
        {
            return;
        }
        self.copy_em_from(other);
        if self.ncol > 1 && other.ym.u.len() == self.ym.u.len() {
            self.ym.u.copy_from_slice(&other.ym.u);
            self.ym.e.copy_from_slice(&other.ym.e);
            self.ym.rho_ext.copy_from_slice(&other.ym.rho_ext);
            self.ym.g_ref.copy_from_slice(&other.ym.g_ref);
        }
        self.buffer_read.copy_from_slice(&other.buffer_read);
    }

    /// n × (N² − 1) floats: carga de color de cada solitón por unidad de norma (depende del gauge).
    pub fn soliton_colour_ptr(&self) -> *const f32 {
        self.ym.sol_colour.as_ptr()
    }

    /// Número de f32 en `soliton_colour_ptr`.
    pub fn soliton_colour_len(&self) -> u32 {
        self.ym.sol_colour.len() as u32
    }

    /// Energía eléctrica ½|Ẽ|² y magnética 2κ(N − Re Tr P) por celda (modo 4 y laboratorio).
    pub fn compute_colour_field_densities(&mut self) {
        if self.ncol <= 1 {
            return;
        }
        let n = self.n;
        let ng = self.ym.ng;
        for i in 0..n {
            let mut el = 0.0;
            let mut mg = 0.0;
            for a in 0..self.dim {
                for c in 0..ng {
                    el += 0.5 * self.ym.e[(a * n + i) * ng + c].powi(2);
                }
                for b in (a + 1)..self.dim {
                    mg += 2.0 * self.em.kappa * (self.ym.n as f64 - self.ym_plaq_p(i, a, b).trace().re);
                }
            }
            self.ym.elec_density[i] = el as f32;
            self.ym.mag_density[i] = mg as f32;
            self.ym.energy_density[i] = (el + mg) as f32;
        }
    }

    pub fn colour_electric_density_ptr(&self) -> *const f32 {
        self.ym.elec_density.as_ptr()
    }

    /// Número de f32 en `colour_electric_density_ptr`.
    pub fn colour_electric_density_len(&self) -> u32 {
        self.ym.elec_density.len() as u32
    }

    /// Ẽ^c_a en un enlace.
    pub fn colour_link_e(&self, x: i32, y: i32, z: i32, axis: u32, component: u32) -> f64 {
        if self.ncol <= 1 {
            return self.link_e(x, y, z, axis);
        }
        let a = (axis as usize).min(self.dim - 1);
        let c = (component as usize).min(self.ym.ng - 1);
        self.ym.e[(a * self.n + self.wrap_cell(x, y, z)) * self.ym.ng + c]
    }

    /// Norma del vector de color proyectada sobre un color de la base (fracción de la norma).
    pub fn colour_fraction(&self, colour: u32) -> f64 {
        if self.ncol <= 1 {
            return 1.0;
        }
        let nco = self.ncol;
        let c = (colour as usize).min(nco - 1);
        let stride = 2 * self.nc * nco;
        let (mut s, mut tot) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            for k in 0..self.nc {
                for a in 0..nco {
                    let o = i * stride + 2 * (k * nco + a);
                    let e = (self.buffer_read[o] as f64).powi(2) + (self.buffer_read[o + 1] as f64).powi(2);
                    tot += e;
                    if a == c {
                        s += e;
                    }
                }
            }
        }
        if tot > 0.0 {
            s / tot
        } else {
            0.0
        }
    }

    /// Centroide (eje 0 o 1) de la densidad del color `colour` con media circular (Wong).
    pub fn colour_centroid(&self, axis: u32, colour: u32) -> f64 {
        let nco = self.ncol.max(1);
        let c = (colour as usize).min(nco - 1);
        let stride = 2 * self.nc * nco;
        let a = (axis as usize).min(self.dim - 1);
        let l = [self.w, self.h, self.d][a] as f64;
        let (mut sc, mut ss) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let mut e = 0.0;
            for k in 0..self.nc {
                let o = i * stride + 2 * (k * nco + c);
                e += (self.buffer_read[o] as f64).powi(2) + (self.buffer_read[o + 1] as f64).powi(2);
            }
            let th = 2.0 * PI * [x, y, z][a] as f64 / l;
            sc += e * th.cos();
            ss += e * th.sin();
        }
        (ss.atan2(sc) / (2.0 * PI) * l).rem_euclid(l)
    }
}

impl QuantumSpacetimeUniverse {
    /// Ω(r) = exp(i·amplitude·ξ^c·T^c): ψ → Ω·ψ, U_a → Ω(r+â)·U_a·Ω(r)†, Ẽ_a → Ad(Ω(r))·Ẽ_a y
    /// G_ref → Ad(Ω)·G_ref. Las densidades, |Ẽ|², Re Tr P y el residuo de Gauss no cambian.
    pub(crate) fn ym_random_gauge(&mut self, amplitude: f32, seed: u32) {
        if !amplitude.is_finite() || self.ncol <= 1 {
            return;
        }
        let n = self.n;
        let ng = self.ym.ng;
        let nco = self.ncol;
        let mut rng = crate::rng::SplitMix64::new(seed as u64 ^ 0x5u64 << 40);
        let omegas: Vec<Mat> = (0..n)
            .map(|_| {
                let mut th = [0.0f64; 8];
                for v in th.iter_mut().take(ng) {
                    *v = -(amplitude as f64) * rng.gauss();
                }
                self.ym.alg.exp_minus_i(&th[..ng])
            })
            .collect();
        // materia (con el motor de espinores, su estado en f64 y sus partes)
        if self.sw.active {
            let ns = self.sw.ns;
            let st = 2 * ns * nco;
            let mut bufs = vec![std::mem::take(&mut self.sw.psi)];
            bufs.extend(std::mem::take(&mut self.sw.parts));
            for b in bufs.iter_mut() {
                for i in 0..n {
                    for k in 0..ns {
                        let o = i * st + 2 * k * nco;
                        let mut v = [cz(); 3];
                        for a in 0..nco {
                            v[a] = Complex::new(b[o + 2 * a], b[o + 2 * a + 1]);
                        }
                        let r = omegas[i].apply(&v);
                        for a in 0..nco {
                            b[o + 2 * a] = r[a].re;
                            b[o + 2 * a + 1] = r[a].im;
                        }
                    }
                }
            }
            self.sw.psi = bufs.remove(0);
            self.sw.parts = bufs;
            self.sw_mirror();
        } else {
            let stride = self.stride();
            for i in 0..n {
                for k in 0..self.nc {
                    let v = self.colour_vec(i, k);
                    let r = omegas[i].apply(&v);
                    let b = i * stride + 2 * k * nco;
                    for a in 0..nco {
                        self.buffer_read[b + 2 * a] = r[a].re as f32;
                        self.buffer_read[b + 2 * a + 1] = r[a].im as f32;
                    }
                }
            }
        }
        let mut tmp = [0.0f64; 8];
        for a in 0..self.dim {
            for i in 0..n {
                let ip = self.nb(i, a, 1);
                let l = a * n + i;
                let u = omegas[ip].mul(&self.ym.link(l)).mul_dag(&omegas[i]);
                self.ym.set_link(l, &u);
                let x = self.ym.alg.combine(&self.ym.e[l * ng..(l + 1) * ng]);
                let ad = omegas[i].mul(&x).mul_dag(&omegas[i]);
                self.ym.alg.components(&ad, &mut tmp);
                self.ym.e[l * ng..(l + 1) * ng].copy_from_slice(&tmp[..ng]);
            }
        }
        for i in 0..n {
            let x = self.ym.alg.combine(&self.ym.g_ref[i * ng..(i + 1) * ng]);
            let ad = omegas[i].mul(&x).mul_dag(&omegas[i]);
            self.ym.alg.components(&ad, &mut tmp);
            self.ym.g_ref[i * ng..(i + 1) * ng].copy_from_slice(&tmp[..ng]);
            let x = self.ym.alg.combine(&self.ym.rho_ext[i * ng..(i + 1) * ng]);
            let ad = omegas[i].mul(&x).mul_dag(&omegas[i]);
            self.ym.alg.components(&ad, &mut tmp);
            self.ym.rho_ext[i * ng..(i + 1) * ng].copy_from_slice(&tmp[..ng]);
        }
    }

    pub(crate) fn add_dirac_colour_packet(&mut self, x0: f64, y0: f64, sigma: f64, kx: f64, ky: f64, col: [C; 3]) {
        if !(x0.is_finite()
            && y0.is_finite()
            && sigma.is_finite()
            && kx.is_finite()
            && ky.is_finite()
            && col.iter().all(|c| c.re.is_finite() && c.im.is_finite()))
            || sigma <= 0.0
        {
            return;
        }
        let nco = self.ncol.max(1);
        let stride = 4 * nco;
        if self.buffer_read.len() < self.n * stride {
            return;
        }
        let spinor = crate::dirac2d::su2_band(&crate::dirac2d::dirac2d_matrix(self.mass as f64, kx, ky), 1).1;
        let cn: f64 = col.iter().take(nco).map(|v| v.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
        let (w, h) = (self.w as f64, self.h as f64);
        let rc = 4.0 * sigma;
        let mut norm = 0.0;
        for i in 0..self.n {
            let (x, y, _) = self.coords(i);
            let dx = crate::util::min_image(x as f64 - x0, w);
            let dy = crate::util::min_image(y as f64 - y0, h);
            if dx.abs() > rc || dy.abs() > rc {
                continue;
            }
            norm += (-(dx * dx + dy * dy) / (sigma * sigma)).exp();
        }
        if norm <= 0.0 {
            return;
        }
        let amp = 1.0 / norm.sqrt();
        for i in 0..self.n {
            let (x, y, _) = self.coords(i);
            let dx = crate::util::min_image(x as f64 - x0, w);
            let dy = crate::util::min_image(y as f64 - y0, h);
            if dx.abs() > rc || dy.abs() > rc {
                continue;
            }
            let e = amp * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            let ph = C::from_polar(e, kx * dx + ky * dy);
            for comp in 0..2 {
                for a in 0..nco {
                    let v = ph * spinor[comp] * col[a] / cn;
                    let o = i * stride + 2 * (comp * nco + a);
                    self.buffer_read[o] += v.re as f32;
                    self.buffer_read[o + 1] += v.im as f32;
                }
            }
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Puntero al campo eléctrico de los enlaces (f64): Ẽ de SU(N) (dim·N_celdas·(N² − 1)) o E de U(1).
    pub fn gauge_e_ptr(&self) -> *const f64 {
        if self.ncol > 1 {
            self.ym.e.as_ptr()
        } else {
            self.em.e.as_ptr()
        }
    }

    /// Número de f64 en `gauge_e_ptr`.
    pub fn gauge_e_len(&self) -> u32 {
        if self.ncol > 1 {
            self.ym.e.len() as u32
        } else {
            self.em.e.len() as u32
        }
    }

    /// Flujo de T^c por plaqueta (a, b) en una celda: 2·Im Tr(T^c·P_ab) (SU(N)).
    pub fn colour_plaquette_flux(&self, x: i32, y: i32, z: i32, component: u32) -> f64 {
        if self.ncol <= 1 {
            return self.plaquette_field(x, y, z, 0, 1);
        }
        let i = self.wrap_cell(x, y, z);
        let p = self.ym_plaq_p(i, 0, 1);
        let c = (component as usize).min(self.ym.ng - 1);
        2.0 * self.ym.alg.t[c].mul(&p).trace().im
    }
}
