//! Una generación (o tres) del modelo estándar, clásica (§2.21 y §2.22): el Dirac 3D de §2.16 en la
//! base de quiralidad, donde cada salto T_a = exp(−iα_a·k_a) actúa como exp(−iσ_a·k_a) sobre los
//! espinores dextrógiros y como exp(+iσ_a·k_a) sobre los levógiros. Los saltos nunca mezclan L y R:
//! cada quiralidad lleva su propia representación gauge.
//! - Campos de Weyl por generación: L = (ν_L, e_L) (doblete, Y = −½), e_R (Y = −1), Q_L = (u_L, d_L)
//!   (doblete × 3 colores, Y = 1/6), u_R (Y = 2/3) y d_R (Y = −1/3): 30 amplitudes por celda.
//! - El salto hacia +a lleva U₂ (dobletes), U₃ (color) y e^{ig′Y·A}; hacia −a, sus adjuntos. Las
//!   corrientes de cada eje se toman antes de su salto y empujan Ẽ₂ (−g²J), Ẽ₃ (−g_s²J) y E_Y (−J).
//! - Las masas salen del doblete de Higgs Φ de §2.12 (Y = ½): con Φ fijo, el término de Yukawa es
//!   una rotación local de ángulo y·|Φ| entre el singlete y la componente del doblete a lo largo de
//!   Φ (Φ̃ = iσ₂Φ* para el u). El mismo paso empuja al Higgs, Π ← Π − ∫∂E_Y/∂Φ* dt, integrado de
//!   forma exacta: como E_Y es invariante gauge, la carga total se conserva celda a celda y la ley de
//!   Gauss sigue exacta. Con tres generaciones, la SVD Y = W·diag(y)·V† separa el flujo en tres.
//! Es clásico: un campo por especie, sin estadística de Fermi ni mar de Dirac, y los dobladores de la
//! caminata (los nodos de las dos quiralidades) también ven a los W.

use crate::linalg::eig;
use crate::rng::SplitMix64;
use crate::spinor::{cx, pauli_eigenbasis};
use crate::sun::{Algebra, Mat, C};
use crate::util::min_image;
use crate::yangmills::YmField;
use crate::QuantumSpacetimeUniverse;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Amplitudes complejas por generación.
pub const NGF: usize = 30;

#[derive(Clone, Copy)]
pub struct FieldDef {
    pub off: usize,
    pub ni: usize,
    pub nc: usize,
    pub y: f64,
    pub right: bool,
}

/// L, e_R, Q_L, u_R, d_R (índice: off + s·(ni·nc) + i·nc + c).
pub const FIELDS: [FieldDef; 5] = [
    FieldDef { off: 0, ni: 2, nc: 1, y: -0.5, right: false },
    FieldDef { off: 4, ni: 1, nc: 1, y: -1.0, right: true },
    FieldDef { off: 6, ni: 2, nc: 3, y: 1.0 / 6.0, right: false },
    FieldDef { off: 18, ni: 1, nc: 3, y: 2.0 / 3.0, right: true },
    FieldDef { off: 24, ni: 1, nc: 3, y: -1.0 / 3.0, right: true },
];

/// Especies de `inject_sm_packet`: (campo, componente de isospín).
pub const SPECIES: [(usize, usize); 7] = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (3, 0), (4, 0)];

type M2 = [[C; 2]; 2];

pub struct Sm {
    pub active: bool,
    pub ng: usize,
    pub quarks: bool,
    pub colour_dynamic: bool,
    pub bosons_dynamic: bool,
    pub reaction: bool,
    pub g: f64,
    pub gp: f64,
    pub gs: f64,
    /// Por sector (e, d, u): valores singulares, W y V (ng × ng, fila mayor) y la matriz Y.
    pub ys: [Vec<f64>; 3],
    pub wm: [Vec<C>; 3],
    pub vm: [Vec<C>; 3],
    pub ymat: [Vec<C>; 3],
    pub psi: Vec<f64>,
    pub tmp: Vec<f64>,
    pub ym3: YmField,
    pub g3_ref: Vec<f64>,
    pub seq: [Vec<usize>; 2],
    pub mat_in: [Vec<M2>; 2],
    pub mat_out: [M2; 2],
    pub ckm: Vec<f64>,
    pub theta: Vec<f64>,
    pub jy: Vec<f64>,
    pub j2: Vec<f64>,
    pub j3: Vec<f64>,
    pub zcell: Vec<C>,
    /// Campo externo opcional de bosón Z' con cambio de sabor b_L ↔ s_L (Fase 14, BSM).
    pub zp_active: bool,
    pub zp_axis: usize,
    pub zp_u_d: [C; 9],
    pub zp_u_l: [C; 9],
}

/// Enlace de un salto: U₂, U₃ (con color dinámico) y las fases de U(1)_Y de los cinco campos.
struct HopLink {
    u2: Mat,
    u3: Option<Mat>,
    ph: [C; 5],
}

impl Sm {
    pub fn new() -> Self {
        Sm {
            active: false,
            ng: 1,
            quarks: true,
            colour_dynamic: true,
            bosons_dynamic: true,
            reaction: true,
            g: 0.6,
            gp: 0.4,
            gs: 0.8,
            ys: [vec![0.4], vec![0.3], vec![0.5]],
            wm: [vec![cx(1.0, 0.0)], vec![cx(1.0, 0.0)], vec![cx(1.0, 0.0)]],
            vm: [vec![cx(1.0, 0.0)], vec![cx(1.0, 0.0)], vec![cx(1.0, 0.0)]],
            ymat: [vec![cx(0.4, 0.0)], vec![cx(0.3, 0.0)], vec![cx(0.5, 0.0)]],
            psi: Vec::new(),
            tmp: Vec::new(),
            ym3: YmField::new(3, 3, 0),
            g3_ref: Vec::new(),
            seq: [vec![0, 1, 2], vec![2, 1, 0]],
            mat_in: [Vec::new(), Vec::new()],
            mat_out: [[[cx(1.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(1.0, 0.0)]]; 2],
            ckm: vec![0.0; 10],
            theta: Vec::new(),
            jy: Vec::new(),
            j2: Vec::new(),
            j3: Vec::new(),
            zcell: Vec::new(),
            zp_active: false,
            zp_axis: 2,
            zp_u_d: [cx(0.0, 0.0); 9],
            zp_u_l: [cx(0.0, 0.0); 9],
        }
    }

    fn rebuild(&mut self) {
        let dag = |m: &M2| -> M2 { [[m[0][0].conj(), m[1][0].conj()], [m[0][1].conj(), m[1][1].conj()]] };
        let mul = |a: &M2, b: &M2| -> M2 {
            let mut r = [[cx(0.0, 0.0); 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
                }
            }
            r
        };
        for p in 0..2 {
            let mut prev: M2 = [[cx(1.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(1.0, 0.0)]];
            let mut mats = Vec::new();
            for &a in &self.seq[p] {
                let v = pauli_eigenbasis(a);
                mats.push(mul(&dag(&v), &prev));
                prev = v;
            }
            self.mat_in[p] = mats;
            self.mat_out[p] = prev;
        }
    }
}

/// SVD de una matriz compleja n × n (fila mayor): Y = W·diag(y)·V†, con y creciente. W sale de los
/// autovectores de Y·Y† y V = Y†·W·diag(1/y).
pub fn svd(y: &[C], n: usize) -> (Vec<f64>, Vec<C>, Vec<C>) {
    let mut a = vec![cx(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = cx(0.0, 0.0);
            for k in 0..n {
                s += y[i * n + k] * y[j * n + k].conj();
            }
            a[i * n + j] = s;
        }
    }
    let e = eig(a, n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| e.values[i].re.partial_cmp(&e.values[j].re).unwrap());
    let sv: Vec<f64> = idx.iter().map(|&i| e.values[i].re.max(0.0).sqrt()).collect();
    let mut w = vec![cx(0.0, 0.0); n * n];
    for (col, &i) in idx.iter().enumerate() {
        // Gram–Schmidt por si hay autovalores casi degenerados
        let mut v = e.vectors[i].clone();
        for prev in 0..col {
            let ov: C = (0..n).map(|r| w[r * n + prev].conj() * v[r]).sum();
            for r in 0..n {
                v[r] -= ov * w[r * n + prev];
            }
        }
        let nn: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
        for r in 0..n {
            w[r * n + col] = v[r] / nn;
        }
    }
    let mut v = vec![cx(0.0, 0.0); n * n];
    for k in 0..n {
        for r in 0..n {
            let mut s = cx(0.0, 0.0);
            for i in 0..n {
                s += y[i * n + r].conj() * w[i * n + k];
            }
            v[r * n + k] = if sv[k] > 1e-14 { s / sv[k] } else { cx(if r == k { 1.0 } else { 0.0 }, 0.0) };
        }
    }
    (sv, w, v)
}

/// Integrales de la rotación de Yukawa en [0, 1]: (sin μ/μ, (1 − cos μ)/μ, ∫cos², ∫sin², ∫sin·cos).
fn rot_integrals(mu: f64) -> (f64, f64, f64, f64, f64) {
    if mu.abs() < 1e-8 {
        return (1.0, 0.5 * mu, 1.0, 0.0, 0.5 * mu);
    }
    let (s, c) = mu.sin_cos();
    let s2 = (2.0 * mu).sin() / (4.0 * mu);
    (s / mu, (1.0 - c) / mu, 0.5 + s2, 0.5 - s2, s * s / (2.0 * mu))
}

/// Rotación de Yukawa de un par (D doblete, S singlete) con Φ efectivo (Φ o Φ̃) fijo y el empuje
/// sobre Π: devuelve (D, S) nuevos y suma −y·∫∂E/∂Φ* a `dpi`.
#[allow(clippy::too_many_arguments)]
fn yukawa_pair(d: [C; 2], s0: C, y: f64, phi: [C; 2], tilde: bool, dpi: &mut [C; 2], push: bool) -> ([C; 2], C) {
    let phie = if tilde { [phi[1].conj(), -phi[0].conj()] } else { phi };
    let nrm = (phie[0].norm_sqr() + phie[1].norm_sqr()).sqrt();
    if nrm < 1e-300 || y == 0.0 {
        return (d, s0);
    }
    let n = [phie[0] / nrm, phie[1] / nrm];
    let chi0 = n[0].conj() * d[0] + n[1].conj() * d[1];
    let dperp = [d[0] - n[0] * chi0, d[1] - n[1] * chi0];
    let mu = y * nrm;
    let (sn, cs) = mu.sin_cos();
    let i = cx(0.0, 1.0);
    let chi1 = chi0 * cs - i * sn * s0;
    let s1 = s0 * cs - i * sn * chi0;
    let dn = [dperp[0] + n[0] * chi1, dperp[1] + n[1] * chi1];
    if push {
        let (f1, f2, ic2, is2, ics) = rot_integrals(mu);
        let (a2, b2) = (chi0.norm_sqr(), s0.norm_sqr());
        if !tilde {
            // X_i = ∫ S*·D_i;  ΔΠ_i = −y·X_i
            let ints = s0.conj() * f1 + i * chi0.conj() * f2;
            let intx = s0.conj() * chi0 * ic2 + chi0.conj() * s0 * is2 + i * (a2 - b2) * ics;
            for k in 0..2 {
                dpi[k] -= (dperp[k] * ints + n[k] * intx) * y;
            }
        } else {
            // Y_i = ∫ D_i*·S;  ΔΠ_1 = +y·Y_2,  ΔΠ_2 = −y·Y_1
            let ints = s0 * f1 - i * chi0 * f2;
            let intx = chi0.conj() * s0 * ic2 + s0.conj() * chi0 * is2 - i * (a2 - b2) * ics;
            let yy = [dperp[0].conj() * ints + n[0].conj() * intx, dperp[1].conj() * ints + n[1].conj() * intx];
            dpi[0] += yy[1] * y;
            dpi[1] -= yy[0] * y;
        }
    }
    (dn, s1)
}

impl QuantumSpacetimeUniverse {
    #[inline(always)]
    fn sm_nt(&self) -> usize {
        NGF * self.sm.ng
    }

    /// Paso de Yukawa de una celda (amplitudes `z`, NGF·ng complejos) con Φ fijo; empuja `dpi`.
    pub(crate) fn sm_yukawa_cell(&self, z: &mut [C], phi: [C; 2], dpi: &mut [C; 2], push: bool) {
        let ng = self.sm.ng;
        // (sector, campo doblete, campo singlete, tilde)
        let sectors: &[(usize, usize, usize, bool)] = if self.sm.quarks { &[(0, 0, 1, false), (1, 2, 4, false), (2, 2, 3, true)] } else { &[(0, 0, 1, false)] };
        for &(sec, fd, fs, tilde) in sectors {
            let (dd, ss) = (FIELDS[fd], FIELDS[fs]);
            let ncol = dd.nc;
            let (w, v, ys) = (&self.sm.wm[sec], &self.sm.vm[sec], &self.sm.ys[sec]);
            for s in 0..2 {
                for c in 0..ncol {
                    // D_g (2) y S_g por generación, rotados a la base de masas
                    let mut dg = [[cx(0.0, 0.0); 2]; 3];
                    let mut sg = [cx(0.0, 0.0); 3];
                    for g in 0..ng {
                        for iso in 0..2 {
                            dg[g][iso] = z[g * NGF + dd.off + s * (2 * ncol) + iso * ncol + c];
                        }
                        sg[g] = z[g * NGF + ss.off + s * ss.nc + c];
                    }
                    let mut dp = [[cx(0.0, 0.0); 2]; 3];
                    let mut sp = [cx(0.0, 0.0); 3];
                    for k in 0..ng {
                        for g in 0..ng {
                            for iso in 0..2 {
                                dp[k][iso] += w[g * ng + k].conj() * dg[g][iso];
                            }
                            sp[k] += v[g * ng + k].conj() * sg[g];
                        }
                    }
                    for k in 0..ng {
                        let (dn, sn) = yukawa_pair(dp[k], sp[k], ys[k], phi, tilde, dpi, push);
                        dp[k] = dn;
                        sp[k] = sn;
                    }
                    for g in 0..ng {
                        let mut dsum = [cx(0.0, 0.0); 2];
                        let mut ssum = cx(0.0, 0.0);
                        for k in 0..ng {
                            for iso in 0..2 {
                                dsum[iso] += w[g * ng + k] * dp[k][iso];
                            }
                            ssum += v[g * ng + k] * sp[k];
                        }
                        for iso in 0..2 {
                            z[g * NGF + dd.off + s * (2 * ncol) + iso * ncol + c] = dsum[iso];
                        }
                        z[g * NGF + ss.off + s * ss.nc + c] = ssum;
                    }
                }
            }
        }
    }

    fn sm_apply_spin(&self, psi: &mut [f64], m: &M2) {
        let nt = self.sm_nt();
        for cell in 0..self.n {
            for g in 0..self.sm.ng {
                for f in FIELDS.iter() {
                    let blk = f.ni * f.nc;
                    let base = cell * nt + g * NGF + f.off;
                    for j in 0..blk {
                        let (p0, p1) = (2 * (base + j), 2 * (base + blk + j));
                        let a = cx(psi[p0], psi[p0 + 1]);
                        let b = cx(psi[p1], psi[p1 + 1]);
                        let na = m[0][0] * a + m[0][1] * b;
                        let nb = m[1][0] * a + m[1][1] * b;
                        psi[p0] = na.re;
                        psi[p0 + 1] = na.im;
                        psi[p1] = nb.re;
                        psi[p1 + 1] = nb.im;
                    }
                }
            }
        }
    }

    /// Dirección del salto de la componente s′ (base propia de σ_a) de un campo: R, +1 avanza;
    /// L, el salto es exp(+iσ_a·k_a), así que +1 retrocede.
    #[inline(always)]
    fn sm_dir(f: &FieldDef, s: usize) -> i32 {
        let base = if s == 0 { 1 } else { -1 };
        if f.right {
            base
        } else {
            -base
        }
    }

    /// Enlace de un salto con sus fases: U₂, U₃ y e^{±iY·θ} de las cinco hipercargas.
    #[inline(always)]
    fn sm_hop_link(&self, l: usize, theta: f64, sign: f64, colour: bool) -> HopLink {
        let mut ph = [cx(1.0, 0.0); 5];
        for (k, f) in FIELDS.iter().enumerate() {
            ph[k] = C::from_polar(1.0, sign * f.y * theta);
        }
        HopLink { u2: self.ym.link(l), u3: if colour { Some(self.sm.ym3.link(l)) } else { None }, ph }
    }

    /// Salto de un eje con las corrientes, enlace por enlace. Para la celda r: lo que avanza llega de
    /// r − â con U(r − â) y lo que retrocede, de r + â con U(r)†. Las corrientes del enlace (a, r) se
    /// toman antes del salto: +lo que sale de r hacia +a (medido en r) y −lo que llega de r + â
    /// (transportado al marco de r), con q_f = g′·Y_f para E_Y.
    #[allow(clippy::too_many_arguments)]
    fn sm_hop_axis(&self, src: &[f64], dst: &mut [f64], a: usize, theta: &[f64], currents: bool, jy: &mut [f64], j2: &mut [f64], j3: &mut [f64]) {
        let n = self.n;
        let nt = self.sm_nt();
        let colour = self.sm.quarks && self.sm.colour_dynamic;
        let cur3 = currents && colour;
        let nfields = if self.sm.quarks { 5 } else { 2 };
        let alg2 = &self.ym.alg;
        let alg3 = &self.sm.ym3.alg;
        let gp = self.sm.gp;
        let mut t = [0.0f64; 8];
        // ρ₃ = Σ ±ψψ† del enlace: J₃^c = Tr(T^c·ρ₃) se proyecta una sola vez por enlace
        let mut rho3 = Mat::zero(3);
        let load = |cell: usize, base: usize, blk: usize| -> [C; 6] {
            let mut v = [cx(0.0, 0.0); 6];
            let o = 2 * (cell * nt + base);
            for j in 0..blk {
                v[j] = cx(src[o + 2 * j], src[o + 2 * j + 1]);
            }
            v
        };
        for i in 0..n {
            let im = self.nb(i, a, -1);
            let ip = self.nb(i, a, 1);
            let (lm, l) = (a * n + im, a * n + i);
            let fwd = self.sm_hop_link(lm, theta[lm], 1.0, colour);
            let bwd = self.sm_hop_link(l, theta[l], -1.0, colour);
            if cur3 {
                rho3.a = [cx(0.0, 0.0); 9];
            }
            for g in 0..self.sm.ng {
                for (fi, f) in FIELDS.iter().enumerate().take(nfields) {
                    let blk = f.ni * f.nc;
                    for s in 0..2 {
                        let dir = Self::sm_dir(f, s);
                        let base = g * NGF + f.off + s * blk;
                        let (mut v, k, dag) = if dir > 0 { (load(im, base, blk), &fwd, false) } else { (load(ip, base, blk), &bwd, true) };
                        if f.ni == 2 {
                            for c in 0..f.nc {
                                let w = [v[c], v[f.nc + c], cx(0.0, 0.0)];
                                let r = if dag { k.u2.apply_dag(&w) } else { k.u2.apply(&w) };
                                v[c] = r[0];
                                v[f.nc + c] = r[1];
                            }
                        }
                        if let (3, Some(u3)) = (f.nc, &k.u3) {
                            for iso in 0..f.ni {
                                let w = [v[iso * 3], v[iso * 3 + 1], v[iso * 3 + 2]];
                                let r = if dag { u3.apply_dag(&w) } else { u3.apply(&w) };
                                v[iso * 3..iso * 3 + 3].copy_from_slice(&r);
                            }
                        }
                        let ph = k.ph[fi];
                        let o = 2 * (i * nt + base);
                        for j in 0..blk {
                            let z = v[j] * ph;
                            dst[o + 2 * j] = z.re;
                            dst[o + 2 * j + 1] = z.im;
                        }
                        if currents {
                            // lo que sale de r hacia +a, sin transportar; lo que llega de r + â, ya en el marco de r
                            let (c, sg) = if dir > 0 { (load(i, base, blk), 1.0) } else { (v, -1.0) };
                            let nrm: f64 = c[..blk].iter().map(|z| z.norm_sqr()).sum();
                            jy[l] += sg * gp * f.y * nrm;
                            if f.ni == 2 {
                                for cc in 0..f.nc {
                                    alg2.charge(&[c[cc], c[f.nc + cc], cx(0.0, 0.0)], &mut t);
                                    for q in 0..3 {
                                        j2[l * 3 + q] += sg * t[q];
                                    }
                                }
                            }
                            if cur3 && f.nc == 3 {
                                for iso in 0..f.ni {
                                    let w = &c[iso * 3..iso * 3 + 3];
                                    for r in 0..3 {
                                        let wr = w[r] * sg;
                                        for q in 0..3 {
                                            rho3.a[r * 3 + q] += wr * w[q].conj();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if cur3 {
                for q in 0..8 {
                    j3[l * 8 + q] += alg3.tr_t(q, &rho3).re;
                }
            }
        }
    }

    /// Un paso del modelo estándar: saltos con corrientes, Yukawa con el empuje sobre Π y, si los
    /// bosones son dinámicos, los subpasos del Higgs con SU(2)×U(1)_Y y los de SU(3).
    pub(crate) fn sm_step(&mut self) {
        let parity = (self.t & 1) as usize;
        let n = self.n;
        let dim = self.dim;
        let mut psi = std::mem::take(&mut self.sm.psi);
        let mut tmp = std::mem::take(&mut self.sm.tmp);
        // búferes de trabajo reservados de antemano (θ = g′·A, corrientes y una celda)
        let mut theta = std::mem::take(&mut self.sm.theta);
        let mut jy = std::mem::take(&mut self.sm.jy);
        let mut j2 = std::mem::take(&mut self.sm.j2);
        let mut j3 = std::mem::take(&mut self.sm.j3);
        let mut z = std::mem::take(&mut self.sm.zcell);
        let nt = self.sm_nt();
        theta.resize(dim * n, 0.0);
        for (t, a) in theta.iter_mut().zip(&self.em.a) {
            *t = self.sm.gp * a;
        }
        let dynamic = self.sm.bosons_dynamic;
        let colour = self.sm.quarks && self.sm.colour_dynamic;
        jy.clear();
        j2.clear();
        j3.clear();
        if dynamic {
            jy.resize(dim * n, 0.0);
            j2.resize(3 * dim * n, 0.0);
            if colour {
                j3.resize(8 * dim * n, 0.0);
            }
        }
        z.resize(nt, cx(0.0, 0.0));
        for s in 0..self.sm.seq[parity].len() {
            let a = self.sm.seq[parity][s];
            let m = self.sm.mat_in[parity][s];
            self.sm_apply_spin(&mut psi, &m);
            self.sm_hop_axis(&psi, &mut tmp, a, &theta, dynamic, &mut jy, &mut j2, &mut j3);
            if self.sm.zp_active && a == self.sm.zp_axis && self.sm.ng == 3 {
                self.sm_apply_zprime_hop(&mut tmp);
            }
            std::mem::swap(&mut psi, &mut tmp);
        }
        let mo = self.sm.mat_out[parity];
        self.sm_apply_spin(&mut psi, &mo);
        // Yukawa con Φ(t) fijo y el empuje exacto sobre Π
        let push = self.sm.reaction;
        for i in 0..n {
            for k in 0..nt {
                z[k] = cx(psi[2 * (i * nt + k)], psi[2 * (i * nt + k) + 1]);
            }
            let phi = [cx(self.higgs.phi[4 * i], self.higgs.phi[4 * i + 1]), cx(self.higgs.phi[4 * i + 2], self.higgs.phi[4 * i + 3])];
            let mut dpi = [cx(0.0, 0.0); 2];
            self.sm_yukawa_cell(&mut z, phi, &mut dpi, push);
            for k in 0..nt {
                psi[2 * (i * nt + k)] = z[k].re;
                psi[2 * (i * nt + k) + 1] = z[k].im;
            }
            if push && dynamic {
                for k in 0..2 {
                    self.higgs.pi[4 * i + 2 * k] += dpi[k].re;
                    self.higgs.pi[4 * i + 2 * k + 1] += dpi[k].im;
                }
            }
        }
        self.sm.psi = psi;
        self.sm.tmp = tmp;
        self.sm.zcell = z;
        self.sm.theta = theta;
        if dynamic {
            let g2 = self.sm.g * self.sm.g;
            for l in 0..dim * n {
                self.em.e[l] -= jy[l];
                for c in 0..3 {
                    self.ym.e[l * 3 + c] -= g2 * j2[l * 3 + c];
                }
            }
            let s = self.em.substeps.max(1);
            let dt = 1.0 / s as f64;
            for _ in 0..s {
                self.higgs_doublet_substep(dt);
            }
            self.ym.reunit_counter += 1;
            if self.ym.reunit_counter % 100 == 0 {
                for l in 0..dim * n {
                    let mut u = self.ym.link(l);
                    u.reunitarize();
                    self.ym.set_link(l, &u);
                }
            }
            if colour {
                let gs2 = self.sm.gs * self.sm.gs;
                for l in 0..8 * dim * n {
                    self.sm.ym3.e[l] -= gs2 * j3[l];
                }
                std::mem::swap(&mut self.ym, &mut self.sm.ym3);
                self.ym_substeps();
                std::mem::swap(&mut self.ym, &mut self.sm.ym3);
            }
        }
        self.sm.jy = jy;
        self.sm.j2 = j2;
        self.sm.j3 = j3;
        self.em.ph_valid = false;
        self.t = self.t.wrapping_add(1);
    }

    // ------------------------------------------------------------------ Gauss

    /// Densidades de carga de los fermiones en una celda: (ρ_Y = Σ q_f·|ψ_f|², ρ₂^c, ρ₃^c).
    pub(crate) fn sm_charges_at(&self, i: usize) -> (f64, [f64; 3], [f64; 8]) {
        let nt = self.sm_nt();
        let mut ry = 0.0;
        let mut r2 = [0.0f64; 3];
        let mut r3 = [0.0f64; 8];
        let mut t = [0.0f64; 8];
        let z = &self.sm.psi[2 * i * nt..2 * (i + 1) * nt];
        let at = |k: usize| cx(z[2 * k], z[2 * k + 1]);
        for g in 0..self.sm.ng {
            for (fi, f) in FIELDS.iter().enumerate() {
                if !self.sm.quarks && fi >= 2 {
                    continue;
                }
                let blk = f.ni * f.nc;
                for s in 0..2 {
                    let base = g * NGF + f.off + s * blk;
                    let nrm: f64 = (0..blk).map(|j| at(base + j).norm_sqr()).sum();
                    ry += self.sm.gp * f.y * nrm;
                    if f.ni == 2 {
                        for c in 0..f.nc {
                            self.ym.alg.charge(&[at(base + c), at(base + f.nc + c), cx(0.0, 0.0)], &mut t);
                            for k in 0..3 {
                                r2[k] += t[k];
                            }
                        }
                    }
                    if f.nc == 3 {
                        for iso in 0..f.ni {
                            self.sm.ym3.alg.charge(&[at(base + iso * 3), at(base + iso * 3 + 1), at(base + iso * 3 + 2)], &mut t);
                            for k in 0..8 {
                                r3[k] += t[k];
                            }
                        }
                    }
                }
            }
        }
        (ry, r2, r3)
    }

    /// G de SU(3) en la celda: cov-div Ẽ₃ − g_s²·ρ₃.
    fn sm_gauss3_at(&self, i: usize, out: &mut [f64]) {
        let n = self.n;
        let mut tmp = [0.0f64; 8];
        let ym3 = &self.sm.ym3;
        for c in 0..8 {
            out[c] = 0.0;
        }
        for a in 0..self.dim {
            let im = self.nb(i, a, -1);
            for c in 0..8 {
                out[c] += ym3.e[(a * n + i) * 8 + c];
            }
            let u = ym3.link(a * n + im);
            let x = ym3.alg.combine(&ym3.e[(a * n + im) * 8..(a * n + im + 1) * 8]);
            let ad = u.mul(&x).mul_dag(&u);
            ym3.alg.components(&ad, &mut tmp);
            for c in 0..8 {
                out[c] -= tmp[c];
            }
        }
        let (_, _, r3) = self.sm_charges_at(i);
        let gs2 = self.sm.gs * self.sm.gs;
        for c in 0..8 {
            out[c] -= gs2 * r3[c];
        }
    }

    pub(crate) fn sm_store_gauss_ref(&mut self) {
        self.higgs_store_gauss_ref();
        if !(self.sm.quarks && self.sm.colour_dynamic) {
            // sin color dinámico no hay campo de SU(3) (ni ley de Gauss que guardar)
            self.sm.g3_ref = Vec::new();
            return;
        }
        let mut g3 = vec![0.0f64; 8 * self.n];
        let mut tmp = [0.0f64; 8];
        for i in 0..self.n {
            self.sm_gauss3_at(i, &mut tmp);
            g3[8 * i..8 * i + 8].copy_from_slice(&tmp);
        }
        self.sm.g3_ref = g3;
    }

    // ------------------------------------------------------------------ operador en Fourier

    /// Operador de un paso en k (enlaces triviales, Φ uniforme, sin empuje): N × N con N = 30·ng.
    pub(crate) fn sm_step_matrix(&self, parity: usize, k: [f64; 3], phi: [C; 2]) -> Vec<C> {
        let nt = self.sm_nt();
        let mut m = vec![cx(0.0, 0.0); nt * nt];
        for col in 0..nt {
            let mut z = vec![cx(0.0, 0.0); nt];
            z[col] = cx(1.0, 0.0);
            for (s, &a) in self.sm.seq[parity].iter().enumerate() {
                let mi = self.sm.mat_in[parity][s];
                sm_spin_vec(&mut z, &mi, self.sm.ng);
                for g in 0..self.sm.ng {
                    for f in FIELDS.iter() {
                        let blk = f.ni * f.nc;
                        for sp in 0..2 {
                            let dir = Self::sm_dir(f, sp);
                            let ph = C::from_polar(1.0, -(dir as f64) * k[a]);
                            for j in 0..blk {
                                z[g * NGF + f.off + sp * blk + j] *= ph;
                            }
                        }
                    }
                }
            }
            let mo = self.sm.mat_out[parity];
            sm_spin_vec(&mut z, &mo, self.sm.ng);
            let mut dpi = [cx(0.0, 0.0); 2];
            self.sm_yukawa_cell(&mut z, phi, &mut dpi, false);
            for row in 0..nt {
                m[row * nt + col] = z[row];
            }
        }
        m
    }

    /// Autovalores y autovectores del operador de dos pasos en k (vacío Φ = (0, v)).
    pub(crate) fn sm_two_step_eigen(&self, k: [f64; 3]) -> crate::linalg::Eigen {
        let nt = self.sm_nt();
        let phi = [cx(0.0, 0.0), cx(self.higgs.v, 0.0)];
        let a = self.sm_step_matrix(0, k, phi);
        let b = self.sm_step_matrix(1, k, phi);
        let mut w = vec![cx(0.0, 0.0); nt * nt];
        for i in 0..nt {
            for j in 0..nt {
                let mut s = cx(0.0, 0.0);
                for l in 0..nt {
                    s += b[i * nt + l] * a[l * nt + j];
                }
                w[i * nt + j] = s;
            }
        }
        eig(w, nt)
    }

    /// |V_CKM| (9) y J de las SVD de Y_u e Y_d: V = W_u†·W_d.
    pub(crate) fn sm_refresh_ckm(&mut self) {
        let ng = self.sm.ng;
        let mut out = vec![0.0f64; 10];
        if ng == 3 {
            let (wu, wd) = (&self.sm.wm[2], &self.sm.wm[1]);
            let mut v = [[cx(0.0, 0.0); 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = cx(0.0, 0.0);
                    for g in 0..3 {
                        s += wu[g * 3 + i].conj() * wd[g * 3 + j];
                    }
                    v[i][j] = s;
                    out[3 * i + j] = s.norm();
                }
            }
            out[9] = (v[0][0] * v[1][1] * v[0][1].conj() * v[1][0].conj()).im;
        } else {
            out[0] = 1.0;
        }
        self.sm.ckm = out;
    }

    fn sm_set_sector(&mut self, sec: usize, y: Vec<C>) {
        let ng = self.sm.ng;
        let (sv, w, v) = svd(&y, ng);
        self.sm.ys[sec] = sv;
        self.sm.wm[sec] = w;
        self.sm.vm[sec] = v;
        self.sm.ymat[sec] = y;
    }

    /// Densidad de un multiplete en una celda (invariante gauge), sumada sobre generaciones.
    pub(crate) fn sm_field_density(&self, i: usize, fi: usize) -> f64 {
        let f = FIELDS[fi];
        let nt = self.sm_nt();
        let blk = 2 * f.ni * f.nc;
        let mut e = 0.0;
        for g in 0..self.sm.ng {
            let k0 = i * nt + g * NGF + f.off;
            for k in k0..k0 + blk {
                e += self.sm.psi[2 * k].powi(2) + self.sm.psi[2 * k + 1].powi(2);
            }
        }
        e
    }

    /// Densidad de una componente (especie) en una celda, sumada sobre generaciones, espín y color.
    pub(crate) fn sm_species_density(&self, i: usize, sp: usize) -> f64 {
        let (fi, iso) = SPECIES[sp.min(6)];
        let f = FIELDS[fi];
        let nt = self.sm_nt();
        let mut e = 0.0;
        for g in 0..self.sm.ng {
            for s in 0..2 {
                for c in 0..f.nc {
                    let k = i * nt + g * NGF + f.off + s * f.ni * f.nc + iso * f.nc + c;
                    e += self.sm.psi[2 * k].powi(2) + self.sm.psi[2 * k + 1].powi(2);
                }
            }
        }
        e
    }
}

/// Aplica una matriz de espín 2×2 a todas las parejas de un vector de amplitudes de una celda.
fn sm_spin_vec(z: &mut [C], m: &M2, ng: usize) {
    for g in 0..ng {
        for f in FIELDS.iter() {
            let blk = f.ni * f.nc;
            for j in 0..blk {
                let (p0, p1) = (g * NGF + f.off + j, g * NGF + f.off + blk + j);
                let (a, b) = (z[p0], z[p1]);
                z[p0] = m[0][0] * a + m[0][1] * b;
                z[p1] = m[1][0] * a + m[1][1] * b;
            }
        }
    }
}

/// Matriz unitaria aleatoria (Gram–Schmidt de columnas gaussianas complejas).
pub fn random_unitary(n: usize, rng: &mut SplitMix64) -> Vec<C> {
    let mut u = vec![cx(0.0, 0.0); n * n];
    for col in 0..n {
        let mut v: Vec<C> = (0..n).map(|_| cx(rng.gauss(), rng.gauss())).collect();
        for prev in 0..col {
            let ov: C = (0..n).map(|r| u[r * n + prev].conj() * v[r]).sum();
            for r in 0..n {
                v[r] -= ov * u[r * n + prev];
            }
        }
        let nn: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for r in 0..n {
            u[r * n + col] = v[r] / nn;
        }
    }
    u
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.21 (escenario 6): 1 o 3 generaciones (L, e_R y, con `quarks`, Q_L, u_R y d_R) más el doblete
    /// de Higgs, con SU(3) (dinámico si `colour_dynamic`), SU(2) con g y U(1)_Y con g′. Valores por
    /// defecto de §3.19: λ = 0,02, v = 1, κ = 0,25, s = 2 y Yukawas 0,4, 0,3 y 0,5. 3D.
    pub fn set_scenario_sm(&mut self, generations: u32, quarks: bool, colour_dynamic: bool, g: f32, g_prime: f32, g_s: f32) {
        if self.dim != 3 {
            self.set_message("El escenario del modelo estándar es 3D.");
            return;
        }
        if generations == 3 && self.n > 24 * 24 * 24 {
            self.set_message("Con tres generaciones la malla está limitada a 24³ (§2.22).");
            return;
        }
        if self.scenario != 6 {
            // los ajustes comunes del escenario (sin gravedad, sin espinores ni Kaplan…)
            self.set_scenario(6);
        }
        self.scenario = 6;
        self.nc = 2;
        let n = self.n;
        let ng = if generations >= 3 { 3 } else { 1 };
        let mut sm = Sm::new();
        sm.ng = ng;
        sm.quarks = quarks;
        sm.colour_dynamic = colour_dynamic;
        sm.g = g as f64;
        sm.gp = g_prime as f64;
        sm.gs = g_s as f64;
        sm.psi = vec![0.0; 2 * NGF * ng * n];
        sm.tmp = sm.psi.clone();
        // el campo de SU(3) solo se reserva con color dinámico (con el color fijo, U₃ = 1 y no se lee)
        sm.ym3 = YmField::new(3, self.dim, if quarks && colour_dynamic { n } else { 0 });
        sm.active = true;
        sm.rebuild();
        self.sm = sm;
        // bosones: el doblete de §2.12 sobre SU(2) (ym) y U(1)_Y (em)
        self.ncol = 1;
        self.ym = YmField::new(2, self.dim, n);
        self.em = crate::em::EmField::new(self.w, self.h, self.d);
        self.em.mode = 2;
        self.em.substeps = 2;
        self.em.kappa = 0.25;
        self.higgs.g = g as f64;
        self.higgs.gp = g_prime as f64;
        self.em.q = 0.5 * g_prime as f64;
        self.higgs.v = 1.0;
        self.higgs.lambda = 0.02;
        self.higgs.mode = 2;
        self.higgs_alloc(4);
        for i in 0..n {
            self.higgs.phi[4 * i + 2] = 1.0;
        }
        let yv = [0.4, 0.3, 0.5];
        for sec in 0..3 {
            let mut y = vec![cx(0.0, 0.0); ng * ng];
            for k in 0..ng {
                y[k * ng + k] = cx(yv[sec], 0.0);
            }
            self.sm_set_sector(sec, y);
        }
        self.sm_refresh_ckm();
        self.t = 0;
        self.render_mode = 0;
        self.sm_store_gauss_ref();
    }

    /// Yukawas (masas y·v en el vacío): en cada generación los mismos (0 = sin masa).
    pub fn set_yukawa_sm(&mut self, y_e: f32, y_d: f32, y_u: f32) {
        if !self.sm.active {
            return;
        }
        let ng = self.sm.ng;
        for (sec, yv) in [y_e, y_d, y_u].into_iter().enumerate() {
            let mut y = vec![cx(0.0, 0.0); ng * ng];
            for k in 0..ng {
                y[k * ng + k] = cx(yv as f64, 0.0);
            }
            self.sm_set_sector(sec, y);
        }
        self.sm_refresh_ckm();
    }

    /// §2.22: tres generaciones con matrices de Yukawa complejas 3×3 (18 f32: re, im por filas). Se
    /// guarda su SVD para el paso.
    pub fn set_yukawa_matrices_sm(&mut self, y_e: &[f32], y_d: &[f32], y_u: &[f32]) {
        if !self.sm.active || self.sm.ng != 3 {
            self.set_message("Las matrices de Yukawa necesitan el escenario 6 con tres generaciones.");
            return;
        }
        for (sec, ym) in [y_e, y_d, y_u].into_iter().enumerate() {
            if ym.len() < 18 {
                continue;
            }
            let y: Vec<C> = (0..9).map(|k| cx(ym[2 * k] as f64, ym[2 * k + 1] as f64)).collect();
            self.sm_set_sector(sec, y);
        }
        self.sm_refresh_ckm();
    }

    /// Matrices de Yukawa aleatorias W·diag(y)·V† con W y V unitarias de la semilla dada y los valores
    /// singulares de §3.20 (e: 0,05, 0,21, 0,43; d: 0,08, 0,17, 0,37; u: 0,11, 0,26, 0,52). Con `real`,
    /// W y V reales (ortogonales): sin fase, J = 0. Devuelve las 54 f32 (e, d, u).
    pub fn random_yukawa_matrices(&self, seed: u32, real: bool) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed as u64 ^ 0x5EED);
        let svs = [[0.05, 0.21, 0.43], [0.08, 0.17, 0.37], [0.11, 0.26, 0.52]];
        let mut out = Vec::with_capacity(54);
        for sv in svs.iter() {
            let mut w = random_unitary(3, &mut rng);
            let mut v = random_unitary(3, &mut rng);
            if real {
                for m in [&mut w, &mut v] {
                    for z in m.iter_mut() {
                        *z = cx(z.re, 0.0);
                    }
                    // re-ortonormalizar la parte real
                    let mut q = vec![cx(0.0, 0.0); 9];
                    for col in 0..3 {
                        let mut c: Vec<C> = (0..3).map(|r| m[r * 3 + col]).collect();
                        for prev in 0..col {
                            let ov: C = (0..3).map(|r| q[r * 3 + prev].conj() * c[r]).sum();
                            for r in 0..3 {
                                c[r] -= ov * q[r * 3 + prev];
                            }
                        }
                        let nn: f64 = c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                        for r in 0..3 {
                            q[r * 3 + col] = c[r] / nn;
                        }
                    }
                    *m = q;
                }
            }
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = cx(0.0, 0.0);
                    for k in 0..3 {
                        s += w[i * 3 + k] * sv[k] * v[j * 3 + k].conj();
                    }
                    out.push(s.re as f32);
                    out.push(s.im as f32);
                }
            }
        }
        out
    }

    /// |V| (9 valores, fila mayor) y J, de las SVD de Y_u e Y_d.
    pub fn ckm_ptr(&self) -> *const f64 {
        self.sm.ckm.as_ptr()
    }

    pub fn ckm(&self) -> Vec<f64> {
        self.sm.ckm.clone()
    }

    /// [Im det[Y_u·Y_u†, Y_d·Y_d†], 2·J·Π(Δm²_u)·Π(Δm²_d), Re det (0), J sacado del conmutador
    /// = Im det/(2·Π(Δm²_u)·Π(Δm²_d))].
    pub fn jarlskog_check(&self) -> Vec<f64> {
        if self.sm.ng != 3 {
            return vec![0.0; 4];
        }
        let hm = |y: &[C]| -> Mat {
            let mut m = Mat::zero(3);
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = cx(0.0, 0.0);
                    for k in 0..3 {
                        s += y[i * 3 + k] * y[j * 3 + k].conj();
                    }
                    m.set(i, j, s);
                }
            }
            m
        };
        let (a, b) = (hm(&self.sm.ymat[2]), hm(&self.sm.ymat[1]));
        let comm = a.mul(&b).add(&b.mul(&a).scale(cx(-1.0, 0.0)));
        let det = comm.det();
        let pd = |s: &[f64]| (s[2] * s[2] - s[1] * s[1]) * (s[2] * s[2] - s[0] * s[0]) * (s[1] * s[1] - s[0] * s[0]);
        let pp = pd(&self.sm.ys[2]) * pd(&self.sm.ys[1]);
        vec![det.im, 2.0 * self.sm.ckm[9] * pp, det.re, det.im / (2.0 * pp)]
    }

    /// Autoestados del operador de dos pasos en k (vacío Φ = (0, v), enlaces triviales): por cada uno,
    /// [ε, peso en ν_L, en e (L + R), en u (L + R), en d (L + R)], ordenados por ε.
    pub fn sm_spectrum_weights(&self, kx: f32, ky: f32, kz: f32) -> Vec<f64> {
        if !self.sm.active {
            return Vec::new();
        }
        let e = self.sm_two_step_eigen([kx as f64, ky as f64, kz as f64]);
        // especie de cada amplitud de la celda: 0 = ν, 1 = e, 2 = u, 3 = d
        let mut kind = vec![0usize; self.sm_nt()];
        for g in 0..self.sm.ng {
            for (fi, f) in FIELDS.iter().enumerate() {
                for s in 0..2 {
                    for iso in 0..f.ni {
                        for c in 0..f.nc {
                            let k = match (fi, iso) {
                                (0, 0) => 0,
                                (0, _) | (1, _) => 1,
                                (2, 0) | (3, _) => 2,
                                _ => 3,
                            };
                            kind[g * NGF + f.off + s * f.ni * f.nc + iso * f.nc + c] = k;
                        }
                    }
                }
            }
        }
        let mut rows: Vec<[f64; 5]> = e
            .values
            .iter()
            .zip(&e.vectors)
            .map(|(l, v)| {
                let mut w = [0.0f64; 4];
                for (k, z) in v.iter().enumerate() {
                    w[kind[k]] += z.norm_sqr();
                }
                let tot: f64 = w.iter().sum::<f64>().max(1e-300);
                [-l.arg() / 2.0, w[0] / tot, w[1] / tot, w[2] / tot, w[3] / tot]
            })
            .collect();
        rows.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        rows.concat()
    }

    /// §2.21: paquete gaussiano de una especie (0 = ν_L, 1 = e_L, 2 = e_R, 3 = u_L, 4 = d_L, 5 = u_R,
    /// 6 = d_R; color para los quarks; primera generación; norma de `set_packet_norm`), con el espinor de la helicidad que le
    /// corresponde a su quiralidad en la dirección de k (−1 para L, +1 para R; espín arriba con k = 0).
    pub fn inject_sm_packet(&mut self, species: u32, x: f32, y: f32, z: f32, sigma: f32, kx: f32, ky: f32, kz: f32, colour: u32) {
        if !self.sm.active {
            return;
        }
        let sp = (species as usize).min(6);
        let (fi, iso) = SPECIES[sp];
        if !self.sm.quarks && fi >= 2 {
            return;
        }
        let f = FIELDS[fi];
        let col = if f.nc == 3 { (colour as usize).min(2) } else { 0 };
        let k = [kx as f64, ky as f64, kz as f64];
        let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
        // autovector de σ·k̂ con autovalor h (h = +1 para R, −1 para L)
        let hsign = if f.right { 1.0 } else { -1.0 };
        let spinor = if kn < 1e-12 {
            [cx(1.0, 0.0), cx(0.0, 0.0)]
        } else {
            let (th, ph) = ((k[2] / kn).clamp(-1.0, 1.0).acos(), k[1].atan2(k[0]));
            if hsign > 0.0 {
                [cx((0.5 * th).cos(), 0.0), C::from_polar((0.5 * th).sin(), ph)]
            } else {
                [cx(-(0.5 * th).sin(), 0.0), C::from_polar((0.5 * th).cos(), ph)]
            }
        };
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let s = (sigma as f64).max(0.5);
        let nt = self.sm_nt();
        let mut amps = vec![cx(0.0, 0.0); self.n];
        let mut norm = 0.0;
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let (dx, dy, dz) = (min_image(px as f64 - x as f64, w), min_image(py as f64 - y as f64, h), min_image(pz as f64 - z as f64, d));
            let arg = (dx * dx + dy * dy + dz * dz) / (2.0 * s * s);
            if arg > 40.0 {
                continue;
            }
            amps[i] = C::from_polar((-arg).exp(), k[0] * dx + k[1] * dy + k[2] * dz);
            norm += amps[i].norm_sqr();
        }
        let sc = self.packet_norm.sqrt() / norm.max(1e-300).sqrt();
        for i in 0..self.n {
            if amps[i].norm_sqr() == 0.0 {
                continue;
            }
            for sidx in 0..2 {
                let kk = i * nt + f.off + sidx * f.ni * f.nc + iso * f.nc + col;
                let v = amps[i] * spinor[sidx] * sc;
                self.sm.psi[2 * kk] += v.re;
                self.sm.psi[2 * kk + 1] += v.im;
            }
        }
    }

    /// Fermiones aleatorios: amplitud gaussiana `amp` en todas las componentes activas (campos fuertes
    /// para las pruebas de Gauss).
    pub fn seed_sm_random(&mut self, amp: f32, seed: u32) {
        if !self.sm.active {
            return;
        }
        let mut rng = SplitMix64::new(seed as u64 ^ 0x5EED5);
        let nt = self.sm_nt();
        for i in 0..self.n {
            for g in 0..self.sm.ng {
                for (fi, f) in FIELDS.iter().enumerate() {
                    if !self.sm.quarks && fi >= 2 {
                        continue;
                    }
                    for j in 0..2 * f.ni * f.nc {
                        let k = i * nt + g * NGF + f.off + j;
                        self.sm.psi[2 * k] += amp as f64 * rng.gauss();
                        self.sm.psi[2 * k + 1] += amp as f64 * rng.gauss();
                    }
                }
            }
        }
    }

    /// Norma de una especie (conversiones ν_L ↔ e_L y e_L ↔ e_R).
    pub fn species_norm(&self, species: u32) -> f64 {
        if !self.sm.active {
            return 0.0;
        }
        (0..self.n).map(|i| self.sm_species_density(i, species as usize)).sum()
    }

    /// Centroide (media circular) de una especie en el eje.
    pub fn species_centroid(&self, species: u32, axis: u32) -> f64 {
        let a = (axis as usize).min(2);
        let l = [self.w as f64, self.h as f64, self.d as f64][a];
        let (mut cs, mut sn) = (0.0, 0.0);
        for i in 0..self.n {
            let e = self.sm_species_density(i, species as usize);
            if e == 0.0 {
                continue;
            }
            let (x, y, z) = self.coords(i);
            let th = 2.0 * PI * [x as f64, y as f64, z as f64][a] / l;
            cs += e * th.cos();
            sn += e * th.sin();
        }
        (sn.atan2(cs) * l / (2.0 * PI)).rem_euclid(l)
    }

    /// Residuos de Gauss [SU(3), SU(2), U(1)_Y] respecto de la referencia guardada.
    pub fn sm_gauss_residuals(&self) -> Vec<f64> {
        if !self.sm.active {
            return vec![0.0; 3];
        }
        if !self.sm.bosons_dynamic {
            // con los bosones fijos (campos externos) no hay ley de Gauss que conservar
            return vec![f64::NAN; 3];
        }
        let r = self.higgs_gauss_residual_parts();
        if !(self.sm.quarks && self.sm.colour_dynamic) {
            // sin color dinámico, Ẽ₃ no responde a las corrientes: la ley de Gauss de SU(3) no aplica
            return vec![f64::NAN, r[0], r[1]];
        }
        let mut g3 = 0.0f64;
        let mut tmp = [0.0f64; 8];
        for i in 0..self.n {
            self.sm_gauss3_at(i, &mut tmp);
            for c in 0..8 {
                g3 = g3.max((tmp[c] - self.sm.g3_ref[8 * i + c]).abs());
            }
        }
        vec![g3, r[0], r[1]]
    }

    /// Guarda la ley de Gauss actual como referencia (tras sembrar el estado).
    pub fn sm_store_gauss_reference(&mut self) {
        if self.sm.active {
            self.sm_store_gauss_ref();
        }
    }

    /// Masas en el vacío (y·v) por sector: [e…, d…, u…], ng valores cada uno, en orden creciente.
    pub fn sm_masses(&self) -> Vec<f64> {
        let v = self.higgs.v;
        self.sm.ys.iter().flat_map(|ys| ys.iter().map(move |y| y * v)).collect()
    }

    /// Bosones dinámicos (por defecto) o fijos (enlaces y Higgs externos).
    pub fn set_sm_bosons_dynamic(&mut self, on: bool) {
        self.sm.bosons_dynamic = on;
    }

    /// El empuje del Yukawa sobre Π (imprescindible para la ley de Gauss; se puede quitar para
    /// comprobarlo).
    pub fn set_sm_higgs_reaction(&mut self, on: bool) {
        self.sm.reaction = on;
    }

    /// Campo externo uniforme en los enlaces de un eje: U₂ = exp(−i·angle·T^gen) y A_Y = ay.
    pub fn set_sm_link_field(&mut self, axis: u32, generator: u32, angle: f32, ay: f32) {
        if !self.sm.active {
            return;
        }
        let a = (axis as usize).min(self.dim - 1);
        let mut th = [0.0f64; 3];
        th[(generator as usize).clamp(1, 3) - 1] = angle as f64;
        let u = self.ym.alg.exp_minus_i(&th);
        for i in 0..self.n {
            self.ym.set_link(a * self.n + i, &u);
            self.em.a[a * self.n + i] = ay as f64;
        }
    }

    /// Añade ruido al Higgs (Φ y Π) para las pruebas de Gauss.
    pub fn perturb_sm_higgs(&mut self, amp: f32, seed: u32) {
        let mut rng = SplitMix64::new(seed as u64 ^ 0x4846);
        for v in self.higgs.phi.iter_mut() {
            *v += amp as f64 * rng.gauss();
        }
        for v in self.higgs.pi.iter_mut() {
            *v += amp as f64 * rng.gauss();
        }
    }

    /// Transformación gauge aleatoria de SU(3)×SU(2)×U(1)_Y (amplitud dada) sobre fermiones, Higgs,
    /// enlaces y campos eléctricos, con las referencias de Gauss. Nada físico debe cambiar.
    pub fn sm_random_gauge(&mut self, amplitude: f32, seed: u32) {
        if !self.sm.active {
            return;
        }
        let n = self.n;
        let dim = self.dim;
        let amp = amplitude as f64;
        let mut rng = SplitMix64::new(seed as u64 ^ 0x6A06E5);
        let alg2 = Algebra::new(2);
        let alg3 = Algebra::new(3);
        let o2: Vec<Mat> = (0..n).map(|_| alg2.exp_minus_i(&[amp * rng.gauss(), amp * rng.gauss(), amp * rng.gauss()])).collect();
        // con el color fijo, los enlaces de SU(3) son la identidad y el salto no los lee: no se giran
        let amp3 = if self.sm.quarks && self.sm.colour_dynamic { amp } else { 0.0 };
        let o3: Vec<Mat> = (0..n).map(|_| alg3.exp_minus_i(&(0..8).map(|_| amp3 * rng.gauss()).collect::<Vec<f64>>())).collect();
        let chi: Vec<f64> = (0..n).map(|_| amp * rng.gauss()).collect();
        let gp = self.sm.gp;
        let nt = self.sm_nt();
        // fermiones
        for i in 0..n {
            for g in 0..self.sm.ng {
                for f in FIELDS.iter() {
                    let blk = f.ni * f.nc;
                    let ph = C::from_polar(1.0, gp * f.y * chi[i]);
                    for s in 0..2 {
                        let base = i * nt + g * NGF + f.off + s * blk;
                        let mut v = [cx(0.0, 0.0); 6];
                        for j in 0..blk {
                            v[j] = cx(self.sm.psi[2 * (base + j)], self.sm.psi[2 * (base + j) + 1]) * ph;
                        }
                        if f.ni == 2 {
                            for c in 0..f.nc {
                                let r = o2[i].apply(&[v[c], v[f.nc + c], cx(0.0, 0.0)]);
                                v[c] = r[0];
                                v[f.nc + c] = r[1];
                            }
                        }
                        if f.nc == 3 {
                            for iso in 0..f.ni {
                                let r = o3[i].apply(&[v[iso * 3], v[iso * 3 + 1], v[iso * 3 + 2]]);
                                for c in 0..3 {
                                    v[iso * 3 + c] = r[c];
                                }
                            }
                        }
                        for j in 0..blk {
                            self.sm.psi[2 * (base + j)] = v[j].re;
                            self.sm.psi[2 * (base + j) + 1] = v[j].im;
                        }
                    }
                }
            }
        }
        // Higgs (Y = ½)
        for i in 0..n {
            let ph = C::from_polar(1.0, 0.5 * gp * chi[i]);
            for fld in [&mut self.higgs.phi, &mut self.higgs.pi] {
                let v = [cx(fld[4 * i], fld[4 * i + 1]) * ph, cx(fld[4 * i + 2], fld[4 * i + 3]) * ph, cx(0.0, 0.0)];
                let r = o2[i].apply(&v);
                fld[4 * i] = r[0].re;
                fld[4 * i + 1] = r[0].im;
                fld[4 * i + 2] = r[1].re;
                fld[4 * i + 3] = r[1].im;
            }
        }
        // enlaces y campos
        let colour = self.sm.quarks && self.sm.colour_dynamic;
        let mut tmp = [0.0f64; 8];
        for a in 0..dim {
            for i in 0..n {
                let ip = self.nb(i, a, 1);
                let l = a * n + i;
                let u = o2[ip].mul(&self.ym.link(l)).mul_dag(&o2[i]);
                self.ym.set_link(l, &u);
                let x = self.ym.alg.combine(&self.ym.e[l * 3..l * 3 + 3]);
                let ad = o2[i].mul(&x).mul_dag(&o2[i]);
                self.ym.alg.components(&ad, &mut tmp);
                self.ym.e[l * 3..l * 3 + 3].copy_from_slice(&tmp[..3]);
                if colour {
                    let u3 = o3[ip].mul(&self.sm.ym3.link(l)).mul_dag(&o3[i]);
                    self.sm.ym3.set_link(l, &u3);
                    let x3 = self.sm.ym3.alg.combine(&self.sm.ym3.e[l * 8..l * 8 + 8]);
                    let ad3 = o3[i].mul(&x3).mul_dag(&o3[i]);
                    self.sm.ym3.alg.components(&ad3, &mut tmp);
                    self.sm.ym3.e[l * 8..l * 8 + 8].copy_from_slice(&tmp[..8]);
                }
                self.em.a[l] += chi[ip] - chi[i];
            }
        }
        // referencias de Gauss
        for i in 0..n {
            let x = self.ym.alg.combine(&self.higgs.g_ref[4 * i..4 * i + 3]);
            let ad = o2[i].mul(&x).mul_dag(&o2[i]);
            self.ym.alg.components(&ad, &mut tmp);
            self.higgs.g_ref[4 * i..4 * i + 3].copy_from_slice(&tmp[..3]);
            if !colour {
                continue;
            }
            let x3 = self.sm.ym3.alg.combine(&self.sm.g3_ref[8 * i..8 * i + 8]);
            let ad3 = o3[i].mul(&x3).mul_dag(&o3[i]);
            self.sm.ym3.alg.components(&ad3, &mut tmp);
            self.sm.g3_ref[8 * i..8 * i + 8].copy_from_slice(&tmp[..8]);
        }
        self.em.ph_valid = false;
    }

    /// Distancia máxima entre las densidades invariantes gauge (la de cada multiplete: L, e_R, Q_L, u_R
    /// y d_R, sumadas sobre isospín, color y espín) de dos universos. ν_L y e_L por separado no son
    /// invariantes: una transformación de SU(2) los mezcla.
    pub fn sm_density_distance(&self, other: &QuantumSpacetimeUniverse) -> f64 {
        if other.n != self.n || !other.sm.active || other.sm.ng != self.sm.ng {
            return f64::NAN;
        }
        let mut d = 0.0f64;
        for i in 0..self.n {
            for fi in 0..5 {
                d = d.max((self.sm_field_density(i, fi) - other.sm_field_density(i, fi)).abs());
            }
        }
        d
    }

    /// Cuasi-energías por paso del operador de dos pasos en k (vacío Φ = (0, v), enlaces triviales):
    /// −arg(λ)/2 ordenadas.
    pub fn sm_spectrum(&self, kx: f32, ky: f32, kz: f32) -> Vec<f64> {
        if !self.sm.active {
            return Vec::new();
        }
        let e = self.sm_two_step_eigen([kx as f64, ky as f64, kz as f64]);
        let mut v: Vec<f64> = e.values.iter().map(|l| -l.arg() / 2.0).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    }

    /// Matriz |V| de los autovectores del operador de dos pasos en k = 0 (tres generaciones): para
    /// cada estado de masa de tipo u y de tipo d (cuasi-energía positiva), su contenido de generación
    /// en u_L y en d_L (color 0); |V|_kl = |a_k†·b_l|, con las masas ordenadas.
    pub fn sm_walk_ckm(&self) -> Vec<f64> {
        if !self.sm.active || self.sm.ng != 3 || !self.sm.quarks {
            return Vec::new();
        }
        let e = self.sm_two_step_eigen([0.0; 3]);
        let comp = |g: usize, iso: usize, s: usize| g * NGF + FIELDS[2].off + s * 6 + iso * 3;
        let gen_vec = |iso: usize, target: f64| -> [C; 3] {
            // de los autovectores con cuasi-energía +target, el de más peso en (iso, color 0)
            let mut best = (0.0f64, [cx(0.0, 0.0); 3]);
            for (k, l) in e.values.iter().enumerate() {
                if (-l.arg() / 2.0 - target).abs() > 1e-6 {
                    continue;
                }
                for s in 0..2 {
                    let v: [C; 3] = [e.vectors[k][comp(0, iso, s)], e.vectors[k][comp(1, iso, s)], e.vectors[k][comp(2, iso, s)]];
                    let nn: f64 = v.iter().map(|z| z.norm_sqr()).sum();
                    if nn > best.0 {
                        let r = nn.sqrt();
                        best = (nn, [v[0] / r, v[1] / r, v[2] / r]);
                    }
                }
            }
            best.1
        };
        let mut out = vec![0.0f64; 9];
        let (yu, yd) = (self.sm.ys[2].clone(), self.sm.ys[1].clone());
        let a: Vec<[C; 3]> = yu.iter().map(|y| gen_vec(0, *y * self.higgs.v)).collect();
        let b: Vec<[C; 3]> = yd.iter().map(|y| gen_vec(1, *y * self.higgs.v)).collect();
        for i in 0..3 {
            for j in 0..3 {
                let ov: C = (0..3).map(|g| a[i][g].conj() * b[j][g]).sum();
                out[3 * i + j] = ov.norm();
            }
        }
        out
    }

    /// Fase 14 (LHCb / CMS 2026): comprobación exacta del mecanismo GIM en el modelo estándar de
    /// tres generaciones. Compara la corriente neutra Z⁰ (T³ − sin²θ_W·Q, diagonal en sabor entre
    /// los autoestados de masa de d_L y u_L) frente a la corriente cargada W¹ (mezcla CKM no nula).
    /// Devuelve [max_offdiag_Z_d, max_offdiag_Z_u, max_offdiag_CKM, |V_ts|, |V_cb|, err_unit_CKM, dispersión_gZ_dL].
    pub fn sm_gim_check(&self) -> Vec<f64> {
        if !self.sm.active || self.sm.ng != 3 || !self.sm.quarks {
            return Vec::new();
        }
        let e = self.sm_two_step_eigen([0.0; 3]);
        let comp = |g: usize, iso: usize, s: usize| g * NGF + FIELDS[2].off + s * 6 + iso * 3;
        let gen_vec = |iso: usize, target: f64| -> [C; 3] {
            let mut best = (0.0f64, [cx(0.0, 0.0); 3]);
            for (k, l) in e.values.iter().enumerate() {
                if (-l.arg() / 2.0 - target).abs() > 1e-6 {
                    continue;
                }
                for s in 0..2 {
                    let v: [C; 3] = [e.vectors[k][comp(0, iso, s)], e.vectors[k][comp(1, iso, s)], e.vectors[k][comp(2, iso, s)]];
                    let nn: f64 = v.iter().map(|z| z.norm_sqr()).sum();
                    if nn > best.0 {
                        let r = nn.sqrt();
                        best = (nn, [v[0] / r, v[1] / r, v[2] / r]);
                    }
                }
            }
            best.1
        };
        let u_vecs: Vec<[C; 3]> = self.sm.ys[2].iter().map(|y| gen_vec(0, *y * self.higgs.v)).collect();
        let d_vecs: Vec<[C; 3]> = self.sm.ys[1].iter().map(|y| gen_vec(1, *y * self.higgs.v)).collect();
        let s2w = self.sm.gp * self.sm.gp / (self.sm.g * self.sm.g + self.sm.gp * self.sm.gp);
        let gz_d = -0.5 + (1.0 / 3.0) * s2w;
        let gz_u = 0.5 - (2.0 / 3.0) * s2w;
        // Corriente neutra Z⁰ en la base de sabor: universal g_Z·I₃ → matriz en base de masas
        let mut max_z_d = 0.0f64;
        let mut max_z_u = 0.0f64;
        let mut gz_diag_min = f64::INFINITY;
        let mut gz_diag_max = f64::NEG_INFINITY;
        for i in 0..3 {
            for j in 0..3 {
                let ov_d: C = (0..3).map(|g| d_vecs[i][g].conj() * (gz_d * d_vecs[j][g])).sum();
                let ov_u: C = (0..3).map(|g| u_vecs[i][g].conj() * (gz_u * u_vecs[j][g])).sum();
                if i == j {
                    gz_diag_min = gz_diag_min.min(ov_d.re);
                    gz_diag_max = gz_diag_max.max(ov_d.re);
                } else {
                    max_z_d = max_z_d.max(ov_d.norm());
                    max_z_u = max_z_u.max(ov_u.norm());
                }
            }
        }
        // Corriente cargada W¹: V_ij = u_i†·d_j
        let mut vmat = [[cx(0.0, 0.0); 3]; 3];
        let mut max_ckm_off = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let ov: C = (0..3).map(|g| u_vecs[i][g].conj() * d_vecs[j][g]).sum();
                vmat[i][j] = ov;
                if i != j {
                    max_ckm_off = max_ckm_off.max(ov.norm());
                }
            }
        }
        let mut unit_err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let s: C = (0..3).map(|k| vmat[k][i].conj() * vmat[k][j]).sum();
                let tgt = if i == j { cx(1.0, 0.0) } else { cx(0.0, 0.0) };
                unit_err = unit_err.max((s - tgt).norm());
            }
        }
        vec![max_z_d, max_z_u, max_ckm_off, vmat[2][1].norm(), vmat[1][2].norm(), unit_err, (gz_diag_max - gz_diag_min).abs()]
    }

    /// Fase 14: activa un campo externo BSM de bosón Z' con cambio de sabor b_L ↔ s_L (ángulo
    /// `theta_bs`) y acoplamiento a μ_L (`theta_mu`) en los enlaces del eje `axis` (solo 3 generaciones).
    /// Con ambos ángulos en 0 se desactiva.
    pub fn set_sm_zprime_field(&mut self, axis: u32, theta_bs: f32, theta_mu: f32) {
        if !self.sm.active || self.sm.ng != 3 {
            self.sm.zp_active = false;
            return;
        }
        let tbs = theta_bs as f64;
        let tmu = theta_mu as f64;
        if tbs == 0.0 && tmu == 0.0 {
            self.sm.zp_active = false;
            return;
        }
        self.sm.zp_axis = (axis as usize).min(self.dim - 1);
        // En base de masas de d_L: U_mass = exp(−i θ_bs σ_x^{(1,2)}) mezclando s (1) y b (2)
        let (sn, cs) = tbs.sin_cos();
        let u_mass_d = [
            cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0),
            cx(0.0, 0.0), cx(cs, 0.0), cx(0.0, -sn),
            cx(0.0, 0.0), cx(0.0, -sn), cx(cs, 0.0),
        ];
        // En base de masas de e_L: fase exp(−i θ_mu) sobre μ_L (generación 1)
        let u_mass_l = [
            cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0),
            cx(0.0, 0.0), C::from_polar(1.0, -tmu), cx(0.0, 0.0),
            cx(0.0, 0.0), cx(0.0, 0.0), cx(1.0, 0.0),
        ];
        // Rotamos a la base de sabor: U_flav = W · U_mass · W†
        let rotate_flav = |w: &[C], um: &[C; 9]| -> [C; 9] {
            let mut tmp = [cx(0.0, 0.0); 9];
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = cx(0.0, 0.0);
                    for k in 0..3 {
                        s += w[i * 3 + k] * um[k * 3 + j];
                    }
                    tmp[i * 3 + j] = s;
                }
            }
            let mut out = [cx(0.0, 0.0); 9];
            for i in 0..3 {
                for j in 0..3 {
                    let mut s = cx(0.0, 0.0);
                    for k in 0..3 {
                        s += tmp[i * 3 + k] * w[j * 3 + k].conj();
                    }
                    out[i * 3 + j] = s;
                }
            }
            out
        };
        self.sm.zp_u_d = rotate_flav(&self.sm.wm[1], &u_mass_d);
        self.sm.zp_u_l = rotate_flav(&self.sm.wm[0], &u_mass_l);
        self.sm.zp_active = true;
    }

    /// Inyecta un autoestado de masa (generación `mass_gen` ∈ 0..ng) de un quark o leptón levógiro
    /// (`species`: 1 = e_L, 3 = u_L, 4 = d_L) rotado por W_sec a la base de sabor.
    #[allow(clippy::too_many_arguments)]
    pub fn inject_sm_mass_state(&mut self, species: u32, mass_gen: u32, x: f32, y: f32, z: f32, sigma: f32, kx: f32, ky: f32, kz: f32, colour: u32) {
        if !self.sm.active {
            return;
        }
        let sp = (species as usize).min(6);
        let (fi, iso) = SPECIES[sp];
        if !self.sm.quarks && fi >= 2 {
            return;
        }
        let ng = self.sm.ng;
        let mg = (mass_gen as usize).min(ng - 1);
        let sec = match (fi, iso) {
            (0, _) | (1, _) => 0,
            (2, 1) | (4, _) => 1,
            _ => 2,
        };
        let f = FIELDS[fi];
        let col = if f.nc == 3 { (colour as usize).min(2) } else { 0 };
        let k = [kx as f64, ky as f64, kz as f64];
        let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
        let hsign = if f.right { 1.0 } else { -1.0 };
        let spinor = if kn < 1e-12 {
            [cx(1.0, 0.0), cx(0.0, 0.0)]
        } else {
            let (th, ph) = ((k[2] / kn).clamp(-1.0, 1.0).acos(), k[1].atan2(k[0]));
            if hsign > 0.0 {
                [cx((0.5 * th).cos(), 0.0), C::from_polar((0.5 * th).sin(), ph)]
            } else {
                [cx(-(0.5 * th).sin(), 0.0), C::from_polar((0.5 * th).cos(), ph)]
            }
        };
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let s = (sigma as f64).max(0.5);
        let nt = self.sm_nt();
        let mut amps = vec![cx(0.0, 0.0); self.n];
        let mut norm = 0.0;
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let (dx, dy, dz) = (min_image(px as f64 - x as f64, w), min_image(py as f64 - y as f64, h), min_image(pz as f64 - z as f64, d));
            let arg = (dx * dx + dy * dy + dz * dz) / (2.0 * s * s);
            if arg > 40.0 {
                continue;
            }
            amps[i] = C::from_polar((-arg).exp(), k[0] * dx + k[1] * dy + k[2] * dz);
            norm += amps[i].norm_sqr();
        }
        let sc = self.packet_norm.sqrt() / norm.max(1e-300).sqrt();
        let blk = f.ni * f.nc;
        let rot = if f.right { &self.sm.vm[sec] } else { &self.sm.wm[sec] };
        for i in 0..self.n {
            let a = amps[i] * sc;
            if a.norm_sqr() == 0.0 {
                continue;
            }
            for g in 0..ng {
                let wg = rot[g * ng + mg];
                for spn in 0..2 {
                    let val = a * wg * spinor[spn];
                    let idx = 2 * (i * nt + g * NGF + f.off + spn * blk + iso * f.nc + col);
                    self.sm.psi[idx] += val.re;
                    self.sm.psi[idx + 1] += val.im;
                }
            }
        }
    }

    /// Norma proyectada de d_L sobre los tres autoestados de masa [d_L, s_L, b_L] (vía W_d†).
    pub fn sm_down_mass_norms(&self) -> Vec<f64> {
        if !self.sm.active || !self.sm.quarks {
            return Vec::new();
        }
        let ng = self.sm.ng;
        let nt = self.sm_nt();
        let f = FIELDS[2];
        let blk = f.ni * f.nc;
        let wd = &self.sm.wm[1];
        let mut norms = vec![0.0f64; ng];
        for i in 0..self.n {
            for s in 0..2 {
                for c in 0..3 {
                    let mut fg = [cx(0.0, 0.0); 3];
                    for g in 0..ng {
                        let o = 2 * (i * nt + g * NGF + f.off + s * blk + 3 + c);
                        fg[g] = cx(self.sm.psi[o], self.sm.psi[o + 1]);
                    }
                    for m in 0..ng {
                        let mut proj = cx(0.0, 0.0);
                        for g in 0..ng {
                            proj += wd[g * ng + m].conj() * fg[g];
                        }
                        norms[m] += proj.norm_sqr();
                    }
                }
            }
        }
        norms
    }
}

impl QuantumSpacetimeUniverse {
    /// Aplica el enlace unitario del Z' BSM sobre d_L (mezcla b_L ↔ s_L) y e_L (fase en μ_L)
    /// justo después del salto a lo largo del eje `zp_axis`.
    fn sm_apply_zprime_hop(&self, dst: &mut [f64]) {
        let nt = self.sm_nt();
        let ud = &self.sm.zp_u_d;
        let ul = &self.sm.zp_u_l;
        let apply3 = |u: &[C; 9], dag: bool, v: [C; 3]| -> [C; 3] {
            let mut r = [cx(0.0, 0.0); 3];
            for i in 0..3 {
                let mut s = cx(0.0, 0.0);
                for j in 0..3 {
                    let m = if dag { u[j * 3 + i].conj() } else { u[i * 3 + j] };
                    s += m * v[j];
                }
                r[i] = s;
            }
            r
        };
        for i in 0..self.n {
            // d_L: FIELDS[2], iso = 1, 3 colores
            if self.sm.quarks {
                let f = FIELDS[2];
                let blk = f.ni * f.nc;
                for s in 0..2 {
                    let dag = Self::sm_dir(&f, s) < 0;
                    for c in 0..3 {
                        let mut v = [cx(0.0, 0.0); 3];
                        for g in 0..3 {
                            let o = 2 * (i * nt + g * NGF + f.off + s * blk + 3 + c);
                            v[g] = cx(dst[o], dst[o + 1]);
                        }
                        let r = apply3(ud, dag, v);
                        for g in 0..3 {
                            let o = 2 * (i * nt + g * NGF + f.off + s * blk + 3 + c);
                            dst[o] = r[g].re;
                            dst[o + 1] = r[g].im;
                        }
                    }
                }
            }
            // e_L: FIELDS[0], iso = 1, 1 color
            let fl = FIELDS[0];
            let blkl = fl.ni * fl.nc;
            for s in 0..2 {
                let dag = Self::sm_dir(&fl, s) < 0;
                let mut v = [cx(0.0, 0.0); 3];
                for g in 0..3 {
                    let o = 2 * (i * nt + g * NGF + fl.off + s * blkl + 1);
                    v[g] = cx(dst[o], dst[o + 1]);
                }
                let r = apply3(ul, dag, v);
                for g in 0..3 {
                    let o = 2 * (i * nt + g * NGF + fl.off + s * blkl + 1);
                    dst[o] = r[g].re;
                    dst[o + 1] = r[g].im;
                }
            }
        }
    }
}
