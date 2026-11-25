//! Caminatas de espinores en f64: el Dirac 3D de 4 componentes (§2.16) y el fermión de Weyl de
//! 2 componentes (§2.17), en 3D o en 2D con k_z fijo.
//!
//! Cada subpaso pasa el espinor a la base propia de su eje (α_a = τ_x⊗σ_a en Dirac, σ_a en Weyl) y
//! mueve sus componentes: las de autovalor +1 una celda en +a y las de −1 una celda en −a. En los
//! bombeos de Higashikawa et al. solo se mueve una de las dos. Con el U(1) de §2.9 o el SU(N) de
//! §2.11, lo que salta de r a r + â toma U_a(r) y lo que va en contra, U_a(r)†. La corriente de cada
//! eje se toma en esa base, antes del salto: la norma que cruza cada enlace es exactamente J, así
//! que la ley de Gauss se conserva exactamente.
//!
//! El estado vive en f64 (las pruebas de Gauss y de las inestabilidades del condensado lo piden) y
//! se copia en f32 a `buffer_read` para el render y los análisis genéricos. Es la función de onda
//! de una partícula, no un campo cuantizado: no hay mar de Dirac ni estadística de Fermi.

use crate::fftn::{wavenumber, FftN};
use crate::linalg::eig;
use crate::sun::{basis_colour, C};
use crate::util::min_image;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use wasm_bindgen::prelude::*;

/// Matriz 4×4 compleja (con 2 componentes se usa el bloque 2×2 de arriba a la izquierda).
pub type M4 = [[C; 4]; 4];

#[inline(always)]
pub fn cx(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

pub fn m4_zero() -> M4 {
    [[cx(0.0, 0.0); 4]; 4]
}

pub fn m4_identity() -> M4 {
    let mut m = m4_zero();
    for i in 0..4 {
        m[i][i] = cx(1.0, 0.0);
    }
    m
}

pub fn m4_mul(a: &M4, b: &M4) -> M4 {
    let mut r = m4_zero();
    for i in 0..4 {
        for j in 0..4 {
            let mut s = cx(0.0, 0.0);
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            r[i][j] = s;
        }
    }
    r
}

pub fn m4_dag(a: &M4) -> M4 {
    let mut r = m4_zero();
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = a[j][i].conj();
        }
    }
    r
}

#[cfg(test)]
pub fn m4_add(a: &M4, b: &M4, sb: C) -> M4 {
    let mut r = *a;
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] += sb * b[i][j];
        }
    }
    r
}

pub fn m4_apply(a: &M4, v: &[C; 4]) -> [C; 4] {
    let mut r = [cx(0.0, 0.0); 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i] += a[i][j] * v[j];
        }
    }
    r
}

/// Producto de Kronecker de dos matrices 2×2 (índice 2·i + j).
pub fn kron2(a: &[[C; 2]; 2], b: &[[C; 2]; 2]) -> M4 {
    let mut r = m4_zero();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    r[2 * i + k][2 * j + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    r
}

pub fn pauli(a: usize) -> [[C; 2]; 2] {
    let (o, z) = (cx(1.0, 0.0), cx(0.0, 0.0));
    match a {
        0 => [[z, o], [o, z]],
        1 => [[z, cx(0.0, -1.0)], [cx(0.0, 1.0), z]],
        2 => [[o, z], [z, cx(-1.0, 0.0)]],
        _ => [[o, z], [z, o]],
    }
}

/// Columnas = autovectores de σ_a (autovalor +1 primero).
pub fn pauli_eigenbasis(a: usize) -> [[C; 2]; 2] {
    let s = FRAC_1_SQRT_2;
    match a {
        0 => [[cx(s, 0.0), cx(s, 0.0)], [cx(s, 0.0), cx(-s, 0.0)]],
        1 => [[cx(s, 0.0), cx(s, 0.0)], [cx(0.0, s), cx(0.0, -s)]],
        _ => [[cx(1.0, 0.0), cx(0.0, 0.0)], [cx(0.0, 0.0), cx(1.0, 0.0)]],
    }
}

/// Matriz 4×4 de Dirac: τ_a ⊗ σ_b (a, b = 0, 1, 2 para x, y, z; 3 = identidad).
pub fn dirac_matrix(tau: usize, sigma: usize) -> M4 {
    kron2(&pauli(tau), &pauli(sigma))
}

#[cfg(test)]
/// β = τ_z ⊗ 1.
pub fn beta4() -> M4 {
    dirac_matrix(2, 3)
}

#[cfg(test)]
/// α_a = τ_x ⊗ σ_a.
pub fn alpha4(a: usize) -> M4 {
    dirac_matrix(0, a)
}

#[cfg(test)]
/// B₂ = τ_y ⊗ 1: anticonmuta con β y con las tres α (la masa "axial" de §2.18).
pub fn b2_matrix() -> M4 {
    dirac_matrix(1, 3)
}

/// Base propia del eje a: Dirac, V_a = H ⊗ R_a (V_a†·α_a·V_a = τ_z⊗σ_z = diag(+, −, −, +));
/// Weyl, R_a (R_a†·σ_a·R_a = σ_z).
pub fn axis_basis(ns: usize, a: usize) -> M4 {
    let r = pauli_eigenbasis(a);
    if ns == 4 {
        kron2(&pauli_eigenbasis(0), &r)
    } else {
        let mut m = m4_zero();
        for i in 0..2 {
            for j in 0..2 {
                m[i][j] = r[i][j];
            }
        }
        m[2][2] = cx(1.0, 0.0);
        m[3][3] = cx(1.0, 0.0);
        m
    }
}

/// Un subpaso: eje y desplazamiento de cada componente en la base propia del eje
/// (+1: avanza en +a; −1: retrocede; 0: se queda).
#[derive(Clone, Copy, Debug)]
pub struct Sub {
    pub axis: usize,
    pub sign: [i8; 4],
}

pub struct SpinorWalk {
    pub active: bool,
    /// Componentes de espín: 4 (Dirac 3D) o 2 (Weyl).
    pub ns: usize,
    pub weyl: bool,
    /// Dirac: orden fijo x, y, z (true) o alternado (false, por defecto).
    pub fixed_order: bool,
    /// Weyl: 0 = orden fijo, 1 = alternado, 2 = bombeos selectivos de espín.
    pub variant: u32,
    /// 2D con k_z fijo: los saltos en z pasan a ser las fases e^{∓ik_z}.
    pub slice: bool,
    pub kz: f64,
    pub psi: Vec<f64>,
    pub tmp: Vec<f64>,
    /// Paquetes evolucionados por partes (cada uno con el mismo campo, §3.13).
    pub parts: Vec<Vec<f64>>,
    pub part_centers: Vec<[f64; 3]>,
    pub parts_mode: bool,
    /// Partes distinguibles: cada paquete es una partícula propia y el campo lo crean
    /// Σ_p ρ(ψ_p) y Σ_p J(ψ_p), sin los términos de interferencia entre partes.
    pub distinct: bool,
    /// (cos qA, sin qA) por enlace en f64.
    pub ph: Vec<f64>,
    pub seq: [Vec<Sub>; 2],
    pub mat_in: [Vec<M4>; 2],
    pub mat_out: [M4; 2],
    /// 0 = masa uniforme (Dirac) o sin masa (Weyl); 1 = masa compleja de la cuerda (§2.18);
    /// 2 = Yukawa con el Higgs adjunto (§3.9, Jackiw–Rebbi).
    pub mass_kind: u32,
    pub mass_cell: Vec<f64>,
    /// Dirección del espín de los paquetes que se inyectan.
    pub spin_dir: [f64; 3],
    /// Centro de referencia para los centroides lineales (el último paquete inyectado).
    pub center: [f64; 3],
    pub defer_mirror: bool,
    /// Flujo neto de norma por eje en el último paso (Σ|P₊ψ|² − |P₋ψ|² de sus subpasos).
    pub flux: [f64; 3],
    /// El mismo flujo para cada parte (la velocidad exacta de cada paquete).
    pub part_flux: Vec<[f64; 3]>,
    pub track_flux: bool,
    /// Cuerda de Higgs: devanado, m₀ y ξ (§2.18); 0 = sin cuerda.
    pub string_winding: i32,
    pub string_m0: f64,
    pub string_xi: f64,
    pub string_cores: [[f64; 2]; 2],
    /// Los paquetes se construyen en el gauge con A ≈ 0 junto a r₀ (momento cinético k₀); sin
    /// esta compensación, un gaussiano en el gauge de Landau lleva el momento cinético −q·A(r₀).
    pub gauge_compensation: bool,
}

impl SpinorWalk {
    pub fn new() -> Self {
        SpinorWalk {
            active: false,
            ns: 4,
            weyl: false,
            fixed_order: false,
            variant: 0,
            slice: false,
            kz: 0.0,
            psi: Vec::new(),
            tmp: Vec::new(),
            parts: Vec::new(),
            part_centers: Vec::new(),
            parts_mode: false,
            distinct: false,
            ph: Vec::new(),
            seq: [Vec::new(), Vec::new()],
            mat_in: [Vec::new(), Vec::new()],
            mat_out: [m4_identity(), m4_identity()],
            mass_kind: 0,
            mass_cell: Vec::new(),
            spin_dir: [0.0, 0.0, 1.0],
            center: [0.0; 3],
            defer_mirror: false,
            flux: [0.0; 3],
            part_flux: Vec::new(),
            track_flux: false,
            string_winding: 0,
            string_m0: 0.6,
            string_xi: 1.5,
            string_cores: [[0.0; 2]; 2],
            gauge_compensation: true,
        }
    }

    /// ¿El orden de los subpasos cambia entre pasos pares e impares?
    pub fn alternating(&self) -> bool {
        if self.weyl {
            self.variant == 1
        } else {
            !self.fixed_order
        }
    }

    /// Pasos por período del operador de Floquet (1 o 2).
    pub fn period(&self) -> usize {
        if self.alternating() {
            2
        } else {
            1
        }
    }

    /// Precalcula las secuencias de subpasos y las matrices de cambio de base.
    pub fn rebuild(&mut self) {
        let ns = self.ns;
        let walk = |axis: usize| -> Sub {
            if ns == 4 {
                Sub { axis, sign: [1, -1, -1, 1] }
            } else {
                Sub { axis, sign: [1, -1, 0, 0] }
            }
        };
        let pump = |axis: usize, plus: bool| -> Sub {
            if plus {
                Sub { axis, sign: [1, 0, 0, 0] }
            } else {
                Sub { axis, sign: [0, -1, 0, 0] }
            }
        };
        let (even, odd): (Vec<Sub>, Vec<Sub>) = if self.weyl && self.variant == 2 {
            // U = U₁⁻·U₃⁻·U₂⁻·U₃⁺·U₁⁺·U₃⁻·U₂⁺·U₃⁺ (de derecha a izquierda)
            let s = vec![pump(2, true), pump(1, true), pump(2, false), pump(0, true), pump(2, true), pump(1, false), pump(2, false), pump(0, false)];
            (s.clone(), s)
        } else {
            let fwd = vec![walk(0), walk(1), walk(2)];
            let back = vec![walk(2), walk(1), walk(0)];
            if self.alternating() {
                (fwd, back)
            } else {
                (fwd.clone(), fwd)
            }
        };
        for (p, s) in [even, odd].into_iter().enumerate() {
            let mut prev = m4_identity();
            let mut mats = Vec::with_capacity(s.len());
            for sub in &s {
                let v = axis_basis(ns, sub.axis);
                mats.push(m4_mul(&m4_dag(&v), &prev));
                prev = v;
            }
            self.mat_out[p] = prev;
            self.mat_in[p] = mats;
            self.seq[p] = s;
        }
    }
}

/// Paso de masa uniforme M(m) = exp(−imβ) = diag(e^{−im}, e^{−im}, e^{im}, e^{im}).
pub fn mass_matrix(m: f64) -> M4 {
    let mut r = m4_zero();
    let (e1, e2) = (C::from_polar(1.0, -m), C::from_polar(1.0, m));
    r[0][0] = e1;
    r[1][1] = e1;
    r[2][2] = e2;
    r[3][3] = e2;
    r
}

/// Proyector P₊ = ½(1 − S/s) de un operador unitario W de dos bandas (S = (W − W†)/2i,
/// s² = Tr S²/n). Devuelve None si s < 10⁻⁹ o si S² no es proporcional a 1 (entonces hay que
/// diagonalizar).
pub fn band_projector_s(w: &M4, ns: usize, band: i32) -> Option<M4> {
    let wd = m4_dag(w);
    let mut s = m4_zero();
    for i in 0..ns {
        for j in 0..ns {
            s[i][j] = (w[i][j] - wd[i][j]) / cx(0.0, 2.0);
        }
    }
    let s2 = m4_mul(&s, &s);
    let tr: f64 = (0..ns).map(|i| s2[i][i].re).sum::<f64>() / ns as f64;
    if tr < 1e-18 {
        return None;
    }
    let mut dev = 0.0f64;
    for i in 0..ns {
        for j in 0..ns {
            let target = if i == j { tr } else { 0.0 };
            dev = dev.max((s2[i][j] - cx(target, 0.0)).norm());
        }
    }
    if dev > 1e-9 * tr.max(1e-12) {
        return None;
    }
    let sn = tr.sqrt();
    let sign = if band >= 0 { -1.0 } else { 1.0 };
    let mut p = m4_zero();
    for i in 0..ns {
        for j in 0..ns {
            p[i][j] = s[i][j] * (0.5 * sign / sn);
        }
        p[i][i] += cx(0.5, 0.0);
    }
    Some(p)
}

/// Proyector sobre la banda (cuasi-energía positiva si band = +1) por diagonalización.
pub fn band_projector_eig(w: &M4, ns: usize, band: i32) -> M4 {
    let mut a = vec![cx(0.0, 0.0); ns * ns];
    for i in 0..ns {
        for j in 0..ns {
            a[i * ns + j] = w[i][j];
        }
    }
    let e = eig(a, ns);
    let mut p = m4_zero();
    for (l, v) in e.values.iter().zip(e.vectors.iter()) {
        let om = -l.arg();
        let want = if band >= 0 { om > 0.0 && om < PI - 1e-12 } else { om < 0.0 };
        if !want {
            continue;
        }
        for i in 0..ns {
            for j in 0..ns {
                p[i][j] += v[i] * v[j].conj();
            }
        }
    }
    p
}

#[inline(always)]
fn ld<const NC: usize>(b: &[f64], o: usize) -> [C; 3] {
    let mut v = [cx(0.0, 0.0); 3];
    for a in 0..NC {
        v[a] = cx(b[o + 2 * a], b[o + 2 * a + 1]);
    }
    v
}

#[inline(always)]
fn st<const NC: usize>(b: &mut [f64], o: usize, v: &[C; 3]) {
    for a in 0..NC {
        b[o + 2 * a] = v[a].re;
        b[o + 2 * a + 1] = v[a].im;
    }
}

impl QuantumSpacetimeUniverse {
    /// Amplitudes complejas por celda del motor de espinores.
    #[inline(always)]
    pub(crate) fn sw_namp(&self) -> usize {
        self.sw.ns * self.ncol
    }

    /// Activa el motor con ns componentes (reserva los búferes).
    pub(crate) fn sw_activate(&mut self, ns: usize, weyl: bool) {
        self.sw.active = true;
        self.sw.ns = ns;
        self.sw.weyl = weyl;
        self.sw.slice = self.dim == 2;
        self.sw.rebuild();
        self.sw_alloc();
    }

    pub(crate) fn sw_alloc(&mut self) {
        let len = self.n * 2 * self.sw.ns * self.ncol.max(1);
        self.sw.psi = vec![0.0; len];
        self.sw.tmp = vec![0.0; len];
        self.sw.parts.clear();
        self.sw.part_centers.clear();
        self.sw.mass_kind = 0;
        self.sw.mass_cell.clear();
        if self.sw.ph.len() != 2 * self.dim * self.n {
            self.sw.ph = vec![0.0; 2 * self.dim * self.n];
        }
    }

    pub(crate) fn sw_clear(&mut self) {
        for v in self.sw.psi.iter_mut() {
            *v = 0.0;
        }
        self.sw.parts.clear();
        self.sw.part_centers.clear();
        self.sw.flux = [0.0; 3];
    }

    /// Copia ψ (f64) en `buffer_read` (f32) para el render y los análisis genéricos.
    pub(crate) fn sw_mirror(&mut self) {
        let len = self.sw.psi.len().min(self.buffer_read.len());
        for k in 0..len {
            self.buffer_read[k] = self.sw.psi[k] as f32;
        }
    }

    /// Densidad de carga de una celda (f64): |ψ|², o Σ_p |ψ_p|² con partes distinguibles.
    #[inline]
    pub(crate) fn sw_density(&self, i: usize) -> f64 {
        let st = 2 * self.sw_namp();
        if self.sw.distinct && !self.sw.parts.is_empty() {
            return self.sw.parts.iter().map(|p| p[i * st..(i + 1) * st].iter().map(|v| v * v).sum::<f64>()).sum();
        }
        self.sw.psi[i * st..(i + 1) * st].iter().map(|v| v * v).sum()
    }

    /// ρ^c(r) = Σ_p Σ_k ψ_pk†·T^c·ψ_pk con partes distinguibles.
    pub(crate) fn sw_colour_density_parts(&self, i: usize, out: &mut [f64]) {
        let ng = self.ym.ng;
        let nco = self.ncol;
        let st = 2 * self.sw.ns * nco;
        for v in out.iter_mut().take(ng) {
            *v = 0.0;
        }
        let mut tmp = [0.0f64; 8];
        for p in &self.sw.parts {
            for k in 0..self.sw.ns {
                let o = i * st + 2 * k * nco;
                let mut v = [cx(0.0, 0.0); 3];
                for a in 0..nco {
                    v[a] = cx(p[o + 2 * a], p[o + 2 * a + 1]);
                }
                self.ym.alg.charge(&v, &mut tmp);
                for c in 0..ng {
                    out[c] += tmp[c];
                }
            }
        }
    }

    /// Vector de color de la componente de espín k en la celda i.
    #[inline]
    pub(crate) fn sw_colour_vec(&self, i: usize, k: usize) -> [C; 3] {
        let nco = self.ncol;
        let b = i * 2 * self.sw.ns * nco + 2 * k * nco;
        let mut v = [cx(0.0, 0.0); 3];
        for a in 0..nco {
            v[a] = cx(self.sw.psi[b + 2 * a], self.sw.psi[b + 2 * a + 1]);
        }
        v
    }

    pub(crate) fn sw_refresh_phases(&mut self) {
        let q = self.em.q;
        let links = self.dim * self.n;
        if self.sw.ph.len() != 2 * links {
            self.sw.ph = vec![0.0; 2 * links];
        }
        for l in 0..links {
            let (s, c) = (q * self.em.a[l]).sin_cos();
            self.sw.ph[2 * l] = c;
            self.sw.ph[2 * l + 1] = s;
        }
    }

    /// Un paso del escenario Dirac 3D o Weyl.
    pub(crate) fn spinor_step(&mut self) {
        match (self.sw.ns, self.ncol) {
            (4, 1) => self.spinor_step_impl::<4, 1>(),
            (4, 2) => self.spinor_step_impl::<4, 2>(),
            (4, 3) => self.spinor_step_impl::<4, 3>(),
            (2, 1) => self.spinor_step_impl::<2, 1>(),
            (2, 2) => self.spinor_step_impl::<2, 2>(),
            _ => self.spinor_step_impl::<2, 3>(),
        }
        if !self.sw.defer_mirror {
            self.sw_mirror();
        }
    }

    fn spinor_step_impl<const NS: usize, const NC: usize>(&mut self) {
        let parity = if self.sw.alternating() { (self.t & 1) as usize } else { 0 };
        let gauge = self.em.mode != 0;
        let dynamic = self.em.mode == 2;
        if gauge && NC == 1 {
            self.sw_refresh_phases();
        }
        let mut psi = std::mem::take(&mut self.sw.psi);
        let mut tmp = std::mem::take(&mut self.sw.tmp);
        let mut parts = std::mem::take(&mut self.sw.parts);
        let mut jbuf = if dynamic {
            if NC == 1 {
                std::mem::take(&mut self.em.j)
            } else {
                std::mem::take(&mut self.ym.j)
            }
        } else {
            Vec::new()
        };
        for v in jbuf.iter_mut() {
            *v = 0.0;
        }
        let track = self.sw.track_flux;
        let mut flux = [0.0f64; 3];
        let mut pflux = std::mem::take(&mut self.sw.part_flux);
        if track {
            pflux.resize(parts.len(), [0.0; 3]);
            pflux.fill([0.0; 3]);
        } else {
            pflux.clear();
        }
        let nsub = self.sw.seq[parity].len();
        let distinct = self.sw.distinct && !parts.is_empty();
        for s in 0..nsub {
            let sub = self.sw.seq[parity][s];
            let m = self.sw.mat_in[parity][s];
            let cur = dynamic && !(self.sw.slice && sub.axis == 2);
            self.sw_apply_matrix::<NS, NC>(&mut psi, &m);
            if cur && !distinct {
                self.sw_currents::<NS, NC>(&psi, &sub, &mut jbuf);
            }
            if track {
                flux[sub.axis] += self.sw_sub_flux::<NS, NC>(&psi, &sub);
            }
            self.sw_shift::<NS, NC>(&psi, &mut tmp, &sub, gauge);
            std::mem::swap(&mut psi, &mut tmp);
            for (ip, p) in parts.iter_mut().enumerate() {
                self.sw_apply_matrix::<NS, NC>(p, &m);
                if cur && distinct {
                    self.sw_currents::<NS, NC>(p, &sub, &mut jbuf);
                }
                if track {
                    pflux[ip][sub.axis] += self.sw_sub_flux::<NS, NC>(p, &sub);
                }
                self.sw_shift::<NS, NC>(p, &mut tmp, &sub, gauge);
                std::mem::swap(p, &mut tmp);
            }
        }
        self.sw_finish::<NS, NC>(&mut psi, parity);
        for p in parts.iter_mut() {
            self.sw_finish::<NS, NC>(p, parity);
        }
        if dynamic {
            if NC == 1 {
                self.em.j = jbuf;
            } else {
                self.ym.j = jbuf;
            }
        }
        self.sw.psi = psi;
        self.sw.tmp = tmp;
        self.sw.parts = parts;
        if track {
            self.sw.flux = flux;
            self.sw.part_flux = pflux;
        }
        self.t = self.t.wrapping_add(1);
    }

    /// Paso inverso (F⁻¹ = F†) sobre un búfer, sin campo dinámico, masa por celda, potencial ni
    /// muros: deshace la vuelta a la base estándar y la masa uniforme, y después cada subpaso en
    /// orden inverso (el desplazamiento inverso es el mismo con los signos cambiados).
    pub(crate) fn sw_step_back_buffer(&self, psi: &mut Vec<f64>, tmp: &mut Vec<f64>, parity: usize) {
        match (self.sw.ns, self.ncol) {
            (4, 1) => self.sw_step_back_impl::<4, 1>(psi, tmp, parity),
            (2, 1) => self.sw_step_back_impl::<2, 1>(psi, tmp, parity),
            (4, 2) => self.sw_step_back_impl::<4, 2>(psi, tmp, parity),
            (4, 3) => self.sw_step_back_impl::<4, 3>(psi, tmp, parity),
            (2, 2) => self.sw_step_back_impl::<2, 2>(psi, tmp, parity),
            _ => self.sw_step_back_impl::<2, 3>(psi, tmp, parity),
        }
    }

    fn sw_step_back_impl<const NS: usize, const NC: usize>(&self, psi: &mut Vec<f64>, tmp: &mut Vec<f64>, parity: usize) {
        let gauge = self.em.mode != 0;
        let vout = self.sw.mat_out[parity];
        let fin = if !self.sw.weyl && self.sw.mass_kind == 0 { m4_mul(&mass_matrix(self.mass as f64), &vout) } else { vout };
        self.sw_apply_matrix::<NS, NC>(psi, &m4_dag(&fin));
        for s in (0..self.sw.seq[parity].len()).rev() {
            let mut sub = self.sw.seq[parity][s];
            for c in 0..4 {
                sub.sign[c] = -sub.sign[c];
            }
            self.sw_shift::<NS, NC>(psi, tmp, &sub, gauge);
            std::mem::swap(psi, tmp);
            self.sw_apply_matrix::<NS, NC>(psi, &m4_dag(&self.sw.mat_in[parity][s]));
        }
    }

    /// Un paso hacia delante sobre un búfer aparte (sin corrientes ni campo dinámico).
    pub(crate) fn sw_step_buffer(&self, psi: &mut Vec<f64>, tmp: &mut Vec<f64>, parity: usize) {
        match (self.sw.ns, self.ncol) {
            (4, 1) => self.sw_step_buffer_impl::<4, 1>(psi, tmp, parity),
            (2, 1) => self.sw_step_buffer_impl::<2, 1>(psi, tmp, parity),
            (4, 2) => self.sw_step_buffer_impl::<4, 2>(psi, tmp, parity),
            (4, 3) => self.sw_step_buffer_impl::<4, 3>(psi, tmp, parity),
            (2, 2) => self.sw_step_buffer_impl::<2, 2>(psi, tmp, parity),
            _ => self.sw_step_buffer_impl::<2, 3>(psi, tmp, parity),
        }
    }

    fn sw_step_buffer_impl<const NS: usize, const NC: usize>(&self, psi: &mut Vec<f64>, tmp: &mut Vec<f64>, parity: usize) {
        let gauge = self.em.mode != 0;
        for s in 0..self.sw.seq[parity].len() {
            let sub = self.sw.seq[parity][s];
            self.sw_apply_matrix::<NS, NC>(psi, &self.sw.mat_in[parity][s]);
            self.sw_shift::<NS, NC>(psi, tmp, &sub, gauge);
            std::mem::swap(psi, tmp);
        }
        self.sw_finish::<NS, NC>(psi, parity);
    }

    #[inline(always)]
    fn sw_apply_matrix<const NS: usize, const NC: usize>(&self, psi: &mut [f64], m: &M4) {
        let st_ = 2 * NS * NC;
        for i in 0..self.n {
            let o = i * st_;
            for a in 0..NC {
                let mut v = [cx(0.0, 0.0); NS];
                for k in 0..NS {
                    v[k] = cx(psi[o + 2 * (k * NC + a)], psi[o + 2 * (k * NC + a) + 1]);
                }
                for k in 0..NS {
                    let mut s = cx(0.0, 0.0);
                    for j in 0..NS {
                        s += m[k][j] * v[j];
                    }
                    psi[o + 2 * (k * NC + a)] = s.re;
                    psi[o + 2 * (k * NC + a) + 1] = s.im;
                }
            }
        }
    }

    /// Corriente del eje en la base propia, antes del salto: U(1), J = Σ₊|p(r)|² − Σ₋|p(r+â)|²;
    /// SU(N), J^c = Σ₊ p†T^c p − Σ₋ w†T^c w con w = U_a(r)†·p₋(r+â).
    fn sw_currents<const NS: usize, const NC: usize>(&self, psi: &[f64], sub: &Sub, j: &mut [f64]) {
        let n = self.n;
        let a = sub.axis;
        let st_ = 2 * NS * NC;
        if NC == 1 {
            for i in 0..n {
                let ip = self.em.nbp[a * n + i] as usize;
                let mut s = 0.0;
                for k in 0..NS {
                    match sub.sign[k] {
                        1 => {
                            let o = i * st_ + 2 * k;
                            s += psi[o] * psi[o] + psi[o + 1] * psi[o + 1];
                        }
                        -1 => {
                            let o = ip * st_ + 2 * k;
                            s -= psi[o] * psi[o] + psi[o + 1] * psi[o + 1];
                        }
                        _ => {}
                    }
                }
                j[a * n + i] += s;
            }
        } else {
            let ng = self.ym.ng;
            let mut t = [0.0f64; 8];
            for i in 0..n {
                let ip = self.em.nbp[a * n + i] as usize;
                let u = self.ym.link(a * n + i);
                let o = (a * n + i) * ng;
                for k in 0..NS {
                    match sub.sign[k] {
                        1 => {
                            let v = ld::<NC>(psi, i * st_ + 2 * k * NC);
                            self.ym.alg.charge(&v, &mut t);
                            for c in 0..ng {
                                j[o + c] += t[c];
                            }
                        }
                        -1 => {
                            let v = u.apply_dag(&ld::<NC>(psi, ip * st_ + 2 * k * NC));
                            self.ym.alg.charge(&v, &mut t);
                            for c in 0..ng {
                                j[o + c] -= t[c];
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Σ_r (Σ₊|p|² − Σ₋|p|²): la norma que el subpaso mueve en +a menos la que mueve en −a.
    fn sw_sub_flux<const NS: usize, const NC: usize>(&self, psi: &[f64], sub: &Sub) -> f64 {
        let st_ = 2 * NS * NC;
        let mut f = 0.0;
        for i in 0..self.n {
            for k in 0..NS {
                let sg = sub.sign[k] as f64;
                if sg == 0.0 {
                    continue;
                }
                let o = i * st_ + 2 * k * NC;
                let mut e = 0.0;
                for c in 0..2 * NC {
                    e += psi[o + c] * psi[o + c];
                }
                f += sg * e;
            }
        }
        f
    }

    /// Desplazamiento *gather* del subpaso, con los enlaces.
    fn sw_shift<const NS: usize, const NC: usize>(&self, src: &[f64], dst: &mut [f64], sub: &Sub, gauge: bool) {
        let n = self.n;
        let a = sub.axis;
        let st_ = 2 * NS * NC;
        if self.sw.slice && a == 2 {
            // 2D con k_z fijo: e^{−ik_z} para lo que avanza y e^{+ik_z} para lo que retrocede
            let (sn, cs) = self.sw.kz.sin_cos();
            for i in 0..n {
                let o = i * st_;
                for k in 0..NS {
                    let sg = sub.sign[k];
                    for c in 0..NC {
                        let p = o + 2 * (k * NC + c);
                        let (re, im) = (src[p], src[p + 1]);
                        match sg {
                            1 => {
                                dst[p] = re * cs + im * sn;
                                dst[p + 1] = im * cs - re * sn;
                            }
                            -1 => {
                                dst[p] = re * cs - im * sn;
                                dst[p + 1] = im * cs + re * sn;
                            }
                            _ => {
                                dst[p] = re;
                                dst[p + 1] = im;
                            }
                        }
                    }
                }
            }
            return;
        }
        let nbm = &self.em.nbm[a * n..(a + 1) * n];
        let nbp = &self.em.nbp[a * n..(a + 1) * n];
        for i in 0..n {
            let im_ = nbm[i] as usize;
            let ip = nbp[i] as usize;
            let o = i * st_;
            for k in 0..NS {
                let sg = sub.sign[k];
                let ko = 2 * k * NC;
                match sg {
                    1 => {
                        // de r − â con U_a(r − â)
                        let so = im_ * st_ + ko;
                        if !gauge {
                            for c in 0..2 * NC {
                                dst[o + ko + c] = src[so + c];
                            }
                        } else if NC == 1 {
                            let l = a * n + im_;
                            let (c, s) = (self.sw.ph[2 * l], self.sw.ph[2 * l + 1]);
                            let (re, imv) = (src[so], src[so + 1]);
                            dst[o + ko] = re * c - imv * s;
                            dst[o + ko + 1] = re * s + imv * c;
                        } else {
                            let v = self.ym.link(a * n + im_).apply(&ld::<NC>(src, so));
                            st::<NC>(dst, o + ko, &v);
                        }
                    }
                    -1 => {
                        // de r + â con U_a(r)†
                        let so = ip * st_ + ko;
                        if !gauge {
                            for c in 0..2 * NC {
                                dst[o + ko + c] = src[so + c];
                            }
                        } else if NC == 1 {
                            let l = a * n + i;
                            let (c, s) = (self.sw.ph[2 * l], -self.sw.ph[2 * l + 1]);
                            let (re, imv) = (src[so], src[so + 1]);
                            dst[o + ko] = re * c - imv * s;
                            dst[o + ko + 1] = re * s + imv * c;
                        } else {
                            let v = self.ym.link(a * n + i).apply_dag(&ld::<NC>(src, so));
                            st::<NC>(dst, o + ko, &v);
                        }
                    }
                    _ => {
                        for c in 0..2 * NC {
                            dst[o + ko + c] = src[o + ko + c];
                        }
                    }
                }
            }
        }
    }

    /// Vuelta a la base estándar, masa (uniforme o por celda), potencial y muros.
    fn sw_finish<const NS: usize, const NC: usize>(&self, psi: &mut [f64], parity: usize) {
        let vout = self.sw.mat_out[parity];
        let uniform_mass = !self.sw.weyl && self.sw.mass_kind == 0;
        let fin = if uniform_mass { m4_mul(&mass_matrix(self.mass as f64), &vout) } else { vout };
        self.sw_apply_matrix::<NS, NC>(psi, &fin);
        let st_ = 2 * NS * NC;
        if NS == 4 && self.sw.mass_kind == 1 {
            // M = cos μ − i·sin μ·(cos θ·β + sin θ·B₂): [[c − ia, −b], [b, c + ia]] en el índice τ
            for i in 0..self.n {
                let (c, a, b) = (self.sw.mass_cell[3 * i], self.sw.mass_cell[3 * i + 1], self.sw.mass_cell[3 * i + 2]);
                let o = i * st_;
                for sgm in 0..2 {
                    for col in 0..NC {
                        let p0 = o + 2 * (sgm * NC + col);
                        let p1 = o + 2 * ((2 + sgm) * NC + col);
                        let u = cx(psi[p0], psi[p0 + 1]);
                        let l = cx(psi[p1], psi[p1 + 1]);
                        let nu = cx(c, -a) * u - l * b;
                        let nl = u * b + cx(c, a) * l;
                        psi[p0] = nu.re;
                        psi[p0 + 1] = nu.im;
                        psi[p1] = nl.re;
                        psi[p1 + 1] = nl.im;
                    }
                }
            }
        } else if NS == 4 && NC == 2 && self.sw.mass_kind == 2 {
            // M = cos μ − i·sin μ·β⊗(n·τ), con (c, s·n) por celda
            for i in 0..self.n {
                let c = self.sw.mass_cell[4 * i];
                let (nx, ny, nz) = (self.sw.mass_cell[4 * i + 1], self.sw.mass_cell[4 * i + 2], self.sw.mass_cell[4 * i + 3]);
                let o = i * st_;
                for k in 0..4 {
                    let b = if k < 2 { 1.0 } else { -1.0 };
                    let p = o + 2 * k * NC;
                    let v0 = cx(psi[p], psi[p + 1]);
                    let v1 = cx(psi[p + 2], psi[p + 3]);
                    // (n·τ)·v
                    let t0 = v0 * nz + v1 * cx(nx, -ny);
                    let t1 = v0 * cx(nx, ny) - v1 * nz;
                    let f = cx(0.0, -b);
                    let r0 = v0 * c + f * t0;
                    let r1 = v1 * c + f * t1;
                    psi[p] = r0.re;
                    psi[p + 1] = r0.im;
                    psi[p + 2] = r1.re;
                    psi[p + 3] = r1.im;
                }
            }
        }
        if self.has_potential {
            for i in 0..self.n {
                let v = self.potential[i] as f64;
                if v == 0.0 {
                    continue;
                }
                let (s, c) = (-v).sin_cos();
                let o = i * st_;
                for k in 0..NS * NC {
                    let (re, im) = (psi[o + 2 * k], psi[o + 2 * k + 1]);
                    psi[o + 2 * k] = re * c - im * s;
                    psi[o + 2 * k + 1] = re * s + im * c;
                }
            }
        }
        if self.has_walls {
            for i in 0..self.n {
                if self.wall[i] != 0 {
                    for k in 0..st_ {
                        psi[i * st_ + k] = 0.0;
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------- operador en Fourier

    /// Operador de un paso en Fourier (sin campo y con la masa uniforme), de la paridad dada.
    pub(crate) fn sw_step_matrix(&self, parity: usize, k: [f64; 3]) -> M4 {
        let ns = self.sw.ns;
        let mut u = m4_identity();
        for (s, sub) in self.sw.seq[parity].iter().enumerate() {
            u = m4_mul(&self.sw.mat_in[parity][s], &u);
            let kk = k[sub.axis];
            let mut d = m4_identity();
            for c in 0..ns {
                d[c][c] = match sub.sign[c] {
                    1 => C::from_polar(1.0, -kk),
                    -1 => C::from_polar(1.0, kk),
                    _ => cx(1.0, 0.0),
                };
            }
            u = m4_mul(&d, &u);
        }
        let vout = self.sw.mat_out[parity];
        let fin = if !self.sw.weyl && self.sw.mass_kind == 0 { m4_mul(&mass_matrix(self.mass as f64), &vout) } else { vout };
        m4_mul(&fin, &u)
    }

    /// Operador de un período (1 o 2 pasos) que empieza en el paso de paridad `start`.
    pub(crate) fn sw_period_matrix(&self, start: usize, k: [f64; 3]) -> M4 {
        if self.sw.alternating() {
            let a = self.sw_step_matrix(start & 1, k);
            let b = self.sw_step_matrix((start + 1) & 1, k);
            m4_mul(&b, &a)
        } else {
            self.sw_step_matrix(0, k)
        }
    }

    /// Proyector de banda del operador de un período en k.
    pub(crate) fn sw_projector(&self, start: usize, k: [f64; 3], band: i32) -> M4 {
        let w = self.sw_period_matrix(start, k);
        let ns = self.sw.ns;
        match band_projector_s(&w, ns, band) {
            Some(p) => p,
            None => {
                // s ≈ 0 (nodo) o S² no proporcional a 1 (orden fijo): diagonalizar
                let wd = m4_dag(&w);
                let mut tr = 0.0;
                for i in 0..ns {
                    for j in 0..ns {
                        tr += ((w[i][j] - wd[i][j]) / cx(0.0, 2.0)).norm_sqr();
                    }
                }
                if tr < 1e-18 {
                    return m4_identity();
                }
                band_projector_eig(&w, ns, band)
            }
        }
    }

    /// Cuasi-energía de la banda +1 por paso en k (con el orden alternado, la del operador de dos
    /// pasos dividida entre 2; los dos niveles de espín son degenerados).
    pub(crate) fn sw_omega(&self, k: [f64; 3]) -> f64 {
        let ns = self.sw.ns as f64;
        let w = self.sw_period_matrix(0, k);
        let tr: C = (0..self.sw.ns).map(|i| w[i][i]).sum();
        let p = self.sw.period() as f64;
        ((tr.re / ns).clamp(-1.0, 1.0)).acos() / p
    }

    /// Las cuasi-energías por paso de los estados de la banda +1 en k (orden fijo: dos niveles
    /// de espín distintos), por diagonalización.
    pub(crate) fn sw_band_levels(&self, k: [f64; 3]) -> Vec<f64> {
        let ns = self.sw.ns;
        let w = self.sw_period_matrix(0, k);
        let mut a = vec![cx(0.0, 0.0); ns * ns];
        for i in 0..ns {
            for j in 0..ns {
                a[i * ns + j] = w[i][j];
            }
        }
        let p = self.sw.period() as f64;
        let mut v: Vec<f64> = eig(a, ns).values.iter().map(|l| -l.arg() / p).filter(|o| *o > 0.0).collect();
        v.sort_by(f64::total_cmp);
        v
    }

    /// Espinor de la banda en k₀ con el espín pedido (Dirac: P·(ξ, 0) o P·(0, ξ); Weyl: P·ξ).
    pub(crate) fn sw_band_spinor(&self, k0: [f64; 3], band: i32) -> [C; 4] {
        let sd = self.sw.spin_dir;
        let nrm = (sd[0] * sd[0] + sd[1] * sd[1] + sd[2] * sd[2]).sqrt().max(1e-12);
        let th = (sd[2] / nrm).clamp(-1.0, 1.0).acos();
        let ph = sd[1].atan2(sd[0]);
        let xi = [cx((0.5 * th).cos(), 0.0), C::from_polar((0.5 * th).sin(), ph)];
        let mut v = [cx(0.0, 0.0); 4];
        if self.sw.ns == 4 {
            let off = if band >= 0 { 0 } else { 2 };
            v[off] = xi[0];
            v[off + 1] = xi[1];
        } else {
            v[0] = xi[0];
            v[1] = xi[1];
        }
        let p = self.sw_projector((self.t & 1) as usize, k0, band);
        let mut u = m4_apply(&p, &v);
        let mut nn: f64 = u.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if nn < 1e-6 {
            // El espín pedido no tiene componente en la banda: probar vectores de la base ortogonal
            for cand in 0..self.sw.ns {
                let mut alt = [cx(0.0, 0.0); 4];
                alt[cand] = cx(1.0, 0.0);
                let trial = m4_apply(&p, &alt);
                let tn: f64 = trial.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if tn > nn {
                    u = trial;
                    nn = tn;
                    if nn >= 1e-6 {
                        break;
                    }
                }
            }
        }
        let nn = nn.max(1e-300);
        for c in u.iter_mut() {
            *c /= nn;
        }
        u
    }

    /// Paquete gaussiano proyectado sobre su banda (§4.3): envolvente con e^{ik₀·r}, espinor de la
    /// banda en k₀ (⊗ color), FFT de cada componente por P(k) y vuelta. Devuelve el paquete (norma
    /// `amount`) en un búfer aparte.
    pub(crate) fn sw_build_packet(&self, c0: [f64; 3], sigma: [f64; 3], k0: [f64; 3], band: i32, colour: [C; 3], project: bool, amount: f64) -> Vec<f64> {
        let ns = self.sw.ns;
        let nco = self.ncol;
        let namp = ns * nco;
        let n = self.n;
        let spinor = self.sw_band_spinor(k0, band);
        let cn: f64 = colour.iter().take(nco).map(|v| v.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let mut comps: Vec<Vec<C>> = vec![vec![cx(0.0, 0.0); n]; namp];
        for i in 0..n {
            let (x, y, z) = self.coords(i);
            let p = [x as f64, y as f64, z as f64];
            let mut arg = 0.0;
            let mut ph = 0.0;
            for a in 0..self.dim {
                let dx = min_image(p[a] - c0[a], dims[a]);
                arg += dx * dx / (2.0 * sigma[a] * sigma[a]);
                ph += k0[a] * dx;
            }
            if arg > 40.0 {
                continue;
            }
            let e = C::from_polar((-arg).exp(), ph);
            for k in 0..ns {
                for a in 0..nco {
                    comps[k * nco + a][i] = e * spinor[k] * colour[a] / cn;
                }
            }
        }
        if project {
            let mut fft = FftN::<f64>::new(self.w, self.h, self.d);
            for c in comps.iter_mut() {
                fft.forward(c);
            }
            let start = (self.t & 1) as usize;
            for i in 0..n {
                let (x, y, z) = self.coords(i);
                let kz = if self.dim == 3 { wavenumber(z, self.d) } else { self.sw.kz };
                let k = [wavenumber(x, self.w), wavenumber(y, self.h), kz];
                let w = self.sw_period_matrix(start, k);
                let p = match band_projector_s(&w, ns, band) {
                    Some(p) => p,
                    None => {
                        let wd = m4_dag(&w);
                        let mut tr = 0.0;
                        for a in 0..ns {
                            for b in 0..ns {
                                tr += ((w[a][b] - wd[a][b]) / cx(0.0, 2.0)).norm_sqr();
                            }
                        }
                        if tr < 1e-18 {
                            continue;
                        }
                        band_projector_eig(&w, ns, band)
                    }
                };
                for a in 0..nco {
                    let mut v = [cx(0.0, 0.0); 4];
                    for k in 0..ns {
                        v[k] = comps[k * nco + a][i];
                    }
                    let r = m4_apply(&p, &v);
                    for k in 0..ns {
                        comps[k * nco + a][i] = r[k];
                    }
                }
            }
            for c in comps.iter_mut() {
                fft.inverse(c);
            }
        }
        let norm: f64 = comps.iter().map(|c| c.iter().map(|v| v.norm_sqr()).sum::<f64>()).sum();
        let s = if norm > 0.0 { (amount / norm).sqrt() } else { 0.0 };
        // Con un campo U(1), el paquete se construye en el gauge donde A ≈ 0 junto a r₀ y se lleva
        // al gauge del campo con e^{iq·A(r₀)·(r − r₀)}: así su momento cinético es k₀ (§2.10).
        let qa = self.sw_packet_gauge_phase(c0);
        let mut out = vec![0.0; n * 2 * namp];
        for i in 0..n {
            let ph = if qa != [0.0; 3] {
                let (x, y, z) = self.coords(i);
                let p = [x as f64, y as f64, z as f64];
                let mut a = 0.0;
                for ax in 0..self.dim {
                    a += qa[ax] * min_image(p[ax] - c0[ax], dims[ax]);
                }
                C::from_polar(s, a)
            } else {
                cx(s, 0.0)
            };
            for k in 0..namp {
                let v = comps[k][i] * ph;
                out[i * 2 * namp + 2 * k] = v.re;
                out[i * 2 * namp + 2 * k + 1] = v.im;
            }
        }
        out
    }

    /// q·A(r₀) promediado en los dos enlaces de cada eje que tocan la celda de r₀ (U(1)).
    fn sw_packet_gauge_phase(&self, c0: [f64; 3]) -> [f64; 3] {
        if self.em.mode == 0 || self.ncol > 1 || !self.sw.gauge_compensation {
            return [0.0; 3];
        }
        let c = self.wrap_cell(c0[0].round() as i32, c0[1].round() as i32, c0[2].round() as i32);
        let n = self.n;
        let q = self.em.q;
        let mut r = [0.0; 3];
        for a in 0..self.dim {
            r[a] = q * 0.5 * (self.em.a[a * n + c] + self.em.a[a * n + self.nb(c, a, -1)]);
        }
        r
    }

    /// Potencial φ de −∇²_red φ = src (media nula) en el toro, por FFT.
    pub(crate) fn poisson_potential(&self, src: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mean = src.iter().sum::<f64>() / n as f64;
        let mut buf: Vec<C> = src.iter().map(|v| cx(v - mean, 0.0)).collect();
        let mut fft = FftN::<f64>::new(self.w, self.h, self.d);
        fft.forward(&mut buf);
        for i in 0..n {
            let (x, y, z) = self.coords(i);
            let mut lam = 2.0 * (1.0 - (2.0 * PI * x as f64 / self.w as f64).cos()) + 2.0 * (1.0 - (2.0 * PI * y as f64 / self.h as f64).cos());
            if self.dim == 3 {
                lam += 2.0 * (1.0 - (2.0 * PI * z as f64 / self.d as f64).cos());
            }
            buf[i] = if lam > 1e-12 { buf[i] / lam } else { cx(0.0, 0.0) };
        }
        fft.inverse(&mut buf);
        buf.iter().map(|c| c.re).collect()
    }

    /// ⟨E_a⟩ del potencial φ (E en los enlaces, −(φ(r + â) − φ(r)), promediado en las dos caras de
    /// cada celda) sobre la densidad `rho`.
    pub(crate) fn field_mean_over(&self, phi: &[f64], rho: &[f64], axis: usize) -> f64 {
        let tot: f64 = rho.iter().sum();
        let mut acc = 0.0;
        for i in 0..self.n {
            if rho[i] == 0.0 {
                continue;
            }
            let (ip, im) = (self.nb(i, axis, 1), self.nb(i, axis, -1));
            let e_here = -(phi[ip] - phi[i]);
            let e_back = -(phi[i] - phi[im]);
            acc += rho[i] * 0.5 * (e_here + e_back);
        }
        acc / tot.max(1e-300)
    }

    pub(crate) fn buffer_density(&self, b: &[f64]) -> Vec<f64> {
        let st = 2 * self.sw_namp();
        (0..self.n).map(|i| b[i * st..(i + 1) * st].iter().map(|v| v * v).sum()).collect()
    }

    /// Suma un búfer al estado (y, en el modo por partes, lo guarda como parte propia).
    pub(crate) fn sw_add_buffer(&mut self, b: Vec<f64>, center: [f64; 3]) {
        for (p, v) in self.sw.psi.iter_mut().zip(b.iter()) {
            *p += *v;
        }
        if self.sw.parts_mode {
            self.sw.parts.push(b);
            self.sw.part_centers.push(center);
        }
        self.sw.center = center;
        self.sw_mirror();
    }

    /// Centroide lineal (relativo a `center`, con imagen mínima) de un búfer.
    pub(crate) fn sw_buffer_centroid(&self, b: &[f64], center: [f64; 3]) -> [f64; 4] {
        let st_ = 2 * self.sw_namp();
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let mut acc = [0.0f64; 3];
        let mut tot = 0.0;
        for i in 0..self.n {
            let e: f64 = b[i * st_..(i + 1) * st_].iter().map(|v| v * v).sum();
            if e == 0.0 {
                continue;
            }
            let (x, y, z) = self.coords(i);
            let p = [x as f64, y as f64, z as f64];
            for a in 0..3 {
                acc[a] += e * min_image(p[a] - center[a], dims[a]);
            }
            tot += e;
        }
        if tot <= 0.0 {
            return [center[0], center[1], center[2], 0.0];
        }
        [center[0] + acc[0] / tot, center[1] + acc[1] / tot, center[2] + acc[2] / tot, tot]
    }

    /// ⟨Σ_a⟩ (Dirac, Σ = 1⊗σ) o ⟨σ_a⟩ (Weyl) de un búfer, sin normalizar, y su norma.
    pub(crate) fn sw_buffer_spin(&self, b: &[f64]) -> [f64; 4] {
        let ns = self.sw.ns;
        let nco = self.ncol;
        let st_ = 2 * ns * nco;
        let mut s = [0.0f64; 4];
        for i in 0..self.n {
            let o = i * st_;
            for blk in 0..ns / 2 {
                for a in 0..nco {
                    let u = cx(b[o + 2 * ((2 * blk) * nco + a)], b[o + 2 * ((2 * blk) * nco + a) + 1]);
                    let d = cx(b[o + 2 * ((2 * blk + 1) * nco + a)], b[o + 2 * ((2 * blk + 1) * nco + a) + 1]);
                    let ud = u.conj() * d;
                    s[0] += 2.0 * ud.re;
                    s[1] += 2.0 * ud.im;
                    s[2] += u.norm_sqr() - d.norm_sqr();
                    s[3] += u.norm_sqr() + d.norm_sqr();
                }
            }
        }
        s
    }

    /// Fracción de la norma en la banda +1 (proyector de la FFT, supone A = 0 o pequeño).
    pub(crate) fn sw_band_fraction(&self, b: &[f64]) -> f64 {
        let ns = self.sw.ns;
        let nco = self.ncol;
        let namp = ns * nco;
        let n = self.n;
        let mut comps: Vec<Vec<C>> = (0..namp).map(|k| (0..n).map(|i| cx(b[i * 2 * namp + 2 * k], b[i * 2 * namp + 2 * k + 1])).collect()).collect();
        let mut fft = FftN::<f64>::new(self.w, self.h, self.d);
        for c in comps.iter_mut() {
            fft.forward(c);
        }
        let start = (self.t & 1) as usize;
        let (mut plus, mut tot) = (0.0, 0.0);
        for i in 0..n {
            let (x, y, z) = self.coords(i);
            let kz = if self.dim == 3 { wavenumber(z, self.d) } else { self.sw.kz };
            let k = [wavenumber(x, self.w), wavenumber(y, self.h), kz];
            let p = self.sw_projector(start, k, 1);
            for a in 0..nco {
                let mut v = [cx(0.0, 0.0); 4];
                for kk in 0..ns {
                    v[kk] = comps[kk * nco + a][i];
                }
                let r = m4_apply(&p, &v);
                for kk in 0..ns {
                    plus += (v[kk].conj() * r[kk]).re;
                    tot += v[kk].norm_sqr();
                }
            }
        }
        if tot > 0.0 {
            plus / tot
        } else {
            0.0
        }
    }

    /// ⟨∂²ω/∂k_a²⟩ promediado sobre el espectro de un búfer (la inercia de un paquete, §2.16).
    pub(crate) fn sw_mean_curvature(&self, b: &[f64], axis: usize) -> f64 {
        let ns = self.sw.ns;
        let nco = self.ncol;
        let namp = ns * nco;
        let n = self.n;
        let mut fft = FftN::<f64>::new(self.w, self.h, self.d);
        let mut weight = vec![0.0f64; n];
        for k in 0..namp {
            let mut c: Vec<C> = (0..n).map(|i| cx(b[i * 2 * namp + 2 * k], b[i * 2 * namp + 2 * k + 1])).collect();
            fft.forward(&mut c);
            for i in 0..n {
                weight[i] += c[i].norm_sqr();
            }
        }
        let tot: f64 = weight.iter().sum();
        let h = 1e-3;
        let mut acc = 0.0;
        for i in 0..n {
            if weight[i] < 1e-14 * tot {
                continue;
            }
            let (x, y, z) = self.coords(i);
            let kz = if self.dim == 3 { wavenumber(z, self.d) } else { self.sw.kz };
            let k = [wavenumber(x, self.w), wavenumber(y, self.h), kz];
            let mut kp = k;
            kp[axis] += h;
            let mut km = k;
            km[axis] -= h;
            let c = (self.sw_omega(kp) - 2.0 * self.sw_omega(k) + self.sw_omega(km)) / (h * h);
            acc += weight[i] * c;
        }
        acc / tot.max(1e-300)
    }

}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.16 y §2.18: false = orden alternado (x, y, z en los pasos pares y z, y, x en los impares;
    /// por defecto), true = orden fijo x, y, z.
    pub fn set_dirac_order(&mut self, fixed: bool) {
        self.sw.fixed_order = fixed;
        self.sw.rebuild();
    }

    pub fn dirac_order_fixed(&self) -> bool {
        self.sw.fixed_order
    }

    /// §2.17: 0 = orden fijo T_z·T_y·T_x (defecto), 1 = orden alternado, 2 = bombeos selectivos de
    /// espín (Higashikawa et al.).
    pub fn set_weyl_variant(&mut self, variant: u32) {
        self.sw.variant = variant.min(2);
        self.sw.rebuild();
    }

    pub fn weyl_variant(&self) -> u32 {
        self.sw.variant
    }

    /// §2.17 (y el Dirac de 4 componentes en un corte 2D): k_z de los saltos en z, que pasan a ser
    /// las fases e^{∓ik_z}. Solo en 2D.
    pub fn set_weyl_kz(&mut self, kz: f32) {
        if kz.is_finite() {
            self.sw.kz = kz as f64;
        }
    }

    pub fn weyl_kz(&self) -> f64 {
        self.sw.kz
    }

    /// Dirección del espín de los paquetes que se inyecten después (por defecto +z).
    pub fn set_packet_spin(&mut self, sx: f32, sy: f32, sz: f32) {
        let v = [sx as f64, sy as f64, sz as f64];
        if v.iter().all(|c| c.is_finite()) && v.iter().map(|c| c * c).sum::<f64>() > 1e-12 {
            self.sw.spin_dir = v;
        }
    }

    /// Evolución por partes (§3.13): cada paquete inyectado se guarda también como parte propia,
    /// que se evoluciona con el mismo campo (las corrientes salen de la suma).
    pub fn set_packet_parts(&mut self, on: bool) {
        self.sw.parts_mode = on;
        if !on {
            self.sw.parts.clear();
            self.sw.part_centers.clear();
        }
    }

    /// Partes distinguibles (cada paquete, una partícula: el campo lo crean Σ_p ρ_p y Σ_p J_p,
    /// sin interferencia entre partes) o indistinguibles (una sola función de onda, por defecto).
    pub fn set_distinguishable_parts(&mut self, on: bool) {
        self.sw.distinct = on;
    }

    /// true (por defecto): el paquete lleva e^{iq·A(r₀)·(r − r₀)} y su momento cinético es k₀.
    /// false: el gaussiano tal cual en el gauge del campo (en el de Landau se mueve con −q·A(r₀)).
    pub fn set_packet_gauge_compensation(&mut self, on: bool) {
        self.sw.gauge_compensation = on;
    }

    pub fn part_count(&self) -> u32 {
        self.sw.parts.len() as u32
    }

    /// Centroide lineal de la parte p (o de todo el estado si p ≥ número de partes) en el eje.
    pub fn part_centroid(&self, p: u32, axis: u32) -> f64 {
        if !self.sw.active {
            return f64::NAN;
        }
        let a = (axis as usize).min(2);
        let p = p as usize;
        if p < self.sw.parts.len() {
            self.sw_buffer_centroid(&self.sw.parts[p], self.sw.part_centers[p])[a]
        } else {
            self.sw_buffer_centroid(&self.sw.psi, self.sw.center)[a]
        }
    }

    pub fn part_norm(&self, p: u32) -> f64 {
        let p = p as usize;
        let b = if p < self.sw.parts.len() { &self.sw.parts[p] } else { &self.sw.psi };
        b.iter().map(|v| v * v).sum()
    }

    /// §2.16: paquete gaussiano proyectado sobre la banda ±1 (Dirac 3D o Weyl), con el color dado
    /// (SU(N)) y el espín de `set_packet_spin`. Inyecta antes de activar el campo: la proyección
    /// supone A = 0.
    pub fn inject_dirac_packet_3d(&mut self, x: f32, y: f32, z: f32, sigma: f32, kx: f32, ky: f32, kz: f32, band: i32, colour: u32) {
        if !self.sw.active {
            self.set_message("inject_dirac_packet_3d necesita el escenario Dirac 3D o Weyl.");
            return;
        }
        if ![x, y, z, sigma, kx, ky, kz].iter().all(|v| v.is_finite()) || sigma <= 0.0 {
            return;
        }
        if self.em.mode != 0 && self.em.a.iter().any(|v| *v != 0.0) {
            self.set_message("Aviso: la proyección sobre la banda supone A = 0; con este campo es aproximada.");
        }
        let col = if self.ncol > 1 { basis_colour(self.ncol, colour as usize % self.ncol) } else { [cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0)] };
        let c0 = [x as f64, y as f64, if self.dim == 3 { z as f64 } else { 0.0 }];
        let k0 = [kx as f64, ky as f64, if self.dim == 3 { kz as f64 } else { self.sw.kz }];
        let s = sigma as f64;
        let b = self.sw_build_packet(c0, [s, s, s], k0, if band < 0 { -1 } else { 1 }, col, true, self.packet_norm);
        self.sw_add_buffer(b, c0);
    }

    /// El mismo paquete con el espinor de la banda en k₀ pero sin proyectar (para comparar).
    pub fn inject_dirac_packet_3d_unprojected(&mut self, x: f32, y: f32, z: f32, sigma: f32, kx: f32, ky: f32, kz: f32, band: i32, colour: u32) {
        if !self.sw.active {
            return;
        }
        if ![x, y, z, sigma, kx, ky, kz].iter().all(|v| v.is_finite()) || sigma <= 0.0 {
            return;
        }
        let col = if self.ncol > 1 { basis_colour(self.ncol, colour as usize % self.ncol) } else { [cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0)] };
        let c0 = [x as f64, y as f64, if self.dim == 3 { z as f64 } else { 0.0 }];
        let k0 = [kx as f64, ky as f64, if self.dim == 3 { kz as f64 } else { self.sw.kz }];
        let s = sigma as f64;
        let b = self.sw_build_packet(c0, [s, s, s], k0, if band < 0 { -1 } else { 1 }, col, false, self.packet_norm);
        self.sw_add_buffer(b, c0);
    }

    /// Norma total del estado del motor de espinores (f64).
    pub fn spinor_norm(&self) -> f64 {
        self.sw.psi.iter().map(|v| v * v).sum()
    }

    /// ⟨Σ_a⟩/norma (a = 0, 1, 2) del estado (Dirac: Σ = 1⊗σ; Weyl: σ).
    pub fn spinor_spin(&self, axis: u32) -> f64 {
        if !self.sw.active {
            return 0.0;
        }
        let s = self.sw_buffer_spin(&self.sw.psi);
        if s[3] > 0.0 {
            s[(axis as usize).min(2)] / s[3]
        } else {
            0.0
        }
    }

    /// Fracción de la norma en la banda +1 (Dirac 3D y Weyl; proyector de la FFT con A = 0).
    pub fn spinor_band_population(&self) -> f64 {
        if !self.sw.active {
            return -1.0;
        }
        self.sw_band_fraction(&self.sw.psi)
    }

    /// Cuasi-energías por paso de la banda +1 en k (Dirac 3D: con el orden fijo, los dos niveles de
    /// espín se separan; con el alternado son iguales).
    pub fn spinor_band_levels(&self, kx: f32, ky: f32, kz: f32) -> Vec<f64> {
        self.sw_band_levels([kx as f64, ky as f64, kz as f64])
    }

    /// Cuasi-energía por paso de la banda +1 en k (Dirac 3D o Weyl, con la masa y el orden actuales).
    pub fn spinor_dispersion(&self, kx: f32, ky: f32, kz: f32) -> f64 {
        self.sw_omega([kx as f64, ky as f64, kz as f64])
    }

    /// ⟨∂²ω/∂k_a²⟩ del estado actual (la inercia del paquete).
    pub fn spinor_mean_curvature(&self, axis: u32) -> f64 {
        if !self.sw.active {
            return f64::NAN;
        }
        self.sw_mean_curvature(&self.sw.psi, (axis as usize).min(2))
    }

    /// ⟨E_axis⟩ de una carga externa gaussiana (como `add_static_charge`: total `charge`, σ =
    /// `sigma`) resuelta con Poisson en el toro, promediado sobre la densidad actual del estado.
    pub fn static_charge_field_mean(&self, x: f32, y: f32, z: f32, charge: f32, sigma: f32, axis: u32) -> f64 {
        if !self.sw.active || ![x, y, z, charge, sigma].iter().all(|v| v.is_finite()) {
            return f64::NAN;
        }
        let n = self.n;
        let c = self.wrap_cell(x as i32, y as i32, z as i32);
        let (cxp, cyp, czp) = self.coords(c);
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let s = (sigma as f64).max(0.3);
        let mut src = vec![0.0f64; n];
        let mut tot = 0.0;
        for i in 0..n {
            let (px, py, pz) = self.coords(i);
            let dx = min_image(px as f64 - cxp as f64, dims[0]);
            let dy = min_image(py as f64 - cyp as f64, dims[1]);
            let dz = if self.dim == 3 { min_image(pz as f64 - czp as f64, dims[2]) } else { 0.0 };
            src[i] = (-(dx * dx + dy * dy + dz * dz) / (2.0 * s * s)).exp();
            tot += src[i];
        }
        if tot <= 1e-300 {
            return 0.0;
        }
        for v in src.iter_mut() {
            *v *= charge as f64 / tot;
        }
        let phi = self.poisson_potential(&src);
        let rho = self.buffer_density(&self.sw.psi);
        self.field_mean_over(&phi, &rho, (axis as usize).min(self.dim - 1))
    }

    /// ⟨E_axis⟩ sobre la parte `dst` del campo de una carga unidad con la forma de la parte `src`.
    pub fn part_field_mean(&self, src: u32, dst: u32, axis: u32) -> f64 {
        let (s, d) = (src as usize, dst as usize);
        if s >= self.sw.parts.len() || d >= self.sw.parts.len() {
            return f64::NAN;
        }
        let mut rs = self.buffer_density(&self.sw.parts[s]);
        let t: f64 = rs.iter().sum();
        for v in rs.iter_mut() {
            *v /= t.max(1e-300);
        }
        let phi = self.poisson_potential(&rs);
        let rd = self.buffer_density(&self.sw.parts[d]);
        self.field_mean_over(&phi, &rd, (axis as usize).min(self.dim - 1))
    }

    /// ⟨∂²ω/∂k_a²⟩ de la parte p.
    pub fn part_mean_curvature(&self, p: u32, axis: u32) -> f64 {
        match self.sw.parts.get(p as usize) {
            Some(b) => self.sw_mean_curvature(b, (axis as usize).min(2)),
            None => f64::NAN,
        }
    }

    /// Energía de la materia: la cuasi-energía media −arg⟨ψ|F|ψ⟩·⟨ψ|ψ⟩, con F el paso siguiente
    /// con los enlaces congelados (§3.13). La referencia usa −Im⟨ψ|F|ψ⟩ = ⟨sin ε⟩, que la subestima
    /// con ε ≈ 0,5; con la fase, la suma con la energía del campo (en unidades de la materia:
    /// `em_energy`/q² con SU(N)) se conserva a menos del 1 %.
    pub fn spinor_matter_energy(&mut self) -> f64 {
        if !self.sw.active {
            return 0.0;
        }
        if self.em.mode != 0 && self.ncol == 1 {
            self.sw_refresh_phases();
        }
        let mut p = self.sw.psi.clone();
        let mut tmp = vec![0.0f64; p.len()];
        let parity = if self.sw.alternating() { (self.t & 1) as usize } else { 0 };
        self.sw_step_buffer(&mut p, &mut tmp, parity);
        let mut ov = cx(0.0, 0.0);
        let mut nn = 0.0;
        for k in 0..p.len() / 2 {
            let (a, b) = (cx(self.sw.psi[2 * k], self.sw.psi[2 * k + 1]), cx(p[2 * k], p[2 * k + 1]));
            ov += a.conj() * b;
            nn += a.norm_sqr();
        }
        -ov.arg() * nn
    }

    /// max |ρ_a − ρ_b| entre el estado de este universo y el de otro con la misma malla (copia con
    /// una transformación gauge: las densidades no deben cambiar).
    pub fn spinor_density_distance(&self, other: &QuantumSpacetimeUniverse) -> f64 {
        if other.n != self.n || !self.sw.active || !other.sw.active || other.sw.psi.len() != self.sw.psi.len() {
            return f64::NAN;
        }
        (0..self.n).map(|i| (self.sw_density(i) - other.sw_density(i)).abs()).fold(0.0, f64::max)
    }

    /// Activa el registro del flujo exacto por eje (Σ|P₊ψ|² − |P₋ψ|² de los subpasos).
    pub fn set_flux_tracking(&mut self, on: bool) {
        self.sw.track_flux = on;
    }

    /// Flujo neto de norma en el eje durante el último paso: la velocidad media exacta (no depende
    /// de las vueltas al toro).
    pub fn spinor_axis_flux(&self, axis: u32) -> f64 {
        self.sw.flux[(axis as usize).min(2)]
    }

    /// Flujo de la parte p en el eje durante el último paso (su velocidad exacta por su norma).
    pub fn part_axis_flux(&self, p: u32, axis: u32) -> f64 {
        self.sw.part_flux.get(p as usize).map(|f| f[(axis as usize).min(2)]).unwrap_or(0.0)
    }

    /// Norma y centroide (media circular en el eje) de la parte del estado a menos de `radius` del
    /// eje (x₀, y₀) paralelo a z. Devuelve [norma, centroide].
    pub fn spinor_tube_stats(&self, x0: f32, y0: f32, radius: f32, axis: u32) -> Vec<f64> {
        if !self.sw.active || !(x0.is_finite() && y0.is_finite() && radius.is_finite()) {
            return vec![0.0, 0.0];
        }
        let st_ = 2 * self.sw_namp();
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let a = (axis as usize).min(2);
        let l = [w, h, d][a];
        let (mut tot, mut cs, mut sn) = (0.0, 0.0, 0.0);
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let dx = min_image(x as f64 - x0 as f64, w);
            let dy = min_image(y as f64 - y0 as f64, h);
            if dx * dx + dy * dy > (radius as f64).powi(2) {
                continue;
            }
            let e: f64 = self.sw.psi[i * st_..(i + 1) * st_].iter().map(|v| v * v).sum();
            let th = 2.0 * PI * [x as f64, y as f64, z as f64][a] / l;
            tot += e;
            cs += e * th.cos();
            sn += e * th.sin();
        }
        let c = (sn.atan2(cs) * l / (2.0 * PI)).rem_euclid(l);
        vec![tot, c]
    }
}
