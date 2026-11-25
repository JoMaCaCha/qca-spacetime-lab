//! Álgebra de SU(N) para N = 2 y 3 (§2.11): matrices complejas N×N en un arreglo fijo de 9
//! entradas (fila mayor), generadores T^c = σ^c/2 o λ^c/2 con Tr(T^a·T^b) = ½·δ_ab, y la
//! exponencial exacta exp(−iθ·T): cerrada en SU(2) y por Cayley–Hamilton en SU(3)
//! (Morningstar y Peardon, 2004). Una serie truncada no conmutaría con Ẽ y rompería la ley de
//! Gauss.

use rustfft::num_complex::Complex;

pub type C = Complex<f64>;

#[derive(Clone, Copy, Debug)]
pub struct Mat {
    pub n: usize,
    pub a: [C; 9],
}

#[inline(always)]
fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

// Núcleos con N fijo: con N constante el compilador desenrolla los bucles y quita las
// comprobaciones de rango. Las matrices viven en un arreglo 3×3 (fila mayor) también con N = 2.
// En wasm32 con +simd128, cada Complex<f64> ocupa un registro f64x2 exacto (IEEE-754 f64).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd_c64 {
    use super::C;
    use core::arch::wasm32::*;

    #[inline(always)]
    pub unsafe fn load(z: &C) -> v128 {
        v128_load(z as *const C as *const v128)
    }

    #[inline(always)]
    pub unsafe fn store(dst: &mut C, v: v128) {
        v128_store(dst as *mut C as *mut v128, v);
    }

    /// Acumula a·b en (p1, p2): resultado final es p1 + flip_lo(p2).
    #[inline(always)]
    pub unsafe fn mac_ab(a: v128, b: v128, b_sw: v128, p1: &mut v128, p2: &mut v128) {
        let a_re = i64x2_shuffle::<0, 0>(a, a);
        let a_im = i64x2_shuffle::<1, 1>(a, a);
        *p1 = f64x2_add(*p1, f64x2_mul(a_re, b));
        *p2 = f64x2_add(*p2, f64x2_mul(a_im, b_sw));
    }

    /// Convierte (p1, p2) en [p1.0 − p2.0, p1.1 + p2.1] (para Σ a·b).
    #[inline(always)]
    pub unsafe fn finish_ab(p1: v128, p2: v128) -> v128 {
        let sign_lo = u64x2(0x8000_0000_0000_0000, 0);
        f64x2_add(p1, v128_xor(p2, sign_lo))
    }

    /// Convierte (p1, p2) en [p1.0 + p2.0, p1.1 − p2.1] (para Σ a*·b).
    #[inline(always)]
    pub unsafe fn finish_conj_a(p1: v128, p2: v128) -> v128 {
        let sign_hi = u64x2(0, 0x8000_0000_0000_0000);
        f64x2_add(p1, v128_xor(p2, sign_hi))
    }

    /// Producto complejo único a·b.
    #[inline(always)]
    pub unsafe fn cmul(a: v128, b: v128) -> v128 {
        let a_re = i64x2_shuffle::<0, 0>(a, a);
        let a_im = i64x2_shuffle::<1, 1>(a, a);
        let b_sw = i64x2_shuffle::<1, 0>(b, b);
        let p1 = f64x2_mul(a_re, b);
        let p2 = f64x2_mul(a_im, b_sw);
        finish_ab(p1, p2)
    }
}

#[inline]
fn mul_n<const N: usize>(a: &[C; 9], b: &[C; 9]) -> [C; 9] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let mut r = [c(0.0, 0.0); 9];
        let mut bv = [f64x2_splat(0.0); 9];
        let mut bsw = [f64x2_splat(0.0); 9];
        for k in 0..N {
            for j in 0..N {
                let v = simd_c64::load(&b[k * 3 + j]);
                bv[k * 3 + j] = v;
                bsw[k * 3 + j] = i64x2_shuffle::<1, 0>(v, v);
            }
        }
        for i in 0..N {
            let mut av = [f64x2_splat(0.0); 3];
            for k in 0..N {
                av[k] = simd_c64::load(&a[i * 3 + k]);
            }
            for j in 0..N {
                let mut p1 = f64x2_splat(0.0);
                let mut p2 = f64x2_splat(0.0);
                for k in 0..N {
                    simd_c64::mac_ab(av[k], bv[k * 3 + j], bsw[k * 3 + j], &mut p1, &mut p2);
                }
                simd_c64::store(&mut r[i * 3 + j], simd_c64::finish_ab(p1, p2));
            }
        }
        return r;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut r = [c(0.0, 0.0); 9];
        for i in 0..N {
            for j in 0..N {
                let mut s = c(0.0, 0.0);
                for k in 0..N {
                    s += a[i * 3 + k] * b[k * 3 + j];
                }
                r[i * 3 + j] = s;
            }
        }
        r
    }
}

#[inline]
fn dag_mul_n<const N: usize>(a: &[C; 9], b: &[C; 9]) -> [C; 9] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let mut r = [c(0.0, 0.0); 9];
        let mut bv = [f64x2_splat(0.0); 9];
        let mut bsw = [f64x2_splat(0.0); 9];
        for k in 0..N {
            for j in 0..N {
                let v = simd_c64::load(&b[k * 3 + j]);
                bv[k * 3 + j] = v;
                bsw[k * 3 + j] = i64x2_shuffle::<1, 0>(v, v);
            }
        }
        for i in 0..N {
            let mut av = [f64x2_splat(0.0); 3];
            for k in 0..N {
                av[k] = simd_c64::load(&a[k * 3 + i]);
            }
            for j in 0..N {
                let mut p1 = f64x2_splat(0.0);
                let mut p2 = f64x2_splat(0.0);
                for k in 0..N {
                    simd_c64::mac_ab(av[k], bv[k * 3 + j], bsw[k * 3 + j], &mut p1, &mut p2);
                }
                simd_c64::store(&mut r[i * 3 + j], simd_c64::finish_conj_a(p1, p2));
            }
        }
        return r;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut r = [c(0.0, 0.0); 9];
        for i in 0..N {
            for j in 0..N {
                let mut s = c(0.0, 0.0);
                for k in 0..N {
                    s += a[k * 3 + i].conj() * b[k * 3 + j];
                }
                r[i * 3 + j] = s;
            }
        }
        r
    }
}

#[inline]
fn mul_dag_n<const N: usize>(a: &[C; 9], b: &[C; 9]) -> [C; 9] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let mut r = [c(0.0, 0.0); 9];
        let mut av = [f64x2_splat(0.0); 9];
        let mut asw = [f64x2_splat(0.0); 9];
        for i in 0..N {
            for k in 0..N {
                let v = simd_c64::load(&a[i * 3 + k]);
                av[i * 3 + k] = v;
                asw[i * 3 + k] = i64x2_shuffle::<1, 0>(v, v);
            }
        }
        for j in 0..N {
            let mut bv = [f64x2_splat(0.0); 3];
            for k in 0..N {
                bv[k] = simd_c64::load(&b[j * 3 + k]);
            }
            for i in 0..N {
                // a·b* = b*·a: reutiliza finish_conj_a con (b, a)
                let mut p1 = f64x2_splat(0.0);
                let mut p2 = f64x2_splat(0.0);
                for k in 0..N {
                    simd_c64::mac_ab(bv[k], av[i * 3 + k], asw[i * 3 + k], &mut p1, &mut p2);
                }
                simd_c64::store(&mut r[i * 3 + j], simd_c64::finish_conj_a(p1, p2));
            }
        }
        return r;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut r = [c(0.0, 0.0); 9];
        for i in 0..N {
            for j in 0..N {
                let mut s = c(0.0, 0.0);
                for k in 0..N {
                    s += a[i * 3 + k] * b[j * 3 + k].conj();
                }
                r[i * 3 + j] = s;
            }
        }
        r
    }
}

#[inline(always)]
fn apply_n<const N: usize>(a: &[C; 9], v: &[C; 3]) -> [C; 3] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let mut r = [c(0.0, 0.0); 3];
        let mut vv = [f64x2_splat(0.0); 3];
        let mut vsw = [f64x2_splat(0.0); 3];
        for j in 0..N {
            let x = simd_c64::load(&v[j]);
            vv[j] = x;
            vsw[j] = i64x2_shuffle::<1, 0>(x, x);
        }
        for i in 0..N {
            let mut p1 = f64x2_splat(0.0);
            let mut p2 = f64x2_splat(0.0);
            for j in 0..N {
                simd_c64::mac_ab(simd_c64::load(&a[i * 3 + j]), vv[j], vsw[j], &mut p1, &mut p2);
            }
            simd_c64::store(&mut r[i], simd_c64::finish_ab(p1, p2));
        }
        return r;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut r = [c(0.0, 0.0); 3];
        for i in 0..N {
            let mut s = c(0.0, 0.0);
            for j in 0..N {
                s += a[i * 3 + j] * v[j];
            }
            r[i] = s;
        }
        r
    }
}

#[inline(always)]
fn apply_dag_n<const N: usize>(a: &[C; 9], v: &[C; 3]) -> [C; 3] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let mut r = [c(0.0, 0.0); 3];
        let mut vv = [f64x2_splat(0.0); 3];
        let mut vsw = [f64x2_splat(0.0); 3];
        for j in 0..N {
            let x = simd_c64::load(&v[j]);
            vv[j] = x;
            vsw[j] = i64x2_shuffle::<1, 0>(x, x);
        }
        for i in 0..N {
            let mut p1 = f64x2_splat(0.0);
            let mut p2 = f64x2_splat(0.0);
            for j in 0..N {
                simd_c64::mac_ab(simd_c64::load(&a[j * 3 + i]), vv[j], vsw[j], &mut p1, &mut p2);
            }
            simd_c64::store(&mut r[i], simd_c64::finish_conj_a(p1, p2));
        }
        return r;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut r = [c(0.0, 0.0); 3];
        for i in 0..N {
            let mut s = c(0.0, 0.0);
            for j in 0..N {
                s += a[j * 3 + i].conj() * v[j];
            }
            r[i] = s;
        }
        r
    }
}

#[inline(always)]
fn load_n<const N: usize>(src: &[f64]) -> [C; 9] {
    let src = &src[..2 * N * N];
    let mut a = [c(0.0, 0.0); 9];
    for i in 0..N {
        for j in 0..N {
            let o = 2 * (i * N + j);
            a[i * 3 + j] = c(src[o], src[o + 1]);
        }
    }
    a
}

#[inline(always)]
fn store_n<const N: usize>(a: &[C; 9], dst: &mut [f64]) {
    let dst = &mut dst[..2 * N * N];
    for i in 0..N {
        for j in 0..N {
            let o = 2 * (i * N + j);
            dst[o] = a[i * 3 + j].re;
            dst[o + 1] = a[i * 3 + j].im;
        }
    }
}

impl Mat {
    pub fn zero(n: usize) -> Self {
        Mat { n, a: [c(0.0, 0.0); 9] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Mat::zero(n);
        for i in 0..n {
            m.a[i * 3 + i] = c(1.0, 0.0);
        }
        m
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> C {
        self.a[i * 3 + j]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, v: C) {
        self.a[i * 3 + j] = v;
    }

    #[inline(always)]
    pub fn mul(&self, o: &Mat) -> Mat {
        let a = if self.n == 2 { mul_n::<2>(&self.a, &o.a) } else { mul_n::<3>(&self.a, &o.a) };
        Mat { n: self.n, a }
    }

    /// self† · o
    #[inline(always)]
    pub fn dag_mul(&self, o: &Mat) -> Mat {
        let a = if self.n == 2 { dag_mul_n::<2>(&self.a, &o.a) } else { dag_mul_n::<3>(&self.a, &o.a) };
        Mat { n: self.n, a }
    }

    /// self · o†
    #[inline(always)]
    pub fn mul_dag(&self, o: &Mat) -> Mat {
        let a = if self.n == 2 { mul_dag_n::<2>(&self.a, &o.a) } else { mul_dag_n::<3>(&self.a, &o.a) };
        Mat { n: self.n, a }
    }

    pub fn trace(&self) -> C {
        let mut s = c(0.0, 0.0);
        for i in 0..self.n {
            s += self.a[i * 3 + i];
        }
        s
    }

    #[inline(always)]
    pub fn add(&self, o: &Mat) -> Mat {
        let mut r = *self;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            use core::arch::wasm32::*;
            for i in 0..9 {
                let s = f64x2_add(simd_c64::load(&self.a[i]), simd_c64::load(&o.a[i]));
                simd_c64::store(&mut r.a[i], s);
            }
            return r;
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            for i in 0..9 {
                r.a[i] += o.a[i];
            }
            r
        }
    }

    pub fn scale(&self, s: C) -> Mat {
        let mut r = *self;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            let sv = simd_c64::load(&s);
            for i in 0..9 {
                simd_c64::store(&mut r.a[i], simd_c64::cmul(simd_c64::load(&self.a[i]), sv));
            }
            return r;
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            for i in 0..9 {
                r.a[i] *= s;
            }
            r
        }
    }

    /// Lee una matriz de un búfer f64 [re, im] por entradas (fila mayor N×N).
    #[inline(always)]
    pub fn load(n: usize, src: &[f64]) -> Mat {
        let a = if n == 2 { load_n::<2>(src) } else { load_n::<3>(src) };
        Mat { n, a }
    }

    #[inline(always)]
    pub fn store(&self, dst: &mut [f64]) {
        if self.n == 2 {
            store_n::<2>(&self.a, dst)
        } else {
            store_n::<3>(&self.a, dst)
        }
    }

    /// Reunitariza (Gram–Schmidt por filas) y corrige la fase del determinante a 1.
    pub fn reunitarize(&mut self) {
        let n = self.n;
        if n == 0 {
            return;
        }
        for i in 0..n {
            for k in 0..i {
                let mut d = c(0.0, 0.0);
                for j in 0..n {
                    d += self.a[k * 3 + j].conj() * self.a[i * 3 + j];
                }
                for j in 0..n {
                    let v = self.a[k * 3 + j] * d;
                    self.a[i * 3 + j] -= v;
                }
            }
            let mut nn = 0.0;
            for j in 0..n {
                nn += self.a[i * 3 + j].norm_sqr();
            }
            if nn > 1e-300 {
                let inv = 1.0 / nn.sqrt();
                for j in 0..n {
                    self.a[i * 3 + j] *= inv;
                }
            } else {
                let mut fixed = false;
                for basis in 0..n {
                    for j in 0..n {
                        self.a[i * 3 + j] = if j == basis { c(1.0, 0.0) } else { c(0.0, 0.0) };
                    }
                    for k in 0..i {
                        let mut d = c(0.0, 0.0);
                        for j in 0..n {
                            d += self.a[k * 3 + j].conj() * self.a[i * 3 + j];
                        }
                        for j in 0..n {
                            let v = self.a[k * 3 + j] * d;
                            self.a[i * 3 + j] -= v;
                        }
                    }
                    let mut bn = 0.0;
                    for j in 0..n {
                        bn += self.a[i * 3 + j].norm_sqr();
                    }
                    if bn > 1e-300 {
                        let inv = 1.0 / bn.sqrt();
                        for j in 0..n {
                            self.a[i * 3 + j] *= inv;
                        }
                        fixed = true;
                        break;
                    }
                }
                if !fixed {
                    for j in 0..n {
                        self.a[i * 3 + j] = if j == i { c(1.0, 0.0) } else { c(0.0, 0.0) };
                    }
                }
            }
        }
        let det = self.det();
        if det.norm_sqr() > 1e-300 {
            let ph = C::from_polar(1.0, -det.arg() / n as f64);
            for v in self.a.iter_mut() {
                *v *= ph;
            }
        }
    }

    pub fn det(&self) -> C {
        if self.n == 2 {
            self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
        } else {
            let a = |i, j| self.get(i, j);
            a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))
        }
    }

    /// U·ψ para un vector de color.
    #[inline(always)]
    pub fn apply(&self, v: &[C; 3]) -> [C; 3] {
        if self.n == 2 {
            apply_n::<2>(&self.a, v)
        } else {
            apply_n::<3>(&self.a, v)
        }
    }

    /// U†·ψ
    #[inline(always)]
    pub fn apply_dag(&self, v: &[C; 3]) -> [C; 3] {
        if self.n == 2 {
            apply_dag_n::<2>(&self.a, v)
        } else {
            apply_dag_n::<3>(&self.a, v)
        }
    }
}

/// Generadores T^c del álgebra (N² − 1 matrices).
pub struct Algebra {
    pub n: usize,
    pub ng: usize,
    pub t: Vec<Mat>,
    /// Entradas no nulas (i, j, T^c_ij) de cada generador: los productos con T^c cuestan 2–3
    /// operaciones en vez de un producto de matrices.
    pub sparse: Vec<Vec<(usize, usize, C)>>,
}

impl Algebra {
    pub fn new(n: usize) -> Self {
        let mut t = Vec::new();
        if n == 2 {
            let mut s1 = Mat::zero(2);
            s1.set(0, 1, c(0.5, 0.0));
            s1.set(1, 0, c(0.5, 0.0));
            let mut s2 = Mat::zero(2);
            s2.set(0, 1, c(0.0, -0.5));
            s2.set(1, 0, c(0.0, 0.5));
            let mut s3 = Mat::zero(2);
            s3.set(0, 0, c(0.5, 0.0));
            s3.set(1, 1, c(-0.5, 0.0));
            t = vec![s1, s2, s3];
        } else if n == 3 {
            let h = 0.5;
            let mut l = vec![Mat::zero(3); 8];
            l[0].set(0, 1, c(h, 0.0));
            l[0].set(1, 0, c(h, 0.0));
            l[1].set(0, 1, c(0.0, -h));
            l[1].set(1, 0, c(0.0, h));
            l[2].set(0, 0, c(h, 0.0));
            l[2].set(1, 1, c(-h, 0.0));
            l[3].set(0, 2, c(h, 0.0));
            l[3].set(2, 0, c(h, 0.0));
            l[4].set(0, 2, c(0.0, -h));
            l[4].set(2, 0, c(0.0, h));
            l[5].set(1, 2, c(h, 0.0));
            l[5].set(2, 1, c(h, 0.0));
            l[6].set(1, 2, c(0.0, -h));
            l[6].set(2, 1, c(0.0, h));
            let s = h / 3f64.sqrt();
            l[7].set(0, 0, c(s, 0.0));
            l[7].set(1, 1, c(s, 0.0));
            l[7].set(2, 2, c(-2.0 * s, 0.0));
            t = l;
        }
        let ng = t.len();
        let sparse = t
            .iter()
            .map(|m| {
                let mut v = Vec::new();
                for i in 0..n {
                    for j in 0..n {
                        let x = m.get(i, j);
                        if x.norm_sqr() > 0.0 {
                            v.push((i, j, x));
                        }
                    }
                }
                v
            })
            .collect();
        Algebra { n, ng, t, sparse }
    }

    /// Tr(T^c·M) = Σ_ij T^c_ij·M_ji.
    #[inline(always)]
    pub fn tr_t(&self, cc: usize, m: &Mat) -> C {
        let mut s = c(0.0, 0.0);
        for &(i, j, x) in &self.sparse[cc] {
            s += x * m.a[j * 3 + i];
        }
        s
    }

    /// X = Σ_c θ^c·T^c (hermítica, de traza nula).
    pub fn combine(&self, theta: &[f64]) -> Mat {
        let mut m = Mat::zero(self.n);
        for (cc, th) in theta.iter().enumerate().take(self.ng) {
            if *th == 0.0 {
                continue;
            }
            for &(i, j, x) in &self.sparse[cc] {
                m.a[i * 3 + j] += x * *th;
            }
        }
        m
    }

    /// Componentes X^c = 2·Re Tr(T^c·X).
    pub fn components(&self, x: &Mat, out: &mut [f64]) {
        for cc in 0..self.ng {
            out[cc] = 2.0 * self.tr_t(cc, x).re;
        }
    }

    /// Im Tr(T^c·M) para todos los c (la fuerza de Wilson).
    #[inline(always)]
    pub fn im_tr(&self, m: &Mat, out: &mut [f64]) {
        for cc in 0..self.ng {
            out[cc] = self.tr_t(cc, m).im;
        }
    }

    /// ψ†·T^c·ψ = Σ_ij conj(ψ_i)·T^c_ij·ψ_j para un vector de color.
    #[inline(always)]
    pub fn charge(&self, v: &[C; 3], out: &mut [f64]) {
        for cc in 0..self.ng {
            let mut s = 0.0;
            for &(i, j, x) in &self.sparse[cc] {
                s += (v[i].conj() * x * v[j]).re;
            }
            out[cc] = s;
        }
    }

    /// exp(−i·Σθ^c·T^c), exacta.
    pub fn exp_minus_i(&self, theta: &[f64]) -> Mat {
        if self.n == 2 {
            let (x, y, z) = (theta[0], theta[1], theta[2]);
            let nrm = (x * x + y * y + z * z).sqrt();
            if nrm < 1e-300 {
                return Mat::identity(2);
            }
            let (cs, sn) = ((0.5 * nrm).cos(), (0.5 * nrm).sin());
            let (nx, ny, nz) = (x / nrm, y / nrm, z / nrm);
            // cos(|θ|/2)·I − i·sin(|θ|/2)·(n·σ)
            let mut m = Mat::zero(2);
            m.set(0, 0, c(cs, -sn * nz));
            m.set(1, 1, c(cs, sn * nz));
            m.set(0, 1, c(-sn * ny, -sn * nx));
            m.set(1, 0, c(sn * ny, -sn * nx));
            return m;
        }
        // SU(3): exp(iQ) con Q = −θ·T por Cayley–Hamilton (Morningstar y Peardon, 2004)
        let q = self.combine(theta).scale(c(-1.0, 0.0));
        exp_i_hermitian3(&q)
    }
}

/// exp(iQ) para Q hermítica de traza nula 3×3 (fórmula de Morningstar y Peardon).
pub fn exp_i_hermitian3(q: &Mat) -> Mat {
    let q2 = q.mul(q);
    let c0_raw = q.det().re; // det Q = Tr(Q³)/3
    let c1 = 0.5 * q2.trace().re;
    if c1 < 1e-30 {
        // Q ≈ 0: exp(iQ) ≈ 1 + iQ − Q²/2
        let mut m = Mat::identity(3);
        for i in 0..9 {
            m.a[i] += q.a[i] * c(0.0, 1.0) - q2.a[i] * 0.5;
        }
        return m;
    }
    let neg = c0_raw < 0.0;
    let c0 = c0_raw.abs();
    let c13 = c1 / 3.0;
    let sq13 = c13.sqrt();
    let c0max = 2.0 * c13 * sq13;
    let ratio = (c0 / c0max).clamp(-1.0, 1.0);
    let th = ratio.acos();
    let (s3, c3) = (th / 3.0).sin_cos();
    let u = sq13 * c3;
    let w = c1.sqrt() * s3;
    let (sw, cw) = w.sin_cos();
    let xi0 = if w.abs() < 0.05 {
        let w2 = w * w;
        1.0 - w2 / 6.0 * (1.0 - w2 / 20.0 * (1.0 - w2 / 42.0))
    } else {
        sw / w
    };
    // e^{−iu} y e^{2iu} = (e^{−iu})^{−2} con una sola llamada a sin_cos
    let (su, cu) = u.sin_cos();
    let emiu = c(cu, -su);
    let e2iu = c(cu * cu - su * su, 2.0 * su * cu);
    let h0 = e2iu * (u * u - w * w) + emiu * c(8.0 * u * u * cw, 2.0 * u * (3.0 * u * u + w * w) * xi0);
    let h1 = e2iu * (2.0 * u) - emiu * c(2.0 * u * cw, -(3.0 * u * u - w * w) * xi0);
    let h2 = e2iu - emiu * c(cw, 3.0 * u * xi0);
    let den = 9.0 * u * u - w * w;
    let (mut f0, mut f1, mut f2) = (h0 / den, h1 / den, h2 / den);
    if neg {
        f0 = f0.conj();
        f1 = -f1.conj();
        f2 = f2.conj();
    }
    let mut m = Mat::identity(3).scale(f0);
    for i in 0..9 {
        m.a[i] += q.a[i] * f1 + q2.a[i] * f2;
    }
    m
}

/// Vector de color unitario de la base: e_colour.
pub fn basis_colour(n: usize, colour: usize) -> [C; 3] {
    let mut v = [c(0.0, 0.0); 3];
    v[colour.min(n.saturating_sub(1)).min(2)] = c(1.0, 0.0);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp_series(alg: &Algebra, theta: &[f64]) -> Mat {
        // exp(−iX) por serie de Taylor larga (solo para comprobar)
        let x = alg.combine(theta).scale(c(0.0, -1.0));
        let mut term = Mat::identity(alg.n);
        let mut sum = Mat::identity(alg.n);
        for k in 1..40 {
            term = term.mul(&x).scale(c(1.0 / k as f64, 0.0));
            sum = sum.add(&term);
        }
        sum
    }

    #[test]
    fn exponential_exact() {
        let mut rng = crate::rng::SplitMix64::new(3);
        for &n in &[2usize, 3] {
            let alg = Algebra::new(n);
            for _ in 0..200 {
                let theta: Vec<f64> = (0..alg.ng).map(|_| 2.0 * rng.gauss()).collect();
                let a = alg.exp_minus_i(&theta);
                let b = exp_series(&alg, &theta);
                for i in 0..9 {
                    assert!((a.a[i] - b.a[i]).norm() < 1e-10, "N = {n}");
                }
                assert!((a.det() - c(1.0, 0.0)).norm() < 1e-10);
            }
            // Tr(T^a T^b) = δ_ab/2
            for p in 0..alg.ng {
                for q in 0..alg.ng {
                    let tr = alg.t[p].mul(&alg.t[q]).trace().re;
                    let want = if p == q { 0.5 } else { 0.0 };
                    assert!((tr - want).abs() < 1e-14);
                }
            }
        }
    }
}
