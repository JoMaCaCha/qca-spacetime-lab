//! FFT en 1, 2 o 3 dimensiones sobre búferes complejos planos con el layout de la malla
//! (índice (z·H + y)·W + x). Los planes y los búferes auxiliares se crean una vez, así que
//! `forward`/`inverse` no asignan memoria (§4.2).

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftNum, FftPlanner};
use std::sync::Arc;

pub struct FftN<T: FftNum> {
    pub w: usize,
    pub h: usize,
    pub d: usize,
    fx: Arc<dyn Fft<T>>,
    ix: Arc<dyn Fft<T>>,
    fy: Arc<dyn Fft<T>>,
    iy: Arc<dyn Fft<T>>,
    fz: Arc<dyn Fft<T>>,
    iz: Arc<dyn Fft<T>>,
    scratch: Vec<Complex<T>>,
    line: Vec<Complex<T>>,
}

impl<T: FftNum> FftN<T> {
    pub fn new(w: usize, h: usize, d: usize) -> Self {
        let mut p = FftPlanner::<T>::new();
        let fx = p.plan_fft_forward(w);
        let ix = p.plan_fft_inverse(w);
        let fy = p.plan_fft_forward(h.max(1));
        let iy = p.plan_fft_inverse(h.max(1));
        let fz = p.plan_fft_forward(d.max(1));
        let iz = p.plan_fft_inverse(d.max(1));
        let mut s = 0;
        for f in [&fx, &ix, &fy, &iy, &fz, &iz] {
            s = s.max(f.get_inplace_scratch_len());
        }
        let zero = Complex::new(T::zero(), T::zero());
        FftN {
            w,
            h,
            d,
            fx,
            ix,
            fy,
            iy,
            fz,
            iz,
            scratch: vec![zero; s.max(1)],
            line: vec![zero; w.max(h).max(d).max(1)],
        }
    }

    pub fn len(&self) -> usize {
        self.w * self.h * self.d
    }

    fn axis_pass(&mut self, data: &mut [Complex<T>], axis: usize, inverse: bool) {
        let (w, h, d) = (self.w, self.h, self.d);
        if data.len() < self.len() || self.len() == 0 {
            return;
        }
        match axis {
            0 => {
                let f = if inverse { &self.ix } else { &self.fx };
                f.process_with_scratch(&mut data[..self.len()], &mut self.scratch);
            }
            1 => {
                if h <= 1 {
                    return;
                }
                let f = if inverse { &*self.iy } else { &*self.fy };
                let line = &mut self.line[..h];
                let scratch = &mut self.scratch;
                for z in 0..d {
                    let plane = z * h * w;
                    for x in 0..w {
                        for y in 0..h {
                            line[y] = data[plane + y * w + x];
                        }
                        f.process_with_scratch(line, scratch);
                        for y in 0..h {
                            data[plane + y * w + x] = line[y];
                        }
                    }
                }
            }
            _ => {
                if d <= 1 {
                    return;
                }
                let f = if inverse { &*self.iz } else { &*self.fz };
                let line = &mut self.line[..d];
                let scratch = &mut self.scratch;
                let wh = w * h;
                for y in 0..h {
                    let yw = y * w;
                    for x in 0..w {
                        let base = yw + x;
                        for z in 0..d {
                            line[z] = data[z * wh + base];
                        }
                        f.process_with_scratch(line, scratch);
                        for z in 0..d {
                            data[z * wh + base] = line[z];
                        }
                    }
                }
            }
        }
    }

    /// Transformada directa sin normalizar.
    pub fn forward(&mut self, data: &mut [Complex<T>]) {
        if self.len() == 0 || data.len() < self.len() {
            return;
        }
        for a in 0..3 {
            self.axis_pass(data, a, false);
        }
    }

    /// Transformada inversa normalizada (divide por N).
    pub fn inverse(&mut self, data: &mut [Complex<T>]) {
        let n = self.len();
        if n == 0 || data.len() < n {
            return;
        }
        for a in 0..3 {
            self.axis_pass(data, a, true);
        }
        if let Some(nf) = T::from_usize(n) {
            let inv = T::one() / nf;
            for v in data[..n].iter_mut() {
                *v = *v * inv;
            }
        }
    }
}

/// Número de onda de red del índice `i` en un eje de longitud `l`, en [−π, π).
#[inline]
pub fn wavenumber(i: usize, l: usize) -> f64 {
    let ii = if i >= (l + 1) / 2 { i as f64 - l as f64 } else { i as f64 };
    let k = 2.0 * std::f64::consts::PI * ii / l as f64;
    if k >= std::f64::consts::PI { k - 2.0 * std::f64::consts::PI } else { k }
}
