//! Espectro por autocorrelación (§2.10) y población de la banda +1 de la caminata de Dirac.
//! c(t) = ⟨ψ₀|ψ(t)⟩ se acumula en cada paso (hasta 8192) y S(ω) = |FFT(c·Hann)|² con relleno ×4:
//! los picos son las cuasi-energías del paso (niveles de Landau sin diagonalizar ninguna matriz).

use crate::dirac2d::{dirac2d_matrix, su2_band};
use crate::rng::SplitMix64;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

pub const AUTOCORR_MAX: usize = 8192;

pub struct AutoCorr {
    pub active: bool,
    pub psi0: Vec<f32>,
    pub c: Vec<Complex<f64>>,
    pub count: usize,
    pub spec: Vec<f32>,
    pub bins: usize,
}

impl AutoCorr {
    pub fn new() -> Self {
        AutoCorr { active: false, psi0: Vec::new(), c: vec![Complex::new(0.0, 0.0); AUTOCORR_MAX], count: 0, spec: Vec::new(), bins: 0 }
    }
}

impl QuantumSpacetimeUniverse {
    /// ⟨ψ₀|ψ⟩ con el estado guardado.
    pub(crate) fn overlap_with_psi0(&self) -> Complex<f64> {
        let p0 = &self.autocorr.psi0;
        let p = &self.buffer_read;
        let len = p0.len().min(p.len());
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let mut k = 0;
        while k + 1 < len {
            let (a, b) = (p0[k] as f64, p0[k + 1] as f64);
            let (c, d) = (p[k] as f64, p[k + 1] as f64);
            // conj(a + ib)·(c + id)
            re += a * c + b * d;
            im += a * d - b * c;
            k += 2;
        }
        Complex::new(re, im)
    }

    /// Acumula c(t) tras cada paso (si está activo).
    pub(crate) fn autocorr_record(&mut self) {
        if self.autocorr.active && self.autocorr.count < AUTOCORR_MAX {
            let v = self.overlap_with_psi0();
            let k = self.autocorr.count;
            self.autocorr.c[k] = v;
            self.autocorr.count += 1;
        }
    }

    /// Fracción de la norma en la banda +1 del Dirac 2D con A uniforme (U(k − qA)); −1 si A no es uniforme.
    pub(crate) fn band_population_2d(&mut self) -> f64 {
        let n = self.n;
        let (w, h) = (self.w, self.h);
        let (mut ax, mut ay) = (0.0, 0.0);
        if self.em.mode != 0 {
            ax = self.em.a[0];
            ay = self.em.a[n];
            for i in 0..n {
                if (self.em.a[i] - ax).abs() > 1e-9 || (self.em.a[n + i] - ay).abs() > 1e-9 {
                    return -1.0;
                }
            }
        }
        let q = if self.em.mode != 0 { self.em.q } else { 0.0 };
        let mut fft = crate::fftn::FftN::<f64>::new(w, h, 1);
        let mut up: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(self.buffer_read[4 * i] as f64, self.buffer_read[4 * i + 1] as f64)).collect();
        let mut dn: Vec<Complex<f64>> = (0..n).map(|i| Complex::new(self.buffer_read[4 * i + 2] as f64, self.buffer_read[4 * i + 3] as f64)).collect();
        fft.forward(&mut up);
        fft.forward(&mut dn);
        let m = self.mass as f64;
        let (mut tot, mut plus) = (0.0, 0.0);
        for i in 0..n {
            let (x, y, _) = self.coords(i);
            let kx = crate::fftn::wavenumber(x, w);
            let ky = crate::fftn::wavenumber(y, h);
            let (_, v) = su2_band(&dirac2d_matrix(m, kx - q * ax, ky - q * ay), 1);
            let ov = v[0].conj() * up[i] + v[1].conj() * dn[i];
            plus += ov.norm_sqr();
            tot += up[i].norm_sqr() + dn[i].norm_sqr();
        }
        if tot > 0.0 {
            plus / tot
        } else {
            0.0
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// §2.10: guarda ψ₀ normalizado (Dirac: ruido blanco; materia: ruido suavizado con el box-blur
    /// de la sopa de radio 4, la banda focalizante) y empieza a acumular c(t) = ⟨ψ₀|ψ(t)⟩.
    /// Reemplaza el campo por ψ₀.
    pub fn start_autocorrelation(&mut self, seed: u32) {
        if self.sw.active {
            self.start_spinor_autocorrelation(seed, 0.0, 0.0, 0.0, 0.0, false);
            return;
        }
        let n = self.n;
        let nc = self.nc;
        let stride = 2 * nc;
        let mut rng = SplitMix64::new(seed as u64 ^ 0xAC0F);
        let dirac = self.scenario != 0 && self.scenario != 1;
        if dirac {
            for v in self.buffer_read.iter_mut().take(n * stride) {
                *v = rng.gauss() as f32;
            }
        } else {
            let mut field = vec![0.0f32; 2 * n];
            for v in field.iter_mut() {
                *v = rng.gauss() as f32;
            }
            let mut line = vec![0.0f32; 2 * self.w.max(self.h).max(self.d)];
            for axis in 0..self.dim {
                for _ in 0..3 {
                    self.box_blur_axis(&mut field, &mut line, axis, 4);
                }
            }
            let inv = 1.0 / (nc as f32).sqrt();
            for i in 0..n {
                for k in 0..nc {
                    self.buffer_read[i * stride + 2 * k] = field[2 * i] * inv;
                    self.buffer_read[i * stride + 2 * k + 1] = field[2 * i + 1] * inv;
                }
            }
        }
        for i in 0..n {
            if self.has_walls && self.wall[i] != 0 {
                for k in 0..stride {
                    self.buffer_read[i * stride + k] = 0.0;
                }
            }
        }
        let norm: f64 = self.buffer_read[..n * stride].iter().map(|v| (*v as f64).powi(2)).sum();
        let s = if norm > 0.0 { (1.0 / norm).sqrt() as f32 } else { 0.0 };
        for v in self.buffer_read[..n * stride].iter_mut() {
            *v *= s;
        }
        self.autocorr.psi0 = self.buffer_read[..n * stride].to_vec();
        self.autocorr.count = 0;
        self.autocorr.active = true;
        self.autocorr_record();
    }

    /// Sigue acumulando c(t) con el estado actual como ψ₀ (sin reemplazar el campo).
    pub fn start_autocorrelation_from_current(&mut self) {
        if self.sw.active {
            self.sw_mirror();
        }
        let n = self.n * self.stride();
        let norm: f64 = self.buffer_read[..n].iter().map(|v| (*v as f64).powi(2)).sum();
        let s = if norm > 0.0 { (1.0 / norm).sqrt() as f32 } else { 0.0 };
        self.autocorr.psi0 = self.buffer_read[..n].iter().map(|v| v * s).collect();
        self.autocorr.count = 0;
        self.autocorr.active = true;
        self.autocorr_record();
    }

    /// Dirac 3D y Weyl: ψ₀ = ruido blanco (f64) normalizado, solo en las celdas a menos de `radius`
    /// de (x₀, y₀, z₀) si radius > 0 (en 3D con `uniform_z`, a menos de radius del eje paralelo a z),
    /// y uniforme en z si `uniform_z` (k_z = 0). Reemplaza el campo por ψ₀ y empieza a acumular c(t).
    pub fn start_spinor_autocorrelation(&mut self, seed: u32, x0: f32, y0: f32, z0: f32, radius: f32, uniform_z: bool) {
        if !self.sw.active {
            self.start_autocorrelation(seed);
            return;
        }
        let n = self.n;
        let st = 2 * self.sw_namp();
        let mut rng = SplitMix64::new(seed as u64 ^ 0xAC0F);
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let plane = self.w * self.h;
        for v in self.sw.psi.iter_mut() {
            *v = 0.0;
        }
        for i in 0..n {
            let (x, y, z) = self.coords(i);
            if radius > 0.0 {
                let dx = crate::util::min_image(x as f64 - x0 as f64, w);
                let dy = crate::util::min_image(y as f64 - y0 as f64, h);
                let dz = if self.dim == 3 && !uniform_z { crate::util::min_image(z as f64 - z0 as f64, d) } else { 0.0 };
                if dx * dx + dy * dy + dz * dz > (radius as f64).powi(2) {
                    continue;
                }
            }
            if uniform_z && z > 0 {
                let src = (i % plane) * st;
                for k in 0..st {
                    self.sw.psi[i * st + k] = self.sw.psi[src + k];
                }
                continue;
            }
            for k in 0..st {
                self.sw.psi[i * st + k] = rng.gauss();
            }
        }
        if self.has_walls {
            for i in 0..n {
                if self.wall[i] != 0 {
                    for k in 0..st {
                        self.sw.psi[i * st + k] = 0.0;
                    }
                }
            }
        }
        let norm: f64 = self.sw.psi.iter().map(|v| v * v).sum();
        let s = if norm > 0.0 { 1.0 / norm.sqrt() } else { 0.0 };
        for v in self.sw.psi.iter_mut() {
            *v *= s;
        }
        self.sw_mirror();
        self.autocorr.psi0 = self.buffer_read[..n * st].to_vec();
        self.autocorr.count = 0;
        self.autocorr.active = true;
        self.autocorr_record();
    }

    pub fn stop_autocorrelation(&mut self) {
        self.autocorr.active = false;
    }

    pub fn autocorrelation_length(&self) -> u32 {
        self.autocorr.count as u32
    }

    /// S(ω) = |Σ_t c(t)·Hann(t)·e^{iωt}|² con relleno de ceros ×4; devuelve el número de bins.
    /// Los pares [ω, S(ω)] quedan en `autocorrelation_spectrum_ptr`, con ω ∈ [−π, π) creciente.
    pub fn autocorrelation_spectrum(&mut self) -> u32 {
        let nt = self.autocorr.count;
        if nt < 4 {
            return 0;
        }
        let l = 4 * nt;
        let mut buf = vec![Complex::new(0.0f64, 0.0); l];
        for t in 0..nt {
            let wv = 0.5 * (1.0 - (2.0 * PI * t as f64 / (nt - 1) as f64).cos());
            buf[t] = self.autocorr.c[t] * wv;
        }
        let mut planner = FftPlanner::<f64>::new();
        // e^{+iωt}: la transformada inversa sin normalizar
        planner.plan_fft_inverse(l).process(&mut buf);
        self.autocorr.spec.clear();
        self.autocorr.spec.resize(2 * l, 0.0);
        for j in 0..l {
            // orden creciente en ω ∈ [−π, π)
            let src = (j + l / 2) % l;
            let mut om = 2.0 * PI * src as f64 / l as f64;
            if om >= PI {
                om -= 2.0 * PI;
            }
            self.autocorr.spec[2 * j] = om as f32;
            self.autocorr.spec[2 * j + 1] = buf[src].norm_sqr() as f32;
        }
        self.autocorr.bins = l;
        l as u32
    }

    /// Peso espectral en [lo, hi]: A(ω) = Σ_{|t|≤T} w(t)·c(t)·e^{iωt} con c(−t) = c(t)* y una ventana
    /// de Hann simétrica (w(0) = 1), integrado en [lo, hi] y dividido entre ∫A dω (= 2π·c(0)): la
    /// fracción de ⟨ψ₀|ψ₀⟩ en los estados con cuasi-energía en el intervalo.
    pub fn autocorrelation_weight(&self, lo: f32, hi: f32) -> f64 {
        let nt = self.autocorr.count;
        if nt < 4 {
            return 0.0;
        }
        let c = &self.autocorr.c[..nt];
        let tmax = (nt - 1) as f64;
        let w = |t: usize| 0.5 * (1.0 + (PI * t as f64 / (tmax + 1.0)).cos());
        let a = |om: f64| -> f64 {
            let mut s = c[0].re;
            for t in 1..nt {
                let e = Complex::from_polar(1.0, om * t as f64);
                s += 2.0 * w(t) * (c[t] * e).re;
            }
            s
        };
        let (lo, hi) = (lo as f64, hi as f64);
        let steps = ((hi - lo) / (PI / (8.0 * nt as f64))).ceil().max(16.0) as usize;
        let dw = (hi - lo) / steps as f64;
        let mut acc = 0.0;
        for j in 0..steps {
            acc += a(lo + (j as f64 + 0.5) * dw) * dw;
        }
        acc / (2.0 * PI * c[0].re.max(1e-300))
    }

    /// Fracción de ∫S(ω) (el espectro de §2.10, |FFT(c·Hann)|²) en |ω| ∈ [lo, hi]: el "peso
    /// espectral" que se ve en la gráfica. Pesa más los estados con mucho solapamiento (picos
    /// aislados) que la fracción de la norma de `autocorrelation_weight`.
    pub fn autocorrelation_spectrum_fraction(&mut self, lo: f32, hi: f32) -> f64 {
        let nb = self.autocorrelation_spectrum() as usize;
        let sp = &self.autocorr.spec;
        let (mut a, mut t) = (0.0f64, 0.0f64);
        for j in 0..nb {
            let (om, s) = (sp[2 * j], sp[2 * j + 1] as f64);
            t += s;
            if om >= lo && om <= hi {
                a += s;
            }
        }
        if t > 0.0 {
            a / t
        } else {
            0.0
        }
    }

    /// Pendiente de −arg c(t) con los valores acumulados (la frecuencia media del estado inicial
    /// mientras |c(t)| no ha decaído; sirve para los relojes de §3.12).
    pub fn autocorrelation_rate(&self) -> f64 {
        let nt = self.autocorr.count;
        if nt < 3 {
            return f64::NAN;
        }
        let c = &self.autocorr.c[..nt];
        let (mut ts, mut ph) = (Vec::with_capacity(nt), Vec::with_capacity(nt));
        let mut acc = 0.0;
        for t in 0..nt {
            if t > 0 {
                acc += crate::util::wrap_angle(c[t].arg() - c[t - 1].arg());
            }
            ts.push(t as f64);
            ph.push(acc);
        }
        -crate::util::fit_linear(&ts, &ph).1
    }

    pub fn autocorrelation_spectrum_ptr(&self) -> *const f32 {
        self.autocorr.spec.as_ptr()
    }

    /// §2.10: fracción de la norma en la banda +1 (escenario Dirac 2D); −1 si A no es uniforme.
    pub fn dirac_band_population(&mut self) -> f64 {
        if self.scenario != 2 || self.dim != 2 {
            return -1.0;
        }
        self.band_population_2d()
    }
}
