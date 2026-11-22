//! Diagnóstico: norma, IPR, centro de masas, momento total, espectro P(k), MIP y el análisis
//! de solitones al estilo de Lenia, ampliado a un inspector de partículas (§4.3).

use crate::fftn::wavenumber;
use crate::util::{min_image, wrap_angle};
use crate::{QuantumSpacetimeUniverse, MAX_SOLITONS, SOLITON_FIELDS};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Umbral de detección: E > DETECT_FACTOR·⟨E⟩ (orientativo, §4.3).
pub const DETECT_FACTOR: f32 = 8.0;
/// Tamaño mínimo de una componente para contar como solitón.
pub const MIN_CELLS: usize = 5;

/// Búferes del análisis de solitones, todos reservados en `new`.
pub struct SolitonAnalyzer {
    pub labels: Vec<i32>,
    pub stack: Vec<u32>,
    pub stats: Vec<f32>,
    pub count: usize,
    acc: Vec<[f64; 7]>,
    peak: Vec<(f32, u32)>,
    lin: Vec<[f64; 3]>,
    rg2: Vec<f64>,
    cur_phase: Vec<f64>,
    // análisis anterior
    prev_count: usize,
    prev_center: Vec<[f64; 3]>,
    prev_rg: Vec<f64>,
    prev_id: Vec<u32>,
    prev_peak: Vec<f32>,
    prev_phase: Vec<f64>,
    prev_t: u32,
    prev_valid: bool,
    used: Vec<bool>,
    taken: Vec<bool>,
    pairs: Vec<(f64, u16, u16)>,
    next_id: u32,
}

impl SolitonAnalyzer {
    pub fn new(n: usize) -> Self {
        SolitonAnalyzer {
            labels: vec![-1; n],
            stack: Vec::with_capacity(n),
            stats: vec![0.0; MAX_SOLITONS * SOLITON_FIELDS],
            count: 0,
            acc: vec![[0.0; 7]; MAX_SOLITONS],
            peak: vec![(0.0, 0); MAX_SOLITONS],
            lin: vec![[0.0; 3]; MAX_SOLITONS],
            rg2: vec![0.0; MAX_SOLITONS],
            cur_phase: vec![0.0; MAX_SOLITONS],
            prev_count: 0,
            prev_center: vec![[0.0; 3]; MAX_SOLITONS],
            prev_rg: vec![0.0; MAX_SOLITONS],
            prev_id: vec![0; MAX_SOLITONS],
            prev_peak: vec![0.0; MAX_SOLITONS],
            prev_phase: vec![0.0; MAX_SOLITONS],
            prev_t: 0,
            prev_valid: false,
            used: vec![false; MAX_SOLITONS],
            taken: vec![false; MAX_SOLITONS],
            pairs: Vec::with_capacity(16 * MAX_SOLITONS),
            next_id: 1,
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.prev_count = 0;
        self.prev_valid = false;
        self.labels.iter_mut().for_each(|v| *v = -1);
    }
}

impl QuantumSpacetimeUniverse {
    #[inline]
    pub(crate) fn density(&self, i: usize) -> f32 {
        let stride = self.stride();
        let b = i * stride;
        let mut e = 0.0f32;
        for k in 0..stride {
            let v = self.buffer_read[b + k];
            e += v * v;
        }
        e
    }

    pub(crate) fn mean_density(&self) -> f64 {
        self.total_norm() / self.n as f64
    }

    #[inline]
    fn neighbors6(&self, i: usize) -> [usize; 6] {
        let (w, h, d) = (self.w, self.h, self.d);
        let (x, y, z) = self.coords(i);
        let mut nb = [usize::MAX; 6];
        nb[0] = self.idx(if x == 0 { w - 1 } else { x - 1 }, y, z);
        nb[1] = self.idx(if x + 1 == w { 0 } else { x + 1 }, y, z);
        nb[2] = self.idx(x, if y == 0 { h - 1 } else { y - 1 }, z);
        nb[3] = self.idx(x, if y + 1 == h { 0 } else { y + 1 }, z);
        if self.dim == 3 {
            nb[4] = self.idx(x, y, if z == 0 { d - 1 } else { z - 1 });
            nb[5] = self.idx(x, y, if z + 1 == d { 0 } else { z + 1 });
        }
        nb
    }

    /// Centro de E^p a lo largo de un eje: la media circular hace de ancla y, sobre ella, se
    /// promedian linealmente los desplazamientos de imagen mínima (no se rompe en el borde).
    /// Con `window = Some((centro, radio))` solo cuentan las celdas a menos de ese radio.
    pub(crate) fn centroid_axis(&self, axis: usize, weight_pow: i32, window: Option<([f64; 3], f64)>) -> f64 {
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let l = dims[axis];
        let weight = |e: f64| -> f64 {
            match weight_pow {
                1 => e,
                2 => e * e,
                p => e.powi(p),
            }
        };
        let inside = |c: [usize; 3]| -> bool {
            match window {
                None => true,
                Some((p, r)) => {
                    let mut r2 = 0.0;
                    for a in 0..self.dim {
                        let dd = min_image(c[a] as f64 - p[a], dims[a]);
                        r2 += dd * dd;
                    }
                    r2 <= r * r
                }
            }
        };
        let (mut sc, mut ss, mut sw) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            if !inside([x, y, z]) {
                continue;
            }
            let e = weight(self.density(i) as f64);
            if e == 0.0 {
                continue;
            }
            let th = 2.0 * PI * [x, y, z][axis] as f64 / l;
            sc += e * th.cos();
            ss += e * th.sin();
            sw += e;
        }
        if sw <= 0.0 {
            return 0.0;
        }
        let c0 = (ss.atan2(sc) / (2.0 * PI) * l).rem_euclid(l);
        let mut s1 = 0.0f64;
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            if !inside([x, y, z]) {
                continue;
            }
            let e = weight(self.density(i) as f64);
            if e == 0.0 {
                continue;
            }
            s1 += e * min_image([x, y, z][axis] as f64 - c0, l);
        }
        (c0 + s1 / sw).rem_euclid(l)
    }

    /// Detecta las componentes conexas toroidales con E > 8·⟨E⟩ (≥ 5 celdas), las mide y las
    /// empareja con el análisis anterior. No asigna memoria.
    pub(crate) fn run_soliton_analysis(&mut self) -> usize {
        let n = self.n;
        let thr = DETECT_FACTOR * self.mean_density() as f32;
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let mut labels = std::mem::take(&mut self.sol.labels);
        let mut stack = std::mem::take(&mut self.sol.stack);
        labels.iter_mut().for_each(|v| *v = -1);
        let mut count = 0usize;
        let is_wall = |s: &Self, i: usize| s.has_walls && s.wall[i] != 0;
        for start in 0..n {
            if labels[start] != -1 || self.density(start) <= thr || is_wall(self, start) {
                continue;
            }
            let label = count as i32;
            stack.clear();
            stack.push(start as u32);
            labels[start] = label;
            let mut acc = [0.0f64; 7];
            let mut peak = (0.0f32, start as u32);
            let mut ncells = 0usize;
            while let Some(ci) = stack.pop() {
                let i = ci as usize;
                ncells += 1;
                let e = self.density(i);
                if e > peak.0 {
                    peak = (e, ci);
                }
                let (x, y, z) = self.coords(i);
                let e64 = e as f64;
                acc[0] += e64;
                let c = [x as f64, y as f64, z as f64];
                for a in 0..3 {
                    let th = 2.0 * PI * c[a] / dims[a];
                    acc[1 + 2 * a] += e64 * th.cos();
                    acc[2 + 2 * a] += e64 * th.sin();
                }
                for j in self.neighbors6(i) {
                    if j != usize::MAX && labels[j] == -1 && self.density(j) > thr && !is_wall(self, j) {
                        labels[j] = label;
                        stack.push(j as u32);
                    }
                }
            }
            if ncells < MIN_CELLS || count >= MAX_SOLITONS {
                // Componente descartada: se marca −2 para no volver a visitarla.
                stack.clear();
                stack.push(start as u32);
                labels[start] = -2;
                while let Some(ci) = stack.pop() {
                    for j in self.neighbors6(ci as usize) {
                        if j != usize::MAX && labels[j] == label {
                            labels[j] = -2;
                            stack.push(j as u32);
                        }
                    }
                }
                continue;
            }
            self.sol.acc[count] = acc;
            self.sol.peak[count] = peak;
            count += 1;
        }
        // centro circular de cada componente
        for s in 0..count {
            let acc = self.sol.acc[s];
            let base = s * SOLITON_FIELDS;
            for a in 0..3 {
                let c = if a < self.dim {
                    (acc[2 + 2 * a].atan2(acc[1 + 2 * a]) / (2.0 * PI) * dims[a]).rem_euclid(dims[a])
                } else {
                    0.0
                };
                self.sol.stats[base + 2 + a] = c as f32;
            }
            self.sol.lin[s] = [0.0; 3];
            self.sol.rg2[s] = 0.0;
        }
        // refinamiento lineal del centro y radio de giro
        for i in 0..n {
            let l = labels[i];
            if l < 0 {
                continue;
            }
            let s = l as usize;
            let e = self.density(i) as f64;
            let (x, y, z) = self.coords(i);
            let base = s * SOLITON_FIELDS;
            let c = [x as f64, y as f64, z as f64];
            for a in 0..self.dim {
                let dd = min_image(c[a] - self.sol.stats[base + 2 + a] as f64, dims[a]);
                self.sol.lin[s][a] += e * dd;
                self.sol.rg2[s] += e * dd * dd;
            }
        }
        for s in 0..count {
            let base = s * SOLITON_FIELDS;
            let norm = self.sol.acc[s][0];
            let mut shift2 = 0.0;
            for a in 0..self.dim {
                let c0 = self.sol.stats[base + 2 + a] as f64;
                let sh = self.sol.lin[s][a] / norm;
                shift2 += sh * sh;
                self.sol.stats[base + 2 + a] = (c0 + sh).rem_euclid(dims[a]) as f32;
            }
            self.sol.stats[base + 1] = norm as f32;
            self.sol.stats[base + 8] = ((self.sol.rg2[s] / norm - shift2).max(0.0)).sqrt() as f32;
            self.sol.stats[base + 9] = self.sol.peak[s].0;
            let (sr, si) = self.sum_psi_at(self.sol.peak[s].1 as usize);
            self.sol.cur_phase[s] = (si as f64).atan2(sr as f64) + self.theta_ref as f64;
        }
        self.sol.labels = labels;
        self.sol.stack = stack;
        self.sol.count = count;
        self.match_solitons();
        count
    }

    /// Empareja con el análisis anterior: centro más cercano dentro de 2× el radio de giro.
    fn match_solitons(&mut self) {
        let count = self.sol.count;
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let dt = self.t.wrapping_sub(self.sol.prev_t) as f64;
        let prev_n = if self.sol.prev_valid { self.sol.prev_count } else { 0 };
        let mut pairs = std::mem::take(&mut self.sol.pairs);
        pairs.clear();
        for s in 0..count {
            let base = s * SOLITON_FIELDS;
            let c = [self.sol.stats[base + 2] as f64, self.sol.stats[base + 3] as f64, self.sol.stats[base + 4] as f64];
            let rg = self.sol.stats[base + 8] as f64;
            for p in 0..prev_n {
                let pc = self.sol.prev_center[p];
                let mut r2 = 0.0;
                for a in 0..self.dim {
                    let dd = min_image(c[a] - pc[a], dims[a]);
                    r2 += dd * dd;
                }
                let lim = 2.0 * rg.max(self.sol.prev_rg[p]) + 1.0;
                if r2 <= lim * lim && pairs.len() < pairs.capacity() {
                    pairs.push((r2, s as u16, p as u16));
                }
            }
        }
        pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for p in 0..MAX_SOLITONS.min(prev_n.max(count) + 1) {
            self.sol.used[p] = false;
            self.sol.taken[p] = false;
        }
        for &(_, s, p) in pairs.iter() {
            let (s, p) = (s as usize, p as usize);
            if self.sol.used[p] || self.sol.taken[s] {
                continue;
            }
            self.sol.used[p] = true;
            self.sol.taken[s] = true;
            let base = s * SOLITON_FIELDS;
            self.sol.stats[base] = self.sol.prev_id[p] as f32;
            let pc = self.sol.prev_center[p];
            let mut speed2 = 0.0;
            for a in 0..3 {
                let v = if a < self.dim && dt > 0.0 { min_image(self.sol.stats[base + 2 + a] as f64 - pc[a], dims[a]) / dt } else { 0.0 };
                self.sol.stats[base + 5 + a] = v as f32;
                speed2 += v * v;
            }
            // ω = m − Δ(arg Σψ + θ_ref)/Δt: se desenrolla respecto a la rotación de la banda,
            // así que vale mientras |m − ω|·Δt < π.
            let omega = if dt > 0.0 { self.mass as f64 - wrap_angle(self.sol.cur_phase[s] - self.sol.prev_phase[p]) / dt } else { f64::NAN };
            self.sol.stats[base + 10] = omega as f32;
            let pk = self.sol.stats[base + 9];
            let pp = self.sol.prev_peak[p];
            self.sol.stats[base + 11] = if speed2.sqrt() > 0.01 {
                1.0
            } else if pp > 0.0 && ((pk - pp) / pp).abs() > 0.2 {
                2.0
            } else {
                0.0
            };
        }
        self.sol.pairs = pairs;
        for s in 0..count {
            if self.sol.taken[s] {
                continue;
            }
            let base = s * SOLITON_FIELDS;
            self.sol.stats[base] = self.sol.next_id as f32;
            self.sol.next_id += 1;
            for a in 0..3 {
                self.sol.stats[base + 5 + a] = 0.0;
            }
            self.sol.stats[base + 10] = f32::NAN;
            self.sol.stats[base + 11] = 0.0;
        }
        for s in 0..count {
            let base = s * SOLITON_FIELDS;
            self.sol.prev_center[s] = [self.sol.stats[base + 2] as f64, self.sol.stats[base + 3] as f64, self.sol.stats[base + 4] as f64];
            self.sol.prev_rg[s] = self.sol.stats[base + 8] as f64;
            self.sol.prev_id[s] = self.sol.stats[base] as u32;
            self.sol.prev_peak[s] = self.sol.stats[base + 9];
            self.sol.prev_phase[s] = self.sol.cur_phase[s];
        }
        self.sol.prev_count = count;
        self.sol.prev_t = self.t;
        self.sol.prev_valid = true;
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// 𝒩 = Σ_r E (acumulada en f64).
    pub fn total_norm(&self) -> f64 {
        let stride = self.stride();
        let mut s = 0.0f64;
        for i in 0..self.n {
            let b = i * stride;
            let mut e = 0.0f32;
            for k in 0..stride {
                let v = self.buffer_read[b + k];
                e += v * v;
            }
            s += e as f64;
        }
        s
    }

    /// Máximo de E en la malla.
    pub fn max_density(&self) -> f32 {
        let mut m = 0.0f32;
        for i in 0..self.n {
            m = m.max(self.density(i));
        }
        m
    }

    /// IPR = N_celdas·ΣE²/(ΣE)²: ≈ 1 + 1/n_c para ruido, ≈ 2 para un campo gaussiano.
    pub fn ipr(&self) -> f64 {
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            let e = self.density(i) as f64;
            s1 += e;
            s2 += e * e;
        }
        if s1 <= 0.0 {
            0.0
        } else {
            self.n as f64 * s2 / (s1 * s1)
        }
    }

    /// Centro lineal de E a lo largo de un eje (anclado en la media circular, §4.3).
    pub fn center_of_mass(&self, axis: u32) -> f64 {
        let a = (axis as usize).min(self.dim - 1);
        self.centroid_axis(a, 1, None)
    }

    /// Centro de E^power dentro de una esfera de radio `half` alrededor de (cx, cy, cz).
    #[allow(clippy::too_many_arguments)]
    pub fn centroid_window(&self, axis: u32, cx: f32, cy: f32, cz: f32, half: f32, power: i32) -> f64 {
        let a = (axis as usize).min(self.dim - 1);
        self.centroid_axis(a, power.max(1), Some(([cx as f64, cy as f64, cz as f64], half as f64)))
    }

    /// P = Σ_canales Σ_k k·|ψ̂(k)|² / N_celdas, con k ∈ [−π, π) (FFT).
    pub fn total_momentum(&mut self, axis: u32) -> f64 {
        let a = (axis as usize).min(self.dim - 1);
        let stride = self.stride();
        let mut total = 0.0f64;
        let (w, h, d) = (self.w, self.h, self.d);
        for k in 0..self.nc * self.ncol {
            for i in 0..self.n {
                self.cbuf[i] = Complex::new(self.buffer_read[i * stride + 2 * k], self.buffer_read[i * stride + 2 * k + 1]);
            }
            let mut buf = std::mem::take(&mut self.cbuf);
            self.fft.forward(&mut buf);
            for i in 0..self.n {
                let (x, y, z) = (i % w, (i / w) % h, i / (w * h));
                let kk = match a {
                    0 => wavenumber(x, w),
                    1 => wavenumber(y, h),
                    _ => wavenumber(z, d),
                };
                total += kk * buf[i].norm_sqr() as f64;
            }
            self.cbuf = buf;
        }
        total / self.n as f64
    }

    /// Detecta, mide y empareja solitones; devuelve cuántos hay.
    pub fn analyze_solitons(&mut self) -> u32 {
        let n = self.run_soliton_analysis() as u32;
        if self.ncol > 1 {
            self.ym_soliton_colours();
        }
        n
    }

    /// Número de solitones del último análisis.
    pub fn soliton_count(&self) -> u32 {
        self.sol.count as u32
    }

    /// n × 12 floats: [id, N, cx, cy, cz, vx, vy, vz, radio de giro, E pico, ω, tipo].
    pub fn soliton_stats_ptr(&self) -> *const f32 {
        self.sol.stats.as_ptr()
    }

    /// Índice del solitón que contiene la celda, o −1.
    pub fn soliton_at(&self, x: i32, y: i32, z: i32) -> i32 {
        let c = self.wrap_cell(x, y, z);
        let l = self.sol.labels[c];
        if l >= 0 && (l as usize) < self.sol.count {
            l
        } else {
            -1
        }
    }

    /// P(k) radial de δ = E/⟨E⟩ − 1, en bins de ancho Δk = 2π/L (L = lado menor) desde k = 0.
    pub fn compute_power_spectrum(&mut self) {
        let mean = self.mean_density();
        let inv = if mean > 0.0 { 1.0 / mean } else { 0.0 };
        for i in 0..self.n {
            self.cbuf[i] = Complex::new((self.density(i) as f64 * inv - 1.0) as f32, 0.0);
        }
        let mut buf = std::mem::take(&mut self.cbuf);
        self.fft.forward(&mut buf);
        let l_min = if self.dim == 3 { self.w.min(self.h).min(self.d) } else { self.w.min(self.h) };
        let dk = 2.0 * PI / l_min as f64;
        self.spectrum.iter_mut().for_each(|v| *v = 0.0);
        self.spectrum_count.iter_mut().for_each(|v| *v = 0);
        let (w, h, d) = (self.w, self.h, self.d);
        let norm = 1.0 / self.n as f64;
        for i in 0..self.n {
            let (x, y, z) = (i % w, (i / w) % h, i / (w * h));
            let kx = wavenumber(x, w);
            let ky = wavenumber(y, h);
            let kz = if self.dim == 3 { wavenumber(z, d) } else { 0.0 };
            let kk = (kx * kx + ky * ky + kz * kz).sqrt();
            let bin = (kk / dk).round() as usize;
            if bin < self.spectrum_len {
                self.spectrum[bin] += (buf[i].norm_sqr() as f64 * norm) as f32;
                self.spectrum_count[bin] += 1;
            }
        }
        for b in 0..self.spectrum_len {
            if self.spectrum_count[b] > 0 {
                self.spectrum[b] /= self.spectrum_count[b] as f32;
            }
        }
        self.cbuf = buf;
    }

    pub fn spectrum_ptr(&self) -> *const f32 {
        self.spectrum.as_ptr()
    }

    pub fn spectrum_len(&self) -> usize {
        self.spectrum_len
    }

    /// Ancho de bin del espectro, Δk = 2π/L.
    pub fn spectrum_dk(&self) -> f64 {
        let l_min = if self.dim == 3 { self.w.min(self.h).min(self.d) } else { self.w.min(self.h) };
        2.0 * PI / l_min as f64
    }

    /// Solo 3D: z del vóxel con E máxima en la columna (x, y); P/2 si la columna está vacía.
    pub fn mip_depth(&self, x: i32, y: i32) -> i32 {
        if self.dim == 2 {
            return 0;
        }
        let xx = crate::util::wrap_i(x, self.w);
        let yy = crate::util::wrap_i(y, self.h);
        let mut best = (0.0f32, (self.d / 2) as i32);
        for z in 0..self.d {
            let e = self.density(self.idx(xx, yy, z));
            if e > best.0 {
                best = (e, z as i32);
            }
        }
        if best.0 < 1e-6 {
            (self.d / 2) as i32
        } else {
            best.1
        }
    }

    /// arg(Σ_k ψ_k) en una celda.
    pub fn phase_at(&self, x: i32, y: i32, z: i32) -> f64 {
        let (sr, si) = self.sum_psi_at(self.wrap_cell(x, y, z));
        (si as f64).atan2(sr as f64)
    }

    /// E en una celda.
    pub fn density_at(&self, x: i32, y: i32, z: i32) -> f32 {
        self.density(self.wrap_cell(x, y, z))
    }

    /// Índice lineal de la celda con E máxima.
    pub fn peak_cell(&self) -> u32 {
        let mut best = (0.0f32, 0usize);
        for i in 0..self.n {
            let e = self.density(i);
            if e > best.0 {
                best = (e, i);
            }
        }
        best.1 as u32
    }

    /// Máximo de E dentro de una esfera de radio `radius` alrededor de (cx, cy, cz).
    pub fn max_density_in_sphere(&self, cx: f32, cy: f32, cz: f32, radius: f32) -> f32 {
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let p = [cx as f64, cy as f64, cz as f64];
        let mut m = 0.0f32;
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let c = [x as f64, y as f64, z as f64];
            let mut r2 = 0.0;
            for a in 0..self.dim {
                let dd = min_image(c[a] - p[a], dims[a]);
                r2 += dd * dd;
            }
            if r2 <= (radius as f64) * (radius as f64) {
                m = m.max(self.density(i));
            }
        }
        m
    }

    /// Norma dentro de una esfera de radio `radius` alrededor de (cx, cy, cz).
    pub fn norm_in_sphere(&self, cx: f32, cy: f32, cz: f32, radius: f32) -> f64 {
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let p = [cx as f64, cy as f64, cz as f64];
        let mut s = 0.0;
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let c = [x as f64, y as f64, z as f64];
            let mut r2 = 0.0;
            for a in 0..self.dim {
                let dd = min_image(c[a] - p[a], dims[a]);
                r2 += dd * dd;
            }
            if r2 <= (radius as f64) * (radius as f64) {
                s += self.density(i) as f64;
            }
        }
        s
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// arg Σ_{|r − c| < radius} (Σ_k ψ_k)(r)·e^{−ik·(r − c)}: la fase de un paquete en su centro,
    /// promediada sobre el núcleo (relojes que viajan con el paquete).
    #[allow(clippy::too_many_arguments)]
    pub fn local_phase(&self, cx: f32, cy: f32, cz: f32, radius: f32, kx: f32, ky: f32, kz: f32) -> f64 {
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let c = [cx as f64, cy as f64, cz as f64];
        let k = [kx as f64, ky as f64, kz as f64];
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let p = [x as f64, y as f64, z as f64];
            let mut r2 = 0.0;
            let mut th = 0.0;
            for a in 0..self.dim {
                let dd = min_image(p[a] - c[a], dims[a]);
                r2 += dd * dd;
                th -= k[a] * dd;
            }
            if r2 > (radius as f64).powi(2) {
                continue;
            }
            let (sr, si) = self.sum_psi_at(i);
            let (cs, sn) = (th.cos(), th.sin());
            re += sr as f64 * cs - si as f64 * sn;
            im += sr as f64 * sn + si as f64 * cs;
        }
        im.atan2(re)
    }

    /// Fase 14 (Rutter et al., PNAS 2026; Sharma et al., Nature 2023): indicador heurístico de
    /// ensamblaje y multiplicidad de copias sobre las componentes conexas por encima de 4·⟨E⟩.
    /// Distingue picos de 1–3 celdas (respiradores/artefactos con alto IPR pero a_i = 0) de
    /// solitones estructurados multicelda con copias repetidas n_c > 1.
    /// Devuelve [A_proxy, a_medio_multicelda, n_multicelda, n_picos_simples, clases_unicas, max_copias, ipr].
    pub fn analyze_soup_complexity(&mut self) -> Vec<f64> {
        let n = self.n;
        let mut sum_e = 0.0f64;
        let mut sum_e2 = 0.0f64;
        for i in 0..n {
            let e = self.density(i) as f64;
            sum_e += e;
            sum_e2 += e * e;
        }
        let ipr = if sum_e > 1e-12 { (n as f64) * sum_e2 / (sum_e * sum_e) } else { 0.0 };
        let thr = (4.0 * sum_e / n.max(1) as f64) as f32;
        if thr <= 1e-6 {
            return vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ipr];
        }
        let dims = [self.w as f64, self.h as f64, self.d as f64];
        let mut visited = vec![false; n];
        let mut stack: Vec<usize> = Vec::with_capacity(256);
        let mut comp_cells: Vec<usize> = Vec::with_capacity(256);
        let mut multi_count = 0usize;
        let mut small_count = 0usize;
        let mut sum_ai = 0.0f64;
        // Clases estructurales: (tamaño_bin, rg_bin, perfil_radial_cuantizado) -> (n_copias, suma_ai)
        let mut classes: std::collections::BTreeMap<(u8, u8, [u8; 4]), (usize, f64)> = std::collections::BTreeMap::new();

        for start in 0..n {
            if visited[start] || self.density(start) <= thr || (self.has_walls && self.wall[start] != 0) {
                continue;
            }
            visited[start] = true;
            stack.clear();
            comp_cells.clear();
            stack.push(start);
            let mut peak_val = 0.0f32;
            let mut peak_idx = start;
            let mut norm = 0.0f64;
            while let Some(ci) = stack.pop() {
                comp_cells.push(ci);
                let e = self.density(ci);
                norm += e as f64;
                if e > peak_val {
                    peak_val = e;
                    peak_idx = ci;
                }
                for nb in self.neighbors6(ci) {
                    if nb != usize::MAX && !visited[nb] && self.density(nb) > thr && !(self.has_walls && self.wall[nb] != 0) {
                        visited[nb] = true;
                        stack.push(nb);
                    }
                }
            }
            let nc = comp_cells.len();
            if nc < 4 {
                small_count += 1;
                continue;
            }
            multi_count += 1;
            let (px, py, pz) = self.coords(peak_idx);
            let pc = [px as f64, py as f64, pz as f64];
            let mut r2_sum = 0.0;
            // Perfil radial en 4 coronas (r² ∈ {0, 1..2, 3..5, 6..12}) relativo al pico
            let mut ring_e = [0.0f64; 4];
            let mut ring_c = [0usize; 4];
            let mut qseq: Vec<u8> = Vec::with_capacity(nc.min(64));
            for &ci in &comp_cells {
                let e = self.density(ci) as f64;
                let (cx, cy, cz) = self.coords(ci);
                let dx = min_image(cx as f64 - pc[0], dims[0]);
                let dy = min_image(cy as f64 - pc[1], dims[1]);
                let dz = if self.dim == 3 { min_image(cz as f64 - pc[2], dims[2]) } else { 0.0 };
                let r2 = dx * dx + dy * dy + dz * dz;
                r2_sum += e * r2;
                let rel = (e / (peak_val as f64).max(1e-12)).clamp(0.0, 1.0);
                let q = if rel > 0.72 { 3 } else if rel > 0.42 { 2 } else if rel > 0.18 { 1 } else { 0 };
                if qseq.len() < 64 {
                    qseq.push(q);
                }
                let rb = if r2 < 0.5 { 0 } else if r2 <= 2.5 { 1 } else if r2 <= 6.5 { 2 } else { 3 };
                ring_e[rb] += rel;
                ring_c[rb] += 1;
            }
            let rg = (r2_sum / norm.max(1e-12)).sqrt();
            // Complejidad morfológica LZ76 aproximada (pasos mínimos de construcción no triviales
            // de la secuencia de niveles y coronas ocupadas de la estructura)
            let mut shells_used = 0usize;
            let mut profile = [0u8; 4];
            for r in 0..4 {
                if ring_c[r] > 0 {
                    shells_used += 1;
                    let avg = ring_e[r] / ring_c[r] as f64;
                    profile[r] = ((avg * 3.0).round() as u8).min(3);
                }
            }
            let lz = lz76_steps(&qseq);
            let ai = ((shells_used.saturating_sub(1) as f64) + (lz as f64).ln_1p() + ((nc as f64) / 4.0).ln().max(0.0)).min(8.0);
            sum_ai += ai;
            let size_bin = ((nc as f64).log2().floor() as u8).min(10);
            let rg_bin = (rg.round() as u8).min(12);
            let entry = classes.entry((size_bin, rg_bin, profile)).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += ai;
        }

        let total_objects = multi_count + small_count;
        let mut a_proxy = 0.0;
        let mut max_copies = 0usize;
        for &(cnt, ai_sum) in classes.values() {
            max_copies = max_copies.max(cnt);
            if cnt > 1 {
                let a_mean = (ai_sum / cnt as f64).min(8.0);
                a_proxy += (a_mean.exp() - 1.0) * (cnt - 1) as f64;
            }
        }
        if total_objects > 0 {
            a_proxy /= total_objects as f64;
        }
        let mean_ai = if multi_count > 0 { sum_ai / multi_count as f64 } else { 0.0 };
        vec![a_proxy, mean_ai, multi_count as f64, small_count as f64, classes.len() as f64, max_copies as f64, ipr]
    }
}

/// Número de frases de Lempel–Ziv (LZ76) de una secuencia corta de símbolos cuantizados.
fn lz76_steps(s: &[u8]) -> usize {
    let n = s.len();
    if n == 0 {
        return 0;
    }
    let mut i = 0usize;
    let mut phrases = 0usize;
    while i < n {
        let mut best_len = 0usize;
        for start in 0..i {
            let mut l = 0usize;
            while i + l < n && s[start + l] == s[i + l] {
                l += 1;
            }
            if l > best_len {
                best_len = l;
            }
        }
        phrases += 1;
        i += best_len + 1;
    }
    phrases
}
