//! Utilidades comunes de los tests: universos con los valores de §3.3 y seguimiento de
//! solitones por centroide con peso E² en una ventana alrededor de la posición anterior.

use crate::util::{min_image, wrap_angle};
use crate::QuantumSpacetimeUniverse as U;

/// Universo 2D con m = 0,5, g = 0,1, E_max = 1, α = 0 y sin tope (las condiciones de §3.3).
pub fn base_2d(w: usize, h: usize) -> U {
    let mut u = U::new(w, h, 1, 0.5, 0.1, 1.0);
    u.set_neighbor_coupling(0.0);
    u.set_capacity(0.0);
    u
}

/// Universo 3D con m = 0,5, g = 0,1, E_max = 1, α = 0 y sin tope.
pub fn base_3d(w: usize, h: usize, d: usize) -> U {
    let mut u = U::new(w, h, d, 0.5, 0.1, 1.0);
    u.set_neighbor_coupling(0.0);
    u.set_capacity(0.0);
    u
}

/// Sigue un objeto compacto: centroide de E² en una esfera de radio `win` alrededor de la
/// última posición, con el desplazamiento acumulado sin envolver.
pub struct Tracker {
    pub pos: [f64; 3],
    pub unwrapped: [f64; 3],
    pub win: f64,
}

impl Tracker {
    pub fn new(x: f64, y: f64, z: f64, win: f64) -> Self {
        Tracker { pos: [x, y, z], unwrapped: [x, y, z], win }
    }

    pub fn update(&mut self, u: &U) -> [f64; 3] {
        let dims = [u.width() as f64, u.height() as f64, u.depth() as f64];
        let dim = u.dim() as usize;
        let mut np = self.pos;
        for a in 0..dim {
            np[a] = u.centroid_window(a as u32, self.pos[0] as f32, self.pos[1] as f32, self.pos[2] as f32, self.win as f32, 2);
        }
        for a in 0..dim {
            let dd = min_image(np[a] - self.pos[a], dims[a]);
            self.unwrapped[a] += dd;
        }
        self.pos = np;
        self.unwrapped
    }
}

/// Frecuencia interna de un objeto: pendiente de −arg Σψ en su pico durante `steps` pasos.
pub fn measure_omega(u: &mut U, steps: usize) -> f64 {
    let w = u.width();
    let h = u.height();
    let mut ts = Vec::with_capacity(steps);
    let mut ph = Vec::with_capacity(steps);
    let mut acc = 0.0;
    let mut last: Option<f64> = None;
    for s in 0..steps {
        let p = u.peak_cell() as usize;
        let (x, y, z) = (p % w, (p / w) % h, p / (w * h));
        let a = u.phase_at(x as i32, y as i32, z as i32);
        if let Some(l) = last {
            acc += wrap_angle(a - l);
        }
        last = Some(a);
        ts.push(s as f64);
        ph.push(acc);
        u.step();
    }
    let (_, slope) = crate::util::fit_linear(&ts, &ph);
    -slope
}

/// Pico de densidad.
pub fn peak(u: &U) -> f32 {
    u.max_density()
}

/// |Σ_t w_t·(y_t − ȳ)·e^{−iωt}|² con ventana de Hann.
pub fn hann_power(ys: &[f64], omega: f64) -> f64 {
    let n = ys.len();
    let mean = ys.iter().sum::<f64>() / n as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (t, y) in ys.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * t as f64 / (n - 1) as f64).cos();
        let (s, c) = (omega * t as f64).sin_cos();
        re += w * (y - mean) * c;
        im -= w * (y - mean) * s;
    }
    re * re + im * im
}

/// Frecuencia del pico más alto del espectro de Hann en [lo, hi] (barrido fino y sección áurea).
pub fn peak_frequency(ys: &[f64], lo: f64, hi: f64) -> f64 {
    let n = ys.len() as f64;
    let step = std::f64::consts::PI / (4.0 * n);
    let mut best = (f64::NEG_INFINITY, lo);
    let mut w = lo;
    while w <= hi {
        let p = hann_power(ys, w);
        if p > best.0 {
            best = (p, w);
        }
        w += step;
    }
    let (mut a, mut b) = ((best.1 - step).max(lo), (best.1 + step).min(hi));
    let g = 0.618_033_988_75;
    for _ in 0..60 {
        let c = b - g * (b - a);
        let d = a + g * (b - a);
        if hann_power(ys, c) > hann_power(ys, d) {
            b = d;
        } else {
            a = c;
        }
    }
    0.5 * (a + b)
}

/// Amplitudes √(a_k² + b_k²) del ajuste por mínimos cuadrados y = c + Σ_k a_k·cos ω_k t + b_k·sin ω_k t.
pub fn amplitudes_at(ys: &[f64], omegas: &[f64]) -> Vec<f64> {
    let m = 1 + 2 * omegas.len();
    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];
    for (t, y) in ys.iter().enumerate() {
        let mut f = vec![1.0];
        for w in omegas {
            let (s, c) = (w * t as f64).sin_cos();
            f.push(c);
            f.push(s);
        }
        for i in 0..m {
            aty[i] += f[i] * y;
            for j in 0..m {
                ata[i][j] += f[i] * f[j];
            }
        }
    }
    // eliminación gaussiana con pivote parcial
    for col in 0..m {
        let piv = (col..m).max_by(|&i, &j| ata[i][col].abs().partial_cmp(&ata[j][col].abs()).unwrap()).unwrap();
        ata.swap(col, piv);
        aty.swap(col, piv);
        for r in 0..m {
            if r != col {
                let f = ata[r][col] / ata[col][col];
                for k in col..m {
                    ata[r][k] -= f * ata[col][k];
                }
                aty[r] -= f * aty[col];
            }
        }
    }
    let p: Vec<f64> = (0..m).map(|i| aty[i] / ata[i][i]).collect();
    (0..omegas.len()).map(|k| p[1 + 2 * k].hypot(p[2 + 2 * k])).collect()
}

/// Dispersión del leapfrog con s subpasos: ω = 2s·arcsin(√(4κ·Σ sin²(k_a/2) + m²)/(2s)).
pub fn leapfrog_omega(kappa: f64, k: f64, m: f64, s: f64) -> f64 {
    2.0 * s * ((4.0 * kappa * (k / 2.0).sin().powi(2) + m * m).sqrt() / (2.0 * s)).asin()
}

/// Picos del espectro por autocorrelación por encima de `frac` del máximo.
pub fn spectrum_peaks(u: &mut U, frac: f64) -> Vec<f64> {
    let nb = u.autocorrelation_spectrum() as usize;
    let sp = &u.autocorr.spec;
    let smax = (0..nb).map(|j| sp[2 * j + 1] as f64).fold(0.0, f64::max);
    let mut peaks = Vec::new();
    for j in 1..nb - 1 {
        let s = sp[2 * j + 1] as f64;
        if s > frac * smax && s >= sp[2 * j - 1] as f64 && s > sp[2 * j + 3] as f64 {
            peaks.push(sp[2 * j] as f64);
        }
    }
    peaks
}
