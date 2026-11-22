//! El fermión de Weyl y el teorema de Nielsen–Ninomiya (§2.17 y §3.14): el operador de Floquet
//! F(k) = a₀ − i·a·σ ∈ SU(2) de las tres variantes, sus nodos (a = 0) con su quiralidad, el
//! número de arrollamiento W₃ (el grado de F: T³ → S³) y la rama quiral en un campo B.
//! La quiralidad neta en cada cuasi-energía es W₃, que vale 0 en la zona completa: el compañero de
//! cada nodo está en la misma cuasi-energía. Un Weyl sin pareja solo aparece en media zona
//! (Higashikawa et al.), que no es un subespacio local.

use crate::fftn::wavenumber;
use crate::rng::SplitMix64;
use crate::spinor::{cx, M4};
use crate::sun::C;
use crate::QuantumSpacetimeUniverse;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

type M2 = [[C; 2]; 2];

fn m2_of(m: &M4) -> M2 {
    [[m[0][0], m[0][1]], [m[1][0], m[1][1]]]
}

fn m2_mul(a: &M2, b: &M2) -> M2 {
    let mut r = [[cx(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    r
}

fn m2_dag(a: &M2) -> M2 {
    [[a[0][0].conj(), a[1][0].conj()], [a[0][1].conj(), a[1][1].conj()]]
}

fn wrapf(a: f64) -> f64 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

impl QuantumSpacetimeUniverse {
    /// F(k) del período de la caminata de Weyl (1 paso con el orden fijo y los bombeos; 2 con el
    /// orden alternado).
    pub(crate) fn weyl_f(&self, k: [f64; 3]) -> M2 {
        m2_of(&self.sw_period_matrix(0, k))
    }

    /// (a₀, a₁, a₂, a₃) de F = a₀ − i·a·σ.
    pub(crate) fn weyl_a(&self, k: [f64; 3]) -> [f64; 4] {
        let f = self.weyl_f(k);
        // F = [[a₀ − ia₃, −a₂ − ia₁], [a₂ − ia₁, a₀ + ia₃]]
        [0.5 * (f[0][0].re + f[1][1].re), -0.5 * (f[1][0].im + f[0][1].im), 0.5 * (f[1][0].re - f[0][1].re), 0.5 * (f[1][1].im - f[0][0].im)]
    }

    /// det(∂a/∂k) por diferencias centradas.
    pub(crate) fn weyl_det_da(&self, k: [f64; 3]) -> f64 {
        let h = 1e-6;
        let mut j = [[0.0f64; 3]; 3];
        for c in 0..3 {
            let mut kp = k;
            kp[c] += h;
            let mut km = k;
            km[c] -= h;
            let (ap, am) = (self.weyl_a(kp), self.weyl_a(km));
            for r in 0..3 {
                j[r][c] = (ap[r + 1] - am[r + 1]) / (2.0 * h);
            }
        }
        j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
    }

    /// W₃ = −(1/24π²)∫ε^{ijk}·Tr(R_iR_jR_k), R_i = F†∂_iF, con la regla del punto medio en una malla
    /// de n³ (n/2 planos en k_z con `half`, la media zona |k_z| < π/2) y R_i por diferencias centradas.
    pub(crate) fn weyl_w3(&self, n: usize, half: bool) -> f64 {
        let dk = 2.0 * PI / n as f64;
        let h = 1e-5;
        let nz = if half { n / 2 } else { n };
        let z0 = if half { -0.5 * PI } else { -PI };
        let mut acc = 0.0;
        for iz in 0..nz {
            for iy in 0..n {
                for ix in 0..n {
                    let k = [-PI + (ix as f64 + 0.5) * dk, -PI + (iy as f64 + 0.5) * dk, z0 + (iz as f64 + 0.5) * dk];
                    let f = self.weyl_f(k);
                    let fd = m2_dag(&f);
                    let mut r = [[[cx(0.0, 0.0); 2]; 2]; 3];
                    for c in 0..3 {
                        let mut kp = k;
                        kp[c] += h;
                        let mut km = k;
                        km[c] -= h;
                        let (fp, fm) = (self.weyl_f(kp), self.weyl_f(km));
                        let mut d = [[cx(0.0, 0.0); 2]; 2];
                        for i in 0..2 {
                            for j in 0..2 {
                                d[i][j] = (fp[i][j] - fm[i][j]) / (2.0 * h);
                            }
                        }
                        r[c] = m2_mul(&fd, &d);
                    }
                    let tr = |a: &M2| a[0][0] + a[1][1];
                    let xyz = tr(&m2_mul(&r[0], &m2_mul(&r[1], &r[2])));
                    let xzy = tr(&m2_mul(&r[0], &m2_mul(&r[2], &r[1])));
                    acc += 3.0 * (xyz - xzy).re;
                }
            }
        }
        -acc * dk * dk * dk / (24.0 * PI * PI)
    }

    /// Resuelve a(k) = 0 por Newton desde k₀ (Jacobiano por diferencias); None si no converge.
    fn weyl_newton(&self, mut k: [f64; 3]) -> Option<[f64; 3]> {
        let h = 1e-7;
        for _ in 0..60 {
            let a = self.weyl_a(k);
            let nrm = (a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt();
            if nrm < 1e-13 {
                return Some(k);
            }
            let mut j = [[0.0f64; 3]; 3];
            for c in 0..3 {
                let mut kp = k;
                kp[c] += h;
                let mut km = k;
                km[c] -= h;
                let (ap, am) = (self.weyl_a(kp), self.weyl_a(km));
                for r in 0..3 {
                    j[r][c] = (ap[r + 1] - am[r + 1]) / (2.0 * h);
                }
            }
            let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
            if det.abs() < 1e-10 {
                return None;
            }
            let dx = crate::util::solve3(j, [-a[1], -a[2], -a[3]]);
            let step = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
            let s = if step > 0.5 { 0.5 / step } else { 1.0 };
            for c in 0..3 {
                k[c] = wrapf(k[c] + s * dx[c]);
            }
        }
        None
    }

    /// Paquete del escenario Weyl 3D (orden fijo o bombeos) filtrado sobre la rama quiral de un nodo
    /// en el campo B uniforme actual (§3.14): cada componente k_z del gaussiano se filtra con el
    /// operador magnético 2D en ese k_z, Σ_t h_t·e^{iε_c t}·F^t, con h_t = sin(w·t)/(πt) por una ventana
    /// de Hann en |t| ≤ T: deja la parte con |ε − ε_c| < w, con ε_c = χ·(k_z − k_z,nodo).
    /// Devuelve el paquete filtrado (con la norma de `set_packet_norm`) y la fracción de la norma
    /// del gaussiano que se lleva la rama.
    pub(crate) fn weyl_filtered_packet(&mut self, c0: [f64; 3], sigma: f64, knode: [f64; 3], chirality: i32, half_width: f64, t_max: usize) -> (Vec<f64>, f64) {
        let (w, h, d) = (self.w, self.h, self.d);
        let plane = w * h;
        let raw = self.sw_build_packet(c0, [sigma; 3], knode, 1, [cx(1.0, 0.0), cx(0.0, 0.0), cx(0.0, 0.0)], false, 1.0);
        let namp = 2;
        // FFT en z de cada (x, y, componente)
        let mut planner = rustfft::FftPlanner::<f64>::new();
        let fwd = planner.plan_fft_forward(d);
        let inv = planner.plan_fft_inverse(d);
        let mut spec: Vec<Vec<Complex<f64>>> = vec![vec![cx(0.0, 0.0); d]; plane * namp];
        for p in 0..plane {
            for c in 0..namp {
                let line = &mut spec[p * namp + c];
                for z in 0..d {
                    let i = z * plane + p;
                    line[z] = cx(raw[i * 2 * namp + 2 * c], raw[i * 2 * namp + 2 * c + 1]);
                }
                fwd.process(line);
            }
        }
        // universo auxiliar 2D (corte con k_z fijo) con el mismo campo
        let mut aux = QuantumSpacetimeUniverse::new(w, h, 1, self.mass, 0.0, 1.0);
        aux.set_scenario(3);
        aux.sw.variant = self.sw.variant;
        aux.sw.rebuild();
        if self.em.mode != 0 {
            aux.em.mode = 1;
            aux.em.q = self.em.q;
            for a in 0..2 {
                for p in 0..plane {
                    aux.em.a[a * plane + p] = self.em.a[a * self.n + p];
                }
            }
            aux.sw_refresh_phases();
        }
        let ht: Vec<f64> = (0..=t_max)
            .map(|t| {
                let s = if t == 0 { half_width / PI } else { (half_width * t as f64).sin() / (PI * t as f64) };
                s * 0.5 * (1.0 + (PI * t as f64 / (t_max as f64 + 1.0)).cos())
            })
            .collect();
        let mut tmp = vec![0.0f64; plane * 2 * namp];
        for j in 0..d {
            let kz = wavenumber(j, d);
            let eps_c = chirality as f64 * wrapf(kz - knode[2]);
            aux.sw.kz = kz;
            let mut psi = vec![0.0f64; plane * 2 * namp];
            for p in 0..plane {
                for c in 0..namp {
                    let v = spec[p * namp + c][j];
                    psi[p * 2 * namp + 2 * c] = v.re;
                    psi[p * 2 * namp + 2 * c + 1] = v.im;
                }
            }
            if psi.iter().all(|v| v.abs() < 1e-300) {
                continue;
            }
            let mut acc: Vec<f64> = psi.iter().map(|v| v * ht[0]).collect();
            for dir in [1i32, -1] {
                let mut f = psi.clone();
                for t in 1..=t_max {
                    if dir > 0 {
                        aux.sw_step_buffer(&mut f, &mut tmp, 0);
                    } else {
                        aux.sw_step_back_buffer(&mut f, &mut tmp, 0);
                    }
                    let ph = Complex::from_polar(ht[t], eps_c * dir as f64 * t as f64);
                    for q in 0..plane * namp {
                        let v = cx(f[2 * q], f[2 * q + 1]) * ph;
                        acc[2 * q] += v.re;
                        acc[2 * q + 1] += v.im;
                    }
                }
            }
            for p in 0..plane {
                for c in 0..namp {
                    spec[p * namp + c][j] = cx(acc[p * 2 * namp + 2 * c], acc[p * 2 * namp + 2 * c + 1]);
                }
            }
        }
        let mut out = vec![0.0f64; self.n * 2 * namp];
        for p in 0..plane {
            for c in 0..namp {
                let line = &mut spec[p * namp + c];
                inv.process(line);
                for z in 0..d {
                    let i = z * plane + p;
                    out[i * 2 * namp + 2 * c] = line[z].re;
                    out[i * 2 * namp + 2 * c + 1] = line[z].im;
                }
            }
        }
        let nrm: f64 = out.iter().map(|v| v * v).sum();
        let s = if nrm > 0.0 { (self.packet_norm / nrm).sqrt() } else { 0.0 };
        for v in out.iter_mut() {
            *v *= s;
        }
        // la FFT de rustfft no normaliza: ida y vuelta multiplican por D
        (out, nrm / (d * d) as f64)
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// W₃ de la variante de Weyl actual (§3.14): malla de n³ (n par), en la zona completa o en la
    /// media zona |k_z| < π/2.
    pub fn weyl_winding(&self, grid: u32, half_zone: bool) -> f64 {
        if !self.sw.weyl {
            return f64::NAN;
        }
        self.weyl_w3((grid.max(4) as usize) & !1, half_zone)
    }

    /// Nodos de F (a = 0) en la zona cúbica, por Newton desde `seeds` semillas aleatorias,
    /// deduplicados módulo 2π: n × 5 valores [k_x, k_y, k_z, sector (0 o π), quiralidad]. La
    /// quiralidad es el signo de det(∂a/∂k), invertido en los nodos π. Los nodos con det ≈ 0 (los
    /// planos degenerados en π de los bombeos) no se listan.
    pub fn weyl_nodes(&self, seeds: u32, seed: u32) -> Vec<f64> {
        if !self.sw.weyl {
            return Vec::new();
        }
        let mut rng = SplitMix64::new(seed as u64 ^ 0x9E3779B9);
        let mut found: Vec<[f64; 5]> = Vec::new();
        for _ in 0..seeds {
            let k0 = [PI * (2.0 * rng.next_f64() - 1.0), PI * (2.0 * rng.next_f64() - 1.0), PI * (2.0 * rng.next_f64() - 1.0)];
            if let Some(k) = self.weyl_newton(k0) {
                let det = self.weyl_det_da(k);
                if det.abs() < 1e-6 {
                    continue;
                }
                let a0 = self.weyl_a(k)[0];
                let sector = if a0 > 0.0 { 0.0 } else { PI };
                let chir = if a0 > 0.0 { det.signum() } else { -det.signum() };
                let dup = found.iter().any(|f| (0..3).all(|c| wrapf(f[c] - k[c]).abs() < 1e-6));
                if !dup {
                    found.push([wrapf(k[0]), wrapf(k[1]), wrapf(k[2]), sector, chir]);
                }
            }
        }
        found.sort_by(|a, b| (a[3], a[4], a[0], a[1], a[2]).partial_cmp(&(b[3], b[4], b[0], b[1], b[2])).unwrap());
        found.into_iter().flatten().collect()
    }

    /// Escenario Weyl 3D (orden fijo o bombeos, con B uniforme): paquete gaussiano isótropo de
    /// anchura σ con portadora en el nodo (k_x, k_y, k_z), filtrado sobre la rama quiral de
    /// quiralidad χ (ventana |ε − ε_c| < half_width, T = steps pasos a cada lado). Usa el espín de
    /// `set_packet_spin` y la norma de `set_packet_norm`.
    pub fn inject_weyl_chiral_packet(&mut self, x: f32, y: f32, z: f32, sigma: f32, kx: f32, ky: f32, kz: f32, chirality: i32, half_width: f32, steps: u32) {
        if !self.sw.active || !self.sw.weyl || self.dim != 3 || self.sw.variant == 1 {
            self.set_message("El filtro de la rama quiral necesita el escenario Weyl 3D con orden fijo o bombeos.");
            return;
        }
        let c0 = [x as f64, y as f64, z as f64];
        let (b, _) = self.weyl_filtered_packet(c0, sigma as f64, [kx as f64, ky as f64, kz as f64], chirality.signum(), half_width as f64, steps.max(8) as usize);
        self.sw_add_buffer(b, c0);
    }

    /// Fracción de la norma de un gaussiano (sin proyectar, con el espín de `set_packet_spin`) que
    /// se lleva la rama quiral del nodo dado, con el mismo filtro que `inject_weyl_chiral_packet`.
    pub fn weyl_chiral_fraction(&mut self, x: f32, y: f32, z: f32, sigma: f32, kx: f32, ky: f32, kz: f32, chirality: i32, half_width: f32, steps: u32) -> f64 {
        if !self.sw.active || !self.sw.weyl || self.dim != 3 || self.sw.variant == 1 {
            return f64::NAN;
        }
        let c0 = [x as f64, y as f64, z as f64];
        self.weyl_filtered_packet(c0, sigma as f64, [kx as f64, ky as f64, kz as f64], chirality.signum(), half_width as f64, steps.max(8) as usize).1
    }

    /// Rama adiabática exacta del escenario Weyl en 2D (k_z fijo, con el campo estático actual): se
    /// diagonaliza el operador del corte en k_z creciente y se sigue el estado por máximo solapamiento
    /// con el anterior (proyectado sobre el grupo de autovectores degenerados), empezando por el
    /// autovalor más cercano a eps0. El paso empieza en dk y se divide entre 2 (hasta dk/32) cuando
    /// el solapamiento baja de 0,98, que es lo que pasa junto a los cruces. Devuelve
    /// [k_z, ε, solapamiento] por punto aceptado. Cada punto cuesta una diagonalización de 2N × 2N.
    pub fn weyl_adiabatic_branch(&mut self, kz0: f32, kz1: f32, dk: f32, eps0: f32) -> Vec<f64> {
        if !self.sw.weyl || self.dim != 2 || self.sw.variant == 1 {
            return Vec::new();
        }
        let n2 = 2 * self.n;
        let saved = self.sw.kz;
        if self.em.mode != 0 {
            self.sw_refresh_phases();
        }
        let mut tmp = vec![0.0f64; 2 * n2];
        let mut spectrum = |u: &mut QuantumSpacetimeUniverse, kz: f64| -> crate::linalg::Eigen {
            u.sw.kz = kz;
            let mut a = vec![cx(0.0, 0.0); n2 * n2];
            for col in 0..n2 {
                let mut v = vec![0.0f64; 2 * n2];
                v[2 * col] = 1.0;
                u.sw_step_buffer(&mut v, &mut tmp, 0);
                for row in 0..n2 {
                    a[row * n2 + col] = cx(v[2 * row], v[2 * row + 1]);
                }
            }
            crate::linalg::eig(a, n2)
        };
        // proyección de p sobre el grupo degenerado del autovalor i: (vector normalizado, |proyección|)
        let project = |e: &crate::linalg::Eigen, i: usize, p: &[C]| -> (Vec<C>, f64) {
            let li = e.values[i];
            let mut basis: Vec<Vec<C>> = Vec::new();
            for j in 0..e.values.len() {
                if (e.values[j] - li).norm() > 1e-6 {
                    continue;
                }
                let mut v = e.vectors[j].clone();
                for b in &basis {
                    let ov: C = b.iter().zip(v.iter()).map(|(x, y)| x.conj() * y).sum();
                    for (vv, bb) in v.iter_mut().zip(b.iter()) {
                        *vv -= ov * bb;
                    }
                }
                let nn: f64 = v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
                if nn > 1e-8 {
                    basis.push(v.iter().map(|c| c / nn).collect());
                }
            }
            let mut proj = vec![cx(0.0, 0.0); p.len()];
            for b in &basis {
                let ov: C = b.iter().zip(p.iter()).map(|(x, y)| x.conj() * y).sum();
                for (pp, bb) in proj.iter_mut().zip(b.iter()) {
                    *pp += ov * bb;
                }
            }
            let nn: f64 = proj.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            (proj.iter().map(|c| c / nn.max(1e-300)).collect(), nn)
        };
        let mut out = Vec::new();
        let (k0, k1) = (kz0 as f64, kz1 as f64);
        let e = spectrum(self, k0);
        let i0 = (0..n2).min_by(|&i, &j| (-e.values[i].arg() - eps0 as f64).abs().partial_cmp(&(-e.values[j].arg() - eps0 as f64).abs()).unwrap()).unwrap();
        let mut prev = e.vectors[i0].clone();
        out.extend([k0, -e.values[i0].arg(), 1.0]);
        let (dk, dir) = ((dk as f64).abs().max(1e-4), if k1 >= k0 { 1.0 } else { -1.0 });
        let mut kz = k0;
        let mut h = dk;
        while (k1 - kz) * dir > 1e-9 {
            let step = h.min((k1 - kz).abs());
            let e = spectrum(self, kz + dir * step);
            let mut best = (0usize, 0.0f64);
            for i in 0..n2 {
                let ov: C = e.vectors[i].iter().zip(prev.iter()).map(|(a, b)| b.conj() * a).sum();
                if ov.norm() > best.1 {
                    best = (i, ov.norm());
                }
            }
            let (v, ov) = project(&e, best.0, &prev);
            if ov < 0.98 && h > dk / 32.0 {
                h *= 0.5;
                continue;
            }
            kz += dir * step;
            out.extend([kz, -e.values[best.0].arg(), ov]);
            prev = v;
            h = (2.0 * h).min(dk);
        }
        self.sw.kz = saved;
        out
    }

    /// Bombeo E ∥ B en el escenario Weyl 2D (§3.14): el estado de la rama quiral en kz0 (el
    /// autovector del corte más cercano a ε = kz0) evoluciona con k_z(t) = kz0 + rate·t. Cada
    /// `every` pasos se guardan [k_z, ε, v_z], con ε = −arg⟨ψ|F(k_z)|ψ⟩ (la cuasi-energía
    /// instantánea) y v_z el flujo exacto del subpaso z (la velocidad media).
    pub fn weyl_pump_run(&mut self, kz0: f32, rate: f32, steps: u32, every: u32) -> Vec<f64> {
        if !self.sw.weyl || self.dim != 2 || self.sw.variant == 1 {
            return Vec::new();
        }
        let n2 = 2 * self.n;
        if self.em.mode != 0 {
            self.sw_refresh_phases();
        }
        self.sw.kz = kz0 as f64;
        let mut tmp = vec![0.0f64; 2 * n2];
        let mut a = vec![cx(0.0, 0.0); n2 * n2];
        for col in 0..n2 {
            let mut v = vec![0.0f64; 2 * n2];
            v[2 * col] = 1.0;
            self.sw_step_buffer(&mut v, &mut tmp, 0);
            for row in 0..n2 {
                a[row * n2 + col] = cx(v[2 * row], v[2 * row + 1]);
            }
        }
        let e = crate::linalg::eig(a, n2);
        let i0 = (0..n2).min_by(|&i, &j| (-e.values[i].arg() - kz0 as f64).abs().partial_cmp(&(-e.values[j].arg() - kz0 as f64).abs()).unwrap()).unwrap();
        for (k, c) in e.vectors[i0].iter().enumerate() {
            self.sw.psi[2 * k] = c.re;
            self.sw.psi[2 * k + 1] = c.im;
        }
        self.sw.track_flux = true;
        let mut out = Vec::new();
        let ev = every.max(1);
        for t in 0..=steps {
            self.sw.kz = kz0 as f64 + rate as f64 * t as f64;
            if t % ev == 0 {
                let eps = self.spinor_matter_energy();
                out.extend([self.sw.kz, eps, self.sw.flux[2]]);
            }
            if t < steps {
                self.step();
            }
        }
        self.sw_mirror();
        out
    }

    /// Anchura en el eje: √⟨(x − x̄)²⟩ con x̄ la media circular (el paquete puede cruzar la costura).
    pub fn spinor_width(&self, axis: u32) -> f64 {
        let a = (axis as usize).min(2);
        let l = [self.w as f64, self.h as f64, self.d as f64][a];
        let st = 2 * self.sw_namp();
        let dens: Vec<f64> = (0..self.n).map(|i| self.sw.psi[i * st..(i + 1) * st].iter().map(|v| v * v).sum()).collect();
        let pos = |i: usize| -> f64 {
            let (x, y, z) = self.coords(i);
            [x as f64, y as f64, z as f64][a]
        };
        let (mut cs, mut sn, mut tot) = (0.0, 0.0, 0.0);
        for i in 0..self.n {
            let th = 2.0 * PI * pos(i) / l;
            cs += dens[i] * th.cos();
            sn += dens[i] * th.sin();
            tot += dens[i];
        }
        if tot <= 0.0 {
            return 0.0;
        }
        let c = sn.atan2(cs) * l / (2.0 * PI);
        let mut acc = 0.0;
        for i in 0..self.n {
            let dx = crate::util::min_image(pos(i) - c, l);
            acc += dens[i] * dx * dx;
        }
        (acc / tot).sqrt()
    }
}
