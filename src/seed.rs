//! Condiciones iniciales y edición de la malla: sopa primordial, gotas de energía, paquetes
//! con momento, barreras de potencial, muros y trampa armónica (§2.4, §4.3).

use crate::rng::SplitMix64;
use crate::util::{min_image, wrap_i};
use crate::QuantumSpacetimeUniverse;
use wasm_bindgen::prelude::*;

impl QuantumSpacetimeUniverse {
    /// Box-blur toroidal separable de radio `r` a lo largo de un eje, sobre un campo complejo
    /// guardado como pares (Re, Im) en `field` (2·N floats). `line` es un búfer auxiliar.
    pub(crate) fn box_blur_axis(&self, field: &mut [f32], line: &mut [f32], axis: usize, r: usize) {
        let (w, h, d) = (self.w, self.h, self.d);
        let (len, count_a, count_b) = match axis {
            0 => (w, h, d),
            1 => (h, w, d),
            _ => (d, w, h),
        };
        if len <= 1 {
            return;
        }
        let win = (2 * r + 1) as f32;
        for b in 0..count_b {
            for a in 0..count_a {
                let index = |s: usize| -> usize {
                    match axis {
                        0 => (b * h + a) * w + s,
                        1 => (b * h + s) * w + a,
                        _ => (s * h + b) * w + a,
                    }
                };
                for s in 0..len {
                    let i = index(s);
                    line[2 * s] = field[2 * i];
                    line[2 * s + 1] = field[2 * i + 1];
                }
                // suma móvil de la ventana [s − r, s + r] con envoltura
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for o in 0..(2 * r + 1) {
                    let s = (o as i64 - r as i64).rem_euclid(len as i64) as usize;
                    sr += line[2 * s] as f64;
                    si += line[2 * s + 1] as f64;
                }
                for s in 0..len {
                    let i = index(s);
                    field[2 * i] = (sr / win as f64) as f32;
                    field[2 * i + 1] = (si / win as f64) as f32;
                    let out = (s as i64 - r as i64).rem_euclid(len as i64) as usize;
                    let inn = (s as i64 + r as i64 + 1).rem_euclid(len as i64) as usize;
                    sr += line[2 * inn] as f64 - line[2 * out] as f64;
                    si += line[2 * inn + 1] as f64 - line[2 * out + 1] as f64;
                }
            }
        }
    }

    /// Envolvente gaussiana normalizada: devuelve la lista implícita de celdas a menos de 3σ
    /// y su peso, llamando a `f(celda, desplazamiento, envolvente)`.
    pub(crate) fn for_gaussian_cells<F: FnMut(usize, [f64; 3], f64)>(&self, cx: f64, cy: f64, cz: f64, sigma: f64, mut f: F) {
        let rc = (3.0 * sigma).ceil().max(1.0) as i32;
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let x0 = cx.round() as i32;
        let y0 = cy.round() as i32;
        let z0 = if self.dim == 3 { cz.round() as i32 } else { 0 };
        let zr = if self.dim == 3 { rc } else { 0 };
        let rc2 = (3.0 * sigma) * (3.0 * sigma);
        // evita visitar dos veces la misma celda si 3σ supera media malla
        let rx = rc.min((self.w as i32 - 1) / 2);
        let ry = rc.min((self.h as i32 - 1) / 2);
        let rz = zr.min((self.d as i32 - 1) / 2);
        for dz in -rz..=rz {
            for dy in -ry..=ry {
                for dx in -rx..=rx {
                    let xi = x0 + dx;
                    let yi = y0 + dy;
                    let zi = z0 + dz;
                    let ddx = min_image(xi as f64 - cx, w);
                    let ddy = min_image(yi as f64 - cy, h);
                    let ddz = if self.dim == 3 { min_image(zi as f64 - cz, d) } else { 0.0 };
                    let r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                    if r2 > rc2 {
                        continue;
                    }
                    let cell = self.wrap_cell(xi, yi, zi);
                    if self.has_walls && self.wall[cell] != 0 {
                        continue;
                    }
                    f(cell, [ddx, ddy, ddz], (-r2 / (2.0 * sigma * sigma)).exp());
                }
            }
        }
    }

    /// Suma un paquete φ(r) = A·env(r)·e^{iθ(r)} con el espinor simétrico (1, …, 1)/√n_c,
    /// normalizado para que la norma añadida sobre un campo vacío sea `amount`.
    pub(crate) fn add_symmetric_packet(&mut self, cx: f64, cy: f64, cz: f64, sigma: f64, amount: f64, k: [f64; 3], phase0: f64) {
        if self.ncol > 1 {
            // con SU(N), el color 0 de la base (§4.3) y la fase global como factor
            let mut col = crate::sun::basis_colour(self.ncol, 0);
            col[0] = rustfft::num_complex::Complex::from_polar(1.0, phase0);
            self.add_colour_packet(cx, cy, cz, sigma, amount, k, col);
            return;
        }
        let mut sum = 0.0f64;
        self.for_gaussian_cells(cx, cy, cz, sigma, |_, _, e| sum += e * e);
        if sum <= 0.0 || amount <= 0.0 {
            return;
        }
        let amp = (amount / sum).sqrt();
        let nc = self.nc;
        let stride = 2 * nc;
        let inv = 1.0 / (nc as f64).sqrt();
        // Con U(1), k es el momento cinético: se suma la fase gauge local q·A(r₀)·(r − r₀).
        let ga = self.matter_gauge_phase(cx, cy, cz);
        let k = [k[0] + ga[0], k[1] + ga[1], k[2] + ga[2]];
        // Sin asignaciones: se toma prestado el búfer mientras se recorre la envolvente.
        let mut buf = std::mem::take(&mut self.buffer_read);
        self.for_gaussian_cells(cx, cy, cz, sigma, |c, r, e| {
            let th = phase0 + k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
            let a = amp * e * inv;
            let (re, im) = ((a * th.cos()) as f32, (a * th.sin()) as f32);
            let b = c * stride;
            for ch in 0..nc {
                buf[b + 2 * ch] += re;
                buf[b + 2 * ch + 1] += im;
            }
        });
        self.buffer_read = buf;
    }

    /// q·A(r₀) promediado en los enlaces de la celda (2D o 3D).
    pub(crate) fn matter_gauge_phase(&self, cx: f64, cy: f64, cz: f64) -> [f64; 3] {
        if self.em.mode == 0 {
            return [0.0; 3];
        }
        let c = self.wrap_cell(cx.round() as i32, cy.round() as i32, cz.round() as i32);
        let n = self.n;
        let q = self.em.q;
        let mut out = [0.0; 3];
        for a in 0..self.dim {
            out[a] = q * 0.5 * (self.em.a[a * n + c] + self.em.a[a * n + self.nb(c, a, -1)]);
        }
        out
    }

    pub(crate) fn sum_psi_at(&self, cell: usize) -> (f32, f32) {
        let stride = self.stride();
        let b = cell * stride;
        let (mut sr, mut si) = (0.0, 0.0);
        for k in 0..self.nc * self.ncol {
            sr += self.buffer_read[b + 2 * k];
            si += self.buffer_read[b + 2 * k + 1];
        }
        (sr, si)
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Sopa primordial (§4.3): ruido blanco complejo, 3 pasadas de box-blur toroidal por eje
    /// (radio 6 en 2D y 4 en 3D), normalizado a ⟨|φ|²⟩ = energy_level y ψ_k = φ/√n_c en todos
    /// los canales. Reemplaza el campo y pone t = 0.
    pub fn seed_random_soup(&mut self, energy_level: f32, seed: u32) {
        let n = self.n;
        let level = if energy_level.is_finite() { energy_level.max(0.0) as f64 } else { 0.0 };
        let mut rng = SplitMix64::new(seed as u64);
        let mut field = std::mem::take(&mut self.buffer_write);
        for i in 0..n {
            field[2 * i] = rng.gauss() as f32;
            field[2 * i + 1] = rng.gauss() as f32;
        }
        let r = if self.dim == 3 { 4 } else { 6 };
        let mut line = vec![0.0f32; 2 * self.w.max(self.h).max(self.d)];
        for axis in 0..self.dim {
            for _ in 0..3 {
                self.box_blur_axis(&mut field, &mut line, axis, r);
            }
        }
        let mut s = 0.0f64;
        let mut cnt = 0usize;
        for i in 0..n {
            if self.has_walls && self.wall[i] != 0 {
                continue;
            }
            s += (field[2 * i] as f64).powi(2) + (field[2 * i + 1] as f64).powi(2);
            cnt += 1;
        }
        let scale = if s > 0.0 { (level * cnt as f64 / s).sqrt() } else { 0.0 };
        let nc = self.nc;
        let nco = self.ncol;
        let stride = 2 * nc * nco;
        let inv = 1.0 / (nc as f64).sqrt();
        self.buffer_read.iter_mut().for_each(|v| *v = 0.0);
        // Con color: φ·χ(r)/√n_c, con χ un vector de color unitario aleatorio y suavizado (§2.11).
        let mut chi = vec![0.0f32; if nco > 1 { 2 * n * nco } else { 0 }];
        if nco > 1 {
            let mut tmp = vec![0.0f32; 2 * n];
            for a in 0..nco {
                for v in tmp.iter_mut() {
                    *v = rng.gauss() as f32;
                }
                for axis in 0..self.dim {
                    for _ in 0..3 {
                        self.box_blur_axis(&mut tmp, &mut line, axis, r);
                    }
                }
                for i in 0..n {
                    chi[2 * (i * nco + a)] = tmp[2 * i];
                    chi[2 * (i * nco + a) + 1] = tmp[2 * i + 1];
                }
            }
        }
        for i in 0..n {
            if self.has_walls && self.wall[i] != 0 {
                continue;
            }
            let re = field[2 * i] as f64 * scale * inv;
            let im = field[2 * i + 1] as f64 * scale * inv;
            let b = i * stride;
            if nco == 1 {
                for k in 0..nc {
                    self.buffer_read[b + 2 * k] = re as f32;
                    self.buffer_read[b + 2 * k + 1] = im as f32;
                }
            } else {
                let mut cn = 0.0f64;
                for a in 0..nco {
                    cn += (chi[2 * (i * nco + a)] as f64).powi(2) + (chi[2 * (i * nco + a) + 1] as f64).powi(2);
                }
                let cn = cn.sqrt().max(1e-30);
                for a in 0..nco {
                    let (cr, ci) = (chi[2 * (i * nco + a)] as f64 / cn, chi[2 * (i * nco + a) + 1] as f64 / cn);
                    let (vr, vi) = (re * cr - im * ci, re * ci + im * cr);
                    for k in 0..nc {
                        self.buffer_read[b + 2 * (k * nco + a)] = vr as f32;
                        self.buffer_read[b + 2 * (k * nco + a) + 1] = vi as f32;
                    }
                }
            }
        }
        field.iter_mut().for_each(|v| *v = 0.0);
        self.buffer_write = field;
        self.t = 0;
        self.theta_ref = 0.0;
        self.sol.reset();
        self.reset_capacity_stats();
    }

    /// Gota de energía: envolvente gaussiana (σ = radius/2, truncada a 3σ), espinor simétrico y
    /// la fase de la celda central (inyección constructiva). Añade exactamente `amount` de norma
    /// sobre un campo vacío.
    pub fn inject_energy_blob(&mut self, x: i32, y: i32, z: i32, radius: f32, amount: f32) {
        if !(radius.is_finite() && amount.is_finite()) || radius <= 0.0 || amount <= 0.0 {
            return;
        }
        let c = self.wrap_cell(x, y, z);
        let (sr, si) = self.sum_psi_at(c);
        let ph = if sr * sr + si * si > 1e-24 { (si as f64).atan2(sr as f64) } else { 0.0 };
        let (cx, cy, cz) = self.coords(c);
        self.add_symmetric_packet(cx as f64, cy as f64, cz as f64, radius as f64 / 2.0, amount as f64, [0.0; 3], ph);
    }

    /// Como `inject_energy_blob`, con el factor de fase e^{i(k·(r − r₀)) + i·phase}. Mantén |k| ≲ 0,6.
    #[allow(clippy::too_many_arguments)]
    pub fn inject_wavepacket(&mut self, x: i32, y: i32, z: i32, radius: f32, amount: f32, kx: f32, ky: f32, kz: f32, phase: f32) {
        if !(radius.is_finite() && amount.is_finite() && kx.is_finite() && ky.is_finite() && kz.is_finite() && phase.is_finite())
            || radius <= 0.0
            || amount <= 0.0
        {
            return;
        }
        let c = self.wrap_cell(x, y, z);
        let (cx, cy, cz) = self.coords(c);
        let kz = if self.dim == 3 { kz as f64 } else { 0.0 };
        self.add_symmetric_packet(cx as f64, cy as f64, cz as f64, radius as f64 / 2.0, amount as f64, [kx as f64, ky as f64, kz], phase as f64);
    }

    /// Variante con centro real (para el laboratorio): igual que `inject_wavepacket`.
    #[allow(clippy::too_many_arguments)]
    pub fn inject_wavepacket_at(&mut self, x: f32, y: f32, z: f32, sigma: f32, amount: f32, kx: f32, ky: f32, kz: f32, phase: f32) {
        if sigma <= 0.0 || amount <= 0.0 || !sigma.is_finite() || !amount.is_finite() {
            return;
        }
        let kz = if self.dim == 3 { kz as f64 } else { 0.0 };
        let cz = if self.dim == 3 { z as f64 } else { 0.0 };
        self.add_symmetric_packet(x as f64, y as f64, cz, sigma as f64, amount as f64, [kx as f64, ky as f64, kz], phase as f64);
    }

    /// Fija V = v (limitado a [−π, π]) en el rectángulo (prisma en 3D). No acumula (§2.4).
    #[allow(clippy::too_many_arguments)]
    pub fn add_barrier(&mut self, x: i32, y: i32, z: i32, w: u32, h: u32, d: u32, v: f32) {
        if !v.is_finite() {
            return;
        }
        let v = v.clamp(-std::f32::consts::PI, std::f32::consts::PI);
        self.has_potential = true;
        let dd = if self.dim == 3 { d.max(1) } else { 1 };
        for dz in 0..dd as i32 {
            for dy in 0..h.min(self.h as u32) as i32 {
                for dx in 0..w.min(self.w as u32) as i32 {
                    let c = self.wrap_cell(x + dx, y + dy, z + dz);
                    self.potential[c] = v;
                }
            }
        }
    }

    /// Muro duro: reflexión perfecta en esas celdas; pone a cero la amplitud que haya dentro.
    #[allow(clippy::too_many_arguments)]
    pub fn add_wall(&mut self, x: i32, y: i32, z: i32, w: u32, h: u32, d: u32) {
        let dd = if self.dim == 3 { d.max(1) } else { 1 };
        let stride = 2 * self.nc_max;
        for dz in 0..dd as i32 {
            for dy in 0..h.min(self.h as u32) as i32 {
                for dx in 0..w.min(self.w as u32) as i32 {
                    let c = self.wrap_cell(x + dx, y + dy, z + dz);
                    self.wall[c] = 1;
                    for k in 0..stride {
                        self.buffer_read[c * stride + k] = 0.0;
                    }
                }
            }
        }
        self.has_walls = self.wall.iter().any(|&v| v != 0);
    }

    /// Trampa armónica V = ½·κ·|r − r₀|² (distancia mínima en el toro), limitada a π.
    /// Suma un obstáculo gaussiano de potencial V·exp(−r²/2σ²) centrado en (x, y, z).
    pub fn add_gaussian_potential(&mut self, x: f32, y: f32, z: f32, sigma: f32, v: f32) {
        if !(x.is_finite() && y.is_finite() && z.is_finite() && sigma.is_finite() && v.is_finite()) || sigma <= 0.0 {
            return;
        }
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let s2 = 2.0 * (sigma as f64).powi(2);
        for i in 0..self.n {
            let (px, py, pz) = self.coords(i);
            let dx = min_image(px as f64 - x as f64, w);
            let dy = min_image(py as f64 - y as f64, h);
            let dz = if self.dim == 3 { min_image(pz as f64 - z as f64, d) } else { 0.0 };
            let add = v as f64 * (-(dx * dx + dy * dy + dz * dz) / s2).exp();
            self.potential[i] = (self.potential[i] as f64 + add).clamp(-std::f64::consts::PI, std::f64::consts::PI) as f32;
        }
        self.has_potential = true;
    }

    pub fn set_harmonic_potential(&mut self, kappa: f32, cx: f32, cy: f32, cz: f32) {
        if !(kappa.is_finite() && cx.is_finite() && cy.is_finite() && cz.is_finite()) {
            return;
        }
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        self.has_potential = kappa != 0.0;
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let dx = min_image(x as f64 - cx as f64, w);
            let dy = min_image(y as f64 - cy as f64, h);
            let dz = if self.dim == 3 { min_image(z as f64 - cz as f64, d) } else { 0.0 };
            let v = 0.5 * kappa as f64 * (dx * dx + dy * dy + dz * dz);
            self.potential[i] = v.clamp(-std::f64::consts::PI, std::f64::consts::PI) as f32;
        }
    }

    /// Multiplica todo el campo por e^{i(k·r)} (empujón brusco a un estado ya relajado).
    pub fn boost_field(&mut self, kx: f32, ky: f32, kz: f32) {
        let stride = self.stride();
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let th = kx as f64 * x as f64 + ky as f64 * y as f64 + kz as f64 * z as f64;
            let (c, s) = (th.cos() as f32, th.sin() as f32);
            let b = i * stride;
            for k in 0..self.nc {
                let re = self.buffer_read[b + 2 * k];
                let im = self.buffer_read[b + 2 * k + 1];
                self.buffer_read[b + 2 * k] = re * c - im * s;
                self.buffer_read[b + 2 * k + 1] = re * s + im * c;
            }
        }
    }

    /// Desplaza todo el campo (dx, dy, dz) celdas en el toro (equivale a mover el resto de la escena).
    pub fn translate_field(&mut self, dx: i32, dy: i32, dz: i32) {
        let stride = self.stride();
        let src = std::mem::take(&mut self.buffer_read);
        let mut dst = std::mem::take(&mut self.buffer_write);
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let j = self.wrap_cell(x as i32 + dx, y as i32 + dy, z as i32 + dz);
            dst[j * stride..(j + 1) * stride].copy_from_slice(&src[i * stride..(i + 1) * stride]);
        }
        self.buffer_read = dst;
        self.buffer_write = src;
    }

    /// Pone a cero la amplitud fuera de |r − r₀| < radius (recorta un solitón relajado y deja
    /// fuera la radiación que emitió al formarse).
    pub fn mask_outside(&mut self, x: f32, y: f32, z: f32, radius: f32) {
        let stride = self.stride();
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        for i in 0..self.n {
            let (cx, cy, cz) = self.coords(i);
            let dx = min_image(cx as f64 - x as f64, w);
            let dy = min_image(cy as f64 - y as f64, h);
            let dz = if self.dim == 3 { min_image(cz as f64 - z as f64, d) } else { 0.0 };
            if dx * dx + dy * dy + dz * dz > (radius as f64) * (radius as f64) {
                for k in 0..stride {
                    self.buffer_read[i * stride + k] = 0.0;
                }
            }
        }
    }

    /// Copia el campo de otro universo con la misma malla y los mismos canales.
    pub fn copy_field_from(&mut self, other: &QuantumSpacetimeUniverse) {
        if other.n == self.n && other.nc == self.nc {
            self.buffer_read.copy_from_slice(&other.buffer_read);
            self.t = other.t;
            self.theta_ref = other.theta_ref;
        }
    }

    /// Suma el campo de otro universo con la misma malla (superposición de escenas preparadas).
    pub fn add_field_from(&mut self, other: &QuantumSpacetimeUniverse, dx: i32, dy: i32, dz: i32, phase: f32) {
        if other.n != self.n || other.nc != self.nc {
            return;
        }
        let stride = self.stride();
        let (c, s) = (phase.cos(), phase.sin());
        for i in 0..self.n {
            let (x, y, z) = self.coords(i);
            let j = self.wrap_cell(x as i32 + dx, y as i32 + dy, z as i32 + dz);
            for k in 0..self.nc {
                let re = other.buffer_read[i * stride + 2 * k];
                let im = other.buffer_read[i * stride + 2 * k + 1];
                self.buffer_read[j * stride + 2 * k] += re * c - im * s;
                self.buffer_read[j * stride + 2 * k + 1] += re * s + im * c;
            }
        }
    }
}

/// Celda envuelta (para usos internos sin `self`).
#[allow(dead_code)]
pub(crate) fn wrap3(x: i32, y: i32, z: i32, w: usize, h: usize, d: usize) -> usize {
    (wrap_i(z, d) * h + wrap_i(y, h)) * w + wrap_i(x, w)
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Gas con velocidades gaussianas (Levkov, Panin y Tkachev, 2018): |ψ̂(k)|² ∝ exp(−k²/k₀²)
    /// con fases aleatorias, en el espinor simétrico y normalizado a ⟨E⟩ = `mean_density`.
    /// Con k₀ = m\*·v es la distribución de velocidades de §3.4. Reemplaza el campo y pone t = 0.
    pub fn seed_gaussian_gas(&mut self, k0: f32, mean_density: f32, seed: u32) {
        let n = self.n;
        let mut rng = SplitMix64::new(seed as u64 ^ 0xB05E_57A2);
        let k0 = (k0 as f64).max(1e-6);
        let mut buf = std::mem::take(&mut self.cbuf);
        for (i, v) in buf.iter_mut().enumerate().take(n) {
            let (x, y, z) = self.coords(i);
            let kx = crate::fftn::wavenumber(x, self.w);
            let ky = crate::fftn::wavenumber(y, self.h);
            let kz = if self.dim == 3 { crate::fftn::wavenumber(z, self.d) } else { 0.0 };
            let amp = (-(kx * kx + ky * ky + kz * kz) / (2.0 * k0 * k0)).exp();
            let th = 2.0 * std::f64::consts::PI * rng.next_f64();
            *v = rustfft::num_complex::Complex::new((amp * th.cos()) as f32, (amp * th.sin()) as f32);
        }
        self.fft.inverse(&mut buf);
        let s: f64 = buf.iter().take(n).map(|c| c.norm_sqr() as f64).sum();
        let scale = if s > 0.0 { ((mean_density as f64 * n as f64) / s).sqrt() } else { 0.0 };
        let nc = self.nc;
        let stride = 2 * nc;
        let inv = 1.0 / (nc as f64).sqrt();
        for i in 0..n {
            let re = (buf[i].re as f64 * scale * inv) as f32;
            let im = (buf[i].im as f64 * scale * inv) as f32;
            for k in 0..nc {
                self.buffer_read[i * stride + 2 * k] = re;
                self.buffer_read[i * stride + 2 * k + 1] = im;
            }
        }
        self.cbuf = buf;
        self.t = 0;
        self.theta_ref = 0.0;
        self.sol.reset();
    }

    /// Fondo uniforme modulado: ρ(x) = ρ̄(1 + δ₀·cos(kx)) con k = 2π·mode/W y fase
    /// S(x) = s_phase·δ₀·cos(kx) en el espinor simétrico (modo de densidad con la velocidad
    /// del modo creciente: s_phase = (m\* / k²)·(d ln δ/dt), §3.4).
    pub fn seed_density_mode(&mut self, rho: f32, delta0: f32, mode: u32, s_phase: f32) {
        let nc = self.nc;
        let stride = 2 * nc;
        let inv = 1.0 / (nc as f64).sqrt();
        let k = 2.0 * std::f64::consts::PI * mode as f64 / self.w as f64;
        for i in 0..self.n {
            let x = (i % self.w) as f64;
            let c = (k * x).cos();
            let r = (rho as f64 * (1.0 + delta0 as f64 * c)).max(0.0);
            let s = s_phase as f64 * delta0 as f64 * c;
            let a = r.sqrt() * inv;
            let (re, im) = ((a * s.cos()) as f32, (a * s.sin()) as f32);
            for ch in 0..nc {
                self.buffer_read[i * stride + 2 * ch] = re;
                self.buffer_read[i * stride + 2 * ch + 1] = im;
            }
        }
        self.t = 0;
        self.theta_ref = 0.0;
        self.sol.reset();
    }

    /// Componente coseno (con signo) del modo de densidad: 2·Σ E·cos(kx)/(N·⟨E⟩), k = 2π·mode/W.
    /// Una onda estacionaria oscila con su frecuencia (el módulo de `density_mode_amplitude` la duplica).
    pub fn density_mode_cos(&self, mode: u32) -> f64 {
        let k = 2.0 * std::f64::consts::PI * mode as f64 / self.w as f64;
        let (mut c, mut tot) = (0.0f64, 0.0f64);
        for i in 0..self.n {
            let x = (i % self.w) as f64;
            let e = self.density(i) as f64;
            c += e * (k * x).cos();
            tot += e;
        }
        if tot > 0.0 {
            2.0 * c / tot
        } else {
            0.0
        }
    }

    /// Amplitud relativa del modo cos(2π·mode·x/W) de la densidad: 2|Σ E e^{−ikx}|/(N·⟨E⟩).
    pub fn density_mode_amplitude(&self, mode: u32) -> f64 {
        let k = 2.0 * std::f64::consts::PI * mode as f64 / self.w as f64;
        let (mut c, mut s, mut tot) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..self.n {
            let x = (i % self.w) as f64;
            let e = self.density(i) as f64;
            c += e * (k * x).cos();
            s += e * (k * x).sin();
            tot += e;
        }
        if tot > 0.0 {
            2.0 * (c * c + s * s).sqrt() / tot
        } else {
            0.0
        }
    }
}
