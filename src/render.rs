//! Visualización en el búfer RGBA preasignado (§4.4). Sin asignaciones por frame.
//! Brillo b = (E_eff/E_max)^0,5; modo 0 = fase (tono) y coherencia (saturación); modo 1 = flujo.
//! El potencial V y los muros se mezclan como tintes translúcidos.

use crate::util::hsv_to_rgb;
use crate::QuantumSpacetimeUniverse;
use wasm_bindgen::prelude::*;

impl QuantumSpacetimeUniverse {
    /// Color de una celda del escenario materia (modos 0 y 1). Devuelve (r, g, b) en [0, 1].
    #[inline]
    fn matter_color(&self, i: usize) -> (f32, f32, f32) {
        let stride = self.stride();
        let b = i * stride;
        let p = &self.buffer_read[b..b + stride];
        let mut e = 0.0f32;
        let (mut sr, mut si) = (0.0f32, 0.0f32);
        let nco = self.ncol;
        for k in 0..self.nc * nco {
            let (re, im) = (p[2 * k], p[2 * k + 1]);
            e += re * re + im * im;
            sr += re;
            si += im;
        }
        if e <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let eff = e / (1.0 + e / self.e_max);
        let v = (eff / self.e_max).clamp(0.0, 1.0).powf(self.brightness_gamma);
        match self.render_mode {
            2 if self.grav.mode != 0 || self.grav.has_ext => {
                // Φ normalizado: azul (bajo) → rojo (pozos), con curvas de nivel cada 10 %
                let (lo, hi) = (self.grav.phi_min, self.grav.phi_max);
                let range = if hi > lo { hi - lo } else { 1.0 };
                let t = ((self.phi_total(i) - lo) / range).clamp(0.0, 1.0);
                let hue = (1.0 - t) * 4.2;
                let frac = (t * 10.0).fract();
                let vv = 0.3 + 0.7 * v;
                let vv = if frac < 0.06 { vv * 0.45 } else { vv };
                hsv_to_rgb(hue, 0.85, vv)
            }
            5 if nco > 1 => self.colour_charge_rgb(i, v),
            4 if nco > 1 && self.em.mode != 0 => {
                let el = self.ym.elec_density[i];
                let mg = self.ym.mag_density[i];
                let tot = (el + mg).max(1e-30);
                let s = (tot / self.ym_dens_max.max(1e-30)).sqrt().min(1.0);
                let base = 0.25 * v;
                (base + s * mg / tot, base + s * el / tot, base + s * (el + mg) / tot)
            }
            1 => {
                // J = (|ψ_E|² − |ψ_W|², |ψ_S|² − |ψ_N|²)
                let pw = |k: usize| {
                    let mut s = 0.0;
                    for a in 0..nco {
                        let o = 2 * (k * nco + a);
                        s += p[o] * p[o] + p[o + 1] * p[o + 1];
                    }
                    s
                };
                let jx = pw(3) - pw(2);
                let jy = pw(1) - pw(0);
                let hue = jy.atan2(jx);
                let sat = ((jx * jx + jy * jy).sqrt() / e).clamp(0.0, 1.0);
                hsv_to_rgb(hue, sat, v)
            }
            _ => {
                let hue = si.atan2(sr) + self.theta_ref;
                let coh = ((sr * sr + si * si) / (self.nc as f32 * e)).clamp(0.0, 1.0);
                hsv_to_rgb(hue, coh, v)
            }
        }
    }

    #[inline]
    fn blend_barrier(&self, rgb: (f32, f32, f32), v: f32, wall: bool) -> (f32, f32, f32) {
        let (mut r, mut g, mut b) = rgb;
        if v != 0.0 {
            let a = (v.abs() / std::f32::consts::PI).min(1.0) * 0.55;
            let (tr, tg, tb) = if v > 0.0 { (1.0, 0.15, 0.1) } else { (0.15, 0.35, 1.0) };
            r = r * (1.0 - a) + tr * a;
            g = g * (1.0 - a) + tg * a;
            b = b * (1.0 - a) + tb * a;
        }
        if wall {
            let a = 0.88;
            r = r * (1.0 - a) + 0.55 * a;
            g = g * (1.0 - a) + 0.55 * a;
            b = b * (1.0 - a) + 0.58 * a;
        }
        (r, g, b)
    }

    #[inline]
    fn put_pixel(&mut self, px: usize, rgb: (f32, f32, f32)) {
        let o = px * 4;
        self.image[o] = (rgb.0.clamp(0.0, 1.0) * 255.0) as u8;
        self.image[o + 1] = (rgb.1.clamp(0.0, 1.0) * 255.0) as u8;
        self.image[o + 2] = (rgb.2.clamp(0.0, 1.0) * 255.0) as u8;
        self.image[o + 3] = 255;
    }

    /// Escenario Dirac 2D: brillo = |ψ↑|² + |ψ↓|² (normalizado al máximo del frame) y, en el modo 3,
    /// tono = atan2(⟨σ_x⟩, ⟨σ_z⟩) y saturación = |⟨σ⟩| proyectado. En los demás modos, la fase de ψ↑.
    fn render_dirac2d(&mut self) {
        let n = self.w * self.h;
        let nco = self.ncol;
        let stride = self.stride();
        let gauge = self.render_mode == 4 && self.em.mode != 0 && nco == 1;
        let bmax = if gauge { self.gauge_bmax() } else { 0.0 };
        let mut emax = 0.0f32;
        for i in 0..n {
            emax = emax.max(self.density(i));
        }
        let inv = if emax > 0.0 { 1.0 / emax } else { 0.0 };
        for i in 0..n {
            let b = i * stride;
            let (mut e, mut sx, mut sz, mut ur_sum, mut ui_sum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
            if nco == 1 {
                let (ur, ui, dr, di) = (self.buffer_read[b], self.buffer_read[b + 1], self.buffer_read[b + 2], self.buffer_read[b + 3]);
                e = ur * ur + ui * ui + dr * dr + di * di;
                sx = 2.0 * (ur * dr + ui * di);
                sz = ur * ur + ui * ui - dr * dr - di * di;
                ur_sum = ur;
                ui_sum = ui;
            } else {
                // Con color (SU(N)), las amplitudes se almacenan como [↑_c0..↑_cN-1, ↓_c0..↓_cN-1]
                for a in 0..nco {
                    let ou = b + 2 * a;
                    let od = b + 2 * (nco + a);
                    let (ur, ui, dr, di) = (self.buffer_read[ou], self.buffer_read[ou + 1], self.buffer_read[od], self.buffer_read[od + 1]);
                    let nu = ur * ur + ui * ui;
                    let nd = dr * dr + di * di;
                    e += nu + nd;
                    sx += 2.0 * (ur * dr + ui * di);
                    sz += nu - nd;
                    ur_sum += ur;
                    ui_sum += ui;
                }
            }
            let v = (e * inv).clamp(0.0, 1.0).powf(self.brightness_gamma);
            let rgb = if self.render_mode == 3 {
                let sat = if e > 0.0 { ((sx * sx + sz * sz).sqrt() / e).clamp(0.0, 1.0) } else { 0.0 };
                hsv_to_rgb(sx.atan2(sz), sat, v)
            } else if self.render_mode == 5 && nco > 1 {
                self.colour_charge_rgb(i, v)
            } else {
                hsv_to_rgb(ui_sum.atan2(ur_sum), 0.8, v)
            };
            let rgb = if gauge { self.tint_gauge(rgb, self.gauge_b_at(i % self.w, i / self.w), bmax) } else { rgb };
            let rgb = if self.render_mode == 3 && self.dmass.active && self.dmass.m.len() == self.n {
                self.tint_dirac_mass(rgb, i)
            } else {
                rgb
            };
            let rgb = self.blend_barrier(rgb, self.potential[i], self.has_walls && self.wall[i] != 0);
            self.put_pixel(i, rgb);
        }
    }

    /// Densidad y espín (⟨Σ_x⟩, ⟨Σ_z⟩) de una celda del motor de espinores (Dirac 3D o Weyl).
    #[inline]
    fn spinor_cell(&self, i: usize) -> (f32, f32, f32, f32, f32) {
        let ns = self.sw.ns;
        let nco = self.ncol;
        let st = 2 * ns * nco;
        let p = &self.sw.psi[i * st..(i + 1) * st];
        let (mut e, mut sx, mut sz, mut re, mut im) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for blk in 0..ns / 2 {
            for a in 0..nco {
                let u = (p[2 * ((2 * blk) * nco + a)], p[2 * ((2 * blk) * nco + a) + 1]);
                let d = (p[2 * ((2 * blk + 1) * nco + a)], p[2 * ((2 * blk + 1) * nco + a) + 1]);
                let (nu, nd) = (u.0 * u.0 + u.1 * u.1, d.0 * d.0 + d.1 * d.1);
                e += nu + nd;
                sx += 2.0 * (u.0 * d.0 + u.1 * d.1);
                sz += nu - nd;
                re += u.0 + d.0;
                im += u.1 + d.1;
            }
        }
        (e as f32, sx as f32, sz as f32, re as f32, im as f32)
    }

    /// Escenarios Dirac 3D y Weyl (§4.4): brillo = |ψ|² normalizado al máximo del frame; modo 3,
    /// tono = atan2(⟨Σ_x⟩, ⟨Σ_z⟩) y saturación = |⟨Σ⟩| proyectado; modo 4 (U(1)), B encima; modo 5
    /// (SU(N)), la carga de color; los demás, la fase de Σψ. En 3D, MIP o corte. Con la cuerda de
    /// Higgs (§2.18), los núcleos (donde la masa se anula) se tiñen de ámbar.
    pub(crate) fn render_spinor(&mut self) {
        let (w, h, d) = (self.w, self.h, self.d);
        let mut emax = 0.0f32;
        for i in 0..self.n {
            let e = self.spinor_cell(i).0;
            self.e_eff[i] = e;
            emax = emax.max(e);
        }
        let inv = if emax > 0.0 { 1.0 / emax } else { 0.0 };
        for v in self.e_eff.iter_mut() {
            *v *= inv * self.e_max;
        }
        if self.ncol > 1 && self.render_mode == 4 && self.em.mode != 0 {
            self.compute_colour_field_densities();
            self.ym_dens_max = self.ym.energy_density.iter().fold(0.0f32, |m, v| m.max(*v));
        }
        let gauge = self.render_mode == 4 && self.em.mode != 0 && self.ncol == 1;
        let bmax = if gauge { self.gauge_bmax() } else { 0.0 };
        for y in 0..h {
            for x in 0..w {
                let cell = if self.dim == 2 {
                    self.idx(x, y, 0)
                } else if self.view == 1 {
                    self.idx(x, y, self.slice)
                } else {
                    let mut best = (-1.0f32, self.idx(x, y, 0));
                    for z in 0..d {
                        let c = self.idx(x, y, z);
                        let e = self.e_eff[c];
                        if e > best.0 {
                            best = (e, c);
                        }
                    }
                    best.1
                };
                let (e, sx, sz, re, im) = self.spinor_cell(cell);
                let v = (e * inv).clamp(0.0, 1.0).powf(self.brightness_gamma);
                let mut rgb = match self.render_mode {
                    3 => {
                        let sat = if e > 0.0 { ((sx * sx + sz * sz).sqrt() / e).clamp(0.0, 1.0) } else { 0.0 };
                        hsv_to_rgb(sx.atan2(sz), sat, v)
                    }
                    5 if self.ncol > 1 => self.colour_charge_rgb(cell, v),
                    4 if self.ncol > 1 && self.em.mode != 0 => {
                        let el = self.ym.elec_density[cell];
                        let mg = self.ym.mag_density[cell];
                        let tot = (el + mg).max(1e-30);
                        let s = (tot / self.ym_dens_max.max(1e-30)).sqrt().min(1.0);
                        let base = 0.4 * v;
                        (base + s * mg / tot, base + s * el / tot, base + s * (el + mg) / tot)
                    }
                    _ => hsv_to_rgb(im.atan2(re), 0.8, v),
                };
                if gauge {
                    rgb = self.tint_gauge(rgb, self.gauge_b_at(x, y), bmax);
                }
                if self.sw.mass_kind == 1 && !self.sw.mass_cell.is_empty() {
                    // 1 − sin μ/sin m₀ en ámbar: los núcleos de la cuerda
                    let c = self.idx(x, y, 0);
                    if 3 * c + 2 < self.sw.mass_cell.len() {
                        let smu = (self.sw.mass_cell[3 * c + 1].powi(2) + self.sw.mass_cell[3 * c + 2].powi(2)).sqrt() as f32;
                        let core = (1.0 - smu / (self.sw.string_m0 as f32).sin().max(1e-6)).clamp(0.0, 1.0);
                        let a = 0.35 * core;
                        rgb = (rgb.0 * (1.0 - a) + a, rgb.1 * (1.0 - a) + 0.65 * a, rgb.2 * (1.0 - a) + 0.15 * a);
                    }
                }
                let (pv, wall) = (self.potential[cell], self.has_walls && self.wall[cell] != 0);
                let rgb = self.blend_barrier(rgb, pv, wall);
                self.put_pixel(y * w + x, rgb);
            }
        }
    }

    /// Escenario 6 (§2.21): brillo = densidad total de los fermiones (MIP o corte); color = mezcla de
    /// los colores de las especies presentes (ν_L verde, e_L azul, e_R cian, u_L rojo, d_L naranja,
    /// u_R magenta, d_R amarillo). Con el modo 6, el fondo muestra |Φ|/v − 1 (ámbar si |Φ| > v,
    /// violeta si |Φ| < v).
    fn render_sm(&mut self) {
        const COLS: [(f32, f32, f32); 7] = [(0.2, 0.9, 0.3), (0.25, 0.45, 1.0), (0.2, 0.9, 0.95), (1.0, 0.25, 0.2), (1.0, 0.6, 0.1), (0.95, 0.3, 0.9), (0.95, 0.9, 0.2)];
        let (w, h, d) = (self.w, self.h, self.d);
        let mut emax = 0.0f32;
        for i in 0..self.n {
            let e: f64 = (0..5).map(|f| self.sm_field_density(i, f)).sum();
            self.e_eff[i] = e as f32;
            emax = emax.max(e as f32);
        }
        let inv = if emax > 0.0 { 1.0 / emax } else { 0.0 };
        let higgs = self.render_mode == 6 && self.higgs.phi.len() == 4 * self.n;
        for y in 0..h {
            for x in 0..w {
                let cell = if self.view == 1 {
                    self.idx(x, y, self.slice.min(d - 1))
                } else {
                    let mut best = (-1.0f32, self.idx(x, y, 0));
                    for z in 0..d {
                        let c = self.idx(x, y, z);
                        if self.e_eff[c] > best.0 {
                            best = (self.e_eff[c], c);
                        }
                    }
                    best.1
                };
                let e = self.e_eff[cell];
                let v = (e * inv).clamp(0.0, 1.0).powf(self.brightness_gamma);
                let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
                if e > 0.0 {
                    for (s, c) in COLS.iter().enumerate() {
                        let f = self.sm_species_density(cell, s) as f32 / e;
                        r += f * c.0;
                        g += f * c.1;
                        b += f * c.2;
                    }
                }
                let mut rgb = (r * v, g * v, b * v);
                if higgs {
                    let p = &self.higgs.phi[4 * cell..4 * cell + 4];
                    let a = ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]).sqrt() / self.higgs.v.max(1e-12) - 1.0) as f32;
                    let t = (a.abs() * 3.0).min(1.0) * 0.6;
                    let tint = if a > 0.0 { (1.0, 0.7, 0.2) } else { (0.6, 0.3, 1.0) };
                    rgb = (rgb.0 + t * tint.0 * (1.0 - v), rgb.1 + t * tint.1 * (1.0 - v), rgb.2 + t * tint.2 * (1.0 - v));
                }
                self.put_pixel(y * w + x, rgb);
            }
        }
    }

    /// Escenario 7 (§2.25): brillo = |ψ|² (MIP o corte), tono = fase de Σψ; con el U(1) dinámico, la
    /// energía eléctrica de fondo en azul tenue.
    fn render_twd(&mut self) {
        let (w, h, d) = (self.w, self.h, self.d);
        let n = self.n;
        let mut emax = 0.0f32;
        for i in 0..n {
            let e: f64 = (0..4).map(|a| self.twd.psi[4 * i + a].norm_sqr()).sum();
            self.e_eff[i] = e as f32;
            emax = emax.max(e as f32);
        }
        let inv = if emax > 0.0 { 1.0 / emax } else { 0.0 };
        for y in 0..h {
            for x in 0..w {
                let cell = if self.view == 1 {
                    self.idx(x, y, self.slice.min(d - 1))
                } else {
                    let mut best = (-1.0f32, self.idx(x, y, 0));
                    for z in 0..d {
                        let c = self.idx(x, y, z);
                        if self.e_eff[c] > best.0 {
                            best = (self.e_eff[c], c);
                        }
                    }
                    best.1
                };
                let v = (self.e_eff[cell] * inv).clamp(0.0, 1.0).powf(self.brightness_gamma);
                let s: rustfft::num_complex::Complex<f64> = (0..4).map(|a| self.twd.psi[4 * cell + a]).sum();
                let mut rgb = hsv_to_rgb(s.im.atan2(s.re) as f32, 0.8, v);
                if self.twd.dynamic {
                    let e2: f64 = (0..3).map(|a| self.twd.e[a * n + cell].powi(2)).sum();
                    let t = ((e2.sqrt() * 4.0) as f32).min(1.0) * 0.4;
                    rgb = (rgb.0, rgb.1 + 0.3 * t * (1.0 - v), rgb.2 + t * (1.0 - v));
                }
                self.put_pixel(y * w + x, rgb);
            }
        }
    }

    /// Escenario 8 (§2.26): MIP de la densidad de energía de Skyrme (o corte central), escalada a la imagen.
    fn render_skyrme(&mut self) {
        let (w, h) = (self.w, self.h);
        let n = self.sky.n;
        if n == 0 || self.sky.dens.len() != n * n * n {
            return;
        }
        let dmax = self.sky.dens.iter().cloned().fold(1e-30f32, f32::max);
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x * n / w, y * n / h);
                let val = if self.view == 1 {
                    self.sky.dens[((n / 2) * n + sy) * n + sx]
                } else {
                    (0..n).map(|z| self.sky.dens[(z * n + sy) * n + sx]).fold(0.0f32, f32::max)
                };
                let v = (val / dmax).clamp(0.0, 1.0).powf(0.6);
                let rgb = (0.05 + 0.95 * v, 0.05 + 0.6 * v * v, 0.12 + 0.3 * v * (1.0 - v));
                self.put_pixel(y * w + x, rgb);
            }
        }
    }

    /// Escenario 9 (§2.27): |ψ|² en la red (los huecos de la alfombra en gris oscuro), escalada a la imagen.
    /// Escenario 10 sin malla: la cascada espacio-tiempo de δn/ρ̄ de la tira (láser; tiempo hacia
    /// abajo, la región supersónica teñida), el mapa ξ·G/ρ̄ del Hawking por lotes (x′ horizontal,
    /// x vertical, ±0,05) o un fondo neutro (Unruh, que se dibuja en el panel).
    fn render_analog(&mut self) {
        let (w, h) = (self.w, self.h);
        let diverging = |v: f32| -> (f32, f32, f32) {
            let v = v.clamp(-1.0, 1.0);
            if v >= 0.0 {
                (0.08 + 0.92 * v, 0.08 + 0.35 * v, 0.10 + 0.1 * v)
            } else {
                (0.08 + 0.1 * -v, 0.08 + 0.45 * -v, 0.10 + 0.9 * -v)
            }
        };
        if self.an.setup == 1 && self.an.map_n > 0 {
            let nm = self.an.map_n;
            for y in 0..h {
                for x in 0..w {
                    let (i, j) = ((h - 1 - y) * nm / h, x * nm / w);
                    let v = (self.an.map[i * nm + j] / 0.05) as f32;
                    self.put_pixel(y * w + x, diverging(v));
                }
            }
            return;
        }
        if self.an.strip.is_none() {
            for p in 0..w * h {
                self.put_pixel(p, (0.06, 0.06, 0.08));
            }
            return;
        }
        // préstamos disjuntos (la tira y la imagen) y sin vectores temporales: nada se asigna aquí
        let st = self.an.strip.as_ref().unwrap();
        let img = &mut self.image;
        let sw = st.w;
        let rows = st.hist_rows;
        let scale = if rows > 0 { st.hist.iter().fold(1e-12f32, |m, v| m.max(v.abs())) } else { 1.0 };
        let gsub = st.g_sub.abs().max(1e-12);
        let dmax = (0..sw).fold(1e-12f32, |m, x| m.max((st.density(x) / st.rho - 1.0).abs() as f32));
        for y in 0..h {
            for x in 0..w {
                let sx = x * sw / w;
                let v = if rows > 0 { st.hist[(y * rows / h) * sw + sx] / scale } else { (st.density(sx) / st.rho - 1.0) as f32 / dmax };
                let (r, g, b) = diverging(v);
                let t = 0.12 * (1.0 - st.g[sx].abs() / gsub) as f32;
                let o = 4 * (y * w + x);
                img[o] = ((r + t).clamp(0.0, 1.0) * 255.0) as u8;
                img[o + 1] = ((g + t).clamp(0.0, 1.0) * 255.0) as u8;
                img[o + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                img[o + 3] = 255;
            }
        }
    }

    fn render_floquet(&mut self) {
        let (w, h) = (self.w, self.h);
        let l = self.flq.l;
        if l == 0 {
            return;
        }
        let pmax = self.flq.psi.iter().map(|z| z.norm_sqr()).fold(1e-30, f64::max);
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x * l / w, l - 1 - y * l / h);
                let i = sy * l + sx;
                let rgb = if !self.flq.mask[i] {
                    (0.12, 0.12, 0.14)
                } else {
                    let v = ((self.flq.psi[i].norm_sqr() / pmax) as f32).sqrt();
                    let base = if (sx + sy) % 2 == 0 { 0.03 } else { 0.06 };
                    (base + 0.95 * v, base + 0.75 * v, base + 0.25 * v)
                };
                self.put_pixel(y * w + x, rgb);
            }
        }
    }

    /// Escenario luz (§4.4): brillo = |ψ|² (normalizado al máximo del frame) sobre un fondo teñido
    /// con n(r) − 1 (azul más intenso cerca de la masa). La trayectoria la dibuja JS encima.
    fn render_light(&mut self) {
        let n = self.w * self.h;
        if self.light.psi.len() != 2 * n {
            return;
        }
        let mut emax = 0.0f64;
        for i in 0..n {
            emax = emax.max(self.light.psi[i].norm_sqr() + self.light.psi[n + i].norm_sqr());
        }
        let inv = if emax > 0.0 { 1.0 / emax } else { 0.0 };
        let c0 = self.light.theta0.cos();
        for i in 0..n {
            let e = (self.light.psi[i].norm_sqr() + self.light.psi[n + i].norm_sqr()) * inv;
            let v = (e as f32).powf(self.brightness_gamma);
            let nn = c0 / self.light.cs[0][i].max(1e-12) - 1.0;
            let tint = ((nn / 0.05).sqrt().min(1.0)) as f32;
            let base = (0.02, 0.03 + 0.10 * tint, 0.05 + 0.35 * tint);
            let rgb = (base.0 + v * 0.95, base.1 + v * 0.85, base.2 + v * 0.55);
            self.put_pixel(i, rgb);
        }
    }

    /// Escenario Kaplan (§2.20): la tira en s ocupa el ancho de la imagen; cada columna es una
    /// barra con la altura de |ψ(s)|² (normalizada al máximo) y el color de ⟨γ⁵⟩ local (rojo +1,
    /// azul −1). El fondo se tiñe con el signo de m(s) y las dos paredes se marcan con líneas.
    fn render_kaplan(&mut self) {
        let (w, h) = (self.w, self.h);
        let ls = self.kap.ls;
        if ls == 0 {
            return;
        }
        let dens = self.kaplan_density();
        if dens.len() < ls {
            return;
        }
        let dmax = dens.iter().cloned().fold(1e-300, f64::max);
        let g5 = crate::spinor::dirac_matrix(0, 3);
        for x in 0..w {
            let s = (x * ls / w).min(ls - 1);
            let col = if self.kap.psi.len() == 4 * ls {
                let v = [self.kap.psi[4 * s], self.kap.psi[4 * s + 1], self.kap.psi[4 * s + 2], self.kap.psi[4 * s + 3]];
                let gv = crate::spinor::m4_apply(&g5, &v);
                let g: f64 = (0..4).map(|c| (v[c].conj() * gv[c]).re).sum();
                (g / dens[s].max(1e-300)) as f32
            } else {
                0.0
            };
            let height = ((dens[s] / dmax).sqrt() * h as f64 * 0.9) as usize;
            let m = self.kap.mass_at(s);
            let wall = s == ls / 4 || s == 3 * ls / 4;
            for y in 0..h {
                let from_bottom = h - 1 - y;
                let rgb = if wall {
                    (0.85, 0.85, 0.85)
                } else if from_bottom < height {
                    let r = 0.5 + 0.5 * col.max(0.0);
                    let b = 0.5 + 0.5 * (-col).max(0.0);
                    (r, 0.35, b)
                } else if m >= 0.0 {
                    (0.05, 0.07, 0.14)
                } else {
                    (0.14, 0.09, 0.03)
                };
                self.put_pixel(y * w + x, rgb);
            }
        }
    }

    /// §4.4: con masa de Dirac por celda, tinte azul si m > 0 y ámbar si m < 0, y las paredes (los
    /// cambios de signo con la vecina de la derecha o de abajo) como líneas finas.
    fn tint_dirac_mass(&self, rgb: (f32, f32, f32), i: usize) -> (f32, f32, f32) {
        let m = self.dmass.m[i];
        let a = 0.22 * (m.abs() / 0.5).min(1.0);
        let (tr, tg, tb) = if m >= 0.0 { (0.25, 0.45, 1.0) } else { (1.0, 0.68, 0.18) };
        let mut c = (rgb.0 * (1.0 - a) + tr * a, rgb.1 * (1.0 - a) + tg * a, rgb.2 * (1.0 - a) + tb * a);
        let (x, y) = (i % self.w, i / self.w);
        let right = self.dmass.m[y * self.w + (x + 1) % self.w];
        let down = self.dmass.m[((y + 1) % self.h) * self.w + x];
        if m * right < 0.0 || m * down < 0.0 {
            c = (c.0 * 0.5 + 0.45, c.1 * 0.5 + 0.45, c.2 * 0.5 + 0.45);
        }
        c
    }

    /// |φ|/v en una celda (cualquier modo del Higgs).
    #[inline]
    fn higgs_rel_abs(&self, i: usize) -> f32 {
        let nf = self.higgs.nf;
        let a: f64 = self.higgs.phi[i * nf..(i + 1) * nf].iter().map(|v| v * v).sum::<f64>().sqrt();
        (a / self.higgs.v.max(1e-9)) as f32
    }

    /// B de la plaqueta xy que se superpone en el modo 6: abeliano y doblete, q·B (U(1), módulo 2π
    /// con la energía compacta); adjunto, el ángulo de la proyección abeliana sobre φ̂.
    #[inline]
    fn higgs_b_display(&self, i: usize) -> f32 {
        let a = self.higgs_plaquette_angle_xy(i);
        a as f32
    }

    /// Modo 6 (§4.4): brillo 1 − |φ|/v (los núcleos de los vórtices y los tubos de flujo se ven
    /// brillantes) con B encima en rojo/azul. El color no usa la fase de φ, que depende del gauge.
    /// En 3D, la MIP toma el máximo de cada columna.
    fn render_higgs(&mut self) {
        let (w, h, d) = (self.w, self.h, self.d);
        let wh = w * h;
        // Una única pasada sobre los vóxeles para calcular brillo y B sin duplicar llamadas a plaquetas.
        let mut bright_buf = vec![0.0f32; wh];
        let mut b_buf = vec![0.0f32; wh];
        let mut bmax = 0.0f32;
        if self.dim == 2 || self.view == 1 {
            let z = if self.dim == 3 { self.slice } else { 0 };
            for y in 0..h {
                for x in 0..w {
                    let px = y * w + x;
                    let i = self.idx(x, y, z);
                    bright_buf[px] = (1.0 - self.higgs_rel_abs(i)).clamp(0.0, 1.0);
                    let b = self.higgs_b_display(i);
                    b_buf[px] = b;
                    bmax = bmax.max(b.abs());
                }
            }
        } else {
            for z in 0..d {
                for y in 0..h {
                    for x in 0..w {
                        let px = y * w + x;
                        let i = self.idx(x, y, z);
                        let br = (1.0 - self.higgs_rel_abs(i)).clamp(0.0, 1.0);
                        if br > bright_buf[px] {
                            bright_buf[px] = br;
                        }
                        let v = self.higgs_b_display(i);
                        if v.abs() > b_buf[px].abs() {
                            b_buf[px] = v;
                            bmax = bmax.max(v.abs());
                        }
                    }
                }
            }
        }
        let gam = self.brightness_gamma.min(1.0);
        for px in 0..wh {
            let v = bright_buf[px].powf(gam);
            let base = (0.92 * v, 0.92 * v, 0.98 * v);
            let c = self.tint_gauge(base, b_buf[px], bmax);
            self.put_pixel(px, c);
        }
    }

    /// B que se dibuja en la columna (x, y): B_xy en 2D y en el corte; max |B| en la MIP.
    fn gauge_b_at(&self, x: usize, y: usize) -> f32 {
        if self.dim == 2 {
            return self.plaquette_b(self.idx(x, y, 0), 0, 1) as f32;
        }
        if self.view == 1 {
            return self.plaquette_b(self.idx(x, y, self.slice), 0, 1) as f32;
        }
        let mut m = 0.0f64;
        for z in 0..self.d {
            let i = self.idx(x, y, z);
            let b2 = self.plaquette_b(i, 0, 1).powi(2) + self.plaquette_b(i, 1, 2).powi(2) + self.plaquette_b(i, 2, 0).powi(2);
            m = m.max(b2);
        }
        m.sqrt() as f32
    }

    /// Modo 4: tiñe con B (rojo si B > 0, azul si B < 0), normalizado al máximo del frame.
    fn tint_gauge(&self, rgb: (f32, f32, f32), b: f32, bmax: f32) -> (f32, f32, f32) {
        if bmax <= 0.0 {
            return rgb;
        }
        let a = (b.abs() / bmax).sqrt().min(1.0) * 0.8;
        let (tr, tg, tb) = if b > 0.0 { (1.0, 0.2, 0.15) } else { (0.2, 0.45, 1.0) };
        (rgb.0 * (1.0 - a) + tr * a, rgb.1 * (1.0 - a) + tg * a, rgb.2 * (1.0 - a) + tb * a)
    }

    fn gauge_bmax(&self) -> f32 {
        let mut m = 0.0f32;
        for y in 0..self.h {
            for x in 0..self.w {
                m = m.max(self.gauge_b_at(x, y).abs());
            }
        }
        m
    }

    /// Modo 5: color de la densidad de carga ρ^c (depende del gauge; |ρ| es invariante).
    fn colour_charge_rgb(&self, i: usize, v: f32) -> (f32, f32, f32) {
        let mut rho = [0.0f64; 8];
        self.colour_density(i, &mut rho);
        let e = self.density(i).max(1e-30) as f64;
        if self.ncol == 3 {
            let s3 = 1.0 / 3f64.sqrt();
            let w = [(0.5, 0.5 * s3), (-0.5, 0.5 * s3), (0.0, -s3)];
            let mut c = [0.0f32; 3];
            for k in 0..3 {
                let p = (rho[2] * w[k].0 + rho[7] * w[k].1) / e;
                c[k] = (3.0 * p).max(0.0).min(1.0) as f32;
            }
            (v * (0.25 + 0.75 * c[0]), v * (0.25 + 0.75 * c[1]), v * (0.25 + 0.75 * c[2]))
        } else {
            let r3 = (rho[2] / e) as f32;
            let tr = ((rho[0] * rho[0] + rho[1] * rho[1]).sqrt() / e) as f32;
            (v * (0.25 + 1.5 * r3.max(0.0)), v * (0.25 + 1.5 * tr), v * (0.25 + 1.5 * (-r3).max(0.0)))
        }
    }

    fn render_matter(&mut self) {
        let (w, h, d) = (self.w, self.h, self.d);
        if self.ncol > 1 && self.render_mode == 4 && self.em.mode != 0 {
            self.compute_colour_field_densities();
            self.ym_dens_max = self.ym.energy_density.iter().fold(0.0f32, |m, v| m.max(*v));
        }
        let gauge = self.render_mode == 4 && self.em.mode != 0 && self.ncol == 1;
        let bmax = if gauge { self.gauge_bmax() } else { 0.0 };
        if self.dim == 2 {
            for i in 0..w * h {
                let c = self.matter_color(i);
                let c = if gauge { self.tint_gauge(c, self.gauge_b_at(i % w, i / w), bmax) } else { c };
                let c = self.blend_barrier(c, self.potential[i], self.has_walls && self.wall[i] != 0);
                self.put_pixel(i, c);
            }
            return;
        }
        if self.view == 1 {
            for y in 0..h {
                for x in 0..w {
                    let px = y * w + x;
                    let cell = self.idx(x, y, self.slice);
                    let c = self.matter_color(cell);
                    let c = if gauge { self.tint_gauge(c, self.gauge_b_at(x, y), bmax) } else { c };
                    let c = self.blend_barrier(c, self.potential[cell], self.has_walls && self.wall[cell] != 0);
                    self.put_pixel(px, c);
                }
            }
            return;
        }
        // MIP 3D con recorrido secuencial z -> y -> x (localidad de caché óptima)
        let wh = w * h;
        let mut best_e = vec![-1.0f32; wh];
        let mut best_cell = vec![0usize; wh];
        let mut best_v = vec![0.0f32; wh];
        let mut any_wall = vec![false; wh];
        for z in 0..d {
            let plane = z * wh;
            for px in 0..wh {
                let c = plane + px;
                let e = self.density(c);
                if e > best_e[px] {
                    best_e[px] = e;
                    best_cell[px] = c;
                }
                let v = self.potential[c];
                if v.abs() > best_v[px].abs() {
                    best_v[px] = v;
                }
                if self.has_walls && self.wall[c] != 0 {
                    any_wall[px] = true;
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                let px = y * w + x;
                let c = self.matter_color(best_cell[px]);
                let c = if gauge { self.tint_gauge(c, self.gauge_b_at(x, y), bmax) } else { c };
                let c = self.blend_barrier(c, best_v[px], any_wall[px]);
                self.put_pixel(px, c);
            }
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Dibuja el estado actual en el búfer RGBA (W × H; en 3D, MIP o corte).
    pub fn render_rgba(&mut self) {
        if self.scenario == 10 && matches!(self.an.setup, 0 | 1 | 4) {
            self.render_analog();
            return;
        }
        if self.scenario == 8 && self.sky.active {
            self.render_skyrme();
            return;
        }
        if self.scenario == 9 && self.flq.active {
            self.render_floquet();
            return;
        }
        if self.twd.active {
            self.render_twd();
            return;
        }
        if self.sm.active {
            self.render_sm();
            return;
        }
        if self.higgs.mode != 0 {
            self.render_higgs();
            return;
        }
        if self.scenario == 5 {
            self.render_kaplan();
            return;
        }
        if self.scenario == 4 {
            self.render_light();
            return;
        }
        if self.sw.active {
            self.render_spinor();
            return;
        }
        if self.scenario == 2 && self.dim == 2 {
            self.render_dirac2d();
            return;
        }
        // e_eff al día solo para la vista volumétrica (view == 2); ahorra un barrido completo por frame en 2D/MIP/corte
        if self.view == 2 {
            let inv = 1.0 / self.e_max;
            for i in 0..self.n {
                let e = self.density(i);
                self.e_eff[i] = e / (1.0 + e * inv);
            }
        }
        self.render_matter();
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Exponente del brillo, b = (E_eff/E_max)^γ. La especificación usa γ = 0,5 (el valor por
    /// defecto); en la vista MIP 3D el máximo de cada columna de fondo ya está cerca de E_max y
    /// γ = 0,5 lo satura todo, así que el frontend sube γ a 2 en esa vista.
    pub fn set_brightness_gamma(&mut self, gamma: f32) {
        if gamma.is_finite() {
            self.brightness_gamma = gamma.clamp(0.1, 4.0);
        }
    }
}
