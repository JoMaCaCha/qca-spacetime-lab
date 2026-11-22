//! Núcleo de la caminata de Grover no lineal (§2.1–§2.5): moneda, fase saturable acoplada a
//! los vecinos, desplazamiento *gather* y tope absoluto con lista de trabajo local.
//! Un único código genérico en el número de canales (NC = 4 en 2D, 6 en 3D).

use crate::util::fast_cos_sin;
use crate::QuantumSpacetimeUniverse;

/// Tolerancia relativa del tope: una celda solo reparte si E > E_cap·(1 + 5·10⁻⁷).
pub(crate) const CAP_TOLERANCE: f32 = 5e-7;

/// Moneda C(m) = cos m·I − i·sin m·G, G = (2/n_c)·J − I, aplicada sin matriz en O(n_c).
/// `psi` contiene n_c pares (Re, Im).
#[inline(always)]
pub fn apply_coin(psi: &mut [f32], nc: usize, cos_m: f32, sin_m: f32) {
    let f = 2.0 / nc as f32;
    let (mut sr, mut si) = (0.0f32, 0.0f32);
    for k in 0..nc {
        sr += psi[2 * k];
        si += psi[2 * k + 1];
    }
    let (fsr, fsi) = (f * sr, f * si);
    for k in 0..nc {
        let re = psi[2 * k];
        let im = psi[2 * k + 1];
        psi[2 * k] = cos_m * re + sin_m * (fsi - im);
        psi[2 * k + 1] = cos_m * im - sin_m * (fsr - re);
    }
}

/// Σ_k ψ_k por pares, (ψ₀ + ψ₁) + (ψ₂ + ψ₃) [+ (ψ₄ + ψ₅)]: el redondeo en f32 no cambia al
/// intercambiar los canales de un par (N↔S, E↔W, F↔B) ni los dos primeros pares, así que el paso
/// conserva exactamente las simetrías de reflexión de la red (un vórtice centrado no gana dipolo).
#[inline(always)]
pub(crate) fn pair_sum<const NC: usize>(p: &[f32]) -> (f32, f32) {
    let r = (p[0] + p[2]) + (p[4] + p[6]);
    let i = (p[1] + p[3]) + (p[5] + p[7]);
    if NC == 6 {
        (r + (p[8] + p[10]), i + (p[9] + p[11]))
    } else {
        (r, i)
    }
}

/// Σ|ψ_k|² con la misma agrupación por pares que `pair_sum`.
#[inline(always)]
pub(crate) fn pair_sum_sq<const NC: usize>(p: &[f32]) -> f32 {
    let e = |k: usize| p[2 * k] * p[2 * k] + p[2 * k + 1] * p[2 * k + 1];
    let s = (e(0) + e(1)) + (e(2) + e(3));
    if NC == 6 {
        s + (e(4) + e(5))
    } else {
        s
    }
}

impl QuantumSpacetimeUniverse {
    /// Paso 1: E_eff = E/(1 + E/E_max) de todas las celdas, antes de la moneda (que conserva E).
    pub(crate) fn compute_e_eff<const NC: usize>(&mut self) {
        let inv = 1.0 / self.e_max;
        let stride = 2 * NC;
        let psi = &self.buffer_read;
        let n = self.n;
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            use core::arch::wasm32::*;
            let v_one = f32x4_splat(1.0);
            let v_inv = f32x4_splat(inv);
            let mut i = 0usize;
            let out_ptr = self.e_eff.as_mut_ptr();
            while i + 4 <= n {
                let e0 = pair_sum_sq::<NC>(&psi[i * stride..(i + 1) * stride]);
                let e1 = pair_sum_sq::<NC>(&psi[(i + 1) * stride..(i + 2) * stride]);
                let e2 = pair_sum_sq::<NC>(&psi[(i + 2) * stride..(i + 3) * stride]);
                let e3 = pair_sum_sq::<NC>(&psi[(i + 3) * stride..(i + 4) * stride]);
                let ev = f32x4(e0, e1, e2, e3);
                let eff = f32x4_div(ev, f32x4_add(v_one, f32x4_mul(ev, v_inv)));
                v128_store(out_ptr.add(i) as *mut v128, eff);
                i += 4;
            }
            while i < n {
                let e = pair_sum_sq::<NC>(&psi[i * stride..(i + 1) * stride]);
                self.e_eff[i] = e / (1.0 + e * inv);
                i += 1;
            }
            return;
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        for (i, out) in self.e_eff.iter_mut().enumerate().take(n) {
            let e = pair_sum_sq::<NC>(&psi[i * stride..(i + 1) * stride]);
            *out = e / (1.0 + e * inv);
        }
    }

    /// Frecuencia no lineal de la celda i: g·[(1 − α)·E_eff + α·⟨E_eff⟩_vecinos] (§2.2).
    #[inline(always)]
    pub(crate) fn omega_nl_at<const NC: usize>(&self, x: usize, y: usize, z: usize) -> f32 {
        let i = self.idx(x, y, z);
        let own = self.e_eff[i];
        if self.alpha == 0.0 {
            return self.g * own;
        }
        let (w, h, d) = (self.w, self.h, self.d);
        let xm = if x == 0 { w - 1 } else { x - 1 };
        let xp = if x + 1 == w { 0 } else { x + 1 };
        let ym = if y == 0 { h - 1 } else { y - 1 };
        let yp = if y + 1 == h { 0 } else { y + 1 };
        let mut nb = self.e_eff[self.idx(xm, y, z)]
            + self.e_eff[self.idx(xp, y, z)]
            + self.e_eff[self.idx(x, ym, z)]
            + self.e_eff[self.idx(x, yp, z)];
        if NC == 6 {
            let zm = if z == 0 { d - 1 } else { z - 1 };
            let zp = if z + 1 == d { 0 } else { z + 1 };
            nb += self.e_eff[self.idx(x, y, zm)] + self.e_eff[self.idx(x, y, zp)];
        }
        self.g * ((1.0 - self.alpha) * own + self.alpha * nb / NC as f32)
    }

    /// Paso 2: moneda + fase (o reflexión en las celdas de muro), in situ sobre `buffer_read`.
    pub(crate) fn coin_phase_pass<const NC: usize>(&mut self) {
        let stride = 2 * NC;
        let (w, h, d) = (self.w, self.h, self.d);
        let (cm, sm) = (self.cos_m, self.sin_m);
        let f = 2.0 / NC as f32;
        let gamma_phi = self.gravity_phase_active();
        let gam = if gamma_phi { self.grav.gamma_eff() } else { 0.0 };

        #[inline(always)]
        fn apply_cell<const NC: usize>(p: &mut [f32], cm: f32, sm: f32, f: f32, ct: f32, st: f32) {
            let (sr, si) = pair_sum::<NC>(p);
            let (fsr, fsi) = (f * sr, f * si);
            let dr = cm * ct + sm * st;
            let di = sm * ct - cm * st;
            let br = sm * (fsi * ct - fsr * st);
            let bi = sm * (-fsr * ct - fsi * st);
            for k in 0..NC {
                let re = p[2 * k];
                let im = p[2 * k + 1];
                p[2 * k] = dr * re - di * im + br;
                p[2 * k + 1] = di * re + dr * im + bi;
            }
        }

        // Ruta SIMD vertical (4 celdas contiguas en paralelo): vectoriza omega_nl, potential y
        // fast_cos_sin_x4 sin ramas en el interior de cada fila cuando no hay muros.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if !self.has_walls && w >= 6 {
            use core::arch::wasm32::*;
            let g_v = f32x4_splat(self.g);
            let c_own = f32x4_splat(self.g * (1.0 - self.alpha));
            let c_nb = f32x4_splat(self.g * self.alpha / NC as f32);
            let alpha_zero = self.alpha == 0.0;
            let has_pot = self.has_potential;
            let e_ptr = self.e_eff.as_ptr();
            let pot_ptr = self.potential.as_ptr();
            let wh = w * h;
            for z in 0..d {
                let zm = if z == 0 { d - 1 } else { z - 1 };
                let zp = if z + 1 == d { 0 } else { z + 1 };
                for y in 0..h {
                    let ym = if y == 0 { h - 1 } else { y - 1 };
                    let yp = if y + 1 == h { 0 } else { y + 1 };
                    let row = (z * h + y) * w;
                    let row_ym = (z * h + ym) * w;
                    let row_yp = (z * h + yp) * w;
                    let row_zm = (zm * h + y) * w;
                    let row_zp = (zp * h + y) * w;
                    // Borde izquierdo x = 0
                    {
                        let mut wnl = self.omega_nl_at::<NC>(0, y, z);
                        if gamma_phi {
                            wnl += gam * self.phi_total(row);
                        }
                        let (ct, st) = fast_cos_sin(self.potential[row] - wnl);
                        apply_cell::<NC>(&mut self.buffer_read[row * stride..(row + 1) * stride], cm, sm, f, ct, st);
                    }
                    // Interior x = 1..w-1 en bloques de 4 celdas
                    let mut x = 1usize;
                    while x + 4 <= w - 1 {
                        let i = row + x;
                        let (ct_arr, st_arr) = unsafe {
                            let own = v128_load(e_ptr.add(i) as *const v128);
                            let mut wnl = if alpha_zero {
                                f32x4_mul(g_v, own)
                            } else {
                                let xm = v128_load(e_ptr.add(i - 1) as *const v128);
                                let xp = v128_load(e_ptr.add(i + 1) as *const v128);
                                let ym_v = v128_load(e_ptr.add(row_ym + x) as *const v128);
                                let yp_v = v128_load(e_ptr.add(row_yp + x) as *const v128);
                                let mut nb = f32x4_add(f32x4_add(xm, xp), f32x4_add(ym_v, yp_v));
                                if NC == 6 {
                                    let zm_v = v128_load(e_ptr.add(row_zm + x) as *const v128);
                                    let zp_v = v128_load(e_ptr.add(row_zp + x) as *const v128);
                                    nb = f32x4_add(nb, f32x4_add(zm_v, zp_v));
                                }
                                f32x4_add(f32x4_mul(c_own, own), f32x4_mul(c_nb, nb))
                            };
                            if gamma_phi {
                                let pv = f32x4(
                                    self.phi_total(i),
                                    self.phi_total(i + 1),
                                    self.phi_total(i + 2),
                                    self.phi_total(i + 3),
                                );
                                wnl = f32x4_add(wnl, f32x4_mul(f32x4_splat(gam), pv));
                            }
                            let theta = if has_pot {
                                f32x4_sub(v128_load(pot_ptr.add(i) as *const v128), wnl)
                            } else {
                                f32x4_neg(wnl)
                            };
                            let (ctv, stv) = crate::util::fast_cos_sin_x4(theta);
                            let mut cta = [0.0f32; 4];
                            let mut sta = [0.0f32; 4];
                            v128_store(cta.as_mut_ptr() as *mut v128, ctv);
                            v128_store(sta.as_mut_ptr() as *mut v128, stv);
                            (cta, sta)
                        };
                        for k in 0..4 {
                            let b = (i + k) * stride;
                            apply_cell::<NC>(&mut self.buffer_read[b..b + stride], cm, sm, f, ct_arr[k], st_arr[k]);
                        }
                        x += 4;
                    }
                    // Resto de la fila (incluye borde derecho x = w - 1)
                    while x < w {
                        let i = row + x;
                        let mut wnl = self.omega_nl_at::<NC>(x, y, z);
                        if gamma_phi {
                            wnl += gam * self.phi_total(i);
                        }
                        let (ct, st) = fast_cos_sin(self.potential[i] - wnl);
                        apply_cell::<NC>(&mut self.buffer_read[i * stride..(i + 1) * stride], cm, sm, f, ct, st);
                        x += 1;
                    }
                }
            }
            let _ = wh;
            return;
        }

        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let i = (z * h + y) * w + x;
                    let b = i * stride;
                    if self.has_walls && self.wall[i] != 0 {
                        // Reflexión perfecta: N↔S, W↔E, F↔B (una permutación unitaria).
                        let p = &mut self.buffer_read[b..b + stride];
                        for pair in 0..NC / 2 {
                            let a = 4 * pair;
                            p.swap(a, a + 2);
                            p.swap(a + 1, a + 3);
                        }
                        continue;
                    }
                    let mut wnl = self.omega_nl_at::<NC>(x, y, z);
                    if gamma_phi {
                        wnl += gam * self.phi_total(i);
                    }
                    let theta = self.potential[i] - wnl;
                    // ψ ← e^{−iθ}·C(m)·ψ
                    let (ct, st) = fast_cos_sin(theta);
                    apply_cell::<NC>(&mut self.buffer_read[b..b + stride], cm, sm, f, ct, st);
                }
            }
        }
    }

    /// Paso 3: desplazamiento *gather* de `buffer_read` a `buffer_write` y `swap`.
    /// Con el tope activo, apunta en la lista de trabajo las celdas que lo superan.
    pub(crate) fn shift_pass<const NC: usize>(&mut self) {
        let stride = 2 * NC;
        let (w, h, d) = (self.w, self.h, self.d);
        let cap = self.e_cap;
        self.cap_stack.clear();
        {
            let src = &self.buffer_read;
            let dst = &mut self.buffer_write;
            for z in 0..d {
                let zm = if z == 0 { d - 1 } else { z - 1 };
                let zp = if z + 1 == d { 0 } else { z + 1 };
                for y in 0..h {
                    let ym = if y == 0 { h - 1 } else { y - 1 };
                    let yp = if y + 1 == h { 0 } else { y + 1 };
                    for x in 0..w {
                        let xm = if x == 0 { w - 1 } else { x - 1 };
                        let xp = if x + 1 == w { 0 } else { x + 1 };
                        let i = (z * h + y) * w + x;
                        let b = i * stride;
                        // N(0) ← (x, y+1); S(1) ← (x, y−1); W(2) ← (x+1, y); E(3) ← (x−1, y)
                        let sn = ((z * h + yp) * w + x) * stride;
                        let ss = ((z * h + ym) * w + x) * stride + 2;
                        let sw = ((z * h + y) * w + xp) * stride + 4;
                        let se = ((z * h + y) * w + xm) * stride + 6;
                        dst[b] = src[sn];
                        dst[b + 1] = src[sn + 1];
                        dst[b + 2] = src[ss];
                        dst[b + 3] = src[ss + 1];
                        dst[b + 4] = src[sw];
                        dst[b + 5] = src[sw + 1];
                        dst[b + 6] = src[se];
                        dst[b + 7] = src[se + 1];
                        if NC == 6 {
                            // F(4) ← (x, y, z+1); B(5) ← (x, y, z−1)
                            let sf = ((zp * h + y) * w + x) * stride + 8;
                            let sb = ((zm * h + y) * w + x) * stride + 10;
                            dst[b + 8] = src[sf];
                            dst[b + 9] = src[sf + 1];
                            dst[b + 10] = src[sb];
                            dst[b + 11] = src[sb + 1];
                        }
                        if cap > 0.0 {
                            let e = pair_sum_sq::<NC>(&dst[b..b + stride]);
                            if e > cap * (1.0 + CAP_TOLERANCE) {
                                self.cap_stack.push_back(i as u32);
                                self.cap_flag[i] = 1;
                            }
                        }
                    }
                }
            }
        }
        std::mem::swap(&mut self.buffer_read, &mut self.buffer_write);
    }

    /// Copia la celda i a la sombra f64 del tope la primera vez que el reparto la toca.
    #[inline(always)]
    fn cap_touch(&mut self, i: usize, stride: usize) {
        if self.cap_touched[i] == 0 {
            self.cap_touched[i] = 1;
            self.cap_touched_list.push(i as u32);
            let b = i * stride;
            for k in 0..stride {
                self.cap_shadow[b + k] = self.buffer_read[b + k] as f64;
            }
        }
    }

    #[inline(always)]
    fn shadow_energy(&self, i: usize, stride: usize) -> f64 {
        let b = i * stride;
        let mut e = 0.0f64;
        for k in 0..stride {
            let v = self.cap_shadow[b + k];
            e += v * v;
        }
        e
    }

    /// Reserva la sombra f64 del tope (fuera de `step`).
    pub(crate) fn ensure_cap_shadow(&mut self) {
        let len = self.n * 2 * self.nc_max;
        if self.cap_shadow.len() < len {
            self.cap_shadow = vec![0.0; len];
        }
    }

    /// Paso 4: tope absoluto E_cap (§2.5). Una celda que lo supera conserva E_cap y reparte el
    /// exceso a partes iguales entre sus vecinas que no son muro, reescalando amplitudes (las
    /// fases se mantienen y la norma total se conserva). Con U(1) dinámico, cada reparto se
    /// anota como corriente en el enlace por el que pasa.
    ///
    /// El reparto se hace en f64 sobre una copia sombra de las celdas que toca y se redondea a
    /// f32 una sola vez al final: en f32 las transferencias pequeñas no son representables, se
    /// pierden o se duplican al redondear y el montón de arena no converge.
    pub(crate) fn capacity_pass<const NC: usize>(&mut self) {
        if self.e_cap <= 0.0 || self.cap_stack.is_empty() {
            return;
        }
        if self.cap_shadow.len() < self.n * 2 * NC {
            // Solo si la sombra no se reservó antes (set_capacity y new la reservan).
            self.ensure_cap_shadow();
        }
        let stride = 2 * NC;
        let cap = self.e_cap as f64;
        // Tolerancia relativa (excesos del orden del redondeo no se reparten). Cola FIFO: rondas.
        let cap_hi = cap * (1.0 + CAP_TOLERANCE as f64);
        let (w, h, d) = (self.w, self.h, self.d);
        let mut iters = 0usize;
        let book_u1 = self.u1_books_capacity();
        while let Some(ci) = self.cap_stack.pop_front() {
            let i = ci as usize;
            self.cap_flag[i] = 0;
            self.cap_touch(i, stride);
            let e = self.shadow_energy(i, stride);
            if e <= cap_hi {
                continue;
            }
            iters += 1;
            if iters > self.cap_max_iters {
                // Corte de seguridad: vacía la lista sin perder norma.
                for &c in &self.cap_stack {
                    self.cap_flag[c as usize] = 0;
                }
                self.cap_stack.clear();
                break;
            }
            let x = i % w;
            let y = (i / w) % h;
            let z = i / (w * h);
            let mut nbrs = [0usize; 6];
            let mut axis_dir = [(0usize, 0i32); 6];
            nbrs[0] = self.idx(x, if y == 0 { h - 1 } else { y - 1 }, z);
            axis_dir[0] = (1, -1);
            nbrs[1] = self.idx(x, if y + 1 == h { 0 } else { y + 1 }, z);
            axis_dir[1] = (1, 1);
            nbrs[2] = self.idx(if x == 0 { w - 1 } else { x - 1 }, y, z);
            axis_dir[2] = (0, -1);
            nbrs[3] = self.idx(if x + 1 == w { 0 } else { x + 1 }, y, z);
            axis_dir[3] = (0, 1);
            if NC == 6 {
                nbrs[4] = self.idx(x, y, if z == 0 { d - 1 } else { z - 1 });
                axis_dir[4] = (2, -1);
                nbrs[5] = self.idx(x, y, if z + 1 == d { 0 } else { z + 1 });
                axis_dir[5] = (2, 1);
            }
            let mut count = 0usize;
            for k in 0..NC {
                if !(self.has_walls && self.wall[nbrs[k]] != 0) {
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let share = (e - cap) / count as f64;
            // Reescala la celda a E_cap.
            let sc = (cap / e).sqrt();
            let b = i * stride;
            for k in 0..stride {
                self.cap_shadow[b + k] *= sc;
            }
            for k in 0..NC {
                let j = nbrs[k];
                if self.has_walls && self.wall[j] != 0 {
                    continue;
                }
                self.cap_touch(j, stride);
                let ej = self.shadow_energy(j, stride);
                let bj = j * stride;
                if ej > 1e-30 {
                    let s = ((ej + share) / ej).sqrt();
                    for c in 0..stride {
                        self.cap_shadow[bj + c] *= s;
                    }
                } else {
                    // Vecina vacía: recibe el espinor de la celda que reparte.
                    let s = (share / cap).sqrt();
                    for c in 0..stride {
                        self.cap_shadow[bj + c] = self.cap_shadow[b + c] * s;
                    }
                }
                if book_u1 {
                    let (axis, dir) = axis_dir[k];
                    self.book_capacity_current(i, j, axis, dir, share);
                }
                if ej + share > cap_hi && self.cap_flag[j] == 0 {
                    self.cap_flag[j] = 1;
                    self.cap_stack.push_back(j as u32);
                }
            }
        }
        // Devuelve las celdas tocadas a f32 (un único redondeo por celda).
        let list = std::mem::take(&mut self.cap_touched_list);
        for &c in &list {
            let i = c as usize;
            let b = i * stride;
            for k in 0..stride {
                self.buffer_read[b + k] = self.cap_shadow[b + k] as f32;
            }
            self.cap_touched[i] = 0;
        }
        self.cap_touched_list = list;
        self.cap_touched_list.clear();
        if iters > 0 {
            self.cap_steps_active += 1;
        }
        self.cap_last_iters = iters as u32;
    }

    /// Un paso completo del escenario materia/vórtices: Moneda → Vibración → Desplazamiento → Capacidad.
    pub(crate) fn grover_step<const NC: usize>(&mut self) {
        self.compute_e_eff::<NC>();
        self.coin_phase_pass::<NC>();
        if self.u1_active() {
            self.u1_matter_step::<NC>();
        } else {
            self.shift_pass::<NC>();
        }
        self.capacity_pass::<NC>();
        self.after_matter_step();
        self.t = self.t.wrapping_add(1);
        self.theta_ref = crate::util::wrap_angle32(self.theta_ref + self.mass);
    }
}
