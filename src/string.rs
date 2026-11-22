//! Fermiones atrapados en una cuerda de Higgs (§2.18): el Dirac 3D (o su corte 2D con k_z fijo) con
//! una masa compleja por celda, M = m₀·f(r)·(cos θ·β + sin θ·B₂), B₂ = τ_y ⊗ 1. El fondo es un par
//! vórtice-antivórtice de devanados ±n a lo largo de z: f = tanh(r₁/ξ)^|n|·tanh(r₂/ξ)^|n| y la
//! fase de una suma de imágenes, con el salto de las costuras quitado por un gradiente uniforme.
//! Cada núcleo atrapa |n| modos que viajan a lo largo de la cuerda en un solo sentido (Jackiw y
//! Rossi; Callan y Harvey). φ es un fondo fijo sin campo gauge: es la cuerda "global" (axiónica).
//! El monopolo de 't Hooft–Polyakov como fondo de un quark isoespinor (modos de Jackiw–Rebbi,
//! §3.9) también está aquí.

use crate::spinor::cx;
use crate::sun::C;
use crate::util::{min_image, wrap_angle};
use crate::QuantumSpacetimeUniverse;
use wasm_bindgen::prelude::*;

impl QuantumSpacetimeUniverse {
    /// Fase del par vórtice-antivórtice en cada celda del plano (x, y), con el salto de las dos
    /// costuras quitado por gradientes uniformes.
    pub(crate) fn string_phase_plane(&self, winding: i32, cores: [[f64; 2]; 2]) -> Vec<f64> {
        let (w, h) = (self.w, self.h);
        let (wf, hf) = (w as f64, h as f64);
        let m = 8i32;
        let n = winding as f64;
        let mut th = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for i in -m..=m {
                    for j in -m..=m {
                        let (ox, oy) = (i as f64 * wf, j as f64 * hf);
                        s += (y as f64 - cores[0][1] - oy).atan2(x as f64 - cores[0][0] - ox);
                        s -= (y as f64 - cores[1][1] - oy).atan2(x as f64 - cores[1][0] - ox);
                    }
                }
                th[y * w + x] = n * s;
            }
        }
        // salto de la costura y: exceso del salto sobre el de las celdas vecinas, promediado en x
        let mut dy = 0.0;
        for x in 0..w {
            let seam = wrap_angle(th[x] - th[(h - 1) * w + x]);
            let inner = 0.5 * (wrap_angle(th[w + x] - th[x]) + wrap_angle(th[(h - 1) * w + x] - th[(h - 2) * w + x]));
            dy += seam - inner;
        }
        dy /= w as f64;
        let mut dx = 0.0;
        for y in 0..h {
            let seam = wrap_angle(th[y * w] - th[y * w + w - 1]);
            let inner = 0.5 * (wrap_angle(th[y * w + 1] - th[y * w]) + wrap_angle(th[y * w + w - 1] - th[y * w + w - 2]));
            dx += seam - inner;
        }
        dx /= h as f64;
        for y in 0..h {
            for x in 0..w {
                th[y * w + x] += dy * y as f64 / hf + dx * x as f64 / wf;
            }
        }
        th
    }

    /// Núcleos del par: (W/4 − ½, H/2 − ½) y (3W/4 − ½, H/2 − ½), en el centro de una plaqueta.
    pub(crate) fn string_cores(&self) -> [[f64; 2]; 2] {
        let (wf, hf) = (self.w as f64, self.h as f64);
        [[0.25 * wf - 0.5, 0.5 * hf - 0.5], [0.75 * wf - 0.5, 0.5 * hf - 0.5]]
    }

    /// Espinor del paquete sencillo de un núcleo: el autovector |τ_x = +1⟩⊗|σ_z = ±1⟩ de α_z = ±1
    /// (+1 en el vórtice con n > 0). Con él se queda en el núcleo ~51 % de una gaussiana de anchura
    /// 2,5 (m₀ = 0,6), que viaja a 0,834. La combinación (1, i, 1, −i)/2 del mismo subespacio se
    /// parece más al modo de la red (se queda el 78 %), pero el paquete sencillo no calcula el modo.
    pub(crate) fn string_core_spinor(positive: bool) -> [C; 4] {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        if positive {
            [cx(s, 0.0), cx(0.0, 0.0), cx(s, 0.0), cx(0.0, 0.0)]
        } else {
            [cx(0.0, 0.0), cx(s, 0.0), cx(0.0, 0.0), cx(s, 0.0)]
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Escenario Dirac en 2D: con `on`, el corte con k_z fijo del Dirac 3D de 4 componentes (el k_z
    /// se fija con `set_weyl_kz`), para los espectros de la cuerda de Higgs; sin él, el Dirac 2D de
    /// 2 componentes de §2.8.
    pub fn set_dirac_slice(&mut self, on: bool) {
        if self.scenario != 2 || self.dim != 2 {
            self.set_message("set_dirac_slice solo actúa en el escenario Dirac en 2D.");
            return;
        }
        let kz = self.sw.kz;
        if on {
            self.nc = 4;
            self.sw_activate(4, false);
            self.sw.kz = kz;
        } else {
            self.sw = crate::spinor::SpinorWalk::new();
            self.nc = 2;
        }
        self.clear();
    }

    /// §2.18: masa compleja con un par vórtice-antivórtice de devanados ±n a lo largo de z
    /// (Dirac 3D o su corte 2D); winding = 0 vuelve a la masa real. Hace falta el orden fijo
    /// (`set_dirac_order(true)`) para ver la cuerda quiral.
    pub fn set_higgs_string(&mut self, winding: i32, m0: f32, xi: f32) {
        if !self.sw.active || self.sw.ns != 4 {
            self.set_message("La cuerda de Higgs necesita el escenario Dirac 3D (o su corte 2D).");
            return;
        }
        if winding == 0 {
            self.sw.mass_kind = 0;
            self.sw.string_winding = 0;
            self.sw.mass_cell.clear();
            return;
        }
        let winding = winding.clamp(-4, 4);
        let (m0, xi) = (m0.clamp(0.0, 1.5) as f64, (xi as f64).max(0.2));
        let cores = self.string_cores();
        let th = self.string_phase_plane(winding, cores);
        let (w, h) = (self.w, self.h);
        let plane = w * h;
        let mut mc = vec![0.0f64; 3 * self.n];
        for i in 0..self.n {
            let p = i % plane;
            let (x, y) = ((p % w) as f64, (p / w) as f64);
            let mut f = 1.0;
            for c in cores {
                let dx = min_image(x - c[0], w as f64);
                let dy = min_image(y - c[1], h as f64);
                f *= ((dx * dx + dy * dy).sqrt() / xi).tanh().powi(winding.abs());
            }
            let (s, c) = (m0 * f).sin_cos();
            mc[3 * i] = c;
            mc[3 * i + 1] = s * th[p].cos();
            mc[3 * i + 2] = s * th[p].sin();
        }
        self.sw.mass_cell = mc;
        self.sw.mass_kind = 1;
        self.sw.string_winding = winding;
        self.sw.string_m0 = m0;
        self.sw.string_xi = xi;
        self.sw.string_cores = cores;
    }

    /// Fase θ del fondo de la cuerda en la celda (x, y) (0 sin cuerda).
    pub fn higgs_string_phase(&self, x: i32, y: i32) -> f64 {
        if self.sw.mass_kind != 1 {
            return 0.0;
        }
        let i = self.wrap_cell(x, y, 0);
        self.sw.mass_cell[3 * i + 2].atan2(self.sw.mass_cell[3 * i + 1])
    }

    /// m₀·f del fondo de la cuerda en la celda (x, y).
    pub fn higgs_string_mass(&self, x: i32, y: i32) -> f64 {
        if self.sw.mass_kind != 1 {
            return self.mass as f64;
        }
        let i = self.wrap_cell(x, y, 0);
        self.sw.mass_cell[3 * i + 1].hypot(self.sw.mass_cell[3 * i + 2]).atan2(self.sw.mass_cell[3 * i])
    }

    /// §3.9: fondo estático de un par monopolo-antimonopolo de 't Hooft–Polyakov para un quark
    /// isoespinor (Dirac 3D con SU(2)): φ̂ del ansatz de Nambu (como `seed_monopole_pair`), |φ| =
    /// v·tanh(r₁/ξ)·tanh(r₂/ξ), enlaces U_a(r) que giran φ̂(r) en φ̂(r + â) (la rotación mínima, así
    /// que D φ̂ = 0) y la masa de Yukawa G·φ^a·(β ⊗ τ^a) con G·v = `gv`. Los polos están en
    /// (W/2, H/2, D/2 ± d/2). El campo gauge queda estático (modo 1).
    pub fn set_monopole_background(&mut self, d: f32, gv: f32, xi: f32) {
        if !self.sw.active || self.sw.ns != 4 || self.ncol != 2 || self.dim != 3 {
            self.set_message("El fondo de monopolos necesita el Dirac 3D con SU(2) (set_dirac_colour(2)).");
            return;
        }
        let (cx0, cy0, cz0) = (0.5 * self.w as f64, 0.5 * self.h as f64, 0.5 * self.d as f64);
        let p1 = [cx0, cy0, cz0 + 0.5 * d as f64];
        let p2 = [cx0, cy0, cz0 - 0.5 * d as f64];
        let sig2 = 2.0 * (0.5 * d as f64).powi(2).max(1.0);
        let xi = (xi as f64).max(0.3);
        let hat = |p: [f64; 3]| -> ([f64; 3], f64) {
            let r1 = [p[0] - p1[0], p[1] - p1[1], p[2] - p1[2]];
            let r2 = [p[0] - p2[0], p[1] - p2[1], p[2] - p2[2]];
            let n1 = (r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]).sqrt().max(1e-9);
            let n2 = (r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]).sqrt().max(1e-9);
            let th1 = (r1[2] / n1).clamp(-1.0, 1.0).acos();
            let th2 = (r2[2] / n2).clamp(-1.0, 1.0).acos();
            let rho2 = r1[0] * r1[0] + r1[1] * r1[1];
            let big = (th1 - th2) * (-rho2 / sig2).exp();
            let ph = r1[1].atan2(r1[0]);
            ([big.sin() * ph.cos(), big.sin() * ph.sin(), big.cos()], (n1 / xi).tanh() * (n2 / xi).tanh())
        };
        let n = self.n;
        let mut mc = vec![0.0f64; 4 * n];
        let mut dirs = vec![[0.0f64; 3]; n];
        for i in 0..n {
            let (x, y, z) = self.coords(i);
            let (dir, h) = hat([x as f64, y as f64, z as f64]);
            let (s, c) = (gv as f64 * h).sin_cos();
            mc[4 * i] = c;
            for k in 0..3 {
                mc[4 * i + 1 + k] = s * dir[k];
            }
            dirs[i] = dir;
        }
        // enlaces: U = cos(α/2) − i·sin(α/2)·m·τ con m = φ̂ × φ̂', α = ángulo entre φ̂ y φ̂'
        for a in 0..3 {
            for i in 0..n {
                let (x, y, z) = self.coords(i);
                let mut q = [x as f64, y as f64, z as f64];
                q[a] += 1.0;
                let (n0, n1) = (dirs[i], hat(q).0);
                let cr = [n0[1] * n1[2] - n0[2] * n1[1], n0[2] * n1[0] - n0[0] * n1[2], n0[0] * n1[1] - n0[1] * n1[0]];
                let sn = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                let cs = (n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]).clamp(-1.0, 1.0);
                let alpha = sn.atan2(cs);
                let m = if sn > 1e-14 { [cr[0] / sn, cr[1] / sn, cr[2] / sn] } else { [0.0, 0.0, 1.0] };
                let (s2, c2) = (0.5 * alpha).sin_cos();
                let mut u = crate::sun::Mat::zero(2);
                u.set(0, 0, cx(c2, -s2 * m[2]));
                u.set(1, 1, cx(c2, s2 * m[2]));
                u.set(0, 1, cx(-s2 * m[1], -s2 * m[0]));
                u.set(1, 0, cx(s2 * m[1], -s2 * m[0]));
                self.ym.set_link(a * n + i, &u);
            }
        }
        self.sw.mass_cell = mc;
        self.sw.mass_kind = 2;
        self.em.mode = 1;
    }

    /// Núcleo `core` (0 = vórtice, 1 = antivórtice): [x, y].
    pub fn higgs_string_core(&self, core: u32) -> Vec<f64> {
        self.sw.string_cores[(core as usize).min(1)].to_vec()
    }

    /// §2.18: gaussiana de anchura `width` en el núcleo (0 = vórtice, 1 = antivórtice) y σ_z a lo
    /// largo de z, con un espinor de α_z = ±sign(n), sin calcular el modo de la red. Parte se queda
    /// en el núcleo y viaja a lo largo de la cuerda; el resto radia al volumen.
    pub fn inject_string_packet(&mut self, core: u32, z: f32, sigma_z: f32, width: f32) {
        if self.sw.mass_kind != 1 || self.sw.string_winding == 0 {
            self.set_message("Primero crea la cuerda con set_higgs_string.");
            return;
        }
        let core = (core as usize).min(1);
        let c = self.sw.string_cores[core];
        // θ ≈ +sign(n)·φ alrededor del vórtice y −sign(n)·φ alrededor del antivórtice
        let positive = (self.sw.string_winding > 0) == (core == 0);
        let sp = Self::string_core_spinor(positive);
        let (w, h, d) = (self.w as f64, self.h as f64, self.d as f64);
        let nco = self.ncol;
        let st = 2 * 4 * nco;
        let mut b = vec![0.0f64; self.n * st];
        let mut norm = 0.0;
        let (wd, sz) = ((width as f64).max(0.3), (sigma_z as f64).max(0.3));
        for i in 0..self.n {
            let (x, y, zz) = self.coords(i);
            let dx = min_image(x as f64 - c[0], w);
            let dy = min_image(y as f64 - c[1], h);
            let dz = if self.dim == 3 { min_image(zz as f64 - z as f64, d) } else { 0.0 };
            let arg = (dx * dx + dy * dy) / (2.0 * wd * wd) + dz * dz / (2.0 * sz * sz);
            if arg > 40.0 {
                continue;
            }
            let e = (-arg).exp();
            for k in 0..4 {
                let v = sp[k] * e;
                b[i * st + 2 * k * nco] = v.re;
                b[i * st + 2 * k * nco + 1] = v.im;
                norm += v.norm_sqr();
            }
        }
        let s = (self.packet_norm / norm.max(1e-300)).sqrt();
        for v in b.iter_mut() {
            *v *= s;
        }
        self.sw_add_buffer(b, [c[0], c[1], if self.dim == 3 { z as f64 } else { 0.0 }]);
    }
}
