//! # Juego de la Vida Cuántico (Rust + WebAssembly)
//!
//! Caminata cuántica discreta en una malla toroidal 2D (W × H, 4 canales) o 3D (W × H × P,
//! 6 canales) con una no linealidad Kerr saturable. Las estructuras localizadas (solitones)
//! *emergen* de reglas locales en un régimen pre-térmico (§3 de la especificación).
//!
//! Qué es y qué no: es una onda **clásica** de campo medio, no lineal y sin entrelazamiento,
//! sobre una red fija. Está inspirada en los autómatas celulares cuánticos propuestos como
//! modelos discretos del espacio-tiempo, pero no es uno de ellos. El nombre "Juego de la Vida
//! Cuántico" ya se usó para autómatas celulares cuánticos unitarios con entrelazamiento
//! (Arrighi y Grattage, 2010; Bleh, Calarco y Montangero, 2012): este modelo no es de ese tipo.
//!
//! Un paso: Moneda → Vibración (fase saturable acoplada a los vecinos) → Desplazamiento → Capacidad.

use rustfft::num_complex::Complex;
use wasm_bindgen::prelude::*;

mod analog;
mod analysis;
mod bosons;
mod condensate;
mod dirac2d;
mod dmrg;
mod em;
mod envelope;
mod ergo;
mod fftn;
mod floquet;
mod flavour;
mod frontier;
mod gravity;
mod higgs;
mod kaplan;
mod light;
mod linalg;
mod metric;
mod mps;
mod noncubic_pheno;
mod open_quantum;
mod pairs;
mod planck;
mod plasma;
mod qgravity;
mod render;
pub mod research_io;
mod rng;
mod seed;
mod skyrme;
mod sm;
mod spacetime;
mod spectral;
mod spinor;
mod string;
mod studies;
mod sun;
mod twd;
mod unruh;
mod util;
pub mod via_a_qca4;
pub mod via_b_causal_pt;
pub mod via_c_schwinger2f;
mod vortex;
mod walk;
mod walls;
mod weyl;
mod yangmills;

#[cfg(test)]
mod tests;

pub use walk::apply_coin;

/// Número máximo de solitones que guarda el análisis.
pub(crate) const MAX_SOLITONS: usize = 2048;
/// Floats por solitón en `soliton_stats_ptr`.
pub(crate) const SOLITON_FIELDS: usize = 12;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
}

/// El universo: la malla, sus búferes planos y todos los modos.
#[wasm_bindgen]
pub struct QuantumSpacetimeUniverse {
    // --- geometría ---
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) d: usize,
    pub(crate) dim: usize,
    /// Canales del escenario activo (4 en 2D, 6 en 3D; 2 en Dirac 2D…).
    pub(crate) nc: usize,
    /// Canales máximos de la malla (4 en 2D, 6 en 3D): tamaño reservado por celda.
    pub(crate) nc_max: usize,
    pub(crate) n: usize,
    // --- parámetros de la regla ---
    pub(crate) mass: f32,
    pub(crate) cos_m: f32,
    pub(crate) sin_m: f32,
    pub(crate) g: f32,
    pub(crate) e_max: f32,
    pub(crate) alpha: f32,
    pub(crate) e_cap: f32,
    // --- campo (doble búfer AoS: [Re ψ0, Im ψ0, Re ψ1, Im ψ1, …] por celda) ---
    pub(crate) buffer_read: Vec<f32>,
    pub(crate) buffer_write: Vec<f32>,
    pub(crate) e_eff: Vec<f32>,
    pub(crate) potential: Vec<f32>,
    pub(crate) wall: Vec<u8>,
    pub(crate) has_walls: bool,
    pub(crate) has_potential: bool,
    // --- tope (§2.5) ---
    pub(crate) cap_stack: std::collections::VecDeque<u32>,
    pub(crate) cap_flag: Vec<u8>,
    pub(crate) cap_shadow: Vec<f64>,
    pub(crate) cap_touched: Vec<u8>,
    pub(crate) cap_touched_list: Vec<u32>,
    pub(crate) cap_max_iters: usize,
    pub(crate) cap_steps_active: u64,
    pub(crate) cap_last_iters: u32,
    pub(crate) steps_total: u64,
    // --- tiempo ---
    pub(crate) t: u32,
    pub(crate) theta_ref: f32,
    // --- escenario y vista ---
    pub(crate) scenario: u32,
    pub(crate) render_mode: u32,
    pub(crate) view: u32,
    pub(crate) slice: usize,
    /// Exponente del brillo b = (E_eff/E_max)^γ (0,5 por defecto, §4.4).
    pub(crate) brightness_gamma: f32,
    pub(crate) image: Vec<u8>,
    pub(crate) message: String,
    // --- análisis ---
    pub(crate) fft: fftn::FftN<f32>,
    pub(crate) cbuf: Vec<Complex<f32>>,
    pub(crate) spectrum: Vec<f32>,
    pub(crate) spectrum_count: Vec<u32>,
    pub(crate) spectrum_len: usize,
    pub(crate) sol: analysis::SolitonAnalyzer,
    pub(crate) radial: Vec<f64>,
    // --- modo gravedad (§2.6) ---
    pub(crate) grav: gravity::Gravity,
    // --- escenario de vórtices (§2.7) ---
    pub(crate) vort: vortex::VortexState,
    // --- modo electromagnetismo U(1) (§2.9) ---
    pub(crate) em: em::EmField,
    // --- espectro por autocorrelación (§2.10) ---
    pub(crate) autocorr: spectral::AutoCorr,
    // --- color: N = 1 (U(1) o sin campo), 2 = SU(2), 3 = SU(3) (§2.11, §2.14) ---
    pub(crate) ncol: usize,
    pub(crate) ym: yangmills::YmField,
    pub(crate) ym_dens_max: f32,
    // --- campo de Higgs (§2.12): con él activo, la materia queda en pausa ---
    pub(crate) higgs: higgs::HiggsField,
    // --- masa de Dirac por celda y Higgs de Yukawa (§2.13) ---
    pub(crate) dmass: walls::DiracMass,
    // --- instantáneas de E para el análisis del modo que crece (§3.16) ---
    pub(crate) modes: condensate::ModeCapture,
    // --- Dirac 3D y Weyl: espinores en f64 (§2.16 y §2.17) ---
    pub(crate) sw: spinor::SpinorWalk,
    /// Norma de los paquetes de Dirac 3D que se inyecten (`set_packet_norm`).
    pub(crate) packet_norm: f64,
    // --- escenario Kaplan (§2.20): tira 1D en la quinta dirección a momento físico fijo ---
    pub(crate) kap: kaplan::Kaplan,
    // --- escenario luz (§2.19): velocidad de la luz local ---
    pub(crate) light: light::Light,
    // --- potencial conforme de la caminata de Dirac 2D (§2.15): m(r) = m₀·Ω(r) ---
    pub(crate) conf_kind: u32,
    pub(crate) conf_amp: f64,
    pub(crate) conf_pos: [f64; 2],
    pub(crate) conf_width: f64,
    pub(crate) conf_m0: f64,
    // --- mezcla de sabores en el Dirac 2D/1D (§2.22) ---
    pub(crate) fl: flavour::Flavour,
    // --- escenario 6: el modelo estándar clásico (§2.21 y §2.22) ---
    pub(crate) sm: sm::Sm,
    // --- laboratorio de pares (§2.23): no usa la malla ---
    pub(crate) pairs: pairs::PairLab,
    // --- producción clásica de bosones (§2.24) ---
    pub(crate) bos: bosons::Bosons,
    // --- escenario 7: motor 3D sin dobladores (§2.25) ---
    pub(crate) twd: twd::Twd,
    // --- escenario 8: skyrmiones (§2.26) ---
    pub(crate) sky: skyrme::Skyrme,
    // --- escenario 9: Floquet anómalo (§2.27) ---
    pub(crate) flq: floquet::Floquet,
    // --- escenario 10: gravedad análoga (§2.28) ---
    pub(crate) an: analog::Analog,
    // --- fase 14: sistemas cuánticos abiertos de pocos modos (falso vacío y Wigner) ---
    pub(crate) oq: open_quantum::OpenQuantum,
}

impl QuantumSpacetimeUniverse {
    #[inline(always)]
    pub(crate) fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.h + y) * self.w + x
    }

    #[inline]
    pub(crate) fn coords(&self, i: usize) -> (usize, usize, usize) {
        (i % self.w, (i / self.w) % self.h, i / (self.w * self.h))
    }

    /// Celda envuelta en el toro a partir de coordenadas con signo (en 2D, z se ignora).
    #[inline]
    pub(crate) fn wrap_cell(&self, x: i32, y: i32, z: i32) -> usize {
        let xx = util::wrap_i(x, self.w);
        let yy = util::wrap_i(y, self.h);
        let zz = if self.dim == 3 { util::wrap_i(z, self.d) } else { 0 };
        self.idx(xx, yy, zz)
    }

    /// Floats por celda: 2·n_c·N (con N colores; N = 1 sin color).
    #[inline]
    pub(crate) fn stride(&self) -> usize {
        2 * self.nc * self.ncol
    }

    pub(crate) fn set_message(&mut self, msg: &str) {
        self.message = msg.to_string();
        #[cfg(target_arch = "wasm32")]
        warn(msg);
    }

    /// m\* de la banda focalizante: ω(k) ≈ m + |k|²/(2m\*), con m\* = dim·tan m (derivado de la
    /// relación de dispersión exacta de la moneda de Grover; m\* ≈ 1,09 en 2D y ≈ 1,64 en 3D con m = 0,5).
    pub(crate) fn band_mass(&self) -> f64 {
        self.dim as f64 * (self.mass as f64).tan()
    }

    // --- Ganchos de los modos que amplían el paso de la materia (fases posteriores) ---
    #[inline(always)]
    pub(crate) fn gravity_phase_active(&self) -> bool {
        (self.grav.mode != 0 || self.grav.has_ext) && self.grav.gamma_eff() != 0.0
    }
    #[inline(always)]
    pub(crate) fn u1_active(&self) -> bool {
        self.em.mode != 0
    }
    pub(crate) fn u1_matter_step<const NC: usize>(&mut self) {
        self.grover_u1_step::<NC>();
    }
    #[inline(always)]
    pub(crate) fn u1_books_capacity(&self) -> bool {
        self.em.mode == 2 && self.em.book_capacity
    }
    pub(crate) fn book_capacity_current(&mut self, from: usize, to: usize, axis: usize, dir: i32, amount: f64) {
        self.em_book(from, to, axis, dir, amount);
    }
    pub(crate) fn after_matter_step(&mut self) {
        if self.ncol > 1 {
            self.ym_after_step();
        } else if self.em.mode != 0 {
            self.em_after_step();
        }
        if self.grav.mode != 0 || self.grav.exp_on {
            self.gravity_after_step();
        }
    }

    fn default_capacity_ratio(dim: usize) -> f32 {
        if dim == 3 {
            30.0
        } else {
            20.0
        }
    }
}

#[wasm_bindgen]
impl QuantumSpacetimeUniverse {
    /// Crea el universo. `depth` = 1 ⇒ 2D (4 canales); `depth` > 1 ⇒ 3D (6 canales).
    /// Valores por defecto de §2: α = 0,5 y E_cap = 20·E_max en 2D o 30·E_max en 3D.
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, depth: usize, mass: f32, g: f32, e_max: f32) -> QuantumSpacetimeUniverse {
        let w = width.clamp(2, 4096);
        let h = height.clamp(1, 4096);
        let d = depth.clamp(1, 256);
        let dim = if d > 1 { 3 } else { 2 };
        let nc_max = 2 * dim;
        let n = w * h * d;
        let e_max = if e_max.is_finite() { e_max.max(1e-6) } else { 1.0 };
        let mass = if mass.is_finite() { mass.clamp(0.0, std::f32::consts::FRAC_PI_2) } else { 0.5 };
        let g = if g.is_finite() { g } else { 0.0 };
        let fft = fftn::FftN::new(w, h, d);
        let l_min = if dim == 3 { w.min(h).min(d) } else { w.min(h) };
        let kmax = std::f64::consts::PI * (dim as f64).sqrt();
        let dk = 2.0 * std::f64::consts::PI / l_min as f64;
        let spectrum_len = (kmax / dk).round() as usize + 1;
        let mut u = QuantumSpacetimeUniverse {
            w,
            h,
            d,
            dim,
            nc: nc_max,
            nc_max,
            n,
            mass,
            cos_m: mass.cos(),
            sin_m: mass.sin(),
            g,
            e_max,
            alpha: 0.5,
            e_cap: Self::default_capacity_ratio(dim) * e_max,
            buffer_read: vec![0.0; n * 2 * nc_max],
            buffer_write: vec![0.0; n * 2 * nc_max],
            e_eff: vec![0.0; n],
            potential: vec![0.0; n],
            wall: vec![0; n],
            has_walls: false,
            has_potential: false,
            cap_stack: std::collections::VecDeque::with_capacity(n + 1),
            cap_flag: vec![0; n],
            cap_shadow: Vec::new(),
            cap_touched: vec![0; n],
            cap_touched_list: Vec::with_capacity(n),
            cap_max_iters: 200_000,
            cap_steps_active: 0,
            cap_last_iters: 0,
            steps_total: 0,
            t: 0,
            theta_ref: 0.0,
            scenario: 0,
            render_mode: 0,
            view: 0,
            slice: d / 2,
            brightness_gamma: 0.5,
            image: vec![0; w * h * 4],
            message: String::new(),
            fft,
            cbuf: vec![Complex::new(0.0, 0.0); n],
            spectrum: vec![0.0; spectrum_len],
            spectrum_count: vec![0; spectrum_len],
            spectrum_len,
            sol: analysis::SolitonAnalyzer::new(n),
            radial: vec![0.0; 256],
            grav: gravity::Gravity::new(w, h, d),
            vort: vortex::VortexState::new(),
            em: em::EmField::new(w, h, d),
            autocorr: spectral::AutoCorr::new(),
            ncol: 1,
            ym: yangmills::YmField::new(1, dim, 0),
            ym_dens_max: 0.0,
            higgs: higgs::HiggsField::new(),
            dmass: walls::DiracMass::new(),
            modes: condensate::ModeCapture::new(),
            sw: spinor::SpinorWalk::new(),
            packet_norm: 1.0,
            kap: kaplan::Kaplan::new(),
            light: light::Light::new(),
            conf_kind: 0,
            conf_amp: 0.0,
            conf_pos: [0.0; 2],
            conf_width: 1.0,
            conf_m0: 0.0,
            fl: flavour::Flavour::new(),
            sm: sm::Sm::new(),
            pairs: pairs::PairLab::new(),
            bos: bosons::Bosons::new(),
            twd: twd::Twd::new(),
            sky: skyrme::Skyrme::new(),
            flq: floquet::Floquet::new(),
            an: analog::Analog::new(),
            oq: open_quantum::OpenQuantum::new(),
        };
        for p in u.image.chunks_mut(4) {
            p[3] = 255;
        }
        // La sombra f64 del tope se reserva aquí si el tope está activo por defecto y la malla
        // no es enorme; en universos auxiliares grandes, al activar el tope.
        if u.e_cap > 0.0 && n <= 64 * 64 * 64 {
            u.ensure_cap_shadow();
        }
        u
    }

    /// Un paso de la regla del escenario activo.
    pub fn step(&mut self) {
        self.steps_total += 1;
        if self.sm.active {
            self.sm_step();
            return;
        }
        if self.twd.active {
            self.twd_step_universe();
            return;
        }
        if self.scenario == 8 {
            // una iteración de L-BFGS por paso: la relajación se ve en vivo
            if self.sky.active {
                self.skyrme_relax(1);
            }
            self.t = self.t.wrapping_add(1);
            return;
        }
        if self.scenario == 9 {
            if self.flq.active {
                self.flq.period();
            }
            self.t = self.t.wrapping_add(1);
            return;
        }
        if self.scenario == 10 && (self.an.strip.is_some() || self.an.setup == 4) {
            // la tira (láser) avanza; el Hawking por lotes y el Unruh no usan la malla
            self.analog_strip_step();
            return;
        }
        if self.fl.active {
            self.flavour_step();
            return;
        }
        if self.higgs.mode != 0 {
            self.higgs_step();
            return;
        }
        if self.scenario == 5 {
            self.kap.step();
            self.t = self.t.wrapping_add(1);
            return;
        }
        if self.scenario == 4 {
            self.light_step();
            return;
        }
        if self.sw.active {
            self.spinor_step();
            self.after_matter_step();
            if self.autocorr.active {
                self.autocorr_record();
            }
            return;
        }
        match self.scenario {
            2 if self.dim == 2 => {
                if self.ncol > 1 {
                    self.dirac2d_colour_step();
                } else {
                    self.dirac2d_step();
                }
                self.after_matter_step();
                if self.dmass.yukawa == 2 {
                    self.yukawa_step();
                }
            }
            _ if self.ncol > 1 => {
                if self.nc == 6 {
                    self.grover_colour_step::<6>()
                } else {
                    self.grover_colour_step::<4>()
                }
            }
            _ => {
                if self.nc == 6 {
                    self.grover_step::<6>()
                } else {
                    self.grover_step::<4>()
                }
            }
        }
        if self.autocorr.active {
            self.autocorr_record();
        }
    }

    /// Varios pasos por frame en una sola llamada JS → wasm.
    pub fn step_n(&mut self, n: u32) {
        // con el motor de espinores, la copia en f32 para el render se hace una vez al final
        let defer = self.sw.active && !self.autocorr.active;
        self.sw.defer_mirror = defer;
        for _ in 0..n {
            self.step();
        }
        if defer {
            self.sw.defer_mirror = false;
            self.sw_mirror();
        }
    }

    /// Último mensaje de aviso o rechazo (por ejemplo, de `set_scenario`).
    pub fn last_message(&self) -> String {
        self.message.clone()
    }

    // ------------------------------------------------------------------ parámetros

    pub fn set_mass(&mut self, m: f32) {
        if !m.is_finite() {
            return;
        }
        self.mass = m.clamp(0.0, std::f32::consts::FRAC_PI_2);
        self.cos_m = self.mass.cos();
        self.sin_m = self.mass.sin();
        // con el potencial conforme, m₀ es la masa del slider: m(r) = m₀·Ω(r) se recalcula
        if self.conf_kind != 0 && self.scenario == 2 && self.dim == 2 && !self.sw.active {
            let (k, a, p, w) = (self.conf_kind, self.conf_amp, self.conf_pos, self.conf_width);
            self.set_conformal_potential(k, a as f32, p[0] as f32, p[1] as f32, w as f32);
        }
    }

    pub fn set_g(&mut self, g: f32) {
        if g.is_finite() {
            self.g = g.clamp(-10.0, 10.0);
        }
    }

    pub fn set_e_max(&mut self, e_max: f32) {
        if e_max.is_finite() {
            self.e_max = e_max.max(1e-6);
        }
    }

    /// α ∈ [0, 1]: peso de la energía de los vecinos en la vibración de cada cuanto (§2.2).
    pub fn set_neighbor_coupling(&mut self, alpha: f32) {
        if alpha.is_finite() {
            self.alpha = alpha.clamp(0.0, 1.0);
        }
    }

    /// Tope absoluto E_cap (§2.5); 0 = sin tope.
    pub fn set_capacity(&mut self, e_cap: f32) {
        if !e_cap.is_finite() {
            return;
        }
        if self.ncol > 1 {
            // §2.5: el reparto por reescalado no es compatible con la ley de Gauss no abeliana.
            self.set_message("Con SU(N) el tope está desactivado: ninguna corriente en los enlaces compensa el reparto (§2.5).");
            return;
        }
        self.e_cap = e_cap.max(0.0);
        if self.e_cap > 0.0 {
            self.ensure_cap_shadow();
        }
    }

    /// Fracción de los pasos en que actuó el tope desde el último `reset_capacity_stats`.
    pub fn capacity_activity(&self) -> f64 {
        if self.steps_total == 0 {
            0.0
        } else {
            self.cap_steps_active as f64 / self.steps_total as f64
        }
    }

    pub fn reset_capacity_stats(&mut self) {
        self.cap_steps_active = 0;
        self.steps_total = 0;
    }

    /// Iteraciones de reparto del último paso en que actuó el tope.
    pub fn capacity_last_iterations(&self) -> u32 {
        self.cap_last_iters
    }

    pub fn mass(&self) -> f32 {
        self.mass
    }
    pub fn g(&self) -> f32 {
        self.g
    }
    pub fn e_max(&self) -> f32 {
        self.e_max
    }
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
    pub fn capacity(&self) -> f32 {
        self.e_cap
    }

    /// m\* = dim·tan(m), la masa efectiva del fondo de la banda focalizante.
    pub fn effective_mass(&self) -> f64 {
        self.band_mass()
    }

    /// Selecciona el escenario: 0 = materia (defecto), 1 = vórtices (§2.7, solo 2D),
    /// 2 = Dirac (§2.8, 2D con 2 componentes), 3 = Weyl, 4 = luz, 5 = Kaplan, 6 = modelo estándar. Reconfigura n_c, la moneda y los valores por
    /// defecto, y limpia el campo. Los escenarios que aún no existen se rechazan con un mensaje.
    pub fn set_scenario(&mut self, scenario: u32) {
        self.message.clear();
        match scenario {
            0 => {
                self.scenario = 0;
                self.nc = self.nc_max;
                self.set_mass(0.5);
                self.g = 0.1;
                self.e_max = 1.0;
                self.alpha = 0.5;
                self.e_cap = Self::default_capacity_ratio(self.dim) * self.e_max;
                self.render_mode = 0;
            }
            1 => {
                if self.dim != 2 {
                    self.set_message("El escenario de vórtices es solo 2D: en 3D los vórtices son líneas (§2.7).");
                    return;
                }
                self.scenario = 1;
                self.nc = self.nc_max;
                self.set_mass(0.5);
                self.g = -0.03;
                self.e_max = 1.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.set_gravity(0, self.grav.gamma);
                self.render_mode = 0;
            }
            2 => {
                self.set_gravity(0, self.grav.gamma);
                self.scenario = 2;
                self.nc = if self.dim == 3 { 4 } else { 2 };
                self.set_mass(0.1);
                self.g = 0.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.render_mode = 3;
            }
            4 => {
                if self.dim != 2 {
                    self.set_message("El escenario luz es solo 2D (§2.19).");
                    return;
                }
                self.set_gravity(0, self.grav.gamma);
                self.set_em(0, self.em.q as f32, self.em.kappa as f32);
                self.scenario = 4;
                self.nc = 2;
                self.set_mass(0.0);
                self.g = 0.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.render_mode = 0;
                self.light = light::Light::new();
                self.light_alloc();
            }
            5 => {
                self.set_gravity(0, self.grav.gamma);
                self.set_em(0, self.em.q as f32, self.em.kappa as f32);
                self.scenario = 5;
                self.nc = self.nc_max;
                self.render_mode = 0;
                self.kap = kaplan::Kaplan::new();
                let (l, m0, k, f) = (self.kap.ls as u32, self.kap.m0 as f32, self.kap.k, self.kap.fixed);
                self.set_scenario_kaplan(l, m0, k[0] as f32, k[1] as f32, k[2] as f32, f);
            }
            7 => {
                if self.dim != 3 || self.w % 2 != 0 || self.h % 2 != 0 || self.d % 2 != 0 {
                    self.set_message("El motor sin dobladores necesita una malla 3D con lados pares (§2.25).");
                    return;
                }
                self.set_gravity(0, self.grav.gamma);
                self.set_em(0, self.em.q as f32, self.em.kappa as f32);
                self.scenario = 7;
                self.nc = 4;
                self.set_mass(0.0);
                self.g = 0.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.render_mode = 0;
            }
            8 | 9 => {
                self.set_gravity(0, self.grav.gamma);
                self.set_em(0, self.em.q as f32, self.em.kappa as f32);
                self.scenario = scenario;
                self.nc = self.nc_max;
                self.g = 0.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.render_mode = 0;
            }
            6 => {
                if self.dim != 3 {
                    self.set_message("El escenario del modelo estándar es 3D (§2.21).");
                    return;
                }
                self.set_gravity(0, self.grav.gamma);
                self.scenario = 6;
                self.nc = 2;
                self.set_mass(0.0);
                self.g = 0.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.render_mode = 0;
            }
            10 => {
                self.set_gravity(0, self.grav.gamma);
                self.set_em(0, self.em.q as f32, self.em.kappa as f32);
                self.scenario = 10;
                self.nc = self.nc_max;
                self.set_mass(if self.dim == 3 { 0.5 } else { 1.4 });
                self.g = if self.dim == 3 { -0.03 } else { -0.00303 };
                self.e_max = 1.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.render_mode = 0;
            }
            3 => {
                self.set_gravity(0, self.grav.gamma);
                self.scenario = 3;
                self.nc = 2;
                self.set_mass(0.0);
                self.g = 0.0;
                self.alpha = 0.0;
                self.e_cap = 0.0;
                self.render_mode = 3;
            }
            _ => {
                self.set_message(&format!("El escenario {scenario} aún no existe en esta versión."));
                return;
            }
        }
        // Cada escenario empieza sin color (U(1)) y sin Higgs; el color se elige después con
        // set_gauge_group (materia) o set_dirac_colour (Dirac).
        if self.sm.active {
            // el modelo estándar deja el U(1)_Y y el SU(2) dinámicos: se vuelve a un campo vacío
            self.em = em::EmField::new(self.w, self.h, self.d);
            self.ym = yangmills::YmField::new(1, self.dim, 0);
            self.ncol = 1;
        }
        self.sm = sm::Sm::new();
        self.fl = flavour::Flavour::new();
        self.twd = twd::Twd::new();
        self.sky = skyrme::Skyrme::new();
        self.flq = floquet::Floquet::new();
        self.an = analog::Analog::new();
        self.higgs.mode = 0;
        self.dmass.active = false;
        self.dmass.yukawa = 0;
        self.conf_kind = 0;
        if self.ncol > 1 {
            self.resize_colour(1);
        }
        // Dirac 3D (4 componentes) y Weyl (2 componentes, 3D o 2D con k_z fijo) usan el motor de
        // espinores en f64; el Dirac 2D de §2.8 sigue con su paso de 2 componentes.
        self.sw = spinor::SpinorWalk::new();
        match self.scenario {
            2 if self.dim == 3 => self.sw_activate(4, false),
            3 => self.sw_activate(2, true),
            _ => {}
        }
        if self.scenario == 6 {
            self.set_scenario_sm(1, true, true, 0.6, 0.4, 0.8);
        }
        if self.scenario == 7 {
            self.set_scenario_twd(0.1, 1.0, 0.3, 0.0, false);
        }
        if self.scenario == 8 {
            self.skyrme_init(1, 0.2, 5.0);
        }
        if self.scenario == 9 {
            self.floquet_anomalous_init(0, 20, 0.0);
            self.floquet_inject(0, 11);
        }
        self.clear();
        if self.scenario == 10 {
            self.set_analog_setup(if self.dim == 3 { 3 } else { 0 });
        }
    }

    pub fn scenario(&self) -> u32 {
        self.scenario
    }

    // ------------------------------------------------------------------ geometría y estado

    pub fn width(&self) -> usize {
        self.w
    }
    pub fn height(&self) -> usize {
        self.h
    }
    pub fn depth(&self) -> usize {
        self.d
    }
    pub fn dim(&self) -> u32 {
        self.dim as u32
    }
    pub fn channels(&self) -> u32 {
        self.nc as u32
    }
    pub fn time_step(&self) -> u32 {
        self.t
    }

    /// Pone el campo a cero; conserva barreras y muros.
    pub fn clear(&mut self) {
        self.buffer_read.iter_mut().for_each(|v| *v = 0.0);
        self.buffer_write.iter_mut().for_each(|v| *v = 0.0);
        self.e_eff.iter_mut().for_each(|v| *v = 0.0);
        self.t = 0;
        self.theta_ref = 0.0;
        self.sol.reset();
        self.reset_capacity_stats();
        if self.sw.active {
            self.sw_clear();
        }
        if self.sm.active {
            self.sm.psi.iter_mut().for_each(|v| *v = 0.0);
        }
        if self.twd.active {
            self.twd.psi.iter_mut().for_each(|v| *v = rustfft::num_complex::Complex::new(0.0, 0.0));
            self.twd.e.iter_mut().for_each(|v| *v = 0.0);
            self.twd.a.iter_mut().for_each(|v| *v = 0.0);
            self.twd.g_ref.iter_mut().for_each(|v| *v = 0.0);
        }
        if self.fl.active {
            self.fl.psi.iter_mut().for_each(|v| *v = rustfft::num_complex::Complex::new(0.0, 0.0));
            self.fl.norms = vec![0.0; 3];
        }
        if self.scenario == 4 {
            for v in self.light.psi.iter_mut() {
                *v = rustfft::num_complex::Complex::new(0.0, 0.0);
            }
            self.light.tracking = false;
            self.light.rounds = 0;
        }
    }

    /// V = 0 y sin muros.
    pub fn clear_barriers(&mut self) {
        self.has_potential = false;
        self.potential.iter_mut().for_each(|v| *v = 0.0);
        self.wall.iter_mut().for_each(|v| *v = 0);
        self.has_walls = false;
    }

    // ------------------------------------------------------------------ vista

    /// 0 = fase, 1 = dirección del flujo, 2 = potencial gravitatorio Φ, 3 = espín (Dirac),
    /// 4 = campo gauge (U(1): B en rojo/azul; SU(N): energía eléctrica en cian y magnética en
    /// magenta), 5 = carga de color (SU(N)), 6 = Higgs (1 − |φ|/v con B encima).
    pub fn set_render_mode(&mut self, mode: u32) {
        self.render_mode = match mode {
            0..=6 => mode,
            _ => 0,
        };
    }

    /// Solo 3D: 0 = MIP a lo largo de z, 1 = corte z = slice, 2 = volumétrica (la dibuja JS).
    pub fn set_view(&mut self, view: u32, slice: u32) {
        self.view = view.min(2);
        self.slice = (slice as usize).min(self.d - 1);
    }

    pub fn get_image_ptr(&self) -> *const u8 {
        self.image.as_ptr()
    }

    pub fn e_eff_ptr(&self) -> *const f32 {
        self.e_eff.as_ptr()
    }
}
