# Tres estudios numéricos de preguntas abiertas en física discreta fundamental (Fase 17)

> **For English-speaking readers:** See [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md) for the English technical summary covering these Phase 17 pilot studies together with their Phase 18–20 extensions and independent verification audits.

**Repositorio:** *QCA Spacetime Lab* (`src/studies.rs`, `src/tests/phase17.rs`, `src/bin/qsu_cli.rs`, Experimentos 128–130 en `index.html`)  
**Verificación reproducible:**
```bash
cargo test --release --lib phase17 -- --nocapture   # tests t183–t185 (2,0 s, 0 avisos)
cargo run --release --bin qsu_cli -- phase17        # barrido nativo multihilo de producción (3,0 s)
wasm-pack build --target web --release              # compilación WASM SIMD (+simd128)
```

---

## Resumen ejecutivo

Este informe documenta tres investigaciones numéricas autocontenidas y reproducibles sobre preguntas abiertas reales en física teórica discreta, ejecutadas sobre la plataforma de la Fase 16 (DMRG variacional, diagonalización exacta por Lanczos, análisis espectral 3D de autómatas celulares cuánticos y Monte Carlo dinámico de conjuntos causales):

1. **Pregunta abierta #1 (Restauración infrarroja de Lorentz vs. hueco de masa en dos especies interactuantes):** *¿Puede una interacción no perturbativa entre dos especies cuánticas con velocidades desnudas distintas ($v_A \neq v_B$) igualar sus velocidades límite infrarrojas sin abrir un hueco de masa ni destruir su conservación de especie $U(1)_A \times U(1)_B$?*  
   **Resultado:** Encontramos un **trilema no perturbativo limpio**. El acoplamiento densidad–densidad que respeta $U(1)_A \times U(1)_B$ (Canal 1A) mantiene el régimen crítico sin hueco pero **no iguala las velocidades** ($\text{mismatch} = 0,607$ frente a $0,639$ en el control); el intercambio isótropo fuerte de Heisenberg (Canal 1B) destruye la fase sin masa abriendo un hueco real ($\Delta_\infty = 0,7441$) con ley de área exacta ($\Delta S = 0,000$); y la mezcla de sabor sin hueco (Canal 1C) **iguala exactamente las dos velocidades colectivas infrarrojas** ($v_A^{\text{IR}} = v_B^{\text{IR}} = 0,818$, $\text{mismatch} = 0,000$, $\Delta_\infty = 0,0000$), pero al precio inevitable de romper la conservación separada $U(1)_A \times U(1)_B$ ($\max|\Delta N_B| = 0,7321$, frente a $< 10^{-14}$ en el control y en 1A). Esto explica de forma no perturbativa por qué una red fija con interacciones fuertes no iguala fácilmente las velocidades de sectores sin mezcla directa (como el fotón y el electrón).

2. **Pregunta abierta #2 (Clasificación triple de autómatas celulares cuánticos 3+1D y construcción de orden $n = 4$):** *¿Qué familias de QCA locales y unitarios en 3+1D satisfacen simultáneamente (i) simetría CPT-par $\omega(\mathbf{k}) = \omega(-\mathbf{k})$ (sin birrefringencia $n = 1$ ni emisión de Cohen–Glashow), (ii) ausencia de dobladores fermiónicos espurios, y (iii) cancelación de la anisotropía cristalina $O(k^2 \Delta x^2)$?*  
   **Resultado:** Ninguna de las 5 familias locales de 2 componentes (SC, BCC ni FCC, en orden fijo o palindrómico), ni el fotón de 6 componentes de Mlodinow–Brun, ni el Wilson–Dirac cúbico estándar (4 componentes) supera las tres restricciones a la vez. **Construimos y verificamos explícitamente una nueva familia palindrómica unitaria de 4 componentes** —combinando el esténcil isotrópico SC+FCC ($\frac{1}{3}:\frac{2}{3}$) con masa de Wilson en el valor óptimo $r = 2/\sqrt{3}$ en paso simétrico de Strang de tiempo finito $\Delta t = 0,25$— que **pasa las tres restricciones simultáneamente** (`asym = 0,00`, `gap = 0,3500` sin dobladores, `spread A₂ = 1,83×10⁻⁵`) y además anula el término isótropo $O(k^2 \Delta x^2)$ entero (`⟨A₂⟩ = +4,02×10⁻⁶`), alcanzando **invariancia de Lorentz exacta de orden $n = 4$** ($\omega(\mathbf{k}) = |\mathbf{k}| + O(|\mathbf{k}|^5)$) para cualquier $\Delta t$ finito con $6 r \Delta t < \pi$, al coste explícito de romper la simetría quiral mediante el término de Wilson.

3. **Pregunta abierta #3 (Retroacción de materia cuántica sobre gravedad cuántica discreta de conjuntos causales 2D):** *¿Cómo modifica un campo escalar cuántico (propagador retardado exacto de Johnston + acción espectral de 1 lazo de Sorkin–Johnston) el diagrama de fases no perturbativo de Surya (2012) en conjuntos causales dinámicos 2D?*  
   **Resultado:** Verificamos primero que el propagador retardado discreto de Johnston reproduce el continuo $\frac{1}{2}J_0(m\tau)$ con error RMS `0,0105` en espolvoreados 2D ($N = 140, m = 0,8$), revelando simultáneamente fluctuaciones intrínsecas no locales $\sigma_G(\text{cono}) = 0,0425 \to \sigma_G(\text{bulk}) = 0,1636$. Al acoplar $N_f = 0, 1, 2$ campos escalares cuánticos a la integral de caminos sobre $\Omega_{2D}$ ($N = 20$, 600 barridos, 6 semillas independientes), la acción espectral $-N_f S_{\text{mat}}$ penaliza configuraciones con cadenas causales profundas: en la fase cristalina de capas ($\beta > 0$) desplaza la dimensión efectiva de Myrheim–Meyer hacia arriba ($d_{\text{MM}} = 1,75 \pm 0,08 \to 2,12 \pm 0,06$ en $\beta = 1,40$, $> 3,5\sigma$) y atenúa $\langle S_{\text{BD}}\rangle/N$ justo tras la transición ($\beta = 0,45$: $+9,76 \pm 0,75 \to +8,25 \pm 0,38$), mientras que en $\beta = -0,25$ ahonda ligeramente la fase bipartita poco profunda ($-6,93 \pm 0,11 \to -7,49 \pm 0,25$). Reportamos explícitamente las barras de error SEM y la metaestabilidad de Metropolis de intercambio simple a $\beta \ge 0,90$.

---

## 1. Estudio 1: Restauración infrarroja de Lorentz vs. hueco de masa en dos especies interactuantes

### 1.1 Planteamiento y modelo hamiltoniano

En el límite perturbativo (Collins et al., *PRL* 2004; Fase 15 y Fase 16), una red discreta genera a un lazo diferencias de velocidad límite $\Delta c \sim O(g^2)$ entre especies sin supresión por $E/E_P$, y el flujo RG de Chadha–Nielsen a acoplamiento débil (QED) apenas reduce $\Delta c$ un $28\,\%$ entre $M_P$ y $m_e$. Una pregunta abierta natural es si **una interacción fuerte no perturbativa** entre dos especies cuánticas A y B con velocidades desnudas muy distintas puede restaurar una única velocidad de la luz infrarroja $v_A^{\text{IR}} = v_B^{\text{IR}}$ manteniendo ambas especies sin masa (críticas) y distinguibles.

Estudiamos la escalera cuántica interactuante de dos piernas (equivalente por Jordan–Wigner a dos especies fermiónicas 1+1D con interacción de cuatro fermiones $\Delta$ dentro de cada pierna y acoplamientos de peldaño $g_{xy}, g_z$ entre especies):
$$H = \sum_{r=0}^{N_r-1} \Big[ v_A (S^x_{A,r} S^x_{A,r+1} + S^y_{A,r} S^y_{A,r+1}) + v_B (S^x_{B,r} S^x_{B,r+1} + S^y_{B,r} S^y_{B,r+1}) + \Delta (S^z_{A,r} S^z_{A,r+1} + S^z_{B,r} S^z_{B,r+1}) + g_{xy}(S^x_{A,r} S^x_{B,r} + S^y_{A,r} S^y_{B,r}) + g_z S^z_{A,r} S^z_{B,r} \Big],$$
con $v_A = 1,00$, $v_B = 0,45$ y $\Delta = 0,25$ en cuatro regímenes:

- **Control ($g_{xy} = 0, g_z = 0$):** dos líquidos de Luttinger independientes con velocidades exactas de Bethe $v_A^{\text{Bethe}} = 1,154$ y $v_B^{\text{Bethe}} = 0,599$.
- **Canal 1A — Densidad–densidad ($g_{xy} = 0, g_z = 0,60$):** interacción fuerte entre especies que **conmuta con $N_A$ y $N_B$ por separado** (simetría exacta $U(1)_A \times U(1)_B$, análoga a un acoplamiento corriente–corriente sin mezcla de sabor).
- **Canal 1B — Intercambio isótropo de Heisenberg ($g_{xy} = 1,40, g_z = 1,40$):** acoplamiento fuerte simétrico en espín entre piernas.
- **Canal 1C — Mezcla de sabor sin hueco ($g_{xy} = 0,75, g_z = 0$):** hibridación directa $A \leftrightarrow B$ por debajo del umbral de banda ($g_{xy} < 2\sqrt{v_A v_B} = 1,3416$). En el límite libre ($\Delta = 0$), ambas bandas cruzan $E = 0$ en $\cos k_F = \pm g_{xy}/(2\sqrt{v_A v_B})$ con idéntica velocidad de Fermi analítica:
$$v_{\text{harm}}(\Delta = 0) = \frac{2 v_A v_B}{v_A + v_B}\sqrt{1 - \frac{g_{xy}^2}{4 v_A v_B}} = 0,5146.$$
(Al incluir la repulsión intracadena $\Delta = 0,25$, la media armónica renormalizada con las velocidades de Bethe $v_A^{\text{Bethe}} = 1,154, v_B^{\text{Bethe}} = 0,599$ sube a $\approx 0,704$.)

### 1.2 Metodología numérica (cero error de truncación en espectro y transporte)

Para evitar artefactos de truncación SVD de MPS en escaleras críticas ($c = 2$), combinamos tres cálculos complementarios en `src/studies.rs`:
1. **Hueco de masa extrapolado $\Delta_\infty$:** diagonalización exacta de Lanczos (sin truncación) en la escalera **periódica** para $N_r \in \{4, 6, 8\}$ pares (hasta $\binom{16}{8} = 12\,870$ estados en $S^z = 0$ y $\binom{16}{9} = 11\,440$ en $S^z = +1$), extrapolando $\Delta E(N_r) = E_0(S^z = +1) - E_0(S^z = 0) = \Delta_\infty + a/N_r$. Usar exclusivamente $N_r$ pares elimina la alternancia de paridad de cadenas abiertas impares y garantiza un estado fundamental singlete único ($N_A = N_B = N_r/2$).
2. **Velocidades colectivas infrarrojas $(v_A^{\text{IR}}, v_B^{\text{IR}})$:** cálculo exacto del factor de estructura dinámico $S_{\alpha\alpha}(q, \omega) = \sum_n |\langle n | S^z_\alpha(q) | 0 \rangle|^2 \delta(\omega - (E_n - E_0))$ por Lanczos complejo hermítico (45 pasos con reortogonalización completa) en el menor momento infrarrojo no nulo $q = 2\pi / 8 = \pi/4$ de la escalera periódica $N_r = 8$, tomando el primer cúmulo acústico infrarrojo con peso espectral $\ge 6\,\%$ (evitando confundirlo con bandas ópticas UV interbanda) y definiendo $v_\alpha^{\text{IR}} = \omega_\alpha^{\text{IR}}(q) / (2\sin(q/2))$.
3. **Transferencia exacta de sabor $\max_t |\Delta N_B(t)|$ y ley de área DMRG:** evolución exacta por RK4 ($dt = 0,08$, 60 pasos, $t \in [0, 4,8]$) tras inyectar $\sigma^+_{A,0}$ sobre el estado fundamental exacto de $N_r = 6$ (normalizando $\langle\psi|N_B|\psi\rangle/\langle\psi|\psi\rangle$), más DMRG variacional en $N_r = 8$ y $N_r = 16$ ($\chi = 24$ nativo, $\chi = 20$ WASM) para contrastar el crecimiento logarítmico crítico ($\Delta S = S(16) - S(8) > 0$) frente a la saturación por ley de área ($\Delta S = 0$).

### 1.3 Resultados numéricos exactos (`qsu_cli study1`, $\chi_{\text{DMRG}} = 24$)

| Régimen | $\Delta_\infty$ | $S(8) \to S(16)$ ($\Delta S$) | $v_A^{\text{IR}}$ | $v_B^{\text{IR}}$ | $\text{mismatch} = \frac{|v_A-v_B|}{\bar v}$ | $\max_t |\Delta N_B(t)|$ | Diagnóstico físico |
|---|---:|---:|---:|---:|---:|---:|---|
| **Control** ($g_{xy}=0, g_z=0$) | `0,0000` | `1,216 → 1,353` (`+0,137`) | `1,137` | `0,587` | **`0,639`** | `1,20×10⁻¹⁴` | Crítico, $v_A \neq v_B$, $U(1)_A\times U(1)_B$ exacto |
| **1A Dens–Dens** ($g_z=0,60$) | `0,0494` | `1,137 → 1,252` (`+0,115`) | `1,172` | `0,626` | **`0,607`** | `6,22×10⁻¹⁵` | Crítico, $U(1)_A\times U(1)_B$ exacto, **NO iguala $v$** |
| **1B Heisenberg** ($g_{xy}=g_z=1,40$) | **`0,7441`** | `0,217 → 0,217` (**`+0,000`**) | `2,245`* | `2,245`* | `0,000`* | `0,3713` | **Hueco de masa real + ley de área exacta** |
| **1C Mezcla** ($g_{xy}=0,75, g_z=0$) | **`0,0000`** | `0,828 → 0,886` (`+0,058`) | **`0,818`** | **`0,818`** | **`0,000`** | **`0,7321`** | Crítico, **iguala $v_A^{\text{IR}} = v_B^{\text{IR}}$**, pero **rompe $U(1)_A\times U(1)_B$** |

*\*En 1B el modo en $q = \pi/4$ tiene dispersión masiva $\omega(q) = \sqrt{\Delta^2 + v^2 q^2} \approx 1,72$, por lo que $\omega/(2\sin(q/2)) = 2,245$ refleja la brecha de masa y no un cono de luz sin masa.*

### 1.4 Controles nulos y límites honestos

- **Control nulo de Bethe:** en el control desacoplado ($g = 0$), nuestras velocidades infrarrojas medidas en $N_r = 8$ (`1,137` y `0,587`) concuerdan dentro del $1,4\text{–}2,0\,\%$ con las velocidades exactas del ansatz de Bethe para la cadena XXZ ($\Delta = 0,25$: `1,154` y `0,599`).
- **Control nulo de simetría:** en Control y en 1A, la transferencia total de carga entre especies es nula a precisión de máquina (`1,20×10⁻¹⁴` y `6,22×10⁻¹⁵`), confirmando la conservación exacta de $U(1)_A \times U(1)_B$.
- **Límites honestos:** el cálculo exacto es en 1+1D con hasta $N_r = 8$ peldaños (16 sitios) y DMRG hasta $N_r = 16$ (32 sitios). En 1A a $\Delta = +0,25$ el acoplamiento $g_z$ de retrodispersión tiene dimensión $K_A + K_B < 2$, por lo que a $g_z$ mucho mayor abriría un hueco de onda de densidad de carga ($\Delta_\infty = 0,0494$ en $g_z = 0,60$ ya muestra esa tendencia incipiente). El mecanismo demostrado no sustituye un cálculo no perturbativo en 3+1D con campos gauge (p. ej. QED/QCD en red anisótropa), pero establece un modelo soluble de referencia para el dilema de Collins et al. (2004).

---

## 2. Estudio 2: Clasificación triple de 8 familias QCA 3+1D y construcción isotrópica de orden $n = 4$

### 2.1 Planteamiento de la triple restricción

Las Fases 13, 15 y 16 demostraron por separado tres obstáculos para cualquier autómata celular cuántico (QCA) local y unitario en 3+1D como candidato cinemático a escala de Planck:
- **(i) Restricción CPT-par ($n \ge 2$):** si $\omega(\mathbf{k}) \neq \omega(-\mathbf{k})$, el fotón autoconjugado es birrefringente ($\Delta x < 1,38\times 10^{-6}\,\ell_P$ al $95\,\%$ CL por Kislat–Krawczynski, Fase 16) y los fermiones sufren emisión Cherenkov en el vacío de Cohen–Glashow ($\Delta x < 4,9\times 10^{-11}\,\ell_P$ por KM3NeT, Fase 16).
- **(ii) Ausencia de dobladores fermiónicos espurios:** fuera del cono físico $|\mathbf{k}| < 0,35$, toda cuasi-energía debe mantener un hueco limpio $\varepsilon_{\min} > 0,30$ en toda la zona de Brillouin $[-\pi,\pi)^3$ (verificado con barrido $22^3$ + refinamiento local en 4 niveles).
- **(iii) Isotropía direccional en $O(k^2 \Delta x^2)$:** el coeficiente cuadrático efectivo $A_2(\hat{\mathbf{n}}) = (\omega_{\text{sim}}(k\hat{\mathbf{n}})/k - 1)/k^2$ (evaluado sobre 400 direcciones de Fibonacci en $S^2$ con $k = 0,03$) debe tener dispersión direccional $\text{spread}(A_2) = \max_{\hat{\mathbf{n}}} A_2 - \min_{\hat{\mathbf{n}}} A_2 < 10^{-4}$.

### 2.2 Nueva construcción analítica: QCA palindrómico isotrópico SC+FCC Wilson–Dirac de orden $n = 4$

Consideremos un espinor de Dirac de 4 componentes ($\beta, \alpha_x, \alpha_y, \alpha_z$ con $\{\alpha_a, \alpha_b\} = 2\delta_{ab}$, $\{\alpha_a, \beta\} = 0$) evolucionado mediante un paso simétrico de Strang exactamente unitario y local en un entorno cúbico $3\times 3\times 3$ (vecinos SC + FCC):
$$U(\mathbf{k}) = e^{-i\frac{\Delta t}{2} M_W(\mathbf{k})\,\beta}\, e^{-i\Delta t\,\mathbf{S}^{\text{iso}}(\mathbf{k})\cdot\boldsymbol{\alpha}}\, e^{-i\frac{\Delta t}{2} M_W(\mathbf{k})\,\beta},$$
donde:
$$S^{\text{iso}}_a(\mathbf{k}) = \frac{1}{3}\sin k_a\,(1 + \cos k_b + \cos k_c) = k_a\left(1 - \frac{k^2}{6}\right) + O(k^5), \qquad M_W(\mathbf{k}) = r \sum_{a=1}^{3} (1 - \cos k_a) = \frac{r}{2} k^2 + O(k^4).$$
Como $\mathbf{S}^{\text{iso}}\cdot\boldsymbol{\alpha}$ anticonmuta con $\beta$, el espectro exacto de $U(\mathbf{k})$ para **cualquier paso temporal finito $\Delta t$** (con $\Delta t\cdot 6r < \pi$, evitando pseudo-dobladores en $\omega\Delta t = \pi$) obedece la identidad cerrada:
$$\cos(\omega\,\Delta t) = \cos(\Delta t\,M_W(\mathbf{k}))\,\cos(\Delta t\,|\mathbf{S}^{\text{iso}}(\mathbf{k})|).$$
Expandiendo en potencias de $k = |\mathbf{k}|$ hasta $O(k^4)$:
- $|\mathbf{S}^{\text{iso}}(\mathbf{k})|^2 = k^2 - \frac{1}{3} k^4 + O(k^6)$ (**exactamente isótropo hasta $O(k^4)$**; el factor $1/3:2/3$ entre SC y FCC elimina el invariante cúbico $\sum k_a^4$).
- $M_W(\mathbf{k})^2 = \frac{r^2}{4} k^4 + O(k^6)$ (**también exactamente isótropo a $O(k^4)$!**).
- En $\cos(\omega\,\Delta t) = 1 - \frac{\Delta t^2}{2}\omega^2 + \frac{\Delta t^4}{24}\omega^4 - \dots$, los términos de orden $\Delta t^4$ se cancelan idénticamente porque $\frac{\Delta t^4}{24}(|\mathbf{S}|^4 + 6|\mathbf{S}|^2 M_W^2 + M_W^4) = \frac{\Delta t^4}{24}(|\mathbf{S}|^2 + M_W^2)^2 + O(k^6)$ tiene el mismo coeficiente que $\frac{\Delta t^4}{24}\omega^4$! Por tanto:
$$\omega^2(\mathbf{k}) = |\mathbf{S}^{\text{iso}}(\mathbf{k})|^2 + M_W(\mathbf{k})^2 + O(k^6) = k^2 + \left(\frac{r^2}{4} - \frac{1}{3}\right) k^4 + O(k^6).$$
Eligiendo el parámetro de Wilson óptimo:
$$r_* = \frac{2}{\sqrt{3}} \approx 1,15470054,$$
**se anulan simultáneamente la anisotropía cristalina $O(k^2)$ Y el término isótropo $O(k^2)$ completo**, logrando $\omega(\mathbf{k}) = |\mathbf{k}| + O(|\mathbf{k}|^5)$ (**invariancia de Lorentz exacta hasta orden $n = 4$**), $\omega(\mathbf{k}) = \omega(-\mathbf{k})$ exacta, y **cero dobladores** en toda la zona de Brillouin ($M_W \ge 2 r_* = 2,309$ en todos los 7 vértices de dobladores!). Con $\Delta t = 0,25$, el máximo de $\Delta t\,M_W = 0,25 \times 6 r_* = 1,732 < \pi$, de modo que tampoco existen pseudo-dobladores de Floquet en $\varepsilon = \pi$.

### 2.3 Tabla de clasificación exhaustiva de las 8 familias (`qsu_cli study2`, rejilla $22^3$ + 4 niveles)

| # | Familia QCA 3+1D | Asimetría CPT $|\omega(\mathbf{k})-\omega(-\mathbf{k})|$ | Hueco dobladores $\varepsilon_{\min}$ | $\text{spread}(A_2)$ | $\langle A_2 \rangle$ | $\max |A_4|$ | ¿Pasa (i)+(ii)+(iii)? |
|---|---|---:|---:|---:|---:|---:|:---:|
| 0 | `SC-Weyl-Fijo (2c)` | `1,49×10⁻²` | `0,0000` | `7,37×10⁻²` | `-3,81×10⁻²` | `0,0036` | No |
| 1 | `SC-Weyl-Pal (2c)` | `0,00×10⁰` | `0,0000` | `2,21×10⁻¹` | `-1,33×10⁻¹` | `0,0584` | No |
| 2 | `BCC-Weyl-Fijo (2c)` | `8,17×10⁻³` | `0,0004` | `7,17×10⁻²` | `-3,04×10⁻²` | `0,0028` | No |
| 3 | `BCC-Weyl-Pal (2c)` | `3,33×10⁻¹⁶` | `0,0002` | `2,06×10⁻¹` | `-1,00×10⁻¹` | `0,0608` | No |
| 4 | `FCC-Weyl-Pal (2c)` | `2,22×10⁻¹⁶` | `0,0024` | `8,22×10⁻²` | `-5,00×10⁻²` | `0,0338` | No |
| 5 | `SC-Fotón-MB (6c)` | `7,45×10⁻³` | `0,0000` | `1,84×10⁻²` | `-9,52×10⁻³` | `0,0002` | No |
| 6 | `SC-Wilson-Dirac (4c, r=1)` | `0,00×10⁰` | `0,3515` | `1,11×10⁻¹` | `+2,50×10⁻²` | `0,0095` | No |
| **7** | **`Iso-SC+FCC-WD n=4 (4c, r=2/√3)`** | **`0,00×10⁰`** | **`0,3500`** | **`1,83×10⁻⁵`** | **`+4,02×10⁻⁶`** | `0,0113` | **SÍ (1/8)** |

*(Nota: el valor residual `1,83×10⁻⁵` en `spread(A₂)` de la Familia 7 es precisamente $A_4 \cdot k_2^2 \approx 0,011 \times 0,03^2 \approx 10^{-5}$ procedente del término $O(k^4)$ evaluado en $k_2 = 0,03$; el hueco `0,3500` coincide exactamente con el radio de exclusión $|\mathbf{k}| = 0,35$ porque no existe ningún mínimo espurio en toda la zona de Brillouin.)*

### 2.4 Límites honestos y precio físico de la construcción

- **Rotura explícita de la simetría quiral:** eliminar los dobladores en 3+1D exige el término de Wilson $M_W(\mathbf{k})\beta$ (que mezcla quiralidades L y R); por tanto, esta familia describe fermiones de Dirac vectoriales, no teorías gauge quirales como la interacción débil del Modelo Estándar.
- **No inmuniza frente a bucles interactuantes:** mejorar la cinemática libre hasta orden $n = 4$ ($O(E^4/E_P^4)$, completamente inobservable hoy por tiempo de vuelo) **no** evita las correcciones radiativas marginales de dimensión 4 (Collins et al., 2004; Estudio 1) en cuanto se acoplan interacciones cuánticas sobre la red fija.
- **Implementación en espacio real:** $e^{-i\Delta t \mathbf{S}^{\text{iso}}\cdot\boldsymbol{\alpha}}$ es exactamente local en espacio de momentos y se factoriza de forma local y unitaria en espacio real mediante compuertas de dos sitios sobre los 3 ejes SC y las 6 diagonales de cara FCC (con precisión de Trotter simétrico de orden $O(\Delta t^2)$ adicional si se descompone por ejes).

---

## 3. Estudio 3: Conjuntos causales dinámicos 2D acoplados a materia cuántica (Johnston + Sorkin–Johnston)

### 3.1 Planteamiento y formulación matemática

En la Fase 16 reprodujimos el diagrama de fases de gravedad pura de conjuntos causales 2D (Surya, *CQG* 2012) sobre el espacio de 2-órdenes $\Omega_{2D}$ con la acción discreta de Benincasa–Dowker $S_{\text{BD}} = N - 2 N_1 + 4 N_2 - 2 N_3$. Una pregunta abierta central de la teoría de conjuntos causales es cómo responde este diagrama de fases al acoplar **campos cuánticos dinámicos** directamente al orden causal sin introducir ninguna malla de fondo.

Implementamos dos ingredientes exactos en `src/studies.rs`:
1. **Propagador retardado discreto de Johnston (*PRL* 103, 180401, 2009):** dada la matriz causal estrictamente triangular superior $C_{ij} = 1$ si $i \prec j$ (y $0$ en caso contrario), el propagador retardado exacto de un campo escalar libre de masa $m$ en 1+1D satisface
$$\left(I + \frac{m^2}{2} C\right) G_R = \frac{1}{2} C,$$
resuelto exactamente por sustitución hacia atrás en $O(N^3/6)$. En el continuo de Minkowski 1+1D, $G_R^{\text{cont}}(x, y) = \frac{1}{2} J_0(m\tau)\,\theta(t_y - t_x - |x_y - x_x|)$.
2. **Acción espectral de 1 lazo de Sorkin–Johnston (Afshordi, Aslanbeigi y Sorkin, *JHEP* 2012):** a partir del operador conmutador antisimétrico de Pauli–Jordan $\Delta = G_R - G_R^T$, definimos la acción efectiva cuántica de un campo escalar libre de masa $m = 0,8$ sobre el conjunto causal $\mathcal{C}$ como
$$S_{\text{mat}}[\mathcal{C}] = \frac{1}{2}\ln\det(I + \Delta^T \Delta),$$
evaluada exactamente mediante factorización de Cholesky $L L^T = I + \Delta^T \Delta \implies S_{\text{mat}} = \sum_{i=1}^N \ln L_{ii}$ (siempre finita, real, estrictamente positiva y exactamente invariante bajo reetiquetado del orden causal). La función de partición acoplada con $N_f \in \{0, 1, 2\}$ campos escalares cuánticos es:
$$Z(\beta, N_f) = \sum_{\mathcal{C} \in \Omega_{2D}} \exp\big(+\beta\,S_{\text{BD}}[\mathcal{C}] - N_f\,S_{\text{mat}}[\mathcal{C}]\big).$$

### 3.2 Parte A: Verificación del propagador de Johnston y varianza no local intrínseca

Sobre 10 espolvoreados de Poisson en un diamante causal de Minkowski 2D ($N = 140$, volumen $V = N$, masa $m = 0,8$, $\tau \in [0, 5]$ en 10 intervalos):
- **Acuerdo con el continuo $\frac{1}{2}J_0(m\tau)$:** error cuadrático medio **`err_rms = 0,0105`** (acuerdo del $\sim 2\,\%$ respecto a $G_R(0) = 0,5$).
- **Firma no local intrínseca:** aunque el promedio $\langle G_R(\tau)\rangle$ reproduce $\frac{1}{2}J_0(m\tau)$ con alta precisión, cada realización exhibe fluctuaciones intrínsecas no locales entre pares a igual tiempo propio $\tau$: **`σ_G(cono, τ ≈ 0,25) = 0,0425`** y **`σ_G(bulk, τ ≈ 4,75) = 0,1636`** (crecen con el volumen del intervalo causal de Alexandrov $\propto \tau^2$ debido a las fluctuaciones discretas de cadenas intermedias).

### 3.3 Parte B: Retroacción de $N_f = 0, 1, 2$ campos cuánticos sobre el diagrama de fases de Surya (`qsu_cli study3`, $N = 20$, 600 barridos, 6 semillas independientes)

Todos los valores muestran **media $\pm$ error estándar de la media (SEM)** calculados sobre 6 cadenas de Markov independientes con semillas distintas:

| $\beta$ | $N_f = 0$ (gravedad pura) $\langle S_{\text{BD}}\rangle/N$ ; $d_{\text{MM}}$ | $N_f = 1$ (1 campo cuántico) $\langle S_{\text{BD}}\rangle/N$ ; $d_{\text{MM}}$ | $N_f = 2$ (2 campos cuánticos) $\langle S_{\text{BD}}\rangle/N$ ; $d_{\text{MM}}$ |
|---:|---|---|---|
| **`-0,25`** | `-6,93 ± 0,11` ; `1,78 ± 0,02` | `-7,34 ± 0,11` ; `1,70 ± 0,03` | `-7,49 ± 0,25` ; `1,76 ± 0,06` |
| **`+0,00`** | `+0,05 ± 0,03` ; `2,02 ± 0,01` | `+0,04 ± 0,04` ; `2,03 ± 0,01` | `+0,14 ± 0,02` ; `2,04 ± 0,01` |
| **`+0,45`** | `+9,76 ± 0,75` ; `1,70 ± 0,07` | `+8,86 ± 0,54` ; `1,69 ± 0,07` | `+8,25 ± 0,38` ; `1,73 ± 0,09` |
| **`+0,90`** | `+8,80 ± 0,65` ; `1,75 ± 0,12` | `+8,19 ± 0,56` ; `1,92 ± 0,13` | `+8,39 ± 0,33` ; `1,94 ± 0,06` |
| **`+1,40`** | `+7,47 ± 0,87` ; `1,75 ± 0,08` | `+6,40 ± 0,62` ; `2,12 ± 0,06` | `+7,17 ± 0,48` ; `1,92 ± 0,07` |
| **`+1,90`** | `+7,62 ± 0,74` ; `1,76 ± 0,10` | `+5,80 ± 0,38` ; `2,07 ± 0,08` | `+7,43 ± 0,79` ; `1,95 ± 0,11` |

*(En el barrido WASM más pequeño $N = 16$, 260 barridos y 4 semillas del Experimento 130, el mismo sentido físico se reproduce: en $\beta = 0$, $d_{\text{MM}}$ sube de `2,017` a `2,118`, y en $\beta = +1,90$, $\langle S_{\text{BD}}\rangle/N$ pasa de `6,56 ± 1,07` a `4,99 ± 0,25`.)*

### 3.4 Interpretación física y límites honestos

1. **Mecanismo de la retroacción cuántica:** la acción espectral $S_{\text{mat}} = \frac{1}{2}\ln\det(I + \Delta^T\Delta)$ crece con la densidad y profundidad de las relaciones causales (autovalores grandes del conmutador de Pauli–Jordan $\Delta$). En consecuencia, el peso $\exp(-N_f S_{\text{mat}})$:
   - **Desfavorece cadenas causales largas y profundas:** en la fase cristalina de capas ($\beta > 0$), empuja la dimensión efectiva de Myrheim–Meyer hacia arriba ($d_{\text{MM}}: 1,75 \pm 0,08 \to 2,12 \pm 0,06$ en $\beta = 1,40$, desplazamiento $> 3,5\sigma$) y reduce $\langle S_{\text{BD}}\rangle/N$ tras la transición ($\beta = 0,45$: $+9,76 \pm 0,75 \to +8,25 \pm 0,38$).
   - **Favorece ligeramente la fase bipartita poco profunda en $\beta = -0,25$:** en un cristal bipartito de dos anticadenas ($N_2 = 0$, $C^2 = 0$), el propagador trunca al primer orden $G_R = \frac{1}{2}C$ y tiene menor determinante espectral, por lo que $N_f > 0$ hace $\langle S_{\text{BD}}\rangle/N$ ligeramente más negativo (`-6,93 ± 0,11 → -7,49 ± 0,25`).
2. **Límites honestos y diagnóstico de error:**
   - **Tamaño finito y dimensión:** el estudio es en **1+1D** con $N = 16\text{–}20$ elementos; no demuestra gravedad tensorial de espín 2 en 3+1D (ni el límite termodinámico $N \to \infty$ de Glaser, O'Connor y Surya, *CQG* 2018).
   - **Metaestabilidad a $\beta \ge 0,90$:** las barras de error grandes (`SEM ~ 0,5–0,8`) y la no monotonicidad entre $N_f = 1$ y $N_f = 2$ en $\beta = 1,40\text{–}1,90$ revelan atrapamiento en mínimos locales cristalinos bajo Metropolis de intercambio simple. Extraer el desplazamiento exacto del punto crítico $\beta_c(N_f)$ para $N = 40\text{–}80$ requeriría *parallel tempering* (intercambio de réplicas en $\beta$).
   - **Acción de materia regularizada:** $S_{\text{mat}} = \frac{1}{2}\ln\det(I + \Delta^T\Delta)$ es un regulador UV positivo y covariante inspirado en el espectro de Sorkin–Johnston, no el determinante lorentziano sin regularizar $\det(\Box + m^2)$ (cuya continuación analítica en conjuntos causales sigue siendo un problema abierto en la literatura).

---

## 4. Conclusión global y siguiente paso hacia publicación arbitrada

De los tres estudios, el que posee la estructura más limpia y autocontenida para evolucionar hacia una nota o artículo arbitrado es la **combinación del Estudio 2 con la corrección de birrefringencia de la Fase 13/16** (`scripts/verify_via_a_mlodinow_brun.py`): aporta un teorema/clasificación completa sobre 8 familias QCA 3+1D locales y unitarias junto con la construcción explícita cerrada del QCA palindrómico isotrópico SC+FCC Wilson–Dirac ($r = 2/\sqrt{3}$) con invariancia de Lorentz exacta de orden $n = 4$ y cero dobladores. Como trabajo de continuación, el Estudio 1 admite escalado DMRG en redes más largas con sector $U(1)$ explícito y el Estudio 3 admite *parallel tempering* sobre $N = 32\text{–}64$ elementos en `qsu_cli` para medir el corrimiento exacto $\beta_c(N_f)$ con control de tamaño finito.
