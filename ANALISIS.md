# Análisis físico

Este documento reúne el análisis físico completo del motor y de sus experimentos. Los números "medidos" salen de los tests de `src/tests/` y del laboratorio de `index.html`, ejecutados con esta implementación.

## Fase 1: el núcleo

### (a) El papel real de la saturación

La regla original justificaba E_eff = E/(1 + E/E_max) como remedio contra "singularidades numéricas". En la red no hay singularidades: el paso conserva la norma exactamente, y la fase vive módulo 2π, así que nada diverge. La saturación hace falta por otras tres razones:

1. **Acota el desfase no lineal por paso a g·E_max.** Es lo que mantiene la dinámica en el régimen pre-térmico (apartado c). Sin saturación, el colapso concentra la norma, g·E crece sin límite y el sistema se calienta hasta el ruido de temperatura infinita. En 3D pasa antes de 500 pasos.
2. **Detiene el colapso crítico de la NLS cúbica en 2D** (supercrítico en 3D). Sin saturación, la NLS focalizante colapsa en un tiempo finito si su energía es negativa (Glassey, 1977).
3. **Fija un tamaño finito para los solitones:** unas 5–10 celdas de diámetro en 2D y unas 8–12 en 3D.

**E puede superar E_max.** E_max es la escala de saturación de la *vibración*, no un límite de la energía almacenada. Lo acotado es E_eff ∈ [0, E_max) y, con ella, la frecuencia no lineal, que nunca supera g·E_max rad/paso. Sin el tope absoluto, los picos llegan a E ≈ 15–50 con E_max = 1. Ese tope (E_cap, §2.5) es otra regla: una pila de arena divisible que reparte el exceso entre las vecinas sin perder norma. Por defecto vale 20·E_max en 2D y 30·E_max en 3D. Tiene que quedar por encima de la densidad natural de los solitones: con 10·E_max actúa en el 74 % de los pasos y destruye la emergencia.

### (b) Por qué la sopa debe estar en la banda focalizante

La moneda C(m) = e^{−imG} tiene, en k = 0, el espinor simétrico (1, …, 1)/√n_c con ω = +m, y las otras n_c − 1 bandas en ω = −m. La relación de dispersión exacta del paso sale del lema del determinante:

  (1/n_c)·Σ_k d_k/(λ − e^{im}·d_k) = 1/(e^{−im} − e^{im}),  con d_k = e^{−i**q**·**e**_k} y λ = e^{−iω}.

Al desarrollar alrededor de **q** = 0 aparece

  ω(**q**) ≈ m + |**q**|²/(2m\*),  con **m\* = dim·tan m**,

es decir, m\* = 2·tan(0,5) = 1,093 en 2D y m\* = 3·tan(0,5) = 1,639 en 3D. Son los valores de la especificación (≈ 1,09 y ≈ 1,6), y la app los usa en el HUD, en el laboratorio y en la marca k\* del panel de diagnóstico.

Ese punto es un **mínimo** de banda, y con g > 0 la fase e^{+ig·E_eff} baja la frecuencia local: la interacción es atractiva para esa banda. Cerca del fondo, el modelo es una NLS saturable i∂_tφ = −∇²φ/(2m\*) − g·f(|φ|²)·φ, con f(E) = E/(1 + E/E_max). Con ruido blanco independiente en los n_c canales, la norma se reparte entre la banda focalizante y las demás, y no condensa nunca. Pasa lo mismo en la NLS discreta, donde las excitaciones localizadas solo aparecen por encima de la línea de temperatura infinita (Rasmussen et al., 2000). Por eso la sopa es ruido suavizado (box-blur de radio 6 en 2D y 4 en 3D, 3 pasadas por eje) en el espinor simétrico.

**Predicción de Bogoliubov** para un fondo uniforme ρ̄: con ε_k = k²/(2m\*) y μ' = g·ρ̄·f′(ρ̄) = g·ρ̄/(1 + ρ̄/E_max)², la tasa de la inestabilidad modulacional es σ(k) = √(ε_k·(2μ' − ε_k)). Su máximo vale σ_max = μ' y aparece en k\* = √(2m\*ρ̄g)/(1 + ρ̄/E_max). Con los valores por defecto, σ_max = 0,0222 por paso y k\* = 0,220. El test 10 mide los dos en una malla de 256² (tolerancias de ±15 % y ±20 %) y pasa.

### (c) Calentamiento de Floquet y régimen pre-térmico

Un paso es un mapa no lineal discreto en el tiempo que conserva la norma, pero ninguna energía. De forma genérica, se calienta hasta un estado de temperatura infinita: ruido a escala de celda con IPR ≈ 1 + 1/n_c (1,25 en 2D y 1,17 en 3D). Lo mismo se ha descrito en caminatas no lineales (Di Molfetta et al., 2015) y en redes de paso discreto (Moorcroft et al., 2025).

La condensación solo aparece si el desfase no lineal por paso es pequeño frente a m:

| g·E_max/m | Qué pasa |
|---|---|
| ≈ 0,2 (valor por defecto) | Los solitones viven ≳ 30 000 pasos en 2D |
| ≈ 0,4 | Se forman y se funden despacio |
| ≈ 0,6 | Decaen en ~6000 pasos |
| ≳ 0,9 | Ruido en menos de 3000 pasos |

El HUD muestra g·E_max/m, avisa por encima de 0,4 y da la vida media orientativa t½ ≈ 10³·(m/(g·E_max))^2,8 pasos. Esa ley de potencias ajusta mejor que una exponencial de Floquet. Es una estimación del modelo base en 2D, no una ley demostrada.

El régimen es **transitorio**. El estado final es siempre ruido; lo que cambia es cuánto tarda en llegar. Por eso los solitones se llaman *longevos* y no estables. No es el equilibrio de una NLS que conserva la energía (Rumpf y Newell, 2001), porque aquí no hay energía conservada.

### (d) Qué cambia entre 2D y 3D

- **Criticidad.** La NLS cúbica es crítica en 2D y supercrítica en 3D. En 3D el colapso, frenado por la saturación, llega hasta la escala de la red: los solitones duran unos 5000 pasos y después la norma pasa a miles de picos de una celda. Esos picos son "breathers" discretos de larga vida (§3.1). En esa fase el IPR sigue alto sin ningún solitón: el HUD lo avisa y el laboratorio cuenta solitones (regiones de ≥ 5 celdas con E > 8·⟨E⟩).
- **Número de solitones.** Son de ~8–12 celdas, así que caben ~2 en 48³ y ~4 en 64³. El 3D sirve para pocas partículas y sus interacciones; para ver poblaciones, el 2D.
- **Coste.** 48³ con 6 canales cuesta ~2,5 veces el trabajo de 256² con 4. En el navegador de la prueba, los dos van a 60 FPS (4 pasos por frame en 2D y 1 en 3D).
- **Visualización.** La vista por defecto es la MIP: un corte solo enseña una fracción de las estructuras. En la MIP, el brillo de la especificación, (E_eff/E_max)^0,5, satura todas las columnas, porque el máximo de 48 vóxeles de fondo ya está cerca de E_max. Por eso el frontend usa ahí el exponente 2. Es un cambio de visualización y se puede ajustar con un slider.
- **Isotropía.** La red cúbica solo es isótropa a orden k²; el "cono de luz" es un octaedro.

### (e) Qué comportamientos se esperan y cuáles no están garantizados

**Verificados con esta implementación:**
- La condensación desde la sopa: IPR > 5 en 2D (tres semillas) y en 3D (48³, tres semillas), frente a IPR < 3 con g = 0 (tests 7, 8 y 11).
- El umbral de existencia (test 12, experimento 1).
- Momento → velocidad (test 13; en el experimento 2, v·m\*/k = 0,941, 0,941 y 0,934).
- La inercia en una trampa (test 14; en el experimento 3, −2,4 %).
- La conservación aproximada del momento (test 15).
- Las colisiones en fase (fusión inelástica) y en contrafase (nunca se fusionan) (test 16, experimento 5).
- La ligadura que crece con la norma (test 116; en el experimento 6, ω = 0,4873, 0,4668, 0,4582, 0,4518 y 0,4437 para N = 150–900).

**No garantizado:**
- La aniquilación en sentido estricto. La norma se conserva, así que un choque destructivo solo puede convertir los solitones en radiación.
- Órbitas estables ("átomos") sin gravedad. La atracción depende de la fase y el par se funde en el primer acercamiento.
- La estabilidad a largo plazo, porque el régimen es pre-térmico.

El anclaje de Peierls–Nabarro (Jenkinson y Weinstein, 2016) no debería notarse con solitones de 5–12 celdas.

### (f) Interpretación en unidades de Planck

**Lo que es coherente:**
- Con 1 celda = ℓ_P y 1 paso = t_P, la velocidad máxima de la caminata, 1 celda/paso, es exactamente c.
- Con ħ = 1, un desfase de φ rad/paso es una energía φ·E_P (E = ħω literal). La masa de la moneda m = 0,5 es 0,5 m_P ≈ 1,1·10⁻⁸ kg, y la vibración máxima de un cuanto, g·E_max = 0,1 E_P.
- El selector de unidades convierte el HUD y el laboratorio a red, Planck o SI. La norma se convierte con la convención 1 unidad = E_P/E_cap: es una convención de visualización, no un resultado.

**Sus límites:**
- Los solitones tienen frecuencia interna ~0,44–0,49 E_P: son objetos de masa de Planck, hipotéticos. Un electrón es 4·10⁻²³ veces más ligero.
- Una red cúbica a la escala de Planck rompe la invariancia de Lorentz. Las observaciones exigen que una dependencia lineal de c con la energía esté por encima de la masa de Planck (Fermi, Abdo et al., 2009).
- El modelo base no tiene gravedad: la interacción entre solitones depende de su fase. La motivación de E_cap como "la energía que colapsaría el cuanto" es solo una motivación.

### (g) Qué propiedades de partícula se verifican

| Propiedad | Verificada | Medida aquí |
|---|---|---|
| Masa mínima (umbral de existencia) | Sí | Norma 50: pico 0,026; norma 300: pico 3,19 |
| Momento → velocidad, p = M·v | Sí | v·m\*/k = 0,94 |
| Masa inercial (trampa armónica) | Sí, apenas depende de la norma (análogo de Galileo) | Ω −2,4 % frente a √(κ/m\*) |
| Conservación del momento | Aproximada | \|ΔP/P\| < 1 % a los 300 pasos |
| Interacción según la fase | Sí | Contrafase: rebote; fase: fusión |
| Defecto de masa (ligadura) | Sí | m − ω crece con la norma |

**No se verifican:**
- Espín.
- Carga (salvo en los escenarios posteriores).
- Estadística cuántica.
- Órbitas ligadas sin gravedad.
- Estabilidad indefinida.

Los solitones son paquetes coherentes de una onda clásica, no partículas cuánticas.

## Fase 2: gravedad (§2.6 y §3.4)

### (h) El modo gravedad: qué emerge, qué se puso a mano y por qué no es relatividad general

**Qué es.** Cada celda guarda un campo Φ cuya fuente es E − ⟨E⟩, y Φ se suma a la vibración: ω_nl ← ω_nl + γ·Φ. Hay dos variantes:
- **Estática:** −∇²_red Φ = 4π(E − ⟨E⟩), resuelta por FFT y anulando k = 0 (el *Jeans swindle*). Es acción instantánea a distancia.
- **Onda:** un integrador local y causal de dos niveles con velocidad c_g ≤ 1/√dim. Viaja como mucho a 0,71·c en 2D y a 0,58·c en 3D.

Es el sistema Newton–Schrödinger, el mismo que emulan los experimentos ópticos de Bekenstein et al. (2015), no la relatividad general. La norma se sigue conservando exactamente, porque la fase es local.

**Qué se puso a mano:**
- Que la fuente sea la energía almacenada: eso hace universal la gravedad.
- La ecuación de Poisson: la ley de Gauss fija la ley de distancias (log r en 2D y 1/r en 3D).

**Qué emerge y no era obvio** (medido con esta implementación, 256², γ = 10⁻⁶):

| Propiedad | Medido aquí | Referencia |
|---|---|---|
| Universalidad, pareja en fase | a = 1,33·10⁻⁵ (predicción 1,27·10⁻⁵) | +3 % |
| Universalidad, pareja en contrafase | 6,0·10⁻⁶: también atrae | 6,3·10⁻⁶ |
| Equivalencia, pruebas de 150 y 300 | 1,78·10⁻⁵ y 2,06·10⁻⁵ (14 % de diferencia; 0,70 y 0,81 de la predicción) | 9 %; 0,89 y 0,81 |
| Ley de distancias 2D, a·d a 30, 45 y 60 celdas | 7,9, 8,3 y 6,9 (×10⁻⁴) | 8,1, 8,3 y 7,0 |
| Causalidad (variante onda): llegada a 40 celdas | t = 89 (el frente geométrico llega en 10 + 40/c_g = 90) | t = 84 |
| Causalidad (variante estática) | t = 11 | inmediata |
| Corrimiento al rojo | −6,96·10⁻⁴ frente a −8,36·10⁻⁴ (83 %): el reloj cercano va más lento | 84 % |
| Órbita ligada | separación 38,5–52,2 con d = 48; giro al 98 % de 2v/d | 45,5–54; 85 % |

Un solitón compuesto y no lineal responde como una masa puntual. Sin gravedad, la interacción depende de la fase y no hay órbitas.

**Por qué no es relatividad general:**
- Φ es un escalar sin dinámica propia más allá de una ecuación de onda.
- La luz no se curva.
- No hay métrica dinámica.
- La intensidad γ se elige para no salir del régimen pre-térmico (γ·ΔΦ ≪ m), no para reproducir G.

Tampoco puede emerger una gravedad termodinámica (Jacobson, 1995; Verlinde, 2017): la caminata solo conserva la norma y es un campo clásico sin entrelazamiento.

### (t), la parte de gravedad y expansión

**La expansión es solo un γ que crece como a.** En el tiempo superconforme, dτ = dt/a², el sistema de Schrödinger–Poisson comóvil es i∂_τψ = −∇²ψ/2m\* + a(τ)·V[ψ]: el término cinético no cambia y la gravedad se multiplica por a. Basta, pues, con γ_ef = γ₀·a(τ), con a(τ) = 1/(1 − Cτ/2)² y C² = 8πγ₀⟨E⟩/(3m\*). Medido en una caja de 64×8×8:
- d ln δ/d ln a = 0,994 (Einstein–de Sitter da 1).
- Sin expansión, la tasa de Jeans sale 0,01975 frente a 0,01936 (+2 %).

Lo que limita la formación de halos en la red: hace falta λ_J ≪ L, es decir, una expansión rápida, y entonces la fase gravitatoria por paso deja de ser pequeña. Por eso la expansión se detiene sola cuando γ_ef·ΔΦ llega a 0,5·m, y el HUD avisa desde 0,05·m.

**El contacto se cancela promediando sobre la fase relativa y la gravedad no.** La interacción de contacto entre dos solitones va como cos φ. La gravedad solo depende de E, que no ve la fase. Con cuatro fases relativas (0, π/2, π y 3π/2), el promedio deja solo la parte gravitatoria. El test 122 (96³, d = 40) lo comprueba: la parte gravitatoria es atractiva, está entre 0,5 y 1,1 veces 2γ⟨∇Φ⟩/1,32 y varía entre fases menos de un 30 %.

**La estrella de Bose en el régimen cinético.** El criterio de Jeans que vale para un gas con velocidades es k_J² = 8πγn/(m\*v²), no el de un condensado frío. Con L̃ = 20 y la mitad del límite de Jeans (48³), el gas nuclea un solitón de Schrödinger–Poisson:
- Test 123: max ρ/⟨ρ⟩ > 60 antes del paso 9000, frente a < 15 sin gravedad, con el r_c del perfil final a menos del 15 % de ρ₀·r_c⁴·γ·m\*² = 0,385.
- El núcleo de Schive (64³) converge en tiempo imaginario a r_c = 7,32 (referencia 7,40), con un error de perfil del 0,6 %. En la caminata pasa a 7,50, 7,54, 7,68 y 7,66 a los 250–1000 pasos, y ρ₀ baja un 14 %, como en la referencia.

### (s), la masa de un solitón ante una fuerza lenta: desviación respecto a la especificación

La especificación (§3.4) dice que un solitón 3D de norma 3000 tiene una masa de 1,32 por unidad de norma ante una fuerza lenta, frente a la masa de banda m\* = 1,64 de un empujón. Con ese 1,32 explica el factor ≈ 1,24 de la gravedad 3D. **Esta implementación no reproduce esa interpretación.**

- El empujón da M = k/v = 1,58–1,65, como en la referencia.
- Con el campo Φ = 75·sin(2πx/64) durante 100 pasos, la fuerza por unidad de norma *del solitón* y su aceleración dan F/a = 1,62: la masa de banda, no 1,32.
- El 1,32 se reproduce exactamente (1,33) al dividir el impulso total por la **norma total de la caja**. Esa norma incluye el ~20 % de radiación que el solitón soltó al formarse y que no viaja con él (en r < 10 solo queda el 80,5 % de la norma). Del mismo modo, el "M = 1,42 con el impulso acumulado una vez apagado" de la referencia se reproduce (1,422) con la misma normalización.
- Conclusión: el 1,32 y el 1,42 son un artefacto de normalización, no una inercia distinta ante fuerzas lentas.

El factor de la gravedad 3D con una fuente congelada sí aparece, algo menor: a/(γ⟨∇Φ⟩/m\*) = 1,29, 1,18, 1,15 y 1,15 a d = 12, 16, 20 y 24, constante desde d = 16 (la referencia daba 1,44, 1,30, 1,25 y 1,24). Su origen queda abierto. No es una masa menor ante una fuerza uniforme, que da m\*. Es compatible con una respuesta del solitón al campo no uniforme de una fuente puntual: deformación de marea y ligadura adicional.

El test 115 comprueba las dos cosas: la normalización de la referencia (1,32 ± 0,1) y la masa por solitón (m\* ± 10 %). El experimento 71 muestra las dos.

## Fase 3: vórtices y Dirac 2D (§2.7, §2.8 y §3.5)

### (i) Qué falta para las partículas del modelo estándar y qué peldaños cubren estos escenarios

**De la sopa no emergen partículas del modelo estándar.** Sus campos (fermiones quirales, gluones, W, Z, fotón y Higgs) se ponen a mano en escenarios aparte. Faltan:
- El espín ½ en el escenario principal.
- La estadística cuántica y la creación de pares: los escenarios son campos clásicos de campo medio.
- Los campos gauge no abelianos cuánticos y el confinamiento.
- Una teoría gauge quiral exacta en la red: Nielsen–Ninomiya alcanza también a la caminata de Floquet.
- Los parámetros (se ponen a mano, como en el propio modelo estándar) y la jerarquía de masas: un electrón pesa 4·10⁻²³ masas de Planck.

Los precedentes serios de partículas emergentes en redes usan cúbits entrelazados (Levin y Wen, 2005), no ondas clásicas.

**Escenario de vórtices: cargas cuantizadas y electromagnetismo análogo de 2+1D.** Con g < 0, el fondo uniforme de la banda simétrica es un superfluido: ξ = 8,3 celdas y c_s = 0,078 celdas/paso. Medido aquí:
- El winding de las plaquetas es entero (error 2·10⁻¹⁶) y la carga total vale 0 siempre.
- Un par se traslada a v = 1/(m\*d), con v·m\*·d̄ = 1,085, 1,100 y 1,079 a 32, 40 y 48 celdas (referencia 1,08–1,10). Es la deriva E×B de la dualidad partícula-vórtice: las cargas no caen una sobre otra.
- Los pares se aniquilan en 12, 44 y 154 pasos con d = 6, 10 y 16 (referencia 15, 45 y 150) y dejan sonido.
- El sonido sigue a Bogoliubov con un error ≤ 0,4 % para k = 0,05–0,25: lineal con c_s para k ≪ 1/ξ y cuadrático para k ≳ 1/ξ. Es la idea del vacío superfluido (Volovik, 2003): una simetría "relativista" con c_s en lugar de c que se rompe a la escala del núcleo.

La ventana de validez: el fondo repulsivo hierve hacia el paso ~4500 por una resonancia paramétrica de Floquet (§2.7). El HUD cuenta los vórtices espontáneos y avisa.

**Escenario Dirac: espín ½ relativista.** El paso de D'Ariano y Perinotti tiene el hamiltoniano efectivo h = σ_z k_x + σ_x k_y + m σ_y: Dirac en 2+1D con c = 1 celda/paso. Medido aquí:
- ω(k) = √(m² + k²) con error ≤ 0,15 % para k ≤ 0,3.
- Fase de Berry: π con m = 0, y 1,757 frente a π(1 − m/E) = 1,737 con m = 0,1.
- Velocidad de grupo máxima: ≤ 1 en toda la zona de Brillouin.
- Velocidades de 0,694, 0,946 y 0,985 con k = 0,1, 0,3 y 0,6.
- Zitterbewegung con frecuencia 0,202 (2m = 0,2) y amplitud 4,55 (1/2m = 5).

Es el espín ½ de **una** partícula: la banda negativa solo es antimateria al cuantizar el campo.

### (s), la dispersión exacta del Dirac 2D y su deriva transversal

El paso tiene cos ω = cos m·cos k_x·cos k_y + sin m·sin k_x·sin k_y (comprobado a 10⁻⁶). El segundo término no es isótropo:
- La velocidad de grupo no es paralela a k: con m = 0,2 y k = (0,3, 0), v_y = −0,1672.
- Con m > π/4, k = 0 deja de ser el mínimo de la banda, que pasa a ±(π/2, π/2) con valor π/2 − m.

**Una deriva que la especificación no menciona.** Un paquete ↑ finito en reposo también deriva, a ⟨k²/E²⟩ ≈ 1/(2σ²m²). Su parte de energía positiva pesa más en k > 0, la negativa en k < 0, y las dos avanzan hacia +x. Con σ = 40 y m = 0,1 la predicción es 0,031 celdas/paso y se mide 0,028. El experimento 18 y el test 33 ajustan la deriva junto con el temblor, y por eso la amplitud sale 4,55 en vez de los 4,77 de la referencia.

## Fase 4: electromagnetismo (§2.9, §2.10, §3.6 y §3.7)

### (j) El modo electromagnetismo

**Por qué la ley de Gauss y la invariancia gauge son exactas en la red.** El campo es la teoría gauge U(1) de Wilson en su forma hamiltoniana (Kogut–Susskind). A vive en los enlaces y la materia salta con las fases de Peierls e^{±iqA}.
- **Invariancia gauge.** La transformación ψ → e^{iqχ}ψ, A_a → A_a + χ(r + â) − χ(r) deja intacto cada salto, porque la fase que gana la amplitud al llegar compensa la de la celda de partida. Es una identidad de la red, no una aproximación.
- **Ley de Gauss.** La corriente J_a(r) = |ψ_{+a}(r)|² − |ψ_{−a}(r + â)|² se toma tras la moneda y antes del salto, así que la continuidad discreta es exacta. La actualización E −= q·J cambia div E exactamente en q·Δρ, y el rotacional del campo magnético no tiene divergencia.

Por eso div E − q(ρ − ⟨ρ⟩) se conserva al redondeo:
- La materia se guarda en f32 (§4.2): el residuo queda en ~5·10⁻⁷ tras 1000–2000 pasos (la referencia, en f64, daba 10⁻¹⁵).
- Una transformación gauge aleatoria a mitad de la simulación deja las densidades iguales a 4·10⁻⁷.

**El tope con U(1).** El tope (§2.5) mueve norma entre celdas sin pasar por los enlaces, así que cada reparto se anota como corriente en el enlace por el que pasa (E_a −= q·J_a).
- Esa contabilidad solo es exacta si el reparto es exacto. En f32, las transferencias pequeñas se pierden o se duplican al redondear, y el montón de arena llegaba al corte de seguridad (200 001 iteraciones).
- Ahora el reparto se hace en f64, sobre una copia sombra de las celdas tocadas, y converge (por ejemplo, en 5739 iteraciones tras una gota de norma 400 con E_cap = 6).
- Con la contabilidad, el residuo de Gauss queda en 5·10⁻⁷ tras 2000 pasos con el tope actuando en el 99 % de ellos; sin ella pasa de 0,1.

**El efecto Aharonov–Bohm.** El haz rodea un muro que encierra un tubo de flujo: la onda nunca toca el campo magnético. Aun así, la intensidad en el centro de la pantalla relativa a los lados es 1,16 con Φ = 0, 0,51 con π/2 y 0,084 con π (referencia 1,19, 0,51 y 0,09). Lo físico es el flujo encerrado, módulo 2π/q; la cuerda de Dirac que une el tubo con su compensador es invisible.

**Medidas en 2D:**
- Ciclotrón: ω_c al 98,1 % de q·B/m\*.
- Fotón: 0,4993 frente a √κ = 0,5.
- Coulomb: los solitones con la misma carga se repelen, con a·d constante (la ley 1/d de 2D).

**Qué cambia en 3D:**
- El fotón tiene dos polarizaciones transversales, las dos a 0,4993.
- El modo longitudinal es gauge puro (B = 0): tras 200 pasos, E = 0 y A no se ha movido.
- La estabilidad exige κ ≤ s²/dim (⅓ con s = 1), porque el autovalor máximo del rotacional-rotacional es 4·dim. El código limita κ a ese valor.
- La fuerza de Coulomb frente a una carga externa fija cae como 1/d² (test 47).
- Magnetostática: dos corrientes antiparalelas, con el campo relajado amortiguando E dentro de cada subpaso, dan B igual a la magnetostática exacta de la red (κ·rot B = J) con un error relativo < 10⁻³.
- La especificación dice que amortiguar fuera de los subpasos diverge. Con este integrador, amortiguar una vez por paso tras los subpasos no diverge; el test comprueba que el código amortigua dentro.

**Por qué es electromagnetismo clásico y no QED.** Es el campo de Maxwell clásico acoplado a una onda de materia de campo medio:
- No hay fotones cuantizados ni creación de pares: la norma se conserva.
- No hay correcciones radiativas.
- En el escenario materia, la luz viaja a c_γ = 0,5·c: cada especie puede tener su propia velocidad límite (Chadha y Nielsen, 1983).

### (k) La electrodinámica clásica del espín ½

**Por qué c_γ = c.** El escenario Dirac usa κ = 1 con s = 2 subpasos: el fotón comparte el cono de luz de la materia. Con c_γ = 0,5, un paquete con k = 0,3 (v = 0,95) va más rápido que sus propios fotones y emite un cono de Cherenkov (experimento 69):
- El máximo de la energía del campo está a 147,5° de la velocidad, frente a los 148,2° del ángulo de Mach.
- Detrás de la carga hay 24 veces más energía que en el control con c_γ = 1. Delante hay la misma, porque en ningún caso llega nada.
- La razón detrás/delante del control no es la de la referencia (5,9 frente a 0,90). La carga arranca de golpe y su campo solo existe dentro del círculo de luz del punto de partida: para llegar a 60 celdas por delante de una carga con v = 0,95 hacen falta 60/(c − v) = 1200 pasos.

**Los niveles de Landau delatan g = 2.** El espectro por autocorrelación en un B uniforme da ±√(m² + 2nqB) con un error ≤ 1,5 %: 0,2727 y −0,2742 frente a ±0,2747; 0,3720 y −0,3739 frente a ±0,3754; 0,4498 y −0,4514 frente a ±0,4543. El nivel n = 0 aparece **solo en +m** (0,0993), sin compañero en −m: el punto cero se cancela con la energía de Zeeman, que es la huella de g = 2, como en el grafeno.

La caminata escalar en el mismo campo da niveles equiespaciados con el ½ del punto cero: 0,5154, 0,5446 y 0,5729 frente a 0,5150, 0,5449 y 0,5749, con (E₀ − m)/(E₁ − E₀) = 0,529.

**Landau–Zener es el exponente de Schwinger, pero aquí no hay pares.** Un estado de la banda −1 empujado a través de k = 0 pasa a la banda +1 con probabilidad exp(−π·m²/(q·E)): 0,0019, 0,0439, 0,2107 y 0,4489 frente a 0,0019, 0,0432, 0,2079 y 0,4559 en espacio real, los mismos valores que la referencia. Es el exponente de Schwinger por modo. Pero la norma se conserva: es una partícula que cambia de banda, y solo sería un par al cuantizar el campo (el mar de Dirac). Eso se hace en el laboratorio de pares (§2.23).

**Más relatividad en la red:**
- Fuerza constante: x(t) sigue la hipérbola relativista con un error máximo del 2,9 % (Newton daría 3 veces más) y la velocidad del centroide se queda en 0,978 < c.
- Ciclotrón relativista: ω_c·E/(qB) = 0,91–0,94.
- Paradoja de Klein: transmisión 0,978, 0, 0,834 y 0,916 con V₀ = 0,1, 0,25, 0,5 y 0,8 (fórmula del escalón de Dirac: 0,979, 0, 0,838 y 0,920).
- Dilatación temporal: el reloj que viaja con el paquete marca ω − k·v de la red, con un 0,2–0,4 % de más por la dispersión en k de un paquete finito.

## Fase 5: Yang–Mills y quarks 2D (§2.11, §2.14, §3.8 y §3.11)

### (l) Yang–Mills clásico

**Cómo se conserva la ley de Gauss no abeliana.** Los enlaces son matrices U_a(r) ∈ SU(N) y el campo eléctrico Ẽ_a(r) vive en el álgebra, en el marco de color de r. La ley de Gauss es covariante: Σ_a [Ẽ_a(r) − Ad(U_a(r − â)†)·Ẽ_a(r − â)] = q·ρ^c(r), con ρ^c = ψ†T^cψ.
- **La fuerza de Wilson no cambia la ley de Gauss.** La fuerza, 2κ·Im Tr(T^c(P + Q)), es el gradiente de una función invariante gauge, W(U). Su divergencia covariante se anula por la identidad de Noether de esa invariancia.
- **Los enlaces giran con el campo.** U ← exp(−iΔt·Ẽ)·U conserva la divergencia, porque Ẽ conmuta con su propia exponencial.
- **La materia entra por su corriente.** El salto de la materia con color es un transporte paralelo, U·χ, así que su carga cumple una continuidad covariante exacta. La actualización Ẽ −= q²·J la compensa.
- **El resultado.** La deriva de G queda en el redondeo de la materia en f32: < 10⁻⁴ en los tests 48 y 72, frente a 8·10⁻¹⁴ de la referencia en f64. Una transformación gauge aleatoria de amplitud 3 deja las densidades iguales a menos de 10⁻⁴.
- **La fuerza es correcta.** Frente a −∂W/∂θ por diferencias finitas, el error relativo es < 10⁻⁵ (test 49).

**Los gluones interactúan entre sí (experimento 32).** Dos pulsos que se cruzan, con amplitud a:
- **Colores que no conmutan.** En F_ab aparece el término −i[A_a, A_b], que acopla los dos pulsos a primer orden. La desviación de la superposición es ∝ a (pendiente 1,00): 1,79·10⁻³ en SU(2) (T¹ y T²) y 0,90·10⁻³ en SU(3) (T¹ y T⁴) con a = 0,001.
- **La razón entre grupos.** SU(3)/SU(2) = 0,500, que es la razón de las constantes de estructura, f₁₄₇/f₁₂₃ = ½.
- **Colores que conmutan.** Queda solo el efecto de la compacidad de exp(−iθT), ∝ a² (pendiente 2,00): 5,6·10⁻⁹ en SU(2) y 1,9·10⁻⁹ en SU(3). Es cinco órdenes de magnitud menos.
- **U(1).** Los fotones se cruzan sin verse: 1,7·10⁻¹³, el redondeo.
- **La velocidad.** Los gluones de cada color viajan a √κ = 0,5 (test 50).

**Los factores de color (experimento 33).** Un quark rojo estático y un solitón de prueba a 30 celdas, restando el control con q = 0. La fuerza es la de Coulomb con el producto de las cargas de color, w₁·w₂:
- **Rojo con rojo** se repele, con el 92,5 % de w_r·Ẽ/m\*.
- **Verde y azul** se atraen, idénticos entre sí, con el 95,2 %. La razón verde/rojo es −0,515, frente a −½ (referencia −0,516).
- **SU(2).** ↓/↑ = −1,030, frente a −1.
- **El color del solitón no cambia,** porque estas cargas están en la subálgebra de Cartan.

**El caos y por qué su escala depende de la dimensión (experimento 34).** Se mide la distancia invariante entre plaquetas de dos copias, una perturbada a 10⁻⁹.
- **SU(2) y SU(3).** Crece de forma exponencial: λ > 0.
- **U(1).** No crece, porque Maxwell es lineal.
- **La escala de λ.** La única escala de la teoría clásica es g²T, con T ∝ ε (la energía por enlace).
  - En 2+1D, g² tiene dimensiones de masa, así que g²T tiene dimensiones de masa², y λ ∝ √ε. Se mide un exponente de 0,516 en SU(2) y de 0,514 en SU(3) (referencia 0,53 y 0,51).
  - En 3+1D, g²T es una masa y λ ∝ ε, la ley de Müller y Trayanov. En 24³, entre ε = 0,089 y 0,46, se mide 1,08 (referencia 1,17 entre 0,088 y 0,55).
  - Con energías mayores, λ satura, porque el grupo es compacto y la energía por enlace está acotada.

**El apantallamiento (experimento 36).** Dos cargas en un baño de gluones con ⟨|Ẽ|²⟩ = 0,074 por enlace:
- **SU(2).** El exceso de energía cerca de la carga, descontado el nivel lejano, es el 2,0 % del Coulomb desnudo (referencia 0,6–1,6 %). Los gluones tienen color y apantallan la carga (Debye), y la energía de su campo se reparte como calor uniforme.
- **U(1) con el mismo baño.** El exceso coincide con Coulomb (1,0000, 0,9997 y 1,0006 en tres capas), porque los fotones no tienen carga.
- **Lo que no se mide.** La longitud de apantallamiento es menor que la resolución (~3 celdas), así que la ley m_D² ∝ g²T no se ha comprobado.

**Por qué hacen falta subpasos.** El leapfrog no lineal tiene un error de energía ∝ (Δt·ω_max)². Un baño excita los modos de la red hasta el corte, y con s = 1 la energía deriva (+11 % en la referencia), otro calentamiento de Floquet. Con s = 2 la energía se conserva a < 10⁻³ en 700 pasos (test 53).
- La energía que conserva el leapfrog es ½Σ Ẽ(t − ½)·Ẽ(t + ½) + W, no ½Σ Ẽ² + W: esta última oscila a O(Δt²). El HUD y los tests usan la primera.

**Por qué la teoría clásica no confina (experimento 35).** El par estático ±10 en T³ tiene el campo de Coulomb.
- **En 2D, la energía crece ∝ ln d.** Los incrementos al duplicar d son 10,97, 10,74 y 9,81, frente a Q²·ln 2/(2π) − Q²·Δ(d²)/(4·Área) = 10,96, 10,74 y 9,86. El segundo término es el fondo neutralizador del toro, y explica el 9,8 de la referencia.
- **En 3D (48³), la energía satura.** U(d) = 3,600, 4,065 y 4,153 con d = 8, 16 y 24. Los incrementos, 0,465 y 0,088, coinciden con el Coulomb del toro por la suma de Ewald (0,462 y 0,088). En el espacio infinito serían 0,497 y 0,166.
- **El campo se ensancha con la distancia.** Su anchura en el plano medio es 0,63–0,75·d en 2D y 0,50·d en 3D.
- **La solución no cambia con la dinámica.** Bajo la dinámica no abeliana completa se mueve 2·10⁻¹⁵ en 100 pasos: las cargas están en la subálgebra de Cartan y la solución es abeliana.
- **Qué sería confinar.** Un tubo de flujo de anchura fija, con energía ∝ d, y gluones con masa. Ambos son efectos cuánticos no perturbativos: la ley del área del lazo de Wilson aparece en la teoría cuántica de la red (desarrollo de acoplamiento fuerte, vórtices del centro), no en la ecuación de movimiento clásica.

**El tope no se combina con SU(N)** (parte de (s)). Con U(1), cada reparto del tope se anota como una corriente escalar en el enlace (§2.5). Con color, el reparto reescala ψ en cada celda:
- La carga ρ^c que pierde la celda donante tiene la dirección de color de esa celda.
- La que gana la receptora tiene la dirección de color de la receptora.
- En general, ninguna corriente de enlace transportada con U las compensa. Por eso el tope se desactiva con SU(N).

### (o), los quarks de Dirac y el Aharonov–Bohm no abeliano

**Los quarks.** La caminata de Dirac 2D con enlaces SU(N) es un espinor de dos componentes ⊗ un vector de color, con κ = 1 y s = 2 (la luz y los gluones comparten el cono de luz). La ley de Gauss y la invariancia gauge se cumplen al redondeo de f32 y la norma se conserva (test 72).
- No es QCD: es una sola onda clásica, sin estadística de Fermi, sin pares del mar y sin confinamiento.

**El Aharonov–Bohm no abeliano (experimento 48).** El haz rodea un bloque absorbente que tapa un tubo de color con holonomía h. La razón centro/lados en la pantalla es:
- **Con h = 1:** 1,453 para ↑, ↓ y (↑+↓)/√2.
- **Con h = −1:** 0,037 para los tres. h = −1 es el centro de SU(2): conmuta con todo y actúa como la misma fase π sobre cualquier color, así que la interferencia destructiva es ciega al color.
- **Con h = iσ₃ = diag(i, −i):** 0,521 (↑) y 0,477 (↓). h da a cada color una fase distinta, ±π/2, y la asimetría entre los dos viene de la caminata, que no es simétrica ante y → −y (§2.8).
- **La superposición.** (↑+↓)/√2 da exactamente la suma incoherente de las dos (0,4980 frente a 0,4980): con h diagonal, cada color sigue su camino y la densidad no tiene términos cruzados.
- **Frente a la referencia.** Los valores absolutos difieren (0,736, 0,184 y 0,336/0,281). Aquí el haz sale con k_y = atan(tan m·tan k_x), que anula la deriva transversal y lo apunta recto al bloque. Sin esa compensación, h = −1 da 0,174, como en la referencia, pero ↑ y ↓ con iσ₃ salen muy asimétricos (0,13 y 1,13). Las razones relativas que pide el test 73 se cumplen en los dos casos.

**La trayectoria de Wong (experimento 70).** El montaje es un quark SU(2) con m = 0,2 y k = 0,3, y enlaces U_x = exp(−i·0,05·T¹) y U_y = exp(−i·b·T²). Hay un campo de color uniforme, F = a·b·T³ (flujo 0,001 y 0,002 por plaqueta).
- **El modelo, sin parámetros ajustados.** Son 20 000 partículas clásicas de la función de Wigner del paquete:
  - se mueven con la velocidad de grupo de la banda +1;
  - su color precesa con el transporte paralelo (a·Δx alrededor de T¹ y b·Δy alrededor de T²);
  - reciben la fuerza de Lorentz de color sobre la carga T³, con el signo calibrado aparte con un campo T³ abeliano en el gauge de Landau.
- **El acuerdo.** La parte que cambia de signo con el color, (y↑ − y↓)/2, sigue al modelo con un residuo rms del 12,1 % (b = 0,02) y del 10,2 % (b = 0,04), frente al 12 % y el 18 % de la referencia.
  - Oscila con la precesión del color: mínimos de −1,26 y −1,12 celdas con b = 0,02, frente a −1,23 y −1,00 del modelo (referencia −1,25 y −1,17).
  - Se duplica con b: −2,43 con b = 0,04 (referencia −2,38).
- **Por qué el modelo usa solo F³.** En el continuo, F_xy = −i[A_x, A_y] es exactamente a·b·T³. La plaqueta de la red tiene además componentes T¹ y T² de orden a²b (−4·10⁻⁵ y 5·10⁻⁵ con b = 0,04), que son un artefacto de medirla en el sistema de color de una esquina.
  - Esas componentes son pequeñas, pero empujan de forma secular, así que importan: con un convenio de esquina, el residuo baja al 5 %; con el otro, sube al 30 %.
  - El modelo de la especificación es la fuerza sobre la carga T³, y es el que se usa.
- **La deriva común.** La parte común (la deriva de §2.8) coincide con el modelo: −46,8 frente a −48,2 celdas en 300 pasos. La referencia da −57,2, con un paquete cuya anchura no especifica; σ = 12 reproduce los valores del test 114.

## Fase 6: Higgs, paredes de dominio y Yukawa (§2.12, §2.13, §3.9, §3.10 y §3.16)

### (m) El mecanismo de Higgs

**Por qué el condensado de la caminata no sirve.** Cargar el fondo uniforme del escenario de vórtices es, a escalas largas, el modelo de Higgs no relativista de los superconductores, y funciona (experimento 37):
- **El Goldstone desaparece.** Con q = 0,05, una onda de densidad oscila a 0,0339, la frecuencia de plasma ω_p = q·√(ρ̄/m\*) = 0,0338. Sin carga es sonido: 0,00176, frente a c_s·k = 0,00192.

**Pero el condensado cargado es inestable.** El rms de E crece con γ = 0,0170 = 0,502·ω_p (referencia 0,51), y sin carga no crece.
- **Es una resonancia paramétrica con los fotones de la red** (§3.16). El condensado (ε₀) se desintegra en un fotón transversal y en materia de otra banda, con ε₀ − ε_b(−k) = ω_fotón(k) módulo 2π.
- **El modo que crece cumple esa condición.** Con fotones lentos (κ = 0,05), en k = (0,982, 0,589) oscila a ω = 0,2554, frente al fotón de red en ese k, 0,2482. Es transversal (2,5 % longitudinal) y el desajuste de la resonancia es 0,0061 (referencia 0,006).
- **En el continuo ese canal está cerrado.** En la red se abre porque las dispersiones están acotadas y se cruzan.
- **Grover no se puede cerrar.** Dos de sus bandas recorren casi todo el círculo de cuasi-energía, así que la condición tiene solución para cualquier velocidad del fotón. Con κ = 0,05 la tasa sigue en 0,614·ω_p.
- **Dirac sí se puede cerrar,** si el fotón máximo de la red queda por debajo de 2m, al precio de fotones lentos. Es del Dirac 3D (§2.16) y se comprueba en la fase 7.
- **Por eso el Higgs es un campo escalar aparte,** como en el modelo estándar.

**El Higgs escalar.** Es un leapfrog sobre los mismos enlaces de §2.9 y §2.11, con sus leyes de Gauss exactas:
- **Abeliano:** la deriva de div E + 2q·Im(φ*π) es < 10⁻¹² en 1000 pasos (48² y 16³). La copia con una transformación gauge aleatoria coincide en |φ| a < 10⁻⁹.
- **Doblete:** las dos derivas son 5·10⁻¹⁵ y 2·10⁻¹⁴ (32²) y 8·10⁻¹⁵ y 1·10⁻¹⁴ (12³).
- **El fotón absorbe al Goldstone.** En 256×4 con q = 0,1, el fotón oscila a 0,07178 y el Higgs a 0,14207, los valores de la dispersión del leapfrog con m_γ² = 2c²q²v² y m_H² = 4λv², a las cinco cifras. La onda de fase oscila a la frecuencia del fotón masivo, porque se ha convertido en su polarización longitudinal. Sin carga, es un Goldstone sin masa (0,01227 = c·k).

**Meissner y cuantización del flujo (experimento 39).** Un cuanto de flujo repartido uniformemente acaba en un vórtice, con 0,99932·2π/q a su alrededor.
- **El decaimiento de B.** Fuera del núcleo (15 < r < 40), B decae como K₀(r/λ) con λ = 7,87, frente a λ_L = 7,07 (referencia 7,35–7,78).
- **Por qué hace falta la energía compacta.** Con la no compacta, el flujo total de cada plano es exactamente cero, porque B[A] es una diferencia.

**Bogomolny, tipo I y tipo II (experimento 40).** La energía de un vórtice, en unidades de 2πc²v², es:
- **Punto crítico:** 0,9995, la cota de Bogomolny.
- **Tipo I (m_H/m_γ = 0,5):** 0,7511.
- **Tipo II (2):** 1,3393.

Con 4 cuantos, la energía por cuanto muestra la fuerza entre vórtices, con los valores de la referencia a cuatro cifras:
- **Bogomolny:** 0,9996, igual que aislado: no hay fuerza.
- **Tipo II:** 1,3397, más que aislado: repulsión.
- **Tipo I:** 0,6432, el 85,6 % del aislado: atracción.

**La energía de un par fijado frente a d (experimento 64).** E(4), E(6) y E(8) menos E(16):
- **Tipo II:** +0,2027, +0,1146 y +0,0536 (referencia +0,200, +0,114 y +0,054).
- **Tipo I:** −0,059, −0,055 y −0,046.
- **Punto crítico:** plana a 0,0014, 0,0011 y 0,0006 de la energía de un vórtice, aunque los núcleos se solapen.

**La red de Abrikosov (experimento 63).** 24 cuantos en tipo II forman 24 vórtices con |ψ₆| = 0,951 y vecinos a 21,14 ± 1,05, la red triangular (referencia 0,951 y 21,1 ± 1,1). En tipo I no hay red (|ψ₆| = 0,35).

**Dispersión a 90° (experimento 78).**
- **De frente,** dos vórtices críticos salen a 90,0°, con v = 0,1 y con v = 0,2.
- **Con parámetro de impacto** b = 4, 8 y 12: 52,3°, 22,0° y 6,0° (referencia 53,1°, 20,6° y 5,9°).
- **El ángulo no depende de la velocidad:** con b = 8 y v = 0,2 sale 20,3°. Es movimiento geodésico en el espacio de moduli (Manton, 1982). El espacio de dos vórtices idénticos es un cono, y la geodésica que pasa por el vértice gira 90°.
- **En tipo II rebotan** a 180°.

**El ángulo de Weinberg (experimento 42).** Con g = 0,3 y g′ = 0,18:
- **Las masas.** W 0,10682, Z 0,12438, Higgs 0,15031 y el fotón sin masa (0,01227 = c·k), a las cinco cifras de la predicción. M_W/M_Z = cos θ_W = 0,8575.
- **La mezcla.** Una onda solo de B pulsa con el fotón y el Z, con amplitudes de E en razón 0,2737, frente a cot²θ_W·ω_γ/ω_Z = 0,2741. Una de W³ lo hace con 0,0355, frente a 0,0355.

**Confinamiento de monopolos (experimento 41 y test 61).** Dos monopolos externos ±2π/q en 48³, con q = 0,2 en el punto crítico:
- **En la fase de Higgs:** E = 70,88, 83,96 y 96,55 con d = 8, 16 y 24 (la referencia, a cuatro cifras). Crece linealmente, con una pendiente de 1,5745, frente a la tensión del vórtice 2D medida, 1,5679.
- **Sin Higgs:** satura como Coulomb, 59,94, 61,30 y 61,88.
- **Qué es y qué no.** Es confinamiento lineal clásico de cargas *magnéticas*. La conjetura de 't Hooft y Mandelstam es que QCD confina el color con la versión dual, y eso es cuántico. Aquí las cargas de color de §2.11 no se confinan. Con un Higgs fundamental en la teoría cuántica de red, las fases de Higgs y de confinamiento están conectadas (Fradkin y Shenker, 1979), pero en el nivel clásico no hay nada parecido que comprobar.
- **Qué queda fuera.** El origen de v, λ y los acoplamientos (se ponen a mano) y la teoría gauge quiral de la fuerza débil, que acopla los W solo a los fermiones levógiros.

### (n) Fermiones quirales en paredes y masa de Yukawa

**Una sola quiralidad por pared.** Donde la masa de Dirac cambia de signo, la ecuación tiene un modo e^{−∫m}, normalizable para un solo autovalor de σ_x, con ω = s·k_y (Jackiw–Rebbi). El espectro exacto del paso en una tira de 96 celdas (test 64 y experimento 43) da:
- **Un estado en el hueco** por k_y y por pared.
- **Su velocidad:** 0,995, 0,955 y 0,875 con m = 0,1, 0,3 y 0,5, frente a cos m = 0,995, 0,955 y 0,878. Las dos paredes van en sentidos opuestos.
- **Su decaimiento:** 0,1002, 0,3044 y 0,5193, frente a atanh(sin m) = 0,1002, 0,3046 y 0,5222.
- **El control.** Un canal sin cambio de signo solo tiene estados con hueco (±0,117).

**Por qué no contradice a Nielsen–Ninomiya.** El teorema impide un fermión quiral solo, no separar las dos quiralidades en el espacio. La quiralidad opuesta vive en la otra pared, y cada rama tiene además una copia de la misma quiralidad en la cuasi-energía π (las dos subredes de la caminata).

**La propagación y la protección.**
- **Un solo sentido.** Un paquete en el modo exacto viaja a 0,955, con el 99 % en la pared y la anchura de 5,66 a 5,74 en 160 pasos.
- **Sin retrodispersión.** Con un obstáculo de potencial, lo reflejado es 8·10⁻⁷ y lo transmitido 0,996: no hay ningún modo que vaya hacia atrás. Un canal no quiral, con el mismo obstáculo, refleja el 61 %. Es la protección de los bordes Hall y de los aislantes topológicos.
- **El borde Hall.** En un B uniforme, el paquete de la pared avanza a 0,946, los del volumen solo giran (deriva 0,0018) y el nivel n = 0 está en +0,2991 en un lado y en −0,2991 en el otro.

**Paredes cerradas (experimento 73).** El modo quiral de un disco se cuantiza en niveles ±0,0165, ±0,0499, ±0,0828 y ±0,1162, es decir (j + ½)·Δ con Δ = 0,0334: semienteros, la fase de Berry π del espín del modo al dar la vuelta.
- **No depende de la forma.** Lo mismo pasa en un cuadrado (Δ = 0,0261) y en una estrella irregular (Δ = 0,0257).
- **El paquete da la vuelta** en un solo sentido en 191, 243 y 247 pasos, sin perder norma al doblar las esquinas.

**La masa de Yukawa.** Con m = y·φ y un Higgs real:
- **En el vacío,** el hueco del fermión es 0,3000 = y·v.
- **El kink relajado** tiene anchura 6,653, frente a 2c/m_H = 6,667, y atrapa una rama quiral de velocidad ±0,9909.
- **Con la reacción del fermión sobre el Higgs,** el paquete viaja a −0,9908 a lo largo del kink, el 97 % sigue ligado y el kink se deforma 5·10⁻⁷.
- **Un kink en movimiento** (u = 0,2, medido 0,1997) arrastra al fermión, que va a (0,199, 0,974) frente a (0,2, v′/γ = 0,971): la composición relativista de velocidades.

**El límite de la red (experimento 46).** Un campo eléctrico a lo largo de la pared empuja k_y por la rama. El paquete sigue en la pared (0,99) mientras q·E·t < 1,5, pasa al volumen (0,22 con q·E·t = 3) y vuelve (0,99) al completar la vuelta de la zona. En el continuo la rama no se acaba.

**Qué es cuántico y aquí no está:** la carga fraccionaria ½ de Jackiw–Rebbi, la anomalía y su flujo desde el volumen. Necesitan el mar de Dirac lleno.

### (o), el monopolo de 't Hooft–Polyakov

**Por qué su carga está cuantizada.** El erizo φ = v·r̂ define un mapa de la esfera del infinito en la esfera del vacío, |φ| = v. Su grado es un entero, π₂(S²) = ℤ, y no cambia con deformaciones continuas: el monopolo es un solitón topológico, no una carga externa.
- **La carga magnética es la del grado:** 4π/g por unidad.
- **Se mide con la proyección abeliana de 't Hooft,** el ángulo de cada plaqueta alrededor de φ̂. Sale 0,9991·4π/g en un cubo de semilado 8 y 0,99998 en uno de 12 (test 78).

**La masa frente a la cota de Bogomolny.** La energía cumple E ≥ κ·∫B·DΦ = 4πκv/g, y la igualdad se alcanza en el límite BPS (λ = 0), con la solución de Prasad–Sommerfield.
- **Con m_H/m_W = 1,265,** la masa (energía en r < R más la cola magnética 2πκ/(g²R)) es 1,2584 con R = 8 y 1,2607 con R = 12, frente a f(1,265) = 1,272 de Forgács et al. (−0,9 %).
- **En el límite BPS,** con el borde exacto, la masa sale 1,0017, 1,0045, 1,0009 y 0,9994 × 4πκv/g con R = 6, 8, 10 y 16, y |DΦ|²/|B|² = 0,994 en 2 < r < 8 (las ecuaciones de Bogomolny).
- **Pares monopolo-antimonopolo.** Se atraen y se aniquilan en el paso 46 (λ = 0, donde el Higgs sin masa dobla la atracción) y en el 104 (λ = 0,5). Sesenta pasos después, el 87 % de la energía está fuera de r = 16: radiación. La aceleración relativa en los 40 primeros pasos es −0,0032 (λ = 0,5) y −0,0132 (λ = 0), frente a −0,0017 y −0,0050 de la fuerza entre cargas puntuales. El signo es el de la atracción y el Higgs sin masa la multiplica (×4,1 aquí, ×2,9 la predicción). El valor absoluto, en cambio, no sale: a la separación inicial los núcleos se solapan y el ansatz de Nambu no es el par relajado. La fuerza de Coulomb magnética es la asintótica.

## Fase 7: Dirac 3D, Weyl, cuerda de Higgs y métricas (§2.15–§2.20 y §3.12–§3.18)

### (o), la gravedad de Nordström

**Por qué basta una masa que dependa de la posición.** La ecuación de Dirac sin masa es invariante conforme: tras reescalar el espinor, Dirac en g = Ω²·η es Dirac plano con masa m·Ω(r) (Arrighi, Facchini y Forets, 2016). La masa por celda de las paredes de §2.13 ya es, por tanto, un espacio-tiempo curvo de esa familia. Con Ω = 1 + Φ es la gravedad escalar de Nordström (1913).

**Lo que da (experimentos 49 y 50, tests 74–76).** En una tira de 1024 celdas con Φ = −0,05·cos(2πx/1024):
- **Principio de equivalencia.** Paquetes en reposo con m₀ = 0,1, 0,2 y 0,3 caen con −2,872, −2,908 y −2,866 (×10⁻⁴), el 0,94 de −∇Φ los tres (referencia −2,90·10⁻⁴, el 94 %). La aceleración no depende de la masa, porque la fuerza, −m₀·∂ω/∂m·∇Φ, y la inercia, ∂²ω/∂k², escalan igual.
- **Corrimiento al rojo.** Los relojes de m₀ = 0,2 en Φ = −0,05, 0 y +0,05 marchan a 0,19004, 0,20004 y 0,21004: las diferencias son m₀·ΔΦ = 0,01000 exactas. El +0,00004 común es la energía cinética del paquete.
- **Una partícula rápida pesa menos.** Ondas planas con k_y = 0, 0,196 y 0,393 caen con −2,908, −1,375 y −0,442 (×10⁻⁴): la razón es 0,473 y 0,152 (referencia 0,474 y 0,152), frente a (m/E)² = 0,513 y 0,208. La dispersión de la red la reduce más a k grande.

**Por qué la observación la descarta.** En la relatividad general la aceleración transversal de una partícula rápida crece, (1 + v²)·∇Φ, y la luz se desvía el doble que en Newton. En Nordström la luz (m₀ = 0) no siente Φ: no se desvía. El eclipse de 1919 midió la desviación de la relatividad general. Para eso hace falta la parte espacial de la métrica, que no es conformemente plana: es el escenario luz (experimento 62, más abajo).

### (p) Schwarzschild en 1+1D y Dirac en 3D

**Schwarzschild en 1+1D (experimento 52, tests 81 y 82).** En dos dimensiones toda métrica es localmente conformemente plana, así que una tira con masa m₀·Ω(x*) es la ecuación de Dirac en cualquier métrica 1+1D, en coordenadas conformes. Con la parte radial de Schwarzschild en coordenada tortuga, x* = r + 2M·ln(r/2M − 1) y Ω² = 1 − 2M/r (M = 40, m₀ = 0,3):
- **Relojes estáticos.** En r/2M = 2, 4 y 10 marchan a 0,21236, 0,26022 y 0,28512, frente a m₀·√(1 − 2M/r) = 0,21213, 0,25981 y 0,28460 (+0,11 %, +0,16 % y +0,18 %; la referencia da 0,21236, 0,26023 y 0,28513). Junto al horizonte (r/2M = 1,25), 0,13550 frente a 0,13416 (1 %): Ω cambia mucho a lo ancho del paquete.
- **La caída.** Un paquete soltado en reposo en r₀ = 240 sigue la geodésica d(x*)/dt = −√(1 − Ω²/Ω₀²) con un error de 1,2 celdas mientras r > 160 y del 3,6 % del recorrido después (marea y dispersión del paquete). Al final se mueve a −0,998 en x* y r = 80,01: se acerca al horizonte a la velocidad de la luz sin cruzarlo nunca en tiempo coordenado (la "estrella congelada").
- **Qué falta.** Es solo la parte radial: no hay parte angular y en 1+1D las ecuaciones de Einstein no tienen dinámica. La radiación de Hawking es cuántica.

**Dirac en 3D: por qué hay que alternar el orden (experimento 53, test 83).** El paso partido M·T_c·T_b·T_a con T_a = exp(−iα_a·k_a) no es exp(−i(α·k + mβ)): como las α no conmutan, el error de la descomposición es un término cruzado que actúa como un acoplamiento espín-órbita ficticio.
- **Con el orden fijo,** los dos niveles de espín se separan en la diagonal (0,1369 y 0,1455 con |k| = 0,1, un 6,1 %), y la velocidad de grupo de cada nivel pasa de c a escala de red.
- **Alternando** (x, y, z en los pasos pares y z, y, x en los impares) la descomposición del operador de dos pasos es simétrica: la degeneración de espín vuelve a ser exacta (3·10⁻¹⁶), ω = √(m² + k²) en el eje y en la diagonal al 2 % (0,1413/0,1411, 0,2233/0,2217, 0,3157/0,3100, los valores de la referencia), y la velocidad de grupo máxima en toda la zona es 0,9957.

**Niveles de Landau en 3D.** Con k_z = 0 hay picos en ±0,1001 (n = 0 en +m y en −m) y en ±0,326, ±0,445 y ±0,535 (n = 1, 2 y 3), del 1 % al 3 % por debajo de √(m² + 2n·q·B). En 3D el nivel cero aparece en los dos signos: la asimetría del Dirac 2D, la "anomalía de paridad", es propia de 2+1D.

**g = 2 (test 86).** Un paquete de helicidad +1 con k = 0,3 (γ = 3,2) en un B uniforme: su espín gira a −0,04375 rad/paso y su velocidad (el flujo exacto de los subpasos) a −0,04364, razón 1,002 (referencia 0,987). El coseno entre los dos no baja de 0,942. Que la helicidad se conserve es la firma de g = 2: con γ = 3,2, una desviación del 1 % en g cambiaría esa razón un 3 %.

### (q) Dirac 3D con campos y el fermión de Weyl

**Por qué proyectar sobre la banda (test 87).** Un espinor constante (el del k central) por una envolvente gaussiana tiene una parte ~(Δk/2m)² en la banda −1, que acelera al revés: con m = 0,8 y σ = 3, el 6,37 % (referencia 6,4 %). El operador de dos pasos W cumple S² = s²·1 exactamente, con S = (W − W†)/2i (error 4·10⁻¹⁶), así que el proyector de la banda +1 es P₊ = ½(1 − S/s), sin diagonalizar. Proyectado, el paquete tiene 0 en la banda −1.

**Por qué la masa inercial no es m (test 88).** Un paquete en reposo acelera con a = q·E·⟨∂²ω/∂k_x²⟩, promediada sobre su espectro. Con m = 0,8 y σ = 4 el promedio es 0,820, mientras que la curvatura en k = 0 da 0,971: la respuesta medida es 0,987 del promedio (referencia 0,987).

**Coulomb en 3D y la tercera ley (experimento 54, tests 89 y 95).** Con una carga externa Q = 5 a d = 16, 24 y 32 de un paquete proyectado, en 80³ con Maxwell dinámico:
- **La aceleración** es el 0,927, 0,972 y 0,977 de q·⟨E⟩·⟨∂²ω/∂k²⟩ con ⟨E⟩ de Poisson en el toro (los valores de la referencia), y a·d² = 0,292, 0,272 y 0,189: 1/d², con la corrección del toro en d = 32. El control con q = 0 no se mueve (5·10⁻¹⁹).
- **Dos paquetes dinámicos** de normas 0,7 y 0,3 (q = 2, d = 28) aceleran con el 0,942 y el 0,974 de lo previsto (referencia 0,930 y 0,966), y |0,7·a₁ + 0,3·a₂| es el 3,5 % de |0,7·a₁| (referencia 3,8 %): la tercera ley.
- **Por qué por partes.** Con el campo dado, la caminata es lineal: ψ = ψ₁ + ψ₂ se puede evolucionar como dos partes con las corrientes de la suma, y coincide con la evolución completa a 10⁻¹⁶ (test 94). Así cada paquete tiene su propio centroide. Cortar el espacio por la mitad sesgaría los centroides con las colas que cruzan la línea media.

**Quarks en 3D (experimento 55, tests 90 y 98).** Con SU(2) y SU(3), la ley de Gauss no abeliana se conserva a 10⁻¹⁴ y la copia con una transformación gauge aleatoria coincide a 10⁻¹⁵ en la densidad y en las plaquetas.
- **Balance de energía.** La energía pasa de la materia al campo (hasta el 24 % con SU(2) y el 21 % con SU(3); referencia 16 % y 21 %) y vuelve. La suma se conserva al 0,7 % con la cuasi-energía media −arg⟨ψ|F|ψ⟩ y la energía del campo en unidades de la materia (`em_energy`/q²).
- **Factores de color.** Dos quarks en reposo a d = 16 (48³, q = 2): rojo-rojo se repelen (+2,15·10⁻⁴) y rojo-verde se atraen (−1,06·10⁻⁴), razón −0,494, frente a los factores clásicos t₁·t₂ = 1/3 y −1/6 (referencia +2,08, −1,04 y −0,501). Con colores de la base y U = 1 al empezar, el campo se queda en T³ y T⁸: es la versión abeliana del problema. Las aceleraciones se integran con el flujo exacto: hacia el paso 30, las colas rápidas de los quarks dan la vuelta al toro de 48 y falsean los centroides.

**Plasma de Dirac (experimento 57, tests 96 y 101).** Un estado uniforme de la banda +1 sobre un fondo neutralizante oscila a la frecuencia de Bogoliubov con Coulomb en la red, √(ε_k² + 2ε_k·ρ̄q²/K²): el 0,995 con ω_p = 0,1 en 32 celdas (referencia 0,994), y el 0,988 y el 1,005 con ω_p = 0,05 y 0,1 en 64. Con ω_p = 0,2 en 64 celdas sale el 0,953 (referencia 0,955): la inestabilidad ya actúa. Después el campo crece con una amplitud de 0,44–0,48·ω_p:
- **El modo que crece** tiene k_x = 2,258 (referencia 2,258), junto a la raíz de m + E(k) = ω_fotón(k), 2,293, y oscila a 1,895, frente al fotón de red en ese k, 1,876 (1 %). La banda negativa pasa a tener la mitad de la norma.
- **Es la resonancia ε₀ − ε_b(−k) = ω_fotón(k).** Con fotones tan rápidos como la materia, el fotón de red llega a la banda negativa.

**Weyl: por qué no escapa de Nielsen–Ninomiya (experimento 56, tests 91–93 y 97).**
- **Los nodos.** Con el orden fijo, F = +1 en k = 0 y (π/2, π/2, π/2), y F = −1 en (π, 0, 0) y (π/2, π/2, −π/2), con det(∂a/∂k) = +1, −1, −1 y +1 a 10⁻⁷. En la zona cúbica hay 4 nodos de cada quiralidad en la cuasi-energía 0 y otros tantos en π: cada sector tiene su doblador.
- **W₃.** Para F ∈ SU(2), la quiralidad neta en 0 y en π valen las dos W₃, el grado de F: T³ → S³. Con las tres variantes sale 0,00000 en la zona completa. Un producto de desplazamientos por eje y monedas en SU(2) tiene W₃ = 0, porque un factor que depende de una sola componente de k tiene grado 0. Es el teorema de Nielsen–Ninomiya para redes de Floquet (Sun et al.).
- **Higashikawa et al.** Con los bombeos selectivos de espín, W₃ = 1,00000 en la media zona |k_z| < π/2: un Weyl sin pareja, pero solo en un subespacio que no es local. En la zona completa W₃ = 0, y el compañero (−1) está en (0, 0, π), en la misma cuasi-energía.
- **La rama quiral en un campo B.** En 2D con k_z fijo, el pico más cercano a 0 está en ±0,0989 con k_z = ±0,1 y en ∓0,0989 junto a π/2: pendiente +0,988 junto al nodo y −0,988 junto al doblador (los valores de la referencia). Con los bombeos, +0,1120 y −0,0874 con k_z = ±0,05 y lo contrario junto a π: pendiente ±1,99, porque en z dan medio paso.
- **El paquete quiral.** En 24×24×128 con 4 cuantos, un gaussiano filtrado sobre la rama quiral de k = 0 viaja a +0,9889 y el de su doblador a −0,9889, y ninguno se ensancha (7,085 → 7,085): su dispersión es lineal. Un gaussiano con espín arriba, sin filtrar, deriva a +0,145, que es la fracción de la rama (0,148) por su velocidad; con espín abajo, −0,029. Esa asimetría es el efecto magnético quiral de una partícula; la anomalía en sí necesita el mar lleno.

### (r) Fermiones en una cuerda de Higgs

**Por qué la masa compleja atrapa |n| modos.** Con un Higgs complejo, la masa de Yukawa tiene fase: M = m₀·f(r)·(cos θ·β + sin θ·B₂), con B₂ = τ_y ⊗ 1, que anticonmuta con β y con las tres α (exactamente: el test 99 da 0). En un vórtice de devanado n, en cada corte hay |n| modos de energía cero (Jackiw y Rossi, 1981): el índice del operador transversal.

**Por qué se mueven en un solo sentido (experimento 58, test 100).** α_z anticonmuta con el hamiltoniano transversal, así que los modos cero son autoestados de α_z, y sobre ellos el hamiltoniano es α_z·k_z: se mueven en un solo sentido, fijado por el signo de n (Callan y Harvey, 1985).
- **Con el orden fijo,** en un corte de 24×24 (m₀ = 0,6, ξ = 1,5), el único pico con |ε| < 0,4 está en +0,0851 en el vórtice y en −0,0851 en el antivórtice con k_z = +0,1, y al revés con −0,1 (referencia ±0,0851; autovalores ±0,0853).
- **En 3D,** un paquete sencillo en cada núcleo (el espinor de α_z = ±1, sin calcular el modo) deja el 51 % en el núcleo, que viaja a +0,833 en el vórtice y a −0,833 en el antivórtice (referencia ±0,833). El resto radia al volumen.
- **Con devanado 2,** la rama del vórtice tiene 2,9 veces el peso espectral de la de devanado 1. Las dos ramas salen casi degeneradas (en el continuo tienen la misma velocidad); la referencia las ve separadas (0,0804 y 0,0871).

**Por qué el orden alternado esconde la quiralidad.** Con el operador de dos pasos, cada núcleo tiene ramas de las dos pendientes, que se hibridan con un hueco: estados en ±0,0211 y ±0,0318 dentro del hueco con k_z = 0, de los dos signos (referencia ±0,021 y ±0,032). Es lo mismo que en la caminata de Weyl: el orden alternado arregla la isotropía, pero esconde la quiralidad.

**Qué falta para la cuerda superconductora de Witten.** Aquí φ es un fondo fijo y el fermión no siente campo gauge: es la cuerda "global" (axiónica). Con el U(1) de §2.12, una masa de Yukawa con fase exige que el fermión tenga carga axial, con las dos quiralidades de cargas distintas; sin eso no hay corriente en la cuerda. El crecimiento de esa corriente y el flujo de anomalía desde el volumen son cuánticos.

**Quarks en un monopolo (experimento 66, test 111).** Un quark isoespinor en el fondo estático de un par de monopolos de 't Hooft–Polyakov (24³, G·v = 0,6), con los enlaces que giran φ̂(r) en φ̂(r + â): un estado aleatorio en el núcleo de cada polo tiene el 6,3 % y el 3,5 % de su espectro con |ε| < 0,1, con picos en ±0,020–0,030 y estados ligados en ±0,099 y ±0,185 (referencia 3–4 %, ±0,028–0,033, ±0,10 y ±0,18). En el vacío lejano, el 0,4 % por debajo de 0,24. Son los modos cero de Jackiw y Rebbi (1976), desdoblados por el efecto túnel entre los polos. Su carga ½ y la catálisis de procesos que violan el número bariónico son cuánticas.

### (s), lo que se aclaró: la luz en una métrica que no es conformemente plana, el efecto Hall del condensado y la rama adiabática

**La luz (experimento 62, test 113).** En el escenario luz, la velocidad local es c(r) = cos θ(r)/2 por paso: un cono isótropo con velocidad 0,4777, 0,4777 y 0,4756 en x, y y la diagonal (θ = 0,3; cos θ/2 = 0,4777).
- **Con n = 1 + 2GM/√(r² + a²)** (Schwarzschild débil en coordenadas isótropas), la luz se desvía hacia la masa: −0,119, −0,058 y −0,055 rad con (GM, b) = (2, 70), (2, 100) y (1, 70), el 0,97, el 0,75 y el 0,94 de la integral eikonal sobre el tramo recorrido (referencia −0,115, −0,051 y −0,050; 1,14, 0,75 y 0,96). Es aproximadamente proporcional a GM (×2,2 al duplicarlo). Con b = 50, el paquete ancho cruza la zona del núcleo y su centroide no sigue un solo rayo (0,63 de la integral).
- **Por qué esta construcción no sirve para una partícula con masa.** Las rotaciones locales R(θ(r)) que hacen isótropo el cono acoplan el término de masa a los gradientes del medio (§3.17). Hace falta una construcción hecha para eso, como la de Arrighi y Facchini (2017).

**El efecto Hall anómalo del condensado (experimento 67, test 112).** Con fotones lentos (κ = 0,1) el condensado de Dirac es estable, y un E_y uniforme lo hace oscilar a dos frecuencias si el espín es perpendicular a E (0,1838 y 0,2254) y a una sola si es paralelo (0,2023): los valores de la referencia (0,1837, 0,2255 y 0,2025). Es la respuesta girotrópica de un medio polarizado con curvatura de Berry: una conductividad Hall σ_H = ω_p²·m\*·Ω que parte el modo en ω± = √(ω_p² + σ_H²/4) ± σ_H/2. El mismo condensado expulsa el campo magnético de dos láminas de corriente: B_z a 1, 3, 6 y 16 celdas vale −35,4, −19,3, −5,8 y −0,8 (×10⁻⁵), los valores de la referencia, frente a la ecuación de London en la red (−30,8, −16,4, −6,4 y −0,6). Sin acoplamiento, B = −50·10⁻⁵ uniforme.

**La rama adiabática del bombeo E ∥ B (experimento 74).** Con k_z creciendo a 10⁻⁴ por paso, el estado de la rama quiral sigue la rama adiabática exacta: sube con velocidad +0,98 hasta la cima de la banda del nivel cero (ε = 0,752 en k_z = 0,806; referencia 0,751 y ~0,8), donde la velocidad cambia de signo, y baja hasta √(2q·B) = 0,416 en k_z = π/2: el nivel n = 1 del doblador, no su rama quiral en ε = 0. Con 5·10⁻⁴ por paso se queda atrás (0,465). En esta caminata el bombeo no lleva un estado de un nodo al otro: el contenido de una partícula de la anomalía es el flujo espectral, y cómo se conectan las ramas por encima de la cima depende de la red.

### (t), lo que se cerró: el corrimiento al rojo, la onda gravitacional, Kaplan y el condensado con c_γ = c

**Métricas que dependen del tiempo (experimento 77, tests 119 y 120).** Con θ uniforme y variable, la construcción del escenario luz es exacta en cada ronda.
- **Expansión.** Con cos θ(t) = c₀/a(t) y a de 1 a 2 en 200 rondas, la frecuencia del modo central es la energía de banda instantánea a 10⁻⁵ (0,27852) y a·ω cambia un 0,98 % mientras a se duplica (1 + z = 2; la diferencia es la dispersión de la red con q = 0,3). El 99,9995 % sigue en la banda superior.
- **Onda gravitacional.** Con h₀ = 0,05 y un período de 120 rondas, la diferencia entre lo que recorren dos paquetes perpendiculares vale −2,04, −4,08, −2,04 y 0,00 celdas en las rondas 30, 60, 90 y 120, frente a −1,98, −3,96, −1,98 y 0 de la integral de las velocidades de banda (referencia −2,06, −4,12 y −2,06). El primer orden del continuo, −2c₀·∫h dt, da −3,65 en la ronda 60; la red añade un 12 %.

**Kaplan a momento físico fijo (experimento 75, test 117).** Para cada k físico el problema es una caminata 1D en la quinta dirección. Con L_s = 64, m₀ = 0,3 y k = 0,1 a lo largo de x, de y o de z, cada pared tiene dos estados en ±0,0955 = ±cos m₀·|k| con el 98 % de la norma a menos de 6 celdas. La pared A tiene ⟨γ⁵⟩ = −0,955 y helicidad −1,000 en el estado de energía positiva (un Weyl levógiro); la B, +0,955 y +1,000 (dextrógiro). En la diagonal la anisotropía del orden fijo da pendientes 0,975 y 0,935 (los valores de la referencia). La quiralidad de cada pared es real y el doblador está en la otra: Nielsen–Ninomiya se cumple como en Kaplan (1992).

**Por qué no hay condensado estable con c_γ = c (experimento 59).** La resonancia ε₀ − ε_b(−k) = ω_fotón(k) solo se evita si ningún estado de materia cae en el arco de cuasi-energías de longitud ω_max justo por debajo del condensado. Con κ = 1 en 3D, el fotón de red llega a 4,19 rad/paso (2√3 ≈ 3,46 con s → ∞), más de la mitad del círculo:
- **Con κ = 1** el fotón cruza m + E(k) en k_x = 2,293 y el campo de un condensado de 12³ se multiplica por 10⁴ en el paso 39.
- **Con κ = 0,1** (fotón máximo 1,11 < 2m = 1,6) no hay cruce: la energía del campo pasa de 1,65·10⁻¹³ a un máximo de 3,1·10⁻¹³ en 3000 pasos. La onda transversal adquiere masa: sin carga oscila a 0,0310 (el fotón de red) y sobre el condensado a 0,1867 y 0,2237 (referencia 0,187 y 0,224), partida en dos por el efecto Hall.
- **Con el orden alternado (período 2) es imposible** con c_γ = c: módulo π, el arco cubre todo. Con el orden fijo, la banda tiene estados justo por debajo de ε₀ por el desdoblamiento de espín. El precio de un condensado estable en estas caminatas es c_γ < c; con la caminata de Grover no hay arreglo.

## Fase 8: el modelo estándar clásico (§2.21, §2.22, §3.19 y §3.20)

### (u) El modelo estándar como teoría clásica de campos

**Por qué la base de quiralidad permite acoplar los W solo a los levógiros.** En la base de quiralidad, el salto de Dirac T_a = exp(−iα_a·k_a) es diagonal por bloques: exp(−iσ_a·k_a) sobre el espinor dextrógiro y exp(+iσ_a·k_a) sobre el levógiro. Los saltos no mezclan L y R; solo los mezcla el paso de masa. Por eso cada quiralidad puede llevar su propio enlace: los dobletes L = (ν_L, e_L) y Q_L = (u_L, d_L) saltan con U(SU(2)), y e_R, u_R y d_R no. El escenario 6 tiene 30 amplitudes por celda (6 de leptones y 24 de quarks) con SU(3)×SU(2)×U(1)_Y en los enlaces y el doblete de Higgs de §2.12.
- **La corriente cargada es V − A (experimento 83, test 128).** Con un W¹ estático de 0,05 por celda en los enlaces x, un ν_L y un u_L sin masa se convierten en e_L y d_L con la fracción 0,4646, 0,9950, 0,6054 y 0,0199 en los pasos 30, 60, 90 y 120: sin²(φt/2) exacto. La norma de e_R cambia en 8·10⁻¹⁴: el W no lo ve. Con la masa de Yukawa del electrón, el e_L creado se mezcla con e_R (hasta el 1,7 % de la norma en 120 pasos): la masa viene del Higgs.
- **Por qué los dobladores impiden que sea exacto.** El campo L de la red es una caminata de Weyl y, como toda caminata local, tiene nodos de las dos quiralidades (Nielsen–Ninomiya, también en las redes de Floquet). Con el orden alternado, el operador de dos pasos tiene 64 nodos, en los puntos con k_a ∈ {0, ±π/2, π}. El W se acopla al campo L en todos ellos, así que la mitad de los "levógiros" de la red son, cerca de su nodo, dextrógiros con carga de SU(2): la teoría completa es vectorial. Para paquetes cerca de k = 0 y campos suaves el acoplamiento es quiral; a escala de red, no.
  - **Los pseudo-dobladores (experimento 81).** En los 32 nodos con un número par de componentes ±π/2 (cuasi-energía 0) el electrón tiene sus ±0,4; en los 32 de cuasi-energía π, el salto anticonmuta con el Yukawa y la masa se cancela entre los dos pasos. Medido: en (0, π/2, π/2), ±0,4000; en (0, 0, π/2), los cuatro estados del electrón están en ±π/2 del operador de dos pasos (λ = −1) a 4·10⁻⁸, sin masa.
  - Las construcciones que sí resuelven el problema son acciones con fermiones de Ginsparg–Wilson (Lüscher, 1999 y 2000), no caminatas; la versión no perturbativa de una teoría no abeliana como el modelo estándar sigue abierta.

**Cómo salen las cargas eléctricas y por qué el Z distingue L de R (experimento 82, test 127).** Las únicas cargas que se ponen a mano son las hipercargas (L: −½, e_R: −1, Q_L: 1/6, u_R: 2/3, d_R: −1/3, Higgs: ½) y el isospín de los dobletes. Un campo uniforme a lo largo de y en la combinación del fotón (enlace T³ con ángulo θ y A_Y = −θ/g′) da a cada estado la fase e^{−iθ(T³ + Y)} = e^{−iθQ}, y su aceleración transversal es proporcional a Q.
- **Fotón.** Relativa a la de e_L: ν_L 0,000, e_R 1,000, u_L y u_R −0,665, d_L y d_R +0,331 (Q/(−1) = 0, 1, −2/3 y +1/3). El fotón ve igual a L y a R porque Q no depende de la quiralidad: Q = T³ + Y vale −1 para e_L (−½ − ½) y para e_R (0 − 1).
- **Z.** Con el ángulo θ en T³ y A_Y = +tan²θ_W·θ/g′, la fase es proporcional a T³ − sin²θ_W·Q. Medido: ν_L −2,611, e_R −1,602, u_L −1,535, u_R +1,067, d_L +2,072, d_R −0,533, frente a −2,600, −1,600, −1,533, +1,067, +2,067 y −0,533. El Z empuja distinto a e_L y a e_R, y ve al neutrino: es la corriente neutra quiral. Distingue L de R porque T³ solo lo tienen los levógiros (los dobletes); el fotón es justo la combinación en la que T³ e Y se juntan en Q.
- sin²θ_W = g′²/(g² + g′²) = 0,3077 con los acoplamientos de §3.19; en la naturaleza es 0,231. Ninguno de los dos números sale de aquí: g y g′ se ponen a mano.

**Por qué el paso de Yukawa tiene que empujar al Higgs.** Con Φ fijo, E_Y = y·Σ_s(L_s†·Φ·e_{R,s} + h.c.) es una rotación local de ángulo μ = y·|Φ| entre e_R y la componente del doblete a lo largo de Φ, χ = Φ†L/|Φ|. Esa rotación cambia la carga de SU(2) y la de U(1)_Y de los fermiones de la celda: e_L tiene T³ = −½ y e_R no tiene isospín. Si nada lo compensa, la ley de Gauss se rompe, porque el campo eléctrico no se ha enterado.
- **La compensación es el empuje sobre el Higgs.** El mismo término empuja Π ← Π − ∫∂E_Y/∂Φ* dt. Como E_Y es invariante gauge, el flujo exacto que genera conserva la carga total, fermiones más Higgs, celda a celda (Noether). Con Φ fijo durante el paso, el empuje se integra de forma cerrada: ΔΠ_i = −y·∫(S*·D_i)dt, con ∫S* = S₀*·sin μ/μ + iχ₀*·(1 − cos μ)/μ y ∫S*χ a partir de las integrales de cos², sin² y sin·cos (para el u, las conjugadas con Φ̃ = iσ₂Φ*).
- **Medido.** Con la reacción, en 10³ con todo dinámico y 100 pasos, las derivas de Gauss son 3·10⁻¹⁵ (SU(3)), 4·10⁻¹⁵ (SU(2)) y 3·10⁻¹⁵ (U(1)_Y), y la copia con una transformación aleatoria de SU(3)×SU(2)×U(1)_Y coincide a 10⁻¹⁵. Sin ella, con leptones aleatorios en 8³, la ley de Gauss deriva 0,60 en SU(2) y 0,61 en U(1)_Y en 50 pasos. Con tres generaciones en 8³, 7·10⁻¹⁵, 5·10⁻¹⁵ y 6·10⁻¹⁵.
- **En el vacío** Φ = (0, v), χ es e_L y la componente perpendicular es el neutrino: no tiene pareja dextrógira y no tiene masa. Los autovalores del operador de dos pasos en k = 0 son ±0,4000 (electrón), ±0,3000 (d) y ±0,5000 (u) con error < 4·10⁻¹⁶, y 0 para el neutrino, con dos estados en lugar de cuatro (una sola helicidad); en k = (π/4, 0, 0) el neutrino está en ±0,78540 = ±|k|.

**La cancelación de anomalías como aritmética de las hipercargas.** Contando los campos dextrógiros como sus conjugados levógiros (Y → −Y), una generación tiene Q_L (Y = 1/6, 6 estados), L (−½, 2), u_R^c (−2/3, 3), d_R^c (+1/3, 3) y e_R^c (+1, 1). Las anomalías que podrían romper las simetrías gauge son sumas sobre ese contenido, y todas se anulan:
- Y³: 6·(1/6)³ + 2·(−½)³ + 3·(−2/3)³ + 3·(1/3)³ + 1 = (1 − 9 − 32 + 4 + 36)/36 = 0.
- Y (gravitacional): 6·(1/6) + 2·(−½) + 3·(−2/3) + 3·(1/3) + 1 = 1 − 1 − 2 + 1 + 1 = 0.
- SU(2)²·Y (solo los dobletes): 3·(1/6) + (−½) = 0. Necesita los 3 colores del quark: sin quarks, los leptones solos son anómalos.
- SU(3)²·Y (solo los quarks): 2·(1/6) − 2/3 + 1/3 = 0. SU(3)³: 2 tripletes y 2 antitripletes. SU(2) global (Witten): 4 dobletes, un número par.
- **Por qué es un requisito cuántico.** La anomalía es la no conservación de una corriente gauge por el efecto del mar de Dirac (el diagrama triangular, o el flujo espectral de §2.17): es de un bucle, proporcional a ħ. En la teoría clásica las corrientes se conservan con cualquier hipercarga, y en este escenario, con un campo por especie y sin mar, no hay nada que cancelar: la aritmética explica por qué las hipercargas son estas, pero la simulación no la necesita. En la red, además, los dobladores hacen vectorial cada especie y la anomalía se cancela especie a especie (Nielsen–Ninomiya), que es otra forma de ver por qué la teoría gauge quiral no es exacta aquí.

**Oscilaciones de sabor y violación de T como interferencia de una sola partícula (experimento 84, test 129).** Con un índice de sabor y la matriz de masas M = U·diag(m)·U† en el paso de masa, un paquete creado en un sabor es una superposición de estados de masa que avanzan con frecuencias distintas. Su sabor oscila; es interferencia de una sola partícula, clásica.
- **Dos sabores** (θ = 0,6, m = 0,10 y 0,16, k₀ = 0,5, σ = 300 en 4096 celdas): P(ν_e → ν_μ) = 0,3535, 0,8378, 0,6655, 0,1172, 0,0851 y 0,6211 en los pasos 100–600, igual que la suma exacta sobre k a 10⁻⁴. El período con las bandas de la red, 2π/(ω₂ − ω₁) = 454 pasos, es más largo que el del continuo, 4πE/Δm² = 403: con k₀ = 0,5 la red reduce ω₂ − ω₁.
- **Tres sabores** (ángulos 0,58, 0,78 y 0,40, m = 0,10, 0,13 y 0,20): con δ = π/2 (J = +0,0757), P(e → μ) − P(μ → e) = +0,310, +0,752, −0,285, −0,655 y +0,481 en los pasos 200–1400; con δ = 0, 4·10⁻¹⁶. La probabilidad de ida y la de vuelta difieren solo si la matriz de mezcla tiene una fase que no se puede quitar redefiniendo los campos, y el tamaño lo fija J = Im(U_e1·U_μ2·U*_e2·U*_μ1). Viola T, y CP por CPT.
- **La CKM de tres generaciones (experimento 85, test 130).** En el escenario 6 la mezcla no se pone en el paso de masa: sale de matrices de Yukawa complejas 3×3. La SVD Y = W·diag(y)·V† separa el flujo en tres Yukawas escalares independientes (pares D′_k = (W†D)_k y S′_k = (V†S)_k), cada uno con su empuje sobre Π. Las masas son los valores singulares por v (las 9 cuasi-energías positivas en k = 0 coinciden a 6·10⁻⁹, con 6 modos nulos de los neutrinos). Los W actúan en la base de generación, así que la matriz que conecta los estados de masa de tipo u con los de tipo d es V = W_u†·W_d: la |V| sacada de los autovectores de la caminata coincide con |W_u†·W_d| a 3·10⁻¹⁵. Su fase la mide J, que también sale del conmutador de las matrices de masa: det[Y_u·Y_u†, Y_d·Y_d†] = 2i·J·Π(Δm²_u)·Π(Δm²_d) a 7·10⁻¹⁵. Con W y V reales, J = 0.
- Los neutrinos con masa (la PMNS) no están en el modelo estándar mínimo: harían falta ν_R o una masa de Majorana. La mezcla de leptones del escenario 6 no es observable, porque los neutrinos no tienen masa.

**Qué falta para que esto sean las partículas observadas.** Todo lo anterior es la teoría clásica de campos del modelo estándar: un campo por especie, ondas de una partícula.
- **Cuantizar los fermiones.** Hacen falta la estadística de Fermi, el mar de Dirac y la creación de pares; un W de este escenario no se desintegra en leptones. Con campos gauge y de Higgs de fondo se puede hacer exactamente evolucionando los modos del mar (§2.23).
- **Cuantizar los campos gauge y el Higgs.** Sin eso no hay confinamiento (el Yang–Mills clásico no confina, §2.11), ni el corrimiento de los acoplamientos con la escala, ni correcciones radiativas.
- **Una teoría gauge quiral en la red.** Los dobladores de la caminata la hacen vectorial (arriba), y la construcción no perturbativa del modelo estándar en la red sigue abierta.
- **Los parámetros.** g, g′, g_s, λ, v, los Yukawa y las fases se ponen a mano, como en el modelo estándar. Por qué hay tres generaciones no lo explica el modelo estándar, y tampoco esto.

## Fase 9: creación de partículas (§2.23, §2.24, §3.21 y §3.22)

### (v) La creación de partículas

**Cuantizar fermiones libres es evolucionar los modos del mar.** Hasta aquí cada fermión era una onda clásica de una partícula. Pero la caminata es lineal en el campo fermiónico, y un estado de Slater (el mar de Dirac lleno) sigue siendo de Slater bajo una evolución lineal: el operador de evolución del campo cuantizado es el de una partícula aplicado a cada modo ocupado. Con campos gauge y de Higgs de fondo esto es exacto (el método de las funciones de modo, Aarts y Smit, 1999), y con la reacción de los fermiones sobre el campo es la aproximación clásico-estadística (Kasper, Hebenstreit y Berges, 2014).
- Incluye la estadística de Fermi sin ponerla a mano: cada modo es un vector unitario, así que su ocupación de la banda positiva nunca pasa de 1. Con q·E = 0,04, la ocupación máxima es 0,552 (referencia 0,52).
- El número de partículas es la ocupación final de la banda positiva, Σ_k |⟨+, k|ψ_k⟩|². Cada partícula es un par, porque el hueco que deja en el mar es la antipartícula.

**Por qué la tasa de Schwinger de la caminata es el doble.** Un campo uniforme barre k a ritmo q·E, y cada modo que cruza el hueco salta con p = exp(−π·m²/(c·q·E)) (Landau–Zener). Con el mar lleno, la tasa es (q·E/2π)·p por hueco.
- La caminata convencional tiene dos huecos idénticos: el de k = 0 y el de Floquet en k = π, un pseudo-doblador. Medido (m = 0,1): 2,93·10⁻⁶, 1,37·10⁻⁴, 1,32·10⁻³ y 5,78·10⁻³ pares por celda y paso con q·E = 0,005–0,04, frente a 2·(q·E/2π)·e^{−πm²/(qE)}: cocientes 0,988, 0,994, 0,996 y 0,996 (los de la referencia).
- La tasa hay que medirla como pendiente entre q·E·t = 1,5 y 3,0: cada modo cruza cada hueco una vez como mucho. Con varias vueltas a la zona, todos los modos recorren la misma trayectoria en k y la interferencia de Stückelberg entre cruces es coherente: no se promedia, y la tasa sale mal (1,17 y 3,7 veces).

**Higgs y W clásicos que oscilan.** Un Higgs homogéneo con m(t) = m_f + δm·cos(m_H·t) es un condensado de cuantos de densidad n_H = ½·m_H·h₀², y cada cuanto se desintegra con Γ = y²·m_H·β³/(8π), la anchura del modelo estándar (Djouadi, 2008). La producción de pares es Γ·n_H = m_H²·δm²·β³/(16π) por volumen y tiempo (Greene y Kofman, 1999). Para el W¹ en los enlaces z, Γ·n_W = M²·θ₀²/(96π); para el Z, cada Weyl da M²·φ_f²/(48π).
- **Cada nodo da la tasa del continuo.** Con la caminata alterna de §2.16, el operador de dos pasos tiene 64 nodos. Medido con 2000 pasos: Higgs 1,0195 Γ·n en (0, 0, 0) con |q| ≤ 0,4 y 1,0055 en los cuatro nodos de cuasi-energía 0 con la ventana estrecha; W 1,0109 y 1,0007 en los nodos de las cuatro clases; Z 1,0089 para L y para R.
- **La red multiplica la tasa.** El Higgs produce 32Γ: en los 32 nodos de cuasi-energía π el salto anticonmuta con el paso de masa y la masa se cancela entre los dos pasos, así que allí el fermión no tiene masa y no se acopla al Higgs (2,8·10⁻⁵ Γ·n, referencia 0,001). El W produce 64Γ: se acopla al campo L en todos los nodos, y la mitad tienen la quiralidad contraria, así que el W de la red también se desintegra en dobladores dextrógiros (Nielsen–Ninomiya en una desintegración). Para leer una sola especie, el laboratorio integra cerca de un nodo.
- El bosón es un condensado clásico y su "desintegración" se lee como producción de pares a primer orden: no es la desintegración de un W o de un Higgs aislados.

**El mar inerte de la caminata convencional en 1D.** La transformación escalonada ψ(x) → (−1)^x·ψ(x) cambia el signo de cada paso (U → −U, porque las dos componentes saltan una celda), así que lleva el mar de cuasi-energía negativa a su complemento, también durante la evolución. La densidad y la corriente no cambian con ella, así que son exactamente las de medio llenado: n = 1 y j = 0 con cualquier campo. Medido: |n − 1| y |j| < 4,8·10⁻¹⁴ con fases de enlace aleatorias que cambian cada paso, y la cuerda entre dos cargas no se mueve (|E − q| = 7·10⁻¹⁵ en 600 pasos).
- Es la forma que toma aquí un problema conocido: en tiempo discreto la energía es modular y el mar tiene una segunda frontera, en ±π, donde crear pares sale gratis. Los pares del hueco de k = 0 los compensa exactamente el de π. Los pseudo-dobladores son de Gupta y Short (2025, 2026); la inestabilidad del vacío en tiempo discreto y su arreglo con fermiones tangentes, de Beenakker (2026).
- **La caminata de Gupta y Short** (U = W·A·B con proyectores rotados un ángulo θ) no tiene dobladores ni, si 2θ > m, pseudo-dobladores: sus bandas no llegan a ±π. Da la tasa de una sola especie con la velocidad de su banda: c_ef = 0,93882 (referencia 0,93881, frente a cos θ = 0,9553) y cocientes 1,000, 1,000, 0,999 y 0,999. Y con la reacción, la cuerda se rompe: con d = 48, el campo dentro cae de 0,4 a 0,096 en 100 pasos, se crean 1,9 pares que se separan y apantallan las cargas (−0,17 junto a +q en el paso 100 y −0,22 a −0,30 después), y la energía del campo (3,84) pasa a los fermiones. La rotura es gradual: la fracción de campo que queda es 0,89 con d = 4 (energía de la cuerda por debajo de 2m: solo polarización) y baja hasta 0,30 con d = 32. Es campo medio y 1D.

**Qué está hecho y qué sigue abierto.**
- Las teorías gauge quirales en la red: hay construcciones exactas para las abelianas libres de anomalías (Lüscher, 1999) y a todos los órdenes perturbativos para las no abelianas (Lüscher, 2000), y avances recientes en formulación hamiltoniana (Thorngren, Preskill y Fidkowski, 2026); la versión no perturbativa del modelo estándar sigue abierta.
- Las caminatas 3D sin dobladores: la versión 3D de Gupta y Short aún tiene soluciones espurias; aquí, en 3D, se sigue con la caminata alterna y sus 64 nodos. El motor de §2.25 (fase 10), que no es de tipo luz, resuelve las dos cosas.
- La rotura de la cuerda cuántica completa ya se observa en simuladores cuánticos (De et al., 2026); la de esta app es su versión de campo medio (Hebenstreit, Berges y Gelfand, 2013).

**Las fracciones del Z y los quarks en un campo de color.**
- El Z actúa sobre cada especie de Weyl como una fase abeliana de amplitud proporcional a T³ − s²Q, así que Γ_f ∝ (T³ − s²Q)² sumado sobre L y R, ×3 en los quarks. Con s² = 0,2312, tres generaciones sin el top y fermiones sin masa: invisible (3ν) 20,54 %, hadrones 69,13 %, leptones cargados 10,33 % y Γ(νν̄)/Γ(ℓ⁺ℓ⁻) = 1,99, frente a 20,00 %, 69,91 % y 10,10 % del PDG. El árbol acierta al ~1 %; faltan las masas (el b y el τ) y las correcciones de QCD y electrodébiles. El invisible cuenta los neutrinos, que no tienen carga pero sí T³.
- Con un campo de color homogéneo a lo largo de n, cada autovalor λ_i de n·T es un canal abeliano de carga |λ_i|·g. Los autovalores dependen de los dos invariantes de Casimir, E^a·E^a y (d_abc·E^a·E^b·E^c)²: con el mismo |E| = 0,06, la tasa es 3,127·10⁻³ a lo largo de T³ (λ = ±½, 0) y 2,895·10⁻³ a lo largo de T⁸ (−0,577, 0,289, 0,289), un 8 % menos, y 2,93·10⁻³ en una dirección aleatoria (cocientes con la suma por canales 0,999 en los tres). A lo largo de T³ el tercer color no tiene carga y no se produce (Nayak, 2005).

**La anomalía, el bosón de Schwinger y el vacío dieléctrico** (1+1D, caminata de Gupta y Short).
- **Anomalía.** Con fermiones sin masa y E constante, el mar lleno desplazado por q·E·t crea carga quiral a ritmo q·E/π, así que dj/dt = c·q·E/π. Medido: cocientes 0,9998 y 0,9985 con θ = 0,3 y 0,9993 y 0,9958 con θ = 0,5. Exige el mar lleno: una partícula sola no tiene anomalía. Con la caminata convencional, dj/dt = 10⁻²¹: el pseudo-doblador de π aporta exactamente lo contrario.
- **Bosón de Schwinger.** Con la reacción, un campo homogéneo pequeño oscila a q·√(c/π) (con la corrección del salto de rana, 2·arcsen(q·√(c/π)/2)): 0,02758, 0,05519 y 0,11062 frente a 0,02757, 0,05515 y 0,11035. Es la masa e/√π del fotón de la QED sin masa de 1+1D (Schwinger, 1962): el fotón se come la anomalía y aparece una partícula compuesta, sin masa en la acción.
- **Polarización del vacío.** Con masa, los pares virtuales apantallan un E pequeño: E₀/ε con ε − 1 = q²·c/(6π·m²) (segundo orden en el acoplo entre bandas, con el elemento dipolar |⟨+|∂_k|−⟩|² = m²c²/(4ε_k⁴)). Con q = 0,1: 0,2076, 0,0518 y 0,0130 frente a 0,2011, 0,0498 y 0,0122 para m = 0,05, 0,1 y 0,2 (los números de la referencia); el error crece con m porque la banda de la red se aparta del continuo en k ~ m.

### (w) La producción clásica de bosones

**Por qué vale un campo bosónico clásico.** Un campo clásico describe bien un bosón cuando la ocupación de sus modos es grande: entonces las fluctuaciones cuánticas son una corrección. La evolución clásica partiendo de una semilla de ½ cuanto por modo es el primer orden de un desarrollo controlado en esas fluctuaciones, exacto a tiempos cortos (Polkovnikov, 2003): es la aproximación clásico-estadística de las simulaciones de recalentamiento (Khlebnikov y Tkachev, 1996). Aquí la semilla es ħ_ef/2 por enlace con ħ_ef = 10⁻⁸, muchísimo menos que los cuantos del condensado, y los exponentes de crecimiento no dependen de ella. Pierde precisión a tiempos largos o en procesos muy no perturbativos.

**W y Z desde un Higgs que oscila.** Con Φ = (0, v + δ₀) uniforme, el modo uniforme cumple φ̈ = −2λ(φ² − v²)φ, y cada modo transverso, Ẅ + (κk̂² + M²(t))W = 0 con M_W² = κg²φ²/2 y M_Z² = κ(g² + g′²)φ²/2: una ecuación de Mathieu. La resonancia paramétrica principal está en ω_k ≈ m_H/2 y solo existe si m_H > 2M: es la versión clásica y estimulada de H → WW y H → ZZ (el recalentamiento electrodébil de Rajantie, Saffin y Copeland, 2001).
- Un modo suelto del W en la capa k̂² = 0,1522 crece con 0,01685, frente al Floquet 0,01686. En 32³ con la semilla, en la fase lineal crecen solo las capas resonantes: el Z en k̂² = 0,0769 con 0,0243 (Floquet 0,0249) y el W en 0,1522 con 0,0164 (0,0169). En t = 300 el 97,3 % de la energía eléctrica de gauge es del Z, el 2,5 % del W y el 0,2 % del fotón.
- **El fotón no sale en la fase lineal** porque no ve a φ: no hay H → γγ en el árbol clásico, igual que en el modelo estándar, donde va por bucles. Su exponente en la fase lineal es 3·10⁻⁶.
- **Fase no lineal.** Cuando los modos resonantes crecen, la redispersión reparte la energía entre todos los modos, fotón incluido: en t = 900, W± 61 %, Z 24 % y fotón 15 %, y el Higgs ha pasado de 526 (t = 300) a 299. La ley de Gauss sigue a 7·10⁻¹⁵.
- **Un Higgs ligero no produce nada:** con m_H = 0,35 < 2M_W, el exponente de Floquet es 0 en todas las capas y la energía de W y Z no crece.

**Por qué Floquet sobre un número entero de períodos.** El exponente de Floquet es el logaritmo del autovalor de la monodromía de un período, dividido por el período. Sobre una ventana que no es un número entero de períodos, la matriz de transferencia mezcla el crecimiento con la fase del ciclo y el exponente sale sesgado (en O(1)/T). El período de φ (10,875 pasos con δ₀ = 0,2) no es entero en pasos, así que se toma el número de períodos entre 15 y 60 que cae más cerca de un número entero de pasos, integrado con el mismo salto de rana y el φ(t) de la red.

**Nielsen–Olesen y Ambjørn–Olesen.** Un gluon cargado (colores 1 y 2, carga adjunta 1) en un campo cromomagnético uniforme tiene niveles de Landau con el término de espín de factor g = 2: ω² = κ(k_z² + (2n + 1)B − 2B). El nivel más bajo (n = 0, espín a lo largo de B) tiene ω² = −κB < 0 y crece con √(κB): 0,05539 y 0,07841 con uno y dos cuantos, frente a 0,05539 y 0,07833 (Nielsen y Olesen, 1978). El W cargado tiene además su masa: ω² = M_W² − κB, inestable solo si κB > M_W² (Ambjørn y Olesen, 1988): 0,04535 frente a 0,04534 con v = 0,15, y estable con v = 0,3 (×1,02 en 500 pasos). Al crecer, el W arrastra a la componente cargada del Higgs (hasta 1,4·10⁻²): es el condensado.
- **Flujo cuantizado y U(1)_Y compacta.** En el toro los enlaces fundamentales exp(−iaT³) tienen que ser periódicos, así que el flujo total está cuantizado: B·L² = 4πm, con una costura retorcida en x = L − 1. Un campo uniforme con flujo neto no cabe en una U(1) no compacta en el toro (la suma de los rotacionales de un campo real periódico es 0): con la energía no compacta, la costura deja una fuerza de 6,6 en el fondo. Con la compacta, (1 − cos(c·B_Y))/c² con c = g′/2, la fuerza es la misma en todas las plaquetas y el fondo es estático a 1,7·10⁻¹⁵.

**Higgs desde un condensado de W.** Un W¹ uniforme en los enlaces z, θ(t) ≈ θ₀·cos(M_W·t), entra en la ecuación del modo radial del Higgs por el gradiente covariante a lo largo de z: ḧ = −[κk̂² + 2λ(3φ² − v²) + 2κ(1 − cos(θ/2))]·h. El último término oscila a 2M_W, así que hay resonancia en ω_k = M_W, y solo existe si M_W > m_H: es la versión clásica y estimulada de WW → HH. Con m_H = 0,15 y θ₀ = 0,15, el modo con k = 3·2π/64 crece con 0,00106, frente a 0,00104 de la integración lineal con la serie θ(t), φ(t) de la propia red (referencia 0,00104 y 0,00105); fuera de resonancia y con m_H = 0,25 no crece (×1,01 y ×1,02). Con θ₀ = 0,6, κθ₀²/4 supera a 2λ: la energía del condensado restaura la simetría y φ cruza 0.

Con esto, el modelo produce clásicamente todos los bosones: W y Z desde el Higgs, el fotón por redispersión, gluones y W desde campos magnéticos, y Higgs desde un condensado de W. Los fermiones van por el mar de (v).

## Fase 10: más allá de la caminata de tipo luz (§2.25–§2.27 y §3.23–§3.25)

### (x) Wilson–Dirac, skyrmiones y Floquet anómalo

**Una sola especie en 3D, a cambio de la velocidad.** Una caminata de tipo luz (una celda por paso) sin dobladores en 3D sigue abierta. El motor de §2.25 renuncia a eso: es el Trotter simétrico (Strang) del hamiltoniano de Wilson–Dirac, H(k) = Σ α_j sen k_j + β(m + r·Σ(1 − cos k_j)). Cada término de enlace actúa sobre dos sitios; los enlaces de una dirección se parten en pares e impares (ladrillo), así que cada capa es un producto exacto de compuertas de 8×8.
- **Un solo Dirac.** En la celda de 2×2×2 con dt = 0,3 y m = 0,1, el nivel más cercano a 0 está en ±dt·m = ±0,0300 y solo en K = 0; los dobladores, con la masa de Wilson m + 2r·(número de componentes π), están en ±0,63 = ±2,1·dt y más arriba. Con dt·E_máx < π las cuasi-energías no llegan a ±π: la distancia mínima es 1,312, así que tampoco hay pseudo-dobladores y el mar de Dirac es estable. Con dt = 0,6 los niveles se acercan a ±π (0,19).
- **Por qué Strang.** El orden simple parte el cono en dos ramas (0,1143 y 0,2226 frente a 0,1790 del Wilson–Dirac con k = (0,1, 0, 0,1)); el simétrico da uno solo (0,1771 con dt = 0,1), con un error O(dt²).
- **El precio:** avanza dt celdas por paso, no una; es la forma estándar de la simulación digital de fermiones de Wilson, no física nueva. Con el U(1) en las compuertas (g(θ) = D·g₀·D†) la carga que mueve cada compuerta es el cambio de |ψ_{x+j}|², así que la continuidad y la ley de Gauss son exactas: 7·10⁻¹⁵ en 60 pasos, y la copia gauge coincide a 6·10⁻¹⁶.
- **No sirve para el acoplamiento quiral.** El término de Wilson mezcla L y R (β es un término de masa): un W que solo vea a los levógiros rompería la invariancia gauge. El motor vale para teorías vectoriales (QED, QCD) y para la anomalía, no para el sector electrodébil quiral.

**Con el mar lleno, una especie.** El Higgs → f f̄ (m_f = 0,1, m_H = 0,4, δm = 10⁻³) da 0,7937 veces la tasa del continuo con dt = 0,1 (referencia 0,793), frente a 32 veces con la caminata alterna (§3.21). La probabilidad de par de un modo es 0,04494, 0,04962 y 0,05021 con dt = 0,1, 0,05 y 0,025 (los números de la referencia), hacia el 0,0503 del Wilson–Dirac continuo. Una reacción homogénea débil oscila y no crece (E/E₀ entre 0,9424 y 1, la polarización del vacío) con la energía conservada a 3,4·10⁻⁸; una fuerte decae por los pares.

**Los artefactos del término de Wilson, y cómo se separan del error de Trotter.**
- **La masa r·k²/2.** A momento k, el término de Wilson añade r·k²/2 a la masa. Por eso la tasa del Higgs es el 78 % del continuo: con la resonancia en k = 0,17, la regla de oro con la dispersión de la red da 0,781. La diferencia con lo medido (0,793) es el error de Trotter, que cae como dt² (0,898 con dt = 0,2). En la probabilidad de un modo, el término separa la red (0,0503) del continuo (0,0432), y el Trotter se ve en cómo 0,0449 → 0,0496 → 0,0502 se acercan a 0,0503 al bajar dt.
- **La masa r·qB/2 del nivel de Landau más bajo.** Con E ∥ B, el nivel más bajo bombea la carga quiral de la anomalía de 3+1D, (qB/2π)(qE/π). El término de Wilson le da una masa r·qB/2 que la suprime por Landau–Zener: con m = 0, 0,279 de la tasa (referencia 0,255); con una masa física m = −r·qB/2 que la cancela, 0,854 (referencia 0,882), que es la eficiencia de dt = 0,2. El cociente, 0,33, es la supresión exp(−π(r·qB/2)²/(qE)) = 0,30. La caminata alterna da 0.
- **Un artefacto de tamaño finito.** Con condiciones periódicas en un toro de 6³ celdas, la energía del mar depende de un A uniforme (los armónicos de las bandas no se promedian en una rejilla finita): esa curvatura negativa hace crecer el campo. Con las antiperiódicas cambia de signo y el campo oscila; decae con el tamaño.

**Bariones que emergen de un campo clásico.** En el modelo de Skyrme (1962) los bariones son solitones topológicos del campo de piones U(x) ∈ SU(2), con el número bariónico como carga topológica.
- **El barión.** El perfil radial del erizo da E = 1,23145 × 12π² (la cota de Faddeev–Bogomolny es 1) con la cola C/r² de piones sin masa, C = 2,1596. En la red 3D, desde un perfil tosco, L-BFGS llega a 1,20682, 1,22179 (con h = 0,2 y 0,15, los de la referencia), con B = −0,9956 y −0,9988 y forma esférica.
- **La ligadura.** Dos erizos en el canal atractivo (isorrotación relativa de π) se funden en un toro con 1,16940 × 12π² por barión, el 95,8 % del barión solo: la ligadura nuclear, de un ~4 % de la masa, sale sola del campo clásico. El toro tiene simetría axial alrededor del eje de la isorrotación y un hueco en el centro (densidad 4,7 frente a 34).
- **Por qué la derivada centrada lo desenrolla.** La derivada centrada vale 0 en k = π: una configuración en damero no cuesta energía, y el solitón se desenrolla por ella (la energía cae de 1,16 a 0,002 × 12π² y B a 10⁻⁴ en 300 iteraciones). La media de las diferencias hacia delante y hacia atrás penaliza todos los modos y es de segundo orden.
- **Qué dice la cuantización.** Girar el erizo como un rotor rígido da M_N = M + 3/(8λ) y M_Δ = M + 15/(8λ). Ajustando a 938,9 y 1232 MeV salen e = 5,445 y F_π = 129,3 MeV, una masa clásica de 865,6 MeV y un radio isoescalar de 0,593 fm: los números de Adkins, Nappi y Witten (1983). **Qué no dice:** F_π sale en 129 MeV, no en el 186 físico; con la masa física del pión y sin la aproximación de rotor rígido, ningún par de parámetros reproduce el nucleón y la Δ (Battye, Krusch y Sutcliffe, 2005). Que el solitón cuantizado sea un fermión de espín ½ es la regla de Finkelstein y Rubinstein (1968), una condición de la cuantización que el campo clásico no muestra. Y es una teoría efectiva de piones: los quarks y los gluones no aparecen.

**La quiralidad nacida del forzamiento, y por qué la geometría no basta.**
- **Floquet anómalo.** Cuatro pasos de transferencia completa por +x, +y, −x y −y devuelven cada partícula del interior a su sitio (probabilidad 1,000), pero en un borde falta un enlace y la partícula avanza en un solo sentido, dos celdas por período: (0, 11) → (0, 5) → (0, 0) → (6, 0) → (12, 0) (Rudner, Lindner, Berg y Levin, 2013). El interior es trivial y la quiralidad está en la frontera, como en las paredes de Kaplan (§2.20); el compañero de quiralidad contraria está en la otra frontera. Con un quinto paso de fases aleatorias en cada sitio, el modo de borde no se dispersa (probabilidad 1,000).
- **En un fractal.** En la alfombra de Sierpinski de nivel 3, cada agujero es una frontera con su propio canal quiral: el borde exterior va en sentido antihorario y el contorno del agujero central, en horario (como en las redes fotónicas fractales de Li et al., 2023). La quiralidad total se compensa entre el borde exterior y los agujeros.
- **El argumento del índice.** Un hamiltoniano con simetría quiral tiene la forma H = [[0, D], [D†, 0]] con D cuadrada, sea cual sea la red. Su índice, dim ker D − dim ker D†, es 0. Medido con el fermión ingenuo y 3 cuantos de flujo: 6 modos casi nulos en la red cuadrada 27×27 (3 de cada quiralidad) y 40 en la alfombra (20 y 20), también con saltos aleatorios (6 y 32); la suma de quiralidades es 0 en los cuatro casos. Una sola especie necesitaría índice 3. Así que ni una red aleatoria (Christ, Friedberg y Lee, 1982), ni un fractal, ni el desorden pueden dar una sola especie quiral sin romper la simetría quiral. Lo que sí quita los dobladores es romperla (el término de Wilson) o separarlos en fronteras (Kaplan, Floquet anómalo).

## Fase 11: gravedad análoga (§2.28, §3.26 y los añadidos de §2.6, §2.7 y §3.4)

### (y) El superfluido de la caminata como laboratorio de gravedad análoga

**La resonancia paramétrica del fondo.** El fondo repulsivo de §2.7 no es estacionario sin más: el paso es un mapa de Floquet, y dos cuantos del condensado (cuasi-energía Ω) pueden pasar a dos excitaciones de bandas distintas cuando ω_a(k₀ + q) + ω_b(k₀ − q) ≡ 2Ω (mod 2π). Es creación de pares por el forzamiento periódico, que en tiempo continuo no existe. El espectro de Floquet–Bogoliubov por bloques de 2n_c × 2n_c en q lo da sin simular:
- **Es universal en c_s/ξ.** La tasa es ≈ 0,30·c_s/ξ = 0,42·ρ̄f′ (c_s/ξ = √2·ρ̄f′ no depende de m\*): 0,301, 0,301 y 0,280 en la tira con m = 0,2, 0,5 y 0,8; 0,301 en 2D (9,45·10⁻⁴ por paso con g = −0,01) y 0,279 en 3D. El acoplamiento entre las dos ramas es ρ̄f′ (la parte local del paso es exp(iΘ′·P) = 1 + iΘ′·P, con P² = 0) y, mientras haya bandas que cumplan la condición, la tasa es una fracción fija de él. Es muy estrecha en q: con m = 0,5 y g = −0,03 está en q = 1,167 (2,84·10⁻³ por paso), y una tira de 4096 celdas con ruido de 10⁻⁹ crece justo ahí a 2,76·10⁻³.
- **Una masa de moneda grande la quita.** Al subir m, las bandas que podrían sumar 2Ω (mod 2π) se separan: 0,172·c_s/ξ con m = 1,0, 0,031 con 1,2 y ninguna con m ≥ 1,35 en la tira; en 3D, 0,061 con 1,35, 0,041 con 1,4 y ninguna con 1,5. El precio es una masa efectiva grande (m\* = 8,9, 11,6 y 42,3), que baja c_s y ξ para el mismo g.
- **Un matiz que la especificación no dice.** Con m = 1,4 y g = −0,03 queda una resonancia estrecha (9,3·10⁻⁵ por paso, anchura ±1,5·10⁻⁴ en q = 0,4786). Una rejilla q = 2π(j + ½)/nq la salta, pero una tira de 4096 celdas la tiene como modo (q = 2π·312/4096), y la caminata crece ahí a 9,33·10⁻⁵ por paso. En la tira del láser (512 celdas, con flujo) ningún q permitido es resonante (< 10⁻¹³).

**El horizonte acústico y el láser de agujero negro.** En un fluido en movimiento el sonido ve una métrica efectiva; donde el flujo supera a c_s hay un horizonte. En la tira, la onda plana de la banda con k₀ = 2π·16/512 fluye a v = 0,01699 (la velocidad de grupo de la banda). Bajar g a g_sub/8 en 10 celdas, con un potencial que compensa la fase no lineal, deja la onda plana como solución exacta (3·10⁻¹³ tras 1000 pasos) con una región supersónica: Mach 0,87 fuera y 2,45 dentro.
- **El toro obliga a tener un horizonte blanco.** El flujo que sale de la región supersónica vuelve a entrar por el otro lado: a un horizonte negro a la entrada le corresponde uno blanco a la salida. Con la dispersión de Bogoliubov, superlumínica, la cavidad entre los dos es un láser: los modos de energía negativa rebotan y crecen (Corley y Jacobson, 1999). Es una inestabilidad clásica, sin ruido cuántico.
- **La tasa** del BdG del continuo con la banda de la caminata es 2,849·10⁻⁴ por paso (2,851·10⁻⁴ en la referencia), y la caminata con ruido de 10⁻⁷ crece a 2,844·10⁻⁴. A los 60 000 pasos la densidad está modulada entre 0,25 y 0,66, con la norma conservada a 2·10⁻⁹. Con m = 0,5 el láser llegaría a 5·10⁻⁴ por paso, pero la resonancia del fondo sería 6 veces más rápida: por eso se usa m = 1,4.

**El Hawking espontáneo.** La caminata es un campo clásico. El vacío cuántico entra como en el método de Wigner truncado (medio cuanto por modo de Bogoliubov), y como las fluctuaciones pequeñas evolucionan linealmente, su covarianza Σ = ⟨zz†⟩ evoluciona exactamente como Σ ← J·Σ·J†. Es el promedio de infinitas realizaciones, sin ruido estadístico; con ruido harían falta ~3·10⁴ realizaciones para ver el lóbulo.
- **El cálculo.** El vacío del fondo uniforme se construye por bloques en q (Tr Σ₀/2N = 0,5134 ≥ ½, como la referencia). En la tira hay una banda exactamente plana (ψ_N − ψ_S) que no se acopla y que, diagonalizando la matriz completa, haría fallar el vacío. No hace falta guardar Σ: las W filas del funcional δn se propagan hacia atrás con la jacobiana en el marco rotante del fondo (22 s el mapa entero en el paso 16 000) y se combinan con Σ₀.
- **Por qué una rampa.** Un salto brusco de g puebla los modos entrantes de k alto, que remontan el flujo y se convierten en modos de Hawking al llegar al horizonte: es Hawking estimulado, y en el paso 8000 llena el cuadrante horizonte × región supersónica de franjas (un rms 13 veces mayor que con la rampa). Con una rampa smoothstep de 6000 pasos el horizonte se forma hacia el paso 3690 sin excitar nada, y solo queda el lóbulo espontáneo.
- **El lóbulo.** La correlación densidad-densidad tiene un lóbulo negativo anclado en el horizonte: el fonón de Hawking, que remonta la región subsónica a |v − c_sub| = 0,0061, correlacionado con su compañero de norma negativa, que va con el flujo. Promediando por rayos, el mínimo está en una velocidad aguas abajo de 0,0029–0,0032 (no en v − c_super = 0,0057), con ξG/ρ̄ ≈ −2,1·10⁻² en el paso 16 000. **El compañero va más despacio** porque es dispersivo: con k·ξ_super ≈ 1,8, la rama de norma negativa existe solo hasta ω_max = 2,9·10⁻⁴ y su velocidad de grupo baja de 0,0057 (ω → 0) a 0 (ω_max). El lóbulo corresponde a frecuencias cerca del corte, como en un horizonte abrupto. La caminata y el continuo (BdG con la banda de la caminata) coinciden al 0,6 % en el cuadrante.

**La ergorregión de un vórtice multicuantizado.** Alrededor de un vórtice de carga κ la velocidad azimutal κ/(m\*r) supera a la del sonido local dentro de r_e: 1,61, 3,18 y 4,73·ξ para κ = 1, 2 y 3. La estimación κ/(m\*c_s) se queda un 10–13 % corta porque el núcleo deprime la densidad. En la ergorregión hay modos de energía negativa, como en un agujero negro en rotación.
- **Por qué solo en ventanas.** Un modo de energía negativa atrapado necesita reflejarse en el borde para crecer: sin borde se iría como superradiancia. La condición de cuantización entre la ergorregión y la pared solo se cumple en ventanas de R/ξ. Con κ = 2, Im ω/(ρ̄f′) es 0 con R/ξ = 7,2–8,4; 0,025, 0,039, 0,041, 0,038 y 0,027 con 8,8–10,3; 0 de 10,6 a 14,5; 0,0175 en 19,3 y 0 en 24,1 (los de la referencia). Con κ = 3, inestable con 9,7 (ℓ = 3) y 14,5 (ℓ = 2).
- **En la caminata** (m = 1,4, ξ = 8, disco de R = 76 = 9,5ξ), el cuadrupolo del núcleo crece de 0,02 a 1,56 en 200 000 pasos, con una pendiente final de 2,65·10⁻⁵ por paso frente a 2,79·10⁻⁵ del BdG radial: el vórtice se divide en dos de carga 1. El dipolo se queda por debajo de 4·10⁻⁵ del cuadrupolo; la división es simétrica porque el paso conserva exactamente la simetría de inversión de la red.

**La reconexión de vórtices.** En 3D los vórtices son líneas. En el fluido de Euler, dos tubos de vorticidad que se cruzan son un candidato a singularidad en tiempo finito; en un superfluido la reconexión ocurre de forma suave a la escala ξ, porque el núcleo no es una línea matemática (Koplik y Levine, 1993). En 96³, dos anillos de R = 20 se acercan de δ = 44 a 10,1 en el paso 640 y reconectan en el 676 (la referencia, 675), sin vórtices espontáneos. Con α = ½ impuesto, δ = A·√(κ|t − t_r|) da A₋ = 0,94 antes y A₊ = 1,7 después: se separan más deprisa de lo que se acercan, como en Gross–Pitaevskii. **Lo que no sale** es el exponente ½ limpio: con δ del orden de ξ (6,8 celdas) y un error de localización de ~0,3 celdas, la red no llega al régimen asintótico (el ajuste libre da α = 0,46 entre δ = 2,5 y 10; la referencia, 0,39). Haría falta ξ de 15–20 celdas, es decir, ≥ 192³.

**Unruh de fermiones con el mar lleno.** El vacío de Dirac restringido a un intervalo es térmico respecto al generador de su flujo conforme, con T = 1/2π (Bisognano y Wichmann, 1975). Cerca de cada borde ese generador es el de Rindler, con una velocidad local que crece linealmente desde el borde: es el efecto Unruh, la parte de la radiación de Hawking que vive junto al horizonte.
- Con el mar de Wilson–Dirac 1D (r = 1, m = 0) en la cadena infinita y el generador de red con pesos x(ℓ − x)/ℓ (sitios en j + ½), los 10 niveles más bajos con ℓ = 100 dan T = 1,0000·(1/2π).
- El espectro de entrelazamiento coincide con 2π·ω a 4 cifras (0,6225, 1,9130, 3,3077 y 4,8122). Con r = 1 y m = 0 la cadena son dos cadenas de Ising críticas, así que es una identidad conocida.
- Con el mar de Floquet del Trotter simétrico de §2.25 sale 0,9993, 0,9956 y 0,9839·T con dt = 0,1, 0,25 y 0,5, un error O(dt²). El orden de las capas importa: repartir la masa en las compuertas de enlace en lugar de una compuerta local central daba ya un 14 % de error con dt = 0,1.

**El núcleo de Thomas–Fermi de la materia oscura autointeractuante.** Con g < 0 y la gravedad estática en 3D, cuando la repulsión domina a la presión cuántica, |g|·f(ρ) − γΦ = μ dentro del núcleo. Con f lineal, el radio π·√(|g|/4πγ) no depende de la masa; con la saturación hay que resolver Lane–Emden y el radio baja algo al crecer N (un 8 % al multiplicar N por 2,4; un núcleo de Schive bajaría un factor 2,4).
- **El problema radial** (m\* = 42,3, g = −0,0074, γ = 6,5·10⁻⁶) da N = 4087, 6732 y 9727 para ρ₀ = 0,15, 0,3 y 0,6. El estado fundamental de GP–Poisson con la misma N tiene un R½ (radio de media densidad) un 1,5, 3,2 y 7,9 % mayor que el de Thomas–Fermi: la presión cuántica pesa más cuanto más satura la densidad.
- **Por qué necesita 128³.** Hace falta R/ξ ≳ 7–10, con ξ en la densidad central. En 64³ (radio lineal ~20), la energía cinética por partícula, (π/R)²/(2m\*), es del orden de la repulsión y no se forma un núcleo de Thomas–Fermi.
- **En 128³** (test 161, nativo), el perfil de Thomas–Fermi de la caja (N = 6925, borde 28,07, R½ = 14,96, los de la referencia) relaja en tiempo imaginario a R½ = 15,46 y ρ₀ = 0,242. La caminata con la gravedad estática lo conserva: R½ y ρ₀ cambian menos de un 0,2 % en 500 pasos.

**Qué prueba la gravedad análoga y qué no.**
- **Prueba** que el sonido de un fluido cuántico tiene a baja energía la cinemática de un campo relativista en una métrica efectiva: horizontes, ergorregiones, el láser, el lóbulo de Hawking y la temperatura de Unruh salen con los números de los experimentos con átomos fríos y helio. La métrica la fija el flujo, no las ecuaciones de Einstein: aquí nada hace que la materia curve esa métrica.
- **No prueba** que el vacío real sea un superfluido (Thébault, 2019). Los horizontes no tienen entropía de Bekenstein–Hawking, y el Hawking espontáneo es el vacío gaussiano de un campo clásico de campo medio, sin correcciones de muchos cuerpos.
- **Por qué no puede emerger una gravedad termodinámica.** Jacobson (1995) obtiene las ecuaciones de Einstein de δQ = T·dS en horizontes locales: necesita un flujo de energía-momento conservado y una entropía de entrelazamiento del vacío proporcional al área. Verlinde (2017) también parte de la entropía de los grados de libertad microscópicos. La caminata solo conserva la norma y es un campo clásico sin entrelazamiento; el Unruh de arriba es el de fermiones libres por el método de §2.23, no un estado de la caminata.

**La no linealidad logarítmica como prueba del criterio pre-térmico.** El vacío superfluido logarítmico (ω_nl = b·ln(E/⟨E⟩), la no linealidad de los "gausones" de Białynicki-Birula y Mycielski) diverge en E → 0, y cabía pensar que eso rompiera el régimen pre-térmico. No lo hace, porque el fondo de radiación nunca deja la densidad en 0. Según §3 (punto 7):
- con b = 0,02 (fase por paso de hasta 0,11 rad) condensa y dura como la saturable;
- con b = 0,05 (0,21–0,23 rad) se funde;
- con b = 0,1 (0,28–0,29 rad) es ruido.

Manda el mismo criterio del punto 1, el rango de la fase no lineal por paso frente a m, no la forma de la no linealidad. Es lo mismo que aquí con la resonancia paramétrica: lo que decide si el superfluido vive es una escala del paso de Floquet, no el detalle del modelo.

## Fase 12: la ecuación efectiva, aprendida de los datos (§2.29 y §3.27)

### (z) Lo que se aprende de los datos y lo que no cabe en el límite de continuo

**Qué recupera la regresión.** La caminata es un autómata, y su límite de continuo para la banda focalizante es una NLS saturable. Aquí esa ecuación sale de la evolución de la caminata sin suponer su forma. Se escribe i∂_tφ de la envolvente (la proyección en el espinor de la banda, demodulada con e^{imt}) como combinación lineal de 11 términos candidatos, y la regresión dispersa (SINDy/PDE-FIND con STRidge) se queda con los que hacen falta.
- **La masa de la banda:** m\* = 0,6146, 1,0905, 2,0586 y 5,1454 con m = 0,3, 0,5, 0,8 y 1,2, frente a dim·tan m = 0,6187, 1,0926, 2,0593 y 5,1443 (a −0,66 % como mucho).
- **El acoplamiento:** g = 0,0997–0,1000 con g = 0,1, y el signo sale con g = −0,05 (c_ρφ = +0,0497).
- **La saturación:** E_max = 0,93–0,99, del cociente de los términos ρφ y ρ²φ (la serie ρ − ρ²/E_max + …).
- **Las simetrías:** ni deriva (−i∂φ, i∂³φ) ni disipación (i∂²φ). La regresión encuentra la conservación de la norma y la simetría x → −x sin que se le digan.
- **Por qué hacen falta ρ ≪ E_max y longitudes de onda largas.** La biblioteca es polinómica. Con ρ̄ = 0,15 el campo aleatorio llega a ρ ≈ 1, donde la serie de la saturación no converge, y los coeficientes salen hasta un 50 % mal. Con k grandes, la dispersión más allá de k⁴ (que la biblioteca no tiene) contamina el coeficiente de ∂². La regresión solo encuentra lo que cabe en su biblioteca, y en el régimen en que la biblioteca es completa.

**Lo que no está en ninguna NLS estándar.** La regresión elige siempre un término (∂²ρ)·φ, una no linealidad con derivadas. Viene de que el espinor de la banda cambia con k (|∂_ku|² = 1/(8 sen²m) en la tira). La densidad que ve la fase no lineal es E = Σ_c|ψ_c|² = ρ + ½|∂_ku|²·∂²ρ, y al proyectar e^{igf}ψ en la banda vuelve a aparecer la misma corrección: la predicción es −g/(8 sen²m), más −gα/4 por el acoplamiento con los vecinos de §2.2.
- Con la biblioteca de 11 términos sale con el signo predicho y la escala 1/sen²m, a 0,70–0,83 de la predicción. El acoplamiento α = 0,5 añade −0,098·g, frente a −0,125·g.
- Restando la evolución lineal exacta y con la biblioteca cúbica completa, los coeficientes tienden a los predichos (−0,50, −0,52 y −1,10·g frente a −0,54, −0,54 y −1,09·g). La diferencia de la biblioteca reducida es la parte de esos términos que no puede separar.
- Los términos imaginarios con dos derivadas cambian según en qué punto del paso se muestrea: son el término de Baker–Campbell–Hausdorff de la división del paso (la fase no lineal y luego la moneda y el desplazamiento), no un invariante de la caminata. No se toman como predicción.

**Por qué no una red neuronal informada por la física (PINN).**
- **Como solucionador** no aporta nada: el paso de la caminata es exacto (la norma y la ley de Gauss a 10⁻¹⁵), y una PINN es una aproximación con errores de 10⁻³–10⁻⁴. Además falla justo en la dinámica oscilatoria de tiempos largos y multiescala de este modelo, porque la pérdida con el residuo de la ecuación se vuelve casi imposible de optimizar (Krishnapriyan et al., 2021).
- **Para descubrir la ecuación** necesita que se le den los términos con coeficientes libres: es esta misma regresión, pero con un optimizador no convexo y estocástico. Aquí la regresión es lineal y determinista, tarda una décima de segundo y sus veredictos son reproducibles, como los demás del laboratorio.

**Lo que ningún modelo de continuo aprendido puede contener.** Con g < 0, la ecuación aprendida es una NLS repulsiva, con espectro de Bogoliubov real y sin ninguna inestabilidad. La caminata con ese signo hierve a 0,30·c_s/ξ (4,7·10⁻³ por paso con g = −0,05) por la resonancia paramétrica de Floquet de §2.7. Esa resonancia acopla la banda focalizante con las otras bandas módulo 2π y solo existe porque el tiempo es discreto. Una ecuación de una componente en tiempo continuo, aprendida con regresión o con una red, es ciega a ella. Lo propio de la red es precisamente lo que el límite de continuo no ve. Para contenerla habría que aprender el mapa de un paso con varias componentes, no una EDP (§8).

**Qué no es.** No es física nueva: la regresión dispersa de ecuaciones es un método conocido (Brunton, Proctor y Kutz, 2016; Rudy et al., 2017), ya usado para reducir redes fotónicas no lineales a ecuaciones de continuo (Smolina et al., 2025), y que el límite de continuo de la caminata es una NLS ya se sabía. Lo que aporta es la comprobación con los números propios de la caminata y la medida de lo que no cabe en ese límite.

## Fase 13: contraste con la física real (§2.30 y §3.28)

### (aa) Qué queda del modelo como descripción del espacio-tiempo real

**Confirmar no, contrastar sí.** El prompt original propone la caminata como un tejido de cuantos de espacio-tiempo a escala de Planck. Esa afirmación no se puede confirmar con los datos de hoy: una violación de la invariancia de Lorentz detectada favorecería una red discreta, pero un resultado nulo solo empuja el paso de red Δx hacia abajo. Sí se puede contrastar, y en parte refutar, por su cinemática: con Δx = ℓ_P y c = 1 celda/paso, cada caminata predice una velocidad que depende de la energía y de la dirección, y los rayos gamma de estallidos lejanos, los electrones de la nebulosa del Cangrejo y los resonadores de laboratorio la acotan.

**La materia del prompt original queda refutada, a cualquier escala.** La banda simétrica de la caminata de Grover solo es isótropa en el régimen no relativista, k ≪ m\*. Con k ≫ m la moneda casi no mezcla los canales, cada uno viaja por su eje y la velocidad tiende a ‖n̂‖∞: 1 en los ejes, 0,707 en la diagonal de una cara y 0,577 en la del cubo (0,9999, 0,7070 y 0,5773 con k = 100·m). La anisotropía depende de k/m y no de kΔx, así que no desaparece con una red más fina. Todas las partículas conocidas tienen m ≲ 10⁻¹⁷·E_P y se observan con k ≫ m; los electrones del Cangrejo, con γ ≈ 3·10⁹, acotan la anisotropía de su velocidad a 7·10⁻¹⁵–2·10⁻¹⁷ (Altschul, 2006). La caminata de Grover predice 0,42 entre ejes y diagonales: queda refutada como materia relativista por más de 13 órdenes de magnitud. Las caminatas de tipo Dirac de la app no tienen el problema, porque su límite sin masa es el cono de Weyl.

**Qué decide si la violación de Lorentz es lineal o cuadrática.** Hay un término lineal en E/E_P si y solo si la dispersión no es simétrica, ω(**k**) ≠ ω(−**k**). Un paso palindrómico (x, y, z y luego z, y, x) de factores con T(−**k**) = T(**k**)† cumple U(−**k**) = U(**k**)†, así que ω es par y la primera corrección es cuadrática, por la misma razón por la que el Trotter simétrico solo tiene errores pares. En la app, el fotón de red, el Weyl y el Dirac alternados (también los fermiones del modelo estándar), los bombeos de Higashikawa y el motor de Wilson–Dirac tienen ω par. Solo el Weyl de orden fijo, la variante con la que se muestra la quiralidad, tiene un término lineal: ω ≈ |k| − k_x·k_y·k_z/|k|.

**Hasta qué escala sobreviven las reglas, por tiempo de vuelo.** Con las cotas de LHAASO para GRB 221009A (E_QG,1 > 10·E_P y E_QG,2 > 6·10⁻⁸·E_P):
- El fotón de red de la app es subluminal en todas las direcciones (A₂ entre −0,094 y −0,010) y solo exige Δx ≲ 7·10⁷·ℓ_P. Las demás caminatas de ω par, entre 3 y 10·10⁷·ℓ_P. Con esas reglas, una red de Planck es indistinguible del continuo: harían falta unos 8 órdenes de magnitud más de sensibilidad.
- La caminata de orden fijo, usada como fotón, quedaría en el límite (Δx < 0,63·ℓ_P con el coeficiente cuadrático medio) si el tiempo de vuelo fuera la única prueba. No lo es: ver la birrefringencia, más abajo.

**La cota de tiempo de vuelo de Mlodinow y Brun, corregida.** Su autómata de QED (Phys. Rev. D 112, 074513, 2025) tiene la misma estructura de orden fijo, con A₁ = −n_x·n_y·n_z.
- Usan como coeficiente típico √(4π/105) = 0,346, pero el promedio de (n_x·n_y·n_z)² sobre las direcciones es 1/105, y su raíz, 0,0976: falta dividir por el ángulo sólido (versión 1 de arXiv; Monte Carlo: 0,009521 frente a 1/105 = 0,009524). Con el valor correcto, su cota pasa de Δx < 0,36·ℓ_P a Δx < 1,25·ℓ_P.
- Además, esa cota depende de cómo esté orientada la red respecto a la fuente. Para orientaciones al azar, la red de Planck queda excluida solo en el 29 % de los casos, o en el 57 % con nuestro Weyl de orden fijo como fotón.
- Con tres fuentes o menos siempre hay una orientación que las pone a todas en los planos n_x·n_y·n_z = 0, así que no dan ninguna cota de tiempo de vuelo.

**El fotón de una regla de orden fijo es birrefringente, y eso lo refuta.**
- Un fotón es su propia antipartícula: E y B son campos reales. Su evolución cumple entonces U(−**k**) = U(**k**)\*, y el espectro en −**k** es el de **k** cambiado de signo. Si ω(**k**) ≠ ω(−**k**), las dos helicidades de un mismo **k** van con ω(**k**) y con ω(−**k**): el vacío es birrefringente a primer orden, con Δω = A₁·k²·Δx (A₁, el coeficiente de la velocidad de grupo).
- Mlodinow y Brun escriben que su fotón no es birrefringente porque las dos helicidades están degeneradas. Su fotón es el vector F = E + iB rotado por R_z(k_z)·R_y(k_y)·R_x(k_x), y el ángulo de esa rotación es exactamente su ec. (42). Con E y B reales, la app mide que la polarización lineal rota −3,72501·10⁻³ rad/paso en k = (0,21; 0,13; 0,07), frente a (ω(**k**) − ω(−**k**))/2 = −3,72494·10⁻³, con helicidades ±0,998 en los dos modos. El fotón de Bisio, D'Ariano y Perinotti (2016), con E y B hermíticos y ω impar, tampoco es degenerado. Y la matriz de 6×6 que publican Mlodinow y Brun (ecs. 16–17, con γ_j = σ_Z ⊗ J_j) tampoco lo es: es diag(R(**k**), R(−**k**)), con cuasi-energías ±0,25279 y ±0,26024 en ese **k**. La degeneración que afirman es un error algebraico de su propio texto.
- Es lo que dice la teoría efectiva. Myers y Pospelov (2003) ya mostraron que una dispersión cúbica del fotón independiente de la helicidad es incompatible con una teoría efectiva, y en la SME todos los coeficientes de dimensión impar son birrefringentes (Kostelecký y Mewes, 2009). La especificación dejaba como pregunta abierta "qué hipótesis de la teoría efectiva rompe el autómata"; la respuesta es que ninguna. Lo nuevo no es el principio, sino su versión de red y su aplicación a dos autómatas publicados.
- La polarización óptica de 72 fuentes acota por separado los 16 coeficientes de dimensión 5 (Kislat y Krawczynski, 2017). En la SME la separación de helicidades es 2p²·Σ Y·k, así que Σ Y·k = A₁·Δx/2. Para A₁ = −c·n_x·n_y·n_z, con la red en cualquier orientación, c·Δx < 2,45·10⁻⁶·ℓ_P (mediana 1,4·10⁻⁶). Los fotones de Mlodinow y Brun y de Bisio et al. (c = 1) exigen Δx < 2,4·10⁻⁶·ℓ_P y nuestro orden fijo, < 1,2·10⁻⁶·ℓ_P: una red de Planck queda excluida en todas las orientaciones, por unos 5,6 órdenes de magnitud. La corrección del √(4π) y el argumento de las orientaciones siguen siendo correctos, pero solo para el tiempo de vuelo, que aquí deja de ser la prueba que manda. Con las cotas de línea de visión de la polarización de rayos gamma de 5 GRB, que las *Data Tables* marcan como deducidas por razones teóricas, la cota baja a c·Δx < 3,4·10⁻¹³·ℓ_P en la peor orientación.

**Lo que sigue sin probarse: la interacción.** Las reglas palindrómicas son compatibles solo como campos libres. Con un regulador de red, las correcciones de un lazo generan violación de Lorentz de dimensión 4 que no se suprime por E/E_P (Collins et al., 2004). En QCD en red anisótropa hay que ajustar por separado la velocidad de la luz del gluón y la de los quarks. En un autómata, ninguna simetría de la red liga el paso de tiempo al espacio, así que cabe esperar lo mismo: que cada especie con interacción tenga su propia velocidad límite. Es una conjetura, no está comprobada. La app no puede ponerla a prueba, porque su no linealidad es de campo medio y no tiene un autómata de muchos cuerpos con interacción (§8, punto 11).

**Qué puede aportar el modelo como física real.**
- **Como teoría del espacio-tiempo:** un mapa de qué reglas de actualización sobreviven a las observaciones.
  - La materia de Grover no sobrevive.
  - Las de orden fijo no sobreviven: su fotón es birrefringente.
  - Las palindrómicas son compatibles e inobservables hoy, como campos libres.
- **Una contribución pequeña pero concreta a la fenomenología de los autómatas cuánticos:** el argumento de que el fotón autoconjugado de los autómatas de orden fijo publicados es birrefringente, en contra de lo que afirman Mlodinow y Brun, y su cota por polarización. La búsqueda de subsunción en contexto aislado (2026-10-25) la da como parcialmente subsumida: el principio es de Myers y Pospelov y de la SME, y la regla palindrómica es casi conocida (composiciones simétricas, Dellar 2025). Queda libre la corrección de esos dos autómatas, que da para un comentario breve.
- **Lo demás que se le pedía** (que la gravedad o el modelo estándar emerjan) ya estaba refutado por los propios resultados del modelo (§3.4 y §3.5).
- **Como física de laboratorio:** lo que propone, por ejemplo los horizontes en redes ópticas de malla, está en §2.28 y §8, y necesita un experimento o la colaboración de un grupo que los tenga.

**Corrección del 2026-10-25.** La primera versión de esta sección daba la red de Planck de orden fijo como "en el límite de LHAASO" y presentaba como aportación la corrección del √(4π) y el argumento de las orientaciones. Las dos cosas fallaban por lo mismo: el análisis solo usaba el tiempo de vuelo, el observable más débil, y aceptaba sin comprobarla la afirmación de Mlodinow y Brun de que su fotón no es birrefringente. Además, el argumento de las orientaciones ya está resuelto en la literatura para el caso birrefringente: Kislat y Krawczynski acotan los 16 coeficientes con 72 fuentes.

## Fase 14: sistemas cuánticos abiertos, complejidad y extensiones BSM

### (ab) Frontera entre el campo medio clásico y los efectos cuánticos abiertos

La fase 14 añade módulos de pocos modos exactos en `f64` y extensiones controladas de los motores existentes para estudiar cinco cuestiones abiertas de la física contemporánea (2024–2026) con el mismo rigor cuantitativo y de control nulo del resto del laboratorio (experimentos 110–115; tests 165–170).

**1. Falso vacío y bloqueo cuántico (*Cosmic Lockdown*, `src/open_quantum.rs` y `src/higgs.rs`).** Motivado por Christie, Joo, Kaplanek, Vennin y Wands (*JCAP* 09, 2026, 125), un pozo doble asimétrico $V(\phi) = \lambda(\phi^2 - v^2)^2 - \epsilon\phi$ en la base truncada de 20 autoestados exactos (paso de Strang unitario exacto + disipador GKLS de Lindblad por RK4) separa tres regímenes:
- **Control clásico:** el estado inicial del pozo falso tiene energía $E_0 = 1,4055 < V_{\text{barrera}} = 2,2103$; una partícula clásica con esa misma energía nunca cruza la barrera ($P_{\text{cl}} = 0$).
- **Túnel unitario ($\gamma = 0$):** pureza $= 1,000000$, deriva de traza $< 10^{-11}$, y el campo tunela hacia el vacío verdadero hasta $P_{\text{máx}} = 0,8727$.
- **Bloqueo cuántico (efecto Zenón):** un entorno frío con corte infrarrojo por debajo de la barrera ($L = \sqrt{2\gamma}\,\Pi_{\text{sub}}\hat\phi\Pi_{\text{sub}}$) suprime las coherencias entre pozos sin inyectar energía térmica por encima de $V_B$: $P_{\text{máx}}$ cae a $0,0727$ con $\gamma = 1$ y a $0,0389$ con $\gamma = 8$. En contraste, un baño de ruido blanco sin corte ($L = \sqrt{2\gamma}\,\hat\phi$) suprime el túnel a $\gamma$ pequeño pero a $\gamma$ muy alto calienta el sistema por encima de la barrera (escape térmico inducido por ruido).
- **Burbujas clásicas en el Higgs abeliano ($q = 0$, sesgo $-\epsilon\,\mathrm{Re}\,\phi$):** una burbuja ya nucleada con $R = 2,5 < R_c \approx 4,44$ colapsa al falso vacío con energía conservada ($|\Delta E/E| < 2\cdot 10^{-3}$), mientras que una supercrítica ($R = 9,0 > R_c$) se expande convirtiendo la malla. La nucleación por túnel pertenece al módulo cuántico abierto; la propagación posterior de la pared es dinámica clásica de campo.

**2. Función de Wigner $W(q,p)$, no gaussianidad y límite de la aproximación de Wigner truncada (`src/open_quantum.rs`).** Inspirado en Ireland y Vennin (*JCAP* 07, 2026, 042):
- **Teorema de Hudson (validez de TWA):** un estado de vacío sometido a compresión paramétrica gaussiana pura ($g_3 = g_4 = 0$, $\gamma = 0$) se comprime fuertemente manteniendo $\mathcal{N}_W = \iint |W|\,dq\,dp - 1 < 10^{-3}$: la distribución es positiva y el muestreo clásico de trayectorias (como en el experimento 103 de Hawking) es exacto.
- **Ruptura de TWA:** al encender autointeracciones cúbicas y cuárticas ($g_3 = 0,14, g_4 = 0,08$), aparecen franjas de interferencia negativas ($\mathcal{N}_W > 0,10$, $W_{\min} < -0,02$) que ningún ensamble de trayectorias clásicas positivas puede reproducir.
- **Decoherencia térmica e higiene epistémica:** encender un baño térmico ($n_{\text{th}} = 3$, $\gamma_{\text{th}} = 0,35$) borra las franjas de un estado gato de Schrödinger ($\alpha = 1,8$: $\mathcal{N}_W > 0,25 \to < 0,02$, pureza $< 0,4$). Esto muestra cuantitativamente por qué las superposiciones macroscópicas no sobreviven en entornos cálidos y húmedos —refutando invocaciones pseudocientíficas de coherencia o telepatía cuántica biológica— y a la vez recuerda el límite conceptual de la decoherencia: deja un estado mezcla impropio, pero no resuelve por sí sola el problema de la medida.

**3. Des-compresión (*de-squeezing*) y pureza frente a entrelazamiento (`src/analog.rs`).** Implementa el resultado central de Micheli, Oshima y Takahashi (*Phys. Rev. D*, 2026) en la matriz de covarianza gaussiana $4\times 4$ de dos modos $(k, -k)$:
- Bajo amortiguamiento térmico ($\gamma_{\text{damp}} = 0,8, n_{\text{th}} = 0,3$), caen simultáneamente la pureza ($\gamma < 0,6$) y la compresión efectiva ($r_{\text{eff}} < 0,75\,r_{\text{libre}}$), manteniéndose la incertidumbre de Heisenberg ($\nu_- \ge \tfrac{1}{2}$).
- **Dos estados con la misma pureza ($\gamma_A = \gamma_B$ a $< 10^{-7}$):** el estado A (compresión más fuerte + amortiguamiento) permanece entrelazado según Simon–Peres–Horodecki ($\tilde\nu_{-,A} < 0,45$, $E_{\mathcal{N},A} > 0,2$), mientras que el estado B (des-comprimido + ruido isótropo) es estrictamente separable ($\tilde\nu_{-,B} \ge 0,5$, $E_{\mathcal{N},B} = 0$). La pureza sola no mide cuán cuántico sigue siendo un estado cosmológico o de Hawking análogo.

**4. Complejidad efectiva (EFT) e índice de ensamblaje de la sopa (`src/envelope.rs` y `src/analysis.rs`).**
- **Compresión EFT:** al pasar de un régimen suave y diluido ($\bar\rho = 0,02, k_c = 0,20$: residuo $0,42\,\%$, compresión $0,339$) a uno denso y de alto $k$ ($\bar\rho = 0,25, k_c = 0,70$: residuo $1,98\,\%$, compresión $0,189$, BIC empeora en $+7,6\cdot 10^5$), la capacidad de la teoría efectiva NLS para comprimir la microdinámica cae de forma medible; y en $g < 0$ la inestabilidad paramétrica de Floquet queda fuera de toda EFT continua de una componente.
- **Proxy de ensamblaje y multiplicidad de copias:** inspirado en Sharma et al. (*Nature*, 2023) y Rutter et al. (*PNAS*, 2026), clasifica componentes conexas por tamaño, perfil radial y pasos LZ76, ponderando la multiplicidad de copias $n_c > 1$. Separa limpiamente tres regímenes que el IPR confunde: ruido inicial ($\text{IPR} < 3, A_{\text{proxy}} < 0,5$), 12 picos artificiales de 1 celda ($\text{IPR} > 20$, pero $A_{\text{proxy}} = 0$) y solitones multicelda repetidos ($A_{\text{proxy}} > 3$, copias $\ge 3$). Es un clasificador morfológico heurístico, no química ni selección biológica.

**5. Polarización del vacío dispersiva (Kramers–Kronig) y fotón oscuro $U(1)_D$ (`src/pairs.rs` y `src/em.rs`).** Motivado por la tensión dispersiva frente a QCD en red en $g-2$ del muón (Boccaletti et al., *Nature*, 2026; Davier et al., *EPJC*, 2024; Beltran et al., *JHEP*, 2026; Darmé et al., *PRD*, 2023):
- En la caminata 1D de Gupta y Short, $\Pi(\omega) = \chi'(\omega) + i\chi''(\omega)$ da apantallamiento puro por debajo del umbral ($\omega < 2m$: $\chi' > 0, |\chi''| \ll \chi'$) y absorción ligada por el teorema óptico a la creación real de pares por encima ($\omega > 2m$: $\chi'' \approx 2\Gamma_{\text{par}}/E_0^2$).
- La integral de Kramers–Kronig sobre la zona de Brillouin converge al continuo $q^2 c_{\text{ef}}/(6\pi m^2)$ ($0,8275$ frente a $0,8043$ en $m = 0,05$, razón $1,029$) y reproduce el apantallamiento estático directo de `homogeneous_sea_run` en $m = 0,10$ ($0,9508$ frente a $0,9494$, a $0,15\,\%$). El $96,7\,\%$ de $\chi'(0)$ viene de la ventana de umbral $[2m, 4m]$, por lo que un sesgo sistemático del $+4\,\%$ en esa ventana desplaza la integral dispersiva en $+3,87\,\%$ —el mecanismo matemático exacto detrás de la sensibilidad de $a_\mu^{\text{HVP}}$ a la discrepancia KLOE vs CMD-3.
- El fotón oscuro $U(1)_D$ acoplado por mezcla cinética $\epsilon$ y masa $m_{A'}$ en Maxwell 2D/3D conserva exactamente la ley de Gauss visible ($< 10^{-13}$) y la energía acoplada, produce oscilaciones $\gamma \leftrightarrow A'$ y exhibe resonancia aguda ante una antena visible en $\omega_{A'} = m_{A'}/\sqrt{1-\epsilon^2}$.

**6. Mecanismo GIM y bosón $Z'$ con cambio de sabor (`src/sm.rs` y `src/pairs.rs`).** Motivado por las anomalías $b \to s \mu^+\mu^-$ en LHCb y CMS (2026):
- En el modelo estándar clásico de 3 generaciones con matrices de Yukawa complejas aleatorias (CKM y fase de Jarlskog no triviales), los elementos fuera de la diagonal de la corriente neutra $Z^0$ entre autoestados de masa de $d_L$ y $u_L$ son $< 10^{-13}$ (cancelación GIM exacta: $W_d^\dagger g_Z W_d = g_Z I_3$). En evolución real 3D, un paquete $b_L$ bajo un campo $Z^0$ tiene fuga a $s_L$ de $< 10^{-24}$.
- Al activar el campo externo BSM $Z'$ ($\theta_{bs} = 0,04$), la conversión $b_L \to s_L$ sigue exactamente $\sin^2(\theta_{bs}\,t)$ (error $< 10^{-6}$), y en el laboratorio de pares 3D la razón de desintegraciones cumple $\Gamma(Z' \to b_L\bar s_L + s_L\bar b_L)/\Gamma(Z' \to \mu_L^+\mu_L^-) = 2,1621$ frente al valor teórico $6\,g_{bs}^2/g_\mu^2 = 2,1600$ ($0,10\,\%$). Límite explícito: esto modela solo la estructura a nivel árbol de una hipótesis $Z'$; la gran incógnita de los datos de LHCb/CMS —los lazos de encanto de QCD no perturbativa a un lazo— queda fuera de un autómata clásico a nivel árbol.

## Fase 15: QFT de muchos cuerpos (MPS), espacio-tiempo covariante y problemas de frontera

### (ac) Muchos cuerpos cuánticos, violación radiativa de Lorentz, conjuntos causales y fronteras abiertas (§8)

La fase 15 da el salto más allá del campo medio clásico en 1+1D y ataca directamente las limitaciones y problemas abiertos señalados en §8 y en el cierre de las fases 12 y 13 (`src/mps.rs`, `src/spacetime.rs`, `src/frontier.rs` y ejecutable nativo multihilo `src/bin/qsu_cli.rs`; experimentos 116–121; tests 171–176).

**1. QFT de muchos cuerpos en 1+1D mediante Matrix Product States (`src/mps.rs`): modelo de Schwinger cuántico.** Mediante TEBD de 2.º orden (Strang), un SVD complejo de Jacobi-Hestenes en `f64` sin dependencias externas y un MPO exacto de flujo eléctrico ($D = 5$, $E \in \{-2,-1,0,+1,+2\}$) con compresión *zip-up* que impone la ley de Gauss cuántica $E_n = \sum_{j \le n}[\frac{1}{2}(\sigma^z_j + (-1)^j) + q_{\text{ext},j}]$:
- **Confinamiento cuántico real ($L = 12, w = 0,5, m = 0,6, J = 0,5, \chi = 16$):** el potencial estático quark-antiquark $V(d) = E_0(d) - E_0(0)$ crece linealmente con la separación: $V(2) = 0,9404$, $V(4) = 1,8643$, $V(6) = 2,7839$ (tensión efectiva $\sigma \approx 0,464$, ligeramente renormalizada por polarización del vacío cuántico respecto a $J = 0,5$), manteniendo $E_{\text{centro}}(4) = 0,8401$ y aumentando el entrelazamiento de von Neumann $S_{\text{vN}}(0) = 0,2050 \to S_{\text{vN}}(4) = 0,4397$.
- **Rotura cuántica de la cuerda en tiempo real ($L = 12, d = 6, w = 0,8, m = 0,15, J = 0,9, \chi = 16$):** el campo eléctrico central cae de $+1,000$ a un mínimo de $0,105$ (final $0,300$) al producirse pares $e^+e^-$ del vacío, mientras la entropía de entrelazamiento en el corte central crece de $0$ hasta $S_{\text{vN},\max} = 1,2157$ con conservación de la energía total ($E_0 = 4,5000 \to E_{\text{fin}} = 4,4693$, deriva $0,68\,\%$).
- **Límite explícito:** esto es QFT genuina de muchos cuerpos en 1+1D (en el espíritu de Pichler et al., *PRX* 2016; Bañuls et al., 2020), no 3+1D: en 3+1D el crecimiento del entrelazamiento hace inviable el MPS clásico en tiempo real.

**2. Violación de Lorentz radiativa a un lazo (Collins et al., 2004) y cono de luz interactuante en MPS (`src/frontier.rs` y `src/mps.rs`).** Resuelve cuantitativamente la prueba pendiente §8.11 de la Fase 13:
- **Autoenergía exacta a un lazo en teoría Yukawa 1+1D ($m_f = 0,25, M_\phi = 1,2, g_Y = 0,5$):** con regulador continuo covariante de Lorentz, $|\delta c^2_{\text{cont}}|$ decrece con el corte $\Lambda$ ($4,965\cdot 10^{-5}$ en $\Lambda = 20 \to 1,381\cdot 10^{-6}$ en $\Lambda = 120$, control nulo). Con regulador de red discreta (Wilson–Dirac $r = 1$ o ingenuo $r = 0$), los modos UV de la zona de Brillouin generan $\delta c^2_{\text{red}} = -4,0964\cdot 10^{-2}$ ($r = 1$) y $-1,0395\cdot 10^{-2}$ ($r = 0$), casi $3\cdot 10^4$ veces por encima del control continuo, con escalado exacto $\propto g_Y^2$ ($\delta c^2(1,0)/\delta c^2(0,5) = 4,00000$) y **sin supresión por $E/E_P$** (operador marginal de dimensión 4). Una segunda especie con acoplamiento distinto adquiere una velocidad límite diferente ($\Delta c = 2,07\cdot 10^{-2}$), ilustrando el problema de ajuste fino (*fine-tuning*) de toda red fija frente a las cotas experimentales $|\Delta c/c| < 10^{-20}$.
- **Velocidad renormalizada en MPS XXZ ($L = 16, w = 0,5, \chi = 16$):** inyectando una excitación fermiónica localizada $\sigma^+_{\text{centro}}$ sobre el vacío interactuante, la interacción repulsiva $\Delta = 0,22$ ($\Delta_{\text{XXZ}} = 0,88$, $v_{\text{Bethe}} = 1,5074$ frente a $1,0000$ libre) acelera la velocidad balística medida $\sqrt{2}\,r_{\text{rms}}$ en $+16,4\,\%$ ($1,7255 \to 2,0080$).

**3. Espacio-tiempo discreto covariante: conjuntos causales y grafo dinámico (`src/spacetime.rs`).**
- **Conjunto causal 1+1D ($N = 200$, boost $\eta = 0,8$, 24 realizaciones):** el grafo causal es **exactamente invariante** bajo el boost ($\Delta\text{enlaces} = 0$, mientras una red cúbica fija se distorsiona en un factor $e^{2\eta} = 4,95\times$); la dimensión fractal de Myrheim–Meyer recupera $d_{\text{MM}} = 1,9924$ (exacto $2$) únicamente del orden parcial; y el d'Alembertiano discreto no local de Sorkin–Benincasa–Dowker $B^{(2)}_\epsilon$ ($\epsilon = 0,35$) sobre $\phi = \frac{1}{2}(t^2 - x^2)$ (exacto $\Box\phi = 2$) da $1,578$ en reposo y $1,578$ exactamente tras el boost (diferencia $< 10^{-14}$). El precio físico es también medible: el grado medio de enlaces causales es $7,70$ y crece como $\sim \ln N$ (no localidad intrínseca a lo largo del cono de luz).
- **Caminata cuántica sobre grafo dinámico ($L = 96$):** con ángulos de moneda locales $\theta(x, t)$ que reaccionan a la densidad de materia ($\lambda_g = 0,85, \gamma_g = 0,35$), la caminata mantiene **unitariedad exacta** ($|\Delta\text{norma}| = 5,55\cdot 10^{-15}$) mientras la masa central reduce la velocidad local de la luz ($c = 0,969 \to 0,470$) y produce un retraso de Shapiro de $4,48$ celdas ($x_{\text{plano}} = 44,16 \to x_{\text{curvo}} = 39,68$). Límite explícito: es un modelo de juguete escalar de métrica emergente, no relatividad general tensorial de espín 2.

**4. Obstrucción de dobladores en caminatas 3D de tipo luz (§8.6, `src/frontier.rs`).** Escaneando la zona de Brillouin 3D con refinamiento local en 4 niveles alrededor del mínimo fuera de $|k| < 0,35$ (y paralelismo nativo `std::thread::scope` sobre 124 combinaciones $(\theta_x, \theta_y, \theta_z)$):
- Weyl 3D estándar ($\theta = 0$): dobladores exactos ($\varepsilon_{\min} = 0,0000$, $c = 1,000$).
- Gupta–Short 3D simétrico ($\theta = 0,4$): el refinamiento revela un cero espurio casi exacto ($\varepsilon_{\min} = 0,0009$, $c = 0,921$).
- Mejor candidato en todo el barrido 3D con $c > 0,25$: $\varepsilon_{\min} = 0,0633$ en $\theta = (0,75; 0,25; 0,75)$ ($c = 0,732$), más de $20\times$ inferior al hueco limpio de Wilson–Dirac ($\approx 1,31$ en Fase 10) —evidencia numérica directa de la obstrucción topológica de Nielsen–Ninomiya en caminatas locales 3D de 2 componentes.

**5. Aprendizaje del mapa discreto multicomponente de Floquet (§8.10, `src/frontier.rs`).** Frente a la ceguera de la EDP continua aprendida en la Fase 12 (que daba tasa $0$ para $g < 0$), ajustar directamente el mapa discreto de un paso sobre los 4 canales y 3 sitios vecinos más el polinomio de fase $\theta(E)$ recupera la matriz de moneda con error $4,93\cdot 10^{-9}$, $\theta'(\bar\rho = 0,5)$ al $0,61\,\%$, y predice la inestabilidad paramétrica de Floquet con exactitud: tasa aprendida $4,4826\cdot 10^{-3}$ frente a exacta $4,5224\cdot 10^{-3}$ (razón $0,9912$) en el número de onda exacto $q_* = 1,160$.

**6. Fermiones CPT-impares y umbral de Cohen–Glashow (§8.11, `src/frontier.rs`).** Para la caminata de Weyl de orden fijo ($A_{1,\text{rms}} = 2/\sqrt{105} = 0,19518$), el canal Cherenkov en el vacío $\nu \to \nu e^+ e^-$ (Cohen y Glashow, *PRL* 107, 181803, 2011) abre a escala de Planck ya en $E_{\text{th}}(\ell_P) = 40,28\text{ TeV}$, de modo que los neutrinos astrofísicos de IceCube ($6\text{ PeV}$) imponen $\Delta x/\ell_P < 3,025\cdot 10^{-7}$ y el evento KM3-230213A de KM3NeT ($220\text{ PeV}$) impone $\Delta x/\ell_P < 6,136\cdot 10^{-12}$. Las reglas palindrómicas anulan la asimetría CPT exactamente ($0,00$) a nivel árbol, remitiendo toda cota residual a los bucles interactuantes del punto 2.

## Fase 16: plataforma de investigación en física fundamental

### (ad) DMRG variacional, flujo RG de Lorentz, gravedad cuántica discreta, Yang–Mills 3+1D cuántico y catálogos reales

La fase 16 dota al proyecto de las herramientas cuantitativas necesarias para atacar problemas abiertos reales de física fundamental (`src/dmrg.rs`, `src/qgravity.rs`, `src/noncubic_pheno.rs` y extensión del CLI nativo multihilo `src/bin/qsu_cli.rs`; experimentos 122–127; tests 177–182).

**1. DMRG variacional de dos sitios (Lanczos + MPO exacto) y restauración infrarroja de Lorentz (`src/dmrg.rs`).**
- **Escalado al continuo y carga central CFT:** sobre la cadena crítica interactuante XXZ en el punto de Razumov–Stroganov ($\Delta = 1/2$, $K = 3/4$, $L \in \{16, 24, 32, 48\}$, $\chi = 32$), la extrapolación de tamaño finito $e(L) = e_\infty + b/L + c/L^2$ da $e_\infty = -0,374988$ frente a la solución exacta de Bethe $-3/8 = -0,375000$ (error relativo $3,20\cdot 10^{-5}$), y el escalado de Calabrese–Cardy en el corte central promediado ($L = 24\dots 48$) recupera $c_{\text{CFT}} = 0,934$ (frente a $c = 1$). (En Heisenberg XXX $\Delta = 1$, el mismo DMRG acierta la energía exacta de Bethe $\frac{1}{4} - \ln 2 = -0,443147$ con error $3,18\cdot 10^{-5}$, pero muestra la conocida supresión logarítmica marginal $SU(2)$ de $c_{\text{ef}}(L=48)$ descrita por Affleck, Laflorencie y Sørensen, 2009.)
- **Problema de Collins et al. (2004) frente al flujo RG de Chadha–Nielsen (1983):** en la escalera DMRG de dos especies acopladas ($v_A = 1,0, v_B = 0,5$), el acoplamiento $g_{AB} = 0,8$ liga las especies ($\Delta E_{\text{lig}} = 0,2595$, $\langle S^z_A S^z_B \rangle = 0,0000 \to -0,1630$). Al integrar el flujo RG exacto a un lazo desde $M_P = 1,22\cdot 10^{19}\text{ GeV}$ hasta $m_e = 5,11\cdot 10^{-4}\text{ GeV}$ ($\Delta\ln\mu = 51,53$), QED débil ($\alpha = 1/137$) solo reduce $\delta c$ en un factor $0,7170 = (\alpha_{\text{IR}}/\alpha_{\text{UV}})^4$ —insuficiente en $> 18$ órdenes de magnitud frente a $|\Delta c/c| < 10^{-20}$— mientras que un sector fuertemente acoplado requiere una dimensión anómala $\gamma_* \ge \gamma_{\min} = 0,818$ ($\gamma_* = 0,90$ suprime $\delta c$ hasta $1,50\cdot 10^{-22}$).

**2. Gravedad cuántica discreta y teoría gauge cuántica 3+1D (`src/qgravity.rs`).**
- **Monte Carlo dinámico de conjuntos causales 2D (Surya, *CQG* 2012):** muestreando $\Omega_{2D}$ ($N = 28\text{–}32$) con la acción exacta de Benincasa–Dowker $S_{\text{BD}} = N - 2N_1 + 4N_2 - 2N_3$ y peso $\exp(+\beta S_{\text{BD}})$ mediante máscaras de bits `u64::count_ones()`, reproduce las tres fases: variedad 2D de Minkowski en $\beta = 0$ ($S/N = +0,030, d_{\text{MM}} = 1,991, R = 0,503$), transición a fase cristalina de capas dominada por $N_2$ en $\beta = +2,0$ ($S/N = +9,353$) y colapso a fase cristalina bipartita en $\beta = -0,25$ ($S/N = -10,374, N_1/N = 3,31$).
- **Yang–Mills cuántico $SU(2)$ en 3+1D euclídeo (Creutz 1980; Kennedy–Pendleton 1985):** el baño térmico exacto en malla $4^4$ ($6^4$ en `qsu_cli`) cierra la ausencia de confinamiento cuántico 3+1D señalada en §2.11: reproduce $\langle P \rangle = 0,2441$ en $\beta = 1,0$ (fuerte $0,2396$), $\langle P \rangle = 0,7238$ en $\beta = 3,0$ (1-lazo débil $0,7500$) y mide confinamiento por ley de área con razón de Creutz $\chi(2,2) = 1,203$ ($\beta = 1,5$) $\to 0,620$ ($\beta = 2,0$).
- **Espín 2 frente a modelos escalares y obstrucción de Weinberg–Witten (1980):** verifica la transversabilidad y traza nula de $h_+, h_\times$ (`0,0`), la rotación exacta de espín 2 a $+45^\circ$ (`2,2e-16`), la razón de deflexión de la luz $2\times$ (Einstein) frente a $0\times$ (Nordström) y $1\times$ (refractivo puro), y la proyección idénticamente nula (`1,8e-16`) de un tensor covariante $T^{\mu\nu}$ sobre estados sin masa de helicidad $|h| = 2$.

**3. Redes 3D no cúbicas, verosimilitud sobre catálogos reales y Kaplan–Sen 2024 (`src/noncubic_pheno.rs`).**
- **Redes no cúbicas 3D:** demuestra que BCC de orden fijo también es CPT-impar ($|\omega(k)-\omega(-k)| = 6,42\cdot 10^{-3}$), cuantifica la anisotropía $O(k^2\Delta x^2)$ de SC ($0,0277 = 1/36$), BCC ($0,0553 = 1/18$) y FCC ($0,0138 = 1/72$), y demuestra que la combinación palindrómica isotrópica $\frac{1}{3}\text{SC} + \frac{2}{3}\text{FCC}$ cancela exactamente la anisotropía direccional en $k^2\Delta x^2$ hasta $1,11\cdot 10^{-6}$ (empujándola a $O(k^4\Delta x^4) = 0,0007$).
- **Perfil $\chi^2$ sobre catálogos astrofísicos reales:** usando las 16 cotas de la Tabla IV de Kislat y Krawczynski (2017) minimizadas sobre $SO(3)$, el fotón de orden fijo de Mlodinow–Brun queda excluido con **$\Delta x < 1,381\cdot 10^{-6}\,\ell_P$ al $95\,\%$ CL** (mediana $9,407\cdot 10^{-7}\,\ell_P$) y **$\Delta x < 3,523\cdot 10^{-6}\,\ell_P$ a $5\sigma$ en cualquier orientación**; usando el extremo conservador al $90\,\%$ CL de KM3-230213A ($110\text{ PeV}$), el Weyl de orden fijo exige $\Delta x < 4,909\cdot 10^{-11}\,\ell_P$; e integra $K_1(\text{GRB 221009A}) = 7,157\cdot 10^{16}\text{ s}$.
- **Fermiones quirales en frontera finita (Kaplan y Sen, *PRL* 2024; Golterman y Shamir, 2025):** en una losa abierta 4+1D ($L = 14, m = 0,45$), aparece un único modo Weyl en $s = 0$ con $\omega = 0,14944 = \sin(0,15)$, $\langle\gamma^5\rangle_L = +0,9994$, decaimiento exponencial $0,3150 \approx (1-m)^2 = 0,3025$ y hueco $1,583$ en los 7 dobladores $k = (\pi, 0, 0)$, mientras la frontera opuesta $s = L-1$ porta $\langle\gamma^5\rangle_R = -0,9994$ con suma neta $-3,11\cdot 10^{-15}$.

*Límite honesto:* el motor provee ahora toda la infraestructura de cálculo de frontera (DMRG, Monte Carlo cuántico 4D, conjuntos causales dinámicos y verosimilitud de catálogos); convertirlo en un descubrimiento de física fundamental publicado exige fijar una única pregunta abierta (p. ej. escalado $L \to \infty$ de una nueva familia de teorías gauge quirales o de restauración RG fuerte) y someter el análisis a revisión por pares.

## Fase 17: tres estudios numéricos de preguntas abiertas

### (ae) Trilema de Lorentz en dos especies, QCA 3+1D isotrópico de orden $n = 4$ y retroacción cuántica en conjuntos causales 2D

La fase 17 utiliza la infraestructura de la fase 16 para abordar de forma cuantitativa, reproducible y con controles nulos tres preguntas abiertas concretas (`src/studies.rs`, `src/tests/phase17.rs`, subcomandos `phase17`/`study1`/`study2`/`study3` en `src/bin/qsu_cli.rs`, experimentos 128–130 en `index.html`, tests 183–185 e informe completo en [`estudio_preguntas_abiertas.md`](estudio_preguntas_abiertas.md)).

**1. Estudio 1 — Restauración infrarroja de Lorentz vs. hueco de masa en dos especies cuánticas acopladas.** Combinando diagonalización exacta de Lanczos en escalera periódica ($N_r \in \{4, 6, 8\}$, hasta $12\,870$ estados, cero error de truncación) para el hueco extrapolado $\Delta_\infty$ y el factor de estructura dinámico infrarrojo $S_{\alpha\alpha}(q = \pi/4, \omega)$, evolución exacta por RK4 de la carga de sabor $\Delta N_B(t)$ en $N_r = 6$ y entropía DMRG $S(8) \to S(16)$ ($\chi = 24$), sobre $v_A = 1,00$, $v_B = 0,45$ y $\Delta = 0,25$:
- **Control ($g = 0$):** $\Delta_\infty = 0,0000$, $S: 1,216 \to 1,353$ ($\Delta S = +0,137$), $v_A^{\text{IR}} = 1,137$ y $v_B^{\text{IR}} = 0,587$ (acuerdo al $1,4\text{–}2,0\,\%$ con las velocidades exactas de Bethe $1,154$ y $0,599$), desajuste $0,639$ y $\max|\Delta N_B| = 1,20\cdot 10^{-14}$.
- **Canal 1A (densidad–densidad $g_z = 0,60$, $U(1)_A \times U(1)_B$ conservado):** $\Delta_\infty = 0,0494$, $\Delta S = +0,115$, $\max|\Delta N_B| = 6,22\cdot 10^{-15}$ (conservación exacta), pero **no iguala las velocidades**: $v_A^{\text{IR}} = 1,172$, $v_B^{\text{IR}} = 0,626$, desajuste $0,607$.
- **Canal 1B (intercambio isótropo de Heisenberg $g_{xy} = g_z = 1,40$):** abre un **hueco de masa real** ($\Delta_\infty = 0,7441$) con **ley de área exacta** ($S(8) = 0,217 \to S(16) = 0,217$, $\Delta S = 0,000$).
- **Canal 1C (mezcla de sabor sin hueco $g_{xy} = 0,75, g_z = 0$):** permanece crítico sin hueco ($\Delta_\infty = 0,0000$, $\Delta S = +0,058$) e **iguala exactamente ambas velocidades infrarrojas** ($v_A^{\text{IR}} = v_B^{\text{IR}} = 0,818$, desajuste $0,000$, valor intermedio coherente con la media armónica renormalizada), pero al precio de romper $U(1)_A \times U(1)_B$ ($\max|\Delta N_B| = 0,7321$). *Conclusión:* trilema no perturbativo en 1+1D que muestra por qué es difícil igualar velocidades límite entre sectores sin mezcla directa (como fotón y electrón) sin abrir un hueco de masa.

**2. Estudio 2 — Clasificación triple de 8 familias QCA 3+1D y construcción isotrópica de orden $n = 4$.** Evaluando simultáneamente (i) asimetría CPT $|\omega(\mathbf{k}) - \omega(-\mathbf{k})| < 10^{-10}$, (ii) ausencia de dobladores fuera de $|\mathbf{k}| < 0,35$ ($\varepsilon_{\min} > 0,30$ en rejilla $22^3$ + 4 niveles de refinamiento) y (iii) isotropía cuadrática $\text{spread}(A_2) < 10^{-4}$ sobre 400 direcciones:
- Ninguna de las 5 familias locales de 2 componentes (`SC/BCC/FCC-Weyl`, orden fijo o palindrómico: $\varepsilon_{\min} \le 0,0024$), ni el fotón de 6 componentes de Mlodinow–Brun (`asym = 7,45e-3`, `gap = 0,0000`), ni `SC-Wilson-Dirac` (`gap = 0,3515`, pero `spread A₂ = 0,111`) supera las tres condiciones.
- **Nueva construcción (Familia 7, `Iso-SC+FCC-WD n=4`, 4 componentes):** el paso unitario palindrómico de Strang con esténcil isotrópico $S^{\text{iso}}_a(\mathbf{k}) = \frac{1}{3}\sin k_a(1 + \cos k_b + \cos k_c)$ y masa de Wilson $M_W(\mathbf{k}) = r\sum_a(1 - \cos k_a)$ en el valor óptimo $r = 2/\sqrt{3}$ con $\Delta t = 0,25$ satisface $\cos(\omega\Delta t) = \cos(\Delta t M_W)\cos(\Delta t |\mathbf{S}^{\text{iso}}|)$, dando `asym = 0,00`, `gap = 0,3500` (cero dobladores), `spread A₂ = 1,83×10⁻⁵`, `⟨A₂⟩ = +4,02×10⁻⁶` y `A₄_max = 0,0113` (**invariancia de Lorentz exacta de orden $n = 4$** para $\Delta t$ finito). *Precio explícito:* rompe la simetría quiral por el término de Wilson y no elimina los corrimientos radiativos $n = 0$ de Collins et al. al añadir interacciones.

**3. Estudio 3 — Conjuntos causales dinámicos 2D acoplados a materia cuántica.**
- **Propagador retardado exacto de Johnston (*PRL* 2009):** en 10 espolvoreados 2D de Minkowski ($N = 140, m = 0,8$), $\langle G_R(\tau)\rangle$ reproduce $\frac{1}{2}J_0(m\tau)$ con `err_rms = 0,0105`, exponiendo a la vez la varianza intrínseca no local del conjunto causal ($\sigma_G(\text{cono}) = 0,0425$, $\sigma_G(\text{bulk}) = 0,1636$).
- **Retroacción de $N_f = 0, 1, 2$ campos cuánticos ($S_{\text{mat}} = \frac{1}{2}\ln\det(I + \Delta^T\Delta)$ por Cholesky exacto en `qsu_cli`, $N = 20$, 600 barridos, 6 semillas):** en $\beta = 0$ la fase de variedad se mantiene en $d_{\text{MM}} = 2,02 \pm 0,01 \to 2,04 \pm 0,01$; en la fase cristalina de capas ($\beta > 0$) la materia penaliza cadenas profundas elevando $d_{\text{MM}}$ ($1,75 \pm 0,08 \to 2,12 \pm 0,06$ en $\beta = 1,40$, $> 3,5\sigma$) y reduciendo $\langle S_{\text{BD}}\rangle/N$ en $\beta = 0,45$ ($+9,76 \pm 0,75 \to +8,25 \pm 0,38$); en la fase bipartita poco profunda ($\beta = -0,25$) ahonda ligeramente la acción ($-6,93 \pm 0,11 \to -7,49 \pm 0,25$). *Límite honesto:* el estudio es 2D, finito ($N \le 20$) y muestra metaestabilidad de Metropolis simple a $\beta \ge 0,90$ (`SEM ~ 0,5–0,8`), por lo que fijar $\beta_c(N_f)$ en el límite termodinámico requeriría *parallel tempering*.

## Fase 18: profundización a nivel artículo en las tres vías (Vías A, B y C)

### (af) Circuito 3D real $n = 4$ con Gauss exacta, *Parallel Tempering* hasta $N = 48$ y confinamiento gauge en Schwinger de dos sabores

La fase 18 lleva cada una de las tres líneas de investigación más allá del nivel piloto mediante infraestructura estadística y de exportación reproducible ([`src/research_io.rs`](src/research_io.rs), archivos CSV/JSON en `datos_investigacion/`), tres módulos especializados (`src/via_a_qca4.rs`, `src/via_b_causal_pt.rs`, `src/via_c_schwinger2f.rs`), tests `t186`–`t188`, experimentos interactivos `131`–`133` en `index.html`, subcomandos `phase18`/`via-a`/`via-b`/`via-c` en `src/bin/qsu_cli.rs` y tres manuscritos borrador independientes:

**1. Vía A — QCA 3+1D isotrópico $n = 4$ en espacio real con $U(1)$ dinámico y fotón `Yee-4 S₄` (`src/via_a_qca4.rs`).**
- **Espacio real $6^3$ con compuertas estrictamente locales de 2 sitios (3 SC + 6 diagonales FCC) y Maxwell dinámico:** mediante caminos en L simétricos sobre enlaces cúbicos para las diagonales FCC y relajación amortiguada de Gauss inicial, 6 pasos completos de Yoshida–Suzuki $S_4$ conservan la norma (`4,76e-14`), la **ley de Gauss (`1,29e-14`)** y la **invariancia gauge $U(1)$ local** (`1,37e-16` en densidad, `1,98e-13` en $\mathbf{E}$).
- **Descubrimiento algebraico en la factorización local ($S_2$ vs. $S_4$, Bloch exacto $32\times 32$):** separar el término $+3r\beta$ de los enlaces SC renormaliza $c$ en $O(\Delta t^2 r^2)$ incluso en $k \to 0$; embeber $\frac{r}{2}\beta$ en cada compuerta SC y subir de $S_2$ a $S_4$ reduce $|c_{\text{ef}}-1|$ en **$17,5\times$** (`2,26e-3 → 1,29e-4`) y $|\langle A_2\rangle|$ en **$84\times$** (`-0,1565 → -1,86e-3`, `spr = 6,06e-4`, hueco BZ `2,3095`, CPT `7,22e-15`).
- **Auditoría de 5 fotones 3+1D:** Mlodinow–Brun (2025), Bisio–D'Ariano–Perinotti (2016) y Eon et al. fijo (2023) son birrefringentes (`asym = 7,45–8,17e-3`), Eon et al. palindrómico es $n = 2$ anisótropo (`spr = 2,78e-2`), y el nuevo **fotón real escalonado `Yee-4 S₄`** pasa todas las pruebas de orden $n = 4$ (`asym = 0`, `spr = 1,04e-5`, `⟨A₂⟩ = -6,64e-6`, `corner_gap = 2,314`).

**2. Vía B — *Parallel Tempering* en conjuntos causales 2D con Sorkin–Johnston e Ising hasta $N = 48$ (`src/via_b_causal_pt.rs`).**
- El intercambio de réplicas sobre 12 temperaturas $\beta \in [-0,25; +1,35]$ elimina la metaestabilidad de la Fase 17 (`swap_acc = 0,12–0,31`, $\tau_{\text{int}} = 0,5\text{–}0,8$ en $\beta = 0$, barras *block bootstrap* $\pm 0,01$ en $d_{\text{MM}}$).
- **Contraste de signo opuesto entre materia cuántica y clásica en $\beta = 0$:** mientras los espines clásicos de Ising sobre enlaces ($J = 0,45$) deprimen fuertemente $d_{\text{MM}}$ hasta `1,746–1,840` ($> 8\sigma$ por debajo de 2) con fuerte frenado crítico ($\tau_{\text{int}} = 17\text{–}71$), la acción cuántica de 1 lazo de Sorkin–Johnston ($N_f = 1, 2$) produce solo un pequeño desplazamiento positivo que **se atenúa con $N$** (`2,052` en $N=20 \to 2,013 \pm 0,011$ en $N=48$), estabilizando la fase de variedad 2D y preservando el pico agudo de susceptibilidad ($\chi_{\text{pico}}$ crece hasta `563` en $N = 48$).

**3. Vía C — Restauración infrarroja de Lorentz por confinamiento gauge en Schwinger 2 sabores + DMRG $U(1)$ por bloques (`src/via_c_schwinger2f.rs`).**
- **Rompiendo el trilema del Estudio 1:** acoplando dos especies ($v_A = 1,00, v_B = 0,45$) con cargas opuestas ($Q_r = n_{A,r} - n_{B,r}$, conservación **exacta $U(1)_A \times U(1)_B$**, cero mezcla de sabor) al mismo campo eléctrico cuántico $J\sum E_r^2$, el sector exacto hasta $N_r = 10$ (**63 504 estados**) muestra que **para todo $J > 0$ ambas velocidades infrarrojas se igualan exactamente** (`mismatch = 0,7586` en $J = 0 \to 0,0000$ en $J = 0,08\text{–}0,50$ y `0,0008` en $J = 1,0$), acercándose en $J$ débil a la predicción analítica de bosonización $\sqrt{v_A v_B} = 0,6708$ (`0,548 → 0,570` al subir $N_r = 8 \to 10$) y suavizándose como $\propto v_A v_B / J$ en acoplamiento fuerte mientras el modo cargado adquiere la masa del bosón de Schwinger ($\omega_{\text{ch}}: 0,128 \to 1,279$).
- **Motor DMRG por bloques de carga $U(1)$ + MPO exacto $D = 7$ sin corte del campo eléctrico:** coincide con Lanczos exacto en $N_r = 6$ a **`5,54×10⁻¹⁰`** y mide en cortes pares $N_r = 8 \to 16$ ($\chi = 48$) la caída esperada de entropía (`1,416 → 0,875`, $-38\,\%$) y de carga central efectiva (`c_eff: 2,39 → 1,55`) debida al confinamiento del modo cargado.

## Fase 19: cierre de los puntos pendientes en las tres vías (Vías A, B y C)

### (ag) Extensión quiral 4+1D $n = 4$, *Parallel Tempering* adaptativo (20 réplicas + *checkpoints* hasta $N = 64$) y límite continuo / brecha BKT en Schwinger 2 sabores

Tests `t189`–`t191` (`src/tests/phase19.rs`), experimentos interactivos `134`–`136` en `index.html`, subcomandos `phase19`/`via-a-chiral`/`via-b-fss`/`via-c-scaling` en `src/bin/qsu_cli.rs` y exportación de datos en `datos_investigacion/` (incluidos 20 *checkpoints* JSON en `datos_investigacion/checkpoints/`):

1. **Vía A — Frontera quiral 4+1D de orden $n = 4$ y verificación bibliográfica (`via_a_chiral_kaplan_sen_n4`, Exp 134):**
   - Incrustar el núcleo isotrópico SC+FCC y el núcleo de cuarto orden Symanzik-4 ($\tilde S_a = \frac{4}{3}\sin k_a - \frac{1}{6}\sin 2k_a$) en la losa finita abierta 4+1D de Kaplan–Sen ($L_s = 14, m = 0,55$) reduce la anisotropía direccional $O(k^2)$ del modo Weyl de frontera en **`4 170×`** (`1,111e-1 → 2,665e-5` en Iso-SC+FCC) y anula ambos `spread(A₂) = 1,895e-4` y `⟨A₂⟩ = -1,065e-4` en Symanzik-4 con cero dobladores (`gap_BZ = 1,483`), manteniendo intacta la cancelación entre fronteras opuestas de Golterman–Shamir ($\langle\gamma^5\rangle_L + \langle\gamma^5\rangle_R = -3,11\times 10^{-15}$).
   - Verificación bibliográfica primaria completada sobre Mlodinow–Brun (*PRD* 112, 074513 / arXiv:2506.20136, afirmación textual de ausencia de birrefringencia junto a la Ec. (48)), Dellar (*PRE* 2011; *JCP* 533, 113991, 2025, QLA 1D de 9 etapas) y Eon et al. (*Quantum* 7, 1179, 2023).
2. **Vía B — *Parallel Tempering* adaptativo de 20 réplicas + *checkpoints* hasta $N = 64$ (`via_b_adaptive_fss`, `qsu_cli via-b-fss`, Exp 135):**
   - La redistribución adaptativa por resistencia logarítmica $(-\ln p_r)^{0,75}$ con $\beta = 0$ fijado eleva `swap_acc` a `0,49 → 0,21–0,24` en $N = 20 \to 64$.
   - El ajuste de tamaño finito sobre 5 tamaños ($N \in \{20, 28, 36, 48, 64\}$) revela que gravedad pura (`d_∞ = 1,9968 + 0,38/N`), SJ $N_f = 1$ (`1,9974 + 0,88/N`) y SJ $N_f = 2$ (`1,9938 + 1,33/N`) convergen todos a **$d_{\text{MM}}(\infty) = 2,00$** con un desplazamiento finito $O(N_f/N)$ y $\tau_{\text{int}} = 0,5\text{–}0,7$, mientras Ising sobre enlaces ($J = 0,45$) colapsa a **$d_{\text{MM}}(\infty) = 1,6115$** ($\tau_{\text{int}} = 52\text{–}104$). La transición principal de Surya deriva suavemente hacia $\beta_c(\infty) \approx 0,06$ con $\chi_{\text{pico}} \propto N^\gamma$ ($\gamma \approx 0,9\text{–}2,0$) y mínimos negativos de Binder.
3. **Vía C — Trayectoria de continuo ($M_S L = \text{const}$), escalado CFT hasta $L = 64$, brecha BKT inducida por $v_A \neq v_B$ y contraste 3+1D Coulomb (`via_c_continuum_and_scaling`, `qsu_cli via-c-scaling`, Exp 136):**
   - A masa física de Schwinger constante ($J = 8/N_r^2$ en $N_r \in \{6, 8, 10\}$), $v_{\text{IR}}$ asciende monótonamente `0,5326 → 0,5596 → 0,5699` con `mismatch = 0,00000` (`v_∞ = 0,6281`, $R^2 = 0,9877$ hacia $\sqrt{v_A v_B} = 0,6708$).
   - El escalado Calabrese–Cardy sobre $N_r \in \{8, 12, 16, 24, 32\}$ ($L = 16\text{–}64$, $\chi = 56$) confirma la reducción exacta a la mitad de la carga central (`c_hi(J=2,5, v_A=v_B) / c_hi(J=0) = 1,159 / 2,329 = 0,498` en $N_r = 16 \to 24$; $J = 1,0$ converge monótonamente `1,58 → 1,50 → 1,42 → 1,33` hasta $L = 64$).
   - **Hallazgo teórico-numérico en acoplamiento fuerte de red ($J \gg v_A$):** el hamiltoniano efectivo de Schrieffer–Wolff sobre el doblete neutro es una cadena XXZ con anisotropía exacta $\Delta_{\text{ef}} = (v_A^2 + v_B^2)/(2 v_A v_B) = 1 + (v_A - v_B)^2/(2 v_A v_B) > 1$ (`1,336` para $v_A = 1,0, v_B = 0,45$), de modo que la propia asimetría $v_A \neq v_B$ induce una brecha BKT de Néel visible como saturación de $S(N_r)$ en $N_r = 24 \to 32$ (`0,8300 → 0,8541`, $c_{\text{hi}} = 0,50$). En contraste con el confinamiento 1+1D (`mis = 0,00000`), el flujo RG 3+1D de QED Coulomb conserva el $69,4\,\%$ del desajuste inicial (`0,7586 → 0,5267`).

## Fase 20: nivel publicación, verificación independiente y activos reproducibles (Vías A, B y C)

### (ah) Fotón real `Yee-4 S₄`, Wang–Landau multicanónico + gran $N_f$, 2-site $U(1)$-TDVP + diagrama BKT y scripts independientes

Tests `t192`–`t194` (`src/tests/phase20.rs`), experimentos interactivos `137`–`139` en `index.html`, subcomandos `phase20`/`via-a-yee4`/`via-b-wl`/`via-c-tdvp` en `src/bin/qsu_cli.rs`, scripts independientes de verificación Python/SciPy (`scripts/verify_via_a_mlodinow_brun.py`, `scripts/verify_via_c_schwinger_exact.py`) y generador automático de figuras vectoriales y tablas LaTeX (`scripts/generate_publication_assets.py` $\to$ `figuras_publicacion/fig_via_{a,b,c}.{pdf,svg,png}` y `tablas_latex.tex`):

1. **Vía A — Acoplamiento real 3D `Iso-SC+FCC S₄` + `Yee-4 S₄` y propagación polarizada $16^3$ (`via_a_coupled_yee4_3d`, Exp 137):**
   - El hamiltoniano local simpléctico $H_{\text{EM}} = \frac{1}{2}\sum_a \langle E_a, M_a E_a\rangle + \frac{1}{2}\sum_{a<b}\|\mathcal{K}_a\mathcal{K}_b F^{(2)}_{ab}\|^2$ acoplado al fermión `Iso-SC+FCC S₄` en la misma malla $6^3$ conserva norma (`4,76e-14`), **ley de Gauss estándar (`8,18e-16`)** e invariancia gauge (`1,25e-16` / `1,62e-14`).
   - En propagación 3D real $16^3$ (`Yoshida–Suzuki S₄`), `Yee-4 S₄` reduce la anisotropía axial-diagonal en **`4 341×`** frente a Yee-2 (`3,24e-3 → 7,46e-7`), conserva energía a `1,10e-6` y suprime la birrefringencia física a $45^\circ$ en $\mathbf{k} = \frac{2\pi}{16}(2,1,1), T = 10$ en **`16 571×`** frente a Mlodinow–Brun (`3,84e-5` vs. `0,6367 rad`). Verificado independientemente en NumPy/SciPy (`k^{4,00}` dispersión, `k^{7,00}` birrefringencia, cota $SO(3)$ `1,358e-6 ℓ_P` al 95 % CL).
2. **Vía B — Wang–Landau multicanónico sobre la barrera de primer orden, 5 semillas y barrido de masa $m$ / FSS a gran $N_f$ (`via_b_wl_and_matter_sweep`, `qsu_cli via-b-wl`, `qsu_cli via-b-multiseed`, Exp 138):**
   - Wang–Landau controlado por planitud + ajuste Berg–Neuhaus + producción con $g(S)$ congelado cruza limpiamente la barrera de Surya ($96\text{–}100\,\%$ de cobertura de bines en Puro $N = 20..48$ y SJ $N_f = 1$ en $N = 20, 36$), midiendo $\chi_{\text{pico}}: 268 \to 1\,135$, calor latente $\Delta s = 9,3 \to 11,6$ y barrera de energía libre $\Delta F = 1,5\text{–}2,7 > 0$ mientras recupera $d_{\text{MM}}(0) = 2,003\text{–}2,025$. La auditoría de 5 semillas independientes (`datos_investigacion/via_b_wl_multiseed.csv`) cuantifica la reproducibilidad entre semillas ($\beta_c = 0,1292 \pm 0,0056 \to 0,0936 \pm 0,0050$ en Puro y $0,1296 \pm 0,0022 \to 0,0884 \pm 0,0028$ en SJ $N_f = 1$, con $\Delta F = 1,30\text{–}2,18 > 0$).
   - El barrido en $\beta = 0$ ($N = 36$) confirma la ausencia de inestabilidad IR en el límite cuasi-sin masa ($m = 0,05$: `d_MM = 2,026 ± 0,008`, $\tau = 0,97$). El escalado de tamaño finito a gran $N_f$ (`datos_investigacion/via_b_large_nf_fss.csv`, $N_f \in \{0, 2, 4, 6, 8\} \times N \in \{24, 36, 48\}$) aclara la naturaleza del salto en $N_f = 8$: a $N$ fijo produce un fuerte desplazamiento hacia órdenes poco profundos (`d_MM = 3,803 ± 0,210, τ = 61,6` en $N = 24$), pero al aumentar $N$ **se relaja hacia la variedad 2D** (`2,568 ± 0,059, τ = 27,7` en $N = 36 \to$ **`2,196 ± 0,034, τ = 6,55` en $N = 48$**), revelando un **cruce de tamaño finito gobernado por la razón $N_f/N$** (competencia entre la entropía superlineal $\ln(N!) \sim N\ln N$ de $\Omega_{2\text{D}}$ y la acción de 1 lazo $N_f S_{\text{mat}}$), en contraste radical con el colapso asintótico de Ising ($d_\infty = 1,61$).
3. **Vía C — 2-site $U(1)$-TDVP en tiempo real ($L = 32$, $2^{32}$ estados), diagrama de fases BKT de Néel y $\chi$-convergencia (`via_c_tdvp_and_bkt_diagram`, `qsu_cli via-c-tdvp`, `qsu_cli via-c-chi-conv`, Exp 139):**
   - Evolucionando paquetes locales $n_{\alpha,c}|\text{GS}\rangle$ (conservación exacta $U(1)_A \times U(1)_B$) en tiempo real con deriva de energía `2,2–2,7×10⁻⁵`, el desajuste balístico cae **$7,9\times$ en $L = 24$** (`0,728 → 0,092`, $v_A, v_B = 0,652, 0,715 \approx \sqrt{v_A v_B}$) y **$5,1\times$ en $L = 32$** (`0,769 → 0,150`).
   - El diagrama BKT ($N_r = 16 \to 24, \chi = 36$) en $J = 2,50$ muestra la transición limpia desde el punto crítico simétrico $\Delta_{\text{ef}} = 1,00$ (`c_eff = +1,21`, `C_Néel(d=4) = +0,104`) hacia la fase Néel profunda en $\Delta_{\text{ef}} = 2,125$ (`c_eff = -3,88`, `C_Néel = +0,441`, $4,24\times$ mayor). La auditoría de dimensión de enlace $\chi \in \{24, 32, 44, 56\}$ (`datos_investigacion/via_c_chi_convergence.csv`) confirma convergencia de 5 cifras en el punto simétrico (`E_24 = -5,455193 → -5,455213`, `c_eff = +1,232 → +1,238`, `C_Néel = +0,10427`) y estabilidad al $1\,\%$ de $C_{\text{Néel}} = +0,435\text{–}+0,445$ ($4,26\times$ superior al control) en la fase Néel profunda, donde el descenso de $S_{24}$ en $\sim\ln 2$ al subir $\chi$ reproduce la ruptura espontánea $\mathbb{Z}_2$ esperada entre los dos estados de Néel cuasi-degenerados. El script independiente `verify_via_c_schwinger_exact.py` confirma los resultados de Rust con discrepancia máxima `1,20e-7`.
