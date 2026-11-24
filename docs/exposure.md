# Mapa de exposición: qué instancias pudo ver cada solucionador

Fuentes:
- Actas de la SAT Competition 2026 (DOI 10.34726/12919).
- Ding et al., IJCAI 2026 (AESAT).
- `scores.csv` oficial de 2026.

Fecha de consulta: 2026-11-23.

Se dice que una instancia estuvo **expuesta** a un solucionador si sus desarrolladores pudieron usarla
(o usar su familia) para tomar decisiones de diseño, ajuste o selección. Esto no se limita a las
instancias de entrenamiento declaradas: la selección de la variante final también cuenta.

| Solucionador | Tipo de IA | Base | Datos de desarrollo declarados | Regla de exposición (conservadora) |
|---|---|---|---|---|
| ae-kissat-mab-sc2025 | ajustado (AESAT, DeepSeek-R1) | Kissat 4.0.2 MAB | 60 instancias al azar de la SAT Comp 2024; evaluado en 2022–2025 | pistas `main_2022`…`main_2025` |
| kissat-mab-eae1 | generado (EAE, 100 % de los cambios por IA) | ae-kissat-mab-sc2025 | marco AESAT (benchmarks de competición) | `main_2022`…`main_2025` |
| kissat-mab-hypre-evolve / -satlution | ajustado | kissat-mab-hypre | AESAT / SATLUTION (benchmarks de competición) | `main_2022`…`main_2025` |
| mergesat-l | generado | mergesat | Main Tracks 2024 y 2025 | `main_2024`, `main_2025` |
| kissat-evolve-rb | generado (KissatEvolve, Gemini) | Kissat | subconjunto de 50 instancias de 2025 y benchmark de 2025 | `main_2025` |
| lymphosat | generado (agentes Claude Opus 4.6 y GPT 5.5) | fallback Kissat 4.0.2 | **80 % de cada familia de la GBD** (189 familias) para entrenar y 20 % para validar | **toda instancia de la GBD anterior a 2026** |
| humanos (kissat, cadical3, satsuma-iter-kissat, kissat-mab-hypre, mergesat) | ninguno | — | ajustados por sus autores durante años con competiciones anteriores | todas las pistas `main_*` ≤ 2025 |

## Consecuencias para el diseño

1. **Ningún conjunto anterior a 2026 está limpio para todos.** Los humanos también se ajustaron con
   competiciones pasadas. La diferencia está en la *intensidad* y la *automatización* del ajuste, no en
   que haya o no exposición.
2. **El único conjunto sin exposición para todos** son las instancias **enviadas por primera vez en 2026**
   (`submissions_2026` sin aparición anterior en ninguna pista). Se entregaron junto con los
   solucionadores, así que ningún desarrollador pudo verlas.
3. **El caso de LymphoSAT** es un experimento natural de memorización: sus emparejadores reconocen
   familias de la GBD por sus rasgos estructurales. La comparación entre las familias que pudo ver y las
   que no, o entre instancias perturbadas y sin perturbar, mide directamente su capacidad de
   generalización.
4. **Robustez frente a perturbaciones que preservan la semántica.** Permutar variables y cláusulas e
   invertir polaridades produce la misma fórmula lógica con otra codificación. Un solucionador que
   generaliza no debería cambiar mucho; uno sobreajustado a la codificación (emparejadores, heurísticas
   afinadas a patrones concretos), sí.

## Lista exacta de entrenamiento de AESAT

`solvers/exposure/aesat_train_hashes.txt` contiene las 60 instancias, extraídas de
`work/data/download.txt` del repositorio oficial (commit `6c9fa38`).
- Las 60 están en la GBD y pertenecen a `main_2024`.
- **4 de ellas reaparecen en `main_2026`.** Los resultados oficiales de 2026 de los solucionadores
  derivados de AESAT incluyen, por tanto, 4 instancias de su propio entrenamiento.
- Coincidencias con las familias de la tesis: *scheduling* 9, *argumentation* 6, *miter* 5,
  *bitvector* 1. El resto pertenece a familias combinatorias.

## Pendiente

- Confirmar los datos de desarrollo de kissat-mab-hypre-evolve/-satlution en su sección de las actas.
