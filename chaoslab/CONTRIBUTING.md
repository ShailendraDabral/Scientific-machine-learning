# Contributing to ChaosLab

Thank you for considering a contribution. ChaosLab is a scientific tool, so the bar for a change is that it is numerically correct, tested, and honest about its limits.

## Getting started

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
npm install
npm run dev     # development server
npm test        # numerical validation tests
```

Node.js 20 or newer is required. There is no backend and nothing to configure.

## Ground rules

1. **Computation stays out of the interface.** Anything that produces a number belongs in `src/core` as a pure function with no DOM access. The UI calls the core; it does not re-implement formulas. The one tolerated exception is cheap display-only arithmetic on stored samples.
2. **No decorative data.** Every plotted value must come from the simulation or from a stated formula evaluated on simulation output. Do not add pre-recorded, random or smoothed curves.
3. **Every numerical change comes with a test.** Tests compare against an analytic result, a conservation law, a published value with a citation, or a convergence order. A test that only checks "it runs" is not sufficient.
4. **Validate inputs and fail loudly.** Reject invalid parameters with `InvalidParameterError`. Never return `NaN` or silently clamp a value.
5. **State assumptions and limits.** New models must fill in `assumptions` and `limitations`; new analyses need an explanation in `src/ui/analysisText.js`.
6. **Do not overstate.** In code comments, documentation and pull requests, report only results you have actually obtained, and say how.
7. **No external services.** The application must keep working as static files with no network calls at run time.

## Adding a dynamical system

1. Create `src/core/systems/<name>.js` exporting an object with: `id`, `name`, `kind`, `paramSchema`, `dimension`, `resolveParams`, `defaultState`, `rhs(t, y, dydt, p)` (writes in place, allocates nothing), `rateScale`, `invariants`, `assumptions`, `limitations`. Add `jacobian` if you have it, and `separable` only if the Hamiltonian is of the form T(p) + V(q).
2. Register it in `src/core/systems/index.js`.
3. Add tests in `tests/`: at minimum, the right-hand side against hand-computed values, equilibria or another analytic property, rejection of invalid parameters, and any conserved quantity.
4. Create `src/experiments/<name>.js` with controls, a 3D view, plots and readouts, and register it in `src/main.js`.
5. Add an entry to `SYS` in `src/ui/analysisLab.js` if the analysis suite should cover it, and a section definition in `src/core/analysis/poincare.js` if a Poincaré section makes sense.
6. Run `npm run docs` to regenerate `docs/MODELS.md` (add the display equations in `scripts/build-docs.js`).

## Adding an integrator

1. Create `src/core/integrators/<name>.js` with `id`, `name`, `order`, `symplectic`, `requiresSeparable`, `evaluationsPerStep`, `stabilityLimit`, `notes`, `createWork(n)` and `step(system, params, t, y, h, out, work)`. `step` must not allocate and must not write to `y`.
2. Register it in `src/core/integrators/index.js`. The convergence-order test in `tests/integrators.test.js` picks it up automatically and will fail if the observed order differs from the declared one.
3. Add its display name and colour in `src/ui/odeLab.js` and `src/ui/analysisLab.js`, and list it in the `numerics.methods` of the experiments it is valid for.

## Adding an analysis

1. Implement it in `src/core/analysis/` as a function that takes and returns plain data, and wire it into `runAnalysis` in `src/core/analysis/index.js`.
2. Test it on a case with a known answer.
3. Add a renderer in `src/ui/analysisLab.js` and an explanation in `src/ui/analysisText.js` covering what is measured, how it is computed, how to read it, and its limits.

## Code style

- Plain ES modules, no framework, no TypeScript build step. Use JSDoc comments for public functions.
- State vectors are `Float64Array`; hot loops avoid allocation.
- Two-space indentation, single quotes, semicolons. Keep modules small and named for what they contain.
- Comments explain the mathematics and the reason for a choice, not the syntax.

## Pull requests

- Keep each pull request to one topic.
- Describe what changed and how you verified it. Paste the output of `npm test`; do not write "tests pass" without having run them.
- If the change affects what is drawn, include a screenshot.
- If you change a model or method, update `assumptions`/`limitations` and regenerate `docs/MODELS.md`.
- The Pages workflow runs `npm test` before building, so a failing test blocks deployment.

## Reporting a problem

Open an issue with: the experiment, the parameter values and initial conditions, the integrator and step size, what you expected (with a reference if it is a scientific claim), and what you observed. A reproducible case in terms of `integrate(...)` from `src/core` is ideal.

## Licence

By contributing you agree that your contribution is released under the MIT License of this project.
