/** Logistic map: descriptor only; the laboratory lives in ui/logisticLab.js. */
export default {
  id: 'logistic', title: 'Logistic map', group: 'Discrete map', kind: 'map',
  tex: [String.raw`x_{n+1} = r\,x_n(1 - x_n)`,
    String.raw`\lambda(r) = \lim_{N\to\infty}\frac1N\sum_{n=0}^{N-1}\ln\lvert r(1-2x_n)\rvert`],
  blurb: 'The bifurcation diagram plots the iterates that remain after a discarded transient, for each r. Below it, the Lyapunov exponent λ(r) separates stable and periodic regions (λ < 0) from chaotic ones (λ > 0). Drag a box on the diagram to zoom; it is recomputed at the new range.',
};
