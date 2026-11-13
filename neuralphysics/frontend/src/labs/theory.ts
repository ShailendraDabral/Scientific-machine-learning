// Mathematical foundations shown in the Operator Lab. Each block is prose followed by display equations.
export interface TheoryBlock { text: string; tex?: string }
export interface Theory { title: string; tagline: string; blocks: TheoryBlock[]; reference: string }

export const THEORY: Record<string, Theory> = {
  deeponet: {
    title: 'DeepONet',
    tagline: 'Learn an operator as a sum of basis functions whose coefficients depend on the input function.',
    blocks: [
      { text: 'The goal is an operator between function spaces: here the map from an initial condition to the solution of the Burgers equation on the whole space-time domain.',
        tex: String.raw`G:\; u_0(x)\;\longmapsto\; u(x,t), \qquad y=(x,t)` },
      { text: 'The universal approximation theorem for operators (Chen and Chen, 1995) states that a continuous operator can be approximated to any accuracy by two networks joined through an inner product.',
        tex: String.raw`\Big|\,G(u)(y)-\sum_{k=1}^{p} b_k\big(u(x_1),\dots,u(x_m)\big)\,t_k(y)\Big|<\varepsilon` },
      { text: 'The branch network b reads the input function at m fixed sensor points. The trunk network t reads the query coordinate y and produces p basis functions. The prediction is their linear combination.',
        tex: String.raw`G_\theta(u)(y)=\sum_{k=1}^{p} b_k(u)\,t_k(y)+b_0` },
      { text: 'Training minimises the relative error against solver-generated solutions. The output can be queried at any y, but the input must always be sampled at the same sensors.',
        tex: String.raw`\mathcal{L}_{\text{data}}=\frac{1}{N}\sum_{i=1}^{N}\frac{\lVert G_\theta(u_i)-G(u_i)\rVert_2}{\lVert G(u_i)\rVert_2}` },
    ],
    reference: 'Lu, Jin, Pang, Zhang and Karniadakis, Nature Machine Intelligence 3 (2021)',
  },
  fno: {
    title: 'Fourier Neural Operator',
    tagline: 'Stack integral-operator layers whose kernels are learned directly in Fourier space.',
    blocks: [
      { text: 'A neural operator generalises a network layer from vectors to functions: a pointwise linear map plus an integral operator with a learned kernel.',
        tex: String.raw`v_{l+1}(x)=\sigma\Big(W_l\,v_l(x)+\int_D \kappa_l(x-y)\,v_l(y)\,\mathrm{d}y\Big)` },
      { text: 'For a translation-invariant kernel the integral is a convolution, which the Fourier transform turns into a product. FNO therefore learns the kernel as complex weights R on the lowest Fourier modes.',
        tex: String.raw`\int_D \kappa(x-y)\,v(y)\,\mathrm{d}y=\mathcal{F}^{-1}\big(R\cdot\mathcal{F}(v)\big)(x),\qquad R=\mathcal{F}(\kappa),\quad |k|\le k_{\max}` },
      { text: 'A lifting map P raises the input to a wider channel space, L Fourier layers act there, and a projection Q returns the solution.',
        tex: String.raw`u=Q\circ\big(\sigma\circ(W_L+\mathcal{K}_L)\big)\circ\dots\circ\big(\sigma\circ(W_1+\mathcal{K}_1)\big)\circ P\,(u_0)` },
      { text: 'The weights live in Fourier space and not on a grid, so the same model can be evaluated on a finer or coarser grid. Each layer costs O(n log n) through the FFT.' },
    ],
    reference: 'Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart and Anandkumar, ICLR 2021',
  },
  pino: {
    title: 'Physics-Informed Neural Operator',
    tagline: 'An FNO trained on data and on the PDE residual of its own output.',
    blocks: [
      { text: 'PINO keeps the Fourier Neural Operator architecture and changes only the loss. Beside the data misfit it penalises the PDE residual and the initial-condition misfit of every predicted field.',
        tex: String.raw`\mathcal{L}=\lambda_{d}\,\mathcal{L}_{\text{data}}+\lambda_{p}\,\big\lVert \partial_t u_\theta+u_\theta\,\partial_x u_\theta-\nu\,\partial_{xx}u_\theta\big\rVert^2+\lambda_{0}\,\big\lVert u_\theta(\cdot,0)-u_0\big\rVert^2` },
      { text: 'The output is a field on a grid, so derivatives are taken on the grid: exactly in x through the Fourier transform, and with fourth-order central differences in t.',
        tex: String.raw`\partial_x u=\mathcal{F}^{-1}\big(\mathrm{i}k\,\mathcal{F}(u)\big),\qquad \partial_t u\approx\frac{-u^{n+2}+8u^{n+1}-8u^{n-1}+u^{n-2}}{12\,\Delta t}` },
      { text: 'The physics term needs no solver data, so it acts as extra supervision on every training input. Time levels next to t = 0 are left out because the initial transient is faster than one time step. The grid derivatives have their own discretisation error, so the residual of the reference data is reported as the floor.' },
      { text: 'PINO sits between two extremes: a PINN uses only physics and solves one instance; an FNO uses only data and learns the whole operator.' },
    ],
    reference: 'Li, Zheng, Kovachki, Jin, Chen, Liu, Azizzadenesheli and Anandkumar (2021), arXiv:2111.03794',
  },
  pinn: {
    title: 'Conventional PINN',
    tagline: 'One network per problem instance, trained on the PDE alone.',
    blocks: [
      { text: 'A PINN represents the solution for a single initial condition and is trained without solution data. Every new input function requires a new optimisation from scratch.',
        tex: String.raw`\min_\theta\;\frac{1}{N_r}\sum_{i}\big|\mathcal{N}[u_\theta](x_i,t_i)\big|^2+\lambda_0\,\frac{1}{N_0}\sum_{j}\big|u_\theta(x_j,0)-u_0(x_j)\big|^2` },
    ],
    reference: 'Raissi, Perdikaris and Karniadakis, Journal of Computational Physics 378 (2019)',
  },
};
