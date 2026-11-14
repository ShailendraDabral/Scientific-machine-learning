import { lorenz } from './lorenz.js';
import { rossler } from './rossler.js';
import { logistic } from './logistic.js';
import { duffing } from './duffing.js';
import { doublePendulum } from './doublePendulum.js';
import { threeBody, presets as threeBodyPresets, toCenterOfMassFrame } from './threeBody.js';

export const systems = { lorenz, rossler, logistic, duffing, doublePendulum, threeBody };
export { lorenz, rossler, logistic, duffing, doublePendulum, threeBody, threeBodyPresets, toCenterOfMassFrame };
