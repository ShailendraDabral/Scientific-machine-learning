/**
 * Error types raised by the numerical core.
 * The core never returns silently-wrong numbers: it either produces a finite
 * result or throws one of these.
 */

/** A parameter, initial state or solver option is outside its valid domain. */
export class InvalidParameterError extends RangeError {
  constructor(message, details = {}) {
    super(message);
    this.name = 'InvalidParameterError';
    this.details = details;
  }
}

/** The computation left the domain of finite numbers or of the model. */
export class NumericalInstabilityError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'NumericalInstabilityError';
    this.details = details;
  }
}
