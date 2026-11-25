// the grading rules (plan §2): from a measured Shape distance to Stars, the verdict and the Match
// pure functions, so they are tested without a GPU; all factors live in one config, tuned after a playtest
//   Tolerance τ = max(noiseFactor × Noise floor, minTolerance)        (per Level, from its Noise floor)
//   Stars by RMS:  3★ ≤ τ,  2★ ≤ star2Factor × τ,  1★ ≤ star1Factor × τ
//   Passed: at least 1★, and the Hausdorff distance under the looser bound hausdorffFactor × τ_H,
//           so a small wrong spot (a missing crease) can't hide in the average
// all distances are divided by the Target's diagonal, like the Shape distance itself

export interface GradingConfig {
  noiseFactor: number;
  minTolerance: number;   // in units of the diagonal: 0.002 = 0.2 %
  star2Factor: number;
  star1Factor: number;
  hausdorffFactor: number;
}

export const DEFAULT_GRADING: GradingConfig = {
  noiseFactor: 3,
  minTolerance: 0.002,
  star2Factor: 3,
  star1Factor: 10,
  hausdorffFactor: 10,
};

export type Stars = 0 | 1 | 2 | 3;

// what the grading needs from a measurement (CPU or GPU)
export interface Distance {
  rms: number;
  hausdorff: number;
}

// measured once when a Level loads
export interface Calibration {
  tolerance: number;          // τ, for the RMS
  hausdorffTolerance: number; // τ_H, for the Hausdorff distance
  startRms: number;           // the starting Learner surface: 0 % Match
}

export interface Grade extends Distance {
  stars: Stars;
  passed: boolean;
  match: number; // percent, 100 = within the Tolerance
}

export function gradingConfig(overrides: Partial<GradingConfig> = {}): GradingConfig {
  return { ...DEFAULT_GRADING, ...overrides };
}

// noise: the Target against itself on a grid shifted by half a cell; start: the starting Learner surface
export function calibrate(noise: Distance, start: Distance, config = DEFAULT_GRADING): Calibration {
  return {
    tolerance: Math.max(config.noiseFactor * noise.rms, config.minTolerance),
    hausdorffTolerance: Math.max(config.noiseFactor * noise.hausdorff, config.minTolerance),
    startRms: start.rms,
  };
}

export function grade(d: Distance, c: Calibration, config = DEFAULT_GRADING): Grade {
  const tau = c.tolerance;
  const byRms: Stars = d.rms <= tau ? 3 : d.rms <= config.star2Factor * tau ? 2 : d.rms <= config.star1Factor * tau ? 1 : 0;
  const stars: Stars = d.hausdorff <= config.hausdorffFactor * c.hausdorffTolerance ? byRms : 0;
  return { ...d, stars, passed: stars >= 1, match: matchPercent(d.rms, c) };
}

// the Match: 0 % at the starting surface, 100 % within the Tolerance, on a log scale in between,
// so that every Star is a visible step on the meter (equal factors of distance are equal steps)
export function matchPercent(rms: number, c: Calibration): number {
  if (rms <= c.tolerance) return 100;
  if (c.startRms <= c.tolerance) return 0; // the start is already within the Tolerance: nothing to scale
  const t = Math.log(c.startRms / rms) / Math.log(c.startRms / c.tolerance);
  return 100 * Math.min(1, Math.max(0, t));
}

// where the Match meter crosses into 1★ and 2★ (3★ is at 100 %), for the marks on the meter
export function starMarks(c: Calibration, config = DEFAULT_GRADING): [number, number] {
  return [matchPercent(config.star1Factor * c.tolerance, c), matchPercent(config.star2Factor * c.tolerance, c)];
}
