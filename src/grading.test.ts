import { describe, expect, it } from 'vitest';
import { calibrate, DEFAULT_GRADING, grade, gradingConfig, matchPercent, starMarks, type Calibration } from './grading.ts';
import { shapeDistance } from './shapeDistance.ts';
import { allTasks } from './target.ts';

const c: Calibration = { tolerance: 0.002, hausdorffTolerance: 0.004, startRms: 0.2 };

describe('calibrate', () => {
  it('sets τ from the Noise floor, but never below the minimum', () => {
    expect(calibrate({ rms: 0.001, hausdorff: 0.003 }, { rms: 0.1, hausdorff: 0.2 }).tolerance).toBeCloseTo(0.003, 12);
    expect(calibrate({ rms: 1e-5, hausdorff: 1e-5 }, { rms: 0.1, hausdorff: 0.2 }).tolerance).toBe(DEFAULT_GRADING.minTolerance);
  });

  it('remembers the starting distance for the Match', () => {
    expect(calibrate({ rms: 0, hausdorff: 0 }, { rms: 0.15, hausdorff: 0.3 }).startRms).toBe(0.15);
  });
});

describe('grade', () => {
  it('gives Stars by the RMS: 3 within τ, 2 within 3τ, 1 within 10τ, else 0', () => {
    const stars = (rms: number) => grade({ rms, hausdorff: 0 }, c).stars;
    expect(stars(0.002)).toBe(3);
    expect(stars(0.006)).toBe(2);
    expect(stars(0.02)).toBe(1);
    expect(stars(0.0201)).toBe(0);
  });

  it('passes from 1 Star on', () => {
    expect(grade({ rms: 0.02, hausdorff: 0 }, c).passed).toBe(true);
    expect(grade({ rms: 0.03, hausdorff: 0 }, c).passed).toBe(false);
  });

  it('fails a surface that is fine on average but wrong in one spot (Hausdorff bound)', () => {
    const g = grade({ rms: 0.001, hausdorff: 0.05 }, c); // RMS alone would give 3 Stars
    expect(g.stars).toBe(0);
    expect(g.passed).toBe(false);
  });

  it('takes the factors from the config, which a Level can override', () => {
    const strict = gradingConfig({ star1Factor: 5 });
    expect(grade({ rms: 0.015, hausdorff: 0 }, c, strict).stars).toBe(0);
    expect(grade({ rms: 0.015, hausdorff: 0 }, c).stars).toBe(1);
  });
});

describe('matchPercent', () => {
  it('is 0 % at the start and 100 % within the Tolerance', () => {
    expect(matchPercent(0.2, c)).toBe(0);
    expect(matchPercent(0.5, c)).toBe(0); // worse than the start
    expect(matchPercent(0.002, c)).toBe(100);
    expect(matchPercent(0, c)).toBe(100);
  });

  it('grows with every halving of the distance by the same step (log scale)', () => {
    const a = matchPercent(0.1, c) - matchPercent(0.2, c);
    const b = matchPercent(0.025, c) - matchPercent(0.05, c);
    expect(a).toBeCloseTo(b, 10);
  });
});

describe('starMarks', () => {
  it('puts the 1 and 2 Star thresholds on the meter, in order', () => {
    const [one, two] = starMarks(c);
    expect(one).toBeCloseTo(matchPercent(0.02, c), 10);
    expect(two).toBeCloseTo(matchPercent(0.006, c), 10);
    expect(0 < one && one < two && two < 100).toBe(true);
  });
});

describe('every Task', () => {
  // CPU reference; at 16 the Noise floor is so coarse that 10τ reaches the start of some Tasks (Level 1: 3.42 % vs 3.33 %),
  // from 32 on every start is far outside; the app measures at 64
  const RES = 32;
  it.each(allTasks())('$id: the start gives 0 Stars, the Target 3 Stars', (task) => {
    const config = gradingConfig(task.grading);
    const noise = shapeDistance(task.target, task.target, { res: RES, learnerShift: 0.5 });
    const start = shapeDistance(task.target, task.start, { res: RES });
    const cal = calibrate(noise, start, config);
    expect(grade(start, cal, config).stars).toBe(0);
    expect(grade(shapeDistance(task.target, task.target, { res: RES }), cal, config).stars).toBe(3);
  });
});
