import { describe, expect, it } from 'vitest';
import {
  emptyProgress, isSolved, isUnlocked, loadProgress, nextLevel, recordHandIn, saveProgress, useHint,
  type ProgressStorage,
} from './progress.ts';

const order = ['level1', 'leaf', 'tent'];

function memoryStorage(): ProgressStorage & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return { data, getItem: (k) => data.get(k) ?? null, setItem: (k, v) => { data.set(k, v); } };
}

describe('unlocking', () => {
  it('opens the first Level, and each next one once the one before is Solved', () => {
    let p = emptyProgress();
    expect(order.map((id) => isUnlocked(p, order, id))).toEqual([true, false, false]);
    p = recordHandIn(p, 'level1', { stars: 1, match: 40 }).progress;
    expect(order.map((id) => isUnlocked(p, order, id))).toEqual([true, true, false]);
  });

  it('does not unlock on a Hand-in that did not pass', () => {
    const p = recordHandIn(emptyProgress(), 'level1', { stars: 0, match: 70 }).progress;
    expect(isSolved(p, 'level1')).toBe(false);
    expect(isUnlocked(p, order, 'leaf')).toBe(false);
  });

  it('knows the next Level, and that the last one has none', () => {
    expect(nextLevel(order, 'level1')).toBe('leaf');
    expect(nextLevel(order, 'tent')).toBeUndefined();
  });
});

describe('recordHandIn', () => {
  it('keeps the Best: more Stars win, then more Match', () => {
    let p = emptyProgress();
    const first = recordHandIn(p, 'leaf', { stars: 2, match: 80 });
    expect(first.result.newBest).toBe(false); // nothing to beat yet
    expect(first.result.firstPass).toBe(true);
    p = first.progress;

    const worse = recordHandIn(p, 'leaf', { stars: 1, match: 95 });
    expect(worse.result.newBest).toBe(false);
    expect(worse.result.best).toEqual({ stars: 2, match: 80 });
    expect(worse.result.firstPass).toBe(false);

    const better = recordHandIn(worse.progress, 'leaf', { stars: 2, match: 90 });
    expect(better.result.newBest).toBe(true);
    expect(better.progress.best.leaf).toEqual({ stars: 2, match: 90 });
  });

  it('caps the Level at 2 Stars once a Hint was used', () => {
    const p = useHint(emptyProgress(), 'leaf');
    const { result } = recordHandIn(p, 'leaf', { stars: 3, match: 100 });
    expect(result.stars).toBe(2);
    expect(result.passed).toBe(true);
  });

  it('does not change the progress it was given', () => {
    const p = emptyProgress();
    recordHandIn(p, 'leaf', { stars: 3, match: 100 });
    expect(p.best).toEqual({});
  });
});

describe('storage', () => {
  it('saves and loads the progress', () => {
    const storage = memoryStorage();
    const p = recordHandIn(useHint(emptyProgress(), 'tent'), 'level1', { stars: 3, match: 100 }).progress;
    expect(saveProgress(storage, p)).toBe(true);
    expect(loadProgress(storage)).toEqual(p);
  });

  it('starts fresh without storage, with broken data, or with invalid entries', () => {
    expect(loadProgress(null)).toEqual(emptyProgress());
    const storage = memoryStorage();
    storage.setItem('shape-match/progress/v1', '{not json');
    expect(loadProgress(storage)).toEqual(emptyProgress());
    storage.setItem('shape-match/progress/v1', JSON.stringify({ best: { a: { stars: 7, match: 1 }, b: { stars: 2, match: 50 } } }));
    expect(loadProgress(storage).best).toEqual({ b: { stars: 2, match: 50 } });
  });

  it('reports a failed save instead of throwing', () => {
    const full: ProgressStorage = { getItem: () => null, setItem: () => { throw new Error('quota'); } };
    expect(saveProgress(full, emptyProgress())).toBe(false);
    expect(saveProgress(null, emptyProgress())).toBe(false);
  });
});
