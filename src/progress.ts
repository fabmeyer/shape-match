// the learner's progress: the Best per Level, which Levels are unlocked, and Hints used
// pure functions on a plain object, kept in localStorage (no server); a missing or broken storage
// only means the progress isn't remembered, the game still works
import type { Stars } from './grading.ts';

const STORAGE_KEY = 'shape-match/progress/v1';
const HINT_CAP: Stars = 2; // using a Hint caps the Level at 2★

export interface LevelBest {
  stars: Stars;
  match: number; // percent
}

export interface Progress {
  best: Record<string, LevelBest>;   // by Task id
  hintUsed: Record<string, boolean>; // by Task id
}

// what a Hand-in brings to the Result screen
export interface HandInResult {
  levelId: string;
  stars: Stars;        // after the Hint cap
  match: number;
  passed: boolean;
  best: LevelBest;     // the Best after this Hand-in
  newBest: boolean;    // better than an earlier Hand-in
  firstPass: boolean;  // this Hand-in Solved the Level for the first time (the next one unlocks)
}

export function emptyProgress(): Progress {
  return { best: {}, hintUsed: {} };
}

export function isSolved(p: Progress, levelId: string): boolean {
  return (p.best[levelId]?.stars ?? 0) >= 1;
}

// the first Level is always open, every other one once the Level before it is Solved
export function isUnlocked(p: Progress, order: string[], levelId: string): boolean {
  const k = order.indexOf(levelId);
  return k === 0 || (k > 0 && isSolved(p, order[k - 1]));
}

export function nextLevel(order: string[], levelId: string): string | undefined {
  const k = order.indexOf(levelId);
  return k >= 0 ? order[k + 1] : undefined;
}

function better(a: LevelBest, b: LevelBest | undefined): boolean {
  return !b || a.stars > b.stars || (a.stars === b.stars && a.match > b.match);
}

export function recordHandIn(p: Progress, levelId: string, grade: { stars: Stars; match: number }): { progress: Progress; result: HandInResult } {
  const stars = (p.hintUsed[levelId] ? Math.min(grade.stars, HINT_CAP) : grade.stars) as Stars;
  const previous = p.best[levelId];
  const handedIn: LevelBest = { stars, match: grade.match };
  const improved = better(handedIn, previous);
  const best = improved ? handedIn : previous!;
  const progress: Progress = { ...p, best: { ...p.best, [levelId]: best } };
  return {
    progress,
    result: {
      levelId,
      stars,
      match: grade.match,
      passed: stars >= 1,
      best,
      newBest: improved && previous !== undefined,
      firstPass: stars >= 1 && !isSolved(p, levelId),
    },
  };
}

export function useHint(p: Progress, levelId: string): Progress {
  return { ...p, hintUsed: { ...p.hintUsed, [levelId]: true } };
}

// ---------------------------------------------------------
// Storage
// ---------------------------------------------------------

export type ProgressStorage = Pick<Storage, 'getItem' | 'setItem'>;

// localStorage, or null where it is blocked (private windows, previews): even reading the property can throw
export function browserStorage(): ProgressStorage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function loadProgress(storage: ProgressStorage | null): Progress {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    if (!raw) return emptyProgress();
    const data = JSON.parse(raw);
    const p = emptyProgress();
    for (const [id, b] of Object.entries(data.best ?? {})) {
      const { stars, match } = b as LevelBest;
      if ([0, 1, 2, 3].includes(stars) && Number.isFinite(match)) p.best[id] = { stars, match };
    }
    for (const [id, used] of Object.entries(data.hintUsed ?? {})) if (used === true) p.hintUsed[id] = true;
    return p;
  } catch {
    return emptyProgress(); // unreadable: start fresh rather than break the game
  }
}

// false if it couldn't be saved (the game goes on, only without memory)
export function saveProgress(storage: ProgressStorage | null, p: Progress): boolean {
  try {
    storage?.setItem(STORAGE_KEY, JSON.stringify(p));
    return storage !== null;
  } catch {
    return false;
  }
}
