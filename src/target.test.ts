import { describe, expect, it } from 'vitest';
import { allTasks, cloneNet, sameLayout, toolsText } from './target.ts';
import { getPoint, getWeight } from './controlNet.ts';
import { checkLayout } from './shapeChecks.ts';

const tasks = allTasks();

describe('Tasks', () => {
  it('have unique ids', () => {
    expect(new Set(tasks.map((t) => t.id)).size).toBe(tasks.length);
  });

  describe.each(tasks)('$id', (task) => {
    it('starts in the Layout of its Target (ADR 0001)', () => {
      expect(checkLayout(task.start)).toEqual([]);
      expect(sameLayout(task.start, task.target)).toBe(true);
    });

    it('does not start at the solution', () => {
      expect(Array.from(task.start.points)).not.toEqual(Array.from(task.target.points));
    });
  });

  it('come in more than one Layout', () => {
    expect(tasks.some((t) => !sameLayout(t.target, tasks[0].target))).toBe(true);
  });
});

describe('cloneNet', () => {
  it('copies points and knots, so editing the clone leaves the Task alone', () => {
    const task = tasks[0];
    const copy = cloneNet(task.start);
    copy.points[2] = 9;
    copy.knotsU[1] = 0.5;
    expect(task.start.points[2]).not.toBe(9);
    expect(task.start.knotsU[1]).not.toBe(0.5);
    expect(sameLayout(copy, task.start)).toBe(false);
  });
});

describe('the final Level (superposition)', () => {
  const task = tasks.find((t) => t.id === 'superposition')!;

  it('comes last on the Board', () => {
    expect(tasks.at(-1)).toBe(task);
  });
  const z = (k: number) => getPoint(task.target, k)[2];

  it('has a ridge, a valley, and a flat point where they cross', () => {
    const at = (i: number, j: number) => z(j * task.target.countU + i);
    expect(at(1, 2)).toBeGreaterThan(0);  // ridge
    expect(at(3, 1)).toBeLessThan(0);     // valley
    expect(at(3, 2)).toBeCloseTo(0, 6);   // up and down add up to nothing
  });

  it('allows moving points only', () => {
    expect(task.tools).toEqual({ movePoints: true, weights: false, knotInsertion: false });
  });
});

describe('Level 2 (weights)', () => {
  const task = tasks.find((t) => t.id === 'weights')!;

  it('starts with the points of the Target and all weights 1', () => {
    for (let k = 0; k < task.start.countU * task.start.countV; k++) {
      getPoint(task.start, k).forEach((c, i) => expect(c).toBeCloseTo(getPoint(task.target, k)[i], 6));
      expect(getWeight(task.start, k)).toBe(1);
    }
  });

  it('is round only with √2/2 at the corners', () => {
    const corners = [...Array(task.target.countU * task.target.countV).keys()].filter((k) => getWeight(task.target, k) < 1);
    expect(corners).toHaveLength(8); // 4 corners of the square, in both rows
    for (const k of corners) expect(getWeight(task.target, k)).toBeCloseTo(Math.SQRT1_2, 6);
  });

  it('allows changing weights only', () => {
    expect(task.tools).toEqual({ movePoints: false, weights: true, knotInsertion: false });
  });
});

describe('titles', () => {
  it('every Level name starts with a capital letter', () => {
    for (const task of tasks) expect(task.title.split('·').pop()!.trim()).toMatch(/^[A-Z]/);
  });
});

describe('toolsText', () => {
  it('says what the learner can do', () => {
    expect(toolsText({ movePoints: false, weights: true, knotInsertion: false })).toBe('In this Level you can change weights.');
    expect(toolsText({ movePoints: true, weights: true, knotInsertion: false })).toBe('In this Level you can move control points, change weights.');
  });
});
