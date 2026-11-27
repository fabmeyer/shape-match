import { describe, expect, it } from 'vitest';
import { allTasks, changedStart, cloneNet, sameLayout, startChange, toolsText } from './target.ts';
import { flatControlNet, getPoint, getWeight, setPoint } from './controlNet.ts';
import { checkLayout } from './shapeChecks.ts';
import { allShapes } from './shapes.ts';

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

describe('starting surfaces of the library shapes', () => {
  it('get harder step by step from the Tent on', () => {
    expect(startChange(2)).toEqual({ squash: 0.25, stretchX: 1.3, stretchY: 1, lean: 0 });
    expect(startChange(5)).toEqual({ squash: 0.25, stretchX: 1.3, stretchY: 0.75, lean: 0 });
    expect(startChange(9)).toEqual({ squash: 0.25, stretchX: 1.3, stretchY: 0.75, lean: 0.35 });
  });

  it('keep the rings of a revolution round: flattened and leaning, never stretched', () => {
    expect(startChange(4, true)).toEqual({ squash: 0.25, stretchX: 1, stretchY: 1, lean: 0.35 });
    expect(startChange(9, true)).toEqual({ squash: 0.25, stretchX: 1, stretchY: 1, lean: 0.35 });
    const vase = tasks.find((t) => t.id === 'vase')!;
    const { countU, countV } = vase.start;
    for (let j = 0; j < countV; j++) { // every ring of 9 points has one radius around its own centre
      const ring = Array.from({ length: countU }, (_, i) => getPoint(vase.start, j * countU + i));
      const ys = ring.map((p) => p[1]);
      const centreY = (Math.max(...ys) + Math.min(...ys)) / 2;
      const r = (p: number[]) => Math.max(Math.abs(p[0]), Math.abs(p[1] - centreY));
      for (const p of ring) expect(r(p)).toBeCloseTo(r(ring[0]), 5);
    }
  });

  it('keep the Layout and the weights of the Target', () => {
    const shapeIds = new Set(allShapes().map((s) => s.id));
    for (const task of tasks.filter((t) => shapeIds.has(t.id))) {
      expect(sameLayout(task.start, task.target)).toBe(true);
      for (let k = 0; k < task.start.countU * task.start.countV; k++) expect(getWeight(task.start, k)).toBeCloseTo(getWeight(task.target, k), 12);
    }
  });

  it('stretch around the middle, lean with the height', () => {
    const net = flatControlNet();
    const k = 3 * net.countU + 2;
    setPoint(net, k, [getPoint(net, k)[0], getPoint(net, k)[1], 1]);
    const start = changedStart(net, { squash: 0.5, stretchX: 2, stretchY: 1, lean: 0.4 });
    const [x, y] = getPoint(net, k);
    const [sx, sy, sz] = getPoint(start, k);
    expect(sx).toBeCloseTo(2 * x, 6); // the flat sheet is centred on 0
    expect(sy).toBeCloseTo(y + 0.4, 6);
    expect(sz).toBeCloseTo(0.5, 6);
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
