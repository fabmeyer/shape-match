import { describe, expect, it } from 'vitest';
import { allTasks, cloneNet, sameLayout } from './target.ts';
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
