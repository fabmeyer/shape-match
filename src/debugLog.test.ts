import { describe, expect, it } from 'vitest';
import { buildReport, DebugLog, errorMessage, netFromData, netToData } from './debugLog.ts';
import { allShapes } from './shapes.ts';
import { evaluate } from './nurbs.ts';

describe('DebugLog', () => {
  it('keeps the newest entries, oldest first, with timestamps', () => {
    let clock = 0;
    const log = new DebugLog(3, () => clock);
    for (let k = 0; k < 5; k++) { clock = k * 10; log.log('edit', { k }); }
    expect(log.entries().map((e) => e.data)).toEqual([{ k: 2 }, { k: 3 }, { k: 4 }]);
    expect(log.entries()[0].t).toBe(20);
  });

  it('hands out a copy, so the report does not change afterwards', () => {
    const log = new DebugLog(5, () => 0);
    log.log('a');
    const copy = log.entries();
    log.log('b');
    expect(copy).toHaveLength(1);
  });
});

describe('netToData / netFromData', () => {
  it('turns a surface into data and back into the same surface', () => {
    const ball = allShapes().find((s) => s.id === 'ball')!.net; // with weights √2/2
    const back = netFromData(JSON.parse(JSON.stringify(netToData(ball))));
    expect(back.knotsU).toEqual(ball.knotsU);
    for (const [u, v] of [[0.1, 0.2], [0.42, 0.6], [0.9, 0.95]]) {
      const a = evaluate(ball, u, v).point;
      const b = evaluate(back, u, v).point;
      a.forEach((c, i) => expect(b[i]).toBeCloseTo(c, 5));
    }
  });

  it('writes points in Cartesian form with their weight, like a Target in code', () => {
    const data = netToData(allShapes().find((s) => s.id === 'ball')!.net);
    expect(data.points[1][3]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(data.points).toHaveLength(data.countU * data.countV);
  });
});

describe('buildReport', () => {
  it('adds the log to the parts', () => {
    const log = new DebugLog(5, () => 7);
    log.log('level', { id: 'leaf' });
    const leaf = allShapes()[0].net;
    const report = buildReport({
      app: { version: 'test', createdAt: '2026-11-25', url: 'http://localhost' },
      environment: { userAgent: 'vitest' },
      task: { id: 'leaf', title: 'Shape 1 · leaf' },
      learner: netToData(leaf),
      state: { view: 'top' },
    }, log);
    expect(report.log).toEqual([{ t: 7, type: 'level', data: { id: 'leaf' } }]);
    expect(JSON.parse(JSON.stringify(report)).learner.countU).toBe(leaf.countU);
  });
});

describe('errorMessage', () => {
  it('names the error and its message', () => {
    expect(errorMessage(new TypeError('bad'))).toBe('TypeError: bad');
    expect(errorMessage('plain')).toBe('plain');
  });
});
