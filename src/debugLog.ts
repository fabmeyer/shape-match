// debugging: an event log and a debug report the learner can copy and paste to someone else
// nobody but the person at the browser sees its console, its surface or its errors; the report hands
// that over: the environment, the Task, the learner's surface as exact data, the state, and the log
// the log stays in memory: nothing is sent, nothing is stored between visits
import type { ControlNet } from './controlNet.ts';

const CAPACITY = 200; // newest events kept

export interface LogEntry {
  t: number;      // ms since the page loaded
  type: string;   // e.g. 'level', 'edit', 'measure', 'hand-in', 'screen', 'error'
  data?: unknown;
}

export class DebugLog {
  echo = false; // also print every event to the console (?debug)
  private items: LogEntry[] = [];
  private capacity: number;
  private now: () => number;

  constructor(capacity = CAPACITY, now: () => number = () => performance.now()) {
    this.capacity = capacity;
    this.now = now;
  }

  log(type: string, data?: unknown): void {
    const entry: LogEntry = { t: Math.round(this.now()), type, data };
    this.items.push(entry);
    if (this.items.length > this.capacity) this.items.shift();
    if (this.echo) console.debug(`[${type}]`, data ?? '');
  }

  // the newest events, oldest first
  entries(): LogEntry[] {
    return [...this.items];
  }
}

// the one log of the app
export const debugLog = new DebugLog();

export function errorMessage(e: unknown): string {
  return e instanceof Error ? `${e.name}: ${e.message}` : String(e);
}

// ---------------------------------------------------------
// A surface as data: the same form a Target is written in (Cartesian points and weights)
// ---------------------------------------------------------

export interface NetData {
  countU: number;
  countV: number;
  p: number;
  q: number;
  knotsU: number[];
  knotsV: number[];
  points: [number, number, number, number][]; // x, y, z, w per control point, index j * countU + i
}

const round = (x: number) => Math.round(x * 1e6) / 1e6; // f32 points: 6 decimals are exact enough

export function netToData(net: ControlNet): NetData {
  const points: [number, number, number, number][] = [];
  for (let k = 0; k < net.countU * net.countV; k++) {
    const w = net.points[k * 4 + 3];
    points.push([round(net.points[k * 4] / w), round(net.points[k * 4 + 1] / w), round(net.points[k * 4 + 2] / w), round(w)]);
  }
  const { countU, countV, p, q } = net;
  return { countU, countV, p, q, knotsU: [...net.knotsU], knotsV: [...net.knotsV], points };
}

export function netFromData(d: NetData): ControlNet {
  const points = new Float32Array(d.points.flatMap(([x, y, z, w]) => [w * x, w * y, w * z, w]));
  return { countU: d.countU, countV: d.countV, p: d.p, q: d.q, knotsU: [...d.knotsU], knotsV: [...d.knotsV], points };
}

// ---------------------------------------------------------
// The report
// ---------------------------------------------------------

export interface DebugReport {
  app: { version: string; createdAt: string; url: string };
  environment: { userAgent: string; adapter?: string; screen?: string };
  task: { id: string; title: string };
  learner: NetData;    // the learner's surface right now
  state: Record<string, unknown>; // camera, mode, calibration, last grade, progress, ...
  log: LogEntry[];
}

export function buildReport(parts: Omit<DebugReport, 'log'>, log: DebugLog = debugLog): DebugReport {
  return { ...parts, log: log.entries() };
}
