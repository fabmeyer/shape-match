// "partition of unity" plot (idea from DesignMentor's Partition of Unity window):
// all B-spline basis functions N_{i,p}(t) of one parameter direction, the current t,
// and a bar with the values at t stacked on each other: they always add up to 1
// dragging in the plot moves t
import { basisFuns, findSpan } from './nurbs.ts';

const SAMPLES = 200;
const PAD = { left: 28, right: 46, top: 12, bottom: 22 }; // CSS pixels; the bar sits in the right margin

// colour-blind friendly (Okabe-Ito), one per basis function
const COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999'];

// all n + 1 basis functions at t (most of them 0)
export function allBasis(count: number, degree: number, knots: number[], t: number): number[] {
  const values = new Array<number>(count).fill(0);
  const span = findSpan(count - 1, degree, t, knots);
  basisFuns(span, t, degree, knots).forEach((value, k) => { values[span - degree + k] = value; });
  return values;
}

export class PartitionView {
  t = 0.5;

  private canvas: HTMLCanvasElement;
  private name: string; // 'u' or 'v'
  private count: number;
  private degree: number;
  private knots: number[];
  private onDrag: (t: number) => void;
  private dragging = false;

  constructor(canvas: HTMLCanvasElement, name: string, count: number, degree: number, knots: number[], onDrag: (t: number) => void) {
    this.canvas = canvas;
    this.name = name;
    this.count = count;
    this.degree = degree;
    this.knots = knots;
    this.onDrag = onDrag;

    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      this.dragging = true;
      this.drag(e);
    });
    canvas.addEventListener('pointermove', (e) => { if (this.dragging) this.drag(e); });
    canvas.addEventListener('pointerup', () => { this.dragging = false; });
    canvas.addEventListener('pointercancel', () => { this.dragging = false; });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  private drag(e: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    const plotWidth = r.width - PAD.left - PAD.right;
    this.t = Math.min(1, Math.max(0, (e.clientX - r.left - PAD.left) / plotWidth));
    this.onDrag(this.t);
  }

  draw(): void {
    const canvas = this.canvas;
    if (canvas.clientWidth === 0) return; // section is collapsed
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);

    const plotW = width - PAD.left - PAD.right;
    const plotH = height - PAD.top - PAD.bottom;
    const x = (t: number) => PAD.left + t * plotW;
    const y = (value: number) => PAD.top + (1 - value) * plotH;
    const { count, degree, knots } = this;

    // axes: 0 and 1, and the knots (distinct values) as ticks
    ctx.font = '11px system-ui, sans-serif';
    ctx.strokeStyle = '#3a3f4b';
    ctx.lineWidth = 1;
    ctx.strokeRect(x(0), y(1), plotW, plotH);
    ctx.setLineDash([3, 3]);
    for (const knot of new Set(knots)) {
      if (knot <= 0 || knot >= 1) continue;
      ctx.beginPath();
      ctx.moveTo(x(knot), y(0));
      ctx.lineTo(x(knot), y(1));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = '#9aa0aa';
    ctx.textAlign = 'right';
    ctx.fillText('1', x(0) - 5, y(1) + 4);
    ctx.fillText('0', x(0) - 5, y(0) + 4);
    ctx.textAlign = 'center';
    for (const knot of new Set(knots)) {
      ctx.fillText(Number.isInteger(knot) ? String(knot) : knot.toFixed(2), x(knot), y(0) + 14);
    }

    // which functions are non-zero at t: span - p ... span
    const span = findSpan(count - 1, degree, this.t, knots);
    const active = (i: number) => i >= span - degree && i <= span;

    // the curves, sampled; the active ones last (on top) and thicker
    const samples: number[][] = [];
    for (let s = 0; s <= SAMPLES; s++) samples.push(allBasis(count, degree, knots, s / SAMPLES));
    const order = [...Array(count).keys()].sort((a, b) => Number(active(a)) - Number(active(b)));
    for (const i of order) {
      ctx.strokeStyle = COLORS[i % COLORS.length];
      ctx.globalAlpha = active(i) ? 1 : 0.35;
      ctx.lineWidth = active(i) ? 2.5 : 1.25;
      ctx.beginPath();
      samples.forEach((values, s) => {
        const px = x(s / SAMPLES);
        const py = y(values[i]);
        if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      // label i at the highest point of the curve
      let best = 0;
      samples.forEach((values, s) => { if (values[i] > samples[best][i]) best = s; });
      ctx.fillStyle = COLORS[i % COLORS.length];
      ctx.textAlign = 'center';
      ctx.fillText(String(i), x(best / SAMPLES), y(samples[best][i]) - 4);
    }
    ctx.globalAlpha = 1;

    // the current t: a line, and a dot on every active curve
    const values = allBasis(count, degree, knots, this.t);
    ctx.strokeStyle = '#ff40bf';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x(this.t), y(0));
    ctx.lineTo(x(this.t), y(1));
    ctx.stroke();
    for (let i = 0; i < count; i++) {
      if (!active(i)) continue;
      ctx.beginPath();
      ctx.arc(x(this.t), y(values[i]), 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS[i % COLORS.length];
      ctx.fill();
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.fillStyle = '#e6e8ec';
    ctx.textAlign = 'center';
    ctx.fillText(this.name, x(this.t), y(1) - 2);

    // the bar: the values at t stacked; together exactly 1
    const barX = x(1) + 14;
    const barW = 16;
    let bottom = 0;
    for (let i = 0; i < count; i++) {
      if (values[i] <= 0) continue;
      ctx.fillStyle = COLORS[i % COLORS.length];
      ctx.fillRect(barX, y(bottom + values[i]), barW, y(bottom) - y(bottom + values[i]));
      bottom += values[i];
    }
    ctx.strokeStyle = '#9aa0aa';
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, y(1), barW, y(0) - y(1));
    ctx.fillStyle = '#9aa0aa';
    ctx.fillText('Σ', barX + barW / 2, y(0) + 14);
  }
}
