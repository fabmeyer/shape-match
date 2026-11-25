import type { ControlNet } from './controlNet.ts';
import { findSpan } from './nurbs.ts';

const MARGIN = 30; // space around the square for the labels, in CSS pixels

/**
 * 2D view of the parameter square [0, 1] x [0, 1]: knots as lines, the knot span
 * that contains the current (u, v) highlighted, and (u, v) as a draggable dot
 * u runs to the right, v runs up
 */
export class ParamSpaceView {
  u = 0.5;
  v = 0.5;

  private canvas: HTMLCanvasElement;
  net: ControlNet; // replaced when a Level loads or the Layout changes
  private onChange: () => void;
  private dragging = false;

  constructor(canvas: HTMLCanvasElement, net: ControlNet, onChange: () => void) {
    this.canvas = canvas;
    this.net = net;
    this.onChange = onChange;

    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      this.dragging = true;
      this.setFromMouse(e);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (this.dragging) this.setFromMouse(e);
    });
    const stop = () => { this.dragging = false; };
    canvas.addEventListener('pointerup', stop);
    canvas.addEventListener('pointercancel', stop);
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  // side length of the square in CSS pixels
  private side(): number {
    return Math.min(this.canvas.clientWidth, this.canvas.clientHeight) - 2 * MARGIN;
  }

  private setFromMouse(e: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    const side = this.side();
    this.u = Math.min(1, Math.max(0, (e.clientX - r.left - MARGIN) / side));
    this.v = Math.min(1, Math.max(0, 1 - (e.clientY - r.top - MARGIN) / side)); // canvas y runs down
    this.draw();
    this.onChange();
  }

  draw(): void {
    const canvas = this.canvas;
    if (canvas.clientWidth === 0) return; // panel is hidden
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * dpr);
    canvas.height = Math.round(canvas.clientHeight * dpr);
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

    const { net } = this;
    const side = this.side();
    const x = (u: number) => MARGIN + u * side;
    const y = (v: number) => MARGIN + (1 - v) * side;

    // the knot span that contains (u, v)
    const spanU = findSpan(net.countU - 1, net.p, this.u, net.knotsU);
    const spanV = findSpan(net.countV - 1, net.q, this.v, net.knotsV);
    const u0 = net.knotsU[spanU], u1 = net.knotsU[spanU + 1];
    const v0 = net.knotsV[spanV], v1 = net.knotsV[spanV + 1];
    ctx.fillStyle = 'rgba(77, 230, 255, 0.18)';
    ctx.fillRect(x(u0), y(v1), x(u1) - x(u0), y(v0) - y(v1));

    // knot lines: a knot that appears several times is drawn thicker
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = '#9aa0aa';
    ctx.strokeStyle = '#6b7280';
    const label = (t: number) => (Number.isInteger(t) ? String(t) : t.toFixed(2));
    for (const [knot, count] of multiplicities(net.knotsU)) {
      ctx.lineWidth = Math.min(count, 4) * 0.75;
      ctx.beginPath();
      ctx.moveTo(x(knot), y(0));
      ctx.lineTo(x(knot), y(1));
      ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(label(knot), x(knot), y(0) + 14);
    }
    for (const [knot, count] of multiplicities(net.knotsV)) {
      ctx.lineWidth = Math.min(count, 4) * 0.75;
      ctx.beginPath();
      ctx.moveTo(x(0), y(knot));
      ctx.lineTo(x(1), y(knot));
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(label(knot), x(0) - 6, y(knot) + 4);
    }
    ctx.fillStyle = '#e6e8ec';
    ctx.textAlign = 'center';
    ctx.fillText('u', x(0.5), y(0) + 27);
    ctx.fillText('v', x(0) - 22, y(0.5) + 4);

    // the current (u, v), same colour as the marker on the surface
    ctx.beginPath();
    ctx.arc(x(this.u), y(this.v), 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#ff40bf';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  }
}

// distinct knot values with how often each appears
function multiplicities(knots: number[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const knot of knots) counts.set(knot, (counts.get(knot) ?? 0) + 1);
  return counts;
}
