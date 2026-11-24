import type { ControlNet } from './controlNet.ts';
import type { SurfaceEval } from './nurbs.ts';

const f = (x: number) => x.toFixed(3);

// fills the formula panel with the numbers for the current (u, v)
// the formulas themselves are static MathML in index.html
export function updateMathPanel(target: HTMLElement, net: ControlNet, s: SurfaceEval): void {
  const { p, q } = net;
  const i0 = s.spanU - p; // first active control point in u
  const j0 = s.spanV - q;

  const basisRow = (name: string, first: number, degree: number, values: number[]) =>
    values.map((value, k) => `<span class="value">${name}<sub>${first + k},${degree}</sub> = ${f(value)}</span>`).join('');

  // table of R_ij: i to the right, j upwards (like the parameter space picture)
  let table = '<table><tr><th></th>';
  for (let k = 0; k <= p; k++) table += `<th>i = ${i0 + k}</th>`;
  table += '</tr>';
  let sum = 0;
  for (let l = q; l >= 0; l--) {
    table += `<tr><th>j = ${j0 + l}</th>`;
    for (let k = 0; k <= p; k++) {
      const r = s.R[l][k];
      sum += r;
      // the larger the share, the stronger the background
      table += `<td style="background: rgba(77, 230, 255, ${(0.75 * Math.min(1, Math.abs(r))).toFixed(3)})">${f(r)}</td>`;
    }
    table += '</tr>';
  }
  table += '</table>';

  target.innerHTML = `
    <p>At <b>u = ${f(s.u)}</b>, <b>v = ${f(s.v)}</b>:</p>
    <p class="muted">u lies in the knot span [${f(net.knotsU[s.spanU])}, ${f(net.knotsU[s.spanU + 1])}),
      so only the control points i = ${i0} … ${s.spanU} count:</p>
    <p>${basisRow('N', i0, p, s.Nu)}</p>
    <p class="muted">v lies in the knot span [${f(net.knotsV[s.spanV])}, ${f(net.knotsV[s.spanV + 1])}),
      so only j = ${j0} … ${s.spanV} count:</p>
    <p>${basisRow('N', j0, q, s.Nv)}</p>
    <p class="muted">Share R<sub>i,j</sub>(u, v) of each active control point:</p>
    ${table}
    <p class="muted">Sum of all shares: ${f(sum)}</p>
    <p><b>S(u, v) = (${f(s.point[0])}, ${f(s.point[1])}, ${f(s.point[2])})</b></p>`;
}
