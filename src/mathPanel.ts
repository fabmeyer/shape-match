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

  // the two stages (the two loops of the compute shader): walkers along u, then a curve along v
  const sub = (a: string, b: string) => `<msub><mi>${a}</mi><mrow>${b}</mrow></msub>`;
  const ij = '<mi>i</mi><mo>,</mo><mi>j</mi>';
  const stageA = `<math display="block">${sub('C', '<mi>j</mi>')}<mo stretchy="false">(</mo><mi>u</mi><mo stretchy="false">)</mo><mo>=</mo>
    <munder><mo>∑</mo><mi>i</mi></munder>${sub('N', '<mi>i</mi><mo>,</mo><mi>p</mi>')}<mo stretchy="false">(</mo><mi>u</mi><mo stretchy="false">)</mo>
    <mspace width="0.2em"></mspace>${sub('P', ij)}</math>`;
  const stageB = `<math display="block"><mi>S</mi><mo stretchy="false">(</mo><mi>u</mi><mo>,</mo><mi>v</mi><mo stretchy="false">)</mo><mo>=</mo>
    <munder><mo>∑</mo><mi>j</mi></munder>${sub('N', '<mi>j</mi><mo>,</mo><mi>q</mi>')}<mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo>
    <mspace width="0.2em"></mspace>${sub('C', '<mi>j</mi>')}<mo stretchy="false">(</mo><mi>u</mi><mo stretchy="false">)</mo></math>`;

  // walkers are homogeneous (w*x, w*y, w*z, w): show the 3D point, and the weight if it is not 1
  const weighted = s.walkers.some((c) => Math.abs(c[3] - 1) > 1e-9);
  let stages = `<table><tr><th>j</th><th>walker C<sub>j</sub>(u)</th>${weighted ? '<th>weight</th>' : ''}<th>N<sub>j,${q}</sub>(v)</th></tr>`;
  for (let l = q; l >= 0; l--) {
    const c = s.walkers[l];
    stages += `<tr><th>${j0 + l}</th><td>(${f(c[0] / c[3])}, ${f(c[1] / c[3])}, ${f(c[2] / c[3])})</td>`
      + (weighted ? `<td>${f(c[3])}</td>` : '') + `<td>${f(s.Nv[l])}</td></tr>`;
  }
  stages += '</table>';

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
    <h3>Two stages: curves of curves</h3>
    <p class="muted"><b class="green">Stage a, along u:</b> each active row j of control points is a curve in u.
      Its point at u is the walker:</p>
    ${stageA}
    <p class="muted"><b class="orange">Stage b, along v:</b> the walkers are the control points of a curve in v.
      Its point at v is the surface point:</p>
    ${stageB}
    ${stages}
    ${weighted ? '<p class="muted">With weights, both sums are taken in homogeneous coordinates (w·x, w·y, w·z, w) and divided by w only at the end.</p>' : ''}
    <h3>All at once</h3>
    <p class="muted">The same result in one step: share R<sub>i,j</sub>(u, v) of each active control point:</p>
    ${table}
    <p class="muted">Sum of all shares: ${f(sum)}</p>
    <p><b>S(u, v) = (${f(s.point[0])}, ${f(s.point[1])}, ${f(s.point[2])})</b></p>`;
}
