// the Result screen after a Hand-in: the reward moment, in the loud style of a party game
//   Passed:     a banner, the Stars pop in one by one, the Match counts up, confetti, "NEW BEST!",
//               then a countdown to the next Level ("Next Level" skips it, "Stay" cancels it)
//   Not passed: a friendly banner, the Match reached, back to the Workbench with the surface unchanged
// all animation is CSS (game.css) except the count-up and the countdown; both respect reduced motion
import type { HandInResult } from './progress.ts';
import { starText } from './taskPanel.ts';

export interface ResultActions {
  nextTitle?: string;      // the next Level, or none after the last one
  countdownMs: number;
  onNext: () => void;      // next Level's Order, or the Board after the last Level
  onStay: () => void;      // back to the Workbench
  onBoard: () => void;
}

const STAR_PATH = 'M50 4 L62 36 L96 38 L69 59 L79 93 L50 73 L21 93 L31 59 L4 38 L38 36 Z';
const CONFETTI_COLORS = ['#ffd23f', '#ff4f9a', '#2f8cff', '#2ecc71', '#ff8a1f', '#a66bff'];
const CONFETTI_PIECES = 90;
const INTRO_MS = 1600;      // banner, Stars and count-up before the countdown starts
const COUNT_UP_MS = 1100;

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function star(earned: boolean, k: number): HTMLElement {
  const wrap = el('span', `result-star${earned ? ' earned' : ''}`);
  wrap.style.setProperty('--k', String(k)); // stagger: one after the other
  wrap.innerHTML = `<svg viewBox="0 0 100 100" aria-hidden="true"><path d="${STAR_PATH}"/></svg>`;
  return wrap;
}

function confetti(): HTMLElement {
  const box = el('div', 'confetti');
  for (let k = 0; k < CONFETTI_PIECES; k++) {
    const piece = el('i', '');
    piece.style.left = `${Math.random() * 100}%`;
    piece.style.background = CONFETTI_COLORS[k % CONFETTI_COLORS.length];
    piece.style.animationDelay = `${Math.random() * 1.2}s`;
    piece.style.animationDuration = `${2.4 + Math.random() * 2}s`;
    piece.style.setProperty('--spin', `${Math.random() < 0.5 ? -1 : 1}`);
    piece.style.width = `${6 + Math.random() * 8}px`;
    box.append(piece);
  }
  return box;
}

function button(label: string, color: string, onClick: () => void): HTMLButtonElement {
  const b = el('button', `party-button ${color}`, label);
  b.addEventListener('click', onClick);
  return b;
}

// fills `root` with the Result; returns a function that stops its timers (the game calls it when leaving)
export function showResult(root: HTMLElement, r: HandInResult, levelTitle: string, actions: ResultActions): () => void {
  const timers: number[] = [];
  let frame = 0;
  const stop = () => {
    timers.forEach(clearTimeout);
    cancelAnimationFrame(frame);
  };

  root.className = `screen result-screen ${r.passed ? 'passed' : 'not-passed'}`;
  const card = el('div', 'result-card');
  const banner = el('div', 'result-banner', r.passed ? 'PASSED!' : r.match >= 50 ? 'SO CLOSE!' : 'NOT YET!');
  const level = el('p', 'result-level', levelTitle);
  const stars = el('div', 'result-stars');
  stars.setAttribute('aria-label', `${r.stars} of 3 Stars`);
  for (let k = 0; k < 3; k++) stars.append(star(k < r.stars, k));

  const match = el('div', 'result-match');
  const count = el('span', 'result-count', '0.0');
  match.append(count, el('span', 'result-unit', '%'), el('span', 'result-match-label', 'Match'));

  card.append(banner, level, stars, match);
  const stickers = el('div', 'stickers');
  if (r.newBest) stickers.append(el('div', 'sticker new-best', 'NEW BEST!'));
  if (r.firstPass && actions.nextTitle) stickers.append(el('div', 'sticker unlocked', 'Next Level unlocked!'));
  if (stickers.childElementCount > 0) card.append(stickers);
  card.append(el('p', 'result-best', `Your best: ${starText(r.best.stars)} ${r.best.match.toFixed(1)} %`));
  if (!r.passed) card.append(el('p', 'result-hint', 'Not passed yet. Compare with the Target and keep shaping!'));

  const buttons = el('div', 'result-buttons');
  if (r.passed) {
    const next = button(actions.nextTitle ? 'Next Level' : 'Back to the Board', 'green countdown', () => { stop(); actions.onNext(); });
    buttons.append(next, button('Stay and try for more Stars', 'blue', () => { stop(); actions.onStay(); }));
    // the countdown: a ring that fills on the Next button, then the game moves on by itself
    timers.push(window.setTimeout(() => {
      const start = performance.now();
      const tick = (now: number) => {
        const t = Math.min(1, (now - start) / actions.countdownMs);
        next.style.setProperty('--progress', String(t));
        next.dataset.seconds = String(Math.ceil((1 - t) * actions.countdownMs / 1000));
        if (t < 1) frame = requestAnimationFrame(tick);
        else actions.onNext();
      };
      frame = requestAnimationFrame(tick);
    }, reducedMotion() ? 0 : INTRO_MS));
  } else {
    buttons.append(button('Back to the Workbench', 'blue', () => { stop(); actions.onStay(); }));
  }
  buttons.append(button('Board', 'purple small', () => { stop(); actions.onBoard(); }));
  card.append(buttons);

  root.replaceChildren(el('div', 'burst'), ...(r.passed ? [confetti()] : []), card);

  // the Match counts up from 0
  if (reducedMotion()) {
    count.textContent = r.match.toFixed(1);
  } else {
    const start = performance.now();
    const up = (now: number) => {
      const t = Math.min(1, (now - start) / COUNT_UP_MS);
      count.textContent = (r.match * (1 - (1 - t) ** 3)).toFixed(1); // ease out
      if (t < 1) frame = requestAnimationFrame(up);
    };
    frame = requestAnimationFrame(up);
  }
  (buttons.firstElementChild as HTMLButtonElement).focus();
  return stop;
}
