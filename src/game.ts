// the game layer: 2D screens in front of the WebGPU canvas (plan §5)
//   Title → Board → Order → Workbench → Hand-in → Result → (next Level's Order | Workbench | Board)
// the Workbench is the editor itself: no screen is shown while the learner works
import './game.css';
import type { Grade } from './grading.ts';
import {
  isUnlocked, loadProgress, nextLevel, recordHandIn, saveProgress,
  type LevelBest, type Progress, type ProgressStorage,
} from './progress.ts';
import { showResult } from './resultScreen.ts';
import { debugLog } from './debugLog.ts';
import { starText } from './taskPanel.ts';
import type { Task } from './target.ts';

// after a passing Hand-in, the game moves on to the next Level by itself after this long
export const RESULT_COUNTDOWN_MS = 4000;

export interface GameHost {
  loadTask(task: Task): void;                  // puts the Task on the Workbench
  showBest(best: LevelBest | undefined): void; // the Best of the current Level in the Task panel
  onWorkbench?(): void;                        // the Workbench is in front again (ends the Hand-in animation)
}

type ScreenName = 'title' | 'board' | 'order' | 'result';

// "Level 1 · Local support" → "Local support"
function shortTitle(task: Task): string {
  return task.title.split('·').pop()!.trim();
}

export class Game {
  private tasks: Task[];
  private order: string[];
  private host: GameHost;
  private storage: ProgressStorage | null;
  private progress: Progress;
  private stopResult?: () => void;
  private screens: Record<ScreenName, HTMLElement>;

  constructor(tasks: Task[], host: GameHost, storage: ProgressStorage | null) {
    this.tasks = tasks;
    this.order = tasks.map((t) => t.id);
    this.host = host;
    this.storage = storage;
    this.progress = loadProgress(storage);
    this.screens = {
      title: document.getElementById('titleScreen')!,
      board: document.getElementById('boardScreen')!,
      order: document.getElementById('orderScreen')!,
      result: document.getElementById('resultScreen')!,
    };
    document.getElementById('titlePlay')!.addEventListener('click', () => this.showBoard());
    document.getElementById('toBoard')!.addEventListener('click', () => this.showBoard());
    document.getElementById('orderBack')!.addEventListener('click', () => this.showBoard());
  }

  // which screen is in front; null = the Workbench
  private show(name: ScreenName | null): void {
    this.stopResult?.();
    this.stopResult = undefined;
    for (const [key, el] of Object.entries(this.screens)) el.hidden = key !== name;
    if (name !== 'result') this.host.onWorkbench?.(); // leaving the Result: the surfaces go back to their places
    debugLog.log('screen', { name: name ?? 'workbench' });
  }

  showTitle(): void {
    this.show('title');
  }

  showBoard(): void {
    const path = document.getElementById('boardPath')!;
    path.replaceChildren(...this.tasks.map((task, k) => {
      const unlocked = isUnlocked(this.progress, this.order, task.id);
      const best = this.progress.best[task.id];
      const li = document.createElement('li');
      li.className = 'board-space' + (unlocked ? '' : ' locked') + (best && best.stars > 0 ? ' solved' : '');
      const button = document.createElement('button');
      button.className = 'space-button';
      button.disabled = !unlocked;
      button.textContent = unlocked ? String(k + 1) : '🔒';
      button.title = unlocked ? task.title : 'Solve the Level before to unlock this one';
      button.setAttribute('aria-label', `${task.title}${unlocked ? '' : ' (locked)'}`);
      button.addEventListener('click', () => this.showOrder(task));
      const name = document.createElement('span');
      name.className = 'space-name';
      name.textContent = shortTitle(task);
      const stars = document.createElement('span');
      stars.className = 'space-stars';
      stars.textContent = starText(best?.stars ?? 0);
      li.append(button, name, stars);
      return li;
    }));
    this.show('board');
  }

  showOrder(task: Task): void {
    if (!isUnlocked(this.progress, this.order, task.id)) return; // locked Levels can't be opened
    document.getElementById('orderTitle')!.textContent = task.title;
    document.getElementById('orderText')!.textContent = task.text;
    const best = this.progress.best[task.id];
    document.getElementById('orderBest')!.textContent = best
      ? `Your best: ${starText(best.stars)} ${best.match.toFixed(1)} %`
      : 'A new order: nobody has made this one yet.';
    const start = document.getElementById('orderStart')!;
    start.onclick = () => this.play(task);
    this.show('order');
    start.focus();
  }

  // the Task on the Workbench (also the developer shortcut ?task=<id>)
  play(task: Task): void {
    this.host.loadTask(task);
    this.host.showBest(this.progress.best[task.id]);
    this.show(null);
  }

  // a Hand-in: remember it, then the Result screen
  handIn(task: Task, grade: Grade): void {
    const { progress, result } = recordHandIn(this.progress, task.id, grade);
    debugLog.log('hand-in', { ...result, rms: grade.rms, hausdorff: grade.hausdorff });
    this.progress = progress;
    saveProgress(this.storage, progress);
    this.host.showBest(result.best);

    const nextId = nextLevel(this.order, task.id);
    const next = nextId ? this.tasks.find((t) => t.id === nextId) : undefined;
    this.show('result');
    this.stopResult = showResult(this.screens.result, result, task.title, {
      nextTitle: next?.title,
      countdownMs: RESULT_COUNTDOWN_MS,
      onNext: () => (next ? this.showOrder(next) : this.showBoard()),
      onStay: () => this.show(null), // back to the Workbench, the surface as it was
      onBoard: () => this.showBoard(),
    });
  }
}
