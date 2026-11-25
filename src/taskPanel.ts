// the Task panel on the Workbench: the Order, the live Match meter, the live Stars and the Hand in button
// grading stays live while editing; only a Hand-in counts (it runs the Shape match check)
import type { Grade, Stars } from './grading.ts';

export function starText(stars: Stars): string {
  return '★'.repeat(stars) + '☆'.repeat(3 - stars);
}

export class TaskPanel {
  onHandIn?: () => void;

  private title = document.getElementById('taskTitle') as HTMLElement;
  private text = document.getElementById('taskText') as HTMLElement;
  private tools = document.getElementById('taskTools') as HTMLElement;
  private matchValue = document.getElementById('matchValue') as HTMLElement;
  private matchFill = document.getElementById('matchFill') as HTMLElement;
  private marks = [document.getElementById('mark1') as HTMLElement, document.getElementById('mark2') as HTMLElement];
  private liveStars = document.getElementById('liveStars') as HTMLElement;
  private best = document.getElementById('bestStars') as HTMLElement;
  private button = document.getElementById('handIn') as HTMLButtonElement;
  private checking = false;
  private ready = false;
  private calibrated = false;

  constructor() {
    this.button.addEventListener('click', () => {
      if (!this.checking && this.calibrated) this.onHandIn?.();
    });
  }

  setTask(title: string, text: string, tools: string): void {
    this.title.textContent = title;
    this.text.textContent = text;
    this.tools.textContent = tools;
    this.setCalibrating();
  }

  // a new Level: no grade until its Noise floor is measured
  setCalibrating(): void {
    this.calibrated = false;
    this.ready = false;
    this.matchValue.textContent = '…';
    this.matchFill.style.width = '0%';
    this.liveStars.textContent = starText(0);
    for (const mark of this.marks) mark.hidden = true;
    this.updateButton();
  }

  show(grade: Grade, marks: [number, number]): void {
    this.calibrated = true;
    this.ready = grade.passed;
    this.matchValue.textContent = `${grade.match.toFixed(1)} %`;
    this.matchFill.style.width = `${grade.match}%`;
    this.matchFill.dataset.stars = String(grade.stars);
    this.liveStars.textContent = starText(grade.stars);
    this.liveStars.dataset.stars = String(grade.stars);
    marks.forEach((m, k) => {
      this.marks[k].hidden = false;
      this.marks[k].style.left = `${m}%`;
    });
    this.updateButton();
  }

  // the learner's Best for this Level (hi-score), or nothing yet
  setBest(best: { stars: Stars; match: number } | undefined): void {
    this.best.textContent = best ? `Best ${starText(best.stars)} ${best.match.toFixed(1)} %` : 'No hand-in yet';
  }

  // while the Shape match check runs on the GPU, the button can't be clicked again
  setChecking(on: boolean): void {
    this.checking = on;
    this.updateButton();
  }

  private updateButton(): void {
    const b = this.button;
    b.disabled = this.checking || !this.calibrated;
    b.classList.toggle('ready', this.ready && !this.checking);
    b.textContent = this.checking ? 'Checking…' : !this.calibrated ? 'Calibrating…' : this.ready ? 'Hand in ★' : 'Hand in';
    b.title = this.ready
      ? 'Your surface is good enough to pass: hand it in'
      : 'Hand in your surface: the Shape match check decides Passed or Not passed';
  }
}
