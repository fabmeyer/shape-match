// live grading on the Workbench: measures the Shape distance on the GPU whenever the net changed
// and turns it into a Grade (grading.ts); also runs the Shape match check of a Hand-in
//   setLevel:  a new Level. Measures its Noise floor and its start (calibration) in the background
//   update():  once per frame. Starts a measurement if the net changed and none is on its way
//   check():   the Shape match check. A fresh measurement of the surface as it is now, after the calibration
import type { ControlNet } from './controlNet.ts';
import { ErrorPass, measureLevel, type GpuShapeDistance } from './errorPass.ts';
import { debugLog } from './debugLog.ts';
import { calibrate, grade, gradingConfig, starMarks, type Calibration, type Grade, type GradingConfig } from './grading.ts';

export interface Measurement {
  distance: GpuShapeDistance;
  grade?: Grade;     // missing while the Level is still calibrating
  calibration?: Calibration;
  starMarks?: [number, number]; // Match where 1★ and 2★ begin
  ms: number;        // round trip to the GPU and back
}

export class Grader {
  onMeasured?: (m: Measurement) => void;
  onError?: (e: unknown) => void;

  private device: GPUDevice;
  private tessellatePipeline: GPUComputePipeline;
  private pass: ErrorPass;
  private dirty = true;      // the net changed since the last measurement
  private measuring = false; // a measurement is on its way back from the GPU
  private config: GradingConfig = gradingConfig();
  private calibration?: Calibration;
  private calibrating: Promise<void> = Promise.resolve();
  private level?: { target: ControlNet; start: ControlNet };
  private levelId = 0;       // a calibration that finishes after the next Level loaded is dropped

  constructor(device: GPUDevice, tessellatePipeline: GPUComputePipeline) {
    this.device = device;
    this.tessellatePipeline = tessellatePipeline;
    this.pass = new ErrorPass(device, tessellatePipeline);
  }

  get resolution(): number {
    return this.pass.resolution;
  }

  // a new error grid: the Noise floor depends on it, so the Level is calibrated again
  set resolution(res: number) {
    this.pass.resolution = res;
    this.recalibrate();
  }

  setLevel(target: ControlNet, start: ControlNet, overrides: Partial<GradingConfig> = {}): void {
    this.config = gradingConfig(overrides);
    this.level = { target, start };
    this.pass.setTarget(target);
    this.recalibrate();
  }

  // a new Learner surface, possibly with another Layout
  setLearner(net: ControlNet): void {
    this.pass.setLearner(net);
    this.dirty = true;
  }

  // the learner's control points changed (drag, weight, undo)
  learnerPointsChanged(points: Float32Array): void {
    this.pass.uploadLearnerPoints(points);
    this.dirty = true;
  }

  // per sample of the Learner surface: its distance to the Target (for the heatmap)
  get learnerDistances(): GPUBuffer | undefined {
    return this.pass.learnerDistances;
  }

  // call once per frame: at most one measurement at a time; changes made meanwhile are measured right after it
  update(): void {
    if (!this.dirty || this.measuring) return;
    this.measuring = true;
    this.dirty = false;
    this.measure()
      .then((m) => this.onMeasured?.(m))
      .catch((e) => this.onError?.(e))
      .finally(() => { this.measuring = false; });
  }

  // the Shape match check of a Hand-in: the surface as it is now, measured again (not the last live value)
  async check(): Promise<{ distance: GpuShapeDistance; grade: Grade }> {
    await this.calibrating;
    const m = await this.measure();
    if (!m.grade) throw new Error('The Level is not calibrated');
    return { distance: m.distance, grade: m.grade };
  }

  private async measure(): Promise<Measurement> {
    const start = performance.now();
    const distance = await this.pass.measure();
    const calibration = this.calibration;
    return {
      distance,
      calibration,
      grade: calibration && grade(distance, calibration, this.config),
      starMarks: calibration && starMarks(calibration, this.config),
      ms: performance.now() - start,
    };
  }

  private recalibrate(): void {
    this.calibration = undefined;
    this.dirty = true;
    if (!this.level) return;
    const id = ++this.levelId;
    const { target, start } = this.level;
    this.calibrating = measureLevel(this.device, this.tessellatePipeline, target, start, this.pass.resolution)
      .then(({ noise, start: startDistance }) => {
        if (id !== this.levelId) return; // another Level loaded meanwhile
        this.calibration = calibrate(noise, startDistance, this.config);
        debugLog.log('calibration', { res: noise.res, noiseRms: noise.rms, noiseHausdorff: noise.hausdorff, ...this.calibration });
        this.dirty = true; // grade the current surface now that the Level is calibrated
      })
      .catch((e) => this.onError?.(e));
  }

  destroy(): void {
    this.pass.destroy();
  }
}
