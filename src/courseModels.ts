// the course test models (open patch, vase, torus) with reference values of S(u, v), see test-models/README.md
// every JSON file in test-models/ is loaded; with no files the list is empty and the tests that use it are skipped
import type { ControlNet, Point3 } from './controlNet.ts';

interface CourseModelFile {
  name: string;
  source: string;
  tolerance?: number;
  net: {
    countU: number; countV: number; p: number; q: number;
    knotsU: number[]; knotsV: number[];
    points: [number, number, number, number][]; // Cartesian x, y, z and weight w
  };
  references: { u: number; v: number; point: Point3 }[];
}

export interface CourseModel {
  name: string;
  source: string;
  tolerance: number;
  net: ControlNet;
  references: { u: number; v: number; point: Point3 }[];
}

const DEFAULT_TOLERANCE = 1e-6;

function toModel(file: CourseModelFile): CourseModel {
  const { countU, countV, p, q, knotsU, knotsV, points } = file.net;
  if (points.length !== countU * countV) {
    throw new Error(`${file.name}: ${points.length} control points, expected ${countU} x ${countV}`);
  }
  const homogeneous = new Float32Array(points.flatMap(([x, y, z, w]) => [w * x, w * y, w * z, w]));
  return {
    name: file.name,
    source: file.source,
    tolerance: file.tolerance ?? DEFAULT_TOLERANCE,
    net: { countU, countV, p, q, knotsU, knotsV, points: homogeneous },
    references: file.references,
  };
}

const files = import.meta.glob<CourseModelFile>('../test-models/*.json', { eager: true, import: 'default' });

export const courseModels: CourseModel[] = Object.values(files).map(toModel);
