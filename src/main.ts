/* -----------------------------------------------------------------------------
  Shape Detector — Vanilla TS + Canvas APIs
  Implements: grayscale → Otsu threshold → CC labeling → contour tracing
  → geometric features → rule-based classification (circle/triangle/rectangle/
  pentagon/star). No external libraries.

  Satisfies challenge constraints:
  - Works purely from ImageData
  - No ML / no OpenCV
  - Computes bounding boxes, centers, areas, confidence
  - Targets IoU / center / area tolerances given in README
  - < 2s per image for the provided sizes
----------------------------------------------------------------------------- */

import "./style.css";
import { SelectionManager } from "./ui-utils.js";
import { EvaluationManager } from "./evaluation-manager.js";

/** ------------------- types exactly as provided by the scaffold ------------------- */
export interface Point { x: number; y: number; }

export interface DetectedShape {
  type: "circle" | "triangle" | "rectangle" | "pentagon" | "star";
  confidence: number;
  boundingBox: { x: number; y: number; width: number; height: number; };
  center: Point;
  area: number;
}

export interface DetectionResult {
  shapes: DetectedShape[];
  processingTime: number;
  imageWidth: number;
  imageHeight: number;
}

/** ------------------- small math helpers ------------------- */
const PI = Math.PI;
const TAU = 2 * Math.PI;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const sqr = (v: number) => v * v;
const dist = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

/** Ramer–Douglas–Peucker simplification for polygonal contour */
function rdp(points: Point[], epsilon: number): Point[] {
  if (points.length < 3) return points.slice();
  const stack: Array<[number, number]> = [[0, points.length - 1]];
  const keep = new Array(points.length).fill(false);
  keep[0] = keep[points.length - 1] = true;

  function perpendicularDistance(p: Point, a: Point, b: Point): number {
    const dx = b.x - a.x, dy = b.y - a.y;
    if (dx === 0 && dy === 0) return Math.hypot(p.x - a.x, p.y - a.y);
    const t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
    const px = a.x + t * dx, py = a.y + t * dy;
    return Math.hypot(p.x - px, p.y - py);
  }

  while (stack.length) {
    const [start, end] = stack.pop()!;
    let maxD = -1, idx = -1;
    for (let i = start + 1; i < end; i++) {
      const d = perpendicularDistance(points[i], points[start], points[end]);
      if (d > maxD) { maxD = d; idx = i; }
    }
    if (maxD > epsilon) {
      keep[idx] = true;
      stack.push([start, idx], [idx, end]);
    }
  }

  const out: Point[] = [];
  for (let i = 0; i < points.length; i++) if (keep[i]) out.push(points[i]);
  return out;
}

/** Monotonic chain convex hull (returns hull in CCW order, unique points) */
function convexHull(pts: Point[]): Point[] {
  if (pts.length <= 3) return pts.slice();
  const p = pts.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);

  const cross = (o: Point, a: Point, b: Point) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Point[] = [];
  for (const pt of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], pt) <= 0) {
      lower.pop();
    }
    lower.push(pt);
  }
  const upper: Point[] = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const pt = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pt) <= 0) {
      upper.pop();
    }
    upper.push(pt);
  }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

/** Shoelace polygon area (positive for CCW) */
function polygonArea(poly: Point[]): number {
  if (poly.length < 3) return 0;
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    a += (poly[j].x * poly[i].y) - (poly[i].x * poly[j].y);
  }
  return Math.abs(a) / 2;
}

/** Perimeter length along an ordered contour */
function pathPerimeter(pts: Point[], closed = true): number {
  if (pts.length < 2) return 0;
  let p = 0;
  for (let i = 1; i < pts.length; i++) p += dist(pts[i - 1], pts[i]);
  if (closed) p += dist(pts[pts.length - 1], pts[0]);
  return p;
}

/** 8-neighborhood directions (Moore neighbor tracing) */
const N8 = [
  { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }, { x: -1, y: 1 },
  { x: -1, y: 0 }, { x: -1, y: -1 }, { x: 0, y: -1 }, { x: 1, y: -1 }
];

/** Trace an ordered boundary around a foreground component (binary) */
function traceContour(bin: Uint8Array, w: number, h: number, sx: number, sy: number): Point[] {
  // Find a boundary start: first foreground pixel with a background neighbor
  let x = sx, y = sy;
  outer: for (let j = sy; j < h; j++) {
    for (let i = (j === sy ? sx : 0); i < w; i++) {
      const idx = j * w + i;
      if (!bin[idx]) continue;
      // if any 4-neighbor is background, it’s a boundary
      if ((i > 0 && !bin[idx - 1]) || (i < w - 1 && !bin[idx + 1]) ||
          (j > 0 && !bin[idx - w]) || (j < h - 1 && !bin[idx + w])) {
        x = i; y = j; break outer;
      }
    }
  }
  const contour: Point[] = [];
  // Moore-Neighbor tracing
  let px = x, py = y, dir = 0, first = true;
  do {
    contour.push({ x: px, y: py });
    // search neighbors starting from dir-2 (turn left)
    let ndir = (dir + 6) % 8;
    let found = false;
    for (let k = 0; k < 8; k++) {
      const d = (ndir + k) % 8;
      const nx = px + N8[d].x, ny = py + N8[d].y;
      if (nx >= 0 && nx < w && ny >= 0 && ny < h && bin[ny * w + nx]) {
        // move to boundary neighbor
        px = nx; py = ny; dir = d; found = true; break;
      }
    }
    if (!found) break;
    if (!first && px === x && py === y) break;
    first = false;
  } while (contour.length < 10000); // safety
  return contour;
}

/** ------------------- image → binary via grayscale + Otsu ------------------- */
function toGrayscale(img: ImageData): Uint8Array {
  const { data, width, height } = img;
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    // luminance (BT.709)
    out[p] = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) | 0;
  }
  return out;
}

function otsuThreshold(gray: Uint8Array): number {
  const hist = new Array<number>(256).fill(0);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
  const total = gray.length;

  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];

  let sumB = 0, wB = 0, maxVar = 0, threshold = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB; if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > maxVar) { maxVar = between; threshold = t; }
  }
  return threshold;
}

function binarize(gray: Uint8Array, w: number, h: number): Uint8Array {
  const t = otsuThreshold(gray);
  const bin = new Uint8Array(w * h);
  // Our shapes are black on white → foreground is (gray <= t)
  for (let i = 0; i < gray.length; i++) bin[i] = gray[i] <= t ? 1 : 0;

  // small opening to remove salt noise (1 iteration 3x3)
  const tmp = new Uint8Array(bin);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++)
          s += tmp[(y + dy) * w + (x + dx)];
      bin[y * w + x] = s >= 5 ? 1 : 0;
    }
  }
  return bin;
}

/** ------------------- connected components (8-connect) ------------------- */
type Component = {
  pixels: Point[];       // sample points (not all, to save memory)
  bbox: { minx: number; miny: number; maxx: number; maxy: number; };
  area: number;          // pixel count
  centroid: Point;       // mean of pixels
  contour: Point[];      // ordered boundary points
  perimeter: number;
  hull: Point[];
  hullArea: number;
};

function extractComponents(bin: Uint8Array, w: number, h: number): Component[] {
  const V = new Uint8Array(w * h); // visited
  const comps: Component[] = [];

  const pushIfInside = (x: number, y: number, q: number[]) => {
    if (x >= 0 && x < w && y >= 0 && y < h) {
      const idx = y * w + x;
      if (!V[idx] && bin[idx]) { V[idx] = 1; q.push(x, y); }
    }
  };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!bin[idx] || V[idx]) continue;

      // BFS
      const q: number[] = [x, y];
      V[idx] = 1;
      let minx = x, maxx = x, miny = y, maxy = y;
      let area = 0, cx = 0, cy = 0;
      const samples: Point[] = [];

      while (q.length) {
        const yy = q.pop()!, xx = q.pop()!;
        const id = yy * w + xx;
        area++; cx += xx; cy += yy;
        if ((area & 15) === 0) samples.push({ x: xx, y: yy }); // downsample storing
        if (xx < minx) minx = xx; if (xx > maxx) maxx = xx;
        if (yy < miny) miny = yy; if (yy > maxy) maxy = yy;

        pushIfInside(xx + 1, yy, q); pushIfInside(xx - 1, yy, q);
        pushIfInside(xx, yy + 1, q); pushIfInside(xx, yy - 1, q);
        pushIfInside(xx + 1, yy + 1, q); pushIfInside(xx - 1, yy - 1, q);
        pushIfInside(xx + 1, yy - 1, q); pushIfInside(xx - 1, yy + 1, q);
      }

      if (area < Math.max(40, (w * h) * 0.001)) continue; // discard tiny specks

      const centroid = { x: cx / area, y: cy / area };
      const contour = traceContour(bin, w, h, minx, miny);
      const perimeter = pathPerimeter(contour, true);
      const hull = convexHull(contour);
      const hullArea = polygonArea(hull);

      comps.push({
        pixels: samples,
        bbox: { minx, miny, maxx, maxy },
        area,
        centroid,
        contour,
        perimeter,
        hull,
        hullArea
      });
    }
  }
  return comps;
}

/** ------------------- classification ------------------- */
type ShapeKind = DetectedShape["type"];

function classify(comp: Component): { kind: ShapeKind; confidence: number } {
  const { bbox, area, centroid, contour, perimeter, hull, hullArea } = comp;
  const bbW = bbox.maxx - bbox.minx + 1;
  const bbH = bbox.maxy - bbox.miny + 1;
  const bbArea = bbW * bbH;

  const extent = area / bbArea;                  // fill ratio inside bbox
  const solidity = hullArea > 0 ? area / hullArea : 0; // convexity measure
  const circularity = perimeter > 0 ? (4 * PI * area) / (perimeter * perimeter) : 0; // 1.0 for perfect circle

  // Ordered + simplified contour to estimate vertices
  const eps = Math.max(2, Math.min(bbW, bbH) * 0.04); // 4% of size
  const simplified = rdp(contour, eps);
  // remove very-close duplicates in the closed polygon
  const verts: Point[] = [];
  for (const p of simplified) {
    if (verts.length === 0 || dist(verts[verts.length - 1], p) > 2) verts.push(p);
  }
  if (verts.length > 2 && dist(verts[0], verts[verts.length - 1]) < 2) verts.pop();
  const vcount = verts.length;

  // --- Heuristics chosen to match provided SVGs (black-on-white) ---

  // 1) Circle check first (stable on circularity + extent)
  if (circularity >= 0.78 && extent >= 0.70 && extent <= 0.90) {
    const conf = clamp01(0.65
      + 0.35 * clamp01((circularity - 0.78) / 0.22)
      + 0.10 * (1 - Math.abs(extent - 0.79) / 0.21)
    );
    return { kind: "circle", confidence: conf };
  }

  // 2) Star: non-convex (low solidity) and many alternating corners
  if (solidity < 0.86 && vcount >= 8) {
    // confidence stronger with lower solidity and more vertices
    const conf = clamp01(0.55 + 0.3 * (0.9 - solidity) + 0.02 * Math.min(12, vcount));
    return { kind: "star", confidence: conf };
  }

  // 3) Polygonal classes by vertex count
  if (vcount === 3) {
    const conf = clamp01(0.65 + 0.25 * (1 - Math.abs(extent - 0.5) / 0.5));
    return { kind: "triangle", confidence: conf };
  }

  if (vcount === 4) {
    // rectangle (works for rotated too) — near 90° angles increases solidity ≈ 1
    const aspect = bbW >= bbH ? bbW / bbH : bbH / bbW;
    const conf = clamp01(0.65
      + 0.2 * Math.min(1, solidity)        // convex
      + 0.15 * clamp01(1 - Math.abs(extent - 0.95)) // filled bbox
      + 0.05 * clamp01(1 - Math.abs(aspect - Math.round(aspect)) / Math.max(1, aspect)) // (soft)
    );
    return { kind: "rectangle", confidence: conf };
  }

  if (vcount === 5) {
    const conf = clamp01(0.6
      + 0.25 * clamp01(1 - Math.abs(extent - 0.70))
      + 0.15 * Math.min(1, solidity)
    );
    return { kind: "pentagon", confidence: conf };
  }

  // Fallback: choose best of circle vs rectangle based on features
  const circleScore = 0.5 * circularity + 0.5 * (1 - Math.abs(extent - 0.79));
  const rectScore = 0.5 * solidity + 0.5 * (extent > 0.85 ? 1 : extent);
  if (circleScore >= rectScore) {
    return { kind: "circle", confidence: clamp01(0.55 + 0.3 * circleScore) };
  }
  return { kind: "rectangle", confidence: clamp01(0.55 + 0.3 * rectScore) };
}

/** ------------------- public API class (as required) ------------------- */
export class ShapeDetector {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  /**
   * MAIN ALGORITHM — detect shapes from ImageData
   * Returns DetectedShape[] with area, center, bounding box, confidence
   */
  async detectShapes(imageData: ImageData): Promise<DetectionResult> {
    const t0 = performance.now();

    const { width: w, height: h } = imageData;
    const gray = toGrayscale(imageData);
    const bin = binarize(gray, w, h);
    const comps = extractComponents(bin, w, h);

    const shapes: DetectedShape[] = comps.map((c) => {
      const { kind, confidence } = classify(c);
      const bb = c.bbox;
      const center = { x: c.centroid.x, y: c.centroid.y };
      const area = c.area;

      return {
        type: kind,
        confidence: clamp01(confidence),
        boundingBox: { x: bb.minx, y: bb.miny, width: bb.maxx - bb.minx + 1, height: bb.maxy - bb.miny + 1 },
        center,
        area
      };
    });

    const processingTime = performance.now() - t0;
    return { shapes, processingTime, imageWidth: w, imageHeight: h };
  }

  loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.width;
        this.canvas.height = img.height;
        this.ctx.drawImage(img, 0, 0);
        const imageData = this.ctx.getImageData(0, 0, img.width, img.height);
        resolve(imageData);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }
}

/** ------------------- app wiring (unchanged) ------------------- */
class ShapeDetectionApp {
  private detector: ShapeDetector;
  private imageInput: HTMLInputElement;
  private resultsDiv: HTMLDivElement;
  private testImagesDiv: HTMLDivElement;
  private evaluateButton: HTMLButtonElement;
  private evaluationResultsDiv: HTMLDivElement;
  private selectionManager: SelectionManager;
  private evaluationManager: EvaluationManager;

  constructor() {
    const canvas = document.getElementById("originalCanvas") as HTMLCanvasElement;
    this.detector = new ShapeDetector(canvas);

    this.imageInput = document.getElementById("imageInput") as HTMLInputElement;
    this.resultsDiv = document.getElementById("results") as HTMLDivElement;
    this.testImagesDiv = document.getElementById("testImages") as HTMLDivElement;
    this.evaluateButton = document.getElementById("evaluateButton") as HTMLButtonElement;
    this.evaluationResultsDiv = document.getElementById("evaluationResults") as HTMLDivElement;

    this.selectionManager = new SelectionManager();
    this.evaluationManager = new EvaluationManager(this.detector, this.evaluateButton, this.evaluationResultsDiv);

    this.setupEventListeners();
    this.loadTestImages().catch(console.error);
  }

  private setupEventListeners(): void {
    this.imageInput.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (file) await this.processImage(file);
    });

    this.evaluateButton.addEventListener("click", async () => {
      const selectedImages = this.selectionManager.getSelectedImages();
      await this.evaluationManager.runSelectedEvaluation(selectedImages);
    });
  }

  private async processImage(file: File): Promise<void> {
    try {
      this.resultsDiv.innerHTML = "<p>Processing...</p>";
      const imageData = await this.detector.loadImage(file);
      const results = await this.detector.detectShapes(imageData);
      this.displayResults(results);
    } catch (error) {
      this.resultsDiv.innerHTML = `<p>Error: ${error}</p>`;
    }
  }

  private displayResults(results: DetectionResult): void {
    const { shapes, processingTime } = results;

    let html = `
      <p><strong>Processing Time:</strong> ${processingTime.toFixed(2)}ms</p>
      <p><strong>Shapes Found:</strong> ${shapes.length}</p>
    `;

    if (shapes.length > 0) {
      html += "<h4>Detected Shapes:</h4><ul>";
      shapes.forEach((shape) => {
        html += `
          <li>
            <strong>${shape.type.charAt(0).toUpperCase() + shape.type.slice(1)}</strong><br>
            Confidence: ${(shape.confidence * 100).toFixed(1)}%<br>
            Center: (${shape.center.x.toFixed(1)}, ${shape.center.y.toFixed(1)})<br>
            Area: ${shape.area.toFixed(1)}px²
          </li>
        `;
      });
      html += "</ul>";
    } else {
      html += "<p>No shapes detected.</p>";
    }

    this.resultsDiv.innerHTML = html;
  }

  private async loadTestImages(): Promise<void> {
    try {
      // your project uses test-images-data.ts (or test-image-data.ts). Import whichever exists.
      let module: any;
      try {
        module = await import("./test-images-data.js");
      } catch {
        module = await import("./test-image-data.js");
      }
      const testImages = module.testImages;
      const imageNames = module.getAllTestImageNames();

      let html =
        '<h4>Click to upload your own image or use test images for detection. Right-click test images to select/deselect for evaluation:</h4><div class="evaluation-controls"><button id="selectAllBtn">Select All</button><button id="deselectAllBtn">Deselect All</button><span class="selection-info">0 images selected</span></div><div class="test-images-grid">';

      html += `
        <div class="test-image-item upload-item" onclick="triggerFileUpload()">
          <div class="upload-icon">📁</div>
          <div class="upload-text">Upload Image</div>
          <div class="upload-subtext">Click to select file</div>
        </div>
      `;

      imageNames.forEach((imageName: string) => {
        const dataUrl = testImages[imageName as keyof typeof testImages];
        const displayName = imageName.replace(/[_-]/g, " ").replace(/\.(svg|png)$/i, "");
        html += `
          <div class="test-image-item" data-image="${imageName}"
               onclick="loadTestImage('${imageName}', '${dataUrl}')"
               oncontextmenu="toggleImageSelection(event, '${imageName}')">
            <img src="${dataUrl}" alt="${imageName}">
            <div>${displayName}</div>
          </div>
        `;
      });

      html += "</div>";
      this.testImagesDiv.innerHTML = html;

      this.selectionManager.setupSelectionControls();

      (window as any).loadTestImage = async (name: string, dataUrl: string) => {
        try {
          const response = await fetch(dataUrl);
          const blob = await response.blob();
          const type = blob.type || "image/svg+xml";
          const file = new File([blob], name, { type });
          const imageData = await this.detector.loadImage(file);
          const results = await this.detector.detectShapes(imageData);
          this.displayResults(results);
          console.log(`Loaded test image: ${name}`);
        } catch (error) {
          console.error("Error loading test image:", error);
        }
      };

      (window as any).toggleImageSelection = (event: MouseEvent, imageName: string) => {
        event.preventDefault();
        this.selectionManager.toggleImageSelection(imageName);
      };

      (window as any).triggerFileUpload = () => this.imageInput.click();
    } catch (error) {
      this.testImagesDiv.innerHTML = `
        <p>Test images not available. Please ensure test-images-data.ts is present.</p>`;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => new ShapeDetectionApp());
