# shape_detector
This project implements a complete geometric shape detection system capable of identifying and classifying shapes such as circles, triangles, rectangles, pentagons, and stars from image data. The solution processes raw pixel data without using any external computer vision libraries, instead relying on core image processing operations and mathematical shape analysis. The detection pipeline includes grayscale conversion, thresholding, connected-component extraction, boundary tracing, convex hull estimation, and vertex-based classification. It is designed to handle clean shapes, noisy scenes, overlapping objects, and rotated shapes while producing precise bounding boxes, center coordinates, area calculations, and confidence scores. This project was created as part of a technical evaluation challenge, following strict constraints and performance requirements.

# Shape Detector — Candidate Submission

This submission implements the `detectShapes()` method using only browser-native
APIs and basic math, as required.

## How it works (high level)

**Pipeline**

1. **Grayscale** conversion (BT.709 luminance).
2. **Otsu thresholding** → binary (black shapes on white).
3. **Morphological denoise** (single 3×3 open-like pass).
4. **8-connected components** to isolate each shape.
5. **Boundary tracing** (Moore neighbor) to get an **ordered contour**.
6. **Geometry features** per component:
   - Area (pixel count), centroid (mean position)
   - Axis-aligned bounding box and extent (area / bbox area)
   - Perimeter from contour length
   - Convex hull (monotonic chain) and **solidity** (area / hull area)
   - RDP simplification to estimate **vertex count**
   - **Circularity** = \( 4πA / P² \)
7. **Classification (rule-based)**:
   - **Circle** if `circularity ≥ 0.78` and `0.70 ≤ extent ≤ 0.90`
   - **Star** if `solidity < 0.86` and `vertex count ≥ 8`
   - **Triangle** if `vertex count == 3`
   - **Rectangle** if `vertex count == 4` (works for rotated too)
   - **Pentagon** if `vertex count == 5`
   - Fallback compares circle-vs-rectangle scores
8. **Confidence** (simple & stable – Option A):
   - Weighted closeness to each class’ ideal features (e.g., circularity near 1.0 for circle,
     solidity near 1.0 for convex polygons, extent near expected values).
   - Returns a number in `[0,1]`.

**Outputs** satisfy the required format:

```ts
type DetectedShape = {
  type: 'circle'|'triangle'|'rectangle'|'pentagon'|'star';
  confidence: number;
  boundingBox: { x:number; y:number; width:number; height:number };
  center: { x:number; y:number };
  area: number;
}

