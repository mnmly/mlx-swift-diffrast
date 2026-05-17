# mlx-swift-diffrast

A Swift / MLX port of [NVIDIA nvdiffrast](https://github.com/NVlabs/nvdiffrast) —
differentiable rasterization for Apple Silicon via Metal.

The original nvdiffrast is CUDA + OpenGL. This port reimplements the four core
ops as JIT-compiled Metal compute kernels behind MLX custom VJPs so they
participate in `grad` / `valueAndGrad` like any other MLX op.

## Status

All four nvdiffrast ops are implemented end-to-end with full gradient
correctness, validated by gradchecks against `MLX.grad` and by three working
inverse-rendering samples that converge against synthetic targets.

| Op | Forward | Backward | Features |
|---|---|---|---|
| `rasterize` | ✅ | ✅ d_pos | rast_db, range mode, depth peeling, tile binning, AABB rejection |
| `interpolate` | ✅ | ✅ d_attr / d_rast / d_rastDB | range mode, instanced mode, batch broadcast, diff_attrs `.all` / `.indices(_)` |
| `texture` | ✅ | ✅ d_tex / d_uv / d_uvDA | bilinear, nearest, trilinear, mipmap-nearest, mip pyramid construction, 3 boundary modes + cube, mipLevelBias (scalar & per-pixel) |
| `antialias` | ✅ | ✅ d_color / d_pos | topology hash, silhouette α coverage, non-manifold edge marking |

Conventions match nvdiffrast exactly:
- `pos` is clip-space `[N, V, 4]` (or `[V, 4]` + ranges); channels are `(x·w, y·w, z·w, w)`.
- `rast[n, h, w, :] = (u, v, z/w, tri_id+1)`; `tri_id+1 == 0` marks an empty pixel.
- Pixel `(h=0, w=0)` is top-left; NDC `+y` is up (vertical flip applied).
- `rastDB[n, h, w, :] = (du/dx, du/dy, dv/dx, dv/dy)` in per-pixel units.

## Usage

```swift
import MLX
import MLXDiffRast

// 1. Rasterize a mesh into per-pixel barycentrics + tri-ids.
let (rast, rastDB) = DiffRast.rasterize(
    pos, tri: tri, resolution: (height: 256, width: 256))

// 2. Interpolate per-vertex attributes (colors, normals, UVs, …) across pixels.
let (color, _)   = DiffRast.interpolate(vertexColors, rast: rast, tri: tri)
let (uv, uvDA)   = DiffRast.interpolate(
    vertexUVs, rast: rast, tri: tri, rastDB: rastDB, diffAttrs: .all)

// 3. Sample textures with mipmap-aware filtering driven by uvDA.
let sampled = DiffRast.texture(
    tex, uv: uv, uvDA: uvDA,
    filterMode: .linearMipmapLinear,
    boundaryMode: .clamp)

// 4. Silhouette-aware antialias — the gradient bridge from pixel loss to geometry.
let topology = DiffRast.antialiasConstructTopologyHash(tri)
let antialiased = DiffRast.antialias(
    color: sampled, rast: rast, pos: pos, tri: tri,
    topologyHash: topology)

// 5. Compute loss and propagate gradients all the way back to pos / colors / tex.
let loss = ((antialiased - target) * (antialiased - target)).mean()
// `MLX.grad(loss-fn)(pos)` gives ∂loss/∂pos through the entire chain.
```

## Inverse-rendering samples

Three runnable examples under `Sources/` validate the full stack against
synthetic targets:

```sh
xcodebuild -scheme diffrast-color-fit -destination 'platform=macOS' build
./path/to/Build/Products/Debug/diffrast-color-fit
```

| Tier | Sample | What it optimizes | Loss |
|---|---|---|---|
| 1 | `diffrast-color-fit` | per-vertex RGB on a fixed triangle | 0.028 → 0 |
| 2 | `diffrast-pose-fit`  | vertex positions of a quad via silhouette gradient | 0.034 → 0.0007 |
| 3 | `diffrast-mesh-fit`  | 7-vertex / 6-triangle hexagonal-pie mesh | 0.034 → 0.0002 |

Each writes an animated GIF of `target | current` side-by-side as the optimizer
runs.

## Build

```sh
swift build
```

Depends on `mlx-swift` 0.31.3 (pinned).

### Running tests

**Use `xcodebuild`, not plain `swift test`** — mlx-swift's `mlx.metallib` GPU
shader library is only emitted by an Xcode build, so the SwiftPM CLI runner
fails with `Failed to load the default metallib`.

```sh
xcodebuild -scheme mlx-swift-diffrast-Package -destination 'platform=macOS' test
```

Or open the package in Xcode and run the test target there. 54 tests total at
the current tag; covers forward correctness, finite-difference gradchecks
against `MLX.grad`, and inverse-rendering convergence.

## Layout

```
Sources/
├── MLXDiffRast/
│   ├── MLXDiffRast.swift        — DiffRast namespace + porting status
│   ├── Rasterize.swift          — rasterize: precompute, bin, fwd, fwd_db, bwd kernels
│   ├── Interpolate.swift        — interpolate basic + DA paths
│   ├── Texture.swift            — bilinear, nearest, trilinear, mipmap-nearest, cube
│   └── Antialias.swift          — topology hash, silhouette block, fwd / d_color / d_pos
├── MLXDiffRastExamples/         — shared helpers for samples (Adam, GIF writer, MLX↔CGImage)
├── ColorFitExample/             — tier-1 sample
├── PoseFitExample/              — tier-2 sample
└── MeshFitExample/              — tier-3 sample
```

## Attribution

This is an independent Swift / Metal reimplementation inspired by the paper
*"Modular Primitives for High-Performance Differentiable Rendering"* (Laine et
al., SIGGRAPH Asia 2020). No source from NVIDIA's nvdiffrast (which is
source-available under a non-OSS license) is used or redistributed here.

See [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) for the full citation,
dependency licenses, and BibTeX. This project is MIT-licensed; see
[`LICENSE`](LICENSE).
