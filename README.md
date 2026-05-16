# mlx-swift-diffrast

A Swift / MLX port of [NVIDIA nvdiffrast](https://github.com/NVlabs/nvdiffrast) —
differentiable rasterization for Apple Silicon via Metal.

The original nvdiffrast is CUDA + OpenGL. This port reimplements the four core ops
as JIT-compiled Metal compute kernels behind MLX custom VJPs so they participate in
`grad` / `valueAndGrad` like any other MLX op.

## Status

| Op | Forward | VJP | Notes |
|---|---|---|---|
| `interpolate` | ✅ | ✅ (attr + rast) | instanced mode only; no `rast_db` / `diff_attrs` yet |
| `rasterize`   | ⛔ | ⛔ | Planned: Metal `MTLRenderPipeline` offscreen pass writing `(u, v, z/w, tri_id+1)` |
| `texture`     | ⛔ | ⛔ | Planned: mipmapped sampler with derivatives |
| `antialias`   | ⛔ | ⛔ | Planned: edge-aware AA pass |

Rasterizer-output convention matches nvdiffrast exactly:
`rast[n, h, w, :] = (u, v, z/w, tri_id+1)`, where `tri_id+1 == 0` marks an empty pixel.

## Usage

```swift
import MLX
import MLXDiffRast

let out = DiffRast.interpolate(attr, rast: rast, tri: tri)
// out: [N, H, W, A]; differentiable w.r.t. attr and rast.
```

## Layout

- `Sources/MLXDiffRast/Interpolate.swift` — public op + Metal kernels (forward, d_attr, d_rast)
- `Sources/MLXDiffRast/MLXDiffRast.swift` — `DiffRast` namespace + porting status
- `Tests/MLXDiffRastTests/` — XCTest suite

## Build

Depends on `mlx-swift` 0.31.3 (pinned). Fetched on first build:

```sh
swift build
```

### Running tests

Use `xcodebuild`, not plain `swift test` — mlx-swift's `mlx.metallib` GPU
shader library is only emitted by an Xcode build, so the SwiftPM CLI runner
fails with `Failed to load the default metallib`.

```sh
xcodebuild -scheme mlx-swift-diffrast -destination 'platform=macOS' test
```

Or open the package in Xcode and run the test target there.

## Porting notes

See `tasks/todo.md` for the op-by-op porting plan.

## Attribution

This is an independent Swift / Metal reimplementation inspired by the paper
*"Modular Primitives for High-Performance Differentiable Rendering"* (Laine et
al., SIGGRAPH Asia 2020). No source from NVIDIA's nvdiffrast (which is
source-available under a non-OSS license) is used or redistributed here.

See [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) for the full citation,
dependency licenses, and BibTeX. This project is MIT-licensed; see
[`LICENSE`](LICENSE).
