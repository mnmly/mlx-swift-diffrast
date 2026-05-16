# mlx-swift-diffrast — porting plan

## Goal
Port NVIDIA nvdiffrast to MLX Swift / Metal, preserving the public API shape of
`nvdiffrast.torch` so PyTorch reference code can be ported with minimal changes.

## Acceptance criteria (overall)
- `interpolate`, `rasterize`, `texture`, `antialias` all available as MLX ops with VJPs.
- Forward outputs match PyTorch nvdiffrast within 1e-4 on a fixed test fixture.
- Gradients match within 1e-3 (atomic-add nondeterminism budget) on the same fixture.
- `swift test` passes on Apple Silicon (macOS 14+).

## Working notes
- `MLXFast.metalKernel` JIT compiles a body string; the function signature
  (`device const float* attr`, etc.) is auto-generated from `inputNames` / `outputNames`.
- Use `template:` for compile-time shape constants (N, H, W, A, V).
- For scatter-add gradients (vertex attrs), pass `atomicOutputs: true` and use
  `atomic_fetch_add_explicit`. Initialize the output array with `initValue: 0`.
- Custom autograd: wrap a closure pair with `CustomFunction { Forward(...); VJP(...) }`.
  Non-differentiable inputs (e.g. `tri` int32 indices) should be *captured*, not passed
  through the input list, otherwise MLX will try to take cotangents w.r.t. them.

## Milestones

### M1 — Package skeleton + `interpolate` (✅ DONE)
- [x] SPM package layout (`Package.swift`, `Sources/MLXDiffRast`, `Tests`)
- [x] Public `DiffRast.interpolate(attr, rast, tri, rastDB:, diffAttrs:)` API
- [x] Forward Metal kernels (basic + DA)
- [x] VJP kernels: d_attr (basic), d_attr_da, d_rast, d_rast_db
- [x] Range mode (`attr` shape `[V, A]`) via Swift-side broadcast
- [x] Batch broadcast (`attr.shape[0] == 1`)
- [x] `rast_db` / `diff_attrs = .all` pixel-derivative branch
- [x] Finite-difference gradcheck for attr/rast/rastDB on both basic and DA paths
- [ ] (deferred) `diff_attrs = .indices([Int])` subset selection — only `.all` supported today
- [ ] (deferred) Validate against PyTorch nvdiffrast on a real triangle mesh fixture
      — gradcheck covers internal correctness; deferring until M2 lands so we can
      compare end-to-end (rasterize → interpolate) against the reference.

Test count: 9/9 passing (forward constant, empty pixels, DA analytic, range-mode,
+ 5 gradchecks). `xcodebuild -scheme mlx-swift-diffrast -destination 'platform=macOS' test`

### M2 — `rasterize`
Decision: pure-compute software rasterizer for v1 (simpler MLX-stream
integration, no MTLTexture↔MLXArray plumbing). Can swap to an
MTLRenderPipeline backend later behind the same API if perf demands.

#### M2.1 — Forward only (✅ DONE)
- [x] `DiffRast.rasterize(pos, tri, resolution)` public API
- [x] Per-pixel compute kernel iterating over all triangles, signed-area edge
      functions, depth test, NDC z range check
- [x] Output layout `(u, v, z/w, tri_id+1)` matches what `interpolate` consumes
- [x] 4 tests including an end-to-end `rasterize → interpolate` check
- [x] Placeholder VJP returns zeros — backward deferred to M2.3

#### M2.2 — Pixel derivatives `rast_db` (✅ DONE)
- [x] Public API now returns `(rast, rastDB)` tuple; `gradDB: Bool = true` default
- [x] Second forward kernel `fwd_db` emits `(du/dx, du/dy, dv/dx, dv/dy)` in
      per-pixel units (chain rule through the NDC↔pixel mapping baked in)
- [x] `rast_db` analytic test on a known triangle
- [x] End-to-end `rasterize(gradDB=true) → interpolate(diffAttrs=.all)` test

#### M2.3 — Backward (✅ DONE)
- [x] Per-pixel compute kernel computes the full chain:
      screen-bary partials → perspective divide → atomic scatter-add into `d_pos`
- [x] Handles both `gradDB=true/false` via a single kernel — the no-DB path
      passes a zero `d_rast_db` so the rast_db terms drop out
- [x] `rast_db` values are recomputed from `pos` inline (saves a tensor input)
- [x] Two gradcheck tests (with and without DB) pass against MLX `grad` vs
      central FD on a fixture chosen to keep all pixels well inside the triangle
      so silhouette-edge discontinuities don't poison FD

**Test fixture lesson:** when gradchecking a rasterizer, pixel centers must not
lie ON triangle edges. Edge pixels flip coverage discretely under tiny `pos`
perturbations, making FD wildly disagree with the (correct) smooth-interior
analytic gradient. See `fullCoverPos()` for a triangle with edges >0.5 NDC
from every pixel center.

#### Deferred
- [ ] Range mode (`pos` shape `[V, 4]` + `ranges` tensor)
- [ ] `DepthPeeler` analog for transparency
- [ ] Replace per-pixel ∀-triangle loop with a tile-based hierarchical pass
      when triangle counts grow beyond a few thousand

### M3 — `texture` (✅ M3.1 DONE — bilinear; mipmap and cube deferred)
- [x] Bilinear forward with three boundary modes (wrap, clamp, zero)
- [x] VJP for `tex` (atomic scatter-add) and `uv` (per-pixel analytic)
- [x] 4 gradchecks across boundary modes + 5 forward correctness tests
- [ ] `texture_construct_mip` + mipmap-linear / linear-mipmap-linear filters
- [ ] `uv_da` / `mip_level_bias`-driven LOD selection
- [ ] Cube textures + `cube` boundary mode

Metal-kernel note: template args (e.g. `BOUNDARY`) are NOT visible inside the
`header:` block — they only resolve in the main kernel body. Helper functions
in the header must take what they need as explicit parameters.

### M4 — `antialias`
- [ ] `antialias_construct_topology_hash` — build edge hash from `tri`.
- [ ] Forward: detect silhouette edges per pixel, blend across them.
- [ ] Backward: gradient flows into `pos` (silhouette geometry) and `color`.

### M5 — Examples + docs
- [ ] DocC catalog
- [ ] Port the "earth" sample from `samples/torch/` as a Swift example target

## Running tests
Use xcodebuild — `swift test` from the CLI fails because mlx-swift's metallib
is only emitted by an Xcode build:
```
xcodebuild -scheme mlx-swift-diffrast -destination 'platform=macOS' test
```

## Open questions
- Should we ship a thin Python bridge so existing PyTorch users can call this from
  Python via MLX's Python bindings, or keep it Swift-only? Defer until M2 lands.
- License: nvdiffrast is NVIDIA-source-available, not OSS. This port is a
  reimplementation from API + paper, no copied code. Confirm with user before publishing.

## Lessons
(populate in `tasks/lessons.md` as we go)
