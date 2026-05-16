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

#### M2.2 — Pixel derivatives `rast_db`
- [ ] Optional second output `[N,H,W,4]` with `(du/dx, du/dy, dv/dx, dv/dy)`.
- [ ] These are constants per triangle (linear barycentrics in screen space),
      so a single per-pixel write suffices once the covering triangle is known.
- [ ] Likely fold into the existing forward kernel under a template flag rather
      than a second kernel pass.

#### M2.3 — Backward
- [ ] Replace the placeholder VJP. Gradients from `d_rast[u,v,z/w]` flow into
      `pos` via:
        1. Chain through perspective divide: d(NDC)/d(clip).
        2. Chain through screen-space barycentric formula: d(u,v)/d(s0,s1,s2).
        3. Depth: d(z_pix)/d(z_k, w_k) at each vertex via the linear combo.
- [ ] Strategy: per-pixel compute kernel that emits per-vertex contributions
      with atomic scatter-add into `d_pos` (analogous to interpolate's d_attr).
- [ ] Add to gradcheck test suite once it lands.

#### Deferred
- [ ] Range mode (`pos` shape `[V, 4]` + `ranges` tensor)
- [ ] `DepthPeeler` analog for transparency
- [ ] Replace per-pixel ∀-triangle loop with a tile-based hierarchical pass
      when triangle counts grow beyond a few thousand

### M3 — `texture`
- [ ] Forward: bilinear / trilinear sampling with mipmap pyramid + boundary modes
      (wrap, clamp, zero, cube). Metal samplers cover most; cube mode needs manual indexing.
- [ ] `texture_construct_mip` — produce mip pyramid as a list of MLXArrays.
- [ ] Backward: gradient w.r.t. `tex`, `uv`, `uv_da`, `mip_level_bias`.

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
