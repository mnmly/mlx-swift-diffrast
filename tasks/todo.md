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
- [x] `diff_attrs = .indices([Int])` subset selection — DA kernels now take a
      `diff_idx` int array and `K` template arg; `.all` is the case K=A with
      identity indices.
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

#### M2.4 — Precompute + AABB rejection (✅ DONE)
- [x] New `precomputeTriDataKernel`: per-(batch, triangle) thread does the
      perspective divide **once** and packs `(sxk, syk, szk for k=0..2, area)`
      as `[N, T, 10]`. Used by both `fwd` and `fwd_db` kernels — no more
      per-pixel perspective divides inside the inner loop.
- [x] Inline AABB rejection — compute `min(sx*), max(sx*), min(sy*), max(sy*)`
      from the precomputed screen verts and skip the edge tests for triangles
      whose screen AABB doesn't contain the pixel.
- [x] Backward unchanged (it doesn't iterate triangles).
- [x] All 40 tests + 3 inverse-rendering samples still pass / converge.

The most user-visible perf win without the bin-list complexity. For large
meshes the inner loop becomes dominated by 4 float comparisons (the AABB
reject) for the vast majority of triangles, since most triangles are far
from any given pixel.

#### M2.5 — Range mode (✅ DONE)
- [x] `rasterize(..., ranges: [N, 2])` with shared `pos: [V, 4]` vertex buffer
- [x] Default `ranges = [[0, T]] × N` synthesized for instanced mode — keeps
      the kernel code path uniform across both layouts
- [x] Range check folded into the precompute kernel (out-of-range triangles
      get `area = 0`, the forward already skips those)
- [x] Differentiable through MLX's broadcast-backward — gradient flows back to
      the shared `[V, 4]` buffer correctly (summed over batches)

#### M2.6 — Depth peeling (✅ DONE)
- [x] `peelLayer: MLXArray?` parameter on `rasterize`. Pass a previous-layer
      output to peel through it: triangles whose pixel z is ≤ the layer's z
      are skipped. Empty pixels in the previous layer (`tri_id+1 == 0`)
      contribute `z = -∞` so all triangles compete normally.
- [x] `PEEL: Bool` template flag on `fwd` / `fwd_db` kernels — non-peel path
      is the existing kernel with the flag false, no runtime overhead.
- [x] Backward unchanged (per-pixel winner lookup from `rast[3]` works the
      same regardless of which layer was selected).

#### Deferred
- [ ] True per-tile bin lists (one allocator pass + one rasterize pass) — the
      next step beyond AABB rejection when triangle counts go past ~10⁴ and
      the per-triangle screen reads start to dominate.

### M3 — `texture`

#### M3.1 — Bilinear (✅ DONE)
- [x] Bilinear forward with three boundary modes (wrap, clamp, zero)
- [x] VJP for `tex` (atomic scatter-add) and `uv` (per-pixel analytic)
- [x] 4 gradchecks across boundary modes + 5 forward correctness tests

#### M3.2 — Mipmap + trilinear + LOD selection (✅ DONE)
- [x] `textureConstructMip(tex, maxLevel:)` — pure MLX-ops pyramid build, so
      gradient through downsampling is auto-differentiated. Stops at first
      odd dim or `maxLevel`.
- [x] `FilterMode.linearMipmapLinear` (trilinear) — per-pixel LOD from
      `uvDA` via `ρ² = max(|ds/dx|² + |dt/dx|², |ds/dy|² + |dt/dy|²)`,
      `lod = 0.5 · log₂(ρ²)`, clamped to `[0, NUM_LEVELS - 1]`. Bilinear
      samples at floor and ceil mip levels, linearly blended by `frac(lod)`.
- [x] Pyramid packed into one flat tensor + (offsets, H, W) metadata so the
      kernel can index any level with arithmetic — `d_packed_pyramid` then
      auto-flows back through `concatenate`/`reshape`/`mean` to `d_tex`.
- [x] `d_uv` gradcheck (multi-level sum chain) and `d_tex` gradcheck pass
      against MLX `grad` + finite differences.

#### M3.3 — `d_uvDA` (LOD chain backward) (✅ DONE)
- [x] Chain rule: `∂out/∂lod = sample₁ - sample₀`, `∂lod/∂ρ² = 1/(2ρ²·ln2)`,
      `∂ρ²/∂A = [A ≥ B]` (and analog for B), `∂A/∂(du/dx) = 2·sx·W` etc.
- [x] Saturated LOD (clamped at 0 or `NUM_LEVELS - 1`) → `d_uvDA = 0`, which
      is the analytically correct derivative of the clamp.
- [x] Gradcheck against MLX `grad` + central FD passes.
- [x] Closes the texture→pos gradient loop: texture loss now flows back
      through `uvDA` → `rast_db` → `pos`, completing the full chain for
      texture-aware geometry optimization.

#### M3.4 — Small texture features (✅ DONE)
- [x] `FilterMode.nearest` — single-texel lookup; cheapest forward, no
      interpolation aliasing. `d_uv` is zero (lookup is piecewise constant);
      `d_tex` is atomic scatter into the chosen texel.
- [x] `mipLevelBias: Float` parameter for trilinear — folded into `uvDA` as
      `uvDA · 2^bias` in Swift before kernel dispatch, so the existing kernels
      and `d_uvDA` chain rule work unchanged.

#### M3.5 — Per-pixel mip bias + linearMipmapNearest (✅ DONE)
- [x] `mipLevelBiasMap: MLXArray?` parameter — per-pixel bias `[N, H, W]` or
      `[N, H, W, 1]`. Folded into `uvDA` as `uvDA · 2^map` in Swift; the
      existing kernels and `d_uvDA` chain rule work unchanged.
- [x] `FilterMode.linearMipmapNearest` — bilinear sample at the *single*
      nearest mip level (rounded LOD). Cheaper than trilinear; same kernels
      via a `MIP_LINEAR: Bool` template arg.

#### M3.6 — Cube textures (✅ DONE — bilinear, d_dir deferred)
- [x] `BoundaryMode.cube` — tex shape `[N, 6, H, W, C]`, uv interpreted as
      a 3D direction vector `[N, H_img, W_img, 3]`.
- [x] Per-pixel cube projection (OpenGL face order +X,-X,+Y,-Y,+Z,-Z) →
      face selection + (sc, tc) → (u, v) → bilinear sample on the chosen face.
- [x] `d_tex` via atomic scatter to the chosen face's four texels.
- [x] `d_dir` is currently zero — chain rule through the per-face projection
      is deferred. Cube textures are typically sampled with constant
      directions (environment lookup) so this doesn't impact common use cases.
- [ ] Cube + mipmap (only `linear` filter supported with `.cube` today).

#### Deferred
- [ ] `nearestMipmapNearest` / `nearestMipmapLinear` filter combinations
      (we ship `nearest`, `linear`, `linearMipmapNearest`, `linearMipmapLinear`
      — covers the common cases).
- [ ] Cube + mipmap support for IBL/environment-map workflows.
- [ ] `d_dir` chain rule for cube textures (needed for surface normal
      optimization with environment lighting).

Metal-kernel note: template args (e.g. `BOUNDARY`) are NOT visible inside the
`header:` block — they only resolve in the main kernel body. Helper functions
in the header must take what they need as explicit parameters.

Metal-kernel note: template args (e.g. `BOUNDARY`) are NOT visible inside the
`header:` block — they only resolve in the main kernel body. Helper functions
in the header must take what they need as explicit parameters.

### M4 — `antialias`

#### M4.1 — Topology + API stub (✅ DONE)
- [x] `antialiasConstructTopologyHash(tri:)` — full Swift implementation,
      returns `[T, 3]` int32 neighbor table. `-1` = boundary, `-2` =
      non-manifold (3+ triangles share that edge — silhouette skipped).
      O(T log T), pure CPU; cache alongside `tri`.
- [x] `antialias(color:, rast:, pos:, tri:, topologyHash:)` API surface as a
      CustomFunction. **Forward is currently identity; d_color is identity;
      d_pos is zero.** Documented as a stub so callers can wire pipelines
      now and upgrade transparently when M4.2 lands.

#### M4.2 — Silhouette blend forward + color backward (✅ DONE)
- [x] Per-pixel kernel walks all 4 neighbors; bg-side-only convention removes
      atomics from the forward
- [x] Topology-hash lookup to validate silhouettes (true silhouettes vs
      depth/overlap discontinuities)
- [x] Sub-pixel coverage `α` from edge-line intersection with the pixel's
      row/column centerline in screen-pixel space
- [x] Forward blend: `out_p = color_p + Σ_dir α_dir · (color_n - color_p)`
- [x] d_color via atomic scatter — `(1 - Σα)` to self, `α_dir` to each
      contributing neighbor
- [x] Gradcheck d_color against MLX `grad` + finite differences

Metal-source lesson worth flagging: when sharing helper logic between
multiple JIT-compiled kernels via MLXFast, **inline the code** through a
Swift string-template substitution rather than via Metal preprocessor
macros or header-defined inline functions. Macro line-continuations and
the `constant`/`device` qualifier matching on kernel-input pointers are
both fragile across the Swift→Metal source boundary. See
`silhouetteBlock` in `Antialias.swift` for the working pattern.

### Examples / inverse-rendering samples (✅ DONE)
- [x] Tier 1: `ColorFitExample` — per-vertex color recovery (loss 0.028 → 0)
- [x] Tier 2: `PoseFitExample` — vertex position recovery via silhouette
      gradient (loss 0.034 → 0.0007)
- [x] Tier 3: `MeshFitExample` — 7-vertex / 6-triangle hexagonal pie mesh fit
      with topology-aware silhouettes (loss → 0.000184)
- [x] Shared `MLXDiffRastExamples` library: animated-GIF writer (ImageIO),
      MLXArray→CGImage, simple Adam

**Antialias semantic bugs found and fixed via the samples** (each invisible
to the M4.2 / M4.3 gradchecks because forward+VJP were self-consistently
wrong):
- Pixel-projection formula had a spurious `-0.5` offset → silhouette edges
  were positioned half a pixel wrong.
- `fg_center_{x,y}` used `pw`/`ph` (this pixel) instead of `nw`/`nh` (the
  neighbor, which is the actual foreground pixel when p is bg).
- Algorithm picked the wrong pixel of the pair to blend: nvdiffrast blends
  whichever pixel has the silhouette edge cutting its *interior*, not
  necessarily the rast-empty one. Replaced with the canonical `β = 0.5 -
  |xe - (pw+0.5)|` formulation.
- Boundary-edge selection picked the first `topology[k] == target` match,
  which is wrong when a triangle has multiple boundary edges. Now picks the
  candidate whose extension geometrically separates `p_center` from
  `neighbor_center` (signed-cross-product test).

#### M4.3 — `pos` backward (✅ DONE — the killer-feature gradient)
- [x] Chain rule:
        `d_loss/d_α` → `d_α/d_(xe|ye)` → `d_(xe|ye)/d_e{0,1}` (line-intersection
        partials) → `d_e/d_pos` (perspective-divide partials)
- [x] Single new kernel `bwdPos` reusing the `silhouetteBlock` per-direction
      stash; both endpoint vertices receive atomic scatter contributions per
      silhouette direction
- [x] `α` saturation handled correctly — gradient zeroed when α clipped to
      0 or 1 (mirrors the analytic derivative of the clamp)
- [x] Gradcheck against MLX `grad` + central FD on a quad fixture
      (two triangles share an interior edge → topology lookups unambiguous;
      `rast` precomputed once outside the loss to fix coverage)
- [x] Non-triviality check confirming the gradient path actually fires
      (at least one pos element receives a non-zero gradient)

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
