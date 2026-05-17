import Foundation
import MLX
import MLXFast

extension DiffRast {

    /// Boundary handling for `texture`. Matches the nvdiffrast names.
    public enum BoundaryMode: Int32 {
        case wrap = 0       // tile the texture (periodic)
        case clamp = 1      // edge-extend
        case zero = 2       // sample 0 outside [0, 1] in either axis
        /// Cube texture mode. `tex` shape must be `[N, 6, H, W, C]` (six
        /// faces stacked along axis 1), and `uv` is interpreted as a 3D
        /// direction vector `[N, H_img, W_img, 3]`. Face order: +X, -X, +Y,
        /// -Y, +Z, -Z (OpenGL convention).
        case cube = 3
    }

    /// Texture filtering mode.
    ///
    /// - `nearest`: single closest texel from level 0. Cheapest and exact
    ///   (no interpolation aliasing) but discontinuous in `uv` — gradient
    ///   w.r.t. `uv` is zero (the lookup is piecewise constant). Use for
    ///   pixel-art textures or exact integer-indexed lookups.
    /// - `linear`: bilinear sampling at the source resolution. Fast and
    ///   robust when sampling at ~source scale; aliases under minification.
    /// - `linearMipmapNearest`: bilinear sample at the *single* nearest mip
    ///   level (rounded from LOD). Cheaper than trilinear and good enough
    ///   when mip-level boundaries don't show as visible banding in the
    ///   target use case.
    /// - `linearMipmapLinear`: classic trilinear filtering. Builds a 2× box-
    ///   filter mip pyramid, derives a per-pixel LOD from `uvDA`, then blends
    ///   bilinear samples from the two adjacent mip levels. Stable under wide
    ///   scale variation, costs an extra log-depth pyramid build.
    public enum FilterMode {
        case nearest
        case linear
        case linearMipmapNearest
        case linearMipmapLinear
    }

    /// Texture sampling — bilinear (`.linear`) or trilinear with mipmaps
    /// (`.linearMipmapLinear`).
    ///
    /// Shapes:
    ///   - `tex`: `[N, H_tex, W_tex, C]`. Batch dim 1 broadcasts to `uv`'s N.
    ///     For mipmap modes, `H_tex` and `W_tex` should be powers of 2 — the
    ///     pyramid stops downsampling at the first odd dimension.
    ///   - `uv`:  `[N, H_img, W_img, 2]` with channels `(u, v)` in `[0, 1]`.
    ///   - `uvDA`: required for `.linearMipmapLinear`. Shape
    ///     `[N, H_img, W_img, 4]` with channels `(du/dx, du/dy, dv/dx, dv/dy)`
    ///     in per-pixel units — the same layout `rasterize`'s `rastDB`
    ///     produces, modulo the upstream chain through `interpolate`.
    /// Returns:
    ///   - `out`: `[N, H_img, W_img, C]`.
    ///
    /// **Differentiability:** w.r.t. `tex`, `uv`, and `uvDA` in trilinear
    /// mode. The `uvDA` gradient flows through the LOD chain rule
    /// (saturating to zero when LOD is clamped at 0 or `NUM_LEVELS - 1`) —
    /// which is what connects a texture-aware pixel loss back to `pos`
    /// through `rast_db` in a full inverse-rendering chain.
    public static func texture(
        _ tex: MLXArray,
        uv: MLXArray,
        uvDA: MLXArray? = nil,
        filterMode: FilterMode = .linear,
        boundaryMode: BoundaryMode = .wrap,
        mipLevelBias: Float = 0,
        mipLevelBiasMap: MLXArray? = nil
    ) -> MLXArray {
        // Cube textures dispatch to a dedicated code path — different tex
        // shape ([N, 6, H, W, C]), uv is a 3D direction vector, and the
        // sampling math has its own per-face projection.
        if boundaryMode == .cube {
            precondition(filterMode == .linear,
                         "texture: cube textures currently support filterMode .linear only")
            precondition(tex.ndim == 5 && tex.shape[1] == 6,
                         "texture: cube tex must be [N, 6, H, W, C] (got \(tex.shape))")
            precondition(uv.ndim == 4 && uv.shape[3] == 3,
                         "texture: cube uv must be [N, H_img, W_img, 3] direction vector (got \(uv.shape))")
            return textureCube(tex, dir: uv)
        }

        precondition(tex.ndim == 4,
                     "texture: tex must be [N, H_tex, W_tex, C] (got \(tex.shape))")
        precondition(uv.ndim == 4 && uv.shape[3] == 2,
                     "texture: uv must be [N, H_img, W_img, 2] (got \(uv.shape))")
        precondition(tex.shape[0] == 1 || tex.shape[0] == uv.shape[0],
                     "texture: tex batch dim \(tex.shape[0]) incompatible with uv batch \(uv.shape[0])")

        switch filterMode {
        case .nearest:
            return textureNearest(tex, uv: uv, boundaryMode: boundaryMode)
        case .linear:
            return textureBilinear(tex, uv: uv, boundaryMode: boundaryMode)
        case .linearMipmapNearest, .linearMipmapLinear:
            precondition(uvDA != nil,
                         "texture: mipmap filterModes require uvDA")
            precondition(uvDA!.shape[0] == uv.shape[0]
                         && uvDA!.shape[1] == uv.shape[1]
                         && uvDA!.shape[2] == uv.shape[2]
                         && uvDA!.shape[3] == 4,
                         "texture: uvDA must be [N, H_img, W_img, 4] (got \(uvDA!.shape))")
            // mipLevelBias and mipLevelBiasMap are folded into uvDA by
            // multiplying by 2^bias — `lod = 0.5·log₂(ρ²·2^(2b)) = 0.5·log₂(ρ²) + b`.
            // Per-pixel: bias map is `[N, H, W]` or `[N, H, W, 1]`, broadcast
            // over the 4 uvDA channels. Gradient flows correctly through MLX
            // autograd (the multiply is a regular op).
            let effectiveUVDA: MLXArray
            if let map = mipLevelBiasMap {
                precondition(map.shape[0] == uv.shape[0]
                             && map.shape[1] == uv.shape[1]
                             && map.shape[2] == uv.shape[2],
                             "texture: mipLevelBiasMap must be [N, H_img, W_img] or [N, H_img, W_img, 1] (got \(map.shape))")
                let mapExpanded = (map.ndim == 3) ? map.expandedDimensions(axis: 3) : map
                let scale = pow(MLXArray(Float(2.0)), mapExpanded)
                // Combine scalar mipLevelBias with per-pixel map.
                let scaleWithScalar = (mipLevelBias == 0)
                    ? scale
                    : scale * MLXArray(pow(2.0, mipLevelBias))
                effectiveUVDA = uvDA! * scaleWithScalar
            } else if mipLevelBias != 0 {
                effectiveUVDA = uvDA! * MLXArray(pow(2.0, mipLevelBias))
            } else {
                effectiveUVDA = uvDA!
            }
            let mipLinear = (filterMode == .linearMipmapLinear)
            return textureMipmap(tex, uv: uv, uvDA: effectiveUVDA,
                                 boundaryMode: boundaryMode, mipLinear: mipLinear)
        }
    }

    /// Build a mip pyramid by 2× box-filter downsampling, stopping when
    /// either spatial dim becomes odd or reaches `maxLevel + 1` entries.
    /// Pure forward; differentiable through MLX's autograd (uses just
    /// `reshape` and `mean`).
    ///
    /// Returns a list of MLXArrays — `[level0 = tex, level1, level2, ...]`.
    /// Level k has shape `[N, H/2^k, W/2^k, C]`.
    ///
    /// You usually don't need to call this directly — `texture(..., filterMode:
    /// .linearMipmapLinear)` builds the pyramid internally. Exposed for users
    /// who want to cache the pyramid across many calls with the same `tex`.
    public static func textureConstructMip(
        _ tex: MLXArray, maxLevel: Int? = nil
    ) -> [MLXArray] {
        precondition(tex.ndim == 4,
                     "textureConstructMip: tex must be [N, H, W, C] (got \(tex.shape))")
        var levels: [MLXArray] = [tex]
        let N = tex.shape[0], C = tex.shape[3]
        var current = tex
        var H = tex.shape[1], W = tex.shape[2]
        while H >= 2 && W >= 2 && H.isMultiple(of: 2) && W.isMultiple(of: 2) {
            if let mx = maxLevel, levels.count > mx { break }
            let H2 = H / 2, W2 = W / 2
            current = current.reshaped([N, H2, 2, W2, 2, C]).mean(axes: [2, 4])
            levels.append(current)
            H = H2; W = W2
        }
        return levels
    }

    // MARK: - Cube texture path (.cube boundary)

    /// Sample a cube texture given a per-pixel direction vector.
    /// Cube layout: `tex` shape `[N, 6, H, W, C]` with face order
    /// `(+X, -X, +Y, -Y, +Z, -Z)` (OpenGL convention).
    ///
    /// Differentiable w.r.t. `tex` (atomic scatter into the chosen face).
    /// `d_dir` is currently zero — the chain rule through the per-face
    /// projection is deferred; cube textures are typically sampled with
    /// constant directions (environment lookup) so the missing gradient
    /// doesn't impact common use cases.
    private static func textureCube(_ tex: MLXArray, dir: MLXArray) -> MLXArray {
        let N = dir.shape[0]
        let Himg = dir.shape[1], Wimg = dir.shape[2]
        let Htex = tex.shape[2], Wtex = tex.shape[3], C = tex.shape[4]

        // Pack tex as a flat tensor [N * 6 * H * W * C] so the kernel indexes
        // freely. Broadcast N=1 if needed.
        var texN = tex
        if texN.shape[0] == 1 && N > 1 {
            texN = broadcast(texN, to: [N, 6, Htex, Wtex, C])
        }
        let texFlat = texN.reshaped([-1])

        let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
            [Self.cubeForwardKernel(
                tex: inputs[0], dir: inputs[1],
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C)]
        }
        let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
            let dTexFlat = Self.cubeBackwardKernel(
                tex: primals[0], dir: primals[1], dOut: cotangents[0],
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C)
            // d_dir is zero (deferred).
            let dDir = MLXArray.zeros(primals[1].shape, dtype: .float32)
            return [dTexFlat, dDir]
        }
        let custom = CustomFunction { Forward(fwd); VJP(vjp) }
        let outFlat = custom([texFlat, dir])[0]
        // The custom function returns a flat-shaped result if forward returned
        // one; cubeForwardKernel returns [N, Himg, Wimg, C] directly so this
        // is fine. Reshape just in case.
        return outFlat.reshaped([N, Himg, Wimg, C])
    }

    // MARK: - Nearest (.nearest) path

    private static func textureNearest(
        _ tex: MLXArray, uv: MLXArray, boundaryMode: BoundaryMode
    ) -> MLXArray {
        let N = uv.shape[0]
        let Himg = uv.shape[1], Wimg = uv.shape[2]
        let Htex = tex.shape[1], Wtex = tex.shape[2], C = tex.shape[3]

        var texN = tex
        if texN.shape[0] == 1 && N > 1 {
            texN = broadcast(texN, to: [N, Htex, Wtex, C])
        }

        let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
            [Self.textureNearestForwardKernel(
                tex: inputs[0], uv: inputs[1],
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C,
                boundary: boundaryMode)]
        }
        // uv has no gradient (nearest is piecewise constant); only d_tex flows.
        let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
            let dTex = Self.textureNearestBackwardKernel(
                tex: primals[0], uv: primals[1], dOut: cotangents[0],
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C,
                boundary: boundaryMode)
            let dUV = MLXArray.zeros(primals[1].shape, dtype: .float32)
            return [dTex, dUV]
        }
        let custom = CustomFunction { Forward(fwd); VJP(vjp) }
        return custom([texN, uv])[0]
    }

    // MARK: - Bilinear (.linear) path — original M3.1 implementation

    private static func textureBilinear(
        _ tex: MLXArray, uv: MLXArray, boundaryMode: BoundaryMode
    ) -> MLXArray {
        let N = uv.shape[0]
        let Himg = uv.shape[1], Wimg = uv.shape[2]
        let Htex = tex.shape[1], Wtex = tex.shape[2], C = tex.shape[3]

        var texN = tex
        if texN.shape[0] == 1 && N > 1 {
            texN = broadcast(texN, to: [N, Htex, Wtex, C])
        }

        let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
            [Self.textureForwardKernel(
                tex: inputs[0], uv: inputs[1],
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C,
                boundary: boundaryMode)]
        }
        let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
            let (dTex, dUV) = Self.textureBackwardKernels(
                tex: primals[0], uv: primals[1], dOut: cotangents[0],
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C,
                boundary: boundaryMode)
            return [dTex, dUV]
        }
        let custom = CustomFunction { Forward(fwd); VJP(vjp) }
        return custom([texN, uv])[0]
    }

    // MARK: - Mipmap path (.linearMipmapLinear and .linearMipmapNearest)

    private static func textureMipmap(
        _ tex: MLXArray, uv: MLXArray, uvDA: MLXArray,
        boundaryMode: BoundaryMode, mipLinear: Bool
    ) -> MLXArray {
        let N = uv.shape[0]
        let Himg = uv.shape[1], Wimg = uv.shape[2]
        let Htex = tex.shape[1], Wtex = tex.shape[2], C = tex.shape[3]

        var texN = tex
        if texN.shape[0] == 1 && N > 1 {
            texN = broadcast(texN, to: [N, Htex, Wtex, C])
        }

        // Build the mip pyramid via MLX ops (auto-differentiable). Then pack
        // into a single flat tensor so the kernel can read all levels with
        // index arithmetic. Per-level dimensions and starting offsets are
        // metadata int32 buffers — non-differentiable.
        let pyramid = textureConstructMip(texN)
        let numLevels = pyramid.count
        var offsets: [Int32] = []
        var levelH: [Int32] = []
        var levelW: [Int32] = []
        var cumulative: Int32 = 0
        for level in pyramid {
            offsets.append(cumulative)
            levelH.append(Int32(level.shape[1]))
            levelW.append(Int32(level.shape[2]))
            cumulative += Int32(level.shape[0] * level.shape[1] * level.shape[2] * level.shape[3])
        }
        // Pack levels along the flat axis. Use stack of `flattened`+`concat`.
        let packed = MLX.concatenated(pyramid.map { $0.reshaped([-1]) }, axis: 0)
        let offsetsArr = MLXArray(offsets, [numLevels])
        let levelHArr = MLXArray(levelH, [numLevels])
        let levelWArr = MLXArray(levelW, [numLevels])

        let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
            [Self.textureTrilinearForwardKernel(
                packed: inputs[0], uv: inputs[1], uvDA: inputs[2],
                offsets: offsetsArr, levelH: levelHArr, levelW: levelWArr,
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C,
                numLevels: numLevels, boundary: boundaryMode, mipLinear: mipLinear)]
        }
        let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
            let (dPacked, dUV, dUVDA) = Self.textureTrilinearBackwardKernels(
                packed: primals[0], uv: primals[1], uvDA: primals[2], dOut: cotangents[0],
                offsets: offsetsArr, levelH: levelHArr, levelW: levelWArr,
                packedSize: Int(cumulative),
                N: N, Himg: Himg, Wimg: Wimg, Htex: Htex, Wtex: Wtex, C: C,
                numLevels: numLevels, boundary: boundaryMode, mipLinear: mipLinear)
            return [dPacked, dUV, dUVDA]
        }
        let custom = CustomFunction { Forward(fwd); VJP(vjp) }
        return custom([packed, uv, uvDA])[0]
    }

    // MARK: - Kernel sources

    /// Shared boundary-mode helper as a Metal header. Returns the integer texel
    /// coord and an out-of-bounds mask (used only by `zero` mode).
    ///
    /// Convention: `t` is the floored real-valued texel coord. Returns coord in
    /// `[0, size)` (after wrap/clamp) and an OOB flag (1.0 = out of bounds).
    private static let boundaryHeader = """
        // mode: 0=wrap, 1=clamp, 2=zero
        inline int apply_boundary(int t, int size, int mode, thread float& oob) {
            if (mode == 2) {
                if (t < 0 || t >= size) { oob = 1.0f; return 0; }
                oob = 0.0f;
                return t;
            }
            if (mode == 1) {
                oob = 0.0f;
                return t < 0 ? 0 : (t >= size ? size - 1 : t);
            }
            // wrap (mode == 0)
            oob = 0.0f;
            int r = t % size;
            if (r < 0) r += size;
            return r;
        }
    """

    /// Forward: bilinear lookup at four neighboring texels.
    /// Texel-center convention: texel (i, j) occupies real coord (j+0.5)/W, (i+0.5)/H.
    /// So `tx = u*W - 0.5`, `ty = v*H - 0.5`.
    private static let fwdSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        float tx = u * (float)Wtex - 0.5f;
        float ty = v * (float)Htex - 0.5f;

        int tx0 = (int)floor(tx);
        int ty0 = (int)floor(ty);
        int tx1 = tx0 + 1;
        int ty1 = ty0 + 1;
        float fx = tx - (float)tx0;
        float fy = ty - (float)ty0;

        float oob_x0, oob_x1, oob_y0, oob_y1;
        int ix0 = apply_boundary(tx0, Wtex, BOUNDARY, oob_x0);
        int ix1 = apply_boundary(tx1, Wtex, BOUNDARY, oob_x1);
        int iy0 = apply_boundary(ty0, Htex, BOUNDARY, oob_y0);
        int iy1 = apply_boundary(ty1, Htex, BOUNDARY, oob_y1);

        float w00 = (1.0f - fx) * (1.0f - fy) * (1.0f - oob_x0) * (1.0f - oob_y0);
        float w10 = (       fx) * (1.0f - fy) * (1.0f - oob_x1) * (1.0f - oob_y0);
        float w01 = (1.0f - fx) * (       fy) * (1.0f - oob_x0) * (1.0f - oob_y1);
        float w11 = (       fx) * (       fy) * (1.0f - oob_x1) * (1.0f - oob_y1);

        uint tex_n_base = pn * (uint)Htex * (uint)Wtex * (uint)C;
        uint out_base   = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;

        for (int c = 0; c < C; ++c) {
            float t00 = tex[tex_n_base + ((uint)iy0 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c];
            float t10 = tex[tex_n_base + ((uint)iy0 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c];
            float t01 = tex[tex_n_base + ((uint)iy1 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c];
            float t11 = tex[tex_n_base + ((uint)iy1 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c];
            out[out_base + (uint)c] = w00 * t00 + w10 * t10 + w01 * t01 + w11 * t11;
        }
    """

    /// d_tex: atomic scatter-add into the four neighboring texels with the same
    /// bilinear weights as the forward.
    private static let gradTexSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        float tx = u * (float)Wtex - 0.5f;
        float ty = v * (float)Htex - 0.5f;

        int tx0 = (int)floor(tx);
        int ty0 = (int)floor(ty);
        int tx1 = tx0 + 1;
        int ty1 = ty0 + 1;
        float fx = tx - (float)tx0;
        float fy = ty - (float)ty0;

        float oob_x0, oob_x1, oob_y0, oob_y1;
        int ix0 = apply_boundary(tx0, Wtex, BOUNDARY, oob_x0);
        int ix1 = apply_boundary(tx1, Wtex, BOUNDARY, oob_x1);
        int iy0 = apply_boundary(ty0, Htex, BOUNDARY, oob_y0);
        int iy1 = apply_boundary(ty1, Htex, BOUNDARY, oob_y1);

        float w00 = (1.0f - fx) * (1.0f - fy) * (1.0f - oob_x0) * (1.0f - oob_y0);
        float w10 = (       fx) * (1.0f - fy) * (1.0f - oob_x1) * (1.0f - oob_y0);
        float w01 = (1.0f - fx) * (       fy) * (1.0f - oob_x0) * (1.0f - oob_y1);
        float w11 = (       fx) * (       fy) * (1.0f - oob_x1) * (1.0f - oob_y1);

        uint dout_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;
        uint dtex_n_base = pn * (uint)Htex * (uint)Wtex * (uint)C;

        for (int c = 0; c < C; ++c) {
            float g = d_out[dout_base + (uint)c];
            atomic_fetch_add_explicit(
                &d_tex[dtex_n_base + ((uint)iy0 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c],
                w00 * g, memory_order_relaxed);
            atomic_fetch_add_explicit(
                &d_tex[dtex_n_base + ((uint)iy0 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c],
                w10 * g, memory_order_relaxed);
            atomic_fetch_add_explicit(
                &d_tex[dtex_n_base + ((uint)iy1 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c],
                w01 * g, memory_order_relaxed);
            atomic_fetch_add_explicit(
                &d_tex[dtex_n_base + ((uint)iy1 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c],
                w11 * g, memory_order_relaxed);
        }
    """

    /// d_uv: per-pixel non-atomic. The bilinear sample is a smooth function of
    /// `(tx, ty)` *between* integer breakpoints; at exact integer values the
    /// gradient is discontinuous but we pick one branch (the one consistent with
    /// `floor`). For `zero` boundary the contribution from an OOB texel is 0
    /// and so is its derivative wrt fx/fy.
    ///
    ///   out_c = (1-fx)(1-fy) t00 + fx(1-fy) t10 + (1-fx) fy t01 + fx fy t11
    ///   d_out_c/d_fx = (1-fy)*(t10 - t00) + fy*(t11 - t01)
    ///   d_out_c/d_fy = (1-fx)*(t01 - t00) + fx*(t11 - t10)
    ///
    /// Chain rule: d_fx/d_u = Wtex, d_fy/d_v = Htex.
    private static let gradUVSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        float tx = u * (float)Wtex - 0.5f;
        float ty = v * (float)Htex - 0.5f;
        int tx0 = (int)floor(tx);
        int ty0 = (int)floor(ty);
        int tx1 = tx0 + 1;
        int ty1 = ty0 + 1;
        float fx = tx - (float)tx0;
        float fy = ty - (float)ty0;

        float oob_x0, oob_x1, oob_y0, oob_y1;
        int ix0 = apply_boundary(tx0, Wtex, BOUNDARY, oob_x0);
        int ix1 = apply_boundary(tx1, Wtex, BOUNDARY, oob_x1);
        int iy0 = apply_boundary(ty0, Htex, BOUNDARY, oob_y0);
        int iy1 = apply_boundary(ty1, Htex, BOUNDARY, oob_y1);

        // Masked texel values: OOB samples count as zero in `zero` mode.
        uint tex_n_base = pn * (uint)Htex * (uint)Wtex * (uint)C;
        uint dout_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;
        float m00 = (1.0f - oob_x0) * (1.0f - oob_y0);
        float m10 = (1.0f - oob_x1) * (1.0f - oob_y0);
        float m01 = (1.0f - oob_x0) * (1.0f - oob_y1);
        float m11 = (1.0f - oob_x1) * (1.0f - oob_y1);

        float dfx_sum = 0.0f;
        float dfy_sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float t00 = tex[tex_n_base + ((uint)iy0 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c] * m00;
            float t10 = tex[tex_n_base + ((uint)iy0 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c] * m10;
            float t01 = tex[tex_n_base + ((uint)iy1 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c] * m01;
            float t11 = tex[tex_n_base + ((uint)iy1 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c] * m11;
            float g  = d_out[dout_base + (uint)c];
            dfx_sum += g * ((1.0f - fy) * (t10 - t00) + fy * (t11 - t01));
            dfy_sum += g * ((1.0f - fx) * (t01 - t00) + fx * (t11 - t10));
        }
        d_uv[uv_base + 0] = dfx_sum * (float)Wtex;
        d_uv[uv_base + 1] = dfy_sum * (float)Htex;
    """

    // MARK: - Kernel instances

    private static let fwdKernel = MLXFast.metalKernel(
        name: "diffrast_texture_fwd",
        inputNames: ["tex", "uv"], outputNames: ["out"],
        source: fwdSource, header: boundaryHeader)

    private static let gradTexKernel = MLXFast.metalKernel(
        name: "diffrast_texture_grad_tex",
        inputNames: ["tex", "uv", "d_out"], outputNames: ["d_tex"],
        source: gradTexSource, header: boundaryHeader,
        atomicOutputs: true)

    private static let gradUVKernel = MLXFast.metalKernel(
        name: "diffrast_texture_grad_uv",
        inputNames: ["tex", "uv", "d_out"], outputNames: ["d_uv"],
        source: gradUVSource, header: boundaryHeader)

    // MARK: - Nearest filter kernels

    /// Single-texel lookup. Texel index = `floor(uv * tex_size)`. Out-of-bounds
    /// behavior follows `BoundaryMode` (with `.zero` → output is zero on OOB).
    private static let nearestFwdSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        int tx = (int)floor(u * (float)Wtex);
        int ty = (int)floor(v * (float)Htex);

        float oob_x, oob_y;
        int ix = apply_boundary(tx, Wtex, BOUNDARY, oob_x);
        int iy = apply_boundary(ty, Htex, BOUNDARY, oob_y);
        float mask = (1.0f - oob_x) * (1.0f - oob_y);

        uint tex_n_base = pn * (uint)Htex * (uint)Wtex * (uint)C;
        uint out_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;
        for (int c = 0; c < C; ++c) {
            float t = tex[tex_n_base + ((uint)iy * (uint)Wtex + (uint)ix) * (uint)C + (uint)c];
            out[out_base + (uint)c] = t * mask;
        }
    """

    /// d_tex for nearest: atomic-add `d_out * mask` into the single chosen texel.
    /// d_uv has no contribution (nearest is piecewise constant in uv).
    private static let nearestGradTexSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        int tx = (int)floor(u * (float)Wtex);
        int ty = (int)floor(v * (float)Htex);

        float oob_x, oob_y;
        int ix = apply_boundary(tx, Wtex, BOUNDARY, oob_x);
        int iy = apply_boundary(ty, Htex, BOUNDARY, oob_y);
        float mask = (1.0f - oob_x) * (1.0f - oob_y);

        uint dout_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;
        uint dtex_n_base = pn * (uint)Htex * (uint)Wtex * (uint)C;
        for (int c = 0; c < C; ++c) {
            float g = d_out[dout_base + (uint)c] * mask;
            atomic_fetch_add_explicit(
                &d_tex[dtex_n_base + ((uint)iy * (uint)Wtex + (uint)ix) * (uint)C + (uint)c],
                g, memory_order_relaxed);
        }
    """

    private static let nearestFwdKernel = MLXFast.metalKernel(
        name: "diffrast_texture_nearest_fwd",
        inputNames: ["tex", "uv"], outputNames: ["out"],
        source: nearestFwdSource, header: boundaryHeader)

    // MARK: - Cube texture kernels

    /// Cube projection from direction (x, y, z) to (face, sc, tc, ma).
    /// Inlined into both fwd and bwd kernels.
    private static let cubeProjectionBlock = """
        float ax = fabs(x), ay = fabs(y), az = fabs(z);
        int face;
        float sc, tc, ma;
        if (ax >= ay && ax >= az) {
            ma = ax;
            if (x > 0.0f) { face = 0; sc = -z; tc = -y; }
            else          { face = 1; sc =  z; tc = -y; }
        } else if (ay >= az) {
            ma = ay;
            if (y > 0.0f) { face = 2; sc =  x; tc =  z; }
            else          { face = 3; sc =  x; tc = -z; }
        } else {
            ma = az;
            if (z > 0.0f) { face = 4; sc =  x; tc = -y; }
            else          { face = 5; sc = -x; tc = -y; }
        }
        float inv_ma = 1.0f / ma;
        float u = (sc * inv_ma + 1.0f) * 0.5f;
        float v = (tc * inv_ma + 1.0f) * 0.5f;
    """

    private static var cubeFwdSource: String { """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint dir_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 3u;
        float x = dir[dir_base + 0];
        float y = dir[dir_base + 1];
        float z = dir[dir_base + 2];

        \(cubeProjectionBlock)

        // Bilinear sample at chosen face. Clamp boundary handling for the
        // face edges (cube seams need special handling we skip for v1).
        float tx_real = u * (float)Wtex - 0.5f;
        float ty_real = v * (float)Htex - 0.5f;
        int tx0 = (int)floor(tx_real);
        int ty0 = (int)floor(ty_real);
        int tx1 = tx0 + 1;
        int ty1 = ty0 + 1;
        float fx = tx_real - (float)tx0;
        float fy = ty_real - (float)ty0;
        int ix0 = max(0, min(Wtex - 1, tx0));
        int ix1 = max(0, min(Wtex - 1, tx1));
        int iy0 = max(0, min(Htex - 1, ty0));
        int iy1 = max(0, min(Htex - 1, ty1));

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 =        fx  * (1.0f - fy);
        float w01 = (1.0f - fx) *        fy;
        float w11 =        fx  *        fy;

        uint face_base = (pn * 6u + (uint)face) * (uint)Htex * (uint)Wtex * (uint)C;
        uint out_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;

        for (int c = 0; c < C; ++c) {
            float t00 = tex[face_base + ((uint)iy0 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c];
            float t10 = tex[face_base + ((uint)iy0 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c];
            float t01 = tex[face_base + ((uint)iy1 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c];
            float t11 = tex[face_base + ((uint)iy1 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c];
            out[out_base + (uint)c] = w00 * t00 + w10 * t10 + w01 * t01 + w11 * t11;
        }
    """ }

    private static var cubeGradTexSource: String { """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint dir_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 3u;
        float x = dir[dir_base + 0];
        float y = dir[dir_base + 1];
        float z = dir[dir_base + 2];

        \(cubeProjectionBlock)

        float tx_real = u * (float)Wtex - 0.5f;
        float ty_real = v * (float)Htex - 0.5f;
        int tx0 = (int)floor(tx_real);
        int ty0 = (int)floor(ty_real);
        int tx1 = tx0 + 1;
        int ty1 = ty0 + 1;
        float fx = tx_real - (float)tx0;
        float fy = ty_real - (float)ty0;
        int ix0 = max(0, min(Wtex - 1, tx0));
        int ix1 = max(0, min(Wtex - 1, tx1));
        int iy0 = max(0, min(Htex - 1, ty0));
        int iy1 = max(0, min(Htex - 1, ty1));

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 =        fx  * (1.0f - fy);
        float w01 = (1.0f - fx) *        fy;
        float w11 =        fx  *        fy;

        uint face_base = (pn * 6u + (uint)face) * (uint)Htex * (uint)Wtex * (uint)C;
        uint dout_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;

        for (int c = 0; c < C; ++c) {
            float g = d_out[dout_base + (uint)c];
            atomic_fetch_add_explicit(
                &d_tex[face_base + ((uint)iy0 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c],
                w00 * g, memory_order_relaxed);
            atomic_fetch_add_explicit(
                &d_tex[face_base + ((uint)iy0 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c],
                w10 * g, memory_order_relaxed);
            atomic_fetch_add_explicit(
                &d_tex[face_base + ((uint)iy1 * (uint)Wtex + (uint)ix0) * (uint)C + (uint)c],
                w01 * g, memory_order_relaxed);
            atomic_fetch_add_explicit(
                &d_tex[face_base + ((uint)iy1 * (uint)Wtex + (uint)ix1) * (uint)C + (uint)c],
                w11 * g, memory_order_relaxed);
        }
    """ }

    private static let cubeFwdKernel = MLXFast.metalKernel(
        name: "diffrast_texture_cube_fwd",
        inputNames: ["tex", "dir"], outputNames: ["out"],
        source: cubeFwdSource)

    private static let cubeGradTexKernel = MLXFast.metalKernel(
        name: "diffrast_texture_cube_grad_tex",
        inputNames: ["tex", "dir", "d_out"], outputNames: ["d_tex"],
        source: cubeGradTexSource, atomicOutputs: true)

    private static let nearestGradTexKernel = MLXFast.metalKernel(
        name: "diffrast_texture_nearest_grad_tex",
        inputNames: ["tex", "uv", "d_out"], outputNames: ["d_tex"],
        source: nearestGradTexSource, header: boundaryHeader,
        atomicOutputs: true)

    // MARK: - Trilinear kernels (mipmap + LOD)

    /// Per-pixel LOD + bilinear sample at level k, packed-pyramid form.
    /// Shared between forward and backward bodies via Swift string interpolation
    /// (see [[feedback-mlxfast-kernel-sharing]] for why not header functions).
    ///
    /// Honors `MIP_LINEAR` template arg: `1` = trilinear (floor + frac blend
    /// of two adjacent levels), `0` = mipmap-nearest (round, single level).
    private static let trilinearBilinearBlock = """
        // Compute LOD from uvDA. Reference scale is the level-0 dimensions.
        float du_dx = uvDA[uvda_base + 0];
        float du_dy = uvDA[uvda_base + 1];
        float dv_dx = uvDA[uvda_base + 2];
        float dv_dy = uvDA[uvda_base + 3];
        float sx = du_dx * (float)Wtex;
        float sy = du_dy * (float)Wtex;
        float tx = dv_dx * (float)Htex;
        float ty = dv_dy * (float)Htex;
        float rho_sq = fmax(sx * sx + tx * tx, sy * sy + ty * ty);
        float lod = 0.5f * log2(fmax(rho_sq, 1e-20f));
        lod = fmax(0.0f, fmin((float)(NUM_LEVELS - 1), lod));
        int k0, k1;
        float frac;
        if (MIP_LINEAR) {
            k0 = (int)floor(lod);
            k1 = k0 + 1; if (k1 >= NUM_LEVELS) k1 = NUM_LEVELS - 1;
            frac = lod - (float)k0;
        } else {
            // Nearest mip level: round to nearest integer, single sample.
            k0 = (int)floor(lod + 0.5f);
            if (k0 >= NUM_LEVELS) k0 = NUM_LEVELS - 1;
            k1 = k0;
            frac = 0.0f;
        }
    """

    /// `BILINEAR_AT_LEVEL_BLOCK(k_var, weights_out, indices_out)` is too messy
    /// as Swift-interpolation; instead we open-code it twice in each kernel.
    /// Inputs available at the call site: u, v, oob masks, tex pointer.
    private static let trilinearFwdSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base   = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        uint uvda_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 4u;
        uint out_base  = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;

        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        \(trilinearBilinearBlock)

        // Bilinear sample helper, inlined for each of the two levels.
        //   tex_pyramid contains all levels concatenated; level k starts at
        //   offsets[k] and is shape (levelH[k], levelW[k], C) for batch n.
        //   The per-batch offset within a level is n * H_k * W_k * C.
        float c0 = 0.0f, c1 = 0.0f;
        uint num_per_batch_at_level0 = 0;  // unused but keep symmetry

        for (int level_idx = 0; level_idx < 2; ++level_idx) {
            int k = (level_idx == 0) ? k0 : k1;
            int Hk = levelH[k];
            int Wk = levelW[k];
            uint level_start = (uint)offsets[k];
            uint level_n_base = level_start + pn * (uint)Hk * (uint)Wk * (uint)C;

            float tx_real = u * (float)Wk - 0.5f;
            float ty_real = v * (float)Hk - 0.5f;
            int tx0 = (int)floor(tx_real);
            int ty0 = (int)floor(ty_real);
            int tx1 = tx0 + 1;
            int ty1 = ty0 + 1;
            float fx = tx_real - (float)tx0;
            float fy = ty_real - (float)ty0;

            float oob_x0, oob_x1, oob_y0, oob_y1;
            int ix0 = apply_boundary(tx0, Wk, BOUNDARY, oob_x0);
            int ix1 = apply_boundary(tx1, Wk, BOUNDARY, oob_x1);
            int iy0 = apply_boundary(ty0, Hk, BOUNDARY, oob_y0);
            int iy1 = apply_boundary(ty1, Hk, BOUNDARY, oob_y1);

            float w00 = (1.0f - fx) * (1.0f - fy) * (1.0f - oob_x0) * (1.0f - oob_y0);
            float w10 = (       fx) * (1.0f - fy) * (1.0f - oob_x1) * (1.0f - oob_y0);
            float w01 = (1.0f - fx) * (       fy) * (1.0f - oob_x0) * (1.0f - oob_y1);
            float w11 = (       fx) * (       fy) * (1.0f - oob_x1) * (1.0f - oob_y1);

            for (int c = 0; c < C; ++c) {
                float t00 = tex_pyramid[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c];
                float t10 = tex_pyramid[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c];
                float t01 = tex_pyramid[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c];
                float t11 = tex_pyramid[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c];
                float sample = w00 * t00 + w10 * t10 + w01 * t01 + w11 * t11;
                if (level_idx == 0) {
                    // Accumulate into output, weighted by (1 - frac).
                    if (c == 0) out[out_base + 0] = (1.0f - frac) * sample;
                    else out[out_base + (uint)c] = (1.0f - frac) * sample;
                } else {
                    out[out_base + (uint)c] += frac * sample;
                }
            }
        }
        (void)c0; (void)c1; (void)num_per_batch_at_level0;
    """

    /// d_tex_pyramid (atomic scatter) + d_uv (per-pixel non-atomic).
    /// Both follow directly from the linearity of bilinear sampling: for each
    /// level k_i (i ∈ {0, 1}), the four texel weights `w**` come straight from
    /// the forward, and the output gradient `d_out` is scaled by `(1 - frac)`
    /// or `frac` for level 0 / level 1 respectively.
    private static let trilinearGradPyramidSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base   = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        uint uvda_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 4u;
        uint dout_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;

        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        \(trilinearBilinearBlock)

        for (int level_idx = 0; level_idx < 2; ++level_idx) {
            int k = (level_idx == 0) ? k0 : k1;
            float scale = (level_idx == 0) ? (1.0f - frac) : frac;
            int Hk = levelH[k];
            int Wk = levelW[k];
            uint level_start = (uint)offsets[k];
            uint level_n_base = level_start + pn * (uint)Hk * (uint)Wk * (uint)C;

            float tx_real = u * (float)Wk - 0.5f;
            float ty_real = v * (float)Hk - 0.5f;
            int tx0 = (int)floor(tx_real);
            int ty0 = (int)floor(ty_real);
            int tx1 = tx0 + 1;
            int ty1 = ty0 + 1;
            float fx = tx_real - (float)tx0;
            float fy = ty_real - (float)ty0;

            float oob_x0, oob_x1, oob_y0, oob_y1;
            int ix0 = apply_boundary(tx0, Wk, BOUNDARY, oob_x0);
            int ix1 = apply_boundary(tx1, Wk, BOUNDARY, oob_x1);
            int iy0 = apply_boundary(ty0, Hk, BOUNDARY, oob_y0);
            int iy1 = apply_boundary(ty1, Hk, BOUNDARY, oob_y1);

            float w00 = (1.0f - fx) * (1.0f - fy) * (1.0f - oob_x0) * (1.0f - oob_y0);
            float w10 = (       fx) * (1.0f - fy) * (1.0f - oob_x1) * (1.0f - oob_y0);
            float w01 = (1.0f - fx) * (       fy) * (1.0f - oob_x0) * (1.0f - oob_y1);
            float w11 = (       fx) * (       fy) * (1.0f - oob_x1) * (1.0f - oob_y1);

            for (int c = 0; c < C; ++c) {
                float g = d_out[dout_base + (uint)c] * scale;
                atomic_fetch_add_explicit(
                    &d_packed[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c],
                    w00 * g, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &d_packed[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c],
                    w10 * g, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &d_packed[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c],
                    w01 * g, memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &d_packed[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c],
                    w11 * g, memory_order_relaxed);
            }
        }
    """

    private static let trilinearGradUVSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base   = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        uint uvda_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 4u;
        uint dout_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;

        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        \(trilinearBilinearBlock)

        float dfx_sum_total = 0.0f, dfy_sum_total = 0.0f;
        // dfx/du and dfy/dv depend on the level's W, H; accumulate per-level
        // and apply the scale factor (W_k for u, H_k for v) inside the loop.

        for (int level_idx = 0; level_idx < 2; ++level_idx) {
            int k = (level_idx == 0) ? k0 : k1;
            float scale = (level_idx == 0) ? (1.0f - frac) : frac;
            int Hk = levelH[k];
            int Wk = levelW[k];
            uint level_start = (uint)offsets[k];
            uint level_n_base = level_start + pn * (uint)Hk * (uint)Wk * (uint)C;

            float tx_real = u * (float)Wk - 0.5f;
            float ty_real = v * (float)Hk - 0.5f;
            int tx0 = (int)floor(tx_real);
            int ty0 = (int)floor(ty_real);
            int tx1 = tx0 + 1;
            int ty1 = ty0 + 1;
            float fx = tx_real - (float)tx0;
            float fy = ty_real - (float)ty0;

            float oob_x0, oob_x1, oob_y0, oob_y1;
            int ix0 = apply_boundary(tx0, Wk, BOUNDARY, oob_x0);
            int ix1 = apply_boundary(tx1, Wk, BOUNDARY, oob_x1);
            int iy0 = apply_boundary(ty0, Hk, BOUNDARY, oob_y0);
            int iy1 = apply_boundary(ty1, Hk, BOUNDARY, oob_y1);

            float m00 = (1.0f - oob_x0) * (1.0f - oob_y0);
            float m10 = (1.0f - oob_x1) * (1.0f - oob_y0);
            float m01 = (1.0f - oob_x0) * (1.0f - oob_y1);
            float m11 = (1.0f - oob_x1) * (1.0f - oob_y1);

            float dfx_level = 0.0f, dfy_level = 0.0f;
            for (int c = 0; c < C; ++c) {
                float t00 = tex_pyramid[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c] * m00;
                float t10 = tex_pyramid[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c] * m10;
                float t01 = tex_pyramid[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c] * m01;
                float t11 = tex_pyramid[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c] * m11;
                float g  = d_out[dout_base + (uint)c] * scale;
                dfx_level += g * ((1.0f - fy) * (t10 - t00) + fy * (t11 - t01));
                dfy_level += g * ((1.0f - fx) * (t01 - t00) + fx * (t11 - t10));
            }
            dfx_sum_total += dfx_level * (float)Wk;
            dfy_sum_total += dfy_level * (float)Hk;
        }
        d_uv[uv_base + 0] = dfx_sum_total;
        d_uv[uv_base + 1] = dfy_sum_total;
    """

    /// d_uvDA: chain rule from `d_out` back through LOD to `(du/dx, du/dy,
    /// dv/dx, dv/dy)`. Zero when LOD is clamped (saturated at 0 or
    /// NUM_LEVELS-1), which is the correct derivative of the clamp.
    ///
    /// Chain:
    ///   ∂(out)/∂lod   = sample₁ - sample₀
    ///   ∂(lod)/∂ρ²    = 1 / (2·ρ²·ln2)
    ///   ∂(ρ²)/∂A      = [A ≥ B] ;  ∂(ρ²)/∂B = [B > A]
    ///   ∂(A)/∂(du/dx) = 2·sx·W ;    ∂(A)/∂(dv/dx) = 2·tx·H
    ///   ∂(B)/∂(du/dy) = 2·sy·W ;    ∂(B)/∂(dv/dy) = 2·ty·H
    private static let trilinearGradUVDASource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)Himg * (uint)Wimg;
        if (pix >= total) return;

        uint pw = pix % (uint)Wimg;
        uint ph = (pix / (uint)Wimg) % (uint)Himg;
        uint pn = pix / ((uint)Himg * (uint)Wimg);

        uint uv_base   = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 2u;
        uint uvda_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * 4u;
        uint dout_base = ((pn * (uint)Himg + ph) * (uint)Wimg + pw) * (uint)C;

        float u = uv[uv_base + 0];
        float v = uv[uv_base + 1];

        float du_dx = uvDA[uvda_base + 0];
        float du_dy = uvDA[uvda_base + 1];
        float dv_dx = uvDA[uvda_base + 2];
        float dv_dy = uvDA[uvda_base + 3];
        float sx = du_dx * (float)Wtex;
        float sy = du_dy * (float)Wtex;
        float tx_ = dv_dx * (float)Htex;
        float ty_ = dv_dy * (float)Htex;
        float Asq = sx * sx + tx_ * tx_;
        float Bsq = sy * sy + ty_ * ty_;
        float rho_sq = fmax(Asq, Bsq);
        float lod_raw = 0.5f * log2(fmax(rho_sq, 1e-20f));

        // Saturated → d_uvDA = 0 (correct derivative of clamp at the boundary).
        if (lod_raw <= 0.0f || lod_raw >= (float)(NUM_LEVELS - 1)) {
            d_uvDA[uvda_base + 0] = 0.0f;
            d_uvDA[uvda_base + 1] = 0.0f;
            d_uvDA[uvda_base + 2] = 0.0f;
            d_uvDA[uvda_base + 3] = 0.0f;
            return;
        }

        int k0 = (int)floor(lod_raw);
        int k1 = k0 + 1; if (k1 >= NUM_LEVELS) k1 = NUM_LEVELS - 1;

        // d_loss/d_lod = Σ_c d_out[c] · (sample₁[c] - sample₀[c]).
        float d_lod = 0.0f;
        for (int level_idx = 0; level_idx < 2; ++level_idx) {
            int k = (level_idx == 0) ? k0 : k1;
            float sign = (level_idx == 0) ? -1.0f : 1.0f;
            int Hk = levelH[k];
            int Wk = levelW[k];
            uint level_start = (uint)offsets[k];
            uint level_n_base = level_start + pn * (uint)Hk * (uint)Wk * (uint)C;

            float tx_real = u * (float)Wk - 0.5f;
            float ty_real = v * (float)Hk - 0.5f;
            int tx0i = (int)floor(tx_real);
            int ty0i = (int)floor(ty_real);
            int tx1i = tx0i + 1;
            int ty1i = ty0i + 1;
            float fx = tx_real - (float)tx0i;
            float fy = ty_real - (float)ty0i;

            float oob_x0, oob_x1, oob_y0, oob_y1;
            int ix0 = apply_boundary(tx0i, Wk, BOUNDARY, oob_x0);
            int ix1 = apply_boundary(tx1i, Wk, BOUNDARY, oob_x1);
            int iy0 = apply_boundary(ty0i, Hk, BOUNDARY, oob_y0);
            int iy1 = apply_boundary(ty1i, Hk, BOUNDARY, oob_y1);

            float w00 = (1.0f - fx) * (1.0f - fy) * (1.0f - oob_x0) * (1.0f - oob_y0);
            float w10 = (       fx) * (1.0f - fy) * (1.0f - oob_x1) * (1.0f - oob_y0);
            float w01 = (1.0f - fx) * (       fy) * (1.0f - oob_x0) * (1.0f - oob_y1);
            float w11 = (       fx) * (       fy) * (1.0f - oob_x1) * (1.0f - oob_y1);

            for (int c = 0; c < C; ++c) {
                float t00 = tex_pyramid[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c];
                float t10 = tex_pyramid[level_n_base + ((uint)iy0 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c];
                float t01 = tex_pyramid[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix0) * (uint)C + (uint)c];
                float t11 = tex_pyramid[level_n_base + ((uint)iy1 * (uint)Wk + (uint)ix1) * (uint)C + (uint)c];
                float sample = w00 * t00 + w10 * t10 + w01 * t01 + w11 * t11;
                d_lod += sign * sample * d_out[dout_base + (uint)c];
            }
        }

        // Chain d_lod into uvDA.
        float ln2 = 0.6931471805599453f;
        float dlod_drho_sq = 1.0f / (2.0f * rho_sq * ln2);
        float A_is_max = (Asq >= Bsq) ? 1.0f : 0.0f;
        float B_is_max = 1.0f - A_is_max;

        d_uvDA[uvda_base + 0] = d_lod * dlod_drho_sq * A_is_max * 2.0f * sx  * (float)Wtex;
        d_uvDA[uvda_base + 1] = d_lod * dlod_drho_sq * B_is_max * 2.0f * sy  * (float)Wtex;
        d_uvDA[uvda_base + 2] = d_lod * dlod_drho_sq * A_is_max * 2.0f * tx_ * (float)Htex;
        d_uvDA[uvda_base + 3] = d_lod * dlod_drho_sq * B_is_max * 2.0f * ty_ * (float)Htex;
    """

    private static let trilinearFwdKernel = MLXFast.metalKernel(
        name: "diffrast_texture_trilinear_fwd",
        inputNames: ["tex_pyramid", "uv", "uvDA", "offsets", "levelH", "levelW"],
        outputNames: ["out"],
        source: trilinearFwdSource, header: boundaryHeader)

    private static let trilinearGradUVDAKernel = MLXFast.metalKernel(
        name: "diffrast_texture_trilinear_grad_uvda",
        inputNames: ["tex_pyramid", "uv", "uvDA", "offsets", "levelH", "levelW", "d_out"],
        outputNames: ["d_uvDA"],
        source: trilinearGradUVDASource, header: boundaryHeader)

    private static let trilinearGradPyramidKernel = MLXFast.metalKernel(
        name: "diffrast_texture_trilinear_grad_pyramid",
        inputNames: ["tex_pyramid", "uv", "uvDA", "offsets", "levelH", "levelW", "d_out"],
        outputNames: ["d_packed"],
        source: trilinearGradPyramidSource, header: boundaryHeader,
        atomicOutputs: true)

    private static let trilinearGradUVKernel = MLXFast.metalKernel(
        name: "diffrast_texture_trilinear_grad_uv",
        inputNames: ["tex_pyramid", "uv", "uvDA", "offsets", "levelH", "levelW", "d_out"],
        outputNames: ["d_uv"],
        source: trilinearGradUVSource, header: boundaryHeader)

    private static let textureTG = 256

    private static func texTmpl(N: Int, Himg: Int, Wimg: Int,
                                Htex: Int, Wtex: Int, C: Int,
                                boundary: BoundaryMode)
        -> [(String, any KernelTemplateArg)]
    {
        [("N", N), ("Himg", Himg), ("Wimg", Wimg),
         ("Htex", Htex), ("Wtex", Wtex), ("C", C),
         ("BOUNDARY", Int(boundary.rawValue))]
    }

    private static func textureForwardKernel(
        tex: MLXArray, uv: MLXArray,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int,
        boundary: BoundaryMode
    ) -> MLXArray {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        return fwdKernel(
            [tex, uv],
            template: texTmpl(N: N, Himg: Himg, Wimg: Wimg,
                              Htex: Htex, Wtex: Wtex, C: C, boundary: boundary),
            grid: (rounded, 1, 1),
            threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Himg, Wimg, C]],
            outputDTypes: [.float32]
        )[0]
    }

    private static func cubeForwardKernel(
        tex: MLXArray, dir: MLXArray,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int
    ) -> MLXArray {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        return cubeFwdKernel(
            [tex, dir],
            template: [("N", N), ("Himg", Himg), ("Wimg", Wimg),
                       ("Htex", Htex), ("Wtex", Wtex), ("C", C)],
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Himg, Wimg, C]],
            outputDTypes: [.float32]
        )[0]
    }

    private static func cubeBackwardKernel(
        tex: MLXArray, dir: MLXArray, dOut: MLXArray,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int
    ) -> MLXArray {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        return cubeGradTexKernel(
            [tex, dir, dOut],
            template: [("N", N), ("Himg", Himg), ("Wimg", Wimg),
                       ("Htex", Htex), ("Wtex", Wtex), ("C", C)],
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[N * 6 * Htex * Wtex * C]],
            outputDTypes: [.float32],
            initValue: 0.0
        )[0]
    }

    private static func textureNearestForwardKernel(
        tex: MLXArray, uv: MLXArray,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int,
        boundary: BoundaryMode
    ) -> MLXArray {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        return nearestFwdKernel(
            [tex, uv],
            template: texTmpl(N: N, Himg: Himg, Wimg: Wimg,
                              Htex: Htex, Wtex: Wtex, C: C, boundary: boundary),
            grid: (rounded, 1, 1),
            threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Himg, Wimg, C]],
            outputDTypes: [.float32]
        )[0]
    }

    private static func textureNearestBackwardKernel(
        tex: MLXArray, uv: MLXArray, dOut: MLXArray,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int,
        boundary: BoundaryMode
    ) -> MLXArray {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        return nearestGradTexKernel(
            [tex, uv, dOut],
            template: texTmpl(N: N, Himg: Himg, Wimg: Wimg,
                              Htex: Htex, Wtex: Wtex, C: C, boundary: boundary),
            grid: (rounded, 1, 1),
            threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Htex, Wtex, C]],
            outputDTypes: [.float32],
            initValue: 0.0
        )[0]
    }

    private static func textureBackwardKernels(
        tex: MLXArray, uv: MLXArray, dOut: MLXArray,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int,
        boundary: BoundaryMode
    ) -> (MLXArray, MLXArray) {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        let tmpl = texTmpl(N: N, Himg: Himg, Wimg: Wimg,
                           Htex: Htex, Wtex: Wtex, C: C, boundary: boundary)
        let dTex = gradTexKernel(
            [tex, uv, dOut], template: tmpl,
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Htex, Wtex, C]], outputDTypes: [.float32],
            initValue: 0.0
        )[0]
        let dUV = gradUVKernel(
            [tex, uv, dOut], template: tmpl,
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Himg, Wimg, 2]], outputDTypes: [.float32]
        )[0]
        return (dTex, dUV)
    }

    private static func triTmpl(
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int,
        numLevels: Int, boundary: BoundaryMode, mipLinear: Bool
    ) -> [(String, any KernelTemplateArg)] {
        [("N", N), ("Himg", Himg), ("Wimg", Wimg),
         ("Htex", Htex), ("Wtex", Wtex), ("C", C),
         ("NUM_LEVELS", numLevels),
         ("BOUNDARY", Int(boundary.rawValue)),
         ("MIP_LINEAR", mipLinear)]
    }

    private static func textureTrilinearForwardKernel(
        packed: MLXArray, uv: MLXArray, uvDA: MLXArray,
        offsets: MLXArray, levelH: MLXArray, levelW: MLXArray,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int,
        numLevels: Int, boundary: BoundaryMode, mipLinear: Bool
    ) -> MLXArray {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        return trilinearFwdKernel(
            [packed, uv, uvDA, offsets, levelH, levelW],
            template: triTmpl(N: N, Himg: Himg, Wimg: Wimg,
                              Htex: Htex, Wtex: Wtex, C: C,
                              numLevels: numLevels, boundary: boundary,
                              mipLinear: mipLinear),
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Himg, Wimg, C]],
            outputDTypes: [.float32]
        )[0]
    }

    private static func textureTrilinearBackwardKernels(
        packed: MLXArray, uv: MLXArray, uvDA: MLXArray, dOut: MLXArray,
        offsets: MLXArray, levelH: MLXArray, levelW: MLXArray, packedSize: Int,
        N: Int, Himg: Int, Wimg: Int, Htex: Int, Wtex: Int, C: Int,
        numLevels: Int, boundary: BoundaryMode, mipLinear: Bool
    ) -> (MLXArray, MLXArray, MLXArray) {
        let pixels = N * Himg * Wimg
        let rounded = ((pixels + textureTG - 1) / textureTG) * textureTG
        let tmpl = triTmpl(N: N, Himg: Himg, Wimg: Wimg,
                           Htex: Htex, Wtex: Wtex, C: C,
                           numLevels: numLevels, boundary: boundary,
                           mipLinear: mipLinear)
        let dPacked = trilinearGradPyramidKernel(
            [packed, uv, uvDA, offsets, levelH, levelW, dOut], template: tmpl,
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[packedSize]], outputDTypes: [.float32],
            initValue: 0.0
        )[0]
        let dUV = trilinearGradUVKernel(
            [packed, uv, uvDA, offsets, levelH, levelW, dOut], template: tmpl,
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Himg, Wimg, 2]], outputDTypes: [.float32]
        )[0]
        let dUVDA = trilinearGradUVDAKernel(
            [packed, uv, uvDA, offsets, levelH, levelW, dOut], template: tmpl,
            grid: (rounded, 1, 1), threadGroup: (textureTG, 1, 1),
            outputShapes: [[N, Himg, Wimg, 4]], outputDTypes: [.float32]
        )[0]
        return (dPacked, dUV, dUVDA)
    }
}
