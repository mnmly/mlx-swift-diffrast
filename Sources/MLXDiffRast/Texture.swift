import Foundation
import MLX
import MLXFast

extension DiffRast {

    /// Boundary handling for `texture`. Matches the nvdiffrast names for the
    /// modes that are implemented today.
    public enum BoundaryMode: Int32 {
        case wrap = 0       // tile the texture (periodic)
        case clamp = 1      // edge-extend
        case zero = 2       // sample 0 outside [0, 1] in either axis
        // .cube is reserved for cube textures (M3 follow-up)
    }

    /// Bilinear texture sampling (M3.1: no mipmap, no cube textures).
    ///
    /// Shapes:
    ///   - `tex`: `[N, H_tex, W_tex, C]`. The minibatch axis may be 1 and is
    ///     broadcast to match `uv`'s batch dim.
    ///   - `uv`:  `[N, H_img, W_img, 2]` with channels `(u, v)` in `[0, 1]`.
    ///     `u` indexes the W axis (horizontal); `v` indexes the H axis with
    ///     `v=0` at the top of the texture (image convention).
    /// Returns:
    ///   - `out`: `[N, H_img, W_img, C]`.
    ///
    /// Differentiable w.r.t. `tex` and `uv`. The mipmap-LOD selection
    /// (`uv_da`, `mip_level_bias`) and the trilinear / cube paths are deferred.
    public static func texture(
        _ tex: MLXArray,
        uv: MLXArray,
        boundaryMode: BoundaryMode = .wrap
    ) -> MLXArray {
        precondition(tex.ndim == 4,
                     "texture: tex must be [N, H_tex, W_tex, C] (got \(tex.shape))")
        precondition(uv.ndim == 4 && uv.shape[3] == 2,
                     "texture: uv must be [N, H_img, W_img, 2] (got \(uv.shape))")
        precondition(tex.shape[0] == 1 || tex.shape[0] == uv.shape[0],
                     "texture: tex batch dim \(tex.shape[0]) incompatible with uv batch \(uv.shape[0])")

        let N = uv.shape[0]
        let Himg = uv.shape[1], Wimg = uv.shape[2]
        let Htex = tex.shape[1], Wtex = tex.shape[2], C = tex.shape[3]

        // Broadcast tex batch dim if needed (kernel does straight indexing).
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
}
