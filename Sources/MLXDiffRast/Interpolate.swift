import Foundation
import MLX
import MLXFast

extension DiffRast {

    /// Pixel-derivative selector for `interpolate`. Currently only `.all` is supported
    /// (matches nvdiffrast `diff_attrs='all'`). A future revision will support
    /// `.indices([Int])` for selecting a subset of attributes.
    public enum DiffAttrs: Equatable {
        case all
    }

    // MARK: - Public API

    /// Interpolate per-vertex attributes across rasterized triangles.
    ///
    /// Mirrors `nvdiffrast.torch.interpolate(attr, rast, tri, rast_db, diff_attrs)`.
    ///
    /// Shapes:
    ///   - `attr`: `[N, V, A]` (instanced) or `[V, A]` (range mode, broadcast across batch).
    ///     The minibatch axis may be 1 and is broadcast to match `rast`.
    ///   - `rast`: `[N, H, W, 4]` from `rasterize` — channels `(u, v, z/w, tri_id+1)`.
    ///   - `tri`:  `[T, 3]` int32 vertex indices.
    ///   - `rastDB`: optional `[N, H, W, 4]` containing `(du/dx, du/dy, dv/dx, dv/dy)`.
    ///     Required when `diffAttrs` is non-nil.
    ///   - `diffAttrs`: when set, compute pixel-space attribute derivatives.
    ///
    /// Returns `(out, outDA)`:
    ///   - `out`:   `[N, H, W, A]` interpolated attributes (zero on empty pixels).
    ///   - `outDA`: `[N, H, W, 2*A]` per-pixel `(dA/dx, dA/dy)` pairs when `diffAttrs == .all`;
    ///              empty tensor `[N, H, W, 0]` otherwise.
    ///
    /// Differentiable w.r.t. `attr`, `rast`, and `rastDB`. `tri` is treated as constant.
    @discardableResult
    public static func interpolate(
        _ attr: MLXArray,
        rast: MLXArray,
        tri: MLXArray,
        rastDB: MLXArray? = nil,
        diffAttrs: DiffAttrs? = nil
    ) -> (out: MLXArray, outDA: MLXArray) {
        precondition(rast.ndim == 4 && rast.shape[3] == 4,
                     "interpolate: rast must be [N, H, W, 4] (got \(rast.shape))")
        precondition(tri.ndim == 2 && tri.shape[1] == 3,
                     "interpolate: tri must be [T, 3] (got \(tri.shape))")
        precondition(attr.ndim == 2 || attr.ndim == 3,
                     "interpolate: attr must be [V, A] or [N, V, A] (got \(attr.shape))")
        if diffAttrs != nil {
            precondition(rastDB != nil,
                         "interpolate: rastDB is required when diffAttrs is set")
            precondition(rastDB!.shape == rast.shape,
                         "interpolate: rastDB must match rast shape (got \(rastDB!.shape) vs \(rast.shape))")
        }

        let N = rast.shape[0], H = rast.shape[1], W = rast.shape[2]

        // Normalize attr to [N, V, A] via broadcast. Range-mode [V,A] gets a leading axis;
        // a [1, V, A] tensor is expanded to [N, V, A] so kernel indexing is uniform.
        var attrN = attr
        if attrN.ndim == 2 {
            attrN = attrN.expandedDimensions(axis: 0)
        }
        if attrN.shape[0] == 1 && N > 1 {
            attrN = broadcast(attrN, to: [N, attrN.shape[1], attrN.shape[2]])
        }
        precondition(attrN.shape[0] == N,
                     "interpolate: attr batch dim \(attrN.shape[0]) incompatible with rast batch \(N)")

        let A = attrN.shape[2]
        let triCaptured = tri.dtype == .int32 ? tri : tri.asType(.int32)

        if diffAttrs == nil {
            // Basic path: single output, no rast_db plumbing.
            let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
                [Self.fwd(attr: inputs[0], rast: inputs[1], tri: triCaptured,
                          N: N, H: H, W: W, A: A)]
            }
            let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
                let (dAttr, dRast) = Self.bwdBasic(
                    attr: primals[0], rast: primals[1], tri: triCaptured,
                    dOut: cotangents[0], N: N, H: H, W: W, A: A)
                return [dAttr, dRast]
            }
            let custom = CustomFunction { Forward(fwd); VJP(vjp) }
            let out = custom([attrN, rast])[0]
            let emptyDA = MLXArray.zeros([N, H, W, 0])
            return (out, emptyDA)
        } else {
            // DA path: two outputs, three differentiable primals.
            let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
                let (o, oda) = Self.fwdDA(
                    attr: inputs[0], rast: inputs[1], rastDB: inputs[2], tri: triCaptured,
                    N: N, H: H, W: W, A: A)
                return [o, oda]
            }
            let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
                let (dAttr, dRast, dRastDB) = Self.bwdDA(
                    attr: primals[0], rast: primals[1], rastDB: primals[2], tri: triCaptured,
                    dOut: cotangents[0], dOutDA: cotangents[1],
                    N: N, H: H, W: W, A: A)
                return [dAttr, dRast, dRastDB]
            }
            let custom = CustomFunction { Forward(fwd); VJP(vjp) }
            let outs = custom([attrN, rast, rastDB!])
            return (outs[0], outs[1])
        }
    }

    // MARK: - Kernel sources

    /// Forward (basic). Writes only `out`.
    private static let fwdSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint w = pix % (uint)W;
        uint h = (pix / (uint)W) % (uint)H;
        uint n = pix / ((uint)H * (uint)W);

        uint rast_base = ((n * (uint)H + h) * (uint)W + w) * 4u;
        float u  = rast[rast_base + 0];
        float v  = rast[rast_base + 1];
        float ti = rast[rast_base + 3];
        int   t  = (int)ti - 1;

        uint out_base = ((n * (uint)H + h) * (uint)W + w) * (uint)A;
        if (t < 0) {
            for (int a = 0; a < A; ++a) out[out_base + (uint)a] = 0.0f;
            return;
        }

        int i0 = tri[t * 3 + 0];
        int i1 = tri[t * 3 + 1];
        int i2 = tri[t * 3 + 2];
        float bw = 1.0f - u - v;
        uint attr_n_base = n * (uint)V * (uint)A;

        for (int a = 0; a < A; ++a) {
            float a0 = attr[attr_n_base + (uint)i0 * (uint)A + (uint)a];
            float a1 = attr[attr_n_base + (uint)i1 * (uint)A + (uint)a];
            float a2 = attr[attr_n_base + (uint)i2 * (uint)A + (uint)a];
            out[out_base + (uint)a] = bw * a0 + u * a1 + v * a2;
        }
    """

    /// Forward (DA). Writes both `out` and `out_da`.
    /// `out_da[..., 2a]   = (a1-a0)*du/dx + (a2-a0)*dv/dx`
    /// `out_da[..., 2a+1] = (a1-a0)*du/dy + (a2-a0)*dv/dy`
    private static let fwdDASource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint w = pix % (uint)W;
        uint h = (pix / (uint)W) % (uint)H;
        uint n = pix / ((uint)H * (uint)W);

        uint rast_base = ((n * (uint)H + h) * (uint)W + w) * 4u;
        float u  = rast[rast_base + 0];
        float v  = rast[rast_base + 1];
        float ti = rast[rast_base + 3];
        int   t  = (int)ti - 1;

        uint out_base  = ((n * (uint)H + h) * (uint)W + w) * (uint)A;
        uint outda_base = ((n * (uint)H + h) * (uint)W + w) * (uint)(2 * A);

        if (t < 0) {
            for (int a = 0; a < A; ++a) out[out_base + (uint)a] = 0.0f;
            for (int k = 0; k < 2 * A; ++k) out_da[outda_base + (uint)k] = 0.0f;
            return;
        }

        int i0 = tri[t * 3 + 0];
        int i1 = tri[t * 3 + 1];
        int i2 = tri[t * 3 + 2];
        float bw = 1.0f - u - v;
        uint attr_n_base = n * (uint)V * (uint)A;

        float udx = rast_db[rast_base + 0];
        float udy = rast_db[rast_base + 1];
        float vdx = rast_db[rast_base + 2];
        float vdy = rast_db[rast_base + 3];

        for (int a = 0; a < A; ++a) {
            float a0 = attr[attr_n_base + (uint)i0 * (uint)A + (uint)a];
            float a1 = attr[attr_n_base + (uint)i1 * (uint)A + (uint)a];
            float a2 = attr[attr_n_base + (uint)i2 * (uint)A + (uint)a];
            out[out_base + (uint)a] = bw * a0 + u * a1 + v * a2;

            float d01 = a1 - a0;
            float d02 = a2 - a0;
            out_da[outda_base + (uint)(2 * a + 0)] = d01 * udx + d02 * vdx;
            out_da[outda_base + (uint)(2 * a + 1)] = d01 * udy + d02 * vdy;
        }
    """

    /// d_attr (basic). Atomic scatter-add from d_out.
    private static let gradAttrSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint w = pix % (uint)W;
        uint h = (pix / (uint)W) % (uint)H;
        uint n = pix / ((uint)H * (uint)W);

        uint rast_base = ((n * (uint)H + h) * (uint)W + w) * 4u;
        float u  = rast[rast_base + 0];
        float v  = rast[rast_base + 1];
        float ti = rast[rast_base + 3];
        int   t  = (int)ti - 1;
        if (t < 0) return;

        int i0 = tri[t * 3 + 0];
        int i1 = tri[t * 3 + 1];
        int i2 = tri[t * 3 + 2];
        float bw = 1.0f - u - v;

        uint dout_base = ((n * (uint)H + h) * (uint)W + w) * (uint)A;
        uint d_attr_n_base = n * (uint)V * (uint)A;

        for (int a = 0; a < A; ++a) {
            float g = d_out[dout_base + (uint)a];
            atomic_fetch_add_explicit(&d_attr[d_attr_n_base + (uint)i0 * (uint)A + (uint)a],
                                      bw * g, memory_order_relaxed);
            atomic_fetch_add_explicit(&d_attr[d_attr_n_base + (uint)i1 * (uint)A + (uint)a],
                                      u  * g, memory_order_relaxed);
            atomic_fetch_add_explicit(&d_attr[d_attr_n_base + (uint)i2 * (uint)A + (uint)a],
                                      v  * g, memory_order_relaxed);
        }
    """

    /// d_attr (DA). Atomic scatter-add from d_out AND d_out_da.
    /// Contribution from d_out_da: for each attribute a, with cotangents
    ///   gx = d_out_da[..., 2a], gy = d_out_da[..., 2a+1]:
    ///   d/d_a0 += -(gx*udx + gy*udy + gx*vdx + gy*vdy)
    ///   d/d_a1 += gx*udx + gy*udy
    ///   d/d_a2 += gx*vdx + gy*vdy
    private static let gradAttrDASource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint w = pix % (uint)W;
        uint h = (pix / (uint)W) % (uint)H;
        uint n = pix / ((uint)H * (uint)W);

        uint rast_base = ((n * (uint)H + h) * (uint)W + w) * 4u;
        float u  = rast[rast_base + 0];
        float v  = rast[rast_base + 1];
        float ti = rast[rast_base + 3];
        int   t  = (int)ti - 1;
        if (t < 0) return;

        int i0 = tri[t * 3 + 0];
        int i1 = tri[t * 3 + 1];
        int i2 = tri[t * 3 + 2];
        float bw = 1.0f - u - v;

        float udx = rast_db[rast_base + 0];
        float udy = rast_db[rast_base + 1];
        float vdx = rast_db[rast_base + 2];
        float vdy = rast_db[rast_base + 3];

        uint dout_base   = ((n * (uint)H + h) * (uint)W + w) * (uint)A;
        uint doutda_base = ((n * (uint)H + h) * (uint)W + w) * (uint)(2 * A);
        uint d_attr_n_base = n * (uint)V * (uint)A;

        for (int a = 0; a < A; ++a) {
            float g  = d_out[dout_base + (uint)a];
            float gx = d_out_da[doutda_base + (uint)(2 * a + 0)];
            float gy = d_out_da[doutda_base + (uint)(2 * a + 1)];

            float u_term = gx * udx + gy * udy;
            float v_term = gx * vdx + gy * vdy;

            float g0 = bw * g - u_term - v_term;
            float g1 = u  * g + u_term;
            float g2 = v  * g + v_term;

            atomic_fetch_add_explicit(&d_attr[d_attr_n_base + (uint)i0 * (uint)A + (uint)a],
                                      g0, memory_order_relaxed);
            atomic_fetch_add_explicit(&d_attr[d_attr_n_base + (uint)i1 * (uint)A + (uint)a],
                                      g1, memory_order_relaxed);
            atomic_fetch_add_explicit(&d_attr[d_attr_n_base + (uint)i2 * (uint)A + (uint)a],
                                      g2, memory_order_relaxed);
        }
    """

    /// d_rast (basic/DA). Per-pixel from d_out only. (out_da does not depend on u,v.)
    private static let gradRastSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint w = pix % (uint)W;
        uint h = (pix / (uint)W) % (uint)H;
        uint n = pix / ((uint)H * (uint)W);

        uint rast_base = ((n * (uint)H + h) * (uint)W + w) * 4u;
        float ti = rast[rast_base + 3];
        int   t  = (int)ti - 1;

        d_rast[rast_base + 2] = 0.0f;
        d_rast[rast_base + 3] = 0.0f;

        if (t < 0) {
            d_rast[rast_base + 0] = 0.0f;
            d_rast[rast_base + 1] = 0.0f;
            return;
        }

        int i0 = tri[t * 3 + 0];
        int i1 = tri[t * 3 + 1];
        int i2 = tri[t * 3 + 2];

        uint dout_base   = ((n * (uint)H + h) * (uint)W + w) * (uint)A;
        uint attr_n_base = n * (uint)V * (uint)A;

        float du = 0.0f, dv = 0.0f;
        for (int a = 0; a < A; ++a) {
            float a0 = attr[attr_n_base + (uint)i0 * (uint)A + (uint)a];
            float a1 = attr[attr_n_base + (uint)i1 * (uint)A + (uint)a];
            float a2 = attr[attr_n_base + (uint)i2 * (uint)A + (uint)a];
            float g  = d_out[dout_base + (uint)a];
            du += (a1 - a0) * g;
            dv += (a2 - a0) * g;
        }
        d_rast[rast_base + 0] = du;
        d_rast[rast_base + 1] = dv;
    """

    /// d_rast_db (DA). Per-pixel from d_out_da. Each (udx, udy, vdx, vdy) channel.
    private static let gradRastDBSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint w = pix % (uint)W;
        uint h = (pix / (uint)W) % (uint)H;
        uint n = pix / ((uint)H * (uint)W);

        uint rast_base = ((n * (uint)H + h) * (uint)W + w) * 4u;
        float ti = rast[rast_base + 3];
        int   t  = (int)ti - 1;

        if (t < 0) {
            d_rast_db[rast_base + 0] = 0.0f;
            d_rast_db[rast_base + 1] = 0.0f;
            d_rast_db[rast_base + 2] = 0.0f;
            d_rast_db[rast_base + 3] = 0.0f;
            return;
        }

        int i0 = tri[t * 3 + 0];
        int i1 = tri[t * 3 + 1];
        int i2 = tri[t * 3 + 2];

        uint doutda_base = ((n * (uint)H + h) * (uint)W + w) * (uint)(2 * A);
        uint attr_n_base = n * (uint)V * (uint)A;

        float d_udx = 0.0f, d_udy = 0.0f, d_vdx = 0.0f, d_vdy = 0.0f;
        for (int a = 0; a < A; ++a) {
            float a0 = attr[attr_n_base + (uint)i0 * (uint)A + (uint)a];
            float a1 = attr[attr_n_base + (uint)i1 * (uint)A + (uint)a];
            float a2 = attr[attr_n_base + (uint)i2 * (uint)A + (uint)a];
            float d01 = a1 - a0;
            float d02 = a2 - a0;
            float gx = d_out_da[doutda_base + (uint)(2 * a + 0)];
            float gy = d_out_da[doutda_base + (uint)(2 * a + 1)];
            d_udx += d01 * gx;
            d_udy += d01 * gy;
            d_vdx += d02 * gx;
            d_vdy += d02 * gy;
        }
        d_rast_db[rast_base + 0] = d_udx;
        d_rast_db[rast_base + 1] = d_udy;
        d_rast_db[rast_base + 2] = d_vdx;
        d_rast_db[rast_base + 3] = d_vdy;
    """

    // MARK: - Kernel instances

    private static let fwdKernel = MLXFast.metalKernel(
        name: "diffrast_interp_fwd",
        inputNames: ["attr", "rast", "tri"], outputNames: ["out"],
        source: fwdSource)

    private static let fwdDAKernel = MLXFast.metalKernel(
        name: "diffrast_interp_fwd_da",
        inputNames: ["attr", "rast", "rast_db", "tri"], outputNames: ["out", "out_da"],
        source: fwdDASource)

    private static let gradAttrKernel = MLXFast.metalKernel(
        name: "diffrast_interp_grad_attr",
        inputNames: ["attr", "rast", "tri", "d_out"], outputNames: ["d_attr"],
        source: gradAttrSource, atomicOutputs: true)

    private static let gradAttrDAKernel = MLXFast.metalKernel(
        name: "diffrast_interp_grad_attr_da",
        inputNames: ["attr", "rast", "rast_db", "tri", "d_out", "d_out_da"],
        outputNames: ["d_attr"],
        source: gradAttrDASource, atomicOutputs: true)

    private static let gradRastKernel = MLXFast.metalKernel(
        name: "diffrast_interp_grad_rast",
        inputNames: ["attr", "rast", "tri", "d_out"], outputNames: ["d_rast"],
        source: gradRastSource)

    private static let gradRastDBKernel = MLXFast.metalKernel(
        name: "diffrast_interp_grad_rast_db",
        inputNames: ["attr", "rast", "tri", "d_out_da"], outputNames: ["d_rast_db"],
        source: gradRastDBSource)

    private static let tg = 256

    private static func gridSpec(_ pixels: Int) -> ((Int, Int, Int), (Int, Int, Int)) {
        let rounded = ((pixels + tg - 1) / tg) * tg
        return ((rounded, 1, 1), (tg, 1, 1))
    }

    private static func tmpl(N: Int, H: Int, W: Int, A: Int, V: Int)
        -> [(String, any KernelTemplateArg)]
    {
        [("N", N), ("H", H), ("W", W), ("A", A), ("V", V)]
    }

    // MARK: - Kernel dispatchers

    private static func fwd(attr: MLXArray, rast: MLXArray, tri: MLXArray,
                            N: Int, H: Int, W: Int, A: Int) -> MLXArray
    {
        let V = attr.shape[1]
        let (g, t) = gridSpec(N * H * W)
        return fwdKernel([attr, rast, tri],
                         template: tmpl(N: N, H: H, W: W, A: A, V: V),
                         grid: g, threadGroup: t,
                         outputShapes: [[N, H, W, A]],
                         outputDTypes: [.float32])[0]
    }

    private static func fwdDA(attr: MLXArray, rast: MLXArray, rastDB: MLXArray, tri: MLXArray,
                              N: Int, H: Int, W: Int, A: Int) -> (MLXArray, MLXArray)
    {
        let V = attr.shape[1]
        let (g, t) = gridSpec(N * H * W)
        let outs = fwdDAKernel([attr, rast, rastDB, tri],
                               template: tmpl(N: N, H: H, W: W, A: A, V: V),
                               grid: g, threadGroup: t,
                               outputShapes: [[N, H, W, A], [N, H, W, 2 * A]],
                               outputDTypes: [.float32, .float32])
        return (outs[0], outs[1])
    }

    private static func bwdBasic(attr: MLXArray, rast: MLXArray, tri: MLXArray, dOut: MLXArray,
                                 N: Int, H: Int, W: Int, A: Int) -> (MLXArray, MLXArray)
    {
        let V = attr.shape[1]
        let (g, t) = gridSpec(N * H * W)
        let dAttr = gradAttrKernel([attr, rast, tri, dOut],
                                   template: tmpl(N: N, H: H, W: W, A: A, V: V),
                                   grid: g, threadGroup: t,
                                   outputShapes: [[N, V, A]], outputDTypes: [.float32],
                                   initValue: 0.0)[0]
        let dRast = gradRastKernel([attr, rast, tri, dOut],
                                   template: tmpl(N: N, H: H, W: W, A: A, V: V),
                                   grid: g, threadGroup: t,
                                   outputShapes: [[N, H, W, 4]], outputDTypes: [.float32])[0]
        return (dAttr, dRast)
    }

    private static func bwdDA(
        attr: MLXArray, rast: MLXArray, rastDB: MLXArray, tri: MLXArray,
        dOut: MLXArray, dOutDA: MLXArray,
        N: Int, H: Int, W: Int, A: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let V = attr.shape[1]
        let (g, t) = gridSpec(N * H * W)
        let tparams = tmpl(N: N, H: H, W: W, A: A, V: V)
        let dAttr = gradAttrDAKernel([attr, rast, rastDB, tri, dOut, dOutDA],
                                     template: tparams, grid: g, threadGroup: t,
                                     outputShapes: [[N, V, A]], outputDTypes: [.float32],
                                     initValue: 0.0)[0]
        let dRast = gradRastKernel([attr, rast, tri, dOut],
                                   template: tparams, grid: g, threadGroup: t,
                                   outputShapes: [[N, H, W, 4]], outputDTypes: [.float32])[0]
        let dRastDB = gradRastDBKernel([attr, rast, tri, dOutDA],
                                       template: tparams, grid: g, threadGroup: t,
                                       outputShapes: [[N, H, W, 4]], outputDTypes: [.float32])[0]
        return (dAttr, dRast, dRastDB)
    }
}
