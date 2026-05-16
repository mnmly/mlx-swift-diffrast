import Foundation
import MLX
import MLXFast

extension DiffRast {

    /// Build the triangle-edge adjacency table consumed by `antialias`.
    ///
    /// Returns an `[T, 3]` int32 tensor where entry `[t, k]` is the index of
    /// the triangle that shares the edge *opposite vertex k* of triangle `t`
    /// (i.e. the edge connecting vertices `tri[t, (k+1)%3]` and
    /// `tri[t, (k+2)%3]`). `-1` marks a boundary edge with no neighbor.
    ///
    /// This is pure Swift / CPU work and is cheap to call once per topology
    /// change; you typically cache the result alongside `tri`.
    public static func antialiasConstructTopologyHash(_ tri: MLXArray) -> MLXArray {
        precondition(tri.ndim == 2 && tri.shape[1] == 3,
                     "antialiasConstructTopologyHash: tri must be [T, 3] (got \(tri.shape))")
        let T = tri.shape[0]
        let triI32 = tri.dtype == .int32 ? tri : tri.asType(.int32)
        let triFlat = triI32.asArray(Int32.self)

        struct EdgeRef { let a: Int32; let b: Int32; let t: Int; let k: Int }
        var edges: [EdgeRef] = []
        edges.reserveCapacity(T * 3)
        for t in 0..<T {
            let v0 = triFlat[t * 3 + 0]
            let v1 = triFlat[t * 3 + 1]
            let v2 = triFlat[t * 3 + 2]
            let pairs: [(Int32, Int32)] = [(v1, v2), (v2, v0), (v0, v1)]
            for (k, (p, q)) in pairs.enumerated() {
                let lo = min(p, q), hi = max(p, q)
                edges.append(EdgeRef(a: lo, b: hi, t: t, k: k))
            }
        }
        edges.sort { l, r in
            if l.a != r.a { return l.a < r.a }
            return l.b < r.b
        }

        var neighbor = [Int32](repeating: -1, count: T * 3)
        var i = 0
        while i < edges.count {
            let e = edges[i]
            var j = i + 1
            while j < edges.count && edges[j].a == e.a && edges[j].b == e.b { j += 1 }
            if j - i == 2 {
                let e0 = edges[i], e1 = edges[i + 1]
                neighbor[e0.t * 3 + e0.k] = Int32(e1.t)
                neighbor[e1.t * 3 + e1.k] = Int32(e0.t)
            }
            i = j
        }
        return MLXArray(neighbor, [T, 3])
    }

    /// Silhouette-aware antialiasing — M4.2 (forward + d_color backward).
    ///
    /// For each pair of 4-connected pixels, if their `rast` tri-ids identify a
    /// *true silhouette* (validated by `topologyHash`), the foreground
    /// triangle's silhouette edge is projected into screen-pixel space and the
    /// 1D coverage `α` of the background pixel by the foreground triangle is
    /// computed from where that edge crosses the pixel's centerline. The
    /// background pixel's output is then `(1 - α) · color_bg + α · color_fg`.
    /// Each pair contributes to at most one pixel; per-direction contributions
    /// compose additively if the same pixel borders multiple silhouettes.
    ///
    /// **Differentiability (M4.2 status):** cotangents flow through `color`
    /// correctly (atomic scatter with the same `α`). `d_pos` is still **zero**.
    /// The chain rule from `α` back through the edge-line intersection into
    /// `pos` is the focus of M4.3 — it mirrors rasterize's M2.3 in shape but
    /// runs through line-intersection algebra instead of barycentrics.
    ///
    /// If `topologyHash` is `nil` it will be computed via
    /// `antialiasConstructTopologyHash(tri)`; pass a cached value when
    /// optimizing inside a hot loop.
    public static func antialias(
        color: MLXArray,
        rast: MLXArray,
        pos: MLXArray,
        tri: MLXArray,
        topologyHash: MLXArray? = nil
    ) -> MLXArray {
        precondition(color.ndim == 4,
                     "antialias: color must be [N, H, W, C] (got \(color.shape))")
        precondition(rast.ndim == 4 && rast.shape[3] == 4 && rast.shape[0..<3] == color.shape[0..<3],
                     "antialias: rast must match color in [N, H, W] and have 4 channels")
        precondition(pos.ndim == 3 && pos.shape[2] == 4,
                     "antialias: pos must be [N, V, 4] (got \(pos.shape))")
        precondition(tri.ndim == 2 && tri.shape[1] == 3,
                     "antialias: tri must be [T, 3] (got \(tri.shape))")

        let N = color.shape[0], H = color.shape[1], W = color.shape[2], C = color.shape[3]
        let T = tri.shape[0], V = pos.shape[1]
        let triI32 = tri.dtype == .int32 ? tri : tri.asType(.int32)
        let hash = topologyHash ?? antialiasConstructTopologyHash(triI32)
        let hashI32 = hash.dtype == .int32 ? hash : hash.asType(.int32)

        let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
            [Self.antialiasForwardKernel(
                color: inputs[0], rast: rast, pos: inputs[1],
                tri: triI32, topology: hashI32,
                N: N, H: H, W: W, C: C, T: T, V: V)]
        }
        let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
            let dColor = Self.antialiasBackwardColorKernel(
                color: primals[0], rast: rast, pos: primals[1],
                tri: triI32, topology: hashI32, dOut: cotangents[0],
                N: N, H: H, W: W, C: C, T: T, V: V)
            return [dColor, MLXArray.zeros(primals[1].shape, dtype: .float32)]
        }
        let custom = CustomFunction { Forward(fwd); VJP(vjp) }
        return custom([color, pos])[0]
    }

    // MARK: - Shared silhouette logic (Metal header)

    /// Header carries only a struct declaration. Helpers are inlined into the
    /// kernel bodies to avoid the address-space-qualifier dance with the
    /// inputs that MLXFast exposes (constant vs device qualifier matching is
    /// finicky enough that duplicating ~30 lines between fwd and bwd is
    /// cheaper than maintaining the wrapper functions correctly).
    private static let aaHeader = """
        struct PixelXY { float x; float y; };
    """

    /// Reused once each in `fwdSource` and `bwdColorSource` — kept as a Swift
    /// constant so the silhouette logic stays single-source.
    private static let silhouetteBlock = """
        int tri_p = (int)rast[rast_base + 3] - 1;
        uint pos_n_base = pn * (uint)V * 4u;

        // Per-direction outputs filled in below; consumed by the kernel body.
        float alpha_arr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint  nbase_arr[4] = {0u, 0u, 0u, 0u};
        bool  active_arr[4] = {false, false, false, false};
        float alpha_sum_local = 0.0f;

        for (int dir = 0; dir < 4; ++dir) {
            int dh = 0, dw = 0;
            if (dir == 0) dw = 1;
            else if (dir == 1) dh = 1;
            else if (dir == 2) dw = -1;
            else dh = -1;

            int nh = (int)ph + dh;
            int nw = (int)pw + dw;
            if (nh < 0 || nh >= H || nw < 0 || nw >= W) continue;

            uint n_rast_base = ((pn * (uint)H + (uint)nh) * (uint)W + (uint)nw) * 4u;
            int tri_n = (int)rast[n_rast_base + 3] - 1;

            if (tri_p == tri_n) continue;
            if (tri_p < 0 && tri_n < 0) continue;

            int fg_tri, target;
            bool p_is_bg;
            if (tri_p >= 0 && tri_n >= 0) {
                // Both covered: silhouette only if they share an edge.
                int k_at_p = -1;
                if (topology[tri_p * 3 + 0] == tri_n) k_at_p = 0;
                else if (topology[tri_p * 3 + 1] == tri_n) k_at_p = 1;
                else if (topology[tri_p * 3 + 2] == tri_n) k_at_p = 2;
                if (k_at_p < 0) continue;
                fg_tri = tri_n; target = tri_p; p_is_bg = true;
            } else if (tri_p >= 0) {
                fg_tri = tri_p; target = -1; p_is_bg = false;
            } else {
                fg_tri = tri_n; target = -1; p_is_bg = true;
            }
            if (!p_is_bg) continue;

            int k = -1;
            if (topology[fg_tri * 3 + 0] == target) k = 0;
            else if (topology[fg_tri * 3 + 1] == target) k = 1;
            else if (topology[fg_tri * 3 + 2] == target) k = 2;
            if (k < 0) continue;

            int i_e0 = tri[fg_tri * 3 + ((k + 1) % 3)];
            int i_e1 = tri[fg_tri * 3 + ((k + 2) % 3)];

            // Project edge endpoints to pixel-space (perspective divide + NDC mapping).
            float w0 = pos[pos_n_base + (uint)i_e0 * 4u + 3];
            float w1 = pos[pos_n_base + (uint)i_e1 * 4u + 3];
            PixelXY e0, e1;
            e0.x = (pos[pos_n_base + (uint)i_e0 * 4u + 0] / w0 + 1.0f) * (float)W * 0.5f - 0.5f;
            e0.y = (1.0f - pos[pos_n_base + (uint)i_e0 * 4u + 1] / w0) * (float)H * 0.5f - 0.5f;
            e1.x = (pos[pos_n_base + (uint)i_e1 * 4u + 0] / w1 + 1.0f) * (float)W * 0.5f - 0.5f;
            e1.y = (1.0f - pos[pos_n_base + (uint)i_e1 * 4u + 1] / w1) * (float)H * 0.5f - 0.5f;

            float alpha = 0.0f;
            if (dh == 0) {
                // Horizontal pair: pw_p = left pixel column.
                int pw_p = dw > 0 ? (int)pw : (int)pw - 1;
                float fg_center_x = (float)(pw_p == (int)pw ? nw : pw) + 0.5f;
                float dyv = e1.y - e0.y;
                if (fabs(dyv) >= 1e-7f) {
                    float row_y = (float)ph + 0.5f;
                    float t = (row_y - e0.y) / dyv;
                    if (t >= 0.0f && t <= 1.0f) {
                        float xe = e0.x + t * (e1.x - e0.x);
                        float pl = (float)pw;
                        float pr = (float)(pw + 1);
                        float clipped = fmin(fmax(xe, pl), pr);
                        alpha = (fg_center_x > xe) ? (pr - clipped) : (clipped - pl);
                    }
                }
            } else {
                int ph_p = dh > 0 ? (int)ph : (int)ph - 1;
                float fg_center_y = (float)(ph_p == (int)ph ? nh : ph) + 0.5f;
                float dxv = e1.x - e0.x;
                if (fabs(dxv) >= 1e-7f) {
                    float col_x = (float)pw + 0.5f;
                    float t = (col_x - e0.x) / dxv;
                    if (t >= 0.0f && t <= 1.0f) {
                        float ye = e0.y + t * (e1.y - e0.y);
                        float pt = (float)ph;
                        float pb = (float)(ph + 1);
                        float clipped = fmin(fmax(ye, pt), pb);
                        alpha = (fg_center_y > ye) ? (pb - clipped) : (clipped - pt);
                    }
                }
            }
            alpha = fmin(fmax(alpha, 0.0f), 1.0f);
            if (alpha <= 0.0f) continue;

            alpha_arr[dir] = alpha;
            nbase_arr[dir] = ((pn * (uint)H + (uint)nh) * (uint)W + (uint)nw) * (uint)C;
            active_arr[dir] = true;
            alpha_sum_local += alpha;
        }
    """

    // MARK: - Forward kernel

    /// Per-pixel: detect silhouettes across 4 neighbors. Per the
    /// `silhouetteBlock` convention, only mutate this pixel's output (no
    /// atomics in forward).
    private static var fwdSource: String { """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint pw = pix % (uint)W;
        uint ph = (pix / (uint)W) % (uint)H;
        uint pn = pix / ((uint)H * (uint)W);

        uint rast_base = ((pn * (uint)H + ph) * (uint)W + pw) * 4u;
        uint color_base = ((pn * (uint)H + ph) * (uint)W + pw) * (uint)C;

        \(silhouetteBlock)

        // Initialize out = color, then accumulate per-direction contributions.
        for (int c = 0; c < C; ++c) {
            float cp = color[color_base + (uint)c];
            float acc = cp;
            for (int dir = 0; dir < 4; ++dir) {
                if (!active_arr[dir]) continue;
                float cn = color[nbase_arr[dir] + (uint)c];
                acc += alpha_arr[dir] * (cn - cp);
            }
            out[color_base + (uint)c] = acc;
        }
    """ }

    /// Backward for d_color:
    ///   out_p = color_p + Σ_dir α_dir · (color_n_dir - color_p)
    ///         = color_p · (1 - Σ α_dir) + Σ α_dir · color_n_dir
    ///   d_color_p   += d_out_p · (1 - Σ α_dir)
    ///   d_color_n_d += d_out_p · α_d        (scatter into the neighbors)
    ///
    /// The first term is local to p (non-atomic). The second term writes to
    /// neighbor positions, so we use atomic scatter-add into d_color.
    private static var bwdColorSource: String { """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint pw = pix % (uint)W;
        uint ph = (pix / (uint)W) % (uint)H;
        uint pn = pix / ((uint)H * (uint)W);

        uint rast_base = ((pn * (uint)H + ph) * (uint)W + pw) * 4u;
        uint dout_base = ((pn * (uint)H + ph) * (uint)W + pw) * (uint)C;

        \(silhouetteBlock)

        for (int c = 0; c < C; ++c) {
            float g = d_out[dout_base + (uint)c];
            atomic_fetch_add_explicit(
                &d_color[dout_base + (uint)c], g * (1.0f - alpha_sum_local),
                memory_order_relaxed);
            for (int dir = 0; dir < 4; ++dir) {
                if (!active_arr[dir]) continue;
                atomic_fetch_add_explicit(
                    &d_color[nbase_arr[dir] + (uint)c], g * alpha_arr[dir],
                    memory_order_relaxed);
            }
        }
    """ }

    // MARK: - Kernel instances + dispatchers

    private static let fwdKernel = MLXFast.metalKernel(
        name: "diffrast_antialias_fwd",
        inputNames: ["color", "rast", "pos", "tri", "topology"],
        outputNames: ["out"],
        source: fwdSource, header: aaHeader)

    private static let bwdColorKernel = MLXFast.metalKernel(
        name: "diffrast_antialias_bwd_color",
        inputNames: ["color", "rast", "pos", "tri", "topology", "d_out"],
        outputNames: ["d_color"],
        source: bwdColorSource, header: aaHeader,
        atomicOutputs: true)

    private static let aaTG = 256

    private static func aaTmpl(N: Int, H: Int, W: Int, C: Int, T: Int, V: Int)
        -> [(String, any KernelTemplateArg)]
    {
        [("N", N), ("H", H), ("W", W), ("C", C), ("T", T), ("V", V)]
    }

    private static func antialiasForwardKernel(
        color: MLXArray, rast: MLXArray, pos: MLXArray, tri: MLXArray, topology: MLXArray,
        N: Int, H: Int, W: Int, C: Int, T: Int, V: Int
    ) -> MLXArray {
        let pixels = N * H * W
        let rounded = ((pixels + aaTG - 1) / aaTG) * aaTG
        return fwdKernel(
            [color, rast, pos, tri, topology],
            template: aaTmpl(N: N, H: H, W: W, C: C, T: T, V: V),
            grid: (rounded, 1, 1), threadGroup: (aaTG, 1, 1),
            outputShapes: [[N, H, W, C]], outputDTypes: [.float32]
        )[0]
    }

    private static func antialiasBackwardColorKernel(
        color: MLXArray, rast: MLXArray, pos: MLXArray, tri: MLXArray, topology: MLXArray,
        dOut: MLXArray,
        N: Int, H: Int, W: Int, C: Int, T: Int, V: Int
    ) -> MLXArray {
        let pixels = N * H * W
        let rounded = ((pixels + aaTG - 1) / aaTG) * aaTG
        return bwdColorKernel(
            [color, rast, pos, tri, topology, dOut],
            template: aaTmpl(N: N, H: H, W: W, C: C, T: T, V: V),
            grid: (rounded, 1, 1), threadGroup: (aaTG, 1, 1),
            outputShapes: [[N, H, W, C]], outputDTypes: [.float32],
            initValue: 0.0
        )[0]
    }
}
