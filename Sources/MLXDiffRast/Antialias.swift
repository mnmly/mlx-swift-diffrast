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

    /// Silhouette-aware antialiasing — fully differentiable.
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
    /// Differentiable w.r.t. both `color` and `pos`. The `pos` gradient flows
    /// through the line-intersection formula and the perspective divide back
    /// into clip-space vertex positions — closing the loop for inverse-
    /// rendering pipelines that optimize geometry against a pixel-space loss.
    /// `α` saturates (clipped to `[0, 1]`) on the boundary; gradients are zero
    /// outside the interior of that interval, which is the analytically correct
    /// behavior for the clamped silhouette.
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
            let dPos = Self.antialiasBackwardPosKernel(
                color: primals[0], rast: rast, pos: primals[1],
                tri: triI32, topology: hashI32, dOut: cotangents[0],
                N: N, H: H, W: W, C: C, T: T, V: V)
            return [dColor, dPos]
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

    /// Reused once each in `fwdSource`, `bwdColorSource`, and `bwdPosSource`.
    ///
    /// **Algorithm:** for each pixel p and each of its 4 neighbors:
    ///   1. If they share a tri-id (or are both empty), skip.
    ///   2. Validate via `topologyHash` that this is a true silhouette pair.
    ///   3. Project the silhouette edge to pixel-space (`e.x = (NDC+1)·W/2`,
    ///      `e.y = (1-NDC)·H/2`, *no* extra `-0.5` offset — pixel centers sit
    ///      at integer-plus-half).
    ///   4. Find where the edge crosses p's row centerline (for horizontal
    ///      neighbors) or column centerline (for vertical neighbors).
    ///   5. If the crossing `xe` falls inside p's own pixel range `[pw, pw+1]`,
    ///      the edge cuts p's interior. The smaller of the two split portions
    ///      is the "wrong-coverage" fraction:
    ///        β = 0.5 - |xe - (pw + 0.5)|   (clamped to [0, 0.5])
    ///      Blend p's color toward the neighbor's by β.
    ///
    /// This differs from a naive "always-blend-the-empty-pixel" scheme: the
    /// pixel that gets blended is the one whose *interior* is cut by the
    /// silhouette edge, which may be the rast-fg pixel just as well as the
    /// rast-bg one — and only ever one of them per pair (the edge crosses at
    /// exactly one location, in exactly one of the pair's two pixels).
    ///
    /// Stashes per-direction state used downstream by `bwdPos`; forward and
    /// `bwdColor` ignore the extras (Metal DCEs them).
    private static let silhouetteBlock = """
        int tri_p = (int)rast[rast_base + 3] - 1;
        uint pos_n_base = pn * (uint)V * 4u;

        float alpha_arr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint  nbase_arr[4] = {0u, 0u, 0u, 0u};
        bool  active_arr[4] = {false, false, false, false};
        float alpha_sum_local = 0.0f;

        // Per-direction state for the pos backward (chain rule through xe).
        int   i_e0_arr[4] = {-1, -1, -1, -1};
        int   i_e1_arr[4] = {-1, -1, -1, -1};
        float e0x_arr[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
        float e0y_arr[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
        float e1x_arr[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
        float e1y_arr[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
        float t_arr[4]    = {0.0f, 0.0f, 0.0f, 0.0f};
        float denom_arr[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // dy (horizontal) or dx (vertical)
        float dalpha_dxy_arr[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // d_β / d_xe (or d_ye), ±1
        bool  is_horiz_arr[4] = {false, false, false, false};

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

            // Identify the foreground triangle of the pair + the silhouette edge.
            int fg_tri, target;
            if (tri_p >= 0 && tri_n >= 0) {
                // Both covered: silhouette only if they share an edge.
                int k_at_p = -1;
                if (topology[tri_p * 3 + 0] == tri_n) k_at_p = 0;
                else if (topology[tri_p * 3 + 1] == tri_n) k_at_p = 1;
                else if (topology[tri_p * 3 + 2] == tri_n) k_at_p = 2;
                if (k_at_p < 0) continue;
                // Use whichever triangle p belongs to as fg; the silhouette
                // edge is shared, so the edge endpoints are identical either
                // way. tri_p is fg here.
                fg_tri = tri_p; target = tri_n;
            } else if (tri_p >= 0) {
                fg_tri = tri_p; target = -1;
            } else {
                fg_tri = tri_n; target = -1;
            }

            // Edge selection — among candidate edges (those whose topology
            // entry matches `target`), pick the one whose line geometrically
            // separates p's center from the neighbor's center. For the
            // both-covered case there's only one candidate (the shared edge);
            // for boundary silhouettes there can be 2–3 boundary edges per
            // triangle and only one of them is the actual silhouette in this
            // pair's pixel pair.
            int i_e0 = -1, i_e1 = -1;
            PixelXY e0, e1;
            {
                float pcx = (float)pw + 0.5f;
                float pcy = (float)ph + 0.5f;
                float ncx = (float)nw + 0.5f;
                float ncy = (float)nh + 0.5f;
                int k_sel = -1;
                for (int kk = 0; kk < 3; ++kk) {
                    if (topology[fg_tri * 3 + kk] != target) continue;
                    int ie0_cand = tri[fg_tri * 3 + ((kk + 1) % 3)];
                    int ie1_cand = tri[fg_tri * 3 + ((kk + 2) % 3)];
                    float wc0 = pos[pos_n_base + (uint)ie0_cand * 4u + 3];
                    float wc1 = pos[pos_n_base + (uint)ie1_cand * 4u + 3];
                    PixelXY E0, E1;
                    E0.x = (pos[pos_n_base + (uint)ie0_cand * 4u + 0] / wc0 + 1.0f) * (float)W * 0.5f;
                    E0.y = (1.0f - pos[pos_n_base + (uint)ie0_cand * 4u + 1] / wc0) * (float)H * 0.5f;
                    E1.x = (pos[pos_n_base + (uint)ie1_cand * 4u + 0] / wc1 + 1.0f) * (float)W * 0.5f;
                    E1.y = (1.0f - pos[pos_n_base + (uint)ie1_cand * 4u + 1] / wc1) * (float)H * 0.5f;
                    float ex = E1.x - E0.x;
                    float ey = E1.y - E0.y;
                    // Signed cross products: which side of edge each center is on.
                    float side_p = ex * (pcy - E0.y) - ey * (pcx - E0.x);
                    float side_n = ex * (ncy - E0.y) - ey * (ncx - E0.x);
                    if (side_p * side_n < 0.0f) {
                        k_sel = kk;
                        i_e0 = ie0_cand; i_e1 = ie1_cand;
                        e0 = E0; e1 = E1;
                        break;
                    }
                }
                if (k_sel < 0) continue;
            }

            float beta = 0.0f;
            float t_val = 0.0f;
            float denom = 0.0f;
            float dbeta_dxy = 0.0f;
            bool is_horiz = (dh == 0);
            float p_center_x = (float)pw + 0.5f;
            float p_center_y = (float)ph + 0.5f;

            if (is_horiz) {
                float dyv = e1.y - e0.y;
                denom = dyv;
                if (fabs(dyv) >= 1e-7f) {
                    float row_y = p_center_y;
                    float tval = (row_y - e0.y) / dyv;
                    t_val = tval;
                    if (tval >= 0.0f && tval <= 1.0f) {
                        float xe = e0.x + tval * (e1.x - e0.x);
                        // Check if xe is interior to p's column [pw, pw+1].
                        if (xe > (float)pw && xe < (float)(pw + 1)) {
                            // β = smaller of the two cuts = 0.5 - |xe - p_center|.
                            float d = xe - p_center_x;
                            beta = 0.5f - fabs(d);
                            // d_β/d_xe = -sign(d). Pick branch by sign.
                            dbeta_dxy = (d > 0.0f) ? -1.0f : 1.0f;
                        }
                    }
                }
            } else {
                float dxv = e1.x - e0.x;
                denom = dxv;
                if (fabs(dxv) >= 1e-7f) {
                    float col_x = p_center_x;
                    float tval = (col_x - e0.x) / dxv;
                    t_val = tval;
                    if (tval >= 0.0f && tval <= 1.0f) {
                        float ye = e0.y + tval * (e1.y - e0.y);
                        if (ye > (float)ph && ye < (float)(ph + 1)) {
                            float d = ye - p_center_y;
                            beta = 0.5f - fabs(d);
                            dbeta_dxy = (d > 0.0f) ? -1.0f : 1.0f;
                        }
                    }
                }
            }
            if (beta <= 0.0f) continue;

            alpha_arr[dir] = beta;
            nbase_arr[dir] = ((pn * (uint)H + (uint)nh) * (uint)W + (uint)nw) * (uint)C;
            active_arr[dir] = true;
            alpha_sum_local += beta;

            i_e0_arr[dir] = i_e0;
            i_e1_arr[dir] = i_e1;
            e0x_arr[dir] = e0.x; e0y_arr[dir] = e0.y;
            e1x_arr[dir] = e1.x; e1y_arr[dir] = e1.y;
            t_arr[dir] = t_val;
            denom_arr[dir] = denom;
            dalpha_dxy_arr[dir] = dbeta_dxy;
            is_horiz_arr[dir] = is_horiz;
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
    /// Backward for `d_pos`. Active when α ∈ (0, 1) (interior, non-saturated);
    /// outside that range `α` is clipped by min/max and contributes no
    /// gradient. Chain rule:
    ///
    ///   d_α/d_e0.x = dalpha_dxy · (1 - t)      (horizontal)
    ///   d_α/d_e1.x = dalpha_dxy · t
    ///   d_α/d_e0.y = dalpha_dxy · (e1.x - e0.x) · (row_y - e1.y) / dy²
    ///   d_α/d_e1.y = dalpha_dxy · -(e1.x - e0.x) · (row_y - e0.y) / dy²
    ///
    /// (Vertical pairs swap x ↔ y and `row_y` ↔ `col_x`.)
    ///
    /// Then chain through perspective divide for each endpoint vertex v:
    ///   d_e.x/d_pos[v, 0] =  W/2 / w_v
    ///   d_e.x/d_pos[v, 3] = -W/2 · nd_x / w_v
    ///   d_e.y/d_pos[v, 1] = -H/2 / w_v
    ///   d_e.y/d_pos[v, 3] =  H/2 · nd_y / w_v
    ///   d_e.{x,y}/d_pos[v, 2] = 0   (z doesn't affect screen position)
    ///
    /// `d_loss/d_pos` = `d_loss/d_α` · `d_α/d_pos`, with
    ///   `d_loss/d_α = Σ_c d_out_p[c] · (color_n[c] - color_p[c])`.
    private static var bwdPosSource: String { """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint pw = pix % (uint)W;
        uint ph = (pix / (uint)W) % (uint)H;
        uint pn = pix / ((uint)H * (uint)W);

        uint rast_base = ((pn * (uint)H + ph) * (uint)W + pw) * 4u;
        uint color_base = ((pn * (uint)H + ph) * (uint)W + pw) * (uint)C;

        \(silhouetteBlock)

        float halfW = (float)W * 0.5f;
        float halfH = (float)H * 0.5f;

        for (int dir = 0; dir < 4; ++dir) {
            if (!active_arr[dir]) continue;
            float alpha = alpha_arr[dir];
            // Skip saturated (clipped) α — d_α is zero through the clamp.
            if (alpha <= 0.0f || alpha >= 1.0f) continue;

            // d_loss/d_α from upstream cotangents.
            float d_alpha = 0.0f;
            uint nb = nbase_arr[dir];
            for (int c = 0; c < C; ++c) {
                float cp = color[color_base + (uint)c];
                float cn = color[nb + (uint)c];
                float g  = d_out[color_base + (uint)c];
                d_alpha += g * (cn - cp);
            }
            if (d_alpha == 0.0f) continue;

            float e0x = e0x_arr[dir], e0y = e0y_arr[dir];
            float e1x = e1x_arr[dir], e1y = e1y_arr[dir];
            float t   = t_arr[dir];
            float denom = denom_arr[dir];
            float dalpha_dxy = dalpha_dxy_arr[dir];
            bool  is_horiz = is_horiz_arr[dir];

            float dade0x, dade0y, dade1x, dade1y;
            if (is_horiz) {
                float row_y = (float)ph + 0.5f;
                float ex_diff = e1x - e0x;
                float inv_denom_sq = 1.0f / (denom * denom);
                dade0x = dalpha_dxy * (1.0f - t);
                dade1x = dalpha_dxy * t;
                dade0y = dalpha_dxy * ex_diff * (row_y - e1y) * inv_denom_sq;
                dade1y = dalpha_dxy * (-ex_diff) * (row_y - e0y) * inv_denom_sq;
            } else {
                float col_x = (float)pw + 0.5f;
                float ey_diff = e1y - e0y;
                float inv_denom_sq = 1.0f / (denom * denom);
                dade0y = dalpha_dxy * (1.0f - t);
                dade1y = dalpha_dxy * t;
                dade0x = dalpha_dxy * ey_diff * (col_x - e1x) * inv_denom_sq;
                dade1x = dalpha_dxy * (-ey_diff) * (col_x - e0x) * inv_denom_sq;
            }

            // Chain through perspective divide for each endpoint and scatter
            // into d_pos. Vertex 0:
            int iv0 = i_e0_arr[dir];
            {
                float wv = pos[pos_n_base + (uint)iv0 * 4u + 3];
                float inv_w = 1.0f / wv;
                float nd_x = pos[pos_n_base + (uint)iv0 * 4u + 0] * inv_w;
                float nd_y = pos[pos_n_base + (uint)iv0 * 4u + 1] * inv_w;

                float d_px = d_alpha * dade0x * halfW * inv_w;
                float d_py = d_alpha * dade0y * (-halfH) * inv_w;
                float d_pw = d_alpha *
                    (dade0x * (-halfW * nd_x) + dade0y * (halfH * nd_y)) * inv_w;

                atomic_fetch_add_explicit(
                    &d_pos[pos_n_base + (uint)iv0 * 4u + 0], d_px,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &d_pos[pos_n_base + (uint)iv0 * 4u + 1], d_py,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &d_pos[pos_n_base + (uint)iv0 * 4u + 3], d_pw,
                    memory_order_relaxed);
            }
            // Vertex 1:
            int iv1 = i_e1_arr[dir];
            {
                float wv = pos[pos_n_base + (uint)iv1 * 4u + 3];
                float inv_w = 1.0f / wv;
                float nd_x = pos[pos_n_base + (uint)iv1 * 4u + 0] * inv_w;
                float nd_y = pos[pos_n_base + (uint)iv1 * 4u + 1] * inv_w;

                float d_px = d_alpha * dade1x * halfW * inv_w;
                float d_py = d_alpha * dade1y * (-halfH) * inv_w;
                float d_pw = d_alpha *
                    (dade1x * (-halfW * nd_x) + dade1y * (halfH * nd_y)) * inv_w;

                atomic_fetch_add_explicit(
                    &d_pos[pos_n_base + (uint)iv1 * 4u + 0], d_px,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &d_pos[pos_n_base + (uint)iv1 * 4u + 1], d_py,
                    memory_order_relaxed);
                atomic_fetch_add_explicit(
                    &d_pos[pos_n_base + (uint)iv1 * 4u + 3], d_pw,
                    memory_order_relaxed);
            }
        }
    """ }

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

    private static let bwdPosKernel = MLXFast.metalKernel(
        name: "diffrast_antialias_bwd_pos",
        inputNames: ["color", "rast", "pos", "tri", "topology", "d_out"],
        outputNames: ["d_pos"],
        source: bwdPosSource, header: aaHeader,
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

    private static func antialiasBackwardPosKernel(
        color: MLXArray, rast: MLXArray, pos: MLXArray, tri: MLXArray, topology: MLXArray,
        dOut: MLXArray,
        N: Int, H: Int, W: Int, C: Int, T: Int, V: Int
    ) -> MLXArray {
        let pixels = N * H * W
        let rounded = ((pixels + aaTG - 1) / aaTG) * aaTG
        return bwdPosKernel(
            [color, rast, pos, tri, topology, dOut],
            template: aaTmpl(N: N, H: H, W: W, C: C, T: T, V: V),
            grid: (rounded, 1, 1), threadGroup: (aaTG, 1, 1),
            outputShapes: [[N, V, 4]], outputDTypes: [.float32],
            initValue: 0.0
        )[0]
    }
}
