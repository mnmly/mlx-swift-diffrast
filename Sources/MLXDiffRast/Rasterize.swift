import Foundation
import MLX
import MLXFast

extension DiffRast {

    /// Rasterize a triangle mesh in clip space (M2.1 forward + M2.2 pixel derivatives).
    ///
    /// This is a pure-compute software rasterizer: each pixel thread iterates over
    /// every triangle and keeps the front-most one. Suitable for the moderate
    /// triangle counts typical of inverse rendering (≲ few thousand triangles).
    /// We can swap to a hardware `MTLRenderPipeline` backend later without changing
    /// the public API.
    ///
    /// Conventions (match nvdiffrast):
    ///   - `pos`:    `[N, V, 4]` in clip space — channels are `(x*w, y*w, z*w, w)`.
    ///               Range mode (`[V, 4]` + `ranges`) is deferred.
    ///   - `tri`:    `[T, 3]` int32 vertex indices.
    ///   - Output `rast`: `[N, H, W, 4]` with channels `(u, v, z/w, tri_id+1)`.
    ///     - `u, v` are linear barycentrics (vertex 0 weight = `1-u-v`).
    ///     - `tri_id+1 == 0` marks empty pixels.
    ///   - Pixel `(h=0, w=0)` is top-left; NDC `+y` is up (vertical flip applied).
    ///
    /// `rastDB` (returned when `gradDB == true`, default) has shape `[N, H, W, 4]`
    /// with channels `(du/dx, du/dy, dv/dx, dv/dy)` in per-pixel units. Empty
    /// `[N, H, W, 0]` tensor when disabled. Feeds the `diffAttrs` branch of
    /// `interpolate` and the mipmap-LOD selection in `texture`.
    ///
    /// **Differentiability (M2.1 limitation):** the VJP currently returns zeros.
    /// Calling `grad` w.r.t. `pos` will silently get zero gradients until M2.3
    /// implements the analytic backward.
    @discardableResult
    public static func rasterize(
        _ pos: MLXArray,
        tri: MLXArray,
        resolution: (height: Int, width: Int),
        gradDB: Bool = true
    ) -> (rast: MLXArray, rastDB: MLXArray) {
        precondition(pos.ndim == 3 && pos.shape[2] == 4,
                     "rasterize: pos must be [N, V, 4] (got \(pos.shape))")
        precondition(tri.ndim == 2 && tri.shape[1] == 3,
                     "rasterize: tri must be [T, 3] (got \(tri.shape))")

        let N = pos.shape[0], V = pos.shape[1]
        let H = resolution.height, W = resolution.width
        let T = tri.shape[0]
        let triI32 = tri.dtype == .int32 ? tri : tri.asType(.int32)

        if !gradDB {
            let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
                [Self.rasterizeForwardKernel(pos: inputs[0], tri: triI32,
                                             N: N, V: V, T: T, H: H, W: W)]
            }
            let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, _ in
                [MLXArray.zeros(primals[0].shape, dtype: .float32)]
            }
            let custom = CustomFunction { Forward(fwd); VJP(vjp) }
            let rast = custom([pos])[0]
            return (rast, MLXArray.zeros([N, H, W, 0]))
        } else {
            let fwd: ([MLXArray]) -> [MLXArray] = { inputs in
                let (r, db) = Self.rasterizeForwardDBKernel(pos: inputs[0], tri: triI32,
                                                           N: N, V: V, T: T, H: H, W: W)
                return [r, db]
            }
            let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, _ in
                [MLXArray.zeros(primals[0].shape, dtype: .float32)]
            }
            let custom = CustomFunction { Forward(fwd); VJP(vjp) }
            let outs = custom([pos])
            return (outs[0], outs[1])
        }
    }

    // MARK: - Forward kernel

    /// Per-pixel rasterization. One thread per pixel; iterate over every triangle.
    ///
    /// Algorithm:
    ///   1. Build pixel-center NDC coords `(ndc_x, ndc_y)` from `(w, h)`.
    ///   2. For each triangle:
    ///      a. Perspective-divide vertex clip positions → screen-space `(sx, sy)`
    ///         and `z/w` (one z per vertex).
    ///      b. Edge functions in screen space. If the pixel is inside the triangle
    ///         (all three edges have the same sign as the signed area), compute
    ///         barycentric weights.
    ///      c. Linear z at the pixel = `bw*z0 + u*z1 + v*z2`.
    ///      d. Depth-test against the current best for this pixel.
    ///   3. Write `(u, v, z, tri+1)` for the winner; `(0,0,0,0)` if no hit.
    ///
    /// Notes:
    ///   - Triangles with non-positive `w` at any vertex are skipped (behind the eye).
    ///   - Visible z range is `[-1, 1]` (NDC); anything outside is rejected.
    ///   - Front-to-back is `z` ascending (smaller z/w = closer).
    private static let fwdSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint pw = pix % (uint)W;
        uint ph = (pix / (uint)W) % (uint)H;
        uint pn = pix / ((uint)H * (uint)W);

        // Pixel-center NDC. h=0 maps to ndc_y = +1 (top), h=H-1 to -1.
        float ndc_x = 2.0f * ((float)pw + 0.5f) / (float)W - 1.0f;
        float ndc_y = 1.0f - 2.0f * ((float)ph + 0.5f) / (float)H;

        float best_z = 1e30f;
        int   best_t = -1;
        float best_u = 0.0f;
        float best_v = 0.0f;

        uint pos_n_base = pn * (uint)V * 4u;

        for (int t = 0; t < T; ++t) {
            int i0 = tri[t * 3 + 0];
            int i1 = tri[t * 3 + 1];
            int i2 = tri[t * 3 + 2];

            float x0 = pos[pos_n_base + (uint)i0 * 4u + 0];
            float y0 = pos[pos_n_base + (uint)i0 * 4u + 1];
            float z0 = pos[pos_n_base + (uint)i0 * 4u + 2];
            float w0 = pos[pos_n_base + (uint)i0 * 4u + 3];

            float x1 = pos[pos_n_base + (uint)i1 * 4u + 0];
            float y1 = pos[pos_n_base + (uint)i1 * 4u + 1];
            float z1 = pos[pos_n_base + (uint)i1 * 4u + 2];
            float w1 = pos[pos_n_base + (uint)i1 * 4u + 3];

            float x2 = pos[pos_n_base + (uint)i2 * 4u + 0];
            float y2 = pos[pos_n_base + (uint)i2 * 4u + 1];
            float z2 = pos[pos_n_base + (uint)i2 * 4u + 2];
            float w2 = pos[pos_n_base + (uint)i2 * 4u + 3];

            // Reject if any vertex is behind the eye (w <= 0).
            if (w0 <= 0.0f || w1 <= 0.0f || w2 <= 0.0f) continue;

            // Perspective divide → NDC screen-space xy and z.
            float sx0 = x0 / w0, sy0 = y0 / w0, sz0 = z0 / w0;
            float sx1 = x1 / w1, sy1 = y1 / w1, sz1 = z1 / w1;
            float sx2 = x2 / w2, sy2 = y2 / w2, sz2 = z2 / w2;

            // Signed area (twice the area, sign indicates winding).
            float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
            if (area == 0.0f) continue;
            float inv_area = 1.0f / area;

            // Edge functions at the pixel. Each e_i has sign matching area iff the
            // pixel is on the inside of the edge opposite vertex i.
            //   e0 = signed area of (p, s1, s2) → weight at vertex 0 (bw)
            //   e1 = signed area of (s0, p, s2) → weight at vertex 1 (u)
            //   e2 = signed area of (s0, s1, p) → weight at vertex 2 (v)
            float e0 = (sx1 - ndc_x) * (sy2 - ndc_y) - (sx2 - ndc_x) * (sy1 - ndc_y);
            float e1 = (sx2 - ndc_x) * (sy0 - ndc_y) - (sx0 - ndc_x) * (sy2 - ndc_y);
            float e2 = (sx0 - ndc_x) * (sy1 - ndc_y) - (sx1 - ndc_x) * (sy0 - ndc_y);

            // Inside if all edges share the sign of `area`.
            if ((e0 * area) < 0.0f || (e1 * area) < 0.0f || (e2 * area) < 0.0f) continue;

            float bw = e0 * inv_area;
            float u  = e1 * inv_area;
            float v  = e2 * inv_area;

            // Screen-space linear z (matches nvdiffrast's stored z/w channel).
            float z_pix = bw * sz0 + u * sz1 + v * sz2;
            if (z_pix < -1.0f || z_pix > 1.0f) continue;
            if (z_pix >= best_z) continue;

            best_z = z_pix;
            best_t = t;
            best_u = u;
            best_v = v;
        }

        uint out_base = ((pn * (uint)H + ph) * (uint)W + pw) * 4u;
        if (best_t < 0) {
            out[out_base + 0] = 0.0f;
            out[out_base + 1] = 0.0f;
            out[out_base + 2] = 0.0f;
            out[out_base + 3] = 0.0f;
        } else {
            out[out_base + 0] = best_u;
            out[out_base + 1] = best_v;
            out[out_base + 2] = best_z;
            out[out_base + 3] = (float)(best_t + 1);
        }
    """

    /// Forward + pixel-derivatives. After the per-triangle loop selects the winner,
    /// recompute its screen-space vertices once and emit `(du/dx, du/dy, dv/dx, dv/dy)`
    /// in per-pixel units. Chain rule:
    ///   d_ndc_x/dx =  2/W   ;   d_ndc_y/dy = -2/H
    ///   du/d_ndc_x = (sy2 - sy0) / area
    ///   du/d_ndc_y = (sx0 - sx2) / area
    ///   dv/d_ndc_x = (sy0 - sy1) / area
    ///   dv/d_ndc_y = (sx1 - sx0) / area
    private static let fwdDBSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint pw = pix % (uint)W;
        uint ph = (pix / (uint)W) % (uint)H;
        uint pn = pix / ((uint)H * (uint)W);

        float ndc_x = 2.0f * ((float)pw + 0.5f) / (float)W - 1.0f;
        float ndc_y = 1.0f - 2.0f * ((float)ph + 0.5f) / (float)H;

        float best_z = 1e30f;
        int   best_t = -1;
        float best_u = 0.0f;
        float best_v = 0.0f;

        uint pos_n_base = pn * (uint)V * 4u;

        for (int t = 0; t < T; ++t) {
            int i0 = tri[t * 3 + 0];
            int i1 = tri[t * 3 + 1];
            int i2 = tri[t * 3 + 2];

            float x0 = pos[pos_n_base + (uint)i0 * 4u + 0];
            float y0 = pos[pos_n_base + (uint)i0 * 4u + 1];
            float z0 = pos[pos_n_base + (uint)i0 * 4u + 2];
            float w0 = pos[pos_n_base + (uint)i0 * 4u + 3];
            float x1 = pos[pos_n_base + (uint)i1 * 4u + 0];
            float y1 = pos[pos_n_base + (uint)i1 * 4u + 1];
            float z1 = pos[pos_n_base + (uint)i1 * 4u + 2];
            float w1 = pos[pos_n_base + (uint)i1 * 4u + 3];
            float x2 = pos[pos_n_base + (uint)i2 * 4u + 0];
            float y2 = pos[pos_n_base + (uint)i2 * 4u + 1];
            float z2 = pos[pos_n_base + (uint)i2 * 4u + 2];
            float w2 = pos[pos_n_base + (uint)i2 * 4u + 3];

            if (w0 <= 0.0f || w1 <= 0.0f || w2 <= 0.0f) continue;

            float sx0 = x0 / w0, sy0 = y0 / w0, sz0 = z0 / w0;
            float sx1 = x1 / w1, sy1 = y1 / w1, sz1 = z1 / w1;
            float sx2 = x2 / w2, sy2 = y2 / w2, sz2 = z2 / w2;

            float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
            if (area == 0.0f) continue;
            float inv_area = 1.0f / area;

            float e0 = (sx1 - ndc_x) * (sy2 - ndc_y) - (sx2 - ndc_x) * (sy1 - ndc_y);
            float e1 = (sx2 - ndc_x) * (sy0 - ndc_y) - (sx0 - ndc_x) * (sy2 - ndc_y);
            float e2 = (sx0 - ndc_x) * (sy1 - ndc_y) - (sx1 - ndc_x) * (sy0 - ndc_y);
            if ((e0 * area) < 0.0f || (e1 * area) < 0.0f || (e2 * area) < 0.0f) continue;

            float bw = e0 * inv_area;
            float u  = e1 * inv_area;
            float v  = e2 * inv_area;

            float z_pix = bw * sz0 + u * sz1 + v * sz2;
            if (z_pix < -1.0f || z_pix > 1.0f) continue;
            if (z_pix >= best_z) continue;

            best_z = z_pix;
            best_t = t;
            best_u = u;
            best_v = v;
        }

        uint out_base = ((pn * (uint)H + ph) * (uint)W + pw) * 4u;
        uint db_base  = out_base;   // same stride: [N, H, W, 4]

        if (best_t < 0) {
            out[out_base + 0] = 0.0f;
            out[out_base + 1] = 0.0f;
            out[out_base + 2] = 0.0f;
            out[out_base + 3] = 0.0f;
            out_db[db_base + 0] = 0.0f;
            out_db[db_base + 1] = 0.0f;
            out_db[db_base + 2] = 0.0f;
            out_db[db_base + 3] = 0.0f;
            return;
        }

        out[out_base + 0] = best_u;
        out[out_base + 1] = best_v;
        out[out_base + 2] = best_z;
        out[out_base + 3] = (float)(best_t + 1);

        // Recompute screen-space vertices for the winning triangle to derive rast_db.
        int i0 = tri[best_t * 3 + 0];
        int i1 = tri[best_t * 3 + 1];
        int i2 = tri[best_t * 3 + 2];
        float w0 = pos[pos_n_base + (uint)i0 * 4u + 3];
        float w1 = pos[pos_n_base + (uint)i1 * 4u + 3];
        float w2 = pos[pos_n_base + (uint)i2 * 4u + 3];
        float sx0 = pos[pos_n_base + (uint)i0 * 4u + 0] / w0;
        float sy0 = pos[pos_n_base + (uint)i0 * 4u + 1] / w0;
        float sx1 = pos[pos_n_base + (uint)i1 * 4u + 0] / w1;
        float sy1 = pos[pos_n_base + (uint)i1 * 4u + 1] / w1;
        float sx2 = pos[pos_n_base + (uint)i2 * 4u + 0] / w2;
        float sy2 = pos[pos_n_base + (uint)i2 * 4u + 1] / w2;
        float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
        float inv_area = 1.0f / area;
        float k_x = 2.0f / (float)W;
        float k_y = -2.0f / (float)H;

        out_db[db_base + 0] = (sy2 - sy0) * inv_area * k_x;   // du/dx
        out_db[db_base + 1] = (sx0 - sx2) * inv_area * k_y;   // du/dy
        out_db[db_base + 2] = (sy0 - sy1) * inv_area * k_x;   // dv/dx
        out_db[db_base + 3] = (sx1 - sx0) * inv_area * k_y;   // dv/dy
    """

    private static let fwdKernel = MLXFast.metalKernel(
        name: "diffrast_rasterize_fwd",
        inputNames: ["pos", "tri"], outputNames: ["out"],
        source: fwdSource)

    private static let fwdDBKernel = MLXFast.metalKernel(
        name: "diffrast_rasterize_fwd_db",
        inputNames: ["pos", "tri"], outputNames: ["out", "out_db"],
        source: fwdDBSource)

    private static let rasterizeTG = 256

    private static func rasterizeForwardKernel(
        pos: MLXArray, tri: MLXArray,
        N: Int, V: Int, T: Int, H: Int, W: Int
    ) -> MLXArray {
        let pixels = N * H * W
        let rounded = ((pixels + rasterizeTG - 1) / rasterizeTG) * rasterizeTG
        return fwdKernel(
            [pos, tri],
            template: [("N", N), ("V", V), ("T", T), ("H", H), ("W", W)],
            grid: (rounded, 1, 1),
            threadGroup: (rasterizeTG, 1, 1),
            outputShapes: [[N, H, W, 4]],
            outputDTypes: [.float32]
        )[0]
    }

    private static func rasterizeForwardDBKernel(
        pos: MLXArray, tri: MLXArray,
        N: Int, V: Int, T: Int, H: Int, W: Int
    ) -> (MLXArray, MLXArray) {
        let pixels = N * H * W
        let rounded = ((pixels + rasterizeTG - 1) / rasterizeTG) * rasterizeTG
        let outs = fwdDBKernel(
            [pos, tri],
            template: [("N", N), ("V", V), ("T", T), ("H", H), ("W", W)],
            grid: (rounded, 1, 1),
            threadGroup: (rasterizeTG, 1, 1),
            outputShapes: [[N, H, W, 4], [N, H, W, 4]],
            outputDTypes: [.float32, .float32]
        )
        return (outs[0], outs[1])
    }
}
