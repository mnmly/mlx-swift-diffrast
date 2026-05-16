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
            let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
                let dPos = Self.rasterizeBackwardKernel(
                    pos: primals[0], tri: triI32,
                    rast: forwardRastForBwd(primals[0], triI32, N, V, T, H, W),
                    dRast: cotangents[0],
                    dRastDB: MLXArray.zeros([N, H, W, 4], dtype: .float32),
                    N: N, V: V, H: H, W: W)
                return [dPos]
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
            let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
                let (r, _) = Self.rasterizeForwardDBKernel(
                    pos: primals[0], tri: triI32, N: N, V: V, T: T, H: H, W: W)
                let dPos = Self.rasterizeBackwardKernel(
                    pos: primals[0], tri: triI32, rast: r,
                    dRast: cotangents[0], dRastDB: cotangents[1],
                    N: N, V: V, H: H, W: W)
                return [dPos]
            }
            let custom = CustomFunction { Forward(fwd); VJP(vjp) }
            let outs = custom([pos])
            return (outs[0], outs[1])
        }
    }

    /// VJP helper: re-run forward to recover `rast` (we need it to identify the
    /// winning triangle per pixel). MLX doesn't cache the primal outputs, so we
    /// recompute. Cheap relative to the backward itself.
    private static func forwardRastForBwd(
        _ pos: MLXArray, _ tri: MLXArray,
        _ N: Int, _ V: Int, _ T: Int, _ H: Int, _ W: Int
    ) -> MLXArray {
        rasterizeForwardKernel(pos: pos, tri: tri, N: N, V: V, T: T, H: H, W: W)
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

    // MARK: - Backward kernel (M2.3)

    /// Backward for rasterize. Chains cotangents `d_rast[u,v,z]` and
    /// `d_rast_db[du/dx, du/dy, dv/dx, dv/dy]` back into `d_pos[n, vk, :]`
    /// through screen-space barycentrics and the perspective divide.
    ///
    /// Per-pixel logic (only the winning triangle contributes — empty pixels skip):
    ///   1. Read `tri_id = rast[..., 3] - 1`, look up vertex indices, recompute
    ///      screen-space `(sxk, syk, szk)` from clip-space `pos`.
    ///   2. Compute partial derivatives ∂u/∂s, ∂v/∂s, ∂area/∂s analytically.
    ///   3. Accumulate the screen-space cotangent for each of the 9 variables
    ///      `(sx0, sy0, sz0, ..., sx2, sy2, sz2)`.
    ///   4. Push through the perspective divide:
    ///        d_xk = d_sxk / wk
    ///        d_yk = d_syk / wk
    ///        d_zk = d_szk / wk
    ///        d_wk = -(sxk d_sxk + syk d_syk + szk d_szk) / wk
    ///   5. Atomic-add into `d_pos` since neighboring pixels share vertices.
    ///
    /// rast_db cotangents:
    ///   rast_db_j = N_j / area where N_j is a simple linear combo of (sxk, syk).
    ///   d(rast_db_j)/∂p = (∂N_j/∂p - rast_db_j · ∂area/∂p) / area
    ///
    /// We recompute `rast_db_j` from `pos` rather than passing it as an input
    /// (saves a tensor + lets a single kernel serve both gradDB modes — the
    /// gradDB=false path just passes zeros for `d_rast_db`).
    private static let bwdSource = """
        uint pix = thread_position_in_grid.x;
        uint total = (uint)N * (uint)H * (uint)W;
        if (pix >= total) return;

        uint pw = pix % (uint)W;
        uint ph = (pix / (uint)W) % (uint)H;
        uint pn = pix / ((uint)H * (uint)W);

        uint rast_base = ((pn * (uint)H + ph) * (uint)W + pw) * 4u;
        float tri_id_f = rast[rast_base + 3];
        int t = (int)tri_id_f - 1;
        if (t < 0) return;

        float u_val = rast[rast_base + 0];
        float v_val = rast[rast_base + 1];
        float bw_val = 1.0f - u_val - v_val;

        float d_u  = d_rast[rast_base + 0];
        float d_v  = d_rast[rast_base + 1];
        float d_z  = d_rast[rast_base + 2];
        // d_rast[3] is tri_id+1 — non-differentiable, ignored.

        float d_db0 = d_rast_db[rast_base + 0];
        float d_db1 = d_rast_db[rast_base + 1];
        float d_db2 = d_rast_db[rast_base + 2];
        float d_db3 = d_rast_db[rast_base + 3];

        float Px = 2.0f * ((float)pw + 0.5f) / (float)W - 1.0f;
        float Py = 1.0f - 2.0f * ((float)ph + 0.5f) / (float)H;

        uint pos_n_base = pn * (uint)V * 4u;
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

        float sx0 = x0 / w0, sy0 = y0 / w0, sz0 = z0 / w0;
        float sx1 = x1 / w1, sy1 = y1 / w1, sz1 = z1 / w1;
        float sx2 = x2 / w2, sy2 = y2 / w2, sz2 = z2 / w2;

        float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
        float inv_area = 1.0f / area;

        // ∂area/∂s
        float dadsx0 = sy1 - sy2;
        float dadsx1 = sy2 - sy0;
        float dadsx2 = sy0 - sy1;
        float dadsy0 = sx2 - sx1;
        float dadsy1 = sx0 - sx2;
        float dadsy2 = sx1 - sx0;

        // ∂e1/∂s   (e1 = (sx2-Px)(sy0-Py) - (sx0-Px)(sy2-Py))
        float de1dsx0 = Py - sy2;
        float de1dsx2 = sy0 - Py;
        float de1dsy0 = sx2 - Px;
        float de1dsy2 = Px - sx0;
        // de1/dsx1 = de1/dsy1 = 0

        // ∂e2/∂s   (e2 = (sx0-Px)(sy1-Py) - (sx1-Px)(sy0-Py))
        float de2dsx0 = sy1 - Py;
        float de2dsx1 = Py - sy0;
        float de2dsy0 = Px - sx1;
        float de2dsy1 = sx0 - Px;
        // de2/dsx2 = de2/dsy2 = 0

        // ∂u/∂p = (∂e1/∂p − u·∂area/∂p) / area    ;  ∂v/∂p analogous
        float dudsx0 = (de1dsx0 - u_val * dadsx0) * inv_area;
        float dudsx1 = (0.0f    - u_val * dadsx1) * inv_area;
        float dudsx2 = (de1dsx2 - u_val * dadsx2) * inv_area;
        float dudsy0 = (de1dsy0 - u_val * dadsy0) * inv_area;
        float dudsy1 = (0.0f    - u_val * dadsy1) * inv_area;
        float dudsy2 = (de1dsy2 - u_val * dadsy2) * inv_area;

        float dvdsx0 = (de2dsx0 - v_val * dadsx0) * inv_area;
        float dvdsx1 = (de2dsx1 - v_val * dadsx1) * inv_area;
        float dvdsx2 = (0.0f    - v_val * dadsx2) * inv_area;
        float dvdsy0 = (de2dsy0 - v_val * dadsy0) * inv_area;
        float dvdsy1 = (de2dsy1 - v_val * dadsy1) * inv_area;
        float dvdsy2 = (0.0f    - v_val * dadsy2) * inv_area;

        // z_pix = sz0 + u·(sz1−sz0) + v·(sz2−sz0)   ⇒
        //   ∂z/∂sxk = ∂u/∂sxk · (sz1−sz0) + ∂v/∂sxk · (sz2−sz0)
        //   ∂z/∂szk = (bw, u, v)
        float dsz10 = sz1 - sz0;
        float dsz20 = sz2 - sz0;

        // Accumulate screen-space cotangents.
        float dsx0 = d_u * dudsx0 + d_v * dvdsx0 + d_z * (dudsx0 * dsz10 + dvdsx0 * dsz20);
        float dsx1 = d_u * dudsx1 + d_v * dvdsx1 + d_z * (dudsx1 * dsz10 + dvdsx1 * dsz20);
        float dsx2 = d_u * dudsx2 + d_v * dvdsx2 + d_z * (dudsx2 * dsz10 + dvdsx2 * dsz20);
        float dsy0 = d_u * dudsy0 + d_v * dvdsy0 + d_z * (dudsy0 * dsz10 + dvdsy0 * dsz20);
        float dsy1 = d_u * dudsy1 + d_v * dvdsy1 + d_z * (dudsy1 * dsz10 + dvdsy1 * dsz20);
        float dsy2 = d_u * dudsy2 + d_v * dvdsy2 + d_z * (dudsy2 * dsz10 + dvdsy2 * dsz20);

        float dsz0_s = d_z * bw_val;
        float dsz1_s = d_z * u_val;
        float dsz2_s = d_z * v_val;

        // rast_db cotangents (zero in non-DB mode since d_rast_db is zeros there).
        // rast_db_j = N_j / area where N_j is a linear combo of sxk/syk:
        //   N_0 = (sy2 - sy0) * Kx     Kx =  2/W
        //   N_1 = (sx0 - sx2) * Ky     Ky = -2/H
        //   N_2 = (sy0 - sy1) * Kx
        //   N_3 = (sx1 - sx0) * Ky
        // ∂(rast_db_j)/∂p = (∂N_j/∂p − rast_db_j · ∂area/∂p) / area
        float Kx =  2.0f / (float)W;
        float Ky = -2.0f / (float)H;
        float rdb0 = (sy2 - sy0) * Kx * inv_area;
        float rdb1 = (sx0 - sx2) * Ky * inv_area;
        float rdb2 = (sy0 - sy1) * Kx * inv_area;
        float rdb3 = (sx1 - sx0) * Ky * inv_area;

        // Indirect (through area) — applies to every variable.
        dsx0 += (-(d_db0 * rdb0 + d_db1 * rdb1 + d_db2 * rdb2 + d_db3 * rdb3) * dadsx0) * inv_area;
        dsx1 += (-(d_db0 * rdb0 + d_db1 * rdb1 + d_db2 * rdb2 + d_db3 * rdb3) * dadsx1) * inv_area;
        dsx2 += (-(d_db0 * rdb0 + d_db1 * rdb1 + d_db2 * rdb2 + d_db3 * rdb3) * dadsx2) * inv_area;
        dsy0 += (-(d_db0 * rdb0 + d_db1 * rdb1 + d_db2 * rdb2 + d_db3 * rdb3) * dadsy0) * inv_area;
        dsy1 += (-(d_db0 * rdb0 + d_db1 * rdb1 + d_db2 * rdb2 + d_db3 * rdb3) * dadsy1) * inv_area;
        dsy2 += (-(d_db0 * rdb0 + d_db1 * rdb1 + d_db2 * rdb2 + d_db3 * rdb3) * dadsy2) * inv_area;

        // Direct ∂N_j/∂p contributions.
        dsy0 += d_db0 * (-Kx) * inv_area;   // ch 0 wrt sy0
        dsy2 += d_db0 * ( Kx) * inv_area;   // ch 0 wrt sy2
        dsx0 += d_db1 * ( Ky) * inv_area;   // ch 1 wrt sx0
        dsx2 += d_db1 * (-Ky) * inv_area;   // ch 1 wrt sx2
        dsy0 += d_db2 * ( Kx) * inv_area;   // ch 2 wrt sy0
        dsy1 += d_db2 * (-Kx) * inv_area;   // ch 2 wrt sy1
        dsx0 += d_db3 * (-Ky) * inv_area;   // ch 3 wrt sx0
        dsx1 += d_db3 * ( Ky) * inv_area;   // ch 3 wrt sx1

        // Perspective divide → clip-space cotangents.
        float inv_w0 = 1.0f / w0;
        float inv_w1 = 1.0f / w1;
        float inv_w2 = 1.0f / w2;

        float dx0 = dsx0 * inv_w0;
        float dy0 = dsy0 * inv_w0;
        float dz0c = dsz0_s * inv_w0;
        float dw0 = -(sx0 * dsx0 + sy0 * dsy0 + sz0 * dsz0_s) * inv_w0;

        float dx1 = dsx1 * inv_w1;
        float dy1 = dsy1 * inv_w1;
        float dz1c = dsz1_s * inv_w1;
        float dw1 = -(sx1 * dsx1 + sy1 * dsy1 + sz1 * dsz1_s) * inv_w1;

        float dx2 = dsx2 * inv_w2;
        float dy2 = dsy2 * inv_w2;
        float dz2c = dsz2_s * inv_w2;
        float dw2 = -(sx2 * dsx2 + sy2 * dsy2 + sz2 * dsz2_s) * inv_w2;

        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i0 * 4u + 0], dx0,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i0 * 4u + 1], dy0,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i0 * 4u + 2], dz0c, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i0 * 4u + 3], dw0,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i1 * 4u + 0], dx1,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i1 * 4u + 1], dy1,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i1 * 4u + 2], dz1c, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i1 * 4u + 3], dw1,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i2 * 4u + 0], dx2,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i2 * 4u + 1], dy2,  memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i2 * 4u + 2], dz2c, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_pos[pos_n_base + (uint)i2 * 4u + 3], dw2,  memory_order_relaxed);
    """

    private static let bwdKernel = MLXFast.metalKernel(
        name: "diffrast_rasterize_bwd",
        inputNames: ["pos", "tri", "rast", "d_rast", "d_rast_db"],
        outputNames: ["d_pos"],
        source: bwdSource,
        atomicOutputs: true)

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

    private static func rasterizeBackwardKernel(
        pos: MLXArray, tri: MLXArray, rast: MLXArray,
        dRast: MLXArray, dRastDB: MLXArray,
        N: Int, V: Int, H: Int, W: Int
    ) -> MLXArray {
        let pixels = N * H * W
        let rounded = ((pixels + rasterizeTG - 1) / rasterizeTG) * rasterizeTG
        return bwdKernel(
            [pos, tri, rast, dRast, dRastDB],
            template: [("N", N), ("V", V), ("H", H), ("W", W)],
            grid: (rounded, 1, 1),
            threadGroup: (rasterizeTG, 1, 1),
            outputShapes: [[N, V, 4]],
            outputDTypes: [.float32],
            initValue: 0.0
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
