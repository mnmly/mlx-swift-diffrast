import Foundation
import MLX

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

        // Each directed (vertex_min, vertex_max, triangle, edge_index) tuple.
        struct EdgeRef { let a: Int32; let b: Int32; let t: Int; let k: Int }
        var edges: [EdgeRef] = []
        edges.reserveCapacity(T * 3)
        for t in 0..<T {
            let v0 = triFlat[t * 3 + 0]
            let v1 = triFlat[t * 3 + 1]
            let v2 = triFlat[t * 3 + 2]
            // Edge k is the edge opposite vertex k.
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
            // Walk run of edges sharing the same (a, b).
            var j = i + 1
            while j < edges.count && edges[j].a == e.a && edges[j].b == e.b { j += 1 }
            let runLen = j - i
            if runLen == 2 {
                let e0 = edges[i], e1 = edges[i + 1]
                neighbor[e0.t * 3 + e0.k] = Int32(e1.t)
                neighbor[e1.t * 3 + e1.k] = Int32(e0.t)
            }
            // runLen == 1 → boundary edge, already -1.
            // runLen > 2 is non-manifold; leave as -1 for first match, ignore the rest.
            i = j
        }
        return MLXArray(neighbor, [T, 3])
    }

    /// Silhouette-aware antialiasing — **forward stub** for the M4 API.
    ///
    /// **Status (M4.1):** this function currently returns `color` unchanged
    /// and routes cotangents straight through to `d_color`; `d_pos` is zero.
    /// The `topologyHash` argument is accepted and validated but unused.
    ///
    /// The full implementation needs:
    ///   1. Per-pixel silhouette detection across right/down neighbors using
    ///      `rast` tri-ids and `topologyHash` to identify which triangle edge
    ///      is the silhouette.
    ///   2. Sub-pixel coverage `α` derived from where that edge crosses the
    ///      pixel boundary in screen space.
    ///   3. Blend `out = color_fg + α * (color_bg - color_fg)`.
    ///   4. VJP that propagates cotangents through `α` into both `color`
    ///      (easy) and `pos` (chain rule through edge-line intersection — the
    ///      hard part, analogous to rasterize's M2.3 backward).
    ///
    /// Implementing that correctly is the focus of M4.2; punting here lets the
    /// API surface land without shipping a wrong gradient.
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
        if let hash = topologyHash {
            precondition(hash.shape == [tri.shape[0], 3],
                         "antialias: topologyHash must be [T, 3] matching tri")
        }

        // M4.1 stub: identity pass with identity-VJP on color, zero on pos.
        // Defining it as a CustomFunction keeps the API consistent with the
        // future M4.2 implementation — drop-in upgrade with no caller changes.
        let fwd: ([MLXArray]) -> [MLXArray] = { inputs in [inputs[0]] }
        let vjp: ([MLXArray], [MLXArray]) -> [MLXArray] = { primals, cotangents in
            // primals: [color, pos] ; only color is differentiable here.
            [cotangents[0], MLXArray.zeros(primals[1].shape, dtype: .float32)]
        }
        let custom = CustomFunction { Forward(fwd); VJP(vjp) }
        return custom([color, pos])[0]
    }
}
