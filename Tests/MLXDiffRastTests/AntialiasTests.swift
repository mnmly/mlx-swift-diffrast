import XCTest
import MLX
@testable import MLXDiffRast

final class AntialiasTests: XCTestCase {

    // MARK: - Topology hash

    /// Two triangles sharing one edge should be marked as each other's neighbor
    /// on that edge; the other two edges of each are boundary.
    func testTopologyHashTwoTrianglesSharedEdge() {
        // Tri 0: vertices (0, 1, 2)
        // Tri 1: vertices (1, 2, 3)     ← shares edge (1, 2) with Tri 0
        // Tri 0's edges (opposite each vertex):
        //   k=0 (opposite v0=0): (v1, v2) = (1, 2) — SHARED with Tri 1
        //   k=1 (opposite v1=1): (v2, v0) = (2, 0) — boundary
        //   k=2 (opposite v2=2): (v0, v1) = (0, 1) — boundary
        // Tri 1's edges:
        //   k=0 (opposite v0=1): (v1, v2) = (2, 3) — boundary
        //   k=1 (opposite v1=2): (v2, v0) = (3, 1) — boundary
        //   k=2 (opposite v2=3): (v0, v1) = (1, 2) — SHARED with Tri 0
        let tri = MLXArray([
            Int32(0), 1, 2,
            Int32(1), 2, 3,
        ], [2, 3])
        let hash = DiffRast.antialiasConstructTopologyHash(tri)
        XCTAssertEqual(hash.shape, [2, 3])
        let flat = hash.asArray(Int32.self)

        XCTAssertEqual(flat[0 * 3 + 0], 1,  "Tri 0 edge 0 (1-2) should neighbor Tri 1")
        XCTAssertEqual(flat[0 * 3 + 1], -1, "Tri 0 edge 1 (2-0) boundary")
        XCTAssertEqual(flat[0 * 3 + 2], -1, "Tri 0 edge 2 (0-1) boundary")
        XCTAssertEqual(flat[1 * 3 + 0], -1, "Tri 1 edge 0 (2-3) boundary")
        XCTAssertEqual(flat[1 * 3 + 1], -1, "Tri 1 edge 1 (3-1) boundary")
        XCTAssertEqual(flat[1 * 3 + 2], 0,  "Tri 1 edge 2 (1-2) should neighbor Tri 0")
    }

    func testTopologyHashSingleTriangleAllBoundary() {
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let hash = DiffRast.antialiasConstructTopologyHash(tri)
        XCTAssertEqual(hash.asArray(Int32.self), [-1, -1, -1])
    }

    /// Edge shared by 3+ triangles is non-manifold. The topology builder
    /// should mark each instance with `-2` so the silhouette algorithm skips
    /// it (it can't unambiguously identify a single "neighbor").
    func testTopologyHashNonManifoldEdgeMarked() {
        // Three triangles all sharing the edge (v0, v1). Vertices v2, v3, v4
        // each pair with the shared edge to form a tri.
        //   Tri 0: (0, 1, 2)
        //   Tri 1: (0, 1, 3)
        //   Tri 2: (0, 1, 4)
        // Edge (0, 1) is the k=2 edge of each (edge opposite vertex 2 = (v0, v1)).
        let tri = MLXArray([
            Int32(0), 1, 2,
            Int32(0), 1, 3,
            Int32(0), 1, 4,
        ], [3, 3])
        let hash = DiffRast.antialiasConstructTopologyHash(tri).asArray(Int32.self)
        // All three tris' k=2 edge should be marked as non-manifold (-2).
        XCTAssertEqual(hash[0 * 3 + 2], -2, "tri 0 k=2 non-manifold")
        XCTAssertEqual(hash[1 * 3 + 2], -2, "tri 1 k=2 non-manifold")
        XCTAssertEqual(hash[2 * 3 + 2], -2, "tri 2 k=2 non-manifold")
        // The other 6 edges are pairwise boundary or shared. Edges (1, 2)
        // and (2, 0) of tri 0 are boundary, similarly for tris 1/2. No two
        // of these match, so they all stay -1.
        for t in 0..<3 {
            XCTAssertEqual(hash[t * 3 + 0], -1, "tri \(t) k=0 should be boundary")
            XCTAssertEqual(hash[t * 3 + 1], -1, "tri \(t) k=1 should be boundary")
        }
    }

    // MARK: - Antialias stub

    // MARK: - M4.2 silhouette blend

    /// All-empty rast (no triangles cover any pixel) → out == color exactly.
    func testNoSilhouetteIsIdentity() {
        let N = 1, H = 3, W = 3, C = 2
        let color = MLXArray((0..<(N * H * W * C)).map { Float($0) },
                             [N, H, W, C])
        let rast = MLXArray.zeros([N, H, W, 4])
        let pos = MLXArray.zeros([1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let out = DiffRast.antialias(color: color, rast: rast, pos: pos, tri: tri)
        XCTAssertEqual(out.asArray(Float.self), color.asArray(Float.self))
    }

    /// Single triangle covering all pixels uniformly → every pair has the same
    /// tri-id, no silhouettes detected, out == color.
    func testFullyCoveredIsIdentity() {
        let N = 1, H = 2, W = 2, C = 1
        let color = MLXArray([Float(1), 2, 3, 4], [N, H, W, C])
        // Triangle filling the screen.
        let pos = MLXArray([
            Float(-3),  3, 0, 1,
            Float(-3), -3, 0, 1,
            Float( 3),  0, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let (rast, _) = DiffRast.rasterize(pos, tri: tri,
                                           resolution: (height: H, width: W),
                                           gradDB: false)
        let out = DiffRast.antialias(color: color, rast: rast, pos: pos, tri: tri)
        XCTAssertEqual(out.asArray(Float.self), color.asArray(Float.self),
                       "All-interior pixels share a tri-id → no silhouette blending")
    }

    /// Single triangle with a silhouette running across pixels: pixels on the
    /// background side of the edge should have their colors pulled toward the
    /// foreground pixels by a positive α.
    func testSilhouetteBlendsCoveredAndEmptyPixels() {
        let N = 1, H = 4, W = 4, C = 1
        // Tilted triangle: covers upper-left half of the image, leaves the
        // bottom-right corner empty, so several pixels lie on a silhouette.
        let pos = MLXArray([
            Float(-3),  3, 0, 1,
            Float(-3), -2.7, 0, 1,
            Float( 2.7), 3, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let (rast, _) = DiffRast.rasterize(pos, tri: tri,
                                           resolution: (height: H, width: W),
                                           gradDB: false)
        let color = MLXArray((0..<(N * H * W * C)).map { Float(10 * (1 + $0)) },
                             [N, H, W, C])

        let out = DiffRast.antialias(color: color, rast: rast, pos: pos, tri: tri)
        let outFlat = out.asArray(Float.self)
        let colorFlat = color.asArray(Float.self)
        let rastFlat = rast.asArray(Float.self)

        // Find a silhouette pixel: empty pixel with at least one covered neighbor.
        // At least one such pixel should have out != color (got pulled toward fg).
        var foundBlend = false
        for h in 0..<H {
            for w in 0..<W {
                let idx = (h * W + w) * 4 + 3
                let isCovered = rastFlat[idx] > 0
                let neighbors: [(Int, Int)] = [(h, w + 1), (h + 1, w), (h, w - 1), (h - 1, w)]
                let hasMixedNeighbor = neighbors.contains { (nh, nw) in
                    guard nh >= 0 && nh < H && nw >= 0 && nw < W else { return false }
                    let nIdx = (nh * W + nw) * 4 + 3
                    return (rastFlat[nIdx] > 0) != isCovered
                }
                if hasMixedNeighbor {
                    if abs(outFlat[h * W + w] - colorFlat[h * W + w]) > 1e-4 {
                        foundBlend = true
                    }
                }
            }
        }
        XCTAssertTrue(foundBlend,
                      "Expected at least one silhouette pixel to be blended toward its neighbor")

        // Fully-interior pixels (all 4 neighbors share the same tri-id) should be unchanged.
        for h in 1..<(H - 1) {
            for w in 1..<(W - 1) {
                let idx = (h * W + w) * 4 + 3
                let myTri = rastFlat[idx]
                var allSame = true
                for nh in (h - 1)...(h + 1) {
                    for nw in (w - 1)...(w + 1) {
                        if rastFlat[(nh * W + nw) * 4 + 3] != myTri { allSame = false }
                    }
                }
                if allSame {
                    XCTAssertEqual(outFlat[h * W + w], colorFlat[h * W + w], accuracy: 1e-5,
                                   "Interior pixel (h=\(h), w=\(w)) should be unchanged")
                }
            }
        }
    }

    /// Gradcheck d_color through the silhouette blend.
    func testGradcheckColor() throws {
        let N = 1, H = 4, W = 4, C = 2
        let pos = MLXArray([
            Float(-3),  3, 0, 1,
            Float(-3), -2.7, 0, 1,
            Float( 2.7), 3, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let (rast, _) = DiffRast.rasterize(pos, tri: tri,
                                           resolution: (height: H, width: W),
                                           gradDB: false)
        let hash = DiffRast.antialiasConstructTopologyHash(tri)

        let color = MLXArray((0..<(N * H * W * C)).map { Float($0) * 0.3 + 1 },
                             [N, H, W, C])
        let wOut = MLXArray((0..<(N * H * W * C)).map { Float(1 + $0 % 7) * 0.1 },
                            [N, H, W, C])

        let loss: (MLXArray) -> MLXArray = { c in
            (DiffRast.antialias(color: c, rast: rast, pos: pos, tri: tri,
                                topologyHash: hash) * wOut).sum()
        }
        let analytic = MLX.grad(loss)(color)
        analytic.eval()
        let analyticFlat = analytic.asArray(Float.self)

        let eps: Float = 1e-3
        let flat = color.asArray(Float.self)
        for i in 0..<flat.count {
            var plus = flat; plus[i] += eps
            var minus = flat; minus[i] -= eps
            let lp = loss(MLXArray(plus, color.shape)).item(Float.self)
            let lm = loss(MLXArray(minus, color.shape)).item(Float.self)
            let fd = (lp - lm) / (2 * eps)
            XCTAssertEqual(analyticFlat[i], fd, accuracy: 5e-3,
                           "color elem \(i): analytic \(analyticFlat[i]) vs fd \(fd)")
        }
    }

    // MARK: - M4.3 silhouette pos gradient

    /// A quad (two triangles sharing a diagonal). Multiple boundary edges plus
    /// the interior shared edge make `topologyHash` lookups unambiguous, which
    /// matters for clean silhouette-edge selection in the kernel.
    ///
    /// Vertices placed off pixel centers so perturbations of size `eps` don't
    /// flip rasterize coverage (`rast` is precomputed once outside the loss).
    private func quadFixture() -> (pos: MLXArray, tri: MLXArray) {
        // NDC quad spanning roughly [-0.45, +0.55]² (off-center on purpose).
        let pos = MLXArray([
            Float(-0.45),  0.55, 0, 1,
            Float( 0.55),  0.55, 0, 1,
            Float( 0.55), -0.45, 0, 1,
            Float(-0.45), -0.45, 0, 1,
        ], [1, 4, 4])
        let tri = MLXArray([
            Int32(0), 1, 2,
            Int32(0), 2, 3,
        ], [2, 3])
        return (pos, tri)
    }

    func testGradcheckPos() throws {
        let N = 1, H = 4, W = 4, C = 2
        let (pos0, tri) = quadFixture()
        let hash = DiffRast.antialiasConstructTopologyHash(tri)
        // Rasterize ONCE with the unperturbed pos. Inside the loss, rast is
        // captured by value — so when FD perturbs `pos`, coverage stays fixed
        // and only the silhouette-edge projection (the differentiable path
        // through `pos` inside antialias) responds.
        let (rast, _) = DiffRast.rasterize(pos0, tri: tri,
                                           resolution: (height: H, width: W),
                                           gradDB: false)

        let color = MLXArray((0..<(N * H * W * C)).map { Float($0) * 0.3 + 1 },
                             [N, H, W, C])
        let wOut = MLXArray((0..<(N * H * W * C)).map { Float(1 + $0 % 5) * 0.1 },
                            [N, H, W, C])

        let loss: (MLXArray) -> MLXArray = { p in
            (DiffRast.antialias(color: color, rast: rast, pos: p, tri: tri,
                                topologyHash: hash) * wOut).sum()
        }
        let analytic = MLX.grad(loss)(pos0)
        analytic.eval()
        let analyticFlat = analytic.asArray(Float.self)

        let eps: Float = 1e-3
        let flat = pos0.asArray(Float.self)
        var numeric = [Float](repeating: 0, count: flat.count)
        for i in 0..<flat.count {
            // z channel of pos doesn't affect screen projection at all → both
            // analytic and FD should report 0. (We still test to confirm.)
            var plus = flat; plus[i] += eps
            var minus = flat; minus[i] -= eps
            let lp = loss(MLXArray(plus, pos0.shape)).item(Float.self)
            let lm = loss(MLXArray(minus, pos0.shape)).item(Float.self)
            numeric[i] = (lp - lm) / (2 * eps)
        }
        for i in 0..<flat.count {
            XCTAssertEqual(analyticFlat[i], numeric[i], accuracy: 5e-3,
                           "pos elem \(i): analytic \(analyticFlat[i]) vs fd \(numeric[i])")
        }
    }

    /// At least one pos element should receive a non-zero gradient — otherwise
    /// the test fixture isn't actually exercising the silhouette path.
    func testPosGradientIsNonTrivial() {
        let N = 1, H = 4, W = 4, C = 1
        let (pos0, tri) = quadFixture()
        let (rast, _) = DiffRast.rasterize(pos0, tri: tri,
                                           resolution: (height: H, width: W),
                                           gradDB: false)
        let color = MLXArray((0..<(N * H * W * C)).map { Float($0) },
                             [N, H, W, C])
        let loss: (MLXArray) -> MLXArray = { p in
            DiffRast.antialias(color: color, rast: rast, pos: p, tri: tri).sum()
        }
        let g = MLX.grad(loss)(pos0).asArray(Float.self)
        XCTAssertTrue(g.contains { abs($0) > 1e-5 },
                      "Expected at least one pos element to receive a nonzero gradient; got all zeros")
    }
}
