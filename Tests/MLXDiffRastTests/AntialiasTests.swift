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

    /// M4.3 placeholder: d_pos should remain zero until the silhouette
    /// gradient is implemented.
    func testPosGradientIsCurrentlyZero() {
        let N = 1, H = 3, W = 3, C = 1
        let pos = MLXArray([
            Float(-3),  3, 0, 1,
            Float(-3), -2.7, 0, 1,
            Float( 2.7), 3, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let (rast, _) = DiffRast.rasterize(pos, tri: tri,
                                           resolution: (height: H, width: W),
                                           gradDB: false)
        let color = MLXArray((0..<(N * H * W * C)).map { Float($0) },
                             [N, H, W, C])
        let loss: (MLXArray) -> MLXArray = { p in
            DiffRast.antialias(color: color, rast: rast, pos: p, tri: tri).sum()
        }
        let g = MLX.grad(loss)(pos)
        for v in g.asArray(Float.self) {
            XCTAssertEqual(v, 0, accuracy: 1e-7,
                           "d_pos must stay zero until M4.3 (silhouette pos gradient)")
        }
    }
}
