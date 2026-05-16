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

    /// The M4.1 stub must be a pass-through forward and identity-on-color VJP.
    /// Once M4.2 lands these expectations will tighten.
    func testAntialiasIsCurrentlyPassThrough() {
        let N = 1, H = 2, W = 2, C = 3
        let color = MLXArray((0..<(N * H * W * C)).map { Float($0) },
                             [N, H, W, C])
        // Rast doesn't need to be meaningful for the stub.
        let rast = MLXArray.zeros([N, H, W, 4])
        let pos = MLXArray.zeros([1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let hash = DiffRast.antialiasConstructTopologyHash(tri)

        let out = DiffRast.antialias(color: color, rast: rast, pos: pos, tri: tri,
                                     topologyHash: hash)
        XCTAssertEqual(out.asArray(Float.self), color.asArray(Float.self))
    }

    func testAntialiasColorGradientIsIdentity() {
        let N = 1, H = 2, W = 2, C = 2
        let color = MLXArray((0..<(N * H * W * C)).map { Float($0) },
                             [N, H, W, C])
        let rast = MLXArray.zeros([N, H, W, 4])
        let pos = MLXArray.zeros([1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let hash = DiffRast.antialiasConstructTopologyHash(tri)
        let w = MLXArray((0..<(N * H * W * C)).map { Float($0) * 0.1 + 0.05 },
                        [N, H, W, C])

        let loss: (MLXArray) -> MLXArray = { c in
            (DiffRast.antialias(color: c, rast: rast, pos: pos, tri: tri,
                                topologyHash: hash) * w).sum()
        }
        let g = MLX.grad(loss)(color)
        XCTAssertEqual(g.asArray(Float.self), w.asArray(Float.self),
                       "Stub VJP should pass cotangent through to color unchanged.")
    }
}
