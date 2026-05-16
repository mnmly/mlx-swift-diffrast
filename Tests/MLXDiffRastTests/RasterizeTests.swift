import XCTest
import MLX
@testable import MLXDiffRast

final class RasterizeTests: XCTestCase {

    /// Single triangle covering the entire image. With vertices at the screen
    /// corners (top-left, bottom-left, top-right in pixel space), the barycentric
    /// at each pixel can be hand-computed from the pixel's NDC coords.
    func testSingleTriangleCoversImage() {
        // V0 = top-left (-1, +1), V1 = bottom-left (-1, -1), V2 = top-right (+1, +1)
        // All at z=0, w=1 → flat plane at NDC z=0.
        //
        // Edge functions in this configuration:
        //   area = (sx1-sx0)*(sy2-sy0) - (sx2-sx0)*(sy1-sy0)
        //        = (0)*(0) - (2)*(-2) = 4
        //   e0(p) = (sx1-px)*(sy2-py) - (sx2-px)*(sy1-py)
        //        = (-1-px)*(1-py)    - (1-px)*(-1-py)
        //   bw = e0 / 4
        //
        // For the center pixel (W/2, H/2) with W=H=2, ndc = (0, 0):
        //   e0 = (-1)*(1) - (1)*(-1) = -1 + 1 = 0 → bw = 0   (on edge v1v2)
        // For pixel (0, 0) (top-left), ndc = (-0.5, +0.5):
        //   e0 = (-0.5)*(0.5) - (1.5)*(-1.5) = -0.25 + 2.25 = 2.0 → bw = 0.5
        //   e1 = (1.5)*(0.5)  - (-0.5)*(0.5) = 0.75 + 0.25 = 1.0  → u = 0.25
        //   e2 = (-0.5)*(-1.5) - (-0.5)*(0.5) = 0.75 + 0.25 = 1.0 → v = 0.25
        let pos = MLXArray([
            Float(-1),  1, 0, 1,   // V0 top-left
            Float(-1), -1, 0, 1,   // V1 bottom-left
            Float( 1),  1, 0, 1,   // V2 top-right
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])

        let rast = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2))
        XCTAssertEqual(rast.shape, [1, 2, 2, 4])
        let flat = rast.asArray(Float.self)

        // Top-left pixel: u = 0.25, v = 0.25, z = 0, tri_id+1 = 1
        XCTAssertEqual(flat[0], 0.25, accuracy: 1e-5)
        XCTAssertEqual(flat[1], 0.25, accuracy: 1e-5)
        XCTAssertEqual(flat[2], 0.0,  accuracy: 1e-5)
        XCTAssertEqual(flat[3], 1.0,  accuracy: 1e-5)

        // This corner-triangle only covers the upper-left half of the image.
        // Bottom-right pixel (ndc 0.5, -0.5) lies outside the line y=x and is empty.
        XCTAssertEqual(flat[(0 * 2 + 0) * 4 + 3], 1.0, accuracy: 1e-5, "(0,0) inside")
        XCTAssertEqual(flat[(0 * 2 + 1) * 4 + 3], 1.0, accuracy: 1e-5, "(0,1) on edge → inside")
        XCTAssertEqual(flat[(1 * 2 + 0) * 4 + 3], 1.0, accuracy: 1e-5, "(1,0) inside")
        XCTAssertEqual(flat[(1 * 2 + 1) * 4 + 3], 0.0, accuracy: 1e-5, "(1,1) outside → empty")
    }

    /// Two coplanar triangles at different depths; closer one wins.
    func testDepthTestPicksFrontTriangle() {
        // Triangle 0: at z=0.5 (farther), covers full screen via the same corner setup.
        // Triangle 1: at z=-0.5 (closer), covers full screen.
        let pos = MLXArray([
            // tri 0 vertices: z = 0.5
            Float(-1),  1, 0.5, 1,
            Float(-1), -1, 0.5, 1,
            Float( 1),  1, 0.5, 1,
            // tri 1 vertices: z = -0.5
            Float(-1),  1, -0.5, 1,
            Float(-1), -1, -0.5, 1,
            Float( 1),  1, -0.5, 1,
        ], [1, 6, 4])
        let tri = MLXArray([
            Int32(0), 1, 2,
            Int32(3), 4, 5,
        ], [2, 3])

        let rast = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2))
        let flat = rast.asArray(Float.self)

        // Every pixel that's inside both should report tri 1 (tri_id+1 = 2) and z=-0.5.
        for h in 0..<2 {
            for w in 0..<2 {
                let base = (h * 2 + w) * 4
                if flat[base + 3] == 0 { continue }
                XCTAssertEqual(flat[base + 2], -0.5, accuracy: 1e-5,
                               "pixel (h=\(h), w=\(w)) depth")
                XCTAssertEqual(flat[base + 3], 2.0, accuracy: 1e-5,
                               "pixel (h=\(h), w=\(w)) should pick front triangle")
            }
        }
    }

    /// Pixels outside the triangle should be exactly zero across all channels.
    func testPixelsOutsideTriangleAreZero() {
        // A small triangle near the top-left corner of the image.
        // V0 = (-1, 1), V1 = (-0.5, 1), V2 = (-1, 0.5). Image is 4×4.
        let pos = MLXArray([
            Float(-1.0),  1.0,  0, 1,
            Float(-0.5),  1.0,  0, 1,
            Float(-1.0),  0.5,  0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let rast = DiffRast.rasterize(pos, tri: tri, resolution: (height: 4, width: 4))
        let flat = rast.asArray(Float.self)

        // Bottom-right pixel (h=3, w=3) ndc ≈ (0.75, -0.75) is far outside.
        let base = (3 * 4 + 3) * 4
        XCTAssertEqual(flat[base + 0], 0)
        XCTAssertEqual(flat[base + 1], 0)
        XCTAssertEqual(flat[base + 2], 0)
        XCTAssertEqual(flat[base + 3], 0)
    }

    /// Sanity: feeding the rast output into `interpolate` produces non-zero attrs
    /// on covered pixels. This is the first end-to-end check across the two ops.
    func testRasterizeIntoInterpolate() {
        // Same full-screen triangle as testSingleTriangleCoversImage.
        let pos = MLXArray([
            Float(-1),  1, 0, 1,
            Float(-1), -1, 0, 1,
            Float( 1),  1, 0, 1,
        ], [1, 3, 4])
        let tri = MLXArray([Int32(0), 1, 2], [1, 3])
        let rast = DiffRast.rasterize(pos, tri: tri, resolution: (height: 2, width: 2))

        // Per-vertex color: v0 = red, v1 = green, v2 = blue.
        let attr = MLXArray([
            Float(1), 0, 0,
            Float(0), 1, 0,
            Float(0), 0, 1,
        ], [1, 3, 3])

        let (out, _) = DiffRast.interpolate(attr, rast: rast, tri: tri)
        XCTAssertEqual(out.shape, [1, 2, 2, 3])

        // On covered pixels each RGB sums to 1 (one-hot per vertex partitions unity).
        // Empty pixels stay zero (interpolate writes 0 when tri_id+1 == 0).
        let flat = out.asArray(Float.self)
        let rastFlat = rast.asArray(Float.self)
        for pix in 0..<4 {
            let sum = flat[pix * 3] + flat[pix * 3 + 1] + flat[pix * 3 + 2]
            let covered = rastFlat[pix * 4 + 3] > 0
            XCTAssertEqual(sum, covered ? 1.0 : 0.0, accuracy: 1e-5,
                           "pixel \(pix) bary sum (covered=\(covered))")
        }
    }
}
