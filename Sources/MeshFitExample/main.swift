// Tier 3: optimize a multi-triangle mesh's vertex positions.
//
// Hexagonal pie: 1 center vertex + 6 outer vertices = 7 vertices, 6 triangles
// sharing the center. Per-vertex colors are a color wheel — the optimizer has
// to land each colored slice in the right place. This stresses:
//   - Multi-triangle topology (silhouette detection across internal edges)
//   - All four silhouette directions (outer hexagon is geometrically rich)
//   - Joint optimization of all 7 positions
//
// Output: animated GIF.

import Foundation
import CoreGraphics
import MLX
import MLXDiffRast
import MLXDiffRastExamples


// -----------------------------------------------------------------------------
// Mesh definition: hexagonal triangle-fan

let imageSize = 96   // a bit larger so the hexagon has room to show its slices

func hexagonPos(center: (Float, Float), radius: Float) -> MLXArray {
    var vals: [Float] = [center.0, center.1, 0, 1]   // v0 = center
    for k in 0..<6 {
        let a = Float(k) * (.pi / 3)                  // 0°, 60°, ..., 300°
        let x = center.0 + radius * cos(a)
        let y = center.1 + radius * sin(a)
        vals.append(contentsOf: [x, y, 0, 1])
    }
    return MLXArray(vals, [1, 7, 4])
}

let tri = MLXArray([
    Int32(0), 1, 2,
    Int32(0), 2, 3,
    Int32(0), 3, 4,
    Int32(0), 4, 5,
    Int32(0), 5, 6,
    Int32(0), 6, 1,
], [6, 3])
let topology = DiffRast.antialiasConstructTopologyHash(tri)

// Color wheel — bright saturated colors around the 6 outer vertices, white
// at the center. Makes mis-placement visually loud.
let attr = MLXArray([
    Float(1.0),  1.0,  1.0,    // v0 center: white
    Float(1.0),  0.2,  0.2,    // v1: red
    Float(1.0),  0.9,  0.2,    // v2: yellow
    Float(0.2),  0.9,  0.3,    // v3: green
    Float(0.2),  0.9,  0.9,    // v4: cyan
    Float(0.3),  0.4,  1.0,    // v5: blue
    Float(1.0),  0.3,  0.9,    // v6: magenta
], [1, 7, 3])

// Target: hexagon centered at origin with radius 0.55 NDC.
let targetPos = hexagonPos(center: (0, 0), radius: 0.55)

func render(_ pos: MLXArray) -> MLXArray {
    let (rast, _) = DiffRast.rasterize(
        pos, tri: tri,
        resolution: (height: imageSize, width: imageSize),
        gradDB: false)
    let (interp, _) = DiffRast.interpolate(attr, rast: rast, tri: tri)
    return DiffRast.antialias(
        color: interp, rast: rast, pos: pos, tri: tri, topologyHash: topology)
}

let targetImage = render(targetPos)
targetImage.eval()

// -----------------------------------------------------------------------------
// Optimization

// Initial: shifted hexagon, slightly larger. Avoids NDC values that map to
// pixel boundaries by adding tiny offsets to break ties.
var pos = hexagonPos(center: (0.13, -0.09), radius: 0.42)

let posMask = MLXArray(
    Array(repeating: [Float(1), 1, 0, 0], count: 7).flatMap { $0 },
    [1, 7, 4]
)

let optimizer = Adam(shape: pos.shape, learningRate: 0.015)
let numSteps = 400
let frameEvery = 8

func lossFn(_ pos: MLXArray) -> MLXArray {
    let rendered = render(pos)
    let diff = rendered - targetImage
    return (diff * diff).mean()
}
let valueAndGradLoss = MLX.valueAndGrad({ (inputs: [MLXArray]) -> [MLXArray] in
    [lossFn(inputs[0])]
})

var frames: [CGImage] = []
let targetVis = targetImage.reshaped([imageSize, imageSize, 3])
frames.append(sideBySide(target: targetVis,
                         current: render(pos).reshaped([imageSize, imageSize, 3])))

print("MeshFitExample: optimizing a 7-vertex / 6-triangle hexagonal pie.")
print("Image \(imageSize)×\(imageSize), \(numSteps) steps, lr=\(optimizer.learningRate)")
print("")

for step in 1...numSteps {
    let (vals, grads) = valueAndGradLoss([pos])
    let lossVal = vals[0].item(Float.self)
    let g = grads[0] * posMask
    pos = optimizer.step(param: pos, grad: g)
    pos.eval()

    if step % frameEvery == 0 || step == numSteps {
        let cur = render(pos).reshaped([imageSize, imageSize, 3])
        frames.append(sideBySide(target: targetVis, current: cur))
    }
    if step % 20 == 0 || step == 1 {
        print(String(format: "  step %3d  loss = %.6f", step, lossVal))
    }
}

let final = pos.asArray(Float.self)
let target = targetPos.asArray(Float.self)
print("\nFinal vertex offsets from target (x, y):")
for v in 0..<7 {
    let dx = final[v * 4 + 0] - target[v * 4 + 0]
    let dy = final[v * 4 + 1] - target[v * 4 + 1]
    let label = v == 0 ? "center" : "outer\(v)"
    print(String(format: "  v%d (%@)  Δ = (%+.4f, %+.4f)", v, label as NSString, dx, dy))
}

let outURL = URL(fileURLWithPath: "mesh_fit.gif", isDirectory: false)
try writeAnimatedGIF(frames, to: outURL, frameDelay: 0.05)
print("\nWrote \(frames.count) frames → \(outURL.path)")
