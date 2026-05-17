// Tier 2: optimize vertex *positions* to match a target image.
//
// This is the killer test for `antialias`. The pipeline is:
//   rasterize(pos) → interpolate(attr, rast) → antialias(color, rast, pos)
// MSE against a target image whose only difference from the current render is
// where the quad sits in the frame.
//
// Without `antialias`'s silhouette pos gradient, the loss would be piecewise
// constant in `pos` (coverage flips discretely at pixel boundaries) and Adam
// would have nothing to follow. With it, sub-pixel `α` is smooth in screen
// space → the optimizer pulls the silhouette toward the target's silhouette.
//
// Output: animated GIF of target | current.

import Foundation
import CoreGraphics
import MLX
import MLXDiffRast
import MLXDiffRastExamples

// -----------------------------------------------------------------------------
// Scene definition

let imageSize = 64

// Quad: two triangles sharing the v0-v2 diagonal. Boundary edges are the four
// sides; the interior diagonal is shared (so the topology lookup unambiguously
// resolves silhouette-side classification).
let tri = MLXArray([
    Int32(0), 1, 2,
    Int32(0), 2, 3,
], [2, 3])
let topology = DiffRast.antialiasConstructTopologyHash(tri)

// Per-vertex color: solid red. Means the *only* gradient signal during
// optimization comes from silhouette pixels — interior of the quad matches
// target everywhere it's covered, contributing zero gradient. Perfect stress
// test for the silhouette pos gradient.
let attr = MLXArray(
    (0..<4).flatMap { _ in [Float(0.95), Float(0.2), Float(0.2)] },
    [1, 4, 3]
)

// Target: quad centered at the origin, ±0.4 NDC.
let targetPos = MLXArray([
    Float(-0.4),  0.4, 0, 1,
    Float( 0.4),  0.4, 0, 1,
    Float( 0.4), -0.4, 0, 1,
    Float(-0.4), -0.4, 0, 1,
], [1, 4, 4])

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

// Initial pose: shifted and squashed so silhouettes overlap with the target but
// aren't aligned. Significant initial loss but a clear path home.
var pos = MLXArray([
    Float(-0.25), 0.55, 0, 1,
    Float( 0.55), 0.55, 0, 1,
    Float( 0.55), -0.25, 0, 1,
    Float(-0.25), -0.25, 0, 1,
], [1, 4, 4])

// Only the (x, y) components of clip-space pos are meaningful here — z is
// degenerate (all at 0) and w fixed at 1. Mask gradients on (z, w) so Adam
// doesn't drift them.
let posMask = MLXArray(
    Array(repeating: [Float(1), 1, 0, 0], count: 4).flatMap { $0 },
    [1, 4, 4]
)

let optimizer = Adam(shape: pos.shape, learningRate: 0.01)
let numSteps = 300
let frameEvery = 6

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

print("PoseFitExample: optimizing quad vertex positions to match a target pose.")
print("Image \(imageSize)×\(imageSize), \(numSteps) steps, lr=\(optimizer.learningRate)")
print("Without antialias the loss would be piecewise constant in pos — this run")
print("only converges because the silhouette α gradient is wired through.")
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

// Final pos diagnostics.
let final = pos.asArray(Float.self)
let target = targetPos.asArray(Float.self)
print("\nFinal vertex offsets from target (x, y) per vertex:")
for v in 0..<4 {
    let dx = final[v * 4 + 0] - target[v * 4 + 0]
    let dy = final[v * 4 + 1] - target[v * 4 + 1]
    print(String(format: "  v%d  Δ = (%+.4f, %+.4f)", v, dx, dy))
}

let outURL = URL(fileURLWithPath: "pose_fit.gif", isDirectory: false)
try writeAnimatedGIF(frames, to: outURL, frameDelay: 0.06)
print("\nWrote \(frames.count) frames → \(outURL.path)")
