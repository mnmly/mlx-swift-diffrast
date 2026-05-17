// Tier 1: optimize per-vertex colors to match a target image.
//
// Fixed triangle, fixed camera. The only differentiable parameter is the
// per-vertex color tensor `attr ∈ ℝ^{1×3×3}`. Pipeline:
//   rasterize → interpolate(attr) → MSE(rendered, target)
// `antialias` isn't needed since colors are smooth and silhouettes don't move.
//
// Output: animated GIF showing target | current side-by-side over the
// optimization.

import Foundation
import CoreGraphics
import MLX
import MLXDiffRast
import MLXDiffRastExamples

// -----------------------------------------------------------------------------
// Scene definition

let imageSize = 64
let tri = MLXArray([Int32(0), 1, 2], [1, 3])
let pos = MLXArray([
    Float(-0.8),  0.8, 0, 1,     // v0: top-left
    Float(-0.8), -0.8, 0, 1,     // v1: bottom-left
    Float( 0.8),  0.0, 0, 1,     // v2: middle-right
], [1, 3, 4])

// Target per-vertex colors: pure red, green, blue.
let targetAttr = MLXArray([
    Float(1), 0, 0,
    Float(0), 1, 0,
    Float(0), 0, 1,
], [1, 3, 3])

// Pre-rasterize once — pos is fixed, so rast/rastDB are constants.
let (rast, _) = DiffRast.rasterize(
    pos, tri: tri, resolution: (height: imageSize, width: imageSize), gradDB: false)

func render(_ attr: MLXArray) -> MLXArray {
    let (out, _) = DiffRast.interpolate(attr, rast: rast, tri: tri)
    return out
}

let targetImage = render(targetAttr)
targetImage.eval()

// -----------------------------------------------------------------------------
// Optimization

// Start with mid-gray at every vertex — nothing in the initial render hints
// at the target gradient.
var attr = MLXArray(Array(repeating: Float(0.5), count: 9), [1, 3, 3])
let optimizer = Adam(shape: attr.shape, learningRate: 0.05)
let numSteps = 120
let frameEvery = 4    // capture one GIF frame every 4 steps → ~30 frames

func lossFn(_ attr: MLXArray) -> MLXArray {
    let rendered = render(attr)
    let diff = rendered - targetImage
    return (diff * diff).mean()
}
let valueAndGradLoss = MLX.valueAndGrad({ (inputs: [MLXArray]) -> [MLXArray] in
    [lossFn(inputs[0])]
})

var frames: [CGImage] = []
let targetRGB = targetImage.reshaped([imageSize, imageSize, 3])
frames.append(sideBySide(target: targetRGB,
                         current: render(attr).reshaped([imageSize, imageSize, 3])))

print("ColorFitExample: optimizing per-vertex RGB to match a target gradient triangle.")
print("Image \(imageSize)×\(imageSize), \(numSteps) steps, lr=\(optimizer.learningRate)")

for step in 1...numSteps {
    let (vals, grads) = valueAndGradLoss([attr])
    let lossVal = vals[0].item(Float.self)
    attr = optimizer.step(param: attr, grad: grads[0])
    attr.eval()

    if step % frameEvery == 0 || step == numSteps {
        let cur = render(attr).reshaped([imageSize, imageSize, 3])
        frames.append(sideBySide(target: targetRGB, current: cur))
    }
    if step % 10 == 0 || step == 1 {
        print(String(format: "  step %3d  loss = %.6f", step, lossVal))
    }
}

let outURL = URL(fileURLWithPath: "color_fit.gif", isDirectory: false)
try writeAnimatedGIF(frames, to: outURL, frameDelay: 0.08)
print("Wrote \(frames.count) frames → \(outURL.path)")
