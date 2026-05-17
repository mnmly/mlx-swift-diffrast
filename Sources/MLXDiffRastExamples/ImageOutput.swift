import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX

/// Convert an `[H, W, C]` float MLXArray (C ∈ {1, 3, 4}, values in [0, 1])
/// to a `CGImage`. Out-of-range values are clamped.
public func mlxArrayToCGImage(_ array: MLXArray) -> CGImage {
    precondition(array.ndim == 3, "expected [H, W, C], got \(array.shape)")
    let H = array.shape[0], W = array.shape[1], C = array.shape[2]
    precondition(C == 1 || C == 3 || C == 4, "channels must be 1, 3, or 4 (got \(C))")
    let flat = array.asArray(Float.self)
    var bytes = [UInt8](repeating: 0, count: H * W * 4)  // RGBA8 output
    for i in 0..<(H * W) {
        if C == 1 {
            let g = UInt8(max(0, min(1, flat[i])) * 255)
            bytes[i * 4 + 0] = g
            bytes[i * 4 + 1] = g
            bytes[i * 4 + 2] = g
        } else {
            for c in 0..<3 {
                bytes[i * 4 + c] = UInt8(max(0, min(1, flat[i * C + c])) * 255)
            }
        }
        bytes[i * 4 + 3] = (C == 4)
            ? UInt8(max(0, min(1, flat[i * C + 3])) * 255)
            : 255
    }
    let data = CFDataCreate(nil, bytes, bytes.count)!
    let provider = CGDataProvider(data: data)!
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
    return CGImage(
        width: W, height: H,
        bitsPerComponent: 8, bitsPerPixel: 32,
        bytesPerRow: W * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: bitmapInfo,
        provider: provider, decode: nil,
        shouldInterpolate: false, intent: .defaultIntent
    )!
}

/// Place `target` on the left and `current` on the right with a 2px separator,
/// returning a single composite `CGImage` (RGB).
public func sideBySide(target: MLXArray, current: MLXArray) -> CGImage {
    precondition(target.shape == current.shape)
    let H = target.shape[0], W = target.shape[1], C = target.shape[2]
    let sep = 2
    let outW = W * 2 + sep
    let space = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(
        data: nil, width: outW, height: H,
        bitsPerComponent: 8, bytesPerRow: outW * 4,
        space: space,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue).rawValue
    )!
    ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: outW, height: H))
    let tgtImg = mlxArrayToCGImage(target)
    let curImg = mlxArrayToCGImage(current)
    ctx.draw(tgtImg, in: CGRect(x: 0, y: 0, width: W, height: H))
    ctx.draw(curImg, in: CGRect(x: W + sep, y: 0, width: W, height: H))
    _ = C  // silence unused warning if optimization removes it
    return ctx.makeImage()!
}

/// Write a sequence of frames as an animated GIF.
///   - frames: per-frame images (must all be the same size)
///   - url:    output file URL
///   - frameDelay: seconds between frames (0.1 ≈ 10 fps)
///   - loop:   loop count (0 = infinite)
public func writeAnimatedGIF(
    _ frames: [CGImage], to url: URL, frameDelay: Double = 0.1, loop: Int = 0
) throws {
    guard !frames.isEmpty else { return }
    let destination = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.gif.identifier as CFString, frames.count, nil
    )
    guard let destination else {
        throw NSError(domain: "MLXDiffRastExamples", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "Failed to create CGImageDestination for \(url.path)"
        ])
    }
    let fileProps: [CFString: Any] = [
        kCGImagePropertyGIFDictionary: [
            kCGImagePropertyGIFLoopCount: loop,
        ] as [CFString: Any]
    ]
    CGImageDestinationSetProperties(destination, fileProps as CFDictionary)
    let frameProps: [CFString: Any] = [
        kCGImagePropertyGIFDictionary: [
            kCGImagePropertyGIFDelayTime: frameDelay,
        ] as [CFString: Any]
    ]
    for frame in frames {
        CGImageDestinationAddImage(destination, frame, frameProps as CFDictionary)
    }
    if !CGImageDestinationFinalize(destination) {
        throw NSError(domain: "MLXDiffRastExamples", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "CGImageDestinationFinalize failed for \(url.path)"
        ])
    }
}
