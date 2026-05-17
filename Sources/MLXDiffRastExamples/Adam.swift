import Foundation
import MLX

/// Plain Adam optimizer for a single MLXArray parameter. State lives on the
/// instance so successive `step` calls accumulate momentum.
///
/// Tracks the standard `m, v, t` Adam state; the bias-corrected updates match
/// PyTorch's `torch.optim.Adam` defaults (`eps = 1e-8`).
public final class Adam {
    public let learningRate: Float
    public let beta1: Float
    public let beta2: Float
    public let eps: Float

    private var m: MLXArray
    private var v: MLXArray
    private var t: Int = 0

    public init(
        shape: [Int],
        learningRate: Float = 1e-2,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        eps: Float = 1e-8
    ) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = MLXArray.zeros(shape, dtype: .float32)
        self.v = MLXArray.zeros(shape, dtype: .float32)
    }

    /// Returns the updated parameter. `param` and `grad` must share the same
    /// shape as the one passed to `init`.
    public func step(param: MLXArray, grad: MLXArray) -> MLXArray {
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        let mHat = m / (1 - pow(Float(beta1), Float(t)))
        let vHat = v / (1 - pow(Float(beta2), Float(t)))
        return param - learningRate * mHat / (sqrt(vHat) + eps)
    }
}
