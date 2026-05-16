import MLX

/// Differentiable rasterization primitives for MLX Swift.
///
/// Port of NVIDIA's nvdiffrast (https://github.com/NVlabs/nvdiffrast). The CUDA/OpenGL
/// kernels are reimplemented as JIT-compiled Metal compute kernels behind MLX custom
/// VJPs so they participate in `grad` / `valueAndGrad`.
///
/// Status:
///   - interpolate: implemented (forward + VJP for `attr` and `rast`)
///   - rasterize:   TODO (Metal render-pass based)
///   - texture:     TODO
///   - antialias:   TODO
///
/// Rasterizer-output layout matches nvdiffrast exactly:
///   `rast[n, h, w, :] = (u, v, z/w, triangle_id_plus_one_as_float)`
///   where `triangle_id_plus_one == 0` marks an empty pixel.
public enum DiffRast {}
