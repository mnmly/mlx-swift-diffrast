# Acknowledgements

## Inspiration

This project is an independent Swift / Metal reimplementation of the
differentiable rasterization primitives originally introduced by:

> Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen,
> Timo Aila. **"Modular Primitives for High-Performance Differentiable
> Rendering."** *ACM Transactions on Graphics* 39(6) (SIGGRAPH Asia 2020).
> [arXiv:2011.03277](https://arxiv.org/abs/2011.03277)

The reference implementation [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast)
is published by NVIDIA Corporation under a source-available (non-OSS) license.
**No source code from nvdiffrast is used, vendored, or redistributed in this
repository.** The Swift / Metal kernels here were written from scratch against
the public API surface described in the paper and the project documentation.

If you use this library in academic work, please cite the original paper above.

```bibtex
@article{Laine2020diffrast,
  title   = {Modular Primitives for High-Performance Differentiable Rendering},
  author  = {Laine, Samuli and Hellsten, Janne and Karras, Tero and Seol, Yeongho
             and Lehtinen, Jaakko and Aila, Timo},
  journal = {ACM Transactions on Graphics},
  year    = {2020},
  volume  = {39},
  number  = {6}
}
```

## Dependencies

| Package | License | Role |
|---|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | MIT | Tensor backend, custom-VJP machinery, JIT-compiled Metal kernels |
| [swift-numerics](https://github.com/apple/swift-numerics) (transitive) | Apache-2.0 | Pulled in by mlx-swift |

## License

This project is released under the MIT License — see [`LICENSE`](LICENSE).
