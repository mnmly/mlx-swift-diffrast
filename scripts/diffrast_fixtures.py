"""
Reference fixture generator: run nvdiffrast on fixed deterministic inputs and
save outputs (forward + backward) into a single .safetensors file. Companion
Swift test loads it, runs the Swift reimplementation, and compares.

Run inside the trellis2 conda env on the CUDA host:

    python /root/diffrast_fixtures.py --out /mnt/c/Users/iam/Downloads/diffrast_fixtures.safetensors

Inputs are constructed deterministically (no RNG dependency) so the Swift side
can rebuild bit-identical inputs and only compare the produced *outputs*
against the saved nvdiffrast result.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import nvdiffrast.torch as dr
from safetensors.torch import save_file


def make_inputs():
    pos = torch.tensor(
        [[
            [-0.7,  0.8, 0.0, 1.0],
            [ 0.8,  0.7, 0.1, 1.0],
            [ 0.7, -0.8, 0.2, 1.0],
            [-0.8, -0.7, 0.0, 1.0],
        ]],
        dtype=torch.float32, device="cuda",
    )
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device="cuda")
    attr = torch.tensor(
        [[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]],
        dtype=torch.float32, device="cuda",
    )
    H_tex = W_tex = 8
    tex = torch.zeros((1, H_tex, W_tex, 3), dtype=torch.float32, device="cuda")
    for y in range(H_tex):
        for x in range(W_tex):
            tex[0, y, x, 0] = x / (W_tex - 1)
            tex[0, y, x, 1] = y / (H_tex - 1)
            tex[0, y, x, 2] = (x + y) / (W_tex + H_tex - 2)
    uv_verts = torch.tensor(
        [[[0.10, 0.10], [0.85, 0.20], [0.90, 0.85], [0.20, 0.90]]],
        dtype=torch.float32, device="cuda",
    )
    return pos, tri, attr, tex, uv_verts


def run_fixtures():
    out = {}

    resolution = (16, 16)
    pos, tri, attr, tex, uv_verts = make_inputs()
    pos.requires_grad_(True)
    attr.requires_grad_(True)
    tex.requires_grad_(True)

    glctx = dr.RasterizeCudaContext()

    # Deterministic cotangents (same construction as the Swift side).
    def fill(shape, fn):
        t = torch.zeros(shape, dtype=torch.float32, device="cuda")
        for idx in torch.cartesian_prod(*[torch.arange(s) for s in shape]):
            t[tuple(idx.tolist())] = fn(*idx.tolist())
        return t

    # ------------------------------------------------------------------
    # 1) rasterize: forward + backward
    # ------------------------------------------------------------------
    rast, rast_db = dr.rasterize(glctx, pos, tri, resolution=resolution)
    out["rast"] = rast.detach().contiguous()
    out["rast_db"] = rast_db.detach().contiguous()

    co_rast = fill((1, 16, 16, 4),
        lambda b, h, w, c: 0.0 if c == 3
        else 0.1 + 0.01 * h + 0.005 * w if c == 0
        else 0.2 + 0.005 * h * w if c == 1
        else 0.3 - 0.002 * (h + w))
    co_rast_db = fill((1, 16, 16, 4),
        lambda b, h, w, c: 0.01 * (1 + (h + w + c) % 3))
    out["co_rast"] = co_rast.contiguous()
    out["co_rast_db"] = co_rast_db.contiguous()

    ((rast * co_rast).sum() + (rast_db * co_rast_db).sum()).backward()
    out["d_pos_from_rast"] = pos.grad.detach().clone()
    pos.grad.zero_()

    # ------------------------------------------------------------------
    # 2) interpolate
    # ------------------------------------------------------------------
    rast2, rast_db2 = dr.rasterize(glctx, pos.detach(), tri, resolution=resolution)
    rast_d = rast2.detach().clone().requires_grad_(True)
    rast_db_d = rast_db2.detach().clone().requires_grad_(True)
    out_interp, out_da = dr.interpolate(attr, rast_d, tri,
                                         rast_db=rast_db_d, diff_attrs="all")
    out["interp_out"] = out_interp.detach().contiguous()
    out["interp_out_da"] = out_da.detach().contiguous()
    co_out = fill((1, 16, 16, 3),
        lambda b, h, w, c: 0.1 + 0.01 * (h + w + c))
    co_out_da = fill((1, 16, 16, 6),
        lambda b, h, w, c: 0.05 - 0.001 * (h - w + c))
    out["co_interp_out"] = co_out.contiguous()
    out["co_interp_out_da"] = co_out_da.contiguous()
    ((out_interp * co_out).sum() + (out_da * co_out_da).sum()).backward()
    out["d_attr_from_interp"] = attr.grad.detach().clone()
    out["d_rast_from_interp"] = rast_d.grad.detach().clone()
    out["d_rast_db_from_interp"] = rast_db_d.grad.detach().clone()
    attr.grad.zero_()

    # ------------------------------------------------------------------
    # 3) texture: bilinear + trilinear
    # ------------------------------------------------------------------
    rast3, rast_db3 = dr.rasterize(glctx, pos.detach(), tri, resolution=resolution)
    uv, uv_da = dr.interpolate(uv_verts, rast3, tri,
                                rast_db=rast_db3, diff_attrs="all")
    out["uv"] = uv.detach().contiguous()
    out["uv_da"] = uv_da.detach().contiguous()

    uv_b = uv.detach().clone().requires_grad_(True)
    sampled = dr.texture(tex, uv_b, filter_mode="linear", boundary_mode="clamp")
    out["tex_bilinear"] = sampled.detach().contiguous()
    co_tex = fill((1, 16, 16, 3),
        lambda b, h, w, c: 0.1 + 0.01 * (h + w + c))
    out["co_tex"] = co_tex.contiguous()
    (sampled * co_tex).sum().backward()
    out["d_tex_from_bilinear"] = tex.grad.detach().clone()
    out["d_uv_from_bilinear"] = uv_b.grad.detach().clone()
    tex.grad.zero_()

    uv_t = uv.detach().clone().requires_grad_(True)
    uvda_t = uv_da.detach().clone().requires_grad_(True)
    sampled_tri = dr.texture(
        tex, uv_t, uvda_t,
        filter_mode="linear-mipmap-linear", boundary_mode="clamp")
    out["tex_trilinear"] = sampled_tri.detach().contiguous()
    (sampled_tri * co_tex).sum().backward()
    out["d_tex_from_trilinear"] = tex.grad.detach().clone()
    out["d_uv_from_trilinear"] = uv_t.grad.detach().clone()
    out["d_uvDA_from_trilinear"] = uvda_t.grad.detach().clone()
    tex.grad.zero_()

    # ------------------------------------------------------------------
    # 4) antialias
    # ------------------------------------------------------------------
    pos.grad = None
    rast4, _ = dr.rasterize(glctx, pos, tri, resolution=resolution)
    color = torch.full((1, 16, 16, 3), 0.7,
                       dtype=torch.float32, device="cuda")
    color.requires_grad_(True)
    aa = dr.antialias(color, rast4, pos, tri)
    out["antialias_out"] = aa.detach().contiguous()
    co_aa = fill((1, 16, 16, 3),
        lambda b, h, w, c: 0.1 + 0.01 * (h - w + c))
    out["co_antialias"] = co_aa.contiguous()
    (aa * co_aa).sum().backward()
    out["d_color_from_antialias"] = color.grad.detach().clone()
    out["d_pos_from_antialias"] = pos.grad.detach().clone()

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing fixtures to {out_path}")
    tensors = run_fixtures()
    # safetensors needs contiguous CPU tensors.
    cpu_tensors = {k: v.cpu().contiguous() for k, v in tensors.items()}
    save_file(cpu_tensors, str(out_path))
    print(f"\nDone. {len(cpu_tensors)} tensors in {out_path}")
    for k, v in cpu_tensors.items():
        print(f"  {k:32s} shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
