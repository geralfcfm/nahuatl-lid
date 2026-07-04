from __future__ import annotations
import numpy as np
import torch

def aggregated_cam(model, specs, device="cpu"):
    model = model.to(device); model.train()  # gradients needed
    target = model.conv[-4]  # last Conv2d block's output (conv layer)
    acts, grads = [], []
    fh = target.register_forward_hook(lambda m, i, o: acts.append(o))
    bh = target.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    cams = []
    try:
        for spec in specs:
            acts.clear(); grads.clear()
            x = spec.unsqueeze(0).to(device).requires_grad_(True)
            out = model(x); model.zero_grad(); out[0, 0].backward()
            g = grads[0][0].detach().cpu().numpy()      # (C, H, W)
            a = acts[0][0].detach().cpu().numpy()       # (C, H, W)
            w = g.mean(axis=(1, 2))
            cam = np.maximum((w[:, None, None] * a).sum(axis=0), 0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam)
    finally:
        fh.remove(); bh.remove()
    return np.mean(cams, axis=0)
