from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model import CRNN_LID
from .data import grouped_folds
from . import train, gradcam, config

def _save(cam, title, path):
    # cam is the Grad-CAM at the last Conv2d block (conv[-4]) resolution, NOT the raw
    # input spectrogram: after two 2x2 max-pools the 64 input Mel bins become H rows
    # (~4 input Mel bins each) and the time frames become W columns. Label the axes at
    # that downsampled conv-feature-map resolution so the index is not misread as a raw
    # 0-63 Mel-bin index.
    h, w = cam.shape
    plt.figure()
    plt.imshow(cam, origin="lower", aspect="auto", cmap="jet")
    plt.title(title)
    plt.colorbar()
    plt.xlabel(f"time (conv frame, 0--{w - 1})")
    plt.ylabel(f"Mel axis (conv row 0--{h - 1}; low $\\rightarrow$ high frequency)")
    plt.yticks(range(0, h, max(1, h // 8)))
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def render_cams(items, filenames, band, norm, device, out_dir, n_per_class=100, epochs=config.EPOCHS):
    tr, va = grouped_folds(filenames, config.K_FOLDS)[0]
    tr_items = [(items[i][0], items[i][1]) for i in tr]
    va_items = [(items[i][0], items[i][1]) for i in va]
    hist = train.train_fold(tr_items, va_items, norm, device=device, epochs=epochs)
    model = CRNN_LID().to(device); model.load_state_dict(hist["best_state"])

    nah = [items[i][0] for i in range(len(items)) if items[i][1] == 1.0][:n_per_class]
    spa = [items[i][0] for i in range(len(items)) if items[i][1] == 0.0][:n_per_class]
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "nahuatl": os.path.join(out_dir, f"cam_nahuatl_{band}_{norm}.png"),
        "spanish": os.path.join(out_dir, f"cam_spanish_{band}_{norm}.png"),
    }
    _save(gradcam.aggregated_cam(model, nah, device), f"GradCAM Nahuatl ({band}/{norm}, n={len(nah)})", paths["nahuatl"])
    _save(gradcam.aggregated_cam(model, spa, device), f"GradCAM Spanish ({band}/{norm}, n={len(spa)})", paths["spanish"])
    return paths
