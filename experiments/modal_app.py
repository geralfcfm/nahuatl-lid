from __future__ import annotations

import modal
from experiments.lid import config, run_matrix, results as results_mod

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("torch", "torchaudio", "scikit-learn", "numpy",
                      "matplotlib", "datasets==2.21.0", "soundfile", "librosa")
         .add_local_python_source("experiments"))
app = modal.App("nahuatl-lid-phase1", image=image)
audio_vol = modal.Volume.from_name("lid-audio", create_if_missing=True)
out_vol = modal.Volume.from_name("lid-outputs", create_if_missing=True)
features_vol = modal.Volume.from_name("lid-features", create_if_missing=True)
hf_vol = modal.Volume.from_name("lid-hf-cache", create_if_missing=True)
cv_ncx_vol = modal.Volume.from_name("lid-cv-ncx", create_if_missing=True)
cv_sei_vol = modal.Volume.from_name("lid-cv-sei", create_if_missing=True)
cv_tar_vol = modal.Volume.from_name("lid-cv-tar", create_if_missing=True)
cv_qxp_vol = modal.Volume.from_name("lid-cv-qxp", create_if_missing=True)

@app.function(gpu="T4", timeout=4 * 60 * 60, memory=16384,
              volumes={"/audio": audio_vol, "/outputs": out_vol,
                       "/features": features_vol, "/hf_cache": hf_vol,
                       "/cv-ncx": cv_ncx_vol,
                       "/cv-sei": cv_sei_vol,
                       "/cv-tar": cv_tar_vol,
                       "/cv-qxp": cv_qxp_vol})
def run(smoke: bool = False, rebuild: bool = False, experiment: str = "baseline", contrast: str = "es") -> dict:
    import os
    os.environ.setdefault("HF_HOME", "/hf_cache")  # persist CIEMPIESS download
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from experiments import cache
    from experiments.preprocess import decode_corpus, features_for_band
    if smoke:
        config.K_FOLDS = 2
    epochs = 1 if smoke else config.EPOCHS
    bands = ["wideband_16k", "bandrestrict_4k", "lowpass_4k"]
    CACHE = "/features"
    print(f"PHASE: start (smoke={smoke}, rebuild={rebuild}, experiment={experiment}, contrast={contrast}, k_folds={config.K_FOLDS}, epochs={epochs})", flush=True)

    if experiment in ("cv", "cv_shuffle", "cv_ablate"):
        from experiments import cv_data
        contrast_dir = None if contrast in cv_data.MIRROR_LANGS else f"/cv-{contrast}/{contrast}"
        print(f"PHASE: {experiment} pair (nahuatl vs {contrast}, dir={contrast_dir})", flush=True)
        raw, filenames = cv_data.build_cv_pair("/cv-ncx/ncx", contrast, contrast_dir=contrast_dir)
        os.makedirs("/outputs/figures", exist_ok=True)

        if experiment == "cv_ablate":
            # LSTM-ablation: same conv front-end, BiLSTM replaced by global avg-pool.
            from experiments.lid.model import CNN_LID
            print("PHASE: cv_ablate (CNN_LID, no BiLSTM)", flush=True)
            res = run_matrix.run_bandwise(raw, filenames, bands, device="cuda", epochs=epochs, model_cls=CNN_LID)
            tag, title = f"cv_{contrast}_cnn", f"cv nahuatl-vs-{contrast} (CNN/no-LSTM)"
        elif experiment == "cv_shuffle":
            # time-shuffle on the GENUINE channel-matched task: does temporal order matter?
            from experiments.lid import transforms
            config.seed_everything(config.SEED)
            print("PHASE: cv_shuffle (time-axis permuted)", flush=True)
            items_by_band = {}
            for band in bands:
                bi = features_for_band(raw, band)
                items_by_band[band] = [(transforms.shuffle_time(s), lbl, fn) for s, lbl, fn in bi]
            res = run_matrix.run_matrix(items_by_band, filenames, device="cuda", epochs=epochs)
            tag, title = f"cv_{contrast}_shuffle", f"cv nahuatl-vs-{contrast} (time-shuffled)"
        else:  # plain cv
            res = run_matrix.run_bandwise(raw, filenames, bands, device="cuda", epochs=epochs)
            tag, title = f"cv_{contrast}", f"cv nahuatl-vs-{contrast}"

        results_mod.write_results(res, f"/outputs/results_{tag}.json")
        for s in res["configs"]:
            accs = [f["val_acc"] for f in s["folds"]]
            plt.figure(); plt.plot(range(1, len(accs) + 1), accs, marker="o")
            plt.title(f"{title}: {s['band']}/{s['norm']} mean {s['mean_acc']*100:.2f}%")
            plt.xlabel("fold"); plt.ylabel("best val acc"); plt.grid(True)
            plt.savefig(f"/outputs/figures/{tag}_acc_{s['band']}_{s['norm']}.png"); plt.close()
        out_vol.commit()
        print(f"PHASE: done ({experiment} {contrast}) -> {res['meta']}", flush=True)
        return res["meta"]

    if experiment == "cv_paired":
        # Paired CNN/CRNN comparison: build the CV pair + GroupKFold folds ONCE and
        # train BOTH architectures on those identical folds (see run_matrix.run_bandwise_paired).
        # This does NOT read the lid-features cache -- like cv/cv_shuffle/cv_ablate, it
        # rebuilds the CV pair from source (raw CV audio, not decode_corpus's cached Mels).
        from experiments import cv_data
        contrast_dir = None if contrast in cv_data.MIRROR_LANGS else f"/cv-{contrast}/{contrast}"
        print(f"PHASE: cv_paired pair (nahuatl vs {contrast}, dir={contrast_dir})", flush=True)
        raw, filenames = cv_data.build_cv_pair("/cv-ncx/ncx", contrast, contrast_dir=contrast_dir)
        # Minimum required config: wideband_16k/none is the reported CNN-vs-CRNN comparison.
        paired_configs = [("wideband_16k", "none")]
        res = run_matrix.run_bandwise_paired(raw, filenames, paired_configs, device="cuda", epochs=epochs)
        out_path = f"/outputs/results_cv_{contrast}_paired.json"
        results_mod.write_results(res, out_path)
        out_vol.commit()
        print(f"PHASE: done (cv_paired {contrast}) -> {res['meta']}", flush=True)
        return res["meta"]

    if experiment == "cv_paired_seeds":
        # Seed-variance for the paired CNN/CRNN comparison + per-epoch learning curves.
        # Builds the CV pair + folds ONCE and trains both architectures under 3 seeds on
        # the SAME folds, isolating training non-determinism from data-draw/fold variance.
        from experiments import cv_data
        contrast_dir = None if contrast in cv_data.MIRROR_LANGS else f"/cv-{contrast}/{contrast}"
        print(f"PHASE: cv_paired_seeds pair (nahuatl vs {contrast}, dir={contrast_dir})", flush=True)
        raw, filenames = cv_data.build_cv_pair("/cv-ncx/ncx", contrast, contrast_dir=contrast_dir)
        res = run_matrix.run_paired_seeds(raw, filenames, "wideband_16k", "none",
                                          seeds=[42, 1, 2], device="cuda", epochs=epochs)
        out_path = f"/outputs/results_cv_{contrast}_paired_seeds.json"
        results_mod.write_results(res, out_path)
        out_vol.commit()
        print(f"PHASE: done (cv_paired_seeds {contrast}) -> {res['meta']}", flush=True)
        return res["meta"]

    if experiment == "cv_nested":
        # Nested speaker-disjoint held-out TEST estimate (outer test disjoint; inner val
        # for checkpoint only). Single config wideband_16k/none, CRNN. Scoped sanity check.
        from experiments import cv_data
        contrast_dir = None if contrast in cv_data.MIRROR_LANGS else f"/cv-{contrast}/{contrast}"
        print(f"PHASE: cv_nested pair (nahuatl vs {contrast}, dir={contrast_dir})", flush=True)
        raw, filenames = cv_data.build_cv_pair("/cv-ncx/ncx", contrast, contrast_dir=contrast_dir)
        res = run_matrix.run_nested_cv(raw, filenames, "wideband_16k", "none",
                                       device="cuda", epochs=epochs)
        out_path = f"/outputs/results_cv_{contrast}_nested.json"
        results_mod.write_results(res, out_path)
        out_vol.commit()
        print(f"PHASE: done (cv_nested {contrast}) -> {res['meta']}", flush=True)
        return res["meta"]

    if experiment in ("shuffle", "degrade"):
        if not cache.is_cached(CACHE):
            raise RuntimeError("feature cache missing; run baseline first to populate lid-features")
        config.seed_everything(config.SEED)
        filenames, _ = cache.load_manifest(CACHE)
        print(f"PHASE: {experiment} from cache", flush=True)
        os.makedirs("/outputs/figures", exist_ok=True)
        if experiment == "shuffle":
            from experiments.lid import transforms
            items_by_band = {}
            for band in bands:
                items = cache.load_band_items(CACHE, band)
                items_by_band[band] = [(transforms.shuffle_time(s), lbl, fn) for s, lbl, fn in items]
            res = run_matrix.run_matrix(items_by_band, filenames, device="cuda", epochs=epochs)
        else:  # degrade
            from experiments.lid import transforms
            raw = cache.load_raw_items(CACHE)
            raw_d = [(transforms.degrade_chunk(c), lbl, fn) for c, lbl, fn in raw]
            res = run_matrix.run_bandwise(raw_d, filenames, bands, device="cuda", epochs=epochs)
        results_mod.write_results(res, f"/outputs/results_{experiment}.json")
        for s in res["configs"]:
            accs = [f["val_acc"] for f in s["folds"]]
            plt.figure(); plt.plot(range(1, len(accs) + 1), accs, marker="o")
            plt.title(f"{experiment}: {s['band']}/{s['norm']} mean {s['mean_acc']*100:.2f}%")
            plt.xlabel("fold"); plt.ylabel("best val acc"); plt.grid(True)
            plt.savefig(f"/outputs/figures/{experiment}_acc_{s['band']}_{s['norm']}.png"); plt.close()
        out_vol.commit()
        print(f"PHASE: done ({experiment}) -> {res['meta']}", flush=True)
        return res["meta"]

    if (not rebuild) and (not smoke) and cache.is_cached(CACHE):
        print("PHASE: loading cached features", flush=True)
        filenames, _ = cache.load_manifest(CACHE)
        items_by_band = {b: cache.load_band_items(CACHE, b) for b in bands}
        best_items_for = lambda band: items_by_band[band]
        print("PHASE: training", flush=True)
        res = run_matrix.run_matrix(items_by_band, filenames, device="cuda", epochs=epochs)
    else:
        print("PHASE: decoding corpus", flush=True)
        if smoke:
            raw, filenames = decode_corpus("/audio/nahuatl", max_files=4, max_chunks_per_file=15)
        else:
            raw, filenames = decode_corpus("/audio/nahuatl")
        if not smoke:
            print("PHASE: caching features to volume", flush=True)
            cache.save_corpus(raw, filenames, bands, CACHE, features_for_band)
            features_vol.commit()
            hf_vol.commit()
        best_items_for = lambda band: features_for_band(raw, band)
        print("PHASE: training", flush=True)
        res = run_matrix.run_bandwise(raw, filenames, bands, device="cuda", epochs=epochs)

    print("PHASE: writing results + figures + GradCAM", flush=True)
    os.makedirs("/outputs/figures", exist_ok=True)
    results_mod.write_results(res, "/outputs/results.json")
    for s in res["configs"]:
        accs = [f["val_acc"] for f in s["folds"]]
        plt.figure(); plt.plot(range(1, len(accs) + 1), accs, marker="o")
        plt.title(f"{s['band']} / {s['norm']} — mean {s['mean_acc']*100:.2f}%")
        plt.xlabel("fold"); plt.ylabel("best val acc"); plt.grid(True)
        plt.savefig(f"/outputs/figures/acc_{s['band']}_{s['norm']}.png"); plt.close()
    from experiments.lid.gradcam_figure import render_cams
    bc = res["meta"]["best_config"]
    best_items = best_items_for(bc["band"])
    render_cams(best_items, filenames, bc["band"], bc["norm"],
                device="cuda", out_dir="/outputs/figures",
                n_per_class=(4 if smoke else 100), epochs=epochs)
    out_vol.commit()
    print(f"PHASE: done -> {res['meta']}", flush=True)
    return res["meta"]

@app.local_entrypoint()
def main(smoke: bool = False, rebuild: bool = False, experiment: str = "baseline", contrast: str = "es") -> None:
    # .spawn() (not .remote()) so the run is true fire-and-forget: it completes
    # server-side even if the local caller disconnects. Pair with `modal run --detach`
    # so the app stays alive after this entrypoint returns. Poll lid-outputs/results.json
    # (or `modal app logs`) for completion — do not wait on the call here.
    call = run.spawn(smoke=smoke, rebuild=rebuild, experiment=experiment, contrast=contrast)
    print(f"spawned run: function_call_id={call.object_id}", flush=True)
