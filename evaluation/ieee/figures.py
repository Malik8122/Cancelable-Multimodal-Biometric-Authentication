"""Part 18: architecture/workflow diagrams (Figs 1-9) and synthetic-calibration figures (Figs 10-11), matplotlib only.

Diagram content is taken from the code (stage, file, dimension); data figures read committed result files.
Run: python -m evaluation.ieee.figures
"""

from __future__ import annotations

import json

import numpy as np

from evaluation.ieee.common import COLORS, CONFIG, COL_W, DBL_W, RESULTS, SYNTHETIC, ieee_style, save_fig, timed
from template_protection.metric_estimation import fitted_hamming_similarity

BOX = {"io": "#eef2f7", "proc": "#dde7f3", "secure": "#fbe3e1", "store": "#e3f1e4", "decide": "#f3e6f7"}


def _flow(ax, boxes, arrows, xlim=(0, 10), ylim=(0, 4)):
    from matplotlib.patches import FancyBboxPatch

    ax.set(xlim=xlim, ylim=ylim)
    ax.axis("off")
    pos = {}
    for key, (x, y, w, h, text, kind) in boxes.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=BOX[kind], ec="#33415c", lw=0.6))
        ax.text(x, y, text, ha="center", va="center", fontsize=5.6, linespacing=1.15)
        pos[key] = (x, y, w, h)
    for a, b, *lab in arrows:
        (xa, ya, wa, ha), (xb, yb, wb, hb) = pos[a], pos[b]
        if abs(ya - yb) < 1e-6:
            start, end = (xa + wa / 2 * np.sign(xb - xa), ya), (xb - wb / 2 * np.sign(xb - xa), yb)
        else:
            start, end = (xa, ya - ha / 2 * np.sign(ya - yb)), (xb, yb + hb / 2 * np.sign(ya - yb))
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": "#33415c", "shrinkA": 0, "shrinkB": 0})
        if lab:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, lab[0], ha="center", fontsize=5, color="#33415c")


def diagrams():
    plt = ieee_style()
    # Fig 1: system architecture
    fig, ax = plt.subplots(figsize=(DBL_W, 2.2))
    _flow(ax, {
        "ui": (1.0, 2.6, 1.7, 1.0, "React frontend\n(camera,\nmicrophone)", "io"),
        "api": (3.1, 2.6, 1.7, 1.0, "FastAPI backend\n/users /enroll\n/authenticate/fusion", "proc"),
        "pipe": (5.2, 2.6, 1.8, 1.0, "Modality pipelines\nface 512-d,\nvoice 192-d (L2)", "proc"),
        "bio": (7.3, 2.6, 1.8, 1.0, "BioHash (HKDF-SHA256 keys)\n256-bit cancelable\ntemplate", "secure"),
        "cmp": (9.2, 2.6, 1.4, 1.0, "Hamming compare\n+ per-modality\ndecision", "decide"),
        "db": (7.3, 0.8, 1.8, 0.9, "SQLite: users, 4 template\nsets/user, audit log\n(no images/embeddings)", "store"),
        "fus": (9.2, 0.8, 1.4, 0.9, "Fusion policy\nALL_REQUIRED\n-> decision + name", "decide"),
    }, [("ui", "api", "HTTPS"), ("api", "pipe"), ("pipe", "bio"), ("bio", "cmp"), ("bio", "db", "enroll"), ("cmp", "fus")], ylim=(0.2, 3.3))
    save_fig(fig, "fig01_system_architecture", CONFIG, "System architecture of the implemented cancelable multimodal authentication system.")

    # Fig 2: enrollment
    fig, ax = plt.subplots(figsize=(DBL_W, 2.4))
    _flow(ax, {
        "name": (0.9, 3.0, 1.5, 0.8, "Display name\nPOST /users\n(server user ID)", "io"),
        "face": (2.9, 3.0, 1.8, 0.8, "Face: 5 guided poses\nquality gates (MTCNN)\n>= 3 valid", "proc"),
        "cent": (5.0, 3.0, 1.7, 0.8, "Centroid of pose\nembeddings\n(L2-normalized)", "proc"),
        "voice": (2.9, 1.3, 1.8, 0.8, "Voice: 2 recordings\ncosine consistency\n>= 0.60 (FAIR opt-in)", "proc"),
        "keys": (7.0, 2.15, 1.8, 0.9, "HKDF keys for 4 sets\n(key_version v..v+3)", "secure"),
        "hash": (8.9, 2.15, 1.6, 0.9, "BioHash x4\n256 bits each", "secure"),
        "db": (8.9, 0.6, 1.6, 0.7, "Store templates only\n(set 1 ACTIVE)", "store"),
    }, [("name", "face"), ("face", "cent"), ("cent", "keys"), ("voice", "keys"), ("keys", "hash"), ("hash", "db")], ylim=(0.1, 3.6))
    save_fig(fig, "fig02_enrollment_workflow", CONFIG, "Enrollment workflow: name, guided face poses, two voice recordings, four keyed template sets.")

    # Fig 3: authentication
    fig, ax = plt.subplots(figsize=(DBL_W, 1.8))
    _flow(ax, {
        "sel": (0.8, 1.6, 1.4, 0.9, "User selects\nenrolled factors", "io"),
        "cap": (2.4, 1.6, 1.4, 0.9, "Capture +\nembed (L2)", "proc"),
        "key": (4.0, 1.6, 1.5, 0.9, "Key of claimed user,\nACTIVE set only", "secure"),
        "hash": (5.6, 1.6, 1.3, 0.9, "BioHash\n256 bits", "secure"),
        "ham": (7.1, 1.6, 1.4, 0.9, "Hamming vs stored\ntemplate", "decide"),
        "dec": (8.7, 1.6, 1.5, 0.9, "face est. cos >= 0.80\nvoice est. dist <= 0.75\nfusion: ALL_REQUIRED", "decide"),
    }, [("sel", "cap"), ("cap", "key"), ("key", "hash"), ("hash", "ham"), ("ham", "dec")], ylim=(0.9, 2.3))
    save_fig(fig, "fig03_authentication_workflow", CONFIG, "Authentication workflow; the stored template is compared only by Hamming similarity.")

    for name, caption, steps in (
        ("fig04_face_pipeline", "Face pipeline as implemented.", [
            ("RGB image", "io"), ("MTCNN detection\n160x160 crop\n(post_process)", "proc"), ("InceptionResnetV1\n(VGGFace2, LFW-tuned)", "proc"),
            ("512-d embedding\nL2-normalized", "proc"), ("BioHash\n256 bits", "secure")]),
        ("fig05_voice_pipeline", "Voice pipeline as implemented.", [
            ("WAV", "io"), ("16 kHz mono, energy VAD,\nRMS norm, 4.0 s\ncentre segment", "proc"), ("80-bin log-mel\nn_fft 400, hop 160", "proc"),
            ("ECAPA-TDNN\n192-d, L2", "proc"), ("BioHash\n256 bits", "secure")]),
        ("fig07_biohash_generation", "BioHash generation (template_protection/): three HKDF-SHA256 seeds drive projection, quantization and permutation.", [
            (r"$\hat{x} = x/\|x\|_2$", "proc"), (r"$p = W\hat{x},\ W \in \mathbb{R}^{256\times D}$" + "\nHaar orthonormal rows\n(projection seed)", "secure"),
            (r"$b_i = [\,p_i > t_i\,]$" + "\n" + r"$t_i \sim \mathcal{N}(0,\,(0.5\,\sigma_p)^2)$" + "\n(threshold seed)", "secure"), ("keyed permutation\n(permutation seed)", "secure"),
            ("256-bit template\n(32 bytes)", "store")]),
    ):
        fig, ax = plt.subplots(figsize=(DBL_W, 1.2))
        n = len(steps)
        boxes = {f"s{k}": (1.0 + k * 2.0, 0.6, 1.75, 0.85, t, kind) for k, (t, kind) in enumerate(steps)}
        _flow(ax, boxes, [(f"s{k}", f"s{k + 1}") for k in range(n - 1)], ylim=(0.1, 1.1))
        save_fig(fig, name, CONFIG, caption)

    # Fig 8: template lifecycle
    fig, ax = plt.subplots(figsize=(DBL_W, 1.6))
    _flow(ax, {
        "s1": (1.0, 1.2, 1.5, 0.8, "Set 1 ACTIVE\nSets 2-4 STANDBY", "store"),
        "rev": (3.2, 1.2, 1.5, 0.8, "Revoke: set 1 ->\nREVOKED (all modalities)", "secure"),
        "pro": (5.4, 1.2, 1.5, 0.8, "Promote oldest\nSTANDBY -> ACTIVE", "store"),
        "exh": (7.6, 1.2, 1.5, 0.8, "Pool exhausted\n(HTTP 409)", "decide"),
        "re": (9.4, 1.2, 1.1, 0.8, "Re-enroll:\nnew key\nversions", "proc"),
    }, [("s1", "rev"), ("rev", "pro"), ("pro", "exh", "after 3"), ("exh", "re")], ylim=(0.6, 1.8))
    save_fig(fig, "fig08_template_lifecycle", CONFIG, "Template-set lifecycle: whole multimodal sets are revoked and promoted together.")

    # Fig 9: fusion
    fig, ax = plt.subplots(figsize=(DBL_W, 1.9))
    _flow(ax, {
        "f": (1.2, 2.2, 2.0, 0.8, "Face: estimated cosine ĉ\nmatch if ĉ >= 0.80", "decide"),
        "v": (1.2, 0.9, 2.0, 0.8, "Voice: estimated distance d̂\nmatch if d̂ <= 0.75", "decide"),
        "sf": (4.0, 2.2, 1.9, 0.8, "s_face = ĉ\n(t = 0.80)", "proc"),
        "sv": (4.0, 0.9, 1.9, 0.8, "s_voice = 1 - d̂²/2\n(t = 0.71875)", "proc"),
        "fu": (6.7, 1.55, 1.9, 1.0, "S = mean(s)\nT = mean(t)\n(equal weights)", "proc"),
        "po": (9.0, 1.55, 1.8, 1.2, "ALL_REQUIRED: all match\nWEIGHTED: S >= T, s >= 0", "decide"),
    }, [("f", "sf"), ("v", "sv"), ("sf", "fu"), ("sv", "fu"), ("fu", "po")], ylim=(0.3, 2.8))
    save_fig(fig, "fig09_fusion_architecture", CONFIG, "Score conversion and fusion: every modality enters fusion on one higher-is-better scale.")


def calibration_figures():
    plt = ieee_style()
    cal = json.loads((RESULTS / "biohash_metric_calibration.json").read_text())
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for m in ("face", "voice"):
        e = cal["modalities"][m]
        c = np.array(e["cosine"])
        ax.errorbar(c, e["hamming_similarity_mean"], yerr=e["hamming_similarity_std"], fmt="o", ms=1.8, lw=0.5, capsize=0,
                    color=COLORS[m], alpha=0.7, label=f"{m} (D={e['embedding_dim']}): mean ± SD")
        dense = np.linspace(-0.2, 1, 300)
        ax.plot(dense, fitted_hamming_similarity(dense, e["coefficients"]), color=COLORS[m], lw=0.8)
    ax.axhline(0.8176, color="k", ls=":", lw=0.6)
    ax.axvline(0.80, color="k", ls=":", lw=0.6)
    ax.text(0.02, 0.83, "face rule: ĉ >= 0.80 (h >= 0.818)", fontsize=5.5)
    ax.set(xlabel="embedding cosine similarity", ylabel="template Hamming similarity (256 bits)", title="Calibration (synthetic pairs)")
    ax.grid(True)
    ax.legend(fontsize=6, loc="upper left")
    save_fig(fig, "fig10_calibration_curve", SYNTHETIC,
             "Offline calibration on 19,600 synthetic pairs per modality through the real BioHash: mean ± SD and fitted cubic-in-angle model.")
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.7, 1.9))
    for m in ("face", "voice"):
        e = cal["modalities"][m]
        c = np.array(e["cosine"])
        res = fitted_hamming_similarity(c, e["coefficients"]) - np.array(e["hamming_similarity_mean"])
        axes[0].plot(c, res, "o-", ms=1.5, lw=0.6, color=COLORS[m], label=m)
        axes[1].plot(c, e["cosine_estimate_std"], lw=0.8, color=COLORS[m], label=m)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set(xlabel="embedding cosine", ylabel="fit − mean (Hamming sim.)", title="Fit residuals")
    axes[1].set(xlabel="embedding cosine", ylabel="SD of single estimate", title="Estimate uncertainty")
    for ax in axes:
        ax.grid(True)
    axes[1].legend(fontsize=6)
    save_fig(fig, "fig11_calibration_residuals", SYNTHETIC, "Synthetic calibration: model residuals and single-estimate standard deviation vs cosine.")


if __name__ == "__main__":
    with timed("part18_diagrams"):
        diagrams()
        calibration_figures()
