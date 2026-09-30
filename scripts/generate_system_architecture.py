"""System architecture figure (vector PDF + SVG + PNG) for the IEEE paper.

Every label reflects the CURRENT implementation (checked against the code on 2026-09-25):
face = MTCNN + bounding-box crop 160x160 (landmark alignment implemented but off by default), InceptionResnetV1 512-D;
voice = 16 kHz, silence trim, RMS normalisation, 4 s, 80 log-mel, ECAPA-TDNN 192-D; BioHash = L2 -> keyed orthonormal projection -> keyed quantisation
-> keyed permutation -> 256 bits; keys = HKDF-SHA256; 4 template sets per enrollment; ALL_REQUIRED fusion; rotation =
compare-and-swap, behind TEMPLATE_ROTATION_ON_FAILED_AUTH (off by default).

Run: python -m scripts.generate_system_architecture
Output: docs/system_architecture.pdf (+ .svg, .png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "system_architecture"

# muted palette
FACE, VOICE = "#2f5d9e", "#1f8a86"
PROT, GRANT, DENY, ROT = "#3d3d3d", "#2e7d32", "#c62828", "#d9822b"
INK, SUB = "#1f2328", "#57606a"
CLIENT_BG, SERVER_BG, DB_BG = "#f1f2f4", "#eaf2fb", "#fdf6dc"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})

W, H = 100.0, 64.0
fig = plt.figure(figsize=(11.0, 7.04))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text="", ec=INK, fc="white", lw=0.9, ls="-", size=7, weight="normal", color=INK, align="center",
        radius=0.6, pad_x=0.8, z=3, va="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", ec=ec, fc=fc, lw=lw,
                                ls=ls, zorder=z))
    if text:
        tx = x + w / 2 if align == "center" else x + pad_x
        ty = y + h / 2 if va == "center" else y + h - 0.9
        ax.text(tx, ty, text, ha=align, va=va, fontsize=size, fontweight=weight, color=color, zorder=z + 1,
                linespacing=1.25)


def label(x, y, text, size=7, weight="normal", color=INK, ha="left", va="center", z=5, style="normal", rotation=0):
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, style=style,
                   linespacing=1.25, rotation=rotation)


def arrow(p, q, color=INK, lw=1.0, ls="-", style="-|>", conn="arc3", z=2, head=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=head, color=color, lw=lw, ls=ls,
                                 connectionstyle=conn, zorder=z, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ security-boundary shading
ax.add_patch(Rectangle((0.4, 14.0), 16.2, 49.6, fc=CLIENT_BG, ec="none", zorder=0))
ax.add_patch(Rectangle((17.0, 14.0), 82.6, 49.6, fc=SERVER_BG, ec="none", zorder=0))
ax.add_patch(Rectangle((76.2, 14.8), 23.2, 27.4, fc=DB_BG, ec="#e3d49a", lw=0.6, zorder=0.5))

# band headers
for x, t in ((0.9, "1  INPUT & CLIENT"), (17.6, "2  API & ORCHESTRATION"), (34.3, "3  MODALITY PROCESSING"),
             (57.2, "4  MATCHING & DECISION"), (76.6, "5  SECURITY RESPONSE & STORAGE")):
    label(x, 62.5, t, size=7.4, weight="bold", color=INK)
label(0.9, 60.9, "CLIENT — capture and display only", size=6.2, color=SUB, style="italic")
label(17.6, 60.9, "SERVER — decisions, thresholds, rotation, keys", size=6.2, color=SUB, style="italic")
label(76.9, 40.9, "DATABASE — templates and metadata only", size=6.2, color="#7a6414", style="italic")

# ------------------------------------------------------------------ 1  input & client
box(4.0, 55.6, 9.0, 3.3, "User", weight="bold")
for i, (t, c) in enumerate((("Camera\nface frame (PNG)", FACE), ("Microphone\nvoice WAV", VOICE))):
    y = 49.2 - i * 4.9
    box(1.6, y, 13.8, 4.0, t, ec=c, lw=1.2, size=6.8)
    arrow((8.5, 55.6 if i == 0 else y + 4.9), (8.5, y + 4.0), color=SUB, lw=0.8, head=5) if i == 0 else None
arrow((8.5, 55.6), (8.5, 53.2), color=SUB, lw=0.8, head=5)
box(1.6, 26.0, 13.8, 12.4, "", ec=INK, lw=1.0)
label(8.5, 36.9, "Browser UI (React)", size=7.2, weight="bold", ha="center")
for j, t in enumerate(("Guided 5-capture face enrollment", "Voice recorder (2 clips)", "Result page")):
    label(2.5, 34.6 - j * 2.1, "•  " + t, size=6.3)
arrow((8.5, 44.3), (8.5, 38.4), color=SUB, lw=0.8, head=5)
for yy in (48.2, 43.3):
    pass
box(1.6, 16.6, 13.8, 7.6, "", ec=INK, lw=1.0)
label(8.5, 22.6, "Result UI", size=7.2, weight="bold", ha="center")
label(8.5, 19.5, "decision · scores\nT1 → T2 rotation panel\n(renders backend result)", size=6.1, ha="center",
      color=SUB)

# client -> server
ax.plot([15.4, 16.8, 16.8], [34.0, 34.0, 52.4], color=INK, lw=1.1, zorder=2, solid_capstyle="butt")
arrow((16.8, 52.4), (17.9, 52.4), color=INK, lw=1.1, head=5)
label(16.25, 43.2, "HTTP(S) multipart", size=5.9, color=SUB, ha="center", z=6, rotation=90)

# ------------------------------------------------------------------ 2  API & orchestration
box(17.9, 45.6, 14.6, 13.6, "", ec=INK, lw=1.0)
label(25.2, 57.8, "FastAPI backend", size=7.2, weight="bold", ha="center")
tags = ["/enroll", "/enroll/face/check-pose", "/authenticate/fusion", "/revoke-template", "/templates", "/audit"]
for k, t in enumerate(tags):
    box(18.7, 55.0 - k * 1.55, 13.0, 1.25, t, ec="#9aa4af", fc="#f7f9fb", lw=0.6, size=6.0, radius=0.35,
        color=INK)
arrow((25.2, 45.6), (25.2, 43.4), color=INK, lw=1.0)
box(17.9, 22.2, 14.6, 21.2, "", ec=INK, lw=1.0)
label(25.2, 42.0, "Authentication service", size=7.2, weight="bold", ha="center")
steps = [("1", "Enrollment check\n409 if not enrolled"), ("2", "One snapshot of the\ntemplate pool"),
         ("3", "Per-modality processing")]
for k, (n, t) in enumerate(steps):
    y = 36.0 - k * 5.4
    box(18.8, y, 12.8, 4.4, "", ec="#9aa4af", fc="white", lw=0.7, radius=0.4)
    label(19.7, y + 2.2, n, size=7.0, weight="bold")
    label(21.2, y + 2.2, t, size=6.3)
    if k < 2:
        arrow((25.2, y), (25.2, y - 1.0), color=SUB, lw=0.8, head=5)

# ------------------------------------------------------------------ 3  modality lanes
lanes = [
    ("FACE", FACE, "MTCNN detection → bounding-box crop 160×160\n→ InceptionResnetV1 → 512-D → L2 normalisation",
     "landmark alignment implemented, off by default"),
    ("VOICE", VOICE, "16 kHz → silence trim → RMS norm. → 4 s\n→ 80 log-mel → ECAPA-TDNN → 192-D → L2 norm.", None),
]
for i, (name, c, flow, note) in enumerate(lanes):
    y = 52.6 - i * 7.6
    box(34.3, y, 21.6, 6.6, "", ec=c, lw=1.3)
    ax.add_patch(Rectangle((34.3, y), 1.1, 6.6, fc=c, ec="none", zorder=4))
    label(36.2, y + 5.3, name, size=6.9, weight="bold", color=c)
    label(36.2, y + 2.6 if note is None else y + 3.0, flow, size=6.2)
    if note:
        label(36.2, y + 0.75, note, size=5.7, color=SUB, style="italic")
    arrow((33.4, y + 3.3), (34.3, y + 3.3), color=c, lw=1.0, head=5)
ax.plot([31.6, 33.4, 33.4], [27.0, 27.0, 55.9], color=INK, lw=1.0, zorder=2)

# merge into template layer
arrow((45.1, 45.0), (45.1, 33.4), color=PROT, lw=1.2)
label(44.5, 39.2, "embedding, in memory only", size=5.9, color=SUB, style="italic", ha="right")

# ------------------------------------------------------------------ cancelable template layer
box(34.3, 16.6, 21.6, 16.8, "", ec=PROT, fc="#f4f4f4", lw=1.6)
label(45.1, 32.1, "CANCELABLE TEMPLATE LAYER", size=7.2, weight="bold", ha="center", color=PROT)
label(45.1, 30.6, "revocable biometric transformation (not a plain hash)", size=5.9, ha="center", color=SUB,
      style="italic")
box(35.3, 25.4, 19.6, 4.3, "HKDF-SHA256 key\nmaster secret + app + user + modality + key version", ec=PROT, lw=0.9,
    size=6.2)
arrow((45.1, 25.4), (45.1, 24.4), color=PROT, lw=0.9, head=5)
box(35.3, 20.1, 19.6, 4.3, "BioHash: keyed orthonormal projection\n→ keyed quantisation → keyed permutation",
    ec=PROT, lw=0.9, size=6.2)
arrow((45.1, 20.1), (45.1, 19.1), color=PROT, lw=0.9, head=5)
box(38.6, 17.2, 13.0, 1.9, "256-bit template (32 bytes)", ec=PROT, fc=PROT, color="white", size=6.4, weight="bold",
    radius=0.4)

# ------------------------------------------------------------------ 4  matching & decision
box(57.2, 54.2, 17.8, 4.6, "Hamming similarity\nvs ACTIVE set Tn", ec=PROT, lw=1.1, size=6.6, weight="bold")
arrow((55.9, 24.4), (57.2, 56.5), color=PROT, lw=1.1, conn="angle3,angleA=0,angleB=90")
for k, (t, c) in enumerate((("Face: est. cosine ≥ 0.80", FACE), ("Voice: est. distance ≤ 0.75", VOICE))):
    y = 50.3 - k * 2.75
    box(57.8, y, 16.6, 2.2, t, ec=c, lw=1.1, size=6.3, radius=0.4)
arrow((66.1, 54.2), (66.1, 52.5), color=INK, lw=0.9, head=5)
arrow((66.1, 47.55), (66.1, 43.6), color=INK, lw=0.9, head=5)
box(57.2, 38.6, 17.8, 5.0, "Fusion policy: ALL_REQUIRED\nevery submitted factor must pass", ec=INK, lw=1.1,
    size=6.5, weight="bold")
box(57.2, 31.0, 8.4, 4.6, "ACCESS\nGRANTED", ec=GRANT, fc=GRANT, color="white", size=6.8, weight="bold")
box(66.6, 31.0, 8.4, 4.6, "ACCESS\nDENIED", ec=DENY, fc=DENY, color="white", size=6.8, weight="bold")
arrow((63.0, 38.6), (61.4, 35.6), color=GRANT, lw=1.1)
arrow((69.2, 38.6), (70.8, 35.6), color=DENY, lw=1.1)
box(60.2, 20.4, 11.8, 5.6, "Audit log\ndecision · failed factors\nset before/after · rotation status", ec=INK, lw=1.0,
    size=6.1)
arrow((61.4, 31.0), (64.0, 26.0), color=GRANT, lw=1.0)
arrow((70.8, 31.0), (68.2, 26.0), color=DENY, lw=1.0)

# ------------------------------------------------------------------ 5  security response & storage
box(77.0, 45.4, 22.2, 13.6, "", ec=ROT, fc="#fdf1e6", lw=1.6)
label(88.1, 57.6, "TEMPLATE ROTATION", size=7.2, weight="bold", ha="center", color=ROT)
label(88.1, 56.1, "compare-and-swap, one commit (when enabled)", size=5.9, ha="center", color=SUB, style="italic")
for k, t in enumerate(("Tn: ACTIVE → REVOKED", "Tn+1: STANDBY → ACTIVE", "all enrolled modalities together",
                       "ALREADY_ROTATED · POOL_EXHAUSTED handled")):
    label(78.2, 54.0 - k * 2.05, "•  " + t, size=6.3, weight="bold" if k < 2 else "normal")
arrow((75.0, 33.3), (77.0, 49.0), color=ROT, lw=1.6, conn="angle3,angleA=0,angleB=90", head=7)
label(77.0, 43.8, "on DENIED", size=6.0, color=ROT, weight="bold", ha="left")

# template pool grid (inside the database region)
arrow((97.2, 45.4), (97.2, 38.9), color=ROT, lw=1.3, head=6)
label(96.6, 43.7, "switches the ACTIVE set", size=5.9, color=ROT, style="italic", ha="right")
gx, gy, cw, ch = 84.4, 30.2, 3.6, 2.0
for j, (t, st) in enumerate((("T1", "ACTIVE"), ("T2", "STANDBY"), ("T3", "STANDBY"), ("T4", "STANDBY"))):
    label(gx + j * cw + cw / 2, gy + 3 * ch + 2.1, t, size=6.6, weight="bold", ha="center")
    label(gx + j * cw + cw / 2, gy + 3 * ch + 0.9, st, size=5.3, ha="center", color=INK if j == 0 else SUB)
for i, (m, c) in enumerate((("Face", FACE), ("Voice", VOICE))):
    y = gy + (2 - i) * ch
    label(gx - 0.5, y + ch / 2, m, size=6.2, ha="right", color=c, weight="bold")
    for j in range(4):
        ax.add_patch(Rectangle((gx + j * cw + 0.25, y + 0.2), cw - 0.5, ch - 0.4, fc="white" if j else "#ffffff",
                               ec=c, lw=1.3 if j == 0 else 0.8, ls="-" if j == 0 else (0, (2.5, 1.6)), zorder=3))
        label(gx + j * cw + cw / 2, y + ch / 2, f"v{j + 1}", size=5.6, ha="center", color=INK if j == 0 else SUB)
label(88.9, 28.9, "4 sets created at enrollment, each under its own key version\n(rows only for enrolled modalities)",
      size=5.6, ha="center", color=SUB, style="italic")

# database cylinder
cx, cy, cwid, chgt = 79.0, 16.2, 19.4, 9.0
ax.add_patch(Rectangle((cx, cy), cwid, chgt, fc="#fff9e6", ec="none", zorder=2))
ax.plot([cx, cx], [cy, cy + chgt], color="#a88a2a", lw=1.0, zorder=3)
ax.plot([cx + cwid, cx + cwid], [cy, cy + chgt], color="#a88a2a", lw=1.0, zorder=3)
ax.add_patch(Ellipse((cx + cwid / 2, cy), cwid, 1.6, fc="#fff9e6", ec="#a88a2a", lw=1.0, zorder=2.5))
ax.add_patch(Rectangle((cx, cy), cwid, 0.02, fc="#fff9e6", ec="none", zorder=2.6))
ax.add_patch(Ellipse((cx + cwid / 2, cy + chgt), cwid, 1.6, fc="#fdf0c4", ec="#a88a2a", lw=1.0, zorder=3))
label(cx + cwid / 2, cy + chgt - 1.7, "SQLite", size=7.0, weight="bold", ha="center")
label(cx + cwid / 2, cy + 3.9, "protected templates (32 B each)\n+ key/set metadata + audit log\n"
      "no images, audio or embeddings\nunique index: one ACTIVE row per user & modality", size=5.9, ha="center")
arrow((72.0, 23.2), (79.0, 21.5), color=INK, lw=0.9, head=5)

# response back to the client (along the bottom of the main area)
arrow((66.1, 20.4), (66.1, 15.0), color=SUB, lw=0.9, style="-", head=5)
arrow((66.1, 15.0), (15.4, 18.6), color=SUB, lw=0.9, conn="angle3,angleA=180,angleB=90", head=5)
label(40.0, 14.35, "JSON response: decision, fused score, rotation (Tn → Tn+1)", size=6.0, ha="center", color=SUB)

# ------------------------------------------------------------------ enrollment strip
box(0.4, 0.5, 99.2, 12.6, "", ec="#c4c9cf", fc="white", lw=0.8, radius=0.4, z=1)
label(1.2, 11.8, "ENROLLMENT", size=7.4, weight="bold")
rows = [
    ("Face", FACE, ["5 captures", "quality gates\n(check-pose)", "≥ 3 valid", "centroid\n(mean, L2)"]),
    ("Voice", VOICE, ["2 recordings", "consistency gate\n(cosine ≥ 0.75)", "first recording"]),
]
for i, (m, c, steps_) in enumerate(rows):
    y = 7.7 - i * 3.35
    label(9.6, y + 1.15, m, size=6.4, weight="bold", color=c, ha="right")
    for k, s in enumerate(steps_):
        x = 10.4 + k * 9.6
        box(x, y, 8.4, 2.35, s, ec=c, lw=0.9, size=5.8, radius=0.35)
        if k:
            arrow((x - 1.2, y + 1.17), (x, y + 1.17), color=c, lw=0.8, head=4)
    last = 10.4 + (len(steps_) - 1) * 9.6 + 8.4
    arrow((last, y + 1.17), (49.0, 5.6), color=c, lw=0.8, head=4, conn="arc3,rad=0.0")
box(49.0, 3.8, 11.6, 3.6, "BioHash under\n4 key versions", ec=PROT, lw=1.2, size=6.4, weight="bold")
arrow((60.6, 5.6), (62.2, 5.6), color=PROT, lw=1.0, head=5)
box(62.2, 3.8, 10.2, 3.6, "T1 ACTIVE\nT2–T4 STANDBY", ec=PROT, lw=1.0, size=6.3)
arrow((72.4, 5.6), (74.0, 5.6), color=PROT, lw=1.0, head=5)
box(74.0, 3.8, 5.6, 3.6, "Database", ec="#a88a2a", fc="#fff9e6", lw=1.0, size=6.3)

# legend
lx, ly = 81.0, 11.3
label(lx, ly, "Legend", size=6.6, weight="bold")
items = [(FACE, "Face"), (VOICE, "Voice"), (PROT, "Template protection"),
         (GRANT, "Access granted"), (DENY, "Access denied"), (ROT, "Rotation")]
for k, (c, t) in enumerate(items):
    col, row = k % 2, k // 2
    x, y = lx + col * 9.4, ly - 1.7 - row * 1.55
    ax.add_patch(Rectangle((x, y - 0.45), 1.4, 0.9, fc=c, ec="none", zorder=3))
    label(x + 1.9, y, t, size=5.9)
y0 = ly - 1.7 - 4 * 1.55
for k, (ls, fc, ec, t) in enumerate(((("-"), "white", INK, "ACTIVE"), ((0, (2.5, 1.6)), "white", INK, "STANDBY"),
                                     ("-", "#d0d3d7", "#9aa4af", "REVOKED"))):
    x = lx + k * 6.2
    ax.add_patch(Rectangle((x, y0 - 0.45), 1.4, 0.9, fc=fc, ec=ec, lw=0.9, ls=ls, zorder=3))
    label(x + 1.9, y0, t, size=5.9)

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None, facecolor="white")
print("written:", *(OUT.with_suffix(f".{e}") for e in ("pdf", "svg", "png")))
