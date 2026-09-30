"""Derive the canonical 5-point landmark template used by `preprocessing.face.FacePreprocessorAligned`.

The template is the MEAN landmark position inside the baseline 160x160 bounding-box crops of the face model's own
fine-tuning images (LFW, min_faces_per_person=20, notebook split seed 42, TRAIN split only). Aligning to it keeps the
face framing the InceptionResnetV1 checkpoint was trained on (a generic template such as ArcFace's 112x112 one frames
the face differently), so the controlled variable in the experiment is geometric normalization, not framing.

The mean is then left-right symmetrized (`symmetrize`) so the canonical face has level eyes and a centred nose.

Run: python -m evaluation.ieee.face_alignment_template
Output: evaluation/results/face_alignment_template.json (the constant in preprocessing/face.py cites this file).
"""

from __future__ import annotations

import json

import numpy as np

from evaluation.ieee.common import DATA, RESULTS

ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])


def landmarks_in_crop(image: np.ndarray, detector) -> np.ndarray | None:
    """5 landmarks of the largest face, in the coordinates of the baseline 160x160 box crop (None if no face)."""
    from PIL import Image

    boxes, _, points = detector.detect(Image.fromarray(image), landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None
    boxes = np.asarray(boxes, dtype=np.float64)
    k = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))  # MTCNN.forward(select_largest=True)
    x1, y1, x2, y2 = boxes[k]
    lm = np.asarray(points[k], dtype=np.float64)
    return np.c_[(lm[:, 0] - x1) * 160.0 / (x2 - x1), (lm[:, 1] - y1) * 160.0 / (y2 - y1)]


def symmetrize(mean: np.ndarray, size: float = 160.0) -> np.ndarray:
    """Left-right symmetric template: average each point with the mirror (x -> size - x) of its counterpart
    (left eye <-> right eye, mouth left <-> mouth right, nose <-> itself). Eyes become level and the nose centred."""
    mirrored = np.c_[size - mean[:, 0], mean[:, 1]][[1, 0, 2, 4, 3]]
    return (mean + mirrored) / 2.0


def main():
    from facenet_pytorch import MTCNN
    from sklearn.datasets import fetch_lfw_people

    lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True, funneled=True, data_home=str(DATA / "lfw"))
    images, labels = (lfw.images * 255).astype(np.uint8), lfw.target
    detector = MTCNN(image_size=160, margin=0, post_process=True)
    pts, keep = [], []
    for i, img in enumerate(images):
        p = landmarks_in_crop(img, detector)
        if p is not None:
            pts.append(p)
            keep.append(i)
    pts, keep = np.array(pts), np.array(keep)
    rng = np.random.default_rng(42)  # training split exactly as kaggle_kernels/face_training cell 8
    by: dict = {}
    for j, lab in enumerate(labels[keep]):
        by.setdefault(lab, []).append(j)
    train = []
    for _, idx in by.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        train += list(idx[: max(1, int(len(idx) * 0.7))])
    T = pts[train]
    mean = T.mean(0)
    out = {
        "template_160_symmetric": np.round(symmetrize(mean), 2).tolist(),
        "template_160_mean": np.round(mean, 2).tolist(),
        "sd_160": np.round(T.std(0), 2).tolist(),
        "order": ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"],
        "coordinate_system": "x right, y down, pixels of the 160x160 baseline crop (MTCNN box, margin 0)",
        "n_faces_detected": int(len(keep)), "n_images": int(len(images)), "n_train_crops": int(len(T)),
        "source": "LFW min_faces_per_person=20 (face model fine-tuning set), train split seed 42",
        "arcface_112_scaled_to_160_for_comparison": np.round(ARCFACE_112 * 160 / 112, 2).tolist(),
    }
    (RESULTS / "face_alignment_template.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
