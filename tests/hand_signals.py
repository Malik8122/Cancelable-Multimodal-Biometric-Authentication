"""Synthetic MediaPipe-Hands-like Z-gesture captures for tests (SYNTHETIC - never evidence of biometric performance).

A "person" has a characteristic way of drawing the Z (proportions, slant, bow of each stroke, how each stroke speeds
up and slows down, how long each stroke takes, pauses at the corners, hand orientation, finger geometry). One
execution adds natural execution-to-execution variation: overall speed (e.g. 1.6 s vs 3.1 s), Z size, position in the
frame, distance to the camera, a few degrees of rotation, slightly different proportions and stroke timing, and
landmark jitter. The output has exactly the wire format the browser sends: raw (un-mirrored) MediaPipe image
coordinates with `"mirrored": true`, because the capture preview is a selfie view - the user draws the Z as they see it
on screen, so in raw camera coordinates it is horizontally flipped.
"""

from __future__ import annotations

import numpy as np

# A canonical right hand in "palm units" (wrist at origin, wrist -> middle MCP = 1, pointing up the image = -y).
_BASE_HAND = np.array([
    [0.00, 0.00], [-0.35, -0.20], [-0.55, -0.45], [-0.70, -0.70], [-0.82, -0.92],    # wrist, thumb 1-4
    [-0.30, -0.95], [-0.34, -1.35], [-0.36, -1.62], [-0.38, -1.85],                   # index 5-8
    [0.00, -1.00], [0.00, -1.45], [0.00, -1.75], [0.00, -2.00],                       # middle 9-12
    [0.25, -0.95], [0.28, -1.35], [0.30, -1.60], [0.31, -1.82],                       # ring 13-16
    [0.45, -0.85], [0.52, -1.15], [0.56, -1.35], [0.58, -1.52],                       # pinky 17-20
])


def _ease(u: np.ndarray, sharpness: float, bias: float) -> np.ndarray:
    """Progress along one stroke: slow start / fast middle / slow end (`sharpness`), skewed early or late (`bias`)."""
    u = np.clip(u, 0.0, 1.0) ** bias
    a, b = u ** sharpness, (1 - u) ** sharpness
    return a / np.maximum(a + b, 1e-12)


class Person:
    """One synthetic user's Z-drawing style (seeded)."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.width = rng.uniform(2.2, 4.0)                  # Z width, palm sizes
        self.aspect = rng.uniform(0.6, 1.3)                 # height / width
        self.slant = rng.uniform(-0.25, 0.25)               # shear of the whole Z
        self.top_ratio = rng.uniform(0.75, 1.15)            # top stroke length / width
        self.bottom_ratio = rng.uniform(0.75, 1.2)          # bottom stroke length / width
        self.bows = rng.uniform(-0.12, 0.12, size=3)        # sideways bow of each stroke (fraction of its length)
        self.stroke_time = rng.dirichlet([6, 7, 6])         # share of the drawing time per stroke
        self.pauses = rng.uniform(0.0, 0.18, size=2)        # seconds held at the two corners
        self.sharpness = rng.uniform(1.2, 2.6, size=3)      # speed profile of each stroke
        self.bias = rng.uniform(0.75, 1.35, size=3)
        self.tilt = rng.uniform(-0.08, 0.08)                # habitual rotation of the whole Z (radians)
        self.finger_scale = rng.uniform(0.85, 1.15, size=5)
        self.curl = rng.uniform(0.0, 0.35)
        self.hand_tilt = rng.uniform(-0.4, 0.4)
        self.wrist_swing = rng.uniform(-0.25, 0.25)         # hand rotation that follows the stroke direction

    def path(self, rng: np.random.Generator, duration: float, fps: float,
             profile_jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(times_s, palm-centre path in user-view palm units, stroke index per frame) of one execution.

        `profile_jitter` > 0 adds NON-UNIFORM tempo variation, as real people show when they draw faster or slower on
        another attempt: each stroke's speed profile, its share of the time and the corner pauses change by up to that
        fraction (pauses by up to twice it). With 0 (default) a slower execution is a uniform stretch of a faster one
        and no extra random numbers are drawn (existing seeds reproduce exactly)."""
        width = self.width * rng.uniform(0.94, 1.06)
        height = width * self.aspect * rng.uniform(0.93, 1.07)
        top = width * self.top_ratio * rng.uniform(0.95, 1.05)
        bottom = width * self.bottom_ratio * rng.uniform(0.95, 1.05)
        corners = np.array([[0.0, 0.0], [top, 0.0], [0.0, height], [bottom, height]])
        corners[:, 0] -= self.slant * corners[:, 1]
        corners += rng.normal(0, 0.04 * width, corners.shape)
        pauses = np.maximum(self.pauses * rng.uniform(0.6, 1.4, 2), 0.0)
        shares = self.stroke_time * rng.uniform(0.9, 1.1, 3)
        sharpness, bias = self.sharpness, self.bias
        if profile_jitter > 0:
            j = profile_jitter
            shares = shares * rng.uniform(1 - j, 1 + j, 3)
            pauses = pauses * rng.uniform(max(0.0, 1 - 2 * j), 1 + 2 * j, 2)
            sharpness = sharpness * rng.uniform(1 - j, 1 + j, 3)
            bias = bias * rng.uniform(1 - j / 2, 1 + j / 2, 3)
        moving = max(duration - pauses.sum(), 0.3)
        stroke_s = moving * shares / shares.sum()
        # Timeline: stroke 1, pause, stroke 2, pause, stroke 3.
        edges = np.cumsum([0, stroke_s[0], pauses[0], stroke_s[1], pauses[1], stroke_s[2]])
        times = np.arange(0.0, edges[-1] + 1e-9, 1.0 / fps)
        points, strokes = [], []
        bows = self.bows + rng.normal(0, 0.02, 3)
        for t in times:
            segment = min(int(np.searchsorted(edges, t, side="right")) - 1, 4)
            if segment % 2 == 1:                       # holding still at a corner
                points.append(corners[(segment + 1) // 2])
                strokes.append(segment // 2)
                continue
            s = segment // 2
            u = (t - edges[segment]) / max(edges[segment + 1] - edges[segment], 1e-9)
            e = _ease(np.array(u), sharpness[s], bias[s])
            a, b = corners[s], corners[s + 1]
            chord = b - a
            normal = np.array([-chord[1], chord[0]])
            points.append(a + chord * e + normal * bows[s] * np.sin(np.pi * e))
            strokes.append(s)
        path = np.asarray(points)
        angle = self.tilt + rng.uniform(-0.1, 0.1)       # a few degrees of rotation between executions
        c, s = np.cos(angle), np.sin(angle)
        path = (path - path.mean(axis=0)) @ np.array([[c, -s], [s, c]]).T
        return times, path, np.asarray(strokes)

    def capture(self, seed: int, duration_s: float | None = None, fps: float = 30.0, dropout: float = 0.0,
                image=(640, 480), extra: dict | None = None, profile_jitter: float = 0.0,
                finger_extension: tuple | None = None, pose_jitter: float = 0.0) -> dict:
        """One execution. `finger_extension` (5 factors, thumb..pinky; 1 = this person's normal) holds the fingers
        differently for this execution - e.g. POINTING = index out, others curled - with the SAME Z movement.
        `pose_jitter` > 0 varies each finger's extension by up to that fraction (natural hand-pose variation)."""
        rng = np.random.default_rng(seed)
        duration = duration_s if duration_s is not None else rng.uniform(1.5, 3.1)
        times, path, strokes = self.path(rng, duration, fps, profile_jitter)
        palm_px = rng.uniform(0.08, 0.13) * image[1]                       # distance to the camera varies
        size = rng.uniform(0.75, 1.3)                                       # a bigger or smaller Z
        span = path.max(axis=0) - path.min(axis=0)
        size = min(size, 0.75 * image[0] / max(span[0] * palm_px, 1e-9), 0.7 * image[1] / max(span[1] * palm_px, 1e-9))
        centre = np.array([rng.uniform(0.38, 0.62) * image[0], rng.uniform(0.42, 0.58) * image[1]])
        hand = _BASE_HAND.copy()
        extension = np.ones(5) if finger_extension is None else np.asarray(finger_extension, dtype=float)
        if pose_jitter > 0:
            extension = extension * rng.uniform(1 - pose_jitter, 1 + pose_jitter, 5)
        for f, tips in enumerate(((2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20))):
            for tip in tips:
                hand[tip] = hand[tip - 1] + (hand[tip] - hand[tip - 1]) * self.finger_scale[f] * (1 - self.curl * 0.3) * extension[f]
        palm_offset = hand[[0, 5, 9, 13, 17]].mean(axis=0)
        frames = []
        for k, t in enumerate(times):
            stamp = round(1000.0 * t + rng.uniform(-3, 3) * (k > 0), 3)
            if rng.random() < dropout:
                frames.append({"t": stamp, "landmarks": None})
                continue
            angle = self.hand_tilt + self.wrist_swing * (strokes[k] - 1) + rng.normal(0, 0.02)
            ca, sa = np.cos(angle), np.sin(angle)
            shape = (hand - palm_offset) @ np.array([[ca, -sa], [sa, ca]]).T
            user_view = centre + (path[k] * size + rng.normal(0, 0.02, 2)) * palm_px + shape * palm_px \
                + rng.normal(0, 0.6, (21, 2))
            x_raw = image[0] - user_view[:, 0]                              # the selfie preview is mirrored
            z = np.linalg.norm(shape, axis=1) * -0.02 + rng.normal(0, 0.003, 21)
            z[0] = 0.0
            lm = np.column_stack([x_raw / image[0], user_view[:, 1] / image[1], z])
            frames.append({"t": stamp, "landmarks": lm.round(5).tolist(), "handedness": "Right"})
        # Keep timestamps strictly increasing despite the jitter.
        for k in range(1, len(frames)):
            if frames[k]["t"] <= frames[k - 1]["t"]:
                frames[k]["t"] = round(frames[k - 1]["t"] + 1.0, 3)
        capture = {"gesture_type": "z", "mirrored": True, "image_width": image[0], "image_height": image[1],
                   "frames": frames, "client_metrics": {"mediapipe_ms_mean": 11.5}}
        capture.update(extra or {})
        return capture


def _hand_frame(centre_user_px, palm_px: float, image=(640, 480), angle: float = 0.0, rng=None) -> list:
    """Landmarks of a canonical open hand whose palm centre is at `centre_user_px` (user view), raw coordinates."""
    rng = rng or np.random.default_rng(0)
    ca, sa = np.cos(angle), np.sin(angle)
    shape = (_BASE_HAND - _BASE_HAND[[0, 5, 9, 13, 17]].mean(axis=0)) @ np.array([[ca, -sa], [sa, ca]]).T
    pts = np.asarray(centre_user_px) + shape * palm_px + rng.normal(0, 0.5, (21, 2))
    z = np.zeros(21)
    return np.column_stack([(image[0] - pts[:, 0]) / image[0], pts[:, 1] / image[1], z]).round(5).tolist()


def polyline_capture(waypoints, duration_s: float = 2.0, fps: float = 30.0, palm_px: float = 50.0, seed: int = 0,
                     image=(640, 480)) -> dict:
    """A hand moved at constant speed along `waypoints` (user-view pixels) - for shape tests: an incomplete Z, a
    backwards Z, a straight swipe, a circle..."""
    rng = np.random.default_rng(seed)
    waypoints = np.asarray(waypoints, dtype=float)
    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    times = np.arange(0.0, duration_s + 1e-9, 1.0 / fps)
    frames = []
    for t in times:
        d = cumulative[-1] * t / duration_s
        i = min(int(np.searchsorted(cumulative, d, side="right")) - 1, len(lengths) - 1)
        u = (d - cumulative[i]) / max(lengths[i], 1e-9)
        point = waypoints[i] + (waypoints[i + 1] - waypoints[i]) * min(u, 1.0)
        frames.append({"t": round(1000 * t, 3), "landmarks": _hand_frame(point, palm_px, image, rng=rng), "handedness": "Right"})
    return {"gesture_type": "z", "mirrored": True, "image_width": image[0], "image_height": image[1], "frames": frames}


#: Hand poses for `Person.capture(finger_extension=...)`: an open hand and a pointing hand (index out, the rest curled -
#: fingertips folded back towards the palm).
OPEN_HAND = (1.0, 1.0, 1.0, 1.0, 1.0)
POINTING_HAND = (0.8, 1.0, 0.25, 0.1, 0.1)

#: A clean Z in user-view pixels: top-left -> top-right -> bottom-left -> bottom-right.
Z_WAYPOINTS = [(200, 150), (440, 150), (200, 330), (440, 330)]


def still_hand_capture(seed: int = 0, frames_n: int = 60) -> dict:
    """A hand held still in front of the camera (INVALID_TRAJECTORY)."""
    capture = Person(seed).capture(seed, duration_s=2.0)
    first = capture["frames"][0]["landmarks"]
    for frame in capture["frames"]:
        frame["landmarks"] = first
    return capture


def no_hand_capture(frames_n: int = 60) -> dict:
    return {"gesture_type": "z", "mirrored": True, "image_width": 640, "image_height": 480,
            "frames": [{"t": k * 33.3, "landmarks": None} for k in range(frames_n)]}


def with_idle(capture: dict, lead_s: float, tail_s: float, fps: float = 30.0, seed: int = 0) -> dict:
    """The same execution with the hand held (nearly) still before and after the gesture - what a real recording
    contains when the user raises the hand, waits, draws, and waits before stopping. Landmark jitter as in capture()."""
    rng = np.random.default_rng(seed)
    frames = [f for f in capture["frames"] if f["landmarks"] is not None]
    step = 1000.0 / fps

    def hold(template, n, start_t):
        base = np.asarray(template["landmarks"])
        out = []
        for k in range(n):
            jitter = np.column_stack([rng.normal(0, 0.6 / 640, 21), rng.normal(0, 0.6 / 480, 21), np.zeros(21)])
            out.append({"t": round(start_t + k * step, 3), "landmarks": (base + jitter).round(5).tolist(), "handedness": "Right"})
        return out

    lead_n, tail_n = int(round(lead_s * fps)), int(round(tail_s * fps))
    lead = hold(frames[0], lead_n, 0.0)
    shift = lead_n * step
    body = [{**f, "t": round(f["t"] + shift, 3)} for f in capture["frames"]]
    tail = hold(frames[-1], tail_n, body[-1]["t"] + step)
    return {**capture, "frames": lead + body + tail}


def at_frame_rate(capture: dict, keep_every: int) -> dict:
    """The same execution tracked at a lower frame rate (every `keep_every`-th frame) - e.g. a camera slowed down by
    poor lighting or a busy browser."""
    return {**capture, "frames": capture["frames"][::keep_every]}


def transformed(capture: dict, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0, angle_deg: float = 0.0,
                cx: float = 0.5, cy: float = 0.5) -> dict:
    """The same execution moved / scaled / rotated in the image (about (cx, cy), isotropic pixels) - position, Z size
    and camera orientation changes with everything else identical."""
    import copy

    out = copy.deepcopy(capture)
    w, h = out["image_width"], out["image_height"]
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    for frame in out["frames"]:
        if frame["landmarks"]:
            lm = np.asarray(frame["landmarks"], dtype=float)
            px = np.column_stack([(lm[:, 0] - cx) * w, (lm[:, 1] - cy) * h])
            px = (px @ np.array([[c, -s], [s, c]]).T) * scale
            lm[:, 0] = cx + px[:, 0] / w + dx
            lm[:, 1] = cy + px[:, 1] / h + dy
            lm[:, 2] *= scale
            frame["landmarks"] = lm.round(6).tolist()
    return out
