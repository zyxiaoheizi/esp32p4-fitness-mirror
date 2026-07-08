import argparse
import json
import math
import socket
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


MP_TO_COCO = {
    0: 0,
    2: 1,
    5: 2,
    7: 3,
    8: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_KEYPOINT_DICT = {name: idx for idx, name in enumerate(COCO_KEYPOINT_NAMES)}
MIN_CROP_KEYPOINT_SCORE = 0.2
TORSO_MP_INDICES = (11, 12, 23, 24)
SHOULDER_MP_INDICES = (11, 12)
HIP_MP_INDICES = (23, 24)
MP_POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

REQ_HEADER = struct.Struct("<4sHHII")
RESP_HEADER = struct.Struct("<4sHHIfII")
KPT_FLOATS = 4
KPT_COUNT = 33
MAX_PEOPLE = 4
MAX_RAW_FRAME_BYTES = 640 * 480 * 2
MAX_JPEG_FRAME_BYTES = 512 * 1024
REQ_MAGIC_RGB565 = b"P4P1"
REQ_MAGIC_RGB332 = b"P4P2"
REQ_MAGIC_JPEG = b"P4J1"
REQ_MAGIC_WEATHER = b"P4W1"
REQ_MAGIC_VOICE = b"P4V1"
RESP_MAGIC = b"P4R1"
RESP_MAGIC_WEATHER = b"P4W2"
RESP_MAGIC_VOICE = b"P4V2"
WEATHER_RESP_HEADER = struct.Struct("<4sII")
VOICE_RESP_HEADER = struct.Struct("<4sIIII")

VOICE_TEXTS = {
    0: "这一次下蹲深度不够，下一次髋部继续向后向下。",
    1: "这一次躯干前倾偏大，起身时抬胸并收紧核心。",
    2: "这一次膝盖轨迹偏移，下一次让膝盖跟随脚尖方向。",
    3: "这一次左右发力不太均衡，下一次两侧同时起身。",
    4: "连续五次动作很稳定，保持这个节奏。",
    5: "训练完成，辛苦了，记得放松和补水。",
    6: "俯卧撑下降幅度不够，下一次胸口再靠近地面一点。",
    7: "俯卧撑时身体线条不够稳定，收紧核心，肩髋脚跟保持一条线。",
    8: "卧推动作幅度不完整，注意下放和推起都做到位。",
    9: "卧推时左右肩肘不够同步，下一次两侧同时发力。",
    10: "引体向上高度不够，下一次胸口主动靠近横杠。",
    11: "引体向上摆动偏大，先稳定身体再发力。",
    12: "硬拉时背部稳定不足，抬胸收紧核心，不要弓背。",
    13: "硬拉时髋部发力不够，先髋后膝，保持杠铃贴近身体。",
    14: "平板支撑时髋部位置偏离，肩髋脚踝尽量保持一条线。",
    15: "平板支撑保持得不错，继续稳住呼吸。",
    16: "俯卧撑时髋部位置偏离，收紧核心，让肩髋脚跟保持一条线。",
    17: "俯卧撑下降幅度还不够，下一次肘部再弯曲一些，胸口靠近地面。",
    18: "卧推时手腕、肘和肩不够同步，下一次保持两侧同时下放和推起。",
    19: "引体向上请先进入底部悬挂姿态，让双手在肩上方入镜。",
    20: "引体向上要控制下降到底部伸展，再开始下一次。",
    21: "硬拉时膝髋节奏不协调，下一次先髋部向后，再配合屈膝。",
    22: "平板支撑中断，重新进入支撑姿态后再开始计时。",
    23: "平板支撑腰部有下塌趋势，收紧核心，继续保持。",
    24: "哑铃弯举幅度不够，下一次继续屈肘到顶点，再控制下放。",
    25: "哑铃弯举有借力或左右不同步，放慢节奏，保持上臂稳定。",
    26: "仰卧起坐幅度不够，下一次用腹部发力卷起到更高位置。",
    27: "仰卧起坐时颈部不够稳定，不要用手猛拉头部。",
}


def decode_rgb565le(data: bytes, width: int, height: int, swap_rb: bool = False) -> np.ndarray:
    expected = width * height * 2
    if len(data) != expected:
        raise ValueError(f"rgb565 body length {len(data)} != expected {expected}")
    raw = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
    r = ((raw >> 11) & 0x1F).astype(np.uint8)
    g = ((raw >> 5) & 0x3F).astype(np.uint8)
    b = (raw & 0x1F).astype(np.uint8)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    if swap_rb:
        r, b = b, r
    rgb[..., 0] = (r << 3) | (r >> 2)
    rgb[..., 1] = (g << 2) | (g >> 4)
    rgb[..., 2] = (b << 3) | (b >> 2)
    return rgb


def decode_rgb332(data: bytes, width: int, height: int, swap_rb: bool = False) -> np.ndarray:
    expected = width * height
    if len(data) != expected:
        raise ValueError(f"rgb332 body length {len(data)} != expected {expected}")
    raw = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
    r = (raw >> 5) & 0x07
    g = (raw >> 2) & 0x07
    b = raw & 0x03
    if swap_rb:
        r, b = b, r
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (r * 255 // 7).astype(np.uint8)
    rgb[..., 1] = (g * 255 // 7).astype(np.uint8)
    rgb[..., 2] = (b * 255 // 3).astype(np.uint8)
    return rgb


def decode_jpeg_rgb(data: bytes, width: int, height: int, swap_rb: bool = False) -> np.ndarray:
    if len(data) < 4 or data[0:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ValueError(f"jpeg marker check failed bytes={len(data)}")
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"jpeg decode failed bytes={len(data)}")
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    if swap_rb:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def result_quality(result: dict) -> float:
    people = result.get("people", [])
    if not people:
        return -1.0
    keypoints = people[0].get("keypoints", [])
    scores = [float(kp.get("score", 0.0)) for kp in keypoints]
    visible = sum(1 for score in scores if score >= 0.22)
    max_score = max(scores) if scores else 0.0
    avg_score = float(np.mean(scores)) if scores else 0.0
    return float(visible) * 1000.0 + max_score * 100.0 + avg_score


def filter_pose_result(result: dict, min_score: float = 0.28, min_valid: int = 4, min_box: float = 0.08) -> dict:
    """Drop unstable single-person false positives before packing to the board."""
    people = []
    for person in result.get("people", [])[:MAX_PEOPLE]:
        keypoints = person.get("keypoints", [])
        scores = [float(kp.get("score", 0.0)) for kp in keypoints]
        if not scores or max(scores) < min_score:
            continue

        def score_at(idx: int) -> float:
            if idx >= len(keypoints):
                return 0.0
            return float(keypoints[idx].get("score", 0.0))

        shoulder_valid = sum(1 for idx in SHOULDER_MP_INDICES if score_at(idx) >= min_score)
        hip_valid = sum(1 for idx in HIP_MP_INDICES if score_at(idx) >= min_score)
        torso_valid = sum(1 for idx in TORSO_MP_INDICES if score_at(idx) >= min_score)
        if shoulder_valid == 0 or torso_valid < 2:
            continue
        if hip_valid == 0 and shoulder_valid < 2:
            continue

        valid = sum(
            1
            for kp in keypoints
            if float(kp.get("score", 0.0)) >= min_score
            and -0.02 <= float(kp.get("x", 0.0)) <= 1.02
            and -0.02 <= float(kp.get("y", 0.0)) <= 1.02
        )
        if valid < min_valid:
            continue
        xs = [float(kp.get("x", 0.0)) for kp in keypoints if float(kp.get("score", 0.0)) >= min_score]
        ys = [float(kp.get("y", 0.0)) for kp in keypoints if float(kp.get("score", 0.0)) >= min_score]
        if xs and ys:
            box_w = max(xs) - min(xs)
            box_h = max(ys) - min(ys)
            if box_w < min_box or box_h < min_box:
                continue
            if box_h < min_box * 1.8 and torso_valid < 3:
                continue
        people.append(person)
    out = dict(result)
    out["people"] = people
    return out


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    if t_e <= 0.0:
        return 1.0
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter1D:
    """OneEuro filter (Casiez et al.), speed-adaptive low-pass for pose landmarks."""

    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 1e-6:
            return self.x_prev
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class PoseSmoother:
    """EMA on the latest person (legacy; low FPS can look like small-range jitter)."""

    def __init__(self, alpha: float = 0.45):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._last: Optional[list] = None

    def apply(self, result: dict) -> dict:
        people = result.get("people", [])
        if not people:
            return result
        keypoints = people[0].get("keypoints", [])
        if not keypoints:
            return result
        if self._last is None or len(self._last) != len(keypoints):
            self._last = [dict(kp) for kp in keypoints]
            return result
        smoothed = []
        for idx, kp in enumerate(keypoints):
            prev = self._last[idx]
            score = float(kp.get("score", 0.0))
            if score < 0.12:
                smoothed.append(dict(kp))
                self._last[idx] = dict(kp)
                continue
            a = self.alpha
            x = a * float(kp.get("x", 0.0)) + (1.0 - a) * float(prev.get("x", 0.0))
            y = a * float(kp.get("y", 0.0)) + (1.0 - a) * float(prev.get("y", 0.0))
            z = a * float(kp.get("z", 0.0)) + (1.0 - a) * float(prev.get("z", 0.0))
            s = a * score + (1.0 - a) * float(prev.get("score", 0.0))
            item = {"x": x, "y": y, "z": z, "score": s}
            smoothed.append(item)
            self._last[idx] = item
        out = dict(result)
        out["people"] = [{"score": people[0].get("score", 0.0), "keypoints": smoothed}]
        return out


class OneEuroPoseSmoother:
    """Per-keypoint OneEuro filter; holds last good pose briefly when a frame is rejected."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.007, hold_frames: int = 8):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.hold_frames = max(0, int(hold_frames))
        self._fx: List[Optional[OneEuroFilter1D]] = []
        self._fy: List[Optional[OneEuroFilter1D]] = []
        self._last_good: Optional[dict] = None
        self._hold_left = 0

    def apply(self, result: dict) -> dict:
        people = result.get("people", [])
        if not people or not people[0].get("keypoints"):
            if self._last_good is not None and self._hold_left > 0:
                self._hold_left -= 1
                return dict(self._last_good)
            return result

        keypoints = people[0]["keypoints"]
        now = time.perf_counter()
        if len(self._fx) != len(keypoints):
            self._fx = [None] * len(keypoints)
            self._fy = [None] * len(keypoints)

        smoothed = []
        for idx, kp in enumerate(keypoints):
            score = float(kp.get("score", 0.0))
            x = float(kp.get("x", 0.0))
            y = float(kp.get("y", 0.0))
            if score < 0.12:
                smoothed.append(dict(kp))
                self._fx[idx] = None
                self._fy[idx] = None
                continue
            if self._fx[idx] is None:
                self._fx[idx] = OneEuroFilter1D(now, x, min_cutoff=self.min_cutoff, beta=self.beta)
                self._fy[idx] = OneEuroFilter1D(now, y, min_cutoff=self.min_cutoff, beta=self.beta)
                sx, sy = x, y
            else:
                sx = self._fx[idx](now, x)
                sy = self._fy[idx](now, y)
            smoothed.append({"x": sx, "y": sy, "z": float(kp.get("z", 0.0)), "score": score})

        out = dict(result)
        out["people"] = [{"score": people[0].get("score", 0.0), "keypoints": smoothed}]
        self._last_good = out
        self._hold_left = self.hold_frames
        return out


def _smooth_crop_region(prev: Optional[dict], new: dict, alpha: float = 0.35) -> dict:
    if prev is None:
        return dict(new)
    a = float(np.clip(alpha, 0.05, 1.0))
    out = {}
    for key in ("y_min", "x_min", "y_max", "x_max", "height", "width"):
        out[key] = a * float(new[key]) + (1.0 - a) * float(prev[key])
    return out


def create_pose_smoother(
    mode: str,
    smooth_alpha: float,
    oneeuro_min_cutoff: float,
    oneeuro_beta: float,
    oneeuro_hold_frames: int,
) -> Optional[object]:
    mode = (mode or "oneeuro").lower()
    if mode in ("none", "off", "0"):
        return None
    if mode in ("ema",):
        if smooth_alpha <= 0.0:
            return None
        return PoseSmoother(smooth_alpha)
    if mode in ("oneeuro", "1euro", "euro"):
        return OneEuroPoseSmoother(oneeuro_min_cutoff, oneeuro_beta, oneeuro_hold_frames)
    raise ValueError(f"unknown smooth mode: {mode}")


class MediaPipeBackend:
    name = "mediapipe_pose_33"

    def __init__(self, model_complexity: int):
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def infer(self, rgb: np.ndarray) -> dict:
        start = time.perf_counter()
        result = self._pose.process(rgb)
        infer_ms = (time.perf_counter() - start) * 1000.0
        people = []
        if result.pose_landmarks:
            keypoints = []
            scores = []
            for lm in result.pose_landmarks.landmark:
                score = float(lm.visibility)
                scores.append(score)
                keypoints.append(
                    {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "score": score,
                    }
                )
            people.append({"score": float(np.mean(scores)), "keypoints": keypoints})
        return {"infer_ms": infer_ms, "people": people}


class YoloPoseBackend:
    name = "yolo11n_pose_17_mapped_to_33"

    def __init__(self, model_path: Path, imgsz: int):
        from ultralytics import YOLO

        if not model_path.exists():
            raise FileNotFoundError(model_path)
        self._model = YOLO(str(model_path))
        self._imgsz = imgsz

    def infer(self, rgb: np.ndarray) -> dict:
        start = time.perf_counter()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        preds = self._model.predict(bgr, imgsz=self._imgsz, verbose=False)
        infer_ms = (time.perf_counter() - start) * 1000.0
        people = []
        if not preds or preds[0].keypoints is None or preds[0].keypoints.data is None:
            return {"infer_ms": infer_ms, "people": people}

        h, w = rgb.shape[:2]
        kpts = preds[0].keypoints.data.cpu().numpy()
        boxes = preds[0].boxes.conf.cpu().numpy() if preds[0].boxes is not None else np.zeros((len(kpts),), dtype=np.float32)
        if len(kpts) == 0:
            return {"infer_ms": infer_ms, "people": people}

        order = np.argsort(-boxes) if len(boxes) == len(kpts) else range(len(kpts))
        for idx in order[:1]:
            mp_points = [{"x": 0.0, "y": 0.0, "z": 0.0, "score": 0.0} for _ in range(KPT_COUNT)]
            visible_scores = []
            for mp_idx, coco_idx in MP_TO_COCO.items():
                x, y, score = kpts[idx, coco_idx, :3]
                score = float(score)
                visible_scores.append(score)
                mp_points[mp_idx] = {
                    "x": float(np.clip(x / max(w, 1), 0.0, 1.0)),
                    "y": float(np.clip(y / max(h, 1), 0.0, 1.0)),
                    "z": 0.0,
                    "score": score,
                }
            people.append({"score": float(np.mean(visible_scores)) if visible_scores else 0.0, "keypoints": mp_points})
        return {"infer_ms": infer_ms, "people": people}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _resize_with_pad_rgb(image_rgb: np.ndarray, input_w: int, input_h: int):
    src_h, src_w = image_rgb.shape[:2]
    scale = min(input_w / src_w, input_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2
    padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


def _init_crop_region(image_h: int, image_w: int) -> dict:
    if image_w > image_h:
        box_h = image_w / image_h
        box_w = 1.0
        y_min = (image_h / 2.0 - image_w / 2.0) / image_h
        x_min = 0.0
    else:
        box_h = 1.0
        box_w = image_h / image_w
        y_min = 0.0
        x_min = (image_w / 2.0 - image_h / 2.0) / image_w
    return {
        "y_min": float(y_min),
        "x_min": float(x_min),
        "y_max": float(y_min + box_h),
        "x_max": float(x_min + box_w),
        "height": float(box_h),
        "width": float(box_w),
    }


def _crop_and_resize_rgb(image_rgb: np.ndarray, crop_region: dict, input_w: int, input_h: int) -> np.ndarray:
    image_h, image_w = image_rgb.shape[:2]
    x0 = float(crop_region["x_min"]) * image_w
    y0 = float(crop_region["y_min"]) * image_h
    crop_w = float(crop_region["width"]) * image_w
    crop_h = float(crop_region["height"]) * image_h
    matrix = np.array(
        [
            [crop_w / float(input_w), 0.0, x0],
            [0.0, crop_h / float(input_h), y0],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image_rgb,
        matrix,
        (input_w, input_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _torso_visible(keypoints: np.ndarray) -> bool:
    return (
        keypoints[COCO_KEYPOINT_DICT["left_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[COCO_KEYPOINT_DICT["right_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
    ) and (
        keypoints[COCO_KEYPOINT_DICT["left_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[COCO_KEYPOINT_DICT["right_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
    )


def _determine_crop_region(keypoints: np.ndarray, image_h: int, image_w: int) -> dict:
    if not _torso_visible(keypoints):
        return _init_crop_region(image_h, image_w)

    target = {}
    for name, idx in COCO_KEYPOINT_DICT.items():
        target[name] = (float(keypoints[idx, 0]) * image_h, float(keypoints[idx, 1]) * image_w)

    center_y = (target["left_hip"][0] + target["right_hip"][0]) / 2.0
    center_x = (target["left_hip"][1] + target["right_hip"][1]) / 2.0

    max_torso_y = max(abs(center_y - target[name][0]) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    max_torso_x = max(abs(center_x - target[name][1]) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    max_body_y = 0.0
    max_body_x = 0.0
    for name, idx in COCO_KEYPOINT_DICT.items():
        if keypoints[idx, 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        max_body_y = max(max_body_y, abs(center_y - target[name][0]))
        max_body_x = max(max_body_x, abs(center_x - target[name][1]))

    crop_half = max(max_torso_x * 1.9, max_torso_y * 1.9, max_body_y * 1.2, max_body_x * 1.2)
    crop_half = min(crop_half, max(center_x, image_w - center_x, center_y, image_h - center_y))
    if crop_half > max(image_w, image_h) / 2.0 or crop_half <= 1.0:
        return _init_crop_region(image_h, image_w)

    crop_len = crop_half * 2.0
    y_min = (center_y - crop_half) / image_h
    x_min = (center_x - crop_half) / image_w
    return {
        "y_min": float(y_min),
        "x_min": float(x_min),
        "y_max": float(y_min + crop_len / image_h),
        "x_max": float(x_min + crop_len / image_w),
        "height": float(crop_len / image_h),
        "width": float(crop_len / image_w),
    }


class MoveNetOnnxBackend:
    name = "movenet_lightning_17_mapped_to_33"

    def __init__(self, model_path: Path, threads: int, use_crop: bool = False, crop_smooth_alpha: float = 0.35):
        import onnxruntime as ort

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        if threads > 0:
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = 1

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_info = self._session.get_inputs()[0]
        self._output_shapes = [out.shape for out in self._session.get_outputs()]
        self._input_name = self._input_info.name
        self._input_type = self._input_info.type
        self._output_names = [out.name for out in self._session.get_outputs()]
        _, input_h, input_w, _ = self._input_info.shape
        self._input_w = int(input_w)
        self._input_h = int(input_h)
        self._use_crop = bool(use_crop)
        self._crop_smooth_alpha = float(np.clip(crop_smooth_alpha, 0.0, 1.0))
        self._crop_region = None
        print(
            f"[pc_pose] movenet model={model_path} input={self._input_info.shape} "
            f"{self._input_type} outputs={self._output_shapes} threads={threads} crop={self._use_crop}",
            flush=True,
        )

    def _prepare_input(self, rgb: np.ndarray):
        if self._use_crop:
            if self._crop_region is None:
                self._crop_region = _init_crop_region(*rgb.shape[:2])
            padded = _crop_and_resize_rgb(rgb, self._crop_region, self._input_w, self._input_h)
            scale = 0.0
            pad_x = 0
            pad_y = 0
        else:
            padded, scale, pad_x, pad_y = _resize_with_pad_rgb(rgb, self._input_w, self._input_h)
        if self._input_type == "tensor(int32)":
            tensor = padded.astype(np.int32)
        elif self._input_type == "tensor(float)":
            tensor = padded.astype(np.float32) / 127.5 - 1.0
        else:
            raise RuntimeError(f"Unsupported MoveNet ONNX input type: {self._input_type}")
        return np.expand_dims(tensor, axis=0), scale, pad_x, pad_y

    @staticmethod
    def _decode_raw_heads(outputs, output_names=None) -> np.ndarray:
        center = heatmap = regress = offset = None
        for idx, out in enumerate(outputs):
            if out.ndim != 4:
                continue
            name = output_names[idx] if output_names and idx < len(output_names) else ""
            channels = int(out.shape[1])
            if "center" in name:
                center = out
            elif "heatmap" in name:
                heatmap = out
            elif "regress" in name:
                regress = out
            elif "offset" in name:
                offset = out
            elif channels == 1:
                center = out
            elif channels == 17:
                heatmap = out
            elif channels == 34 and regress is None:
                regress = out
            elif channels == 34:
                offset = out

        if center is None or heatmap is None or regress is None or offset is None:
            shapes = [tuple(o.shape) for o in outputs]
            raise RuntimeError(f"Could not identify MoveNet raw heads from shapes: {shapes}")

        center_map = _sigmoid(center[0, 0])
        out_h, out_w = center_map.shape
        center_grid_y, center_grid_x = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        center_prior = np.sqrt((center_grid_y - (out_h / 2.0)) ** 2 + (center_grid_x - (out_w / 2.0)) ** 2)
        center_y, center_x = np.unravel_index(int(np.argmax(center_map / (center_prior + 1.8))), center_map.shape)

        heat = _sigmoid(heatmap[0])
        reg = np.transpose(regress[0], (1, 2, 0)).reshape(out_h, out_w, 17, 2)
        off = np.transpose(offset[0], (1, 2, 0)).reshape(out_h, out_w, 17, 2)

        regressed = reg[center_y, center_x]
        grid_y, grid_x = np.mgrid[0:out_h, 0:out_w].astype(np.float32)

        keypoints = np.zeros((17, 3), dtype=np.float32)
        for k in range(17):
            ry = float(center_y) + float(regressed[k, 0])
            rx = float(center_x) + float(regressed[k, 1])
            distance = np.sqrt((grid_y - ry) ** 2 + (grid_x - rx) ** 2)
            weighted_heat = heat[k] / (distance + 1.8)
            ky, kx = np.unravel_index(int(np.argmax(weighted_heat)), weighted_heat.shape)

            keypoints[k, 0] = (float(ky) + float(off[ky, kx, k, 0])) / float(out_h)
            keypoints[k, 1] = (float(kx) + float(off[ky, kx, k, 1])) / float(out_w)
            keypoints[k, 2] = float(heat[k, ky, kx])

        return keypoints

    def infer(self, rgb: np.ndarray) -> dict:
        total_start = time.perf_counter()
        prepare_start = total_start
        input_tensor, scale, pad_x, pad_y = self._prepare_input(rgb)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
        forward_start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: input_tensor})
        forward_ms = (time.perf_counter() - forward_start) * 1000.0
        decode_start = time.perf_counter()

        if len(outputs) == 1:
            keypoints_raw = np.squeeze(outputs[0]).reshape(-1, 3)
        else:
            keypoints_raw = self._decode_raw_heads(outputs, self._output_names)

        if keypoints_raw.shape[1] >= 3:
            keypoints_raw = keypoints_raw.copy()
            keypoints_raw[:, 0] = np.clip(keypoints_raw[:, 0], 0.0, 1.0)
            keypoints_raw[:, 1] = np.clip(keypoints_raw[:, 1], 0.0, 1.0)

        h, w = rgb.shape[:2]

        def build_people(swap_xy: bool):
            mp_points = [{"x": 0.0, "y": 0.0, "z": 0.0, "score": 0.0} for _ in range(KPT_COUNT)]
            visible_scores = []
            original_keypoints = np.zeros((17, 3), dtype=np.float32)
            for mp_idx, coco_idx in MP_TO_COCO.items():
                a0, a1, score = keypoints_raw[coco_idx, :3]
                score = float(score)
                if swap_xy:
                    y_norm = float(a1)
                    x_norm = float(a0)
                else:
                    y_norm = float(a0)
                    x_norm = float(a1)
                if self._use_crop and self._crop_region is not None:
                    x_norm_orig = float(self._crop_region["x_min"]) + float(x_norm) * float(self._crop_region["width"])
                    y_norm_orig = float(self._crop_region["y_min"]) + float(y_norm) * float(self._crop_region["height"])
                    x = x_norm_orig * w
                    y = y_norm_orig * h
                else:
                    if scale > 0:
                        px = float(x_norm) * self._input_w
                        py = float(y_norm) * self._input_h
                        x = (px - pad_x) / scale
                        y = (py - pad_y) / scale
                    else:
                        x = float(x_norm) * w
                        y = float(y_norm) * h
                    x_norm_orig = x / max(w, 1)
                    y_norm_orig = y / max(h, 1)
                visible_scores.append(score)
                original_keypoints[coco_idx] = [y_norm_orig, x_norm_orig, score]
                mp_points[mp_idx] = {
                    "x": float(np.clip(x / max(w, 1), 0.0, 1.0)),
                    "y": float(np.clip(y / max(h, 1), 0.0, 1.0)),
                    "z": 0.0,
                    "score": score,
                }
            return mp_points, visible_scores, original_keypoints

        def pose_quality(points, scores):
            if not scores:
                return -1.0
            mean_score = float(np.mean(scores))
            valid = sum(1 for s in scores if s >= 0.18)
            xs = [p["x"] for p in points if p["score"] >= 0.18]
            ys = [p["y"] for p in points if p["score"] >= 0.18]
            box_bonus = 0.0
            if xs and ys:
                box_w = max(xs) - min(xs)
                box_h = max(ys) - min(ys)
                if box_w > 0.02 and box_h > 0.02:
                    aspect = box_w / box_h
                    box_bonus = 1.0 if 0.2 <= aspect <= 5.0 else 0.0
            torso_bonus = 0.0
            try:
                ls = points[11]
                rs = points[12]
                lh = points[23]
                rh = points[24]
                torso_bonus = 1.0 if all(p["score"] >= 0.18 for p in (ls, rs, lh, rh)) else 0.0
            except Exception:
                torso_bonus = 0.0
            return valid * 1000.0 + mean_score * 100.0 + box_bonus * 10.0 + torso_bonus

        mp_points, visible_scores, original_keypoints = build_people(False)
        score = pose_quality(mp_points, visible_scores)

        if self._use_crop:
            if visible_scores and max(visible_scores) >= 0.18:
                new_crop = _determine_crop_region(original_keypoints, h, w)
            else:
                new_crop = _init_crop_region(h, w)
            if self._crop_smooth_alpha > 0.0:
                self._crop_region = _smooth_crop_region(self._crop_region, new_crop, self._crop_smooth_alpha)
            else:
                self._crop_region = new_crop

        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        infer_ms = (time.perf_counter() - total_start) * 1000.0
        people = []
        if visible_scores and max(visible_scores) >= 0.18:
            people.append({"score": float(np.mean(visible_scores)), "keypoints": mp_points})
        return {
            "infer_ms": infer_ms,
            "prepare_ms": prepare_ms,
            "forward_ms": forward_ms,
            "decode_ms": decode_ms,
            "people": people,
        }


def create_backend(args):
    if args.backend == "movenet":
        backend = MoveNetOnnxBackend(
            Path(args.movenet_model),
            args.movenet_threads,
            args.movenet_crop,
            crop_smooth_alpha=args.movenet_crop_smooth,
        )
        print(f"[pc_pose] backend={backend.name}", flush=True)
        return backend

    if args.backend in ("auto", "mediapipe"):
        try:
            backend = MediaPipeBackend(args.model_complexity)
            print(f"[pc_pose] backend={backend.name}", flush=True)
            return backend
        except Exception as exc:
            if args.backend == "mediapipe":
                raise
            print(f"[pc_pose] mediapipe unavailable: {exc}", flush=True)

    model_path = Path(args.yolo_model)
    backend = YoloPoseBackend(model_path, args.yolo_imgsz)
    print(f"[pc_pose] backend={backend.name} model={model_path}", flush=True)
    return backend


def recv_exact(sock, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def pack_result(result: dict, frame_seq: int = 0) -> bytes:
    people = result.get("people", [])[:MAX_PEOPLE]
    payload = bytearray()
    valid_count = 0
    for person in people:
        keypoints = person.get("keypoints", [])
        for i in range(KPT_COUNT):
            kp = keypoints[i] if i < len(keypoints) else {}
            x = float(kp.get("x", 0.0))
            y = float(kp.get("y", 0.0))
            z = float(kp.get("z", 0.0))
            score = float(kp.get("score", kp.get("visibility", 0.0)))
            if score >= 0.22 and -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05:
                valid_count += 1
            payload.extend(struct.pack("<ffff", x, y, z, score))

    infer_ms = float(result.get("infer_ms", 0.0))
    header = RESP_HEADER.pack(
        RESP_MAGIC,
        len(people),
        KPT_COUNT,
        valid_count,
        infer_ms,
        len(payload),
        int(frame_seq) & 0xFFFFFFFF,
    )
    return header + payload


class PoseTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        handler_class,
        backend,
        auto_color: bool,
        debug_dir: Optional[Path],
        debug_every: int,
        log_every: int,
        min_score: float,
        min_valid: int,
        min_box: float,
        smooth_mode: str,
        smooth_alpha: float,
        oneeuro_min_cutoff: float,
        oneeuro_beta: float,
        oneeuro_hold_frames: int,
        weather_location: str,
        weather_interval: float,
        voice_enabled: bool,
        voice_rate: int,
        voice_volume: int,
    ):
        super().__init__(server_address, handler_class)
        self.backend = backend
        self.auto_color = auto_color
        self.debug_dir = debug_dir
        self.debug_every = max(0, int(debug_every))
        self.log_every = max(0, int(log_every))
        self.min_score = float(min_score)
        self.min_valid = int(min_valid)
        self.min_box = float(min_box)
        self.smoother = create_pose_smoother(
            smooth_mode, smooth_alpha, oneeuro_min_cutoff, oneeuro_beta, oneeuro_hold_frames
        )
        self.infer_lock = threading.Lock()
        self.debug_count = 0
        self.stats_lock = threading.Lock()
        self.stats_start = time.perf_counter()
        self.stats_frames = 0
        self.stats_fps = 0.0
        self.weather_location = weather_location
        self.weather_interval = max(60.0, float(weather_interval))
        self.weather_lock = threading.Lock()
        self.weather_text = "天气: 待更新"
        self.weather_next_update = 0.0
        self.voice_enabled = bool(voice_enabled)
        self.voice_rate = int(np.clip(voice_rate, -10, 10))
        self.voice_volume = int(np.clip(voice_volume, 0, 100))
        self.voice_lock = threading.Lock()
        self.voice_cache: Dict[int, Tuple[int, int, int, bytes]] = {}
        if self.debug_dir and self.debug_every > 0:
            self.debug_dir.mkdir(parents=True, exist_ok=True)


def record_server_frame(server: PoseTCPServer) -> float:
    now = time.perf_counter()
    with server.stats_lock:
        server.stats_frames += 1
        elapsed = now - server.stats_start
        if elapsed >= 1.0:
            server.stats_fps = server.stats_frames / elapsed
            server.stats_frames = 0
            server.stats_start = now
        return float(server.stats_fps)


def save_debug_frame(server: PoseTCPServer, rgb: np.ndarray, result: dict, frame_seq: int, color_name: str) -> None:
    if not server.debug_dir or server.debug_every <= 0:
        return
    server.debug_count += 1
    if server.debug_count % server.debug_every != 0:
        return

    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]
    people = result.get("people", [])
    for person_idx, person in enumerate(people[:MAX_PEOPLE]):
        keypoints = person.get("keypoints", [])
        for a_idx, b_idx in MP_POSE_EDGES:
            if a_idx >= len(keypoints) or b_idx >= len(keypoints):
                continue
            a = keypoints[a_idx]
            b = keypoints[b_idx]
            if float(a.get("score", 0.0)) >= 0.22 and float(b.get("score", 0.0)) >= 0.22:
                ax = int(np.clip(float(a.get("x", 0.0)) * w, 0, w - 1))
                ay = int(np.clip(float(a.get("y", 0.0)) * h, 0, h - 1))
                bx = int(np.clip(float(b.get("x", 0.0)) * w, 0, w - 1))
                by = int(np.clip(float(b.get("y", 0.0)) * h, 0, h - 1))
                cv2.line(out, (ax, ay), (bx, by), (0, 255, 255), 2, cv2.LINE_AA)
        for kp in keypoints:
            if float(kp.get("score", 0.0)) >= 0.22:
                x = int(np.clip(float(kp.get("x", 0.0)) * w, 0, w - 1))
                y = int(np.clip(float(kp.get("y", 0.0)) * h, 0, h - 1))
                cv2.circle(out, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(out, f"person {person_idx}", (8, 22 + person_idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    frame_path = server.debug_dir / f"frame_{frame_seq:06d}_{color_name}.jpg"
    cv2.imwrite(str(frame_path), out)


def decode_frame_payload(magic: bytes, width: int, height: int, payload_len: int):
    if magic == REQ_MAGIC_RGB565:
        return width * height * 2, MAX_RAW_FRAME_BYTES, decode_rgb565le, "rgb565"
    if magic == REQ_MAGIC_RGB332:
        return width * height, MAX_RAW_FRAME_BYTES, decode_rgb332, "rgb332"
    if magic == REQ_MAGIC_JPEG:
        return payload_len, MAX_JPEG_FRAME_BYTES, decode_jpeg_rgb, "jpeg"
    raise ValueError(f"bad magic {magic!r}")


def normalize_weather_desc(desc: str) -> str:
    text = (desc or "").strip().lower()
    if not text:
        return "未知"
    if any(word in text for word in ("thunder", "雷")):
        return "雷雨"
    if any(word in text for word in ("snow", "sleet", "blizzard", "雪")):
        return "雪"
    if any(word in text for word in ("rain", "drizzle", "shower", "雨")):
        return "雨"
    if any(word in text for word in ("fog", "mist", "haze", "雾", "霾")):
        return "雾"
    if any(word in text for word in ("overcast", "阴")):
        return "阴"
    if any(word in text for word in ("cloud", "多云", "云")):
        return "多云"
    if any(word in text for word in ("clear", "sunny", "晴")):
        return "晴"
    return desc[:10]


def fetch_weather_text(location: str, timeout: float = 3.0) -> str:
    loc = "" if not location or location.lower() == "auto" else urllib.parse.quote(location)
    url = f"https://wttr.in/{loc}?format=j1"
    req = urllib.request.Request(url, headers={"User-Agent": "fitness-mirror/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    current = (data.get("current_condition") or [{}])[0]
    descs = current.get("weatherDesc") or [{}]
    desc = normalize_weather_desc(str(descs[0].get("value", "")))
    temp = str(current.get("temp_C", "--"))
    humidity = str(current.get("humidity", "--"))
    return f"天气: {temp}C {desc} 湿度{humidity}%"


def get_weather_text(server) -> str:
    now = time.monotonic()
    with server.weather_lock:
        if now < server.weather_next_update and server.weather_text:
            return server.weather_text
        last_text = server.weather_text

    try:
        text = fetch_weather_text(server.weather_location)
        next_update = now + server.weather_interval
    except Exception as exc:
        text = last_text or "天气: 获取失败"
        next_update = now + 60.0
        print(f"[pc_pose] weather fetch failed: {exc}", flush=True)

    text = text[:80]
    with server.weather_lock:
        server.weather_text = text
        server.weather_next_update = next_update
    return text


def pack_weather_result(server, frame_seq: int = 0) -> bytes:
    payload = get_weather_text(server).encode("utf-8", errors="replace")
    return WEATHER_RESP_HEADER.pack(RESP_MAGIC_WEATHER, len(payload), int(frame_seq) & 0xFFFFFFFF) + payload


def _powershell_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def synthesize_voice_pcm(text: str, rate: int = 0, volume: int = 100) -> Tuple[int, int, int, bytes]:
    with tempfile.TemporaryDirectory(prefix="pc_pose_tts_") as tmp:
        wav_path = Path(tmp) / "voice.wav"
        script = "\n".join(
            [
                "Add-Type -AssemblyName System.Speech",
                f"$text = {_powershell_quote(text)}",
                f"$path = {_powershell_quote(str(wav_path))}",
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                f"$synth.Rate = {int(np.clip(rate, -10, 10))}",
                f"$synth.Volume = {int(np.clip(volume, 0, 100))}",
                "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
                "16000, "
                "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
                "[System.Speech.AudioFormat.AudioChannel]::Mono)",
                "$synth.SetOutputToWaveFile($path, $fmt)",
                "$synth.Speak($text) | Out-Null",
                "$synth.Dispose()",
            ]
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            check=True,
            timeout=8.0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        with wave.open(str(wav_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            bits_per_sample = wav.getsampwidth() * 8
            pcm = wav.readframes(wav.getnframes())
    if bits_per_sample != 16 or channels <= 0 or sample_rate <= 0 or not pcm:
        raise RuntimeError(f"bad tts wav format sr={sample_rate} ch={channels} bits={bits_per_sample} bytes={len(pcm)}")
    return int(sample_rate), int(channels), int(bits_per_sample), pcm


def fallback_voice_pcm(prompt_id: int) -> Tuple[int, int, int, bytes]:
    sample_rate = 16000
    duration = 0.42 if prompt_id != 5 else 0.62
    freq = 660.0 if prompt_id in (4, 5) else 440.0
    frames = int(sample_rate * duration)
    pcm = bytearray(frames * 2)
    for i in range(frames):
        envelope = min(1.0, i / 800.0, (frames - i) / 800.0)
        sample = int(math.sin(2.0 * math.pi * freq * (i / sample_rate)) * envelope * 9000)
        struct.pack_into("<h", pcm, i * 2, sample)
    return sample_rate, 1, 16, bytes(pcm)


def get_voice_pcm(server, prompt_id: int) -> Tuple[int, int, int, bytes]:
    prompt_id = int(prompt_id)
    with server.voice_lock:
        cached = server.voice_cache.get(prompt_id)
        if cached is not None:
            return cached

    text = VOICE_TEXTS.get(prompt_id, "动作已完成，继续保持。")
    if not getattr(server, "voice_enabled", True):
        audio = fallback_voice_pcm(prompt_id)
    else:
        try:
            audio = synthesize_voice_pcm(text, server.voice_rate, server.voice_volume)
            print(f"[pc_pose] voice cached id={prompt_id} bytes={len(audio[3])}", flush=True)
        except Exception as exc:
            print(f"[pc_pose] voice tts failed id={prompt_id}: {exc}", flush=True)
            audio = fallback_voice_pcm(prompt_id)

    with server.voice_lock:
        server.voice_cache[prompt_id] = audio
    return audio


def pack_voice_result(server, prompt_id: int, frame_seq: int = 0) -> bytes:
    sample_rate, channels, bits_per_sample, payload = get_voice_pcm(server, prompt_id)
    return VOICE_RESP_HEADER.pack(
        RESP_MAGIC_VOICE,
        int(sample_rate),
        int(channels),
        int(bits_per_sample),
        len(payload),
    ) + payload


def infer_frame(server: PoseTCPServer, frame: bytes, magic: bytes, width: int, height: int, payload_len: int, frame_seq: int):
    expected_payload_len, max_payload_len, decoder, frame_format = decode_frame_payload(magic, width, height, payload_len)
    if width <= 0 or height <= 0 or payload_len != expected_payload_len or payload_len > max_payload_len:
        raise ValueError(f"bad frame w={width} h={height} bytes={payload_len} max={max_payload_len}")
    if len(frame) != payload_len:
        raise ValueError(f"frame body length {len(frame)} != expected {payload_len}")

    rgb = decoder(frame, width, height, swap_rb=False)
    color_name = frame_format
    with server.infer_lock:
        result = server.backend.infer(rgb)
        if server.auto_color:
            rgb_swapped = decoder(frame, width, height, swap_rb=True)
            swapped_result = server.backend.infer(rgb_swapped)
            if result_quality(swapped_result) > result_quality(result):
                result = swapped_result
                color_name = frame_format.replace("rgb", "bgr", 1)
        result = filter_pose_result(result, server.min_score, server.min_valid, server.min_box)
        if server.smoother is not None:
            result = server.smoother.apply(result)
    response = pack_result(result, frame_seq)
    save_debug_frame(server, rgb, result, frame_seq, color_name)
    return response, result, color_name


class PoseTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        print(f"[pc_pose] connected {peer}", flush=True)
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as exc:
            print(f"[pc_pose] {peer} TCP_NODELAY failed: {exc}", flush=True)
        while True:
            frame_seq = 0
            try:
                header = recv_exact(self.request, REQ_HEADER.size)
                magic, width, height, payload_len, frame_seq = REQ_HEADER.unpack(header)
                if magic == REQ_MAGIC_WEATHER:
                    self.request.sendall(pack_weather_result(self.server, frame_seq))
                    continue
                if magic == REQ_MAGIC_VOICE:
                    if payload_len:
                        recv_exact(self.request, payload_len)
                    self.request.sendall(pack_voice_result(self.server, int(width), frame_seq))
                    continue
                frame = recv_exact(self.request, payload_len)
                response, result, color_name = infer_frame(self.server, frame, magic, width, height, payload_len, frame_seq)
                self.request.sendall(response)
                server_fps = record_server_frame(self.server)
                people = result.get('people', [])
                first = people[0]['keypoints'][0] if people and people[0].get('keypoints') else None
                if first is not None:
                    first_s = f" first=({first.get('x', 0.0):.3f},{first.get('y', 0.0):.3f},{first.get('score', 0.0):.3f})"
                else:
                    first_s = " first=None"
                if self.server.log_every > 0 and frame_seq % self.server.log_every == 0:
                    print(
                        f"[pc_pose] tcp frame={frame_seq} {width}x{height} people={len(people)} "
                        f"infer={result.get('infer_ms', 0.0):.1f} ms fps={server_fps:.1f} "
                        f"prep={result.get('prepare_ms', 0.0):.1f} fwd={result.get('forward_ms', 0.0):.1f} "
                        f"dec={result.get('decode_ms', 0.0):.1f} color={color_name} q={result_quality(result):.1f} "
                        f"bytes={len(response)}{first_s}",
                        flush=True,
                    )
            except ConnectionError:
                break
            except Exception as exc:
                print(f"[pc_pose] {peer} error: {exc}", flush=True)
                try:
                    self.request.sendall(RESP_HEADER.pack(RESP_MAGIC, 0, KPT_COUNT, 0, 0.0, 0, frame_seq))
                except Exception:
                    break
        print(f"[pc_pose] disconnected {peer}", flush=True)


class PoseUDPServer(socketserver.UDPServer):
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        handler_class,
        backend,
        auto_color: bool,
        debug_dir: Optional[Path],
        debug_every: int,
        log_every: int,
        min_score: float,
        min_valid: int,
        min_box: float,
        smooth_mode: str,
        smooth_alpha: float,
        oneeuro_min_cutoff: float,
        oneeuro_beta: float,
        oneeuro_hold_frames: int,
        weather_location: str,
        weather_interval: float,
        voice_enabled: bool,
        voice_rate: int,
        voice_volume: int,
    ):
        super().__init__(server_address, handler_class)
        self.backend = backend
        self.auto_color = auto_color
        self.debug_dir = debug_dir
        self.debug_every = max(0, int(debug_every))
        self.log_every = max(0, int(log_every))
        self.min_score = float(min_score)
        self.min_valid = int(min_valid)
        self.min_box = float(min_box)
        self.smoother = create_pose_smoother(
            smooth_mode, smooth_alpha, oneeuro_min_cutoff, oneeuro_beta, oneeuro_hold_frames
        )
        self.infer_lock = threading.Lock()
        self.debug_count = 0
        self.stats_lock = threading.Lock()
        self.stats_start = time.perf_counter()
        self.stats_frames = 0
        self.stats_fps = 0.0
        self.latest_lock = threading.Condition()
        self.latest_packet = None
        self.worker = threading.Thread(target=self._udp_latest_worker, name="pose_udp_latest", daemon=True)
        self.worker.start()
        if self.debug_dir and self.debug_every > 0:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def submit_latest(self, data: bytes, client_address):
        with self.latest_lock:
            self.latest_packet = (data, client_address)
            self.latest_lock.notify()

    def _udp_latest_worker(self):
        while True:
            with self.latest_lock:
                while self.latest_packet is None:
                    self.latest_lock.wait()
                data, client_address = self.latest_packet
                self.latest_packet = None
            process_udp_packet(self, data, client_address, self.socket)


def process_udp_packet(server: PoseUDPServer, data: bytes, client_address, sock) -> None:
    peer = f"{client_address[0]}:{client_address[1]}"
    frame_seq = 0
    try:
        if len(data) < REQ_HEADER.size:
            raise ValueError(f"short udp datagram {len(data)}")
        magic, width, height, payload_len, frame_seq = REQ_HEADER.unpack(data[: REQ_HEADER.size])
        frame = data[REQ_HEADER.size :]
        response, result, color_name = infer_frame(server, frame, magic, width, height, payload_len, frame_seq)
        sock.sendto(response, client_address)
        server_fps = record_server_frame(server)
        people = result.get("people", [])
        first = people[0]["keypoints"][0] if people and people[0].get("keypoints") else None
        if first is not None:
            first_s = f" first=({first.get('x', 0.0):.3f},{first.get('y', 0.0):.3f},{first.get('score', 0.0):.3f})"
        else:
            first_s = " first=None"
        if server.log_every > 0 and frame_seq % server.log_every == 0:
            print(
                f"[pc_pose] udp frame={frame_seq} {width}x{height} people={len(people)} "
                f"infer={result.get('infer_ms', 0.0):.1f} ms fps={server_fps:.1f} "
                f"prep={result.get('prepare_ms', 0.0):.1f} fwd={result.get('forward_ms', 0.0):.1f} "
                f"dec={result.get('decode_ms', 0.0):.1f} color={color_name} q={result_quality(result):.1f} "
                f"bytes={len(response)}{first_s}",
                flush=True,
            )
    except Exception as exc:
        print(f"[pc_pose] udp {peer} error: {exc}", flush=True)
        try:
            sock.sendto(RESP_HEADER.pack(RESP_MAGIC, 0, KPT_COUNT, 0, 0.0, 0, frame_seq), client_address)
        except Exception:
            pass


class PoseUDPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data, sock = self.request
        self.server.submit_latest(bytes(data), self.client_address)


def main():
    root = Path(__file__).resolve().parents[2]
    default_yolo = root / "tools" / "esp-dl-upstream" / "models" / "coco_pose" / "models" / "yolo11n-pose.onnx"
    default_movenet = (
        root
        / "tools"
        / "movenet_lightning_test"
        / "models"
        / "movenet_singlepose_lightning_espdl_stripped.onnx"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["auto", "mediapipe", "yolo", "movenet"], default="auto")
    parser.add_argument("--model-complexity", type=int, default=2)
    parser.add_argument("--yolo-model", default=str(default_yolo))
    parser.add_argument("--yolo-imgsz", type=int, default=320)
    parser.add_argument("--movenet-model", default=str(default_movenet))
    parser.add_argument("--movenet-threads", type=int, default=0)
    parser.add_argument(
        "--movenet-crop",
        action="store_true",
        help="MoveNet crop tracking on each received frame. Disable when the board already "
        "sends a center-cropped 192x192 patch (avoids double-crop jitter).",
    )
    parser.add_argument(
        "--movenet-crop-smooth",
        type=float,
        default=0.35,
        help="EMA on crop box when --movenet-crop is enabled (0 = no crop smoothing).",
    )
    parser.add_argument("--no-auto-color", action="store_true", help="Do not try RGB565 and BGR565 decoding and choose the better pose result.")
    parser.add_argument("--min-score", type=float, default=0.22, help="Minimum max keypoint score to accept a person.")
    parser.add_argument("--min-valid", type=int, default=6, help="Minimum visible keypoints to accept a person.")
    parser.add_argument("--min-box", type=float, default=0.08, help="Minimum normalized bbox width/height to accept a person.")
    parser.add_argument(
        "--smooth-mode",
        choices=["oneeuro", "ema", "none"],
        default="none",
        help="Temporal filter: oneeuro (recommended), ema, or none.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.0,
        help="EMA alpha when --smooth-mode=ema (ignored for oneeuro).",
    )
    parser.add_argument(
        "--oneeuro-min-cutoff",
        type=float,
        default=1.2,
        help="OneEuro filter min cutoff (Hz): lower = less jitter, more lag.",
    )
    parser.add_argument(
        "--oneeuro-beta",
        type=float,
        default=0.007,
        help="OneEuro filter speed coefficient: higher = less lag on fast motion.",
    )
    parser.add_argument(
        "--oneeuro-hold-frames",
        type=int,
        default=8,
        help="Keep last good skeleton for N frames when a frame is filtered out.",
    )
    parser.add_argument("--debug-dir", type=Path, help="Optional directory for annotated received-frame snapshots.")
    parser.add_argument("--debug-every", type=int, default=0, help="Save one annotated debug frame every N received frames.")
    parser.add_argument("--log-every", type=int, default=10, help="Print one timing/FPS line every N received frames. Use 0 to silence per-frame logs.")
    parser.add_argument("--weather-location", default="auto", help="Location for wttr.in weather, e.g. auto, Guangzhou, Shanghai.")
    parser.add_argument("--weather-interval", type=float, default=600.0, help="Weather cache interval in seconds.")
    parser.add_argument("--no-voice", action="store_true", help="Disable Windows TTS and return a short fallback tone for voice prompts.")
    parser.add_argument("--voice-rate", type=int, default=0, help="Windows TTS rate from -10 to 10.")
    parser.add_argument("--voice-volume", type=int, default=100, help="Windows TTS volume from 0 to 100.")
    args = parser.parse_args()

    backend = create_backend(args)
    server_kwargs = dict(
        backend=backend,
        auto_color=not args.no_auto_color,
        debug_dir=args.debug_dir,
        debug_every=args.debug_every,
        log_every=args.log_every,
        min_score=args.min_score,
        min_valid=args.min_valid,
        min_box=args.min_box,
        smooth_mode=args.smooth_mode,
        smooth_alpha=args.smooth_alpha,
        oneeuro_min_cutoff=args.oneeuro_min_cutoff,
        oneeuro_beta=args.oneeuro_beta,
        oneeuro_hold_frames=args.oneeuro_hold_frames,
        weather_location=args.weather_location,
        weather_interval=args.weather_interval,
        voice_enabled=not args.no_voice,
        voice_rate=args.voice_rate,
        voice_volume=args.voice_volume,
    )
    server = PoseTCPServer((args.host, args.port), PoseTCPHandler, **server_kwargs)
    udp_server = PoseUDPServer((args.host, args.port), PoseUDPHandler, **server_kwargs)
    threading.Thread(target=udp_server.serve_forever, name="pose_udp", daemon=True).start()
    print(f"[pc_pose] listening udp://{args.host}:{args.port}", flush=True)
    print(f"[pc_pose] listening tcp://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
