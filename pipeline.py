"""GPU pipeline: detect and track objects, apply a rule, render with NVENC."""
import colorsys
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
MAX_FRAMES = 3600
COCO_CLASSES = [0, 1, 2, 3, 5, 7, 62, 63, 67]
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
               7: "truck", 62: "tv", 63: "laptop", 67: "cell phone"}
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = DEVICE == 0
MASK_PADDING_FRAMES = 2  # also mask neighbouring frames to hide detection flicker

_detector = None
_pose = None


def _models():
    global _detector, _pose
    if _detector is None:
        _detector = YOLO("yolo11n.pt")
        _pose = YOLO("yolo11n-pose.pt")
    return _detector, _pose


def device_name() -> str:
    return torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU"


def _color_name(pixels: np.ndarray) -> str:
    if pixels.size == 0:
        return "gray"
    b, g, r = np.median(pixels.reshape(-1, 3), axis=0) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.22:
        return "black"
    if s < 0.2:
        return "white" if v > 0.75 else "gray"
    hue = h * 360
    for limit, name in ((15, "red"), (40, "orange"), (70, "yellow"), (165, "green"),
                        (255, "blue"), (330, "purple")):
        if hue < limit:
            return name
    return "red"


def _region_color(frame: np.ndarray, box: list, is_person: bool) -> str:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    top, bottom = (0.2, 0.5) if is_person else (0.25, 0.75)
    crop = frame[int(y1 + h * top):int(y1 + h * bottom), int(x1 + w * 0.25):int(x2 - w * 0.25)]
    return _color_name(crop)


def _iou(a: list, b: list) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _face_box(keypoints: np.ndarray, confidence: np.ndarray, width: int, height: int):
    """Estimate a square face box from the nose, eye and ear keypoints."""
    points = keypoints[:5][confidence[:5] > 0.3]
    if len(points) < 2:
        return None
    cx, cy = points.mean(axis=0)
    span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    half = max(span * 1.1, 8)
    return [max(0.0, float(cx - half)), max(0.0, float(cy - half)),
            min(float(width), float(cx + half)), min(float(height), float(cy + half))]


def detect(video: Path, on_progress) -> dict:
    detector, pose = _models()
    # Tracker state lives on the predictor, so reset it for each new video.
    detector.predictor = None

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("无法读取这个视频文件")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > MAX_FRAMES:
        capture.release()
        raise ValueError(f"视频过长：{total} 帧，演示版上限 {MAX_FRAMES} 帧")

    items, index = [], 0
    options = {"imgsz": 640, "device": DEVICE, "half": HALF, "verbose": False, "conf": 0.3}

    def add(kind, box, track_id, color):
        items.append({
            "frame": index, "box": box, "kind": kind, "track_id": track_id,
            "t": round(index / fps, 3),
            "cx": (box[0] + box[2]) / 2 / width, "cy": (box[1] + box[3]) / 2 / height,
            "w": (box[2] - box[0]) / width, "h": (box[3] - box[1]) / height,
            "color": color,
        })

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tracked = detector.track(frame, persist=True, classes=COCO_CLASSES,
                                 tracker="bytetrack.yaml", **options)[0].boxes
        people = []
        for i, box_tensor in enumerate(tracked.xyxy.cpu().numpy()):
            box = [float(v) for v in box_tensor]
            kind = CLASS_NAMES[int(tracked.cls[i])]
            track_id = int(tracked.id[i]) if tracked.id is not None else -1
            color = _region_color(frame, box, kind == "person")
            add(kind, box, track_id, color)
            if kind == "person":
                people.append((box, track_id, color))

        if people:
            result = pose.predict(frame, **options)[0]
            if result.keypoints is not None and result.keypoints.conf is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                points = result.keypoints.xy.cpu().numpy()
                confidence = result.keypoints.conf.cpu().numpy()
                for pose_box, kp, conf in zip(boxes, points, confidence):
                    face = _face_box(kp, conf, width, height)
                    if face is None:
                        continue
                    best = max(people, key=lambda p: _iou(p[0], [float(v) for v in pose_box]))
                    add("face", face, best[1], best[2])

        index += 1
        if index % 5 == 0 and total:
            on_progress(index / total)

    capture.release()
    if index == 0:
        raise ValueError("视频里没有可读取的帧")
    return {"fps": fps, "width": width, "height": height, "frames": index, "items": items}


def load_or_detect(video: Path, digest: str, on_progress) -> tuple:
    """Return (detections, cache_hit). Detection is reused for an identical video."""
    cache = ROOT / "data" / "cache" / f"{digest}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8")), True
    detections = detect(video, on_progress)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(detections), encoding="utf-8")
    return detections, False


def apply_rule(code: str, items: list) -> list:
    public = [{k: v for k, v in item.items() if k not in ("frame", "box")} for item in items]
    completed = subprocess.run(
        [sys.executable, "-I", str(ROOT / "sandbox.py")],
        input=json.dumps({"code": code, "items": public}).encode("ascii"),
        capture_output=True, timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("规则执行失败: " + completed.stderr.decode(errors="replace")[-500:])
    return json.loads(completed.stdout)


def render(video: Path, output: Path, detections: dict, decisions: list, on_progress) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg")

    frames_to_boxes = defaultdict(list)
    masked = 0
    for item, decision in zip(detections["items"], decisions):
        if decision:
            masked += 1
            for f in range(item["frame"] - MASK_PADDING_FRAMES, item["frame"] + MASK_PADDING_FRAMES + 1):
                frames_to_boxes[f].append(item["box"])

    width, height = detections["width"], detections["height"]
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(detections["fps"]), "-i", "-", "-i", str(video),
        "-map", "0:v", "-map", "1:a?", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "24", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(output),
    ]
    log_path = output.with_suffix(".ffmpeg.log")
    capture = cv2.VideoCapture(str(video))
    with open(log_path, "wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                for box in frames_to_boxes.get(index, ()):
                    _pixelate(frame, box)
                process.stdin.write(frame.tobytes())
                index += 1
                if index % 10 == 0:
                    on_progress(index / detections["frames"])
        except BrokenPipeError:
            pass  # ffmpeg exited early; its return code is checked below
        finally:
            capture.release()
            try:
                process.stdin.close()
            except OSError:
                pass
            process.wait()
    if process.returncode != 0:
        raise RuntimeError("NVENC 编码失败: " + log_path.read_text(errors="replace")[-500:])
    return masked


def _pixelate(frame: np.ndarray, box: list) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    h, w = region.shape[:2]
    small = cv2.resize(region, (max(1, w // 14), max(1, h // 14)), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
