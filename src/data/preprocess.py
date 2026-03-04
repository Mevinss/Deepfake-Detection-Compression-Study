"""
Data preprocessing utilities for deepfake detection.

Handles video frame extraction, face detection/cropping with MediaPipe or MTCNN,
and augmentation that mimics video compression artifacts.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from facenet_pytorch import MTCNN
    import torch
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Face detector wrappers
# ---------------------------------------------------------------------------

class MediaPipeFaceDetector:
    """Face detector backed by MediaPipe."""

    def __init__(self, min_detection_confidence: float = 0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe is not installed. Run: pip install mediapipe")
        mp_face = mp.solutions.face_detection
        self._detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) bounding boxes in pixel coordinates."""
        results = self._detector.process(frame_rgb)
        boxes: List[Tuple[int, int, int, int]] = []
        if not results.detections:
            return boxes
        h, w = frame_rgb.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(int(bb.width * w), w - x)
            bh = min(int(bb.height * h), h - y)
            boxes.append((x, y, bw, bh))
        return boxes


class MTCNNFaceDetector:
    """Face detector backed by MTCNN (facenet-pytorch)."""

    def __init__(self, device: Optional[str] = None):
        if not MTCNN_AVAILABLE:
            raise ImportError(
                "facenet-pytorch is not installed. Run: pip install facenet-pytorch"
            )
        import torch as _torch
        _device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        self._mtcnn = MTCNN(keep_all=True, device=_device)

    def detect(self, frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) bounding boxes."""
        boxes_raw, _ = self._mtcnn.detect(frame_rgb)
        if boxes_raw is None:
            return []
        result = []
        for box in boxes_raw:
            x1, y1, x2, y2 = [int(v) for v in box]
            result.append((x1, y1, x2 - x1, y2 - y1))
        return result


def get_face_detector(backend: str = "mediapipe", **kwargs):
    """Factory that returns a face detector by backend name."""
    if backend == "mediapipe":
        return MediaPipeFaceDetector(**kwargs)
    if backend == "mtcnn":
        return MTCNNFaceDetector(**kwargs)
    raise ValueError(f"Unknown face detector backend: {backend!r}")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    output_dir: str,
    max_frames: int = 300,
    frame_interval: int = 1,
) -> List[str]:
    """Extract up to *max_frames* frames from a video file using OpenCV.

    Args:
        video_path: Path to the source video.
        output_dir: Directory where extracted frames (PNG) will be saved.
        max_frames: Maximum number of frames to extract.
        frame_interval: Extract one frame every *frame_interval* frames.

    Returns:
        List of paths to the saved frame images.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    saved: List[str] = []
    frame_idx = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            saved_count += 1
        frame_idx += 1

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Face cropping
# ---------------------------------------------------------------------------

def crop_faces(
    image: np.ndarray,
    detector,
    target_size: Tuple[int, int] = (224, 224),
    margin: float = 0.2,
) -> List[np.ndarray]:
    """Detect faces in *image* and return cropped/resized face patches.

    Args:
        image: BGR image (as returned by OpenCV).
        detector: Face detector instance with a ``detect`` method.
        target_size: (width, height) to resize each crop.
        margin: Relative margin to add around the detected bounding box.

    Returns:
        List of BGR face crops resized to *target_size*.
    """
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = detector.detect(frame_rgb)
    crops: List[np.ndarray] = []
    h, w = image.shape[:2]

    for x, y, bw, bh in boxes:
        mx = int(bw * margin)
        my = int(bh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, target_size)
        crops.append(crop)

    return crops


# ---------------------------------------------------------------------------
# Video compression helpers
# ---------------------------------------------------------------------------

def compress_video_ffmpeg(
    input_path: str,
    output_path: str,
    crf: int = 23,
    codec: str = "libx264",
    preset: str = "medium",
) -> str:
    """Re-encode a video with H.264 at a given CRF level using ffmpeg.

    Args:
        input_path: Source video path.
        output_path: Destination video path.
        crf: Constant Rate Factor (lower = better quality; typical range 0–51).
        codec: FFmpeg video codec name.
        preset: FFmpeg encoding preset (e.g. ``"medium"``, ``"fast"``).

    Returns:
        Path to the compressed video.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-an",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def apply_compression_to_image(
    image: np.ndarray,
    quality: int = 75,
    use_jpeg: bool = True,
) -> np.ndarray:
    """Simulate compression artifacts on a single image using JPEG encoding.

    Args:
        image: BGR image as a NumPy array.
        quality: JPEG quality (1–100; lower = more artifacts).
        use_jpeg: If True, encode/decode through JPEG; otherwise returns original.

    Returns:
        Image array with simulated compression artifacts.
    """
    if not use_jpeg:
        return image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(
    image_size: int = 224,
    simulate_compression: bool = True,
    jpeg_quality_range: Tuple[int, int] = (40, 95),
) -> A.Compose:
    """Return albumentations training transform pipeline.

    Optionally includes JPEG compression artifacts to mimic video codec output.
    """
    transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(std_range=(0.05, 0.2), p=0.3),
    ]

    if simulate_compression:
        transforms.append(
            A.ImageCompression(
                quality_range=jpeg_quality_range,
                p=0.7,
            )
        )

    transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Return albumentations validation/test transform pipeline (no augmentation)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_compression_transforms(
    image_size: int = 224,
    jpeg_quality: int = 75,
) -> A.Compose:
    """Return transforms that apply a fixed JPEG compression quality level.

    Used during robustness evaluation to simulate a specific CRF level.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.ImageCompression(
                quality_range=(jpeg_quality, jpeg_quality),
                p=1.0,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# ---------------------------------------------------------------------------
# Utility: CRF → approximate JPEG quality mapping
# ---------------------------------------------------------------------------

CRF_TO_JPEG_QUALITY: dict = {
    0: 95,
    23: 80,
    32: 55,
    40: 30,
    51: 10,
}


def crf_to_jpeg_quality(crf: int) -> int:
    """Map a H.264 CRF value to an approximate JPEG quality for simulation."""
    if crf in CRF_TO_JPEG_QUALITY:
        return CRF_TO_JPEG_QUALITY[crf]
    # Linear interpolation between nearest known anchors
    keys = sorted(CRF_TO_JPEG_QUALITY.keys())
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= crf <= hi:
            t = (crf - lo) / (hi - lo)
            q_lo = CRF_TO_JPEG_QUALITY[lo]
            q_hi = CRF_TO_JPEG_QUALITY[hi]
            return max(1, int(q_lo + t * (q_hi - q_lo)))
    return 10  # fallback for CRF > 51
