"""
Hand detector — compatible with MediaPipe 0.10.x (Tasks API).
Auto-downloads hand_landmarker.task on first run (~8 MB).
"""
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Model asset ───────────────────────────────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_DIR  = Path(__file__).parent.parent.parent / "models" / "mediapipe"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"

# MediaPipe hand connections (pairs of landmark indices) for drawing
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _ensure_model() -> str:
    """Download hand_landmarker.task if not already cached. Returns path string."""
    if _MODEL_PATH.exists():
        return str(_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading hand_landmarker.task (~8 MB) …")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info("Downloaded: %s", _MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download MediaPipe hand landmarker model: {e}\n"
            f"Manual download: {_MODEL_URL}\n"
            f"Place at: {_MODEL_PATH}"
        ) from e
    return str(_MODEL_PATH)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    landmarks_flat:    np.ndarray = field(default_factory=lambda: np.zeros(126))
    landmarks_per_hand: list      = field(default_factory=list)
    handedness:        list       = field(default_factory=list)
    bounding_boxes:    list       = field(default_factory=list)
    num_hands:         int        = 0
    annotated_frame:   np.ndarray = None


# ── Detector ──────────────────────────────────────────────────────────────────

class HandDetector:
    """
    MediaPipe 0.10.x Tasks-API hand detector.
    Drop-in replacement for the old mp.solutions.hands interface.
    """

    def __init__(
        self,
        max_num_hands: int   = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence:  float = 0.6,
        model_complexity: int = 0,          # kept for API compat, unused
    ):
        import mediapipe as mp

        model_path = _ensure_model()

        base_opts = mp.tasks.BaseOptions(model_asset_path=model_path)
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options                = base_opts,
            running_mode                = mp.tasks.vision.RunningMode.IMAGE,
            num_hands                   = max_num_hands,
            min_hand_detection_confidence = min_detection_confidence,
            min_hand_presence_confidence  = min_detection_confidence,
            min_tracking_confidence       = min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self._mp_image   = mp.Image                 # class ref for wrapping frames
        self._ImageFormat = mp.ImageFormat

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run hand detection on an RGB uint8 frame.
        Returns DetectionResult with landmarks, bounding boxes, annotated frame.
        """
        import mediapipe as mp

        # Upscale small frames — Tasks API needs ≥300px for reliable detection
        h, w = frame.shape[:2]
        scale = 1.0
        if min(h, w) < 300:
            scale = 300 / min(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
            h, w = frame.shape[:2]

        annotated = frame.copy()
        result    = DetectionResult(annotated_frame=annotated)

        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection = self._landmarker.detect(mp_img)

        if not detection.hand_landmarks:
            return result

        h, w = frame.shape[:2]
        landmarks_per_hand: list[np.ndarray] = []
        bounding_boxes: list[tuple]          = []
        handedness_list: list[str]           = []

        for hand_idx, hand_lms in enumerate(detection.hand_landmarks):
            # (21, 3) array in normalised coords
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])
            landmarks_per_hand.append(lm_array)

            # Draw skeleton on annotated frame
            self._draw_landmarks(annotated, lm_array, w, h)

            # Pixel bounding box
            xs  = lm_array[:, 0] * w
            ys  = lm_array[:, 1] * h
            pad = 20
            bounding_boxes.append((
                int(xs.min()) - pad, int(ys.min()) - pad,
                int(xs.max()) + pad, int(ys.max()) + pad,
            ))

            # Handedness
            if detection.handedness and hand_idx < len(detection.handedness):
                label = detection.handedness[hand_idx][0].category_name
            else:
                label = "Unknown"
            handedness_list.append(label)

        # Flatten to (126,) — 2 hands × 21 × 3
        flat = np.zeros(126)
        for i, lm in enumerate(landmarks_per_hand[:2]):
            flat[i * 63:(i + 1) * 63] = lm.flatten()

        result.landmarks_flat     = flat
        result.landmarks_per_hand = landmarks_per_hand
        result.handedness         = handedness_list
        result.bounding_boxes     = bounding_boxes
        result.num_hands          = len(landmarks_per_hand)
        result.annotated_frame    = annotated
        return result

    def close(self):
        self._landmarker.close()

    # ── drawing helper ────────────────────────────────────────────────────────

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, lm: np.ndarray, w: int, h: int):
        pts = [(int(lm[i, 0] * w), int(lm[i, 1] * h)) for i in range(21)]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, (0, 150, 80),    1)
