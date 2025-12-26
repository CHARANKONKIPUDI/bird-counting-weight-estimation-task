from ultralytics import YOLO
import numpy as np
from app.utils import bbox_has_motion, is_red_dominant


class BirdDetector:
    def __init__(self, model_path):
        """
        YOLO-based bird detector with:
        - geometry filtering
        - red-object rejection
        - motion as helper (not mandatory)
        """
        self.model = YOLO(model_path)

    def detect(self, frame, motion_mask=None):
        results = self.model(
            frame,
            conf=0.05,       # low threshold, we filter later
            iou=0.5,
            imgsz=640,
            verbose=False
        )

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # -----------------------------
                # Geometry checks
                # -----------------------------
                bw = x2 - x1
                bh = y2 - y1
                area = bw * bh
                aspect = bw / (bh + 1e-6)

                # Bird-like size
                if area < 150 or area > 40000:
                    continue

                # Bird-like shape
                if aspect < 0.4 or aspect > 2.5:
                    continue

                # -----------------------------
                # Color filter (reject red objects)
                # -----------------------------
                if is_red_dominant(frame, (x1, y1, x2, y2)):
                    continue

                # -----------------------------
                # Motion (OPTIONAL)
                # -----------------------------
                if motion_mask is not None:
                    has_motion = bbox_has_motion(
                        motion_mask,
                        (x1, y1, x2, y2)
                    )
                else:
                    has_motion = True

                # Reject only if weak + static
                if not has_motion and conf < 0.20:
                    continue

                detections.append([x1, y1, x2, y2, conf])

        return np.asarray(detections, dtype=np.float32)
