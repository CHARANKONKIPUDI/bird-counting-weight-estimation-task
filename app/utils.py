import cv2
import numpy as np

# ==============================
# FOREGROUND AREA ESTIMATION
# ==============================
def estimate_foreground_area(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cv2.countNonZero(clean)


# ==============================
# MOTION MASK (VIDEO SAFE)
# ==============================
_prev_gray = None

def reset_motion_state():
    global _prev_gray
    _prev_gray = None


def get_motion_mask(frame, thresh=15):
    global _prev_gray

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if _prev_gray is None:
        _prev_gray = gray
        return np.zeros_like(gray, dtype=np.uint8)

    diff = cv2.absdiff(gray, _prev_gray)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    _prev_gray = gray
    return mask


def bbox_has_motion(motion_mask, bbox, min_ratio=0.03):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = motion_mask.shape

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    roi = motion_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    motion_ratio = np.count_nonzero(roi) / roi.size
    return motion_ratio > min_ratio


# ==============================
# RED COLOR FILTER (IMPORTANT)
# ==============================
def is_red_dominant(frame, bbox, red_ratio_thresh=0.25):
    """
    Returns True if the bounding box is dominated by red color.
    Used to reject red objects (feeders, lights, clothes, etc.)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = frame.shape

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = mask1 | mask2
    red_ratio = np.count_nonzero(red_mask) / red_mask.size

    return red_ratio > red_ratio_thresh


# ==============================
# ANNOTATION (GREEN BOX + WEIGHT)
# ==============================
def draw_annotations(frame, tracks=None, bird_count=None, motion_mask=None):
    """
    tracks can be:
    - (x1,y1,x2,y2,id)
    - (x1,y1,x2,y2,id,weight)
    """

    # Optional heatmap (OFF by default)
    if motion_mask is not None:
        heat = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.85, heat, 0.15, 0)

    if tracks is not None:
        for t in tracks:
            if len(t) == 5:
                x1, y1, x2, y2, track_id = t
                weight = None
            else:
                x1, y1, x2, y2, track_id, weight = t

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            label = f"ID {track_id}"
            if weight is not None:
                label += f" | {int(weight)} g"

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    if bird_count is not None:
        cv2.putText(
            frame,
            f"Estimated Birds: {bird_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    return frame
