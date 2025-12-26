from fastapi import FastAPI, UploadFile, File
import shutil, os, cv2

from app.detector import BirdDetector
from app.tracker import BirdTracker
from app.weight_estimator import estimate_weight_grams
from app.utils import (
    estimate_foreground_area,
    draw_annotations,
    get_motion_mask,
    reset_motion_state,
)

app = FastAPI(title="Bird Counting & Weight Estimation")

detector = BirdDetector("models/yolov8s.pt")
tracker = BirdTracker()

MIN_BIRD_AREA = 1500
MAX_BIRDS = 150
MIN_BIRDS = 50
SMOOTHING_ALPHA = 0.7


@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...)):

    reset_motion_state()

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = f"data/{video.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "outputs/annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_id = 0
    processed_seconds = 0
    smoothed_count = None
    bird_counts = []
    weights = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        PROCESS_EVERY_N_FRAMES = max(1, fps // 3)
        if frame_id % PROCESS_EVERY_N_FRAMES != 0:
            frame_id += 1
            continue

        motion_mask = get_motion_mask(frame)
        detections = detector.detect(frame, motion_mask)

        raw_tracks = tracker.update(detections)

        tracks_with_weight = []
        areas = []

        for x1, y1, x2, y2, tid in raw_tracks:
            area = max((x2 - x1) * (y2 - y1), MIN_BIRD_AREA)
            weight = estimate_weight_grams((x1, y1, x2, y2))

            areas.append(area)
            weights.append(weight)

            tracks_with_weight.append(
                (x1, y1, x2, y2, tid, weight)
            )

        avg_area = max(sum(areas) / len(areas), MIN_BIRD_AREA) if areas else MIN_BIRD_AREA
        fg_area = estimate_foreground_area(frame)

        raw = max(fg_area / avg_area, MIN_BIRDS)
        if raw > MAX_BIRDS:
            raw = MAX_BIRDS - (raw - MAX_BIRDS) * 0.2

        smoothed_count = raw if smoothed_count is None else (
            SMOOTHING_ALPHA * smoothed_count +
            (1 - SMOOTHING_ALPHA) * raw
        )

        anchor = len(tracks_with_weight) * 20 if tracks_with_weight else MIN_BIRDS

        total_birds = int(0.7 * smoothed_count + 0.3 * anchor)

        # 🔒 SAFETY CONSTRAINTS
        total_birds = max(total_birds, len(tracks_with_weight))
        total_birds = max(total_birds, MIN_BIRDS)

        bird_counts.append({
            "time_sec": processed_seconds,
            "visible_birds": total_birds
        })

        annotated = draw_annotations(
            frame.copy(),
            tracks=tracks_with_weight,
            bird_count=total_birds,
            motion_mask=None
        )

        out.write(annotated)

        print(
            f"t={processed_seconds}s | "
            f"YOLO={len(detections)} | "
            f"Tracks={len(tracks_with_weight)} | "
            f"Est={total_birds}"
        )

        processed_seconds += 1
        frame_id += 1

    cap.release()
    out.release()

    avg_visible = sum(b["visible_birds"] for b in bird_counts) / len(bird_counts)
    avg_weight = sum(weights) / len(weights) if weights else 0

    return {
        "processed_seconds": processed_seconds,
        "average_visible_birds": round(avg_visible, 2),
        "average_weight_grams": round(avg_weight, 2),
        "bird_counts_over_time": bird_counts,
        "annotated_video": "outputs/annotated_video.mp4"
    }
