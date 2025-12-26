🐦 Bird Counting & Weight Estimation System
📌 Overview

This project implements an end-to-end bird counting and weight estimation system using computer vision and deep learning.
The system processes input videos to detect birds, track them over time, estimate flock size, and compute a proxy-based weight estimate per bird.

A FastAPI backend exposes an API that accepts video uploads and returns:

Estimated bird count over time

Average visible birds

Average estimated weight

Annotated output video

🎥 Demo Video (Annotated Output)

👉 Annotated video with detections, tracking IDs, and weight overlay:
🔗 https://drive.google.com/file/d/1-eeVeaLVvFIfdEil_NgBtNwbHq3Aom-A/view?usp=sharing

🚀 Key Features

YOLOv8-based bird detection

SORT-based multi-object tracking (stable IDs, no double counting)

Hybrid bird counting strategy

Detection + tracking for sparse scenes

Foreground area estimation for dense flocks

Supports static and moving birds

Red object rejection to reduce false positives

Green bounding boxes with:

Bird ID

Estimated weight (proxy)

REST API using FastAPI

Annotated output video generation

🧠 Approach & Methodology
1️⃣ Detection

YOLOv8 is used to detect bird-like objects from video frames.

Geometry-based filters applied:

Bounding box area

Aspect ratio

Red-colored objects (feeders, lights, clothing) are rejected using HSV color analysis.

Motion is optional — static birds are also detected.

2️⃣ Tracking

SORT (Simple Online Realtime Tracking) is used.

Assigns stable IDs to birds across frames.

Prevents double counting.

Handles short occlusions and brief disappearances.

3️⃣ Bird Counting Logic

A hybrid estimation strategy is used:

Visible tracked birds act as an anchor.

Foreground area estimation supports dense flock scenarios.

Temporal smoothing reduces sudden spikes.

Safety constraints applied:

Estimated count ≥ visible tracked birds

Estimated count ≥ minimum threshold

4️⃣ Weight Estimation (Proxy-Based)

Bird weight is estimated using bounding box area:

weight ≈ pixel_area × scale_factor


This provides a relative proxy, not ground-truth biological weight.

Can be calibrated using real bird measurements for actual grams.

⚠️ Important Note on Weight

The reported bird weight is an approximation based on bounding box area.
It should be interpreted as a relative indicator, not a medical or biological measurement.

📂 Project Structure
bird-counting-weight-estimation/
├── app/
│   ├── main.py              # FastAPI application
│   ├── detector.py          # YOLO-based bird detection
│   ├── tracker.py           # SORT-based tracking
│   ├── utils.py             # Motion, color filtering, annotations
│   └── weight_estimator.py  # Weight proxy calculation
├── models/
│   └── yolov8s.pt            # YOLO model weights
├── outputs/
│   └── sample_response.json  # Example API response
├── data/
│   └── (ignored videos)
├── README.md
├── requirements.txt
└── .gitignore


🎥 Large video files are intentionally excluded from the repository and provided via Google Drive.

🔧 Installation
1️⃣ Clone the Repository
git clone https://github.com/CHARANKONKIPUDI/bird-counting-weight-estimation-task.git
cd bird-counting-weight-estimation-task

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the Application
Start FastAPI Server
uvicorn app.main:app --reload

API Documentation

Open in browser:

http://127.0.0.1:8000/docs

📡 API Endpoints
🔹 Health Check

GET /health

Response:

{
  "status": "OK"
}

🔹 Analyze Video

POST /analyze_video

Request

Content-Type: multipart/form-data

Key: video

Value: .mp4 video file

Example

curl -X POST "http://127.0.0.1:8000/analyze_video" \
     -F "video=@sample_video.mp4"

📤 Sample Response
{
  "processed_seconds": 120,
  "average_visible_birds": 135.42,
  "average_weight_grams": 162.8,
  "bird_counts_over_time": [
    { "time_sec": 0, "visible_birds": 98 },
    { "time_sec": 1, "visible_birds": 104 }
  ],
  "annotated_video": "outputs/annotated_video.mp4"
}

🎯 Output Details

Green bounding boxes

Stable bird IDs

Estimated weight displayed per bird

Total estimated bird count overlay

🧪 Limitations

Weight estimation is proxy-based and requires calibration for real grams.

Performance depends on video resolution and lighting conditions.

Very small or heavily occluded birds may be missed.

🔮 Future Improvements

Real-world weight calibration

Species classification

Advanced tracking (DeepSORT / ByteTrack)

Real-time RTSP stream support

CSV / analytics export

✅ Conclusion

This project demonstrates a robust and practical solution for bird counting and relative weight estimation using modern computer vision techniques.
The system balances accuracy, performance, and explainability, making it suitable for real-world monitoring applications.

👤 Author

Candidate Name: KONKIPUDI MANI SAI CHARAN
Role Applied: ML / AI Engineer Intern
Company: Kuppismart Solutions (Livestockify)