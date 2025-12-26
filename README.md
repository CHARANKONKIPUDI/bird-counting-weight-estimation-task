🐦 Bird Counting & Weight Estimation System
📌 Overview

This project implements an end-to-end bird counting and weight estimation system using computer vision and deep learning.
The system detects birds from video, tracks them over time to avoid double counting, estimates flock size, and provides a proxy-based weight estimation.

A FastAPI backend exposes an API to upload videos and returns:

Estimated bird count over time

Average visible birds

Average estimated weight

Annotated output video

🚀 Features

YOLOv8-based bird detection

SORT-based multi-object tracking (stable IDs)

Hybrid bird counting:

Detection + tracking (sparse scenes)

Foreground area estimation (dense flocks)

Static and moving birds supported

Red object rejection (to reduce false positives)

Green bounding boxes with IDs

Per-bird weight proxy displayed on boxes

REST API using FastAPI

Annotated video output

🧠 Approach & Methodology
1️⃣ Detection

Uses YOLOv8 to detect bird-like objects.

Applies geometry filters (area & aspect ratio).

Red-colored objects (feeders, lights, clothes) are rejected using HSV color analysis.

Motion is optional (static birds are still detected).

2️⃣ Tracking

Uses SORT (Simple Online Realtime Tracking).

Assigns stable IDs to birds.

Prevents double counting.

Handles short occlusions and brief disappearances.

3️⃣ Counting Logic

A hybrid estimation strategy is used:

Visible tracked birds anchor the count.

Foreground area estimation helps in dense flock scenarios.

Temporal smoothing prevents sudden spikes.

Safety constraints:

Estimated count ≥ visible birds

Estimated count ≥ minimum threshold

4️⃣ Weight Estimation (Proxy-Based)

Weight is estimated from bounding box area:

weight ≈ pixel_area × scale_factor


This is a relative proxy, not ground-truth weight.

Calibration with real bird measurements can convert it to actual grams.

⚠️ Important Note on Weight

The reported bird weight is an approximation based on bounding box area.
It should be interpreted as a relative indicator, not a medical or biological measurement.

📂 Project Structure
app/
├── main.py                 # FastAPI application
├── detector.py             # YOLO-based bird detection
├── tracker.py              # SORT-based tracking
├── utils.py                # Motion, color filtering, annotation
├── weight_estimator.py     # Weight proxy calculation
models/
├── yolov8s.pt              # YOLO model weights
outputs/
├── annotated_video.mp4     # Output video
data/
├── input_video.mp4
README.md
requirements.txt

🔧 Installation
1️⃣ Clone the Repository
git clone <your-repo-url>
cd bird-counting

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the Application
Start FastAPI Server
uvicorn app.main:app --reload

API Docs

Open in browser:

http://127.0.0.1:8000/docs

📡 API Endpoints
🔹 Health Check
GET /health


Response:

{ "status": "OK" }

🔹 Analyze Video
POST /analyze_video


Request:

Multipart form-data

Key: video

Value: .mp4 video file

Example using curl:

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

🎥 Output

outputs/annotated_video.mp4

Green bounding boxes

Stable IDs

Estimated weight on each box

Total estimated birds displayed

🧪 Limitations

Weight estimation is proxy-based (requires calibration for real grams).

Performance depends on video resolution and lighting.

YOLO bird class may miss very small or heavily occluded birds.

🔮 Future Improvements

True weight calibration using real-world measurements

Species classification

Improved occlusion handling (DeepSORT / ByteTrack)

Real-time RTSP stream support

Export CSV analytics

✅ Conclusion

This system demonstrates a robust, practical solution for bird counting and relative weight estimation using modern computer vision techniques.
The design balances accuracy, performance, and explainability, making it suitable for real-world monitoring scenarios.

👤 Author

Candidate Name: KONKIPUDI MANI SAI CHARAN
Role Applied: ML / AI Engineer Intern
Company: Kuppismart Solutions (Livestockify)