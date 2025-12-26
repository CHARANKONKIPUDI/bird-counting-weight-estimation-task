import numpy as np
from sort import Sort


class BirdTracker:
    def __init__(self):
        """
        SORT-based tracker tuned for birds:
        - Birds move fast and may disappear briefly
        - We allow short gaps but avoid ghost tracks
        """
        self.tracker = Sort(
            max_age=10,        # frames to keep track alive without detection
            min_hits=1,        # birds may appear briefly
            iou_threshold=0.15 # low IOU because birds flap & shift fast
        )

    def update(self, detections):
        """
        Input :
            detections -> np.array [[x1,y1,x2,y2,conf], ...]

        Output:
            tracks -> [(x1,y1,x2,y2,track_id), ...]
        """

        # No detections → no tracks
        if detections is None or len(detections) == 0:
            return []

        # Ensure correct dtype & shape for SORT
        detections = np.asarray(detections, dtype=np.float32)

        # Run tracker
        tracked = self.tracker.update(detections)

        # Format output
        output = []
        for t in tracked:
            x1, y1, x2, y2, track_id = t
            output.append((
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                int(track_id)
            ))

        return output
