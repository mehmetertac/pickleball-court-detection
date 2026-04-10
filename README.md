# Pickleball court detection

YOLO-pose court keypoint parsing, temporal smoothing (`CourtStateManager`), and homography helpers for mapping pixels to court coordinates.

```bash
pip install pickleball-court-detection
```

Environment:

- `PICKLEBALL_COURT_MODEL` — path to court weights (default: `models/court_detector/best.pt` under this package’s install root).
