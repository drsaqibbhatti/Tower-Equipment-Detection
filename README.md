# Power Line Galloping Detection System (Video Monitoring + Real-Time Alerts)

Video-based monitoring system for detecting **abnormal oscillations (galloping)** in overhead power transmission lines.  
Designed for **real-time detection + alerts** to support proactive maintenance and improve grid stability.

Project page: https://drsaqibbhatti.com/projects/power-line-galloping.html

---

## Overview

### Problem
- Power line galloping can cause outages and equipment damage.
- Manual monitoring is impractical for large transmission networks.
- Early detection is needed for preventive maintenance.
- Outdoor conditions are challenging (wind/weather/distance/camera angle).

### Solution
- Video-based detection using motion/time-series analysis (CV + DL).
- Real-time monitoring with an alert-ready pipeline.
- Robust handling of environmental variation and camera positioning.

---

## Key Features
- Real-time inference on **video/webcam**
- Training script included (YOLO-style dataset)
- Export to ONNX (deployment-friendly)
- Optional **FastAPI** endpoint for integration

---

## Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy

---

## Repository Structure
