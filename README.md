# Transmission Tower Equipment Defect Detection (Detection + Segmentation)

Computer vision inspection system for **transmission tower equipment defects** including **cracks, corrosion, loose components, and surface damage**.  
Built for **field robustness** (weather, lighting, viewing angles) and preventive maintenance.

Project page: https://drsaqibbhatti.com/projects/tower-equipment-detection.html

---

## Overview
Manual transmission tower inspection is dangerous, time-consuming, and difficult to scale. This project applies deep learning to **localize and classify multiple defect types** from tower imagery to support safer and earlier maintenance decisions.

---

## Problem
- Multiple defect types (cracks, corrosion, loose components, surface damage).
- Challenging field conditions: weather, lighting variation, and diverse viewing angles.
- Need for early detection to prevent failures and reduce risky manual inspections. 

---

## Solution
- **Detection + segmentation** pipeline for precise localization of defects and damaged regions. 
- **Multi-class detection** across multiple failure modes. 
- **Data augmentation** strategy to improve robustness under field conditions. 

---

## My Role
- Designed the end-to-end computer vision pipeline for defect detection from images. 
- Developed deep learning models for detection + segmentation. 
- Validated performance across multiple defect types and conditions. 

---

## Tech Stack
- **Python**
- **PyTorch**
- **YOLO**
- **OpenCV** 

---

## Output
- Defect localization via bounding boxes (detection) and masks (segmentation)
- Class labels for defect type (multi-class)
- Visual overlays for verification (inspection-friendly)

> Note: Some defect taxonomy details and datasets may be confidential.

