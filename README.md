# Traffic Instance Segmentation Dashboard

Web dashboard for comparing YOLOv26s-seg and RF-DETR Small on traffic instance
segmentation, qualitative inference, and red-light violation detection.

## Contents

- `app.py`: Flask backend and inference routes.
- `templates/`: Flask pages for introduction, metrics, qualitative results, and inference.
- `src/`: Vite/React UI source from the original prototype.
- `static/test_images/`: sample images for local inference.
- `models/README.md`: notes for model checkpoints that are intentionally not committed.

## Run Flask App

1. Create and activate a Python virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Put model checkpoints in `models/`:
   - `yolov26s_seg.pt`
   - `RF-DETR_Small.pt`
4. Start the backend:
   `python app.py`
5. Open `http://localhost:5000`.

## Run React Prototype

1. Install Node.js dependencies:
   `npm install`
2. Start Vite:
   `npm run dev`
3. Open `http://localhost:3000`.

## Notes

Large model checkpoints, generated result files, logs, virtual environments,
`node_modules`, and build outputs are ignored so the repository stays suitable
for GitHub.
