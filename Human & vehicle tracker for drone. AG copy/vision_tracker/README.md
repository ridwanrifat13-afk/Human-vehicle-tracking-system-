# Vision Tracker

A local human and ground vehicle tracking system designed with a drone-ready architecture.

## Architecture

- `sources/`: video acquisition interface and webcam source
- `detection/`: Ultralytics YOLOv8 detection module
- `tracking/`: ByteTrack-style tracking module with ID assignment
- `prediction/`: Kalman filter motion prediction
- `control/`: drone command interface placeholder
- `visualization/`: Tkinter-based UI

## Run locally

1. Install dependencies:

```bash
cd "Human & vehicle tracker for drone. AG/vision_tracker"
pip install -r requirements.txt
```

2. Run the tracker from the project folder:

```bash
python main.py
```

3. If you want to run the built-in tests:

```bash
python -m unittest discover -s tests
```

## Notes

- Uses webcam input via OpenCV.
- Detection is restricted to `person`, `car`, `bus`, `truck`, `motorcycle`, and includes a placeholder for `tank`.
- `NullController` is a future hook for drone control.
- The UI is implemented in Tkinter so it stays the same when a drone source is added later.
- You can click a bounding box to select one or more targets; when targets are selected, only the selected boxes are shown.
- Press `C` to clear the selection and show all tracked boxes again.
- Coordinates of selected targets are shown in the top-right corner.
- Logging is enabled for startup, frame errors, and runtime exceptions.
- If the webcam does not open, make sure macOS camera permissions are granted for your terminal and Python interpreter.

- Uses webcam input via OpenCV.
- Detection is restricted to `person`, `car`, `bus`, `truck`, `motorcycle`, and includes a placeholder for `tank`.
- `NullController` is a future hook for drone control.
- The UI is implemented in Tkinter so it stays the same when a drone source is added later.
