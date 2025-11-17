from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

model = YOLO('yolo11s.pt')
tracker = DeepSort(max_age=30, n_init=3)

selected_person_id = None
boxes_with_ids = []
scale_x, scale_y = 1.0, 1.0
view_mode = 'all'
frame_counter = 0
DETECTION_INTERVAL = 1  # Tune this number: higher = faster, less frequent detection

def mouse_callback(event, x, y, flags, param):
    global selected_person_id, boxes_with_ids, scale_x, scale_y, view_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        clicked_id = None
        for (tid, (x1, y1, x2, y2)) in boxes_with_ids:
            if x1 <= original_x <= x2 and y1 <= original_y <= y2:
                clicked_id = tid
                break
        if clicked_id is not None:
            if view_mode == 'all':
                selected_person_id = clicked_id
                view_mode = 'single'
                print(f"Switched to SINGLE view: Person {selected_person_id}")
            elif view_mode == 'single':
                if selected_person_id == clicked_id:
                    view_mode = 'all'
                    selected_person_id = None
                    print("Switched back to ALL view")
                else:
                    selected_person_id = clicked_id
                    print(f"Changed selection to Person {selected_person_id}")
        else:
            if view_mode == 'single':
                view_mode = 'all'
                selected_person_id = None
                print("Switched back to ALL view (clicked outside)")

def detect_and_interactive_tag(video_path=r"D:\CCTV\cctv2.mp4", display_width=1200):
    global boxes_with_ids, selected_person_id, scale_x, scale_y, view_mode, frame_counter

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    cv2.namedWindow('Person Detection Interactive Tagging')
    cv2.setMouseCallback('Person Detection Interactive Tagging', mouse_callback)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        height, width = frame.shape[:2]

        if view_mode == 'all' or (view_mode == 'single' and (frame_counter % DETECTION_INTERVAL == 0)):
            # Resize frame for detection to speed up (e.g., width 640)
            det_width = 640
            det_height = int(height * det_width / width)
            det_frame = cv2.resize(frame, (det_width, det_height))

            results = model(det_frame)[0]

            detections = []
            # Map coords back to original frame size
            scale_x_det = width / det_width
            scale_y_det = height / det_height

            for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                label = model.names[int(cls_id)]
                if label == 'person' and conf > 0.3:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    x1 = int(x1 * scale_x_det)
                    y1 = int(y1 * scale_y_det)
                    x2 = int(x2 * scale_x_det)
                    y2 = int(y2 * scale_y_det)
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox, conf.cpu().item(), label))

            # Update tracker with detections
            tracks = tracker.update_tracks(detections, frame=frame)
        else:
            # Predict the tracks without new detection
            tracks = tracker.predict()

        boxes_with_ids = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            boxes_with_ids.append((tid, (x1, y1, x2, y2)))

        if view_mode == 'all':
            for tid, (x1, y1, x2, y2) in boxes_with_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            drew_selected = False
            for tid, (x1, y1, x2, y2) in boxes_with_ids:
                if tid == selected_person_id:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"Selected Person {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    drew_selected = True
                    break
            if not drew_selected:
                view_mode = 'all'
                selected_person_id = None
                print("Selected person lost, switched back to ALL view")

        aspect_ratio = height / width
        display_height = int(display_width * aspect_ratio)

        scale_x = width / display_width
        scale_y = height / display_height

        resized_frame = cv2.resize(frame, (display_width, display_height))
        cv2.imshow('Person Detection Interactive Tagging', resized_frame)

        elapsed = time.time() - start_time
        wait_time = max(1, int((1/30 - elapsed) * 1000))  # assuming target 30 FPS

        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            print("Exit requested by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

detect_and_interactive_tag()

