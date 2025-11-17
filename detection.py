import sys
import cv2
import numpy as np
import torch
import torchreid
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class ReIDModel:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        ).to(self.device)
        self.model.eval()
        self.transform = torchreid.data.transforms.build_transforms(
            height=256, width=128, transforms=[], norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225]
        )


    def extract_embedding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 256))
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(img_t).cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding[0]


class DetectionThread(QThread):
    results_ready = pyqtSignal(object, object, object)


    def __init__(self, model, tracker, reid_model):
        super().__init__()
        self.model = model
        self.tracker = tracker
        self.reid_model = reid_model
        self.frame = None
        self.running = True
        self.frame_count = 0
        self.last_tracks = []
        self.last_embs = {}


    def run(self):
        while self.running:
            if self.frame is not None:
                self.frame_count += 1
                frame = self.frame
                self.frame = None


                if self.frame_count % 2 == 0:
                    img_h, img_w = frame.shape[:2]
                    det_width = 640
                    det_height = int(img_h * det_width / img_w)
                    det_frame = cv2.resize(frame, (det_width, det_height))


                    results = self.model(det_frame, device='cuda')[0]


                    scale_x_det = img_w / det_width
                    scale_y_det = img_h / det_height


                    detections = []
                    crops = []
                    for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                        label = self.model.names[int(cls_id)]
                        if label == 'person' and conf > 0.3:
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            x1 = max(0, min(img_w - 1, int(x1 * scale_x_det)))
                            y1 = max(0, min(img_h - 1, int(y1 * scale_y_det)))
                            x2 = max(0, min(img_w - 1, int(x2 * scale_x_det)))
                            y2 = max(0, min(img_h - 1, int(y2 * scale_y_det)))


                            if x2 > x1 and y2 > y1:
                                bbox = [x1, y1, x2 - x1, y2 - y1]
                                detections.append((bbox, conf.cpu().item(), label))
                                crops.append(frame[y1:y2, x1:x2])


                    tracks = self.tracker.update_tracks(detections, frame=frame)


                    embs = {}
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        tid = track.track_id
                        ltrb = list(map(int, track.to_ltrb()))
                        emb = None
                        for bbox, crop in zip([d[0] for d in detections], crops):
                            iou = self.iou(ltrb, bbox)
                            if iou > 0.7 and crop.size > 0:
                                emb = self.reid_model.extract_embedding(crop)
                                break
                        if emb is not None:
                            embs[tid] = emb


                    self.last_tracks = tracks
                    self.last_embs = embs
                    self.results_ready.emit(tracks, frame, embs)
                else:
                    self.results_ready.emit(self.last_tracks, frame, self.last_embs)


            self.msleep(10)


    def update_frame(self, frame):
        if self.frame is None:
            self.frame = frame


    def stop(self):
        self.running = False
        self.wait()


    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou


class VideoWidget(QWidget):
    person_selected = pyqtSignal(int, object)  # camera_id, embedding or None


    def __init__(self, video_path, camera_id):
        super().__init__()


        self.camera_id = camera_id


        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_duration = 1.0 / self.fps
        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        self.display_width = 1000
        self.display_height = int(self.img_h * self.display_width / self.img_w)


        # Set up video writer to save output video with detections
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f'output_camera_{self.camera_id}.mp4'
        self.out_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.img_w, self.img_h))


        self.model = YOLO('yolo11s.pt')
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.reid_model = ReIDModel(device='cuda')


        self.frame_counter = 0
        self.boxes_with_ids = []
        self.embeddings = {}


        self.selected_embedding = None
        self.selected_person_id = None  # ID of the selected person in THIS camera


        # Scale factors - calculated AFTER each frame like in your sample
        self.scale_x = 1.0
        self.scale_y = 1.0


        self.video_label = QLabel()
        self.video_label.setFixedSize(self.display_width, self.display_height)
        self.video_label.setStyleSheet("background-color: black;")
        
        # DIRECT mouse event assignment like in your sample
        self.video_label.mousePressEvent = self.mouse_press_event


        self.det_thread = DetectionThread(self.model, self.tracker, self.reid_model)
        self.det_thread.results_ready.connect(self.update_tracks)
        self.det_thread.start()


        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        interval_ms = max(1, int(self.frame_duration * 1000))
        self.timer.start(interval_ms)


    def mouse_press_event(self, event):
        x = event.pos().x()
        y = event.pos().y()


        # Convert click coordinates to original image coordinates (like your sample)
        click_x = int(x * self.scale_x)
        click_y = int(y * self.scale_y)


        print(f"DEBUG: Click at display coords: ({x}, {y}), image coords: ({click_x}, {click_y})")


        # Check if click hits any bounding box
        for tid, (x1, y1, x2, y2) in self.boxes_with_ids:
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                if self.selected_person_id == tid:
                    # Clicking on already selected person - deselect
                    self.selected_person_id = None
                    print(f"DEBUG: Person {tid} deselected")
                    self.person_selected.emit(self.camera_id, None)
                else:
                    # Select this person
                    self.selected_person_id = tid
                    emb = self.embeddings.get(tid, None)
                    if emb is not None:
                        print(f"DEBUG: Person {tid} selected")
                        self.person_selected.emit(self.camera_id, emb)
                    else:
                        print(f"DEBUG: Person {tid} clicked but no embedding")
                return
        
        # Click outside any person - clear selection
        if self.selected_person_id is not None:
            self.selected_person_id = None
            print("DEBUG: Clicked outside - clearing selection")
            self.person_selected.emit(self.camera_id, None)


    def set_global_selected(self, embedding):
        self.selected_embedding = embedding
        print(f"DEBUG: Camera {self.camera_id} - Global embedding {'set' if embedding is not None else 'cleared'}")


    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.det_thread.stop()
            self.out_writer.release()  # Release writer when done saving video
            return


        self.frame_counter += 1
        self.det_thread.update_frame(frame)
        self.draw_frame_info(frame)  # This draws detections & info on the frame


        # Write the output frame with drawings to the video file
        self.out_writer.write(frame)


        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(self.display_width, self.display_height, Qt.IgnoreAspectRatio)
        self.video_label.setPixmap(pixmap)


        # Calculate scaling factors AFTER setting pixmap (like your sample)
        self.scale_x = self.img_w / self.display_width
        self.scale_y = self.img_h / self.display_height


    def update_tracks(self, tracks, frame, embs):
        self.boxes_with_ids = []
        self.embeddings = embs
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            self.boxes_with_ids.append((tid, (x1, y1, x2, y2)))


    def draw_frame_info(self, frame):
        line_height = 15
        x_pos = 10
        y_pos = 20


        cv2.rectangle(frame, (0, 0), (340, 140), (30, 30, 30), -1)


        info = [
            f"Frame: {self.frame_counter}",
            f"Detections: {len(self.boxes_with_ids)}",
            f"Tracked: {', '.join(str(tid) for tid, _ in self.boxes_with_ids)}",
            f"Selected: {self.selected_person_id if self.selected_person_id else 'None'}"
        ]


        for i, text in enumerate(info):
            cv2.putText(frame, text, (x_pos, y_pos + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        # Draw all bounding boxes
        for tid, (x1, y1, x2, y2) in self.boxes_with_ids:
            # Determine color and thickness
            if tid == self.selected_person_id:
                # This person is selected in this camera - RED
                color = (0, 0, 255)
                thickness = 3
                label = f"SELECTED P{tid}"
            else:
                # Check if this person matches global selection from other camera
                emb = self.embeddings.get(tid, None)
                if emb is not None and self.selected_embedding is not None:
                    cos_sim = np.dot(emb, self.selected_embedding)
                    if cos_sim > 0.7:
                        # Matched person from other camera - RED
                        color = (0, 0, 255)
                        thickness = 3
                        label = f"MATCHED P{tid}"
                    else:
                        # Regular person - GREEN
                        color = (0, 255, 0)
                        thickness = 2
                        label = f"P{tid}"
                else:
                    # Regular person - GREEN
                    color = (0, 255, 0)
                    thickness = 2
                    label = f"P{tid}"


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


class MainWindow(QWidget):
    def __init__(self, video_paths):
        super().__init__()
        self.setWindowTitle("CCTV Multi-View Player")
        self.layout = QGridLayout()
        self.setLayout(self.layout)


        self.players = []
        rows = 1
        cols = len(video_paths)
        self.selected_embedding = None


        for i, path in enumerate(video_paths):
            player = VideoWidget(path, camera_id=i)
            player.person_selected.connect(self.on_person_selected)
            self.players.append(player)
            r = i // cols
            c = i % cols
            self.layout.addWidget(player.video_label, r, c)


        self.resize(cols * 1080, rows * 720)


    def on_person_selected(self, camera_id, embedding):
        print(f"DEBUG: MainWindow - Camera {camera_id} selection changed")
        self.selected_embedding = embedding
        
        # Update all players with the new global selection
        for player in self.players:
            player.set_global_selected(self.selected_embedding)


if __name__ == '__main__':
    app = QApplication(sys.argv)


    video_paths = [
        "D:/CCTV/19번 CAM 20250917_10시 54분(3분).mp4"
    ]


    main_win = MainWindow(video_paths)
    main_win.show()


    sys.exit(app.exec_())
