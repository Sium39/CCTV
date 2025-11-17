import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QVBoxLayout,
    QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class VideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()

        # Video capture and properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_duration = 1.0 / self.fps

        # Original video dimensions
        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Display size maintaining exact aspect ratio (width fixed 1200)
        self.display_width = 1200
        self.display_height = int(self.img_h * self.display_width / self.img_w)

        # Detection and tracking models
        self.model = YOLO('yolo11s.pt')
        self.tracker = DeepSort(max_age=30, n_init=3)

        self.frame_counter = 0
        self.boxes_with_ids = []
        self.selected_person_id = None
        self.view_mode = "all"  # View all or a single focused person

        self.playback_speed = 1.0
        self.paused = False

        # Scale factors and offset (no offset needed since no letterboxing)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # UI setup
        self.video_label = QLabel()
        self.video_label.setFixedSize(self.display_width, self.display_height)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.mousePressEvent = self.mouse_press_event

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 40)  # 0.1x to 4x speed
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.change_speed)

        self.speed_label = QLabel("Speed: 1.0x")

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.speed_label)
        controls_layout.addWidget(self.speed_slider)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        interval_ms = max(1, int(self.frame_duration * 1000 / self.playback_speed))
        self.timer.start(interval_ms)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.setText("Resume" if self.paused else "Pause")

    def change_speed(self, val):
        self.playback_speed = val / 10.0
        self.speed_label.setText(f"Speed: {self.playback_speed:.1f}x")
        interval_ms = max(1, int(self.frame_duration * 1000 / self.playback_speed))
        self.timer.setInterval(interval_ms)

    def mouse_press_event(self, event):
        x = event.pos().x()
        y = event.pos().y()

        # Since no letterboxing, simply scale click coordinates to original image
        click_x = int(x * self.scale_x)
        click_y = int(y * self.scale_y)

        for tid, (x1, y1, x2, y2) in self.boxes_with_ids:
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                if self.view_mode == "all":
                    self.selected_person_id = tid
                    self.view_mode = "single"
                    print(f"Switched to SINGLE view: Person {tid}")
                elif self.view_mode == "single":
                    if self.selected_person_id == tid:
                        self.selected_person_id = None
                        self.view_mode = "all"
                        print("Switched back to ALL view")
                    else:
                        self.selected_person_id = tid
                        print(f"Changed selection to Person {tid}")
                break
        else:
            if self.view_mode == "single":
                self.selected_person_id = None
                self.view_mode = "all"
                print("Clicked outside any person: switched to ALL view")

    def next_frame(self):
        if self.paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return

        self.frame_counter += 1
        img_h, img_w = frame.shape[:2]

        # Resize for detection speed
        det_width = 640
        det_height = int(img_h * det_width / img_w)
        det_frame = cv2.resize(frame, (det_width, det_height))

        results = self.model(det_frame)[0]

        scale_x_det = img_w / det_width
        scale_y_det = img_h / det_height

        detections = []
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            label = self.model.names[int(cls_id)]
            if label == 'person' and conf > 0.3:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())

                # Clip and scale to original frame size
                x1 = max(0, min(img_w - 1, int(x1 * scale_x_det)))
                y1 = max(0, min(img_h - 1, int(y1 * scale_y_det)))
                x2 = max(0, min(img_w - 1, int(x2 * scale_x_det)))
                y2 = max(0, min(img_h - 1, int(y2 * scale_y_det)))

                if x2 > x1 and y2 > y1:
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    detections.append((bbox, conf.cpu().item(), label))

        self.tracks = self.tracker.update_tracks(detections, frame=frame)

        self.boxes_with_ids = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            self.boxes_with_ids.append((tid, (x1, y1, x2, y2)))

        self.draw_frame_info(frame)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Scale pixmap exactly to display size (no aspect ratio correction)
        pixmap = pixmap.scaled(self.display_width, self.display_height, Qt.IgnoreAspectRatio)
        self.video_label.setPixmap(pixmap)

        # Calculate scaling factors for mouse click mapping (no offset needed)
        self.scale_x = self.img_w / self.display_width
        self.scale_y = self.img_h / self.display_height
        self.offset_x = 0
        self.offset_y = 0

    def draw_frame_info(self, frame):
        line_height = 25
        x_pos = 10
        y_pos = 30

        cv2.rectangle(frame, (0, 0), (320, 170), (30, 30, 30), -1)

        info = [
            f"Frame: {self.frame_counter}",
            f"Detections: {len(self.boxes_with_ids)}",
            f"Tracked: {', '.join(str(tid) for tid, _ in self.boxes_with_ids)}",
            f"Speed: {self.playback_speed:.1f}x",
            f"Paused: {'Yes' if self.paused else 'No'}",
            f"Mode: {'FOCUSED on Person ' + str(self.selected_person_id) if self.view_mode == 'single' else 'ALL PERSONS'}"
        ]

        for i, text in enumerate(info):
            cv2.putText(frame, text, (x_pos, y_pos + i * line_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.view_mode == 'all':
            for tid, (x1, y1, x2, y2) in self.boxes_with_ids:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            for tid, (x1, y1, x2, y2) in self.boxes_with_ids:
                if tid == self.selected_person_id:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"Selected Person {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    break

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer("D:/CCTV/cctv2.mp4")  # Modify as needed
    player.setWindowTitle("CCTV Project with Person Tracking")
    player.resize(1220, int(player.display_height * 1220 / player.display_width))
    player.show()
    sys.exit(app.exec_())
