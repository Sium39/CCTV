# import sys
# import cv2
# import numpy as np
# import torch
# import torchreid
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout
# from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
# from PyQt5.QtGui import QImage, QPixmap
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from PIL import Image

# class ReIDModel:
#     def __init__(self, device='cuda'):
#         self.device = device
#         self.model = torchreid.models.build_model(
#             name='osnet_x1_0', num_classes=1000, pretrained=True
#         ).to(self.device)
#         self.model.eval()
#         # Unpack transforms: (train_tfms, test_tfms)
#         _, test_transforms = torchreid.data.transforms.build_transforms(
#             height=256, width=128, transforms=[],
#             norm_mean=[0.485, 0.456, 0.406],
#             norm_std=[0.229, 0.224, 0.225]
#         )
#         self.transform = test_transforms

#     def extract(self, img: np.ndarray):
#         # img is BGR NumPy array
#         if img is None or img.size == 0:
#             return None
#         h, w = img.shape[:2]
#         if h < 64 or w < 32:
#             return None
#         # Convert to RGB PIL Image
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         pil = Image.fromarray(img_rgb)
#         # Apply transforms
#         t = self.transform(pil).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             emb = self.model(t).cpu().numpy()[0]
#         return emb / np.linalg.norm(emb)


# class DetectionThread(QThread):
#     results_ready = pyqtSignal(object, object, object)

#     def __init__(self, model, tracker, reid):
#         super().__init__()
#         self.model, self.tracker, self.reid = model, tracker, reid
#         self.frame = None
#         self.running = True
#         self.frame_count = 0
#         self.last = ([], {})

#     def run(self):
#         while self.running:
#             if self.frame is None:
#                 self.msleep(10)
#                 continue
#             self.frame_count += 1
#             frame = self.frame.copy()
#             self.frame = None

#             if self.frame_count % 2 == 0:
#                 h, w = frame.shape[:2]
#                 det_w = 640
#                 det_h = int(h * det_w / w)
#                 det = cv2.resize(frame, (det_w, det_h))
#                 res_list = self.model(det, device='cuda')
#                 res = res_list[0] if isinstance(res_list, list) else res_list

#                 sx, sy = w / det_w, h / det_h
#                 dets, crops = [], []
#                 for box, cls, conf in zip(
#                         res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
#                     if self.model.names[int(cls)] == 'person' and conf > 0.5:
#                         x1, y1, x2, y2 = map(int, box.cpu().numpy())
#                         x1, y1 = int(x1 * sx), int(y1 * sy)
#                         x2, y2 = int(x2 * sx), int(y2 * sy)
#                         if x2 > x1 and y2 > y1 and x2 - x1 >= 32 and y2 - y1 >= 64:
#                             dets.append(([x1, y1, x2 - x1, y2 - y1],
#                                          conf.cpu().item(), 'person'))
#                             crops.append(frame[y1:y2, x1:x2])

#                 tracks = self.tracker.update_tracks(dets, frame=frame)
#                 embs = {}
#                 for t in tracks:
#                     if not t.is_confirmed():
#                         continue
#                     tid = t.track_id
#                     ltrb = list(map(int, t.to_ltrb()))
#                     best_iou, best_crop = 0, None
#                     for (bx, by, bw, bh), crop in zip(
#                             [d[0] for d in dets], crops):
#                         xA = max(ltrb[0], bx)
#                         yA = max(ltrb[1], by)
#                         xB = min(ltrb[2], bx + bw)
#                         yB = min(ltrb[3], by + bh)
#                         inter = max(0, xB - xA) * max(0, yB - yA)
#                         areaA = (ltrb[2] - ltrb[0]) * (ltrb[3] - ltrb[1])
#                         areaB = bw * bh
#                         iou = inter / float(areaA + areaB - inter + 1e-5)
#                         if iou > best_iou and iou > 0.5:
#                             best_iou, best_crop = iou, crop
#                     emb = self.reid.extract(best_crop) if best_crop is not None else None
#                     if emb is not None:
#                         embs[tid] = emb

#                 self.last = (tracks, embs)
#             tracks, embs = self.last
#             self.results_ready.emit(tracks, frame, embs)
#             self.msleep(10)

#     def update_frame(self, f):
#         if self.frame is None:
#             self.frame = f

#     def stop(self):
#         self.running = False
#         self.wait()

# class VideoWidget(QWidget):
#     person_selected = pyqtSignal(int, object)

#     def __init__(self, path, cam_id):
#         super().__init__()
#         self.cam_id = cam_id

#         self.cap = cv2.VideoCapture(path)
#         self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
#         self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.dw, self.dh = 700, int(self.h * 700 / self.w)

#         self.model = YOLO('yolo11s.pt')
#         self.tracker = DeepSort(max_age=30, n_init=3)
#         self.reid = ReIDModel()

#         # Initialize state
#         self.tracks = []
#         self.boxes = []
#         self.embs = {}
#         self.global_emb = None
#         self.selected_id = None

#         self.sim_th, self.high_th = 0.6, 0.75

#         self.label = QLabel()
#         self.label.setFixedSize(self.dw, self.dh)
#         self.label.setStyleSheet("background:black;")
#         self.label.mousePressEvent = self.on_click

#         self.thread = DetectionThread(self.model, self.tracker, self.reid)
#         self.thread.results_ready.connect(self.update_tracks)
#         self.thread.start()

#         self.timer = QTimer()
#         self.timer.timeout.connect(self.next_frame)
#         self.timer.start(int(1000 / self.fps))

#     def on_click(self, e):
#         x, y = e.pos().x(), e.pos().y()
#         gx, gy = int(x * self.w / self.dw), int(y * self.h / self.dh)
#         for tid, (x1, y1, x2, y2) in self.boxes:
#             if x1 <= gx <= x2 and y1 <= gy <= y2:
#                 emb = self.embs.get(tid)
#                 self.selected_id = None if self.selected_id == tid else tid
#                 self.person_selected.emit(
#                     self.cam_id, None if self.selected_id is None else emb)
#                 return
#         if self.selected_id is not None:
#             self.selected_id = None
#             self.person_selected.emit(self.cam_id, None)

#     def set_global(self, emb):
#         self.global_emb = emb

#     def update_tracks(self, tracks, frame, embs):
#         self.tracks, self.embs = tracks, embs
#         self.boxes = []
#         for t in self.tracks:
#             if not t.is_confirmed():
#                 continue
#             tid = t.track_id
#             x1, y1, x2, y2 = map(int, t.to_ltrb())
#             self.boxes.append((tid, (x1, y1, x2, y2)))

#     def next_frame(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             self.timer.stop()
#             self.thread.stop()
#             return

#         self.thread.update_frame(frame)
#         self.draw(frame)

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb.shape
#         img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#         pm = QPixmap.fromImage(img).scaled(
#             self.dw, self.dh, Qt.IgnoreAspectRatio)
#         self.label.setPixmap(pm)

#     def draw(self, frame):
#         cv2.rectangle(frame, (0, 0), (400, 160), (30, 30, 30), -1)
#         for tid, (x1, y1, x2, y2) in self.boxes:
#             color, th, lbl = (0, 255, 0), 2, f"P{tid}"
#             if tid == self.selected_id:
#                 color, th, lbl = (0, 0, 255), 3, f"SEL{tid}"
#             elif self.global_emb is not None and tid in self.embs:
#                 sim = np.dot(self.embs[tid], self.global_emb)
#                 if sim > self.high_th:
#                     color, th, lbl = (0, 0, 255), 3, f"M{tid}({sim:.2f})"
#                 elif sim > self.sim_th:
#                     color, th, lbl = (0, 165, 255), 3, f"?{tid}({sim:.2f})"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)
#             cv2.putText(frame, lbl, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# class MainWindow(QWidget):
#     def __init__(self, paths):
#         super().__init__()
#         self.setWindowTitle("CCTV Multi-View Player")
#         self.grid = QGridLayout(self)
#         self.widgets = []
#         for i, p in enumerate(paths):
#             w = VideoWidget(p, i)
#             w.person_selected.connect(self.on_select)
#             self.widgets.append(w)
#             self.grid.addWidget(w.label, 0, i)
#         self.resize(len(paths) * 720, 520)

#     def on_select(self, cid, emb):
#         for w in self.widgets:
#             w.set_global(emb)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     mw = MainWindow(["D:/CCTV/video1.mp4", "D:/CCTV/video3.mp4"])
#     mw.show()
#     sys.exit(app.exec_())
import sys
import cv2
import numpy as np
import torch
import torchreid
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
from collections import deque
from threading import Lock
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe frame buffer with explicit synchronization"""
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
        self.lock = Lock()
        self.condition = QWaitCondition()
        self.mutex = QMutex()
    
    def put(self, frame, frame_id):
        with self.lock:
            if frame is not None:
                self.buffer.append((frame_id, frame.copy()))
                logger.debug(f"Frame {frame_id} added to buffer. Buffer size: {len(self.buffer)}")
            self.condition.wakeAll()
    
    def get(self, timeout_ms=100):
        """Get oldest frame from buffer. Returns (frame_id, frame) or (None, None) if empty"""
        with self.lock:
            if self.buffer:
                frame_id, frame = self.buffer.popleft()
                logger.debug(f"Frame {frame_id} retrieved from buffer. Remaining: {len(self.buffer)}")
                return frame_id, frame
        return None, None
    
    def is_empty(self):
        with self.lock:
            return len(self.buffer) == 0
    
    def clear(self):
        with self.lock:
            self.buffer.clear()


class TrackState:
    """Manages track lifecycle and embedding storage"""
    def __init__(self):
        self.active_tracks = {}
        self.lock = Lock()
        self.max_track_age = 120  # frames
    
    def update_embedding(self, track_id, embedding, timestamp, confidence=1.0):
        with self.lock:
            self.active_tracks[track_id] = {
                'embedding': embedding,
                'timestamp': timestamp,
                'confidence': confidence
            }
            logger.debug(f"Updated embedding for track {track_id} (confidence: {confidence:.3f})")
    
    def get_embedding(self, track_id):
        with self.lock:
            if track_id in self.active_tracks:
                return self.active_tracks[track_id]['embedding']
        return None
    
    def get_all_embeddings(self, current_timestamp):
        """Get all valid embeddings and remove stale ones"""
        with self.lock:
            # Remove stale tracks
            stale_ids = [tid for tid, data in self.active_tracks.items() 
                        if current_timestamp - data['timestamp'] > self.max_track_age]
            
            for tid in stale_ids:
                del self.active_tracks[tid]
                logger.debug(f"Removed stale track {tid}")
            
            result = {tid: data['embedding'] for tid, data in self.active_tracks.items()}
            logger.debug(f"Active tracks with embeddings: {len(result)}")
            return result
    
    def clear_track(self, track_id):
        with self.lock:
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
                logger.debug(f"Cleared track {track_id}")


class ReIDModel:
    """Re-identification model with batch inference support and error handling"""
    def __init__(self, device='cuda'):
        self.device = device
        try:
            self.model = torchreid.models.build_model(
                name='osnet_x1_0', num_classes=1000, pretrained=True
            ).to(self.device)
            self.model.eval()
            logger.info(f"ReID model loaded on device: {device}")
            
            _, test_transforms = torchreid.data.transforms.build_transforms(
                height=256, width=128, transforms=[],
                norm_mean=[0.485, 0.456, 0.406],
                norm_std=[0.229, 0.224, 0.225]
            )
            self.transform = test_transforms
        except Exception as e:
            logger.error(f"Failed to load ReID model: {e}")
            raise

    def extract(self, img: np.ndarray):
        """Extract single embedding with quality validation"""
        try:
            if img is None or img.size == 0:
                return None, 0.0
            
            h, w = img.shape[:2]
            if h < 64 or w < 32:
                logger.debug(f"Rejected crop: too small ({h}x{w})")
                return None, 0.0
            
            # Quality checks
            if h / w < 0.5 or h / w > 3.0:
                logger.debug("Rejected crop: extreme aspect ratio")
                return None, 0.3
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            
            t = self.transform(pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(t).cpu().numpy()[0]
            
            norm = np.linalg.norm(emb)
            confidence = min(1.0, norm / 10.0)
            normalized_emb = emb / (norm + 1e-8)
            
            return normalized_emb, confidence
        except Exception as e:
            logger.error(f"Error extracting Re-ID embedding: {e}")
            return None, 0.0

    def batch_extract(self, imgs: list):
        """Extract multiple embeddings in single GPU batch"""
        try:
            if not imgs or all(img is None for img in imgs):
                return [None] * len(imgs), [0.0] * len(imgs)
            
            valid_indices = []
            valid_pils = []
            
            for i, img in enumerate(imgs):
                if img is None or img.size == 0:
                    continue
                h, w = img.shape[:2]
                if h < 64 or w < 32:
                    continue
                if h / w < 0.5 or h / w > 3.0:
                    continue
                
                valid_indices.append(i)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                valid_pils.append(self.transform(pil))
            
            results = [None] * len(imgs)
            confidences = [0.0] * len(imgs)
            
            if valid_pils:
                logger.debug(f"Batch extracting {len(valid_pils)} embeddings from {len(imgs)} crops")
                batch = torch.stack(valid_pils).to(self.device)
                with torch.no_grad():
                    embs = self.model(batch).cpu().numpy()
                
                for idx, orig_idx in enumerate(valid_indices):
                    emb = embs[idx]
                    norm = np.linalg.norm(emb)
                    confidence = min(1.0, norm / 10.0)
                    results[orig_idx] = emb / (norm + 1e-8)
                    confidences[orig_idx] = confidence
            
            return results, confidences
        except Exception as e:
            logger.error(f"Error in batch Re-ID extraction: {e}")
            return [None] * len(imgs), [0.0] * len(imgs)


class DetectionThread(QThread):
    """Improved detection thread with proper synchronization and debugging"""
    results_ready = pyqtSignal(object, object, object)

    def __init__(self, model, tracker, reid):
        super().__init__()
        self.model = model
        self.tracker = tracker
        self.reid = reid
        
        self.frame_buffer = FrameBuffer(max_size=3)
        self.track_state = TrackState()
        self.running = True
        self.frame_counter = 0
        
        # Detection configuration
        self.detection_interval = 2
        self.detection_width = 640
        self.min_conf = 0.5
        self.min_detection_size = (32, 64)
        
        logger.info("DetectionThread initialized")

    def run(self):
        try:
            logger.info("DetectionThread started")
            while self.running:
                frame_id, frame = self.frame_buffer.get(timeout_ms=50)
                
                if frame is None:
                    self.msleep(10)
                    continue
                
                self.frame_counter += 1
                h, w = frame.shape[:2]
                logger.debug(f"Processing frame {self.frame_counter}, ID: {frame_id}, shape: {h}x{w}")
                
                # Perform detection on every frame for better tracking
                try:
                    self._detect_and_track(frame, frame_id, h, w)
                except Exception as e:
                    logger.error(f"Error in detection cycle: {e}")
                    # Emit empty results to maintain consistency
                    self.results_ready.emit([], frame, {})
        
        except Exception as e:
            logger.error(f"DetectionThread fatal error: {e}")
            self.running = False

    def _detect_and_track(self, frame, frame_id, h, w):
        """Perform detection, tracking, and batch Re-ID extraction"""
        try:
            # Downscale for detection
            det_w = self.detection_width
            det_h = int(h * det_w / w)
            logger.debug(f"Resizing frame from {w}x{h} to {det_w}x{det_h}")
            
            frame_resized = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Frame resized successfully to {frame_resized.shape}")
            
            # Run detection with proper error handling
            logger.debug("Running YOLO detection...")
            results = self.model(frame_resized, verbose=False)
            logger.debug(f"YOLO returned {len(results)} result(s)")
            
            if not results or len(results) == 0:
                logger.warning("No detection results returned")
                self.results_ready.emit([], frame, {})
                return
            
            res = results[0]
            logger.debug(f"Result type: {type(res)}, has boxes: {hasattr(res, 'boxes')}")
            
            if not hasattr(res, 'boxes') or res.boxes is None:
                logger.warning("Detection result has no boxes attribute")
                self.results_ready.emit([], frame, {})
                return
            
            # Extract detections with proper coordinate handling
            sx, sy = w / det_w, h / det_h
            logger.debug(f"Scale factors: sx={sx}, sy={sy}")
            
            dets, crops = [], []
            detection_count = 0
            
            for box_idx, (box, cls, conf) in enumerate(zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf)):
                try:
                    # Handle tensor vs numpy
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    if isinstance(cls, torch.Tensor):
                        cls = cls.cpu().numpy()
                    if isinstance(conf, torch.Tensor):
                        conf = conf.cpu().item()
                    
                    cls_id = int(cls)
                    class_name = self.model.names.get(cls_id, 'unknown')
                    logger.debug(f"Box {box_idx}: class={class_name}, conf={conf:.3f}")
                    
                    if class_name == 'person' and conf > self.min_conf:
                        x1, y1, x2, y2 = box
                        
                        # Scale back with proper rounding
                        x1 = max(0, int(np.round(x1 * sx)))
                        y1 = max(0, int(np.round(y1 * sy)))
                        x2 = min(w, int(np.round(x2 * sx)))
                        y2 = min(h, int(np.round(y2 * sy)))
                        
                        bw, bh = x2 - x1, y2 - y1
                        
                        logger.debug(f"Scaled box: ({x1},{y1})-({x2},{y2}), size: {bw}x{bh}")
                        
                        if bw >= self.min_detection_size[0] and bh >= self.min_detection_size[1]:
                            dets.append(([x1, y1, bw, bh], conf, 'person'))
                            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
                            crops.append(crop)
                            detection_count += 1
                            logger.info(f"Valid detection {detection_count}: person at ({x1},{y1})-({x2},{y2})")
                        else:
                            logger.debug(f"Detection rejected: too small {bw}x{bh}")
                    else:
                        logger.debug(f"Detection rejected: not person or low confidence")
                
                except Exception as e:
                    logger.error(f"Error processing box {box_idx}: {e}")
                    continue
            
            logger.info(f"Found {detection_count} valid person detections")
            
            # Update tracker
            try:
                logger.debug(f"Updating tracker with {len(dets)} detections")
                tracks = self.tracker.update_tracks(dets, frame=frame)
                logger.info(f"Tracker returned {len(tracks)} tracks")
            except Exception as e:
                logger.error(f"Tracker update failed: {e}")
                tracks = []
            
            # Batch extract embeddings
            if tracks and crops:
                self._batch_extract_embeddings(tracks, dets, crops, frame_id)
            
            # Get embeddings dictionary
            embeddings_dict = self.track_state.get_all_embeddings(frame_id)
            logger.debug(f"Emitting results: {len(tracks)} tracks, {len(embeddings_dict)} embeddings")
            
            self.results_ready.emit(tracks, frame, embeddings_dict)
        
        except Exception as e:
            logger.error(f"Fatal error in _detect_and_track: {e}")
            self.results_ready.emit([], frame, {})

    def _batch_extract_embeddings(self, tracks, dets, crops, frame_id):
        """Extract embeddings using batch inference"""
        try:
            embedding_crops = []
            track_indices = []
            
            logger.debug(f"Extracting embeddings for {len(tracks)} tracks from {len(crops)} crops")
            
            for t_idx, t in enumerate(tracks):
                if not t.is_confirmed():
                    logger.debug(f"Track {t_idx} not confirmed, skipping")
                    continue
                
                tid = t.track_id
                ltrb = list(map(int, t.to_ltrb()))
                logger.debug(f"Track {tid}: ltrb={ltrb}")
                
                best_iou, best_crop_idx = 0, -1
                for det_idx, ((bx, by, bw, bh), _, _) in enumerate(dets):
                    iou = self._compute_iou(ltrb, [bx, by, bx + bw, by + bh])
                    if iou > best_iou and iou > 0.3:
                        best_iou, best_crop_idx = iou, det_idx
                        logger.debug(f"  Detection {det_idx}: IoU={iou:.3f}")
                
                if best_crop_idx >= 0 and best_crop_idx < len(crops):
                    embedding_crops.append(crops[best_crop_idx])
                    track_indices.append((tid, best_iou))
                    logger.debug(f"Track {tid}: matched to detection {best_crop_idx} (IoU={best_iou:.3f})")
            
            # Batch extract
            if embedding_crops:
                logger.debug(f"Batch extracting {len(embedding_crops)} embeddings")
                embeddings, confidences = self.reid.batch_extract(embedding_crops)
                
                for (tid, iou), emb, conf in zip(track_indices, embeddings, confidences):
                    if emb is not None:
                        final_conf = min(conf, max(0.5, iou))
                        self.track_state.update_embedding(tid, emb, frame_id, final_conf)
                        logger.info(f"Extracted embedding for track {tid} (conf: {final_conf:.3f})")
                    else:
                        logger.warning(f"Failed to extract embedding for track {tid}")
        
        except Exception as e:
            logger.error(f"Error in batch embedding extraction: {e}")

    @staticmethod
    def _compute_iou(ltrb1, ltrb2):
        """Compute Intersection over Union between two boxes"""
        try:
            x1_min, y1_min, x1_max, y1_max = ltrb1
            x2_min, y2_min, x2_max, y2_max = ltrb2
            
            # Validate coordinates
            if x1_min >= x1_max or y1_min >= y1_max:
                logger.debug(f"Invalid box1: {ltrb1}")
                return 0.0
            if x2_min >= x2_max or y2_min >= y2_max:
                logger.debug(f"Invalid box2: {ltrb2}")
                return 0.0
            
            xA = max(x1_min, x2_min)
            yA = max(y1_min, y2_min)
            xB = min(x1_max, x2_max)
            yB = min(y1_max, y2_max)
            
            inter = max(0, xB - xA) * max(0, yB - yA)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - inter + 1e-8
            
            iou = inter / union
            return min(1.0, max(0.0, iou))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error computing IoU: {e}")
            return 0.0

    def update_frame(self, frame):
        """Add frame to buffer"""
        self.frame_buffer.put(frame, self.frame_counter)

    def stop(self):
        logger.info("Stopping DetectionThread")
        self.running = False
        self.wait()


class AdaptiveThresholdManager:
    """Manages adaptive similarity thresholds based on context"""
    def __init__(self, low_threshold=0.60, high_threshold=0.75):
        self.base_low = low_threshold
        self.base_high = high_threshold
        self.similarity_history = deque(maxlen=100)
        self.lock = Lock()
    
    def update_history(self, similarity):
        with self.lock:
            self.similarity_history.append(similarity)
    
    def get_thresholds(self, crowd_density=None):
        """Adaptive thresholds based on statistics"""
        with self.lock:
            if len(self.similarity_history) < 10:
                return self.base_low, self.base_high
            
            similarities = np.array(list(self.similarity_history))
            mean = similarities.mean()
            std = similarities.std()
            
            low_th = max(0.5, mean - 1.5 * std)
            high_th = min(0.95, mean + 0.5 * std)
            
            if crowd_density and crowd_density > 0.8:
                low_th = min(low_th + 0.05, 0.70)
                high_th = min(high_th + 0.05, 0.85)
            
            return low_th, high_th
    
    def get_match_label(self, similarity, low_th=None, high_th=None):
        """Get match label with confidence"""
        if low_th is None or high_th is None:
            low_th, high_th = self.get_thresholds()
        
        if similarity > high_th:
            return "MATCH", (0, 0, 255)
        elif similarity > low_th:
            return "POTENTIAL", (0, 165, 255)
        else:
            return "NOMATCH", (0, 255, 0)


class VideoWidget(QWidget):
    """Improved video widget with comprehensive error handling"""
    person_selected = pyqtSignal(int, object)

    def __init__(self, path, cam_id):
        super().__init__()
        self.cam_id = cam_id
        
        try:
            logger.info(f"Initializing camera {cam_id} with video: {path}")
            
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.dw, self.dh = 700, int(self.h * 700 / self.w)
            
            logger.info(f"Camera {cam_id}: {self.w}x{self.h} @ {self.fps}fps, display: {self.dw}x{self.dh}")
        except Exception as e:
            logger.error(f"Error initializing camera {cam_id}: {e}")
            raise
        
        # Initialize models
        try:
            logger.info(f"Loading models for camera {cam_id}")
            
            # Check YOLO model file
            model_path = 'yolo11s.pt'
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} not found, will be downloaded")
            
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded, device: {self.model.device}")
            
            self.tracker = DeepSort(max_age=30, n_init=3)
            logger.info("DeepSORT tracker initialized")
            
            self.reid = ReIDModel(device='cuda' if torch.cuda.is_available() else 'cpu')
            logger.info("ReID model initialized")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
        
        # State
        self.tracks = []
        self.boxes = []
        self.embeddings_dict = {}
        self.global_emb = None
        self.selected_id = None
        self.threshold_manager = AdaptiveThresholdManager()
        
        # UI
        self.label = QLabel()
        self.label.setFixedSize(self.dw, self.dh)
        self.label.setStyleSheet("background:black;")
        self.label.mousePressEvent = self.on_click
        
        # Detection thread
        self.thread = DetectionThread(self.model, self.tracker, self.reid)
        self.thread.results_ready.connect(self.update_tracks)
        self.thread.start()
        logger.info(f"Detection thread started for camera {cam_id}")
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(int(1000 / self.fps))
        
        self.frame_count = 0
        logger.info(f"Camera {cam_id} initialization complete")

    def on_click(self, e):
        """Handle person selection"""
        try:
            x, y = e.pos().x(), e.pos().y()
            gx = int(x * self.w / self.dw)
            gy = int(y * self.h / self.dh)
            
            logger.debug(f"Click at ({x},{y}) -> ({gx},{gy})")
            
            for tid, (x1, y1, x2, y2) in self.boxes:
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    emb = self.embeddings_dict.get(tid)
                    is_selected = self.selected_id == tid
                    self.selected_id = None if is_selected else tid
                    self.person_selected.emit(
                        self.cam_id, None if self.selected_id is None else emb
                    )
                    logger.info(f"Camera {self.cam_id}: Person {tid} {'deselected' if is_selected else 'selected'}")
                    return
            
            if self.selected_id is not None:
                self.selected_id = None
                self.person_selected.emit(self.cam_id, None)
        except Exception as e:
            logger.error(f"Error in on_click: {e}")

    def set_global(self, emb):
        """Set global reference embedding"""
        self.global_emb = emb
        if emb is None:
            logger.info(f"Camera {self.cam_id}: Global embedding cleared")
        else:
            logger.info(f"Camera {self.cam_id}: Global embedding set")

    def update_tracks(self, tracks, frame, embeddings_dict):
        """Update track information"""
        try:
            self.tracks = tracks
            self.embeddings_dict = embeddings_dict
            self.boxes = []
            
            for t in self.tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                ltrb = t.to_ltrb()
                if ltrb is None:
                    continue
                x1, y1, x2, y2 = map(int, ltrb)
                self.boxes.append((tid, (x1, y1, x2, y2)))
        except Exception as e:
            logger.error(f"Error updating tracks: {e}")

    def next_frame(self):
        """Capture and display next frame"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Camera {self.cam_id}: End of video reached")
                self.timer.stop()
                self.thread.stop()
                return
            
            self.frame_count += 1
            logger.debug(f"Camera {self.cam_id}: Frame {self.frame_count}")
            
            self.thread.update_frame(frame)
            self.draw(frame)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pm = QPixmap.fromImage(img).scaled(self.dw, self.dh, Qt.IgnoreAspectRatio)
            self.label.setPixmap(pm)
        except Exception as e:
            logger.error(f"Error in next_frame: {e}")

    def draw(self, frame):
        """Draw bounding boxes and labels"""
        try:
            cv2.rectangle(frame, (0, 0), (500, 120), (30, 30, 30), -1)
            
            y_offset = 20
            cv2.putText(frame, f"Camera {self.cam_id} | Tracks: {len(self.boxes)} | Frame: {self.frame_count}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            low_th, high_th = self.threshold_manager.get_thresholds()
            cv2.putText(frame, f"Thresholds: {low_th:.2f}/{high_th:.2f}", 
                       (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.putText(frame, f"Embeddings: {len(self.embeddings_dict)}", 
                       (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for tid, (x1, y1, x2, y2) in self.boxes:
                color = (0, 255, 0)
                thickness = 2
                label = f"P{tid}"
                
                if tid == self.selected_id:
                    color = (0, 0, 255)
                    thickness = 3
                    label = f"SEL{tid}"
                elif self.global_emb is not None and tid in self.embeddings_dict:
                    sim = np.dot(self.embeddings_dict[tid], self.global_emb)
                    self.threshold_manager.update_history(sim)
                    
                    match_label, match_color = self.threshold_manager.get_match_label(
                        sim, low_th, high_th
                    )
                    
                    if match_label == "MATCH":
                        color, thickness = match_color, 3
                        label = f"M{tid}({sim:.2f})"
                    elif match_label == "POTENTIAL":
                        color, thickness = match_color, 2
                        label = f"?{tid}({sim:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception as e:
            logger.error(f"Error in draw: {e}")


class MainWindow(QWidget):
    """Main application window"""
    def __init__(self, paths):
        super().__init__()
        self.setWindowTitle("CCTV Multi-View Player (Fixed & Debugged)")
        self.grid = QGridLayout(self)
        self.widgets = []
        
        try:
            logger.info(f"Initializing MainWindow with {len(paths)} cameras")
            for i, p in enumerate(paths):
                logger.info(f"Loading camera {i}: {p}")
                w = VideoWidget(p, i)
                w.person_selected.connect(self.on_select)
                self.widgets.append(w)
                self.grid.addWidget(w.label, 0, i)
            
            self.resize(len(paths) * 720, 520)
            logger.info(f"MainWindow ready with {len(self.widgets)} cameras")
        except Exception as e:
            logger.error(f"Error initializing MainWindow: {e}")
            raise

    def on_select(self, cam_id, emb):
        """Broadcast selected person to all cameras"""
        try:
            if emb is None:
                logger.info(f"Clearing global embedding")
            else:
                logger.info(f"Broadcasting embedding from camera {cam_id}")
            
            for w in self.widgets:
                w.set_global(emb)
        except Exception as e:
            logger.error(f"Error in on_select: {e}")


if __name__ == '__main__':
    try:
        logger.info("="*80)
        logger.info("Application Starting")
        logger.info("="*80)
        
        app = QApplication(sys.argv)
        mw = MainWindow(["D:/CCTV/video1.mp4", "D:/CCTV/video3.mp4"])
        mw.show()
        
        logger.info("Application shown, entering event loop")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)