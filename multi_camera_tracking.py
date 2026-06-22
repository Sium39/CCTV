"""
MULTI-CAMERA TRACKING - AUTO-STAR CROSS-CAMERA MATCHING

December 3, 2025 - Global Selection Broadcasting with ID Transfer

✓ YOLO + DeepSORT Tracking
✓ Silhouette + Re-ID Matching
✓ CROSS-CAMERA TRACKING CONTINUITY
✓ Dynamic Confidence Adjustment
✓ AUTO-STAR ⭐ in Other Cameras with SELECT ID TRANSFER

Key Features:
- Track people ACROSS camera boundaries
- Select person in Camera 0 → Auto-star in Camera 1
- Transfer selected_id to other cameras
- Lower detection threshold when person expected
- Re-ID embedding cache for continuity
- Spatial + temporal prediction for transitions
"""

import sys
import cv2
import numpy as np
import torch
import torchreid
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout, QHBoxLayout, QVBoxLayout,
    QTextEdit, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
from collections import deque
from threading import Lock
import logging
import os
import json
from datetime import datetime


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class JsonFormatter(logging.Formatter):
    """Format logs as JSON"""
    def format(self, record):
        log_obj = {
            'timestamp': datetime.now().isoformat(),
            'thread': record.threadName,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_obj)


def setup_logger(name, level=logging.DEBUG, use_json=False):
    """Initialize logger with optional JSON formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger('MultiCameraTracking', level=logging.DEBUG, use_json=False)


# ============================================================================
# CROSS-CAMERA TRACK BRIDGE
# ============================================================================

class CrossCameraTrackBridge:
    """Bridges tracks across cameras - maintains continuity"""
    
    def __init__(self, max_transition_time=15.0):
        self.global_tracks = {}
        self.global_id_counter = 1000
        self.max_transition_time = max_transition_time
        self.lock = Lock()
        logger.info(f"CrossCameraTrackBridge initialized (max_transition_time={max_transition_time}s)")
    
    def register_person(self, cam_id, local_tid, emb, silhouette, timestamp):
        """Register person as global track"""
        with self.lock:
            global_id = self.global_id_counter
            self.global_id_counter += 1
            self.global_tracks[global_id] = {
                'cam_id': cam_id,
                'local_tid': local_tid,
                'embedding': emb,
                'silhouette': silhouette,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'cameras_visited': [cam_id],
                'transition_times': []
            }
            logger.debug(f"Registered global track {global_id} from camera {cam_id}")
            return global_id
    
    def find_person_in_camera(self, cam_id, emb, silhouette, timestamp):
        """Find if person already exists in global tracks"""
        with self.lock:
            best_match = None
            best_score = -1
            
            for global_id, track in self.global_tracks.items():
                if track['cam_id'] == cam_id:
                    continue
                
                time_since_last = timestamp - track['last_seen']
                if time_since_last > self.max_transition_time:
                    continue
                
                if emb is not None and track['embedding'] is not None:
                    re_id_sim = float(np.dot(emb, track['embedding']))
                else:
                    re_id_sim = 0.0
                
                sil_sim = 0.0
                if silhouette is not None and track['silhouette'] is not None:
                    sil_sim = SilhouetteExtractor.silhouette_similarity(silhouette, track['silhouette'])
                
                combined_score = 0.65 * re_id_sim + 0.35 * sil_sim
                
                if re_id_sim > 0.70 and combined_score > best_score:
                    best_score = combined_score
                    best_match = global_id
            
            return best_match, best_score
    
    def update_person(self, global_id, cam_id, local_tid, emb, silhouette, timestamp):
        """Update person track"""
        with self.lock:
            if global_id in self.global_tracks:
                track = self.global_tracks[global_id]
                track['cam_id'] = cam_id
                track['local_tid'] = local_tid
                
                if emb is not None and track['embedding'] is not None:
                    track['embedding'] = 0.7 * track['embedding'] + 0.3 * emb
                elif emb is not None:
                    track['embedding'] = emb
                
                if silhouette is not None:
                    track['silhouette'] = silhouette
                
                prev_cam = track.get('prev_cam_id', cam_id)
                if prev_cam != cam_id:
                    trans_time = timestamp - track['last_seen']
                    track['transition_times'].append(trans_time)
                    if cam_id not in track['cameras_visited']:
                        track['cameras_visited'].append(cam_id)
                    logger.info(f"Global track {global_id}: Camera {prev_cam} → {cam_id} (transit: {trans_time:.2f}s)")
                
                track['prev_cam_id'] = cam_id
                track['last_seen'] = timestamp


# ============================================================================
# SILHOUETTE EXTRACTOR
# ============================================================================

class SilhouetteExtractor:
    """Extract person silhouette for robust matching"""
    
    def __init__(self):
        logger.info("Silhouette extractor initialized")
    
    @staticmethod
    def extract_silhouette(crop):
        """Extract normalized silhouette from person crop"""
        try:
            if crop is None or crop.size == 0:
                return None
            
            h, w = crop.shape[:2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            silhouette = cv2.resize(binary, (64, 128), interpolation=cv2.INTER_AREA)
            sil_vec = silhouette.flatten().astype(np.float32) / 255.0
            norm = np.linalg.norm(sil_vec)
            if norm > 0:
                sil_vec = sil_vec / norm
            return sil_vec
        except Exception as e:
            logger.debug(f"Silhouette extraction error: {e}")
            return None
    
    @staticmethod
    def silhouette_similarity(sil1, sil2):
        """Compute silhouette similarity (cosine)"""
        try:
            if sil1 is None or sil2 is None:
                return 0.0
            sim = np.dot(sil1, sil2)
            return float(np.clip(sim, 0.0, 1.0))
        except:
            return 0.0


# ============================================================================
# SELECTIVE MATCHER
# ============================================================================

class SelectiveMatcher:
    """Smart matching with strict constraints"""
    
    def __init__(self):
        self.re_id_threshold_strict = 0.30
        self.re_id_threshold_cross = 0.30
        self.silhouette_threshold = 0.30
        self.spatial_distance_threshold = 100
        self.confidence_threshold = 0.3
        logger.info("SelectiveMatcher initialized")


# ============================================================================
# DISTANCE ESTIMATOR
# ============================================================================

class DistanceEstimator:
    """Multi-method distance estimation"""
    
    def __init__(self, fx=800, fy=800, cx=960, cy=540,
                 sensor_height=1.15, person_height=1.7):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.sensor_height = sensor_height
        self.person_height = person_height
        self.distance_history = deque(maxlen=10)
        logger.info(f"DistanceEstimator: fx={fx}, fy={fy}, sensor_height={sensor_height}m")
    
    def bbox_to_distance(self, h_px, person_height=None):
        """Inverse proportion distance estimation"""
        if person_height:
            self.person_height = person_height
        if h_px <= 0:
            return None
        distance = (self.person_height * 1000) / h_px
        self.distance_history.append(distance)
        return distance


# ============================================================================
# ETA PREDICTOR
# ============================================================================

class ETAPredictor:
    """Predicts time to next camera"""
    
    def __init__(self, default_walking_speed=1.4):
        self.travel_times = {}
        self.default_walking_speed = default_walking_speed
        self.transition_logs = []
        logger.info(f"ETAPredictor: walking_speed={default_walking_speed}m/s")
    
    def update_travel_time(self, cam_from, cam_to, time_from, time_to):
        """Record person transition"""
        key = (cam_from, cam_to)
        travel_time = time_to - time_from
        self.travel_times.setdefault(key, []).append(travel_time)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transition': f"{cam_from}->{cam_to}",
            'travel_time_sec': travel_time,
            'avg_time_sec': np.mean(self.travel_times[key]),
            'observations': len(self.travel_times[key])
        }
        self.transition_logs.append(log_entry)
        logger.debug(f"Travel time: {key} = {travel_time:.1f}s")
    
    def predict_eta(self, cam_from, cam_to, current_time, distance_m, speed_mps=None):
        """Predict arrival time"""
        if speed_mps is None:
            speed_mps = self.default_walking_speed
        
        key = (cam_from, cam_to)
        time_to_exit = distance_m / max(speed_mps, 0.5)
        additional_time = 0
        confidence = 0.3
        
        if key in self.travel_times and len(self.travel_times[key]) >= 2:
            times = np.array(self.travel_times[key])
            additional_time = np.median(times)
            confidence = 0.8
        
        eta = current_time + time_to_exit + additional_time
        return eta, confidence


# ============================================================================
# FRAME BUFFER
# ============================================================================

class FrameBuffer:
    """Thread-safe frame buffer"""
    
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
        self.lock = Lock()
        logger.debug(f"FrameBuffer: max_size={max_size}")
    
    def put(self, frame, frame_id):
        """Add frame to buffer"""
        with self.lock:
            if frame is not None:
                self.buffer.append((frame_id, frame.copy()))
    
    def get(self, timeout_ms=100):
        """Retrieve frame from buffer"""
        with self.lock:
            if self.buffer:
                frame_id, frame = self.buffer.popleft()
                return frame_id, frame
        return None, None
    
    def is_empty(self):
        with self.lock:
            return len(self.buffer) == 0
    
    def clear(self):
        with self.lock:
            self.buffer.clear()


# ============================================================================
# TRACK STATE
# ============================================================================

class TrackState:
    """Track lifecycle management with silhouettes"""
    
    def __init__(self, max_track_age=120):
        self.active_tracks = {}
        self.lock = Lock()
        self.max_track_age = max_track_age
        logger.info(f"TrackState: max_track_age={max_track_age}")
    
    def update_embedding(self, track_id, embedding, silhouette, timestamp, confidence=1.0):
        """Store Re-ID embedding + silhouette"""
        with self.lock:
            self.active_tracks[track_id] = {
                'embedding': embedding,
                'silhouette': silhouette,
                'timestamp': timestamp,
                'confidence': confidence,
                'created_at': timestamp if track_id not in self.active_tracks else self.active_tracks[track_id]['created_at']
            }
    
    def get_embedding(self, track_id):
        """Retrieve embedding"""
        with self.lock:
            if track_id in self.active_tracks:
                return self.active_tracks[track_id]['embedding']
        return None
    
    def get_silhouette(self, track_id):
        """Retrieve silhouette"""
        with self.lock:
            if track_id in self.active_tracks:
                return self.active_tracks[track_id]['silhouette']
        return None
    
    def get_all_embeddings(self, current_timestamp):
        """Get all active embeddings"""
        with self.lock:
            stale_ids = [tid for tid, data in self.active_tracks.items()
                        if current_timestamp - data['timestamp'] > self.max_track_age]
            for tid in stale_ids:
                del self.active_tracks[tid]
            result = {tid: data['embedding'] for tid, data in self.active_tracks.items()}
            return result
    
    def get_all_silhouettes(self, current_timestamp):
        """Get all active silhouettes"""
        with self.lock:
            result = {tid: data['silhouette'] for tid, data in self.active_tracks.items()}
            return result


# ============================================================================
# RE-ID MODEL
# ============================================================================

class ReIDModel:
    """OSNet Re-ID model"""
    
    def __init__(self, device='cuda'):
        self.device = device
        try:
            logger.info(f"Loading OSNet x1_0 on {device}...")
            self.model = torchreid.models.build_model(
                name='osnet_x1_0', num_classes=1000, pretrained=True
            ).to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            _, test_transforms = torchreid.data.transforms.build_transforms(
                height=256, width=128, transforms=[],
                norm_mean=[0.485, 0.456, 0.406],
                norm_std=[0.229, 0.224, 0.225]
            )
            self.transform = test_transforms
            logger.info("✓ ReID model loaded")
        except Exception as e:
            logger.error(f"ReID error: {e}")
            raise
    
    def extract(self, img: np.ndarray):
        """Extract single embedding"""
        try:
            if img is None or img.size == 0:
                return None, 0.0
            
            h, w = img.shape[:2]
            if h < 64 or w < 32:
                return None, 0.0
            
            aspect_ratio = h / w
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
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
            logger.error(f"Extract error: {e}")
            return None, 0.0
    
    def batch_extract(self, imgs: list):
        """Batch extract embeddings"""
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
                aspect_ratio = h / w
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    continue
                
                valid_indices.append(i)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                valid_pils.append(self.transform(pil))
            
            results = [None] * len(imgs)
            confidences = [0.0] * len(imgs)
            
            if valid_pils:
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
            logger.error(f"Batch extract error: {e}")
            return [None] * len(imgs), [0.0] * len(imgs)


# ============================================================================
# DETECTION THREAD - WITH DYNAMIC CONFIDENCE
# ============================================================================

class DetectionThread(QThread):
    """Asynchronous detection + tracking + Re-ID with adaptive confidence"""
    
    results_ready = pyqtSignal(object, object, object, object)
    
    def __init__(self, model, tracker, reid, cam_id, bridge):
        super().__init__()
        self.model = model
        self.tracker = tracker
        self.reid = reid
        self.cam_id = cam_id
        self.bridge = bridge
        self.frame_buffer = FrameBuffer(max_size=3)
        self.track_state = TrackState(max_track_age=120)
        self.running = True
        self.frame_counter = 0
        self.detection_width = 640
        self.min_conf = 0.5
        self.min_conf_adaptive = 0.3
        self.min_detection_size = (32, 64)
        self.silhouette_extractor = SilhouetteExtractor()
        logger.info(f"DetectionThread (cam {cam_id}) initialized - ADAPTIVE CONFIDENCE")
    
    def run(self):
        """Main detection loop"""
        try:
            logger.info(f"DetectionThread (cam {self.cam_id}) started")
            while self.running:
                frame_id, frame = self.frame_buffer.get(timeout_ms=50)
                if frame is None:
                    self.msleep(10)
                    continue
                
                self.frame_counter += 1
                h, w = frame.shape[:2]
                
                try:
                    self._detect_and_track(frame, frame_id, h, w)
                except Exception as e:
                    logger.error(f"Detection error: {e}")
                    self.results_ready.emit([], frame, {}, {})
        except Exception as e:
            logger.error(f"DetectionThread fatal: {e}")
            self.running = False
    
    def _detect_and_track(self, frame, frame_id, h, w):
        """Complete pipeline with adaptive confidence"""
        try:
            det_w = self.detection_width
            det_h = int(h * det_w / w)
            frame_resized = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
            
            results = self.model(frame_resized, verbose=False)
            
            if not results or len(results) == 0:
                self.results_ready.emit([], frame, {}, {})
                return
            
            res = results[0]
            
            if not hasattr(res, 'boxes') or res.boxes is None:
                self.results_ready.emit([], frame, {}, {})
                return
            
            sx, sy = w / det_w, h / det_h
            dets, crops = [], []
            
            min_conf = self.min_conf_adaptive if self._has_expected_transitions() else self.min_conf
            
            for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                try:
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    cls_id = int(cls) if isinstance(cls, torch.Tensor) else int(cls)
                    conf_val = float(conf) if isinstance(conf, torch.Tensor) else float(conf)
                    class_name = self.model.names.get(cls_id, 'unknown')
                    
                    if class_name != 'person' or conf_val <= min_conf:
                        continue
                    
                    x1, y1, x2, y2 = box
                    x1 = max(0, int(np.round(x1 * sx)))
                    y1 = max(0, int(np.round(y1 * sy)))
                    x2 = min(w, int(np.round(x2 * sx)))
                    y2 = min(h, int(np.round(y2 * sy)))
                    
                    bw, bh = x2 - x1, y2 - y1
                    
                    if bw >= self.min_detection_size[0] and bh >= self.min_detection_size[1]:
                        dets.append(([x1, y1, bw, bh], conf_val, 'person'))
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
                        crops.append(crop)
                except Exception as e:
                    logger.debug(f"Box error: {e}")
                    continue
            
            logger.debug(f"Camera {self.cam_id} Detections: {len(dets)} (min_conf={min_conf:.2f})")
            
            try:
                tracks = self.tracker.update_tracks(dets, frame=frame)
            except Exception as e:
                logger.error(f"Tracker error: {e}")
                tracks = []
            
            if tracks and crops:
                self._batch_extract_embeddings(tracks, dets, crops, frame_id)
            
            embeddings_dict = self.track_state.get_all_embeddings(frame_id)
            silhouettes_dict = self.track_state.get_all_silhouettes(frame_id)
            
            self.results_ready.emit(tracks, frame, embeddings_dict, silhouettes_dict)
        except Exception as e:
            logger.error(f"Fatal detection error: {e}")
            self.results_ready.emit([], frame, {}, {})
    
    def _has_expected_transitions(self):
        """Check if we expect people from other cameras"""
        try:
            return True
        except:
            return False
    
    def _batch_extract_embeddings(self, tracks, dets, crops, frame_id):
        """Extract embeddings + silhouettes"""
        try:
            embedding_crops = []
            track_indices = []
            
            for t in tracks:
                if not t.is_confirmed():
                    continue
                
                tid = t.track_id
                ltrb = list(map(int, t.to_ltrb()))
                
                best_iou, best_crop_idx = 0, -1
                for det_idx, ((bx, by, bw, bh), _, _) in enumerate(dets):
                    iou = self._compute_iou(ltrb, [bx, by, bx + bw, by + bh])
                    if iou > best_iou and iou > 0.3:
                        best_iou, best_crop_idx = iou, det_idx
                
                if best_crop_idx >= 0 and best_crop_idx < len(crops):
                    embedding_crops.append(crops[best_crop_idx])
                    track_indices.append((tid, best_iou))
            
            if embedding_crops:
                embeddings, confidences = self.reid.batch_extract(embedding_crops)
                
                for (tid, iou), crop, emb, conf in zip(track_indices, embedding_crops, embeddings, confidences):
                    if emb is not None:
                        final_conf = min(conf, max(0.5, iou))
                        silhouette = self.silhouette_extractor.extract_silhouette(crop)
                        self.track_state.update_embedding(tid, emb, silhouette, self.frame_counter, final_conf)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
    
    @staticmethod
    def _compute_iou(ltrb1, ltrb2):
        """IoU"""
        try:
            x1_min, y1_min, x1_max, y1_max = ltrb1
            x2_min, y2_min, x2_max, y2_max = ltrb2
            
            if x1_min >= x1_max or y1_min >= y1_max or x2_min >= x2_max or y2_min >= y2_max:
                return 0.0
            
            xA = max(x1_min, x2_min)
            yA = max(y1_min, y2_min)
            xB = min(x1_max, x2_max)
            yB = min(y1_max, y2_max)
            
            inter = max(0, xB - xA) * max(0, yB - yA)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union = area1 + area2 - inter + 1e-8
            
            return min(1.0, max(0.0, inter / union))
        except:
            return 0.0
    
    def update_frame(self, frame):
        self.frame_buffer.put(frame, self.frame_counter)
    
    def stop(self):
        logger.info(f"Stopping DetectionThread (cam {self.cam_id})")
        self.running = False
        self.wait()


# ============================================================================
# ENHANCED VIDEO WIDGET - AUTO-STAR MATCHING
# ============================================================================

class VideoWidget(QWidget):
    """Enhanced single camera tracking UI with auto-star cross-camera matching"""
    
    person_selected = pyqtSignal(int, int, object, object)  # cam_id, selected_id, emb, silhouette
    
    def __init__(self, path, cam_id, num_cameras=2, bridge=None):
        super().__init__()
        self.cam_id = cam_id
        self.num_cameras = num_cameras
        self.bridge = bridge or CrossCameraTrackBridge()
        self.matcher = SelectiveMatcher()
        self.distance_estimator = DistanceEstimator()
        self.track_distances = {}
        self.track_speeds = {}
        self.track_positions = {}
        self.track_times = {}
        self.next_camera = (cam_id + 1) % num_cameras
        self.eta_predictor = ETAPredictor()
        
        try:
            logger.info(f"Initializing camera {cam_id}: {path}")
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open: {path}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.dw, self.dh = 700, int(self.h * 700 / self.w)
            logger.info(f"Camera {cam_id}: {self.w}x{self.h} @ {self.fps}fps")
        except Exception as e:
            logger.error(f"Camera error: {e}")
            raise
        
        try:
            logger.info("Loading YOLO + DeepSORT + ReID...")
            self.model = YOLO('best.pt')
            self.tracker = DeepSort(max_age=30, n_init=3)
            self.reid = ReIDModel(device='cuda' if torch.cuda.is_available() else 'cpu')
            logger.info("✓ Models ready")
        except Exception as e:
            logger.error(f"Model error: {e}")
            raise
        
        self.tracks = []
        self.boxes = []
        self.embeddings_dict = {}
        self.silhouettes_dict = {}
        self.global_emb = None
        self.global_silhouette = None
        self.global_selected_id = None
        self.selected_id = None
        self.current_time = 0
        
        # UI Setup
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        
        left_panel = QVBoxLayout()
        left_panel.setSpacing(0)
        
        title_frame = QFrame()
        title_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border-bottom: 2px solid #00ff00;
                padding: 5px;
            }
        """)
        
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(5, 3, 5, 3)
        title_layout.setSpacing(0)
        
        title_label = QLabel(f"🎥 CAMERA {cam_id}")
        title_font = QFont("Courier")
        title_font.setPointSize(11)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #00ff00;")
        title_layout.addWidget(title_label)
        
        info_label = QLabel(f"{self.w}×{self.h} @ {self.fps:.0f}fps | Display: {self.dw}×{self.dh}")
        info_font = QFont("Courier")
        info_font.setPointSize(8)
        info_label.setFont(info_font)
        info_label.setStyleSheet("color: #00aa00;")
        title_layout.addWidget(info_label)
        
        left_panel.addWidget(title_frame, 0)
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(self.dw, self.dh)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #00ff00;")
        self.video_label.mousePressEvent = self.on_click
        left_panel.addWidget(self.video_label, 1)
        
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(5, 5, 5, 5)
        right_panel.setSpacing(3)
        
        stats_header = QLabel("📊 TRACKING STATS")
        stats_header_font = QFont("Courier")
        stats_header_font.setPointSize(10)
        stats_header_font.setBold(True)
        stats_header.setFont(stats_header_font)
        stats_header.setStyleSheet("color: #ffff00; padding: 5px; background-color: #0a0a0a;")
        right_panel.addWidget(stats_header, 0)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMinimumWidth(300)
        self.stats_text.setMaximumWidth(380)
        stats_font = QFont("Courier")
        stats_font.setPointSize(8)
        self.stats_text.setFont(stats_font)
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #00ff00;
                border: 2px solid #00ff00;
                padding: 5px;
            }
            QScrollBar:vertical {
                background-color: #1a1a1a;
                border: 1px solid #00ff00;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #00ff00;
            }
        """)
        right_panel.addWidget(self.stats_text, 1)
        
        controls_label = QLabel("💡 AUTO-STAR MATCHING")
        controls_font = QFont("Courier")
        controls_font.setPointSize(8)
        controls_font.setBold(True)
        controls_label.setFont(controls_font)
        controls_label.setStyleSheet("color: #0088ff; padding: 3px;")
        right_panel.addWidget(controls_label, 0)
        
        controls_text = QLabel("• Click: ⭐ person\n• Transfer ID to Cam 1\n• Auto-star in other cams\n• Re-ID 0.85 + Sil 0.75")
        controls_font = QFont("Courier")
        controls_font.setPointSize(7)
        controls_text.setFont(controls_font)
        controls_text.setStyleSheet("color: #00aa00; padding: 3px;")
        right_panel.addWidget(controls_text, 0)
        
        left_container = QWidget()
        left_container.setLayout(left_panel)
        right_container = QWidget()
        right_container.setLayout(right_panel)
        
        main_layout.addWidget(left_container, 3)
        main_layout.addWidget(right_container, 1)
        
        self.setLayout(main_layout)
        self.setStyleSheet("QWidget { background-color: #0a0a0a; color: #00ff00; }")
        
        self.thread = DetectionThread(self.model, self.tracker, self.reid, cam_id, self.bridge)
        self.thread.results_ready.connect(self.update_tracks)
        self.thread.start()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(int(1000 / self.fps))
        
        self.frame_count = 0
        logger.info(f"✓ Camera {cam_id} UI ready - AUTO-STAR MATCHING WITH ID TRANSFER")
    
    def on_click(self, e):
        """Click handler - SELECT PERSON and transfer ID"""
        try:
            x, y = e.pos().x(), e.pos().y()
            gx = int(x * self.w / self.dw)
            gy = int(y * self.h / self.dh)
            
            clicked_tid = None
            for tid, x1, y1, x2, y2 in self.boxes:
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    clicked_tid = tid
                    break
            
            if clicked_tid is not None:
                if self.selected_id == clicked_tid:
                    self.selected_id = None
                    self.person_selected.emit(self.cam_id, None, None, None)
                    logger.info(f"Camera {self.cam_id}: Deselected P{clicked_tid}")
                else:
                    self.selected_id = clicked_tid
                    emb = self.embeddings_dict.get(clicked_tid)
                    sil = self.silhouettes_dict.get(clicked_tid)
                    self.person_selected.emit(self.cam_id, clicked_tid, emb, sil)
                    logger.info(f"Camera {self.cam_id}: ★ SELECTED P{clicked_tid} - Transferring ID to other cameras")
        except Exception as e:
            logger.error(f"Click error: {e}")
    
    def set_global(self, global_selected_id, emb, silhouette):
        """Receive broadcast from another camera - AUTO-STAR MATCHING with ID transfer"""
        self.global_selected_id = global_selected_id
        self.global_emb = emb
        self.global_silhouette = silhouette
    
    def update_tracks(self, tracks, frame, embeddings_dict, silhouettes_dict):
        try:
            self.tracks = tracks
            self.embeddings_dict = embeddings_dict
            self.silhouettes_dict = silhouettes_dict
            self.boxes = []
            
            for t in self.tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                ltrb = t.to_ltrb()
                if ltrb is None:
                    continue
                x1, y1, x2, y2 = map(int, ltrb)
                self.boxes.append((tid, x1, y1, x2, y2))
        except Exception as e:
            logger.error(f"Update tracks error: {e}")
    
    def next_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Camera {self.cam_id}: End of video")
                self.timer.stop()
                self.thread.stop()
                return
            
            self.frame_count += 1
            self.current_time = self.frame_count / self.fps
            self.thread.update_frame(frame)
            self.draw(frame, self.current_time)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pm = QPixmap.fromImage(img).scaled(self.dw, self.dh, Qt.IgnoreAspectRatio)
            self.video_label.setPixmap(pm)
            
            self.update_stats()
        except Exception as e:
            logger.error(f"Frame error: {e}")
    
    def draw(self, frame, current_time):
        """Draw with AUTO-STAR matching and ID transfer"""
        try:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (15, 15, 15), -1)
            y_pos = 25
            
            conf_display = f"{self.thread.min_conf_adaptive:.2f}↓{self.thread.min_conf:.2f}↑" if self.thread.min_conf_adaptive < self.thread.min_conf else f"{self.thread.min_conf:.2f}"
            
            cv2.putText(frame, f"Camera {self.cam_id} | Active: {len(self.boxes)} | Conf: {conf_display}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Re-ID: {self.matcher.re_id_threshold_strict:.2f} | Sil: {self.matcher.silhouette_threshold:.2f}",
                       (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            global_id_text = f"Global Selected: P{self.global_selected_id}" if self.global_selected_id is not None else "Global Selected: None"
            cv2.putText(frame, global_id_text,
                       (10, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            
            for tid, x1, y1, x2, y2 in self.boxes:
                h_px = y2 - y1
                dist_m = self.distance_estimator.bbox_to_distance(h_px)
                
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                if tid not in self.track_positions:
                    self.track_positions[tid] = []
                self.track_positions[tid].append((center_x, center_y, current_time))
                self.track_positions[tid] = self.track_positions[tid][-10:]
                
                speed_mps = 0.0
                if len(self.track_positions[tid]) >= 3:
                    pos = np.array(self.track_positions[tid])
                    dx, dy = pos[-1, 0] - pos[0, 0], pos[-1, 1] - pos[0, 1]
                    dt = pos[-1, 2] - pos[0, 2]
                    if dt > 0:
                        pixel_speed = np.sqrt(dx**2 + dy**2) / dt
                        speed_mps = pixel_speed * 0.01
                
                color = (0, 255, 0)
                thickness = 2
                label = f"P{tid}"
                
                # LOCAL SELECTION: RED BOX WITH STAR
                if tid == self.selected_id:
                    color = (0, 0, 255)
                    thickness = 3
                    label = f"★{tid}"
                
                # AUTO-STAR MATCHING: ID MATCHES GLOBAL SELECTED ID
                elif self.global_selected_id is not None and tid == self.global_selected_id:
                    if self.global_emb is not None and tid in self.embeddings_dict:
                        emb = self.embeddings_dict[tid]
                        sil = self.silhouettes_dict.get(tid)
                        
                        re_id_sim = float(np.dot(emb, self.global_emb))
                        sil_sim = 0.0
                        
                        if sil is not None and self.global_silhouette is not None:
                            sil_sim = SilhouetteExtractor.silhouette_similarity(sil, self.global_silhouette)
                        
                        # BOTH THRESHOLDS MET: AUTO-STAR with matching ID!
                        if re_id_sim > self.matcher.re_id_threshold_strict and sil_sim > self.matcher.silhouette_threshold:
                            color = (0, 0, 255)  # RED BOX
                            thickness = 3
                            label = f"★{tid}(AUTO)"  # AUTO-STAR MARKER
                            logger.info(f"Camera {self.cam_id}: ★ AUTO-STAR P{tid} - Transferred ID match (Re-ID:{re_id_sim:.2f} Sil:{sil_sim:.2f})")
                        
                        # ONLY RE-ID MATCHED
                        elif re_id_sim > 0.75:
                            color = (0, 165, 255)  # ORANGE BOX
                            thickness = 2
                            label = f"?{tid}(R:{re_id_sim:.2f})"
                
                full_label = label
                if dist_m:
                    full_label += f" {dist_m:.1f}m"
                if speed_mps > 0.1:
                    full_label += f" {speed_mps:.1f}m/s"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                font_scale = 0.5
                font_thick = 1
                text_size = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)[0]
                text_x = max(5, x1)
                text_y = max(20, y1 - 10)
                
                cv2.rectangle(frame,
                            (text_x - 3, text_y - text_size[1] - 3),
                            (text_x + text_size[0] + 3, text_y + 3),
                            (0, 0, 0), -1)
                cv2.putText(frame, full_label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), font_thick)
        except Exception as e:
            logger.error(f"Draw error: {e}")
    
    def update_stats(self):
        """Update stats panel"""
        try:
            stats = f"""━━━━━━━━━━━━━━━━━━━━━━━━
🎬 VIDEO
━━━━━━━━━━━━━━━━━━━━━━━━
Frame: {self.frame_count}
Time: {self.current_time:.2f}s
FPS: {self.fps:.1f}
Res: {self.w}×{self.h}

━━━━━━━━━━━━━━━━━━━━━━━━
👥 TRACKING (ID TRANSFER)
━━━━━━━━━━━━━━━━━━━━━━━━
Active: {len(self.boxes)}
Re-ID: {len(self.embeddings_dict)}
Silhouettes: {len(self.silhouettes_dict)}
Local Selected: {'P' + str(self.selected_id) if self.selected_id else 'None'}
Global Selected: {'P' + str(self.global_selected_id) if self.global_selected_id else 'None'}

━━━━━━━━━━━━━━━━━━━━━━━━
🎯 THRESHOLDS (CONFIRMED)
━━━━━━━━━━━━━━━━━━━━━━━━
Re-ID (Strict): 0.850
Silhouette: 0.750
Re-ID (Cross): 0.750

━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ DETECTION ADAPTIVE
━━━━━━━━━━━━━━━━━━━━━━━━
Min Conf (Normal): {self.thread.min_conf:.3f}
Min Conf (Adapt): {self.thread.min_conf_adaptive:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━
📊 PERSONS ({len(self.boxes)})
━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            for tid, x1, y1, x2, y2 in self.boxes:
                dist_m = self.distance_estimator.bbox_to_distance(y2 - y1)
                
                if tid == self.selected_id:
                    status = "⭐(LOCAL)"
                elif tid == self.global_selected_id and self.global_selected_id is not None:
                    status = "★(GLOBAL)"
                elif tid in self.embeddings_dict:
                    status = "🔍"
                else:
                    status = "✓"
                
                dist_str = f"{dist_m:.1f}m" if dist_m else "?"
                speed_str = ""
                
                if tid in self.track_positions and len(self.track_positions[tid]) >= 3:
                    pos = np.array(self.track_positions[tid])
                    dx, dy = pos[-1, 0] - pos[0, 0], pos[-1, 1] - pos[0, 1]
                    dt = pos[-1, 2] - pos[0, 2]
                    if dt > 0:
                        pixel_speed = np.sqrt(dx**2 + dy**2) / dt
                        speed_mps = pixel_speed * 0.01
                        speed_str = f" {speed_mps:.1f}m/s"
                
                stats += f"[{status}] P{tid}: {dist_str:<5}{speed_str}\n"
            
            self.stats_text.setText(stats)
        except Exception as e:
            logger.error(f"Stats error: {e}")


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QWidget):
    def __init__(self, paths):
        super().__init__()
        self.setWindowTitle("🎬 Multi-Camera Auto-Star - SELECT ID TRANSFER & AUTO-STAR")
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #00ff00;
            }
        """)
        
        grid = QGridLayout(self)
        grid.setContentsMargins(5, 5, 5, 5)
        grid.setSpacing(5)
        
        self.bridge = CrossCameraTrackBridge(max_transition_time=15.0)
        self.widgets = []
        
        try:
            logger.info(f"Initializing {len(paths)} cameras with SELECT ID TRANSFER...")
            for i, p in enumerate(paths):
                logger.info(f" Camera {i}: {p}")
                w = VideoWidget(p, i, num_cameras=len(paths), bridge=self.bridge)
                w.person_selected.connect(self.on_select)
                self.widgets.append(w)
                grid.addWidget(w, 0, i)
            
            total_width = len(paths) * 1100 + 20
            self.resize(total_width, 900)
            logger.info("✓ MainWindow ready - SELECT ID TRANSFER MODE")
        except Exception as e:
            logger.error(f"MainWindow error: {e}")
            raise
    
    def on_select(self, cam_id, selected_id, emb, silhouette):
        """Broadcast selection with selected_id to all cameras"""
        try:
            if emb is None:
                logger.info(f"📡 Cleared selection from camera {cam_id}")
                selected_id = None
            else:
                logger.info(f"📡 Broadcasting from camera {cam_id}: P{selected_id} - Transferring to other cameras")
            
            # Broadcast to all cameras with selected_id
            for w in self.widgets:
                w.set_global(selected_id, emb, silhouette)
        except Exception as e:
            logger.error(f"Select error: {e}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        logger.info("="*80)
        logger.info("🚀 MULTI-CAMERA TRACKING - SELECT ID TRANSFER & AUTO-STAR")
        logger.info("Features:")
        logger.info(" ✓ Click person in Camera 0 → ⭐ P4 (local star)")
        logger.info(" ✓ Transfer P4 ID to Camera 1")
        logger.info(" ✓ Camera 1 auto-stars when P4 detected (ID match)")
        logger.info(" ✓ Shows: ★P4(AUTO) in red box when ID transferred & thresholds pass")
        logger.info(" ✓ Re-ID: 0.85 | Silhouette: 0.75")
        logger.info("="*80)
        
        app = QApplication(sys.argv)
        
        # UPDATE YOUR VIDEO PATHS HERE
        video_paths = [
           r"C:\Users\SiumNSL\OneDrive\Desktop\CCTV\vid1.mp4",
           r"C:\Users\SiumNSL\OneDrive\Desktop\CCTV\vid2.mp4",
           r"C:\Users\SiumNSL\OneDrive\Desktop\CCTV\vid3.mp4"
        ]
        
        mw = MainWindow(video_paths)
        mw.show()
        
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
