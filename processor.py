import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class TrafficProcessor:
    def __init__(self, model_path='yolov8n.pt', confidence=0.35):
        # 1. GPU Setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Loading Model on: {device.upper()}")
        
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        
        # 2. History for Line Crossing Logic
        # We need this to know if a car was *above* the line previously
        self.track_history = defaultdict(lambda: []) 

    def process_frame(self, frame, current_counts, counted_ids):
        # 3. Optimization: Resize to 640px width
        original_h, original_w = frame.shape[:2]
        new_w = 640
        scale = new_w / original_w
        new_h = int(original_h * scale)
        
        small_frame = cv2.resize(frame, (new_w, new_h))
        
        # 4. Define the Gate Line (at 55% of the screen height)
        line_y = int(new_h * 0.55)
        
        # 5. Run Tracking
        results = self.model.track(small_frame, persist=True, conf=self.confidence, verbose=False)
        
        # Draw the Gate Line on the original frame (scale it up)
        # We draw on 'frame' (the high-res one) for the user to see
        display_line_y = int(line_y / scale)
        cv2.line(frame, (0, display_line_y), (original_w, display_line_y), (255, 255, 0), 2)
        cv2.putText(frame, "COUNTING GATE", (10, display_line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        active_positions = [] # For GIS Map
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            class_ids = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                # Calculate Centroid (on small frame)
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                
                # --- GIS DATA PREP ---
                # Normalize position (0.0 - 1.0)
                x_norm = cx / new_w
                y_norm = cy / new_h
                active_positions.append((x_norm, y_norm))
                
                # --- LINE CROSSING LOGIC ---
                # Get previous position history
                prev_y = cy 
                history = self.track_history[track_id]
                if len(history) > 0:
                    prev_y = history[-1]
                
                # Update history (keep last 10 frames)
                history.append(cy)
                if len(history) > 10: history.pop(0)
                
                # Check for Crossing
                # Incoming: Moved from ABOVE (y < line) to BELOW (y >= line)
                if prev_y < line_y and cy >= line_y:
                    if track_id not in counted_ids:
                        self._update_counts(cls_id, current_counts, "Incoming")
                        counted_ids.add(track_id)
                        # Visual Flash Green
                        cv2.line(frame, (0, display_line_y), (original_w, display_line_y), (0, 255, 0), 4)

                # Outgoing: Moved from BELOW (y > line) to ABOVE (y <= line)
                elif prev_y > line_y and cy <= line_y:
                    if track_id not in counted_ids:
                        self._update_counts(cls_id, current_counts, "Outgoing")
                        counted_ids.add(track_id)
                        # Visual Flash Purple
                        cv2.line(frame, (0, display_line_y), (original_w, display_line_y), (255, 0, 255), 4)

                # --- VISUALIZATION (Scale back to original) ---
                x1, y1, x2, y2 = (box / scale).astype(int)
                raw_label = self.model.names[cls_id]
                
                color = (0, 255, 0) if track_id in counted_ids else (0, 165, 255) # Green if counted, Orange if tracking
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{raw_label} #{track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, current_counts, counted_ids, active_positions

    def _update_counts(self, cls_id, counts, direction):
        raw_label = self.model.names[cls_id]
        if raw_label in ['car', 'motorcycle', 'bus', 'truck']:
            cat = 'Bike' if raw_label == 'motorcycle' else raw_label.capitalize()
            key = f"{direction}_{cat}"
            if key in counts:
                counts[key] += 1