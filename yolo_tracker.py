"""
YOLOv8 Person Detection & Tracking - File Upload Version
Gradio Video component issue ko bypass kar ke File component use kiya
"""

import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
import os
from datetime import datetime
from PIL import Image
import shutil

class IoUTracker:
    """IoU-based tracker for person detection"""
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.max_frames_lost = 30
        
    def compute_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['frames_lost'] += 1
                if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                    del self.tracks[track_id]
            return []
        
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for det_idx in unmatched_detections:
                iou = self.compute_iou(track['box'], detections[det_idx])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx != -1:
                self.tracks[track_id]['box'] = detections[best_det_idx]
                self.tracks[track_id]['frames_lost'] = 0
                self.tracks[track_id]['frames_tracked'] += 1
                matched_tracks.append((track_id, detections[best_det_idx]))
                unmatched_detections.remove(best_det_idx)
            else:
                self.tracks[track_id]['frames_lost'] += 1
                if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                    del self.tracks[track_id]
        
        for det_idx in unmatched_detections:
            self.tracks[self.next_id] = {
                'box': detections[det_idx],
                'frames_lost': 0,
                'frames_tracked': 1
            }
            matched_tracks.append((self.next_id, detections[det_idx]))
            self.next_id += 1
        
        return matched_tracks

def process_frame(frame, model, tracker, colors):
    """Process single frame"""
    results = model(frame, classes=[0], verbose=False, device='cpu')
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            if conf > 0.4:
                detections.append([int(x1), int(y1), int(x2), int(y2)])
    
    tracked_objects = tracker.update(detections)
    
    for track_id, box in tracked_objects:
        x1, y1, x2, y2 = box
        color = colors[track_id % len(colors)]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label = f"Person {track_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame, len(tracked_objects), len(tracker.tracks)

def process_video_realtime(video_file):
    """Process uploaded video file"""
    if video_file is None:
        yield None, "❌ Please upload a video file!", {}, None
        return
    
    try:
        # Get uploaded file path
        video_path = video_file.name
        
        yield None, f"🔄 Loading video: {os.path.basename(video_path)}", {}, None
        
        # Load model
        try:
            model = YOLO('yolov8n.pt')
            yield None, "✅ Model loaded! Starting processing...", {}, None
        except Exception as e:
            yield None, f"❌ Model Error: {str(e)}\n\nRun: pip install ultralytics", {}, None
            return
        
        tracker = IoUTracker(iou_threshold=0.3)
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
            (255, 165, 0), (0, 128, 255), (128, 255, 0), (255, 0, 128)
        ]
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield None, f"❌ Cannot open video!\n\nFile: {video_path}", {}, None
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        yield None, f"✅ Video: {width}x{height} @ {fps}fps\n📊 Total Frames: {total_frames}\n\n🎬 Processing...", {}, None
        
        # Create output
        output_dir = "tracked_videos"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"tracked_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        unique_people_set = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, active_people, total_unique = process_frame(
                frame.copy(), model, tracker, colors
            )
            
            # Add overlay
            info_text = f"Frame: {frame_count}/{total_frames} | Active: {active_people} | Total: {total_unique}"
            text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(processed_frame, (5, 5), 
                         (text_size[0] + 15, text_size[1] + 20), (0, 0, 0), -1)
            cv2.putText(processed_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            out.write(processed_frame)
            
            unique_people_set.update([tid for tid, _ in tracker.tracks.items()])
            total_detections += active_people
            
            # Convert for display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            result_text = f"""
🔴 PROCESSING... Frame {frame_count}/{total_frames}

📊 Live Stats:
━━━━━━━━━━━━━━━━━━━━━━━━
• Active People: {active_people}
• Total Unique: {total_unique}
• Progress: {(frame_count/total_frames)*100:.1f}%
• Time: {frame_count/fps:.1f}s / {total_frames/fps:.1f}s
            """
            
            stats = {
                "Frame": f"{frame_count}/{total_frames}",
                "Active": active_people,
                "Unique": total_unique,
                "Progress": f"{(frame_count/total_frames)*100:.1f}%"
            }
            
            # Yield every 2nd frame
            if frame_count % 2 == 0:
                yield pil_image, result_text, stats, None
        
        cap.release()
        out.release()
        
        # Final results
        final_text = f"""
✅ COMPLETE!

📊 FINAL STATS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Frames: {frame_count}
✓ Unique People: {len(unique_people_set)}
✓ Total Detections: {total_detections}
✓ Avg/Frame: {total_detections/frame_count:.2f}

🎥 VIDEO INFO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ FPS: {fps}
✓ Resolution: {width}x{height}
✓ Duration: {frame_count/fps:.2f}s

📁 OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ File: {output_path}
✓ Size: {os.path.getsize(output_path)/(1024*1024):.2f} MB

✅ Download button below!
        """
        
        final_stats = {
            "Frames": frame_count,
            "Unique People": len(unique_people_set),
            "Detections": total_detections,
            "Avg/Frame": round(total_detections/frame_count, 2),
            "Output": output_path,
            "Size (MB)": round(os.path.getsize(output_path)/(1024*1024), 2)
        }
        
        yield pil_image, final_text, final_stats, output_path
        
    except Exception as e:
        import traceback
        error_msg = f"❌ ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        yield None, error_msg, {}, None

# Gradio Interface
with gr.Blocks(theme=gr.themes.Ocean(), title="YOLOv8 Person Tracker") as app:
    gr.Markdown("""
    # 🎥 YOLOv8 Person Detection & Tracking
    ### Upload video file → Watch live detection → Download tracked video
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Video File")
            
            # FILE UPLOAD instead of Video component
            video_input = gr.File(
                label="Click to Upload Video",
                file_types=["video"],
                type="filepath"
            )
            
            process_btn = gr.Button(
                "🎬 Start Detection & Tracking", 
                variant="primary", 
                size="lg"
            )
            
            gr.Markdown("""
            **📝 Steps:**
            1. Click above to upload video
            2. Click "Start Detection"
            3. Watch live on right →
            4. Download when done
            
            **✅ Formats:**
            - MP4, AVI, MOV, MKV
            - Any size (be patient!)
            - Any resolution
            """)
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔴 Live Detection")
            live_output = gr.Image(
                label="Real-Time Tracking",
                height=400
            )
    
    gr.Markdown("### 📊 Processing Info & Stats")
    
    with gr.Row():
        result_text = gr.Textbox(
            label="Status & Results",
            lines=12
        )
        stats_json = gr.JSON(label="Live Statistics")
    
    gr.Markdown("### 📥 Download Processed Video")
    
    download_output = gr.File(label="Download Tracked Video")
    
    gr.Markdown("""
    ---
    ## 🚀 What This Does
    
    | Step | Action |
    |------|--------|
    | 1️⃣ | Detects people in every frame |
    | 2️⃣ | Assigns unique ID to each person |
    | 3️⃣ | Tracks people using IoU algorithm |
    | 4️⃣ | Shows live detection feed |
    | 5️⃣ | Saves tracked video with boxes & IDs |
    
    ## 🎯 Features
    
    ✅ **Live Preview** - Watch detection happen in real-time  
    ✅ **Person-Only** - Ignores cars, objects, etc.  
    ✅ **Unique IDs** - Each person gets consistent ID  
    ✅ **Color Coded** - Different colors for different people  
    ✅ **IoU Tracking** - Advanced tracking algorithm  
    ✅ **Downloadable** - Get processed video file  
    
    ## ⚙️ Settings
    
    - **Model:** YOLOv8n (Fastest)
    - **Confidence:** 40%
    - **IoU Threshold:** 0.3
    - **Device:** CPU (works everywhere)
    
    ---
    **Fixed:** File upload instead of Video component to avoid permission errors
    """)
    
    # Connect
    process_btn.click(
        fn=process_video_realtime,
        inputs=[video_input],
        outputs=[live_output, result_text, stats_json, download_output]
    )

if __name__ == "__main__":
    print("=" * 70)
    print("🎥 YOLOv8 Person Detection & Tracking System")
    print("=" * 70)
    print("🔧 Fix Applied: File Upload component (no permission issues)")
    print("📁 Output Directory: tracked_videos/")
    print("🌐 Starting Gradio interface...")
    print("=" * 70)
    app.launch(share=True, inbrowser=True)