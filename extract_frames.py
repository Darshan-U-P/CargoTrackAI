import cv2
import os
from glob import glob

# ================= CONFIG =================
VIDEO_FOLDER = "dataset"      # folder containing videos
OUTPUT_ROOT = "real_frames"      # root output folder
FRAME_SKIP = 5                   # save every Nth frame
# ==========================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)

video_paths = glob(os.path.join(VIDEO_FOLDER, "*.mp4")) + \
              glob(os.path.join(VIDEO_FOLDER, "*.avi"))

print(f"Found {len(video_paths)} videos.")

for video_path in video_paths:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(OUTPUT_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    save_id = 0

    print(f"\nProcessing {video_name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP == 0:
            save_path = os.path.join(output_dir, f"{video_name}_{save_id}.jpg")
            cv2.imwrite(save_path, frame)
            save_id += 1

        frame_id += 1

    cap.release()

    print(f"Saved {save_id} frames from {video_name}")

print("\nAll videos processed.")