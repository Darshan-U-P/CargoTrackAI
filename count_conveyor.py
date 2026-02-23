from ultralytics import YOLO
import cv2
import numpy as np

# ================= CONFIG =================
MODEL_PATH = "runs/detect/train7/weights/best.onnx"
VIDEO_PATH = "dataset/1.mp4"

CONF_THRESHOLD = 0.6
MIN_BOX_AREA = 4000
SMOOTHING_ALPHA = 0.75

LINE_A_RATIO = 0.45   # Upper exit line
LINE_B_RATIO = 0.65   # Lower entry line
# ==========================================


# Load ONNX model (Ultralytics handles ONNXRuntime internally)
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

ret, frame = cap.read()
if not ret:
    print("Video error")
    exit()

h, w = frame.shape[:2]
line_A = int(h * LINE_A_RATIO)
line_B = int(h * LINE_B_RATIO)

# Tracking state
smooth_centers = {}
object_state = {}
count = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("ONNX + Ultralytics Bottom→Top Counting Started... Press Q to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 Use built-in Ultralytics tracker (ByteTrack)
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    r = results[0]

    if r.boxes.id is not None:

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):

            obj_id = int(obj_id)
            x1, y1, x2, y2 = box

            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue

            center_y = int((y1 + y2) / 2)

            # ---- Smooth vertical center ----
            if obj_id in smooth_centers:
                prev_y = smooth_centers[obj_id]
                center_y = int(
                    SMOOTHING_ALPHA * prev_y +
                    (1 - SMOOTHING_ALPHA) * center_y
                )

            smooth_centers[obj_id] = center_y

            # ---- Initialize state ----
            if obj_id not in object_state:
                object_state[obj_id] = "bottom"

            state = object_state[obj_id]

            # -------- BOTTOM ZONE --------
            if center_y > line_B:
                object_state[obj_id] = "bottom"

            # -------- MIDDLE ZONE --------
            elif line_A <= center_y <= line_B:
                if state == "bottom":
                    object_state[obj_id] = "inside"

            # -------- TOP EXIT ZONE --------
            elif center_y < line_A:
                if state == "inside":
                    count += 1
                    object_state[obj_id] = "counted"

            # ---- Draw Box ----
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        f"ID {obj_id}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0), 2)

    # ---- Draw Lines ----
    cv2.line(frame, (0, line_A), (w, line_A), (255, 0, 0), 3)
    cv2.line(frame, (0, line_B), (w, line_B), (0, 0, 255), 3)

    cv2.putText(frame,
                f"Count: {count}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3)

    cv2.imshow("ONNX Dual-Line Bottom→Top Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Final Count:", count)