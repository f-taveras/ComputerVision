import cv2 as cv
import imutils
import argparse
from ultralytics import YOLO
import numpy as np





FLOOR_POLYGON = np.array([[100, 300], [700, 300], [750, 600], [50, 600]])

def point_inside_polygon(x, y, polygon):
    return cv.pointPolygonTest(polygon, (x, y), False) >= 0

def detect_yolov8(frame, model):
    results = model(frame)[0]
    person_count = 0
    in_floor_count = 0

    # Draw the floor area
    cv.polylines(frame, [FLOOR_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label == 'person' and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_count += 1
            center_bottom = (int((x1 + x2) / 2), y2)

            # Check if the bottom center point is inside the floor area
            in_floor = point_inside_polygon(center_bottom[0], center_bottom[1], FLOOR_POLYGON)

            if in_floor:
                color = (0, 255, 0)
                in_floor_count += 1
                status_text = f"IN FLOOR {in_floor_count}"
            else:
                color = (0, 0, 255)
                status_text = "OUTSIDE"

            # Draw bounding box and status
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"Person {person_count}: {status_text}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv.putText(frame, 'Status: Detecting ...', (40, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.putText(frame, f'Total: {person_count} | In Floor: {in_floor_count}', (40, 70), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.imshow("output", frame)

    return frame

def use_video(path, writer, model):
    cap = cv.VideoCapture(path)

    if not cap.isOpened():
        print("Video not found or cannot be opened.")
        return

    print("Detecting ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame = detect_yolov8(frame, model)

        if writer is not None:
            writer.write(frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def argsparser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", help="Path to video file")
    arg_parse.add_argument("-m", "--model", default="yolov8n.pt", help="Path to YOLOv8 model file (e.g., yolov8n.pt)")
    return vars(arg_parse.parse_args())


if __name__ == "__main__":
    args = argsparser()
    video_path = args['video']
    model_path = args['model']

    # Load YOLOv8 model
    model = YOLO(model_path)

    writer = None
    use_video(video_path, writer, model)
