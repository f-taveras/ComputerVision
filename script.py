import cv2 as cv
import imutils
import argparse
from ultralytics import YOLO

# Detection using YOLOv8
def detect_yolov8(frame, model):
    results = model(frame)[0]
    person_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label == 'person' and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_count += 1
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"Person {person_count}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.putText(frame, 'Status: Detecting ...', (40, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.putText(frame, f'Total: {person_count}', (40, 70), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
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
