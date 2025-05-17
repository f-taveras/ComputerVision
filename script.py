import cv2 as cv
import numpy as np

# Load video (or camera)
video = cv.VideoCapture("videos/indians.mp4")  # Use 0 for webcam
backSub = cv.createBackgroundSubtractorMOG2()

if not video.isOpened():
    print("Error opening the video")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame for performance and consistency
    frame_resized = cv.resize(frame, (1080, 720))

    # Apply background subtraction
    fg_mask = backSub.apply(frame_resized)

    # Threshold to filter out shadows (MOG2 outputs gray for shadows)
    _, mask_thresh = cv.threshold(fg_mask, 100, 255, cv.THRESH_BINARY)

    # Clean up the mask using erosion (removes noise)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv.morphologyEx(mask_thresh, cv.MORPH_ERODE, kernel)

    # Find contours from the binary mask
    contours, _ = cv.findContours(mask_eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw all contours for debugging (optional)
    frame_with_contours = frame_resized.copy()
    cv.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)

    # Minimum area to consider as a person
    min_contour_area = 1000

    # Filter large enough contours
    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]

    # Final output frame with bounding boxes
    frame_out = frame_resized.copy()
    for cnt in large_contours:
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Filter based on aspect ratio and area (optional)
        if 0.3 < aspect_ratio < 3 and area < 50000:
            # Draw bounding box
            cv.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Draw area label
            cv.putText(frame_out, f"{int(area)}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show results
    cv.imshow("Bounding Box", frame_out)

    # Exit with 'q'
    if cv.waitKey(30) & 0xFF == ord("q"):
        break

video.release()
cv.destroyAllWindows()
