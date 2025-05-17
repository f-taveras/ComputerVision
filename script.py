import cv2 as cv
import numpy as np
import imutils
import argparse

# Load video
# video = cv.VideoCapture("videos/dogovideo.mp4")
# backSub = cv.createBackgroundSubtractorMOG2()
# resized_video = cv.resize(video, (1080, 720))



# if not video.isOpened():
#     print("Error opening the video")
#     exit()


# detecting function. Will draw bounding boxes around detected objects and enumarete them as "Person 1", "Person 2", etc.

def detect(resized_video):
    bounding_boxes, weights = hog.detectMultiScale(resized_video, winStride=(4, 4), padding=(8, 8), scale=1.05)

    person = 1
    for x, y, w, h in bounding_boxes:
        cv.rectangle(resized_video, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(resized_video, f"Person {person}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        person += 1

    cv.putText(resized_video, 'Status: Detecting ...', (40, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.putText(resized_video, f'Total: {person - 1}', (40, 70), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv.imshow('output', resized_video)

    return resized_video


# Human detection function

def humanDetection(args):
    video_path = args['video']

    writer = None
    if video_path is not None:
        print("[INFO] opening video file...")
        useVideo(video_path, writer)


# Gets video and resizes it 

def useVideo(path, writer): # change video for path to use args in the console

    video = cv.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print("Video not found")
        return
    
    print("Detecting ..")
    while video.isOpened():
        check, frame = video.read()
        
        if check:
            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            frame = detect(frame)

            if writer is not None:
                writer.write(frame)

            key = cv.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()



def argsparser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to video file")
    args = vars(arg_parse.parse_args())

    return args


if __name__ == "__main__":
    # Initialize HOG person detector
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    args = argsparser()
    humanDetection(args)



