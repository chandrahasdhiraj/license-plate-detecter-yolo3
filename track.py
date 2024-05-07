from ultralytics import YOLO
import cv2 as cv

# load yolo8 model
model = YOLO('yolov8n.pt')

# load video
video_path = './test.mp4'
capture = cv.VideoCapture(video_path)
ret = True

# read frames
while ret:
    ret, frame = capture.read()

    if ret:
        # detect object and track objects from model
        results = model.track(frame, persist=True)

        # plot results
        frame_ = results[0].plot()

        # visualize
        cv.imshow('Frame',frame_)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
