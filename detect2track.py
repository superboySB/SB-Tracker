from ultralytics import YOLO
import cv2
import torch
import time
model = YOLO('/workspace/YOLOv8-TensorRT/yolov8m-oiv7.engine', task='detect')

# Open the video file
# video_path = "video/video4.mp4"
# cap = cv2.VideoCapture(video_path)#video

cap = cv2.VideoCapture(0)#USB

#CSI camera
# cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12,framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER)

count=0
max_id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while cap.isOpened():
    # Read a frame from the video
    success,frame = cap.read()

    if success:
        loop_start=cv2.getTickCount()
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results=model.track(source=frame,persist=True)
        for result in results:
            if result.boxes.id is not None:
                if count<result.boxes.id.cpu().numpy().astype(int)[-1]:
                    count=result.boxes.id.cpu().numpy().astype(int)[-1]
        #fps   
        loop_time=cv2.getTickCount()-loop_start
        total_time=loop_time/(cv2.getTickFrequency())
        fps=int(1/total_time)  

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame,"total %d"%count,[40,40], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        cv2.putText(annotated_frame,"fps %d"%fps,[40,80], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

        # Display the annotated frame
        cv2.imshow("YOLOv8 onnx Tracking", annotated_frame) 
        print(fps)
   # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()