# base code taken from: https://stackoverflow.com/questions/68969235/select-roi-on-video-stream-while-its-playing
import numpy as np
import cv2
from loguru import logger
from model import SiamMask
from argparse import ArgumentParser


# Our ROI, defined by two points
def on_mouse(event, x, y, flags, userdata):
    global p1, p2, drawing, show_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click down (select first point)
        drawing = True
        show_drawing = True
        p1 = x, y
        p2 = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        # Drag to second point
        if drawing:
            p2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # Left click up (select second point)
        drawing = False
        p2 = x, y


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--cam_id", type=int, default=0, required=False)
    args = parser.parse_args()

    logger.info("Load in model...")
    model = SiamMask(args.model)

    logger.info("Load in camera feed...")
    cap = cv2.VideoCapture(args.cam_id, cv2.CAP_V4L)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    # Global variable set:
    p1, p2 = (0, 0), (0, 0)
    drawing = False
    show_drawing = False
    track_initialized = False

    if not cap.isOpened():
        exit()

    cv2.setMouseCallback("Frame", on_mouse)

    while True:
        val, fr = cap.read()
        if not val:
            break

        if show_drawing:
            # Fix p2 to be always within the frame
            p2 = (
                0 if p2[0] < 0 else (p2[0] if p2[0] < fr.shape[1] else fr.shape[1]),
                0 if p2[1] < 0 else (p2[1] if p2[1] < fr.shape[0] else fr.shape[0]),
            )
            cv2.rectangle(fr, p1, p2, (0, 0, 255), 2)
            avg_y = (p1[1] + p2[1]) // 2
            avg_x = (p1[0] + p2[0]) // 2

        pressed = cv2.waitKey(1)
        if pressed in [13, 32]:
            # Pressed Enter or Space to use ROI
            drawing = False
            show_drawing = False
            # here do something with ROI points values (p1 and p2)
            logger.info("Model set initialization")
            x, y, w, h = p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]
            print(x, y, w, h)
            model.init(fr, (x,y,w,h))
            track_initialized = True
        elif pressed in [ord("c"), ord("C"), 27]:
            # Pressed C or Esc to cancel ROI
            drawing = False
            show_drawing = False
            track_initialized = False
        elif pressed in [ord("q"), ord("Q")]:
            # Pressed Q to exit
            break

        if track_initialized:
            mask = model.forward(fr)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]

            if len(contours) != 0 and np.max(cnt_area) > 100:
                contour = contours[np.argmax(cnt_area)]  # use max area polygon
                polygon = contour.reshape(-1, 2)
                fr[:, :, 2] = (mask > 0) * 255 + (mask == 0) * fr[:, :, 2]
                fr = cv2.polylines(fr, [polygon], True, (0,0,255), 3)

        cv2.imshow("Frame", fr)
        
    cap.release()
    cv2.destroyAllWindows()