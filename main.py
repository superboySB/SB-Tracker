import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from models import TRTModule  # isort:skip
from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox
from nanosam.utils.predictor import Predictor
from perception.sam_tracker import Tracker

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    # For tracker
    predictor = Predictor(
        args.image_encoder,
        args.mask_decoder
    )
    tracker = Tracker(predictor)

    mask = None
    point = None

    def init_track(event,x,y,flags,param):
        global mask, point
        if event == cv2.EVENT_LBUTTONDBLCLK:
            mask = tracker.init(image, point=(x, y))
            point = (x, y)

    cap = cv2.VideoCapture(0)  # 使用USB摄像头
    cv2.namedWindow('result')
    cv2.setMouseCallback('result',init_track)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Cannot get streaming")
            break
        bgr = frame
        image = bgr.copy()
        
        if tracker.token is not None:
            mask, point, sam_box = tracker.update(image)

            # Draw mask,box,center point
            if mask is not None:
                bin_mask = (mask[0,0].detach().cpu().numpy() < 0)
                green_image = np.zeros_like(image)
                green_image[:, :] = (0, 185, 118)
                green_image[bin_mask] = 0

                image = cv2.addWeighted(image, 0.4, green_image, 0.6, 0)

                # 绘制SAM识别的box和center
                # cv2.rectangle 需要传入左上角和右下角的坐标
                # box 应该是形如 [min_x, min_y, max_x, max_y] 的数组
                start_point = (sam_box[0], sam_box[1])  # 左上角
                end_point = (sam_box[2], sam_box[3])  # 右下角
                color = (0, 255, 0)  # 绿色
                thickness = 2  # 线条的粗细

                image = cv2.rectangle(image, start_point, end_point, color, thickness)
                assert point is not None
                image = cv2.circle(
                    image,
                    point,
                    5,
                    (0, 185, 118),
                    -1
                )
            else:
                print("Lost!!! Please click again.")
                tracker.reset()
                mask = None
                continue
        else:
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
            tensor = torch.asarray(tensor, device=device)
            # inference
            data = Engine(tensor)

            bboxes, scores, labels = det_postprocess(data)
            if bboxes.numel() == 0:
                print('No object detected!')
            else:
                bboxes -= dwdh
                bboxes /= ratio

                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    cls = CLASSES[cls_id]
                    color = COLORS[cls]
                    cv2.rectangle(image, bbox[:2], bbox[2:], color, 2)
                    cv2.putText(image,
                                f'{cls}:{score:.2f}', (bbox[0], bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, color,
                                thickness=2)

        cv2.imshow('result', image)

        ret = cv2.waitKey(1)

        if ret == ord('q'):
            print("Quit.")
            break
        elif ret == ord('r'):
            print("Reset manually!!! Please click again.")
            tracker.reset()
            mask = None

    cap.release()
    cv2.destroyAllWindows()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/workspace/YOLOv8-TensorRT/yolov8s.engine", help='Engine file')
    parser.add_argument('--device', type=str, default='cuda:0', help='TensorRT infer device')
    parser.add_argument("--image_encoder", type=str, default="/opt/nanosam/data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="/opt/nanosam/data/mobile_sam_mask_decoder.engine")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
