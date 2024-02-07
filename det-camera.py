import argparse
from pathlib import Path
import cv2
import torch
from models import TRTModule  # isort:skip
from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    cap = cv2.VideoCapture(0)  # 使用USB摄像头

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        bgr = frame
        draw = bgr.copy()
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
            cv2.imshow('result', draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.2f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color,
                        thickness=2)

        cv2.imshow('result', draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/worksapce/YOLOv8-TensorRT/yolov8s.engine", help='Engine file')
    parser.add_argument('--device', type=str, default='cuda:0', help='TensorRT infer device')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
