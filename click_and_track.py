# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import argparse
from nanosam.utils.predictor import Predictor
from sam_tracker import Tracker


def init_track(event,x,y,flags,param):
    global mask, point
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mask = tracker.init(image, point=(x, y))
        point = (x, y)

# This function reads the results of the KLT tracker (output bounding boxes and estimations) and updates the input bounding boxes and predictions for the next iteration.
def customUpdate(inBoxes, inPreds, outBoxes, outEstim):
    with inBoxes.lock_cpu() as inBoxes_cpu, inPreds.lock_cpu() as inPreds_cpu, \
         outBoxes.rlock_cpu() as outBoxes_cpu, outEstim.rlock_cpu() as outEstim_cpu:
        inBoxes_ = inBoxes_cpu.view(np.recarray)
        inPreds_ = inPreds_cpu.view(np.recarray)
        outBoxes_ = outBoxes_cpu.view(np.recarray)
        outEstim_ = outEstim_cpu.view(np.recarray)
 
        for i in range(outBoxes.size):
            # If the track status of a bounding box is lost, it assigns lost to the corresponding input bouding box.
            if outBoxes_[i].tracking_status == vpi.KLTTrackStatus.LOST:
                inBoxes_[i].tracking_status = vpi.KLTTrackStatus.LOST
                print("Lost")
                continue
 
            # If the template status is update needed, it updates the input bounding box with the corresponding output, making its prediction the identity matrix (fixing the bounding box).
            if outBoxes_[i].template_status == vpi.KLTTemplateStatus.UPDATE_NEEDED:
                inBoxes_[i] = outBoxes_[i]
                inPreds_[i] = np.eye(3)
            else:
                # If the update is not needed, just update the input prediction by the corresponding output estimation.
                inBoxes_[i].template_status = vpi.KLTTemplateStatus.UPDATE_NOT_NEEDED
                inPreds_[i] = outEstim_[i]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="/opt/nanosam/data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="/opt/nanosam/data/mobile_sam_mask_decoder.engine")
    args = parser.parse_args()

    predictor = Predictor(
        args.image_encoder,
        args.mask_decoder
    )

    tracker = Tracker(predictor)

    mask = None
    point = None
    box = None

    cap = cv2.VideoCapture(0)


    cv2.namedWindow('image')
    cv2.setMouseCallback('image',init_track)
    switch_on_yolo = False

    while True:
        re, image = cap.read()

        if not re:
            print("Cannot get streaming")
            break

        if switch_on_yolo:
            pass
            # cvGray, imgReference = convertFrameImage(image)
            # outBoxes = klt(imgReference,update=customUpdate)
            # print(outBoxes)
            # inPreds = klt.in_predictions()
            # inPreds.size = 1
            # inBoxes.size = 1
            # klt_box = writeOutputBox(inBoxes, inPreds)
            # klt_center = box_to_centroid(klt_box)
            # # 绘制KLT识别的box和center
            # if np.any(klt_box != 0):
            #     print("using klt tracker and detect successfully!!!")
            #     start_point = (klt_box[0], klt_box[1])  # 左上角
            #     end_point = (klt_box[2], klt_box[3])  # 右下角
            #     color = (255, 0, 0)  # 绿色
            #     thickness = 2  # 线条的粗细
            #     image = cv2.rectangle(image, start_point, end_point, color, thickness)
            #     assert point is not None
            #     image = cv2.circle(
            #         image,
            #         klt_center,
            #         5,
            #         (185, 118, 0),
            #         -1
            #     )
            #     switch_on_klt = True
            # else:
            #     print("using klt tracker but get lost, try to reopen it by SAM!!!!")
            #     switch_on_klt = False
            #     del klt

        if tracker.token is not None:
            mask, sam_center, sam_box = tracker.update(image)

            # Draw box
            if sam_box is not None:
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
                    sam_center,
                    5,
                    (0, 185, 118),
                    -1
                )

                # TODO: 开始维护yolo跟踪器
                if switch_on_yolo is False:
                    print("switch on yolo tracker!!!")
                    switch_on_yolo = True
            else:
                print("Lost!!! Please click again.")
                tracker.reset()
                mask = None
                box = None
                switch_on_yolo = False
                del klt
                continue

                
        cv2.imshow("image", image)

        ret = cv2.waitKey(1)

        if ret == ord('q'):
            print("Quit.")
            break
        elif ret == ord('r'):
            print("Reset manually!!! Please click again.")
            tracker.reset()
            mask = None
            box = None
            switch_on_yolo = False
            del klt


    cv2.destroyAllWindows()
