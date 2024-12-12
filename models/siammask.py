import os 
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
from loguru import logger

class SiamMask:
    def __init__(self, onnx_path:str, save_opt_onnxruntime:bool=True, do_profiling:bool=False) -> None:
        # Variables for pre and postprocessing for init and track
        self.exemplar_size = 127
        self.context_amount = 0.5
        self.instance_size = 255
        self.out_size = 127
        self.total_stride = 8
        self.base_size = 8
        self.segmentation_thres = 0.3
        self.avg_chans = None

        # ONNX Setup:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()

        if do_profiling is True and not "CUDAExecutionProvider" in providers:
            do_profiling = False
            logger.warning("Cannot profile without CUDA provider and CuPY. Ignoring profile parameter...")

        if do_profiling:
            import cupy as cp
            self.cp = cp
            self.t1 = cp.cuda.Event()
            self.t2 = cp.cuda.Event()
            self.infer_init = 0.0
            self.avg_infer_reg = []
        self.do_profiling = do_profiling

        if save_opt_onnxruntime:
            optimized_path = "{}_ort.onnx".format(Path(onnx_path).stem)
            if os.path.exists(optimized_path):
                onnx_path = optimized_path
            else:
                sess_options.optimized_model_filepath = optimized_path

        self.model = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=sess_options
        )

        # Extra stuff:
        self.dummy_z_feat = np.zeros((1, 256, 7, 7), dtype=np.float32)
        self.mask_color = (0, 0, 255)

        # 预分配输入用的数组，减少重复创建
        # 这些数组在 init 和 forward 时大小不变，只需更新值
        self.target_pos0_array = np.zeros((2,), dtype=np.float32)
        self.target_sz0_array = np.zeros((2,), dtype=np.float32)
        self.scale_x_array = np.zeros((1,), dtype=np.float64)
        self.z_features0_array = self.dummy_z_feat.copy()
        self.first_time_array = np.zeros((1,), dtype=bool)

    def init(self, im: np.ndarray, coordinates: tuple) -> None:
        x, y, w, h = coordinates

        self.avg_chans = np.mean(im, axis=(0, 1))
        target_pos0 = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        target_sz0 = np.array([w, h], dtype=np.float32)

        wc_z = target_sz0[0] + self.context_amount * sum(target_sz0)
        hc_z = target_sz0[1] + self.context_amount * sum(target_sz0)
        s_z = round(np.sqrt(wc_z * hc_z))

        im1 = np.zeros((1, 3, self.instance_size, self.instance_size), dtype=np.float32)
        _im1 = self.__preprocess__(
            im, target_pos0, self.exemplar_size, s_z, self.avg_chans
        )
        im1[:, :, :127, :127] = _im1

        # 更新输入数组，而不是每次都创建新数组
        self.target_pos0_array[:] = target_pos0
        self.target_sz0_array[:] = target_sz0
        self.scale_x_array[0] = 100
        self.z_features0_array[:] = self.dummy_z_feat
        self.first_time_array[0] = True

        if self.do_profiling: self.t1.record()
        outputs = self.model.run(
            ["output", "target_pos1", "target_sz1", "z_features1", "delta_yx"],
            {
                "im": im1,
                "target_pos0": self.target_pos0_array,
                "target_sz0": self.target_sz0_array,
                "scale_x": self.scale_x_array,
                "z_features0": self.z_features0_array,
                "first_time": self.first_time_array,
            },
        )
        if self.do_profiling:
            self.t2.record()
            self.cp.cuda.runtime.deviceSynchronize()
            profile_time = self.cp.cuda.get_elapsed_time(self.t1, self.t2)
            self.infer_init = profile_time

        self.target_pos = outputs[1]
        self.target_sz = outputs[2]
        self.z_feature = outputs[3]

    def forward(self, im: np.ndarray) -> np.ndarray:
        assert (
            not self.avg_chans is None
        ), "All variables not initialized properly. Did you run init?"

        wc_x = self.target_sz[1] + self.context_amount * sum(self.target_sz)
        hc_x = self.target_sz[0] + self.context_amount * sum(self.target_sz)
        s_x = np.sqrt(wc_x * hc_x)
        scale_x = self.exemplar_size / s_x
        d_search = (self.instance_size - self.exemplar_size) / 2
        pad = d_search / scale_x
        s_x = s_x + 2 * pad

        crop_box = [
            self.target_pos[0] - round(s_x) / 2,
            self.target_pos[1] - round(s_x) / 2,
            round(s_x),
            round(s_x),
        ]
        im1 = self.__preprocess__(
            im, self.target_pos, self.instance_size, round(s_x), self.avg_chans
        )
        im1 = np.expand_dims(im1, axis=0)  # shape: (1, 3, 255, 255)

        # 更新输入数组
        self.target_pos0_array[:] = self.target_pos
        self.target_sz0_array[:] = self.target_sz
        self.scale_x_array[0] = scale_x
        self.z_features0_array[:] = self.z_feature
        self.first_time_array[0] = False

        if self.do_profiling: self.t1.record()
        outputs = self.model.run(
            ["output", "target_pos1", "target_sz1", "z_features1", "delta_yx"],
            {
                "im": im1,
                "target_pos0": self.target_pos0_array,
                "target_sz0": self.target_sz0_array,
                "scale_x": self.scale_x_array,
                "z_features0": self.z_features0_array,
                "first_time": self.first_time_array,
            },
        )
        if self.do_profiling:
            self.t2.record()
            self.cp.cuda.runtime.deviceSynchronize()
            profile_time = self.cp.cuda.get_elapsed_time(self.t1, self.t2)
            self.avg_infer_reg.append(profile_time)

        mask = outputs[0]
        self.target_pos = outputs[1]
        self.target_sz = outputs[2]
        self.z_feature = outputs[3]
        delta_yx = outputs[4]

        s = crop_box[2] / self.instance_size
        sub_box = [
            crop_box[0] + (delta_yx[1] - self.base_size / 2) * self.total_stride * s,
            crop_box[1] + (delta_yx[0] - self.base_size / 2) * self.total_stride * s,
            s * self.exemplar_size,
            s * self.exemplar_size,
        ]
        s = self.out_size / sub_box[2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im.shape[1] * s, im.shape[0] * s]

        a = (im.shape[1] - 1) / back_box[2]
        b = (im.shape[0] - 1) / back_box[3]
        c = -a * back_box[0]
        d = -b * back_box[1]
        mapping = np.array([[a, 0, c], [0, b, d]], dtype=np.float32)
        crop = cv2.warpAffine(
            mask,
            mapping,
            (im.shape[1], im.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
        target_mask = (crop > self.segmentation_thres).astype(np.uint8)
        return target_mask

    def __preprocess__(self, im, pos, model_sz, original_sz, avg_chans) -> np.ndarray:
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = round(pos[0] - c)
        context_xmax = context_xmin + sz - 1
        context_ymin = round(pos[1] - c)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0.0, -context_xmin))
        top_pad = int(max(0.0, -context_ymin))
        right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            te_im = np.zeros(
                (r + top_pad + bottom_pad, c + left_pad + right_pad, k), dtype=im.dtype
            )
            te_im[top_pad : top_pad + r, left_pad : left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad : left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad :, left_pad : left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad :, :] = avg_chans
            im_patch_original = te_im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]
        else:
            im_patch_original = im[
                int(context_ymin) : int(context_ymax + 1),
                int(context_xmin) : int(context_xmax + 1),
                :,
            ]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
        else:
            im_patch = im_patch_original

        im_patch = np.transpose(im_patch, (2, 0, 1))  # C*H*W
        # 确保数据为float32并连续，减少每次推理转换开销
        im_patch = im_patch.astype(np.float32, copy=False)
        im_patch = np.ascontiguousarray(im_patch)
        return im_patch
