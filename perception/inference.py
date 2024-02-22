import argparse
from copy import deepcopy
from typing import Any, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from .inferencer import SAMDecoderInferencer, SAMEncoderInferencer
from torchvision.transforms.functional import resize


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """

        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def preprocess(x, img_size):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

    return x


def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
    img_size = 1024
    masks = torch.tensor(masks)
    orig_im_size = torch.tensor(orig_im_size)

    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_coords(coords, original_size, new_size):
    old_h, old_w = original_size
    new_h, new_w = new_size
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords



def calculate_bounding_box(mask):
    """
    Calculate the bounding box coordinates from the mask.
    """
    pos = np.where(mask)
    x_min = np.min(pos[1])
    x_max = np.max(pos[1])
    y_min = np.min(pos[0])
    y_max = np.max(pos[0])
    return x_min, y_min, x_max, y_max

def show_bounding_box(ax, bbox, color='red', linewidth=2):
    """
    Display the bounding box on the image.
    """
    x_min, y_min, x_max, y_max = bbox
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor=color, facecolor='none', linewidth=linewidth))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model type.")
    parser.add_argument("--encoder_engine", type=str, required=True, help="TRT engine.")
    parser.add_argument("--decoder_engine", type=str, required=True, help="TRT engine.")
    parser.add_argument("--img_path", type=str, default="assets/bus.jpg")
    parser.add_argument("--out_path", type=str, default="assets/efficientvit_sam_demo_tensorrt.png")
    parser.add_argument("--mode", type=str, default="point", choices=["point", "boxes"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--boxes", type=str, default=None)
    args = parser.parse_args()

    trt_encoder = SAMEncoderInferencer(args.encoder_engine, batch_size=1)

    raw_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    origin_image_size = raw_img.shape[:2]

    if args.model in ["l0", "l1", "l2"]:
        img = preprocess(raw_img, img_size=512)
    elif args.model in ["xl0", "xl1"]:
        img = preprocess(raw_img, img_size=1024)
    else:
        raise NotImplementedError

    image_embedding = trt_encoder.infer(img)
    image_embedding = image_embedding[0].reshape(1, 256, 64, 64)

    input_size = get_preprocess_shape(*origin_image_size, long_side_length=1024)

    H, W, _ = raw_img.shape
    point = np.array(yaml.safe_load(args.point or f"[[[{W // 2}, {H // 2}, {1}]]]"), dtype=np.float32)
    point_coords = point[..., :2]
    point_labels = point[..., 2]
    orig_point_coords = deepcopy(point_coords)
    orig_point_labels = deepcopy(point_labels)
    point_coords = apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)

    inputs = (image_embedding, point_coords, point_labels)

    trt_decoder = SAMDecoderInferencer(args.decoder_engine, num=point.shape[1], batch_size=1)
    low_res_masks, _ = trt_decoder.infer(inputs)
    low_res_masks = low_res_masks.reshape(1, 1, 256, 256)

    masks = mask_postprocessing(low_res_masks, origin_image_size)
    masks = masks > 0.0

    plt.imshow(raw_img)
    
    bbox = calculate_bounding_box(masks[0].squeeze().numpy())
    show_bounding_box(plt.gca(), bbox)
    show_mask(masks[0], plt.gca())
    show_points(orig_point_coords, orig_point_labels, plt.gca())
    plt.axis("off")
    plt.savefig(args.out_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
    print(f"Result saved in {args.out_path}")


