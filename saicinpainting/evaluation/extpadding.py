from typing import Tuple
from PIL import Image
import torch
from torch import nn
import numpy as np
import numpy.typing as npt
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.evaluation.utils import move_to_device


def _mask_resize(mask_tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Return resized mask.

    Parameters:
    - mask_tensor(torch.Tensor): given by batch['mask'] shape (1, 1, W, H)
    - ratio(float): ratio value for objective width or height.

    Returns:
    - mask_tensor(torch.Tensor): shape (1, 1, W * ratio, H * ratio)
    """
    original_img = _convert_tensor_to_pil_img(mask_tensor, mode="L")
    width, height = original_img.size
    new_width, new_height = map(lambda x: round(x * ratio), (width, height))

    paste_left, paste_top = map(
        lambda t: round(abs(t[0] - t[1]) / 2),
        zip((width, height), (new_width, new_height)),
    )

    if ratio > 1:
        canvas = Image.new(original_img.mode, (new_width, new_height), (0,))
        canvas.paste(original_img, (paste_left, paste_top))
    elif ratio < 1:
        canvas = Image.new(original_img.mode, (width, height), (0,))

        canvas.paste(
            original_img.resize(
                (new_width, new_height),
                resample=Image.Resampling.BICUBIC,
                reducing_gap=3.0,
            ),
            (paste_left, paste_top),
        )
    return default_collate([_convert_pil_img_to_tensor(canvas, mode="L")])


def _add_duplicated_padding(
    img_tensor: torch.Tensor, ratio: float
) -> torch.Tensor:
    """
    Return padded image tensor.

    Parameters:
    - img_tensor(torch.Tensor): given by batch['image'] shape (1, 3, W, H)
    - ratio(float): ratio value for objective width or height.

    Returns:
    - img_tensor(torch.Tensor): shape (1, 3, W * ratio, H * ratio)
    """
    original_img = _convert_tensor_to_pil_img(img_tensor, mode="RGB")
    width, height = original_img.size
    new_width, new_height = map(lambda x: round(x * ratio), (width, height))

    paste_left, paste_top = map(
        lambda t: round(abs(t[0] - t[1]) / 2),
        zip((width, height), (new_width, new_height)),
    )

    canvas = original_img.resize(
        (new_width, new_height),
        resample=Image.Resampling.BICUBIC,
        reducing_gap=3.0,
    )
    if ratio > 1:
        canvas.paste(original_img, (paste_left, paste_top))
    elif ratio < 1:
        original_img.paste(canvas, (paste_left, paste_top))
        canvas = original_img.copy()
    return _convert_pil_img_to_tensor(canvas)


def _convert_tensor_to_pil_img(
    tensor: torch.Tensor, mode="RGB"
) -> Image.Image:
    """
    Retore batch['image'] torch.Tensor to PIL.Image object.

    Parameters:
    - tensor(torch.Tensor): torch.Tensor given with batch[key], shape(1, C, W, H)
    - mode(str): PIL.Image palette mode. Default value is "RGB"

    Returns:
    - img: PIL.Image object
    """
    if mode == "L":
        return Image.fromarray(
            (np.array(tensor[0][0]) * 255).astype("int8"),
            mode=mode,
        )
    else:
        return Image.fromarray(
            np.transpose(
                (np.array(tensor[0]) * 255).astype("int8"), (1, 2, 0)
            ),
            mode=mode,
        )


def _convert_pil_img_to_tensor(
    pil_img: Image.Image, mode="RGB"
) -> torch.Tensor:
    """
    Convert PIL.Image object to torch.Tensor which has same property with
    batch['image'] tensor.

    note: Inverse of _convert_tensor_to_pil_img

    Parameters:
    - img(PIL.Image.Image): PIL.Image object
    - mode(str): PIL.Image palette mode. Default value is "RGB"

    Returns:
    - tensor(torch.Tensor): shape(1, C, W, H)
    """

    img = np.array(pil_img.convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype("float32") / 255
    return default_collate([out_img])


def _convert_arr_to_pil_img(arr: npt.NDArray[np.float32], mode="RGB") -> Image.Image:
    """
    Retore np.array which is loaded by saicinpainting.evaluation.data.load_image
    to PIL.Image object.

    Parameters:
    - arr(np.array): np.array loaded by saicinpainting.evaluation.data.load_image
    - mode(str): PIL.Image palette mode. Default value is "RGB"

    Returns:
    - img: PIL.Image object
    """
    return Image.fromarray((arr * 255).astype("int8"), mode=mode)


def _convert_pil_img_to_arr(pil_img: Image.Image, mode="RGB") -> npt.NDArray[np.float32]:
    """
    Convert PIL.Image object to np.array which has same property with what would
    be returned from saicinpainting.evaluation.data.load_image.

    note: Inverse of _convert_arr_to_pil_img

    Parameters:
    - img(PIL.Image.Image): PIL.Image object
    - mode(str): PIL.Image palette mode. Default value is "RGB"

    Returns:
    - arr(np.array): np.array same with saicinpainting.evaluation.data.load_image
    """

    img = np.array(pil_img.convert(mode))
    out_img = img.astype("float32") / 255
    return out_img


def _crop_image(img_arr: npt.NDArray[np.float32], expanded_ratio: float) -> npt.NDArray[np.float32]:
    """
    Return np.array of cropped image.

    Parameters:
    - img_arr(np.array): padded image. given by cur_res
    - expanded_ratio(float): The ratio that the given image has been expanded.

    Returns:
    - img_arr(np.array): cropped image array
    """
    original_img = _convert_arr_to_pil_img(img_arr, "RGB")
    width, height: Tuple[int, int] = original_img.size

    crop_width, crop_height: Tuple[int, int] = map(
        lambda x: round(
            (x / expanded_ratio)
            if expanded_ratio > 1
            else (x * expanded_ratio)
        ),
        (width, height),
    )

    box_left, box_top: Tuple[int, int] = map(
        lambda t: round(abs(t[0] - t[1]) / 2),
        zip((width, height), (crop_width, crop_height)),
    )

    new_size: Tuple[int, int] = (
        (crop_width, crop_height) if expanded_ratio > 1 else (width, height)
    )
    return _convert_pil_img_to_arr(
        original_img.resize(
            new_size,
            box=(box_left, box_top, width - box_left, height - box_top),
            resample=Image.Resampling.BICUBIC,
            reducing_gap=3.0,
        ),
        mode="RGB",
    )


def extpadding_predict(
    batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    predict_config: OmegaConf,
) -> npt.NDArray[np.float32]:
    """
    Return np.array of predicted image.
    This process includes exterior padding preprocess and cropping.

    config.extpadding float value is required in predict_config.
    """
    with torch.no_grad():
        batch["image"] = _add_duplicated_padding(
            batch["image"], predict_config.extpadding
        )
        batch["mask"] = _mask_resize(batch["mask"], predict_config.extpadding)
        batch = move_to_device(batch, device)
        batch["mask"] = (batch["mask"] > 0) * 1

        batch = model(batch)
        cur_res = (
            batch[predict_config.out_key][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        unpad_to_size = batch.get("unpad_to_size", None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]
        return _crop_image(cur_res, predict_config.extpadding)
