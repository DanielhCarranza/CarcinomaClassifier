import fastai
import torch
import numpy as np
import cv2
from fastai.vision import load_learner, Image, open_image, pil2tensor, ImageSegment, ImageList, SegmentationLabelList
import PIL
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

def open_mk(fn, div: bool=False, convert_mode: str='L', cls: type=ImageSegment)->Image:
    "Return `Image` object created from image in file `fn`."
    with warnings.catch_warnings():
        # EXIF warning from TiffPlugin
        warnings.simplefilter("ignore", UserWarning)
        x = PIL.Image.open(fn).convert(convert_mode)
    x = pil2tensor(x, np.float32)
    x[x > 0] = 1
    if div:
        x.div_(255)
    return cls(x)


class GlandSegmentationLabelList(SegmentationLabelList):
    def open(self, fn): return open_mk(fn)


class GlandSegmentationItemList(ImageList):
    _label_cls = GlandSegmentationLabelList


def accuracy_gland(input, target):
    target = target.squeeze(1)
    mask = target > 0
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


def show_mask(image: torch.Tensor, mask: torch.Tensor, alpha=0.5, cmap='tab20', figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.imshow(torch.squeeze(mask, dim=0), cmap=cmap, alpha=alpha)
    plt.grid(False)
    plt.show()


class Inference():
    def __init__(self, path, file_name):
        """ Inference 
        :path = path to the .pkl file
        :file_name
        return: class prediction
        """
        self.learner = load_learner(path, file_name)
        self.learner.model.float()

    def __call__(self, img):
        img = open_image(img) if not isinstance(
            img, fastai.vision.image.Image) else img
        masks = self.learner.predict(img) # get mask prediction
        # transform img
        img = self.learner.data.one_item(img)[0]
        if getattr(self.learner.data, 'norm', False):
            img = self.learner.data.denorm(img)
        img = torch.squeeze(img, dim=0)
        image = Image(img)
        return image, masks[0]


if __name__=="__main__":
    path_local_models = Path('models')
    model_file = 'gland_unet_224-93d.pkl'
    predict_img_seg = Inference(path_local_models, model_file)

    img_file = Path('seg_img.bmp')  # path to img
    image, mask_pred = predict_img_seg(open_image(img_file))
    image.show(y=mask_pred, figsize=(10,10), alpha=0.6)

    # show_mask(image.data, mask_pred.data, alpha=0.6)