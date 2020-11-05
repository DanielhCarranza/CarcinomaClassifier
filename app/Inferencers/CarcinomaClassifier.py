import fastai
import cv2
from fastai.callbacks.hooks import hook_output
from fastai.vision import  load_learner, pil2tensor, PIL,ImageList,Image
import numpy as np 
import matplotlib.pyplot as plt

def heatMap(x, data, learner, size=(0, 720, 720, 0)):
    """HeatMap"""

    # Evaluation mode
    m = learner.model.eval()

    # Denormalize the image
    xb, _ = data.one_item(x)
    xb_im = Image(data.denorm(xb)[0])
    # xb = xb.cuda()

    # hook the activations
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0, 1].backward()

    # Activations
    acts = hook_a.stored[0].cpu()

    # Avg of the activations
    avg_acts = acts.mean(0)

    # Show HeatMap
    _, ax = plt.subplots(figsize=(15,15))
    xb_im.show(ax)
    ax.imshow(avg_acts, alpha=0.8, extent=size,
              interpolation='bilinear', cmap='magma')
    plt.show()


def open_im(img_stream):
    img = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)
    img = pil2tensor(PIL.Image.fromarray(img), np.float32)
    img = img.div_(255)
    return Image(img)

def open_img(fn):
    im = cv2.imread(str(fn)) # open .tif images 
    img = pil2tensor(PIL.Image.fromarray(im), np.float32)
    img = img.div_(255)
    return Image(img)

class CancerImageList(ImageList):
  def open(self, fn): return open_img(fn)

class Inference():

    def __init__(self, path, file_name:str):
        """ Inference 
        :path = path to the .pkl file
        :file_name
        return: class prediction
        """
        self.learner = load_learner(path, file_name)
        self.learner.model.float()
        self.classes = self.learner.data.single_ds.y.classes
    def __call__(self, img, print_probs=False, heatmap=False):
        img = open_img(img) if not isinstance(
            img, Image) else img
        pred_class, idx, probs = self.learner.predict(img)
        if print_probs:
          for i, p in enumerate(probs):
              print(f"{self.classes[i]} {(p*100):.3f} %")
        if heatmap:
            self.show_heatmap(img)
        return pred_class, idx, probs

    def show_heatmap(self, img: Image):
        heatMap(img, self.learner.data, self.learner, size=(0, 1024, 1024, 0))
        


# path_local_models='models'
# model_file = 'breast_cancer_1024-98acc.pkl'
# predict_img = Inference(path_local_models, model_file)

# img_file= 'invasive_img.tif' # path to img 
# pred_class, idx, probs= predict_img(img_file)

