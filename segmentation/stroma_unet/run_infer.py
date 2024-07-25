import numpy as np
import cv2
import os

import albumentations as albu
import segmentation_models_pytorch as smp

import torch


# Device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Compile
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def preprocess(image):
    preproces = get_preprocessing(preprocessing_fn)

    image = cv2.resize(image, (512, 512))
    sample = preproces(image=image)
    image = sample['image']

    return image


def inference_stroma(image, stroma_model):
    # Image
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

    # Stroma mask prediction
    pr_mask = stroma_model.to(device).predict(x_tensor)
    pr_mask = pr_mask.squeeze().cpu().numpy()
    pr_mask = pr_mask[1]

    # "Smooth" probabilities: from range [-20,20] to [0,1] using linear transformation
    pr_mask = pr_mask / 40 + 0.5

    # Rescale to be between 0 and 1
    pr_mask = np.maximum(pr_mask, 0)
    pr_mask = np.minimum(pr_mask, 1)

    return pr_mask


def predict_stroma(img, stroma_model):

    # Preprocess image
    image = preprocess(img)

    # Stroma network
    stroma = inference_stroma(image, stroma_model)
    # stroma[(np.amax(prob_mask, 0) <= 0)] = 0

    # Resize to original size
    orig_shape = img.shape[:2]
    stroma = cv2.resize(stroma, orig_shape, interpolation=cv2.INTER_NEAREST)

    return stroma


def main_with_args():

    return

def main():

    main_with_args()


if __name__=='__main__':
    main()