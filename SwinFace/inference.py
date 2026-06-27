import argparse

import cv2
import numpy as np
import torch

from model import build_model


def load_swinface_model(cfg, weight, device='cpu'):
    model = build_model(cfg)
    dict_checkpoint = torch.load(weight, map_location=device)
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def inference(cfg, weight, imgs, device='cpu'):
    model = load_swinface_model(cfg, weight, device=device)
    return inference_model(model, imgs)


def prepare_images(imgs):
    if isinstance(imgs, str):
        imgs = [cv2.imread(imgs)]
        if imgs[0] is None:
            raise FileNotFoundError(f"Image not found: {imgs[0]}")

    if isinstance(imgs, np.ndarray):
        if imgs.ndim == 3:
            imgs = [imgs]
        elif imgs.ndim == 4:
            pass
        else:
            raise ValueError("Input numpy array must be shape HxWx3 or NxHxWx3")

    elif isinstance(imgs, (list, tuple)):
        if len(imgs) == 0:
            raise ValueError("Input image list is empty")
        imgs = [cv2.imread(img) if isinstance(img, str) else img for img in imgs]
        for idx, img in enumerate(imgs):
            if img is None:
                raise FileNotFoundError(f"Image not found at list index {idx}")

    else:
        raise TypeError("imgs must be a file path, numpy array, or list/tuple of images")

    processed_images = []
    for img in imgs:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Each image must have shape HxWx3")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        processed_images.append(img)

    batch = np.stack(processed_images, axis=0).astype(np.float32)
    batch = torch.from_numpy(batch)
    batch.div_(255).sub_(0.5).div_(0.5)
    return batch


@torch.no_grad()
def inference_model(model, imgs):
    imgs = prepare_images(imgs)
    if next(model.parameters()).is_cuda:
        imgs = imgs.to(next(model.parameters()).device)

    output = model(imgs)
    return {k: v.cpu().numpy() for k, v in output.items()}


def interpret_swinface_output(face_output):
    """
    Interpret SwinFace model output into human-readable format.
    For binary attributes (2 values): convert logits to probabilities.
    For continuous values: output as is.
    Skip Recognition (embedding vector).
    """
    interpreted = {}
    
    # Attribute labels for binary classifications
    binary_attrs = {
        'Attractive': ['Not Attractive', 'Attractive'],
        'Blurry': ['Sharp', 'Blurry'],
        'Chubby': ['Not Chubby', 'Chubby'],
        'Heavy Makeup': ['No Heavy Makeup', 'Heavy Makeup'],
        'Gender': ['Male', 'Female'],
        'Oval Face': ['Not Oval', 'Oval Face'],
        'Pale Skin': ['Not Pale', 'Pale Skin'],
        'Smiling': ['Not Smiling', 'Smiling'],
        'Young': ['Old', 'Young'],
        'Bald': ['Not Bald', 'Bald'],
        'Bangs': ['No Bangs', 'Bangs'],
        'Black Hair': ['Not Black', 'Black Hair'],
        'Blond Hair': ['Not Blond', 'Blond Hair'],
        'Brown Hair': ['Not Brown', 'Brown Hair'],
        'Gray Hair': ['Not Gray', 'Gray Hair'],
        'Receding Hairline': ['Normal', 'Receding Hairline'],
        'Straight Hair': ['Not Straight', 'Straight Hair'],
        'Wavy Hair': ['Not Wavy', 'Wavy Hair'],
        'Wearing Hat': ['No Hat', 'Wearing Hat'],
        'Arched Eyebrows': ['Not Arched', 'Arched Eyebrows'],
        'Bags Under Eyes': ['No Bags', 'Bags Under Eyes'],
        'Bushy Eyebrows': ['Not Bushy', 'Bushy Eyebrows'],
        'Eyeglasses': ['No Glasses', 'Eyeglasses'],
        'Narrow Eyes': ['Normal Eyes', 'Narrow Eyes'],
        'Big Nose': ['Normal Nose', 'Big Nose'],
        'Pointy Nose': ['Not Pointy', 'Pointy Nose'],
        'High Cheekbones': ['Low Cheekbones', 'High Cheekbones'],
        'Rosy Cheeks': ['No Rosy', 'Rosy Cheeks'],
        'Wearing Earrings': ['No Earrings', 'Wearing Earrings'],
        'Sideburns': ['No Sideburns', 'Sideburns'],
        "Five O'Clock Shadow": ['No Shadow', "Five O'Clock Shadow"],
        'Big Lips': ['Normal Lips', 'Big Lips'],
        'Mouth Slightly Open': ['Closed', 'Mouth Slightly Open'],
        'Mustache': ['No Mustache', 'Mustache'],
        'Wearing Lipstick': ['No Lipstick', 'Wearing Lipstick'],
        'No Beard': ['Beard', 'No Beard'],
        'Double Chin': ['No Double Chin', 'Double Chin'],
        'Goatee': ['No Goatee', 'Goatee'],
        'Wearing Necklace': ['No Necklace', 'Wearing Necklace'],
        'Wearing Necktie': ['No Necktie', 'Wearing Necktie'],
    }
    
    for attr, values in face_output.items():
        if attr == 'Recognition':
            # Skip embedding vector
            continue
        
        if attr == 'Age':
            # Continuous value
            interpreted[attr] = float(values)
        elif attr == 'Expression':
            # Multi-class (7 classes)
            expression_labels = ['Neutral', 'Happiness', 'Sadness', 'Anger', 'Surprise', 'Disgust', 'Fear']
            logits = values
            pred_idx = int(np.argmax(logits))
            interpreted[attr] = expression_labels[pred_idx]
        elif len(values) == 2 and attr in binary_attrs:
            # Binary classification with logits
            logits = values
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))  # for numerical stability
            probs = exp_logits / np.sum(exp_logits)
            pred_idx = int(np.argmax(probs))
            labels = binary_attrs[attr]
            interpreted[attr] = {
                'prediction': labels[pred_idx],
                'confidence': float(probs[pred_idx])
            }
    
    return interpreted


def save_face_results(interpreted_output, filename):
    """Save interpreted face results to a text file."""
    with open(filename, 'w') as f:
        for attr, result in interpreted_output.items():
            if isinstance(result, dict):
                # Binary attribute with confidence
                f.write(f"{attr}: {result['prediction']} (confidence: {result['confidence']:.4f})\n")
            elif isinstance(result, float):
                # Continuous value (Age)
                f.write(f"{attr}: {result:.2f}\n")
            else:
                # Other (Expression, etc.)
                f.write(f"{attr}: {result}\n")


class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512

if __name__ == "__main__":

    cfg = SwinFaceCfg()
    parser = argparse.ArgumentParser(description='PyTorch ArcFace inference')
    parser.add_argument('--weight', type=str, default='<your path>/checkpoint_step_79999_gpu_0.pt')
    parser.add_argument('--imgs', type=str, default="<your path>/test.jpg",
                        help='Single image path or comma-separated list of image paths')
    args = parser.parse_args()

    img_input = [path.strip() for path in args.imgs.split(',')] if ',' in args.imgs else args.imgs
    outputs = inference(cfg, args.weight, img_input)
    print('Batch output shapes:')
    for k, v in outputs.items():
        print(f'  {k}: {v.shape}')
