import argparse
from pathlib import Path

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
    Interpret output from the analysis model trained by ``train.py``.

    The model returns batched heads with shapes ``(B, 7)``, ``(B, 3)`` and
    ``(B, 10)`` for expression, pose and landmarks respectively.  A single
    input returns one dictionary for backwards compatibility; a batch returns
    one dictionary per input image.
    """
    expression_labels = [
        'Neutral', 'Happiness', 'Sadness', 'Anger', 'Surprise', 'Disgust', 'Fear'
    ]
    expected_shapes = {'Expression': 7, 'Pose': 3, 'Landmarks': 10}
    heads = {}
    batch_size = None

    for name, size in expected_shapes.items():
        if name not in face_output:
            raise KeyError(f"Missing model output head: {name}")
        values = np.asarray(face_output[name])
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != size:
            raise ValueError(
                f"{name} output must have shape (B, {size}), got {values.shape}"
            )
        if batch_size is None:
            batch_size = values.shape[0]
        elif values.shape[0] != batch_size:
            raise ValueError("Model output heads have different batch sizes")
        heads[name] = values

    interpreted = []
    for index in range(batch_size):
        expression_logits = heads['Expression'][index]
        interpreted.append({
            'Expression': {
                'prediction': expression_labels[int(np.argmax(expression_logits))],
                'confidence': float(np.max(
                    np.exp(expression_logits - np.max(expression_logits)) /
                    np.sum(np.exp(expression_logits - np.max(expression_logits)))
                )),
            },
            'Pose': {
                'pitch': float(heads['Pose'][index, 0]),
                'yaw': float(heads['Pose'][index, 1]),
                'roll': float(heads['Pose'][index, 2]),
            },
            'Landmarks': heads['Landmarks'][index].astype(float).tolist(),
        })
    return interpreted[0] if batch_size == 1 else interpreted

def euler_to_rotation_matrix(pitch, yaw, roll):
    """pitch, yaw, roll in radians -> 3x3 rotation matrix."""
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
 
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])
 
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])
 
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])
 
    return Rz @ Ry @ Rx


def draw_swinface_results(image, interpreted_output, output_path=None, vector_length=None):
    """Draw normalized landmarks and pose axes, then save the image.

    Pose values are assumed to be radians and are interpreted as pitch, yaw,
    and roll in that order.  The image is expected to be a BGR OpenCV image.
    """
    if isinstance(interpreted_output, list):
        if len(interpreted_output) != 1:
            raise ValueError("Pass one interpreted result per image")
        interpreted_output = interpreted_output[0]
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape HxWx3")

    annotated = image.copy()
    height, width = annotated.shape[:2]
    points = np.asarray(interpreted_output['Landmarks'], dtype=np.float32).reshape(5, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    points = np.rint(points).astype(int)

    landmark_names = ['left eye', 'right eye', 'nose', 'left mouth', 'right mouth']
    for name, (x, y) in zip(landmark_names, points):
        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(annotated, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)

    center = tuple(np.mean(points, axis=0).astype(int))
    if vector_length is None:
        vector_length = max(30, int(min(width, height) * 0.22))
    pose = interpreted_output['Pose']
    pitch, yaw, roll = pose['pitch'], pose['yaw'], pose['roll']
    center_array = np.asarray(center)

    # Project the rotated 3D axes onto the image plane.
    R = euler_to_rotation_matrix(pitch, yaw, roll)
 
    # Unit vectors along each axis in 3D
    axes_3d = np.array([
        [1, 0, 0],  # X - red
        [0, 1, 0],  # Y - green
        [0, 0, 1],  # Z - blue
    ], dtype=np.float64)
 
    rotated = axes_3d @ R.T  # rotate each axis vector
 
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: red, green, blue
    labels = ["X (pitch axis)", "Y (yaw axis)", "Z (roll axis)"]
 
    # Sort by depth (farthest first) so closer axes are drawn on top
    order = np.argsort(rotated[:, 2])
 
    for i in order:
        vec, color, label = rotated[i], colors[i], labels[i]
 
        # Perspective scale factor: >1 when the tip is toward the viewer
        # (positive z), <1 when it points away (negative z).
        z = vec[2]
        perspective_scale = 300 / (300 - z * 20)
 
        end_x = int(center_array[0] + vec[0] * 100 * perspective_scale)
        end_y = int(center_array[1] - vec[1] * 100 * perspective_scale)  # flip y for image coords
 
        # Line thickness also scales with depth for a stronger 3D cue
        thickness = int(np.clip(3 * perspective_scale, 1, 3))
 
        cv2.arrowedLine(annotated, center_array, (end_x, end_y), color, thickness, tipLength=0.2)
        cv2.putText(annotated, label, (end_x + 5, end_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if output_path is not None and not cv2.imwrite(output_path, annotated):
        raise OSError(f"Could not write annotated image: {output_path}")
    return annotated


def save_face_results(interpreted_output, filename):
    """Save interpreted face results to a text file."""
    with open(filename, 'w') as f:
        for attr, result in interpreted_output.items():
            if attr == 'Expression':
                f.write(f"{attr}: {result['prediction']} (confidence: {result['confidence']:.4f})\n")
            elif attr == 'Pose':
                f.write(
                    f"{attr}: pitch={result['pitch']:.4f}, "
                    f"yaw={result['yaw']:.4f}, roll={result['roll']:.4f}\n"
                )
            elif attr == 'Landmarks':
                points = result
                f.write(f"{attr}: ")
                f.write(', '.join(
                    f"({points[index]:.4f}, {points[index + 1]:.4f})"
                    for index in range(0, len(points), 2)
                ))
                f.write('\n')
            else:
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
    parser.add_argument('--folder', type=str,
                        help='Folder of images to display sequentially')
    parser.add_argument('--output', type=str, default='swinface_annotated.jpg',
                        help='Path for the image with landmarks and pose axes')
    args = parser.parse_args()

    if args.folder:
        image_paths = sorted(
            path for path in Path(args.folder).iterdir()
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        )
        if not image_paths:
            raise FileNotFoundError(f'No supported images found in folder: {args.folder}')

        model = load_swinface_model(cfg, args.weight)
        window_name = 'SwinFace inference (press q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            for image_path in image_paths:
                input_image = cv2.imread(str(image_path))
                if input_image is None:
                    print(f'Skipping unreadable image: {image_path}')
                    continue
                output = inference_model(model, input_image)
                interpreted = interpret_swinface_output(output)
                annotated = draw_swinface_results(input_image, interpreted)
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
    else:
        img_input = [path.strip() for path in args.imgs.split(',')] if ',' in args.imgs else args.imgs
        outputs = inference(cfg, args.weight, img_input)
        print('Batch output shapes:')
        for k, v in outputs.items():
            print(f'  {k}: {v.shape}')
        interpreted = interpret_swinface_output(outputs)
        if isinstance(img_input, str):
            input_image = cv2.imread(img_input)
            draw_swinface_results(input_image, interpreted, args.output)
            print(f'Annotated image saved to {args.output}')
        else:
            print('Use one image path with --output to save an annotated image.')
