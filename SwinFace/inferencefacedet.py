import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import torch

from model import build_model

@torch.no_grad()
def inference(model, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    
    output = model(img)

    return output

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

def facedetector(img, model, conf=0.4):
    results = model(img, conf=conf)[0]
    return results.boxes.xyxy.cpu().numpy()
    
def crop_box(img, box):
    x1, y1, x2, y2 = map(int, box)

    # Safety clamp
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    return img[y1:y2, x1:x2]


if __name__ == "__main__":

    cfg = SwinFaceCfg()
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight', type=str, default='./checkpoint_step_79999_gpu_0.pt')
    parser.add_argument('--img', type=str, default="./screenshot.png")
    args = parser.parse_args()
    
    img = cv2.imread(args.img)
    facemodel = YOLO("yolov8n-face.pt")

    boxes = facedetector(img, facemodel, conf=0.4)
    
    attrmodel = build_model(cfg)
    dict_checkpoint = torch.load(args.weight)
    attrmodel.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    attrmodel.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    attrmodel.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    attrmodel.om.load_state_dict(dict_checkpoint["state_dict_om"])

    attrmodel.eval()
    
    for box in boxes:
        output = inference(attrmodel, crop_box(img, box))
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, f"{output['Age'][0]}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )
    
    cv2.imwrite("out.jpg", img)
    
