import os
import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector

# paths
config = 'configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'
checkpoint = 'work_dirs/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_20.pth'
img_dir = 'data/WIDERFace/WIDER_test/images'       # your test folder
out_dir = 'outputs/inferred/'  # where to save results
score_thr = 0.3                # confidence threshold

os.makedirs(out_dir, exist_ok=True)

# load model
model = init_detector(config, checkpoint, device='cuda:0')

# run inference on all images
for root, dirs, files in os.walk(img_dir):
    for fname in files:
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(root, fname)
        result = inference_detector(model, img_path)

        # save visualized result
        out_path = os.path.join(out_dir, fname)
        model.show_result(
        img_path,
        result,
        score_thr=score_thr,
        out_file=out_path
        )
        print(f'Saved: {out_path}')

print(f'\nDone! Results saved to {out_dir}')
