import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def filter_annotations(ann_dir, min_size=64):
    removed_boxes = 0
    removed_files = 0
    kept_boxes = 0
    dirr = "/".join(ann_dir.split("/")[0:2])

    for root_dir, dirs, files in os.walk(ann_dir):
        for fname in files:
            if not fname.endswith('.xml'):
                continue

            xml_path = os.path.join(root_dir, fname)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            objects = root.findall('object')
            to_remove = []
            small_bboxes = []
            
            img_path = os.path.join(dirr, root.find('path').text)
            img = cv2.imread(img_path)

            for obj in objects:
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))

                w = xmax - xmin
                h = ymax - ymin

                if w < min_size or h < min_size:
                    small_bboxes.append((xmin, ymin, xmax, ymax))
                    to_remove.append(obj)
                    removed_boxes += 1
                else:
                    kept_boxes += 1
            if len(to_remove):
                for obj in to_remove:
                    root.remove(obj)
                
                # Apply Gaussian blur to small bbox areas
                for xmin, ymin, xmax, ymax in small_bboxes:
                    region = img[ymin:ymax, xmin:xmax]
                    blurred_region = cv2.GaussianBlur(region, (31, 31), 0)
                    img[ymin:ymax, xmin:xmax] = blurred_region
                
                cv2.imwrite(img_path, img)   
                print(img_path, " modificada")

            # if no faces left, remove the file entirely
            if len(root.findall('object')) == 0:
                os.remove(xml_path)
                removed_files += 1
            else:
                tree.write(xml_path)

    print(f'Done.')
    print(f'  Removed boxes : {removed_boxes}')
    print(f'  Kept boxes    : {kept_boxes}')
    print(f'  Removed files : {removed_files} (no faces left after filtering)')

# run on both train and val
filter_annotations('data/WIDERFace/WIDER_train/Annotations', min_size=100)
filter_annotations('data/WIDERFace/WIDER_val/Annotations', min_size=100)
