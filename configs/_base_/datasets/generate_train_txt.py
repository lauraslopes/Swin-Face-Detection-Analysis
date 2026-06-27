import scipy.io as sio
import os

mat = sio.loadmat('data/WIDERFace/wider_face_split/wider_face_train.mat')
os.makedirs('data/WIDERFace', exist_ok=True)

with open('data/WIDERFace/train.txt', 'w') as f:
    for item in mat['file_list'][0]:
        for fname in item[0]:
            f.write(fname[0][0] + '\n')
