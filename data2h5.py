'''
clip image and make jpg to h5
'''

import cv2 as cv
import os
import h5py
import PIL
from PIL import Image
from torchvision import transforms


def clip_image(root):
    image_names = os.listdir(root)
    image_paths = [os.path.join(root, name) for name in image_names]
    count = 0
    for path in image_paths:
        image = cv.imread(path)
        h, w, _ = image.shape
        for i in range(0, h, 200):
            for j in range(0, w, 200):
                if (i + 480 >= h) or (j + 480 >= w):
                    continue
                img = image[i:i+480, j:j+480]
                clipdata_path = './clipdata'
                save_path = os.path.join(clipdata_path, str(count) + '.jpg')
                cv.imwrite(save_path, img)
                print('Save ' + save_path)
                count += 1

def create_h5_file(root, scale):
    h5_file = h5py.File('./h5data_x4.h5', 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    image_names = os.listdir(root)
    image_paths = [os.path.join(root, name) for name in image_names]
    count = 0
    for image_path in image_paths:
        hr = Image.open(image_path).convert('RGB')
        lr = hr.resize((hr.width // scale, hr.height // scale), resample=Image.BICUBIC)
        hr = transforms.ToTensor()(hr)
        lr = transforms.ToTensor()(lr)
        lr_group.create_dataset(str(count), data=lr)
        hr_group.create_dataset(str(count), data=hr)
        print('convert' + image_path + ' to h5')
        count += 1
    h5_file.close()

if __name__=='__main__':
    clip_image('./data')
    create_h5_file('./clipdata', 4)