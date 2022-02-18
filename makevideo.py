'''
make images to video
'''

import argparse
import glob
import os
import cv2 as cv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='./output_xyj', type=str)
    parser.add_argument("--output", default='./output_xyj.avi', type=str)
    ARGS = parser.parse_args()

    len = len(glob.glob(os.path.join(ARGS.input, '*.jpg')))
    img_name_list = [ARGS.input + '/' + str(i+1) + '_out.jpg' for i in range(len)]
    img = cv.imread(img_name_list[0])
    size = img.shape
    video = cv.VideoWriter(ARGS.output, cv.VideoWriter_fourcc(*'MJPG'), 25, (size[1], size[0]))
    for i, image_path in enumerate(img_name_list):
        res = cv.imread(image_path)
        video.write(res)
        print('Already make ', i, '/', len, ' images')
