'''
clip video to images
'''

import os
import argparse
import cv2 as cv


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='./xyj.mp4', type=str)
    parser.add_argument("--output", default="./xyj", type=str)
    ARGS = parser.parse_args()

    mkdir(ARGS.output)
    video = cv.VideoCapture(ARGS.input)
    ret, image = video.read()
    count = 0
    while ret:
        count += 1
        cv.imwrite(os.path.join(ARGS.output, str(count) + '.jpg'), image)
        ret, image = video.read()
        print('Already clip ', count, ' images')
