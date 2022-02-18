'''
show the result
'''

import argparse
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=0, type=int)
    parser.add_argument("--lr", default='', type=str)
    parser.add_argument("--sr", default='', type=str)
    ARGS = parser.parse_args()

    lr = cv.imread(ARGS.lr)
    sr = cv.imread(ARGS.sr)
    lr_size = lr.shape
    sr_size = sr.shape
    lrlr = cv.resize(lr, (sr_size[1], sr_size[0]))
    
    if not ARGS.model:
        res = np.concatenate((lrlr, sr), axis=1)
        cv.imwrite('res_static.jpg', res)
    else:
        res = lrlr
        video = cv.VideoWriter('res_dynamic.avi', cv.VideoWriter_fourcc(*'MJPG'), 50, (sr_size[1], sr_size[0]))
        for i in range(sr_size[1]):
            res[:, i, :] = sr[:, i, :]
            video.write(res)
