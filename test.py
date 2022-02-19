'''
test
'''

import os
from model import *
import glob
import cv2 as cv
from torchvision import transforms
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def test(scale=4, model='./Models/x4_best.pt', input='./input', output='./output', gpu=0):
    mkdir(output)
    Model = AAN(scale).cuda(gpu)
    state_dict = torch.load(model)
    Model.load_state_dict(state_dict)
    for image_path in glob.glob(os.path.join(input, '*.jpg')):
        print('Test ' + image_path)
        image = cv.imread(image_path)
        image = transforms.ToTensor()(image).unsqueeze(0).cuda(gpu)
        with torch.no_grad():
            result = Model(image).clamp(0.0, 1.0)
        result = result.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        output_path = image_path.replace(input, output)
        cv.imwrite(output_path, result)
        print('Save ' + output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--model", default="./Models/x4_best.pt", type=str)
    parser.add_argument("--input", default="./input", type=str)
    parser.add_argument("--output", default="./output", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    ARGS = parser.parse_args()

    test(ARGS.scale, ARGS.model, ARGS.input, ARGS.output, ARGS.gpu)
