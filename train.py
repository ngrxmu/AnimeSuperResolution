'''
train
'''

from data import MyDataSet
from model import *
from tqdm import tqdm
import os
import argparse
from torch.utils.data import DataLoader
import copy
from torch.autograd import Variable


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_psnr(sr, hr):
    sr_ = sr.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255)
    hr_ = hr.squeeze(0).mul_(255).add_(0.5).clamp_(0, 255)
    sr_ = 16.0 + (65.738 * sr_[0] + 129.057 * sr_[1] + 25.046 * sr_[2])/256.0
    hr_ = 16.0 + (65.738 * hr_[0] + 129.057 * hr_[1] + 25.046 * hr_[2])/256.0
    psnr = 10.0 * torch.log10((255.0**2)/((sr_-hr_)**2).mean())
    return psnr

def train(scale, root, gpu, load_from_dict, dict_path):
    logfile = open('./x4_log.txt', 'w', buffering = 1)
    dataset = MyDataSet(root, scale=scale)
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    Model = AAN(scale).cuda(gpu)
    if load_from_dict:
        state_dict = torch.load(dict_path)
        Model.load_state_dict(state_dict)
        print('load dict success!')
        logfile.write('load dict success!' + '\n')
    optimizer = torch.optim.Adam(Model.parameters(), lr=5e-4)
    criterion = L1_Charbonnier_loss().cuda(gpu)

    best_weight = copy.deepcopy(Model.state_dict())
    best_epoch = 0
    best_psnr = 0

    for epoch in range(300):
        # train
        train_loss = 0
        Model.train()
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            lr = Variable(data[0]).cuda(gpu)
            hr = Variable(data[1]).cuda(gpu)
            sr = Model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('train epoch ', epoch, ' total loss ', train_loss)
        logfile.write('train epoch ' + str(epoch) + ' total loss ' + str(train_loss) + '\n')
        if (epoch+1) % 5 == 0:
            torch.save(Model.state_dict(), './Models/x4_epoch{}.pt'.format(epoch+1))
        # val
        Model.eval()
        val_loss = 0
        val_psnr = 0
        for item in val_dataloader:
            lr = item[0].cuda(gpu)
            hr = item[1].cuda(gpu)
            with torch.no_grad():
                sr = Model(lr)
                loss = criterion(sr, hr)
                val_loss += loss.item()
            val_psnr += calculate_psnr(sr, hr)
        avg_psnr = val_psnr/len(val_dataloader)
        print('val epoch ', epoch, ' total loss ', val_loss, ' avg_psnr ', avg_psnr)
        logfile.write('val epoch ' + str(epoch) + ' total loss ' + str(val_loss) + ' avg_psnr ' + str(avg_psnr.item()) + '\n')
        # save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_weight = copy.deepcopy(Model.state_dict())
            print('Save best model')
            logfile.write('Save best model' + '\n')
            torch.save(best_weight, './Models/x4_best.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--root", default="./h5data_x4.h5", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--load_from_dict", default=1, type=int)
    parser.add_argument("--dict_path", default="./Models/x4_pretrained_model.pt", type=str)
    ARGS = parser.parse_args()
    
    mkdir('./Models')
    train(ARGS.scale, ARGS.root, ARGS.gpu, ARGS.load_from_dict, ARGS.dict_path)