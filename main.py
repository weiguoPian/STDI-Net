from loader.dataLoader import bikeDataNYC
from Glob.glob import p_parse

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


from model.Net import STDI_Net
import random


def adjust_learning_rate(optimizer, epoch, lr):
    decay = epoch//50
    lr = lr*(0.1**decay)
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr

def trainSTDINet(args):
    train_dataset = bikeDataNYC(args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True)
    val_dataset = bikeDataNYC(args, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=512, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)
    
    seed = 0
    torch.manual_seed(seed)

    model = STDI_Net(args)
    

    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-5)

    if args.cuda:
        torch.cuda.manual_seed(seed)
        model = model.cuda()
    
    train_loss_list = []
    BEST_VAL = None
    for epoch in range(args.max_epoches):
        train_loss = 0.0
        step = 0
        for _, pack in enumerate(train_loader):
            step += 1
            pack = list(map(lambda item: item.numpy(), pack))
            seqs = torch.Tensor(pack[0:-2])
            labels = torch.Tensor(pack[-2])
            hour_feature = torch.Tensor(pack[-1])
            if args.cuda:
                seqs = seqs.cuda()
                labels = labels.cuda()
                hour_feature = hour_feature.cuda()
            out = model.forward(seqs, hour_feature)
            labels = labels.view(labels.size(0), -1)
            loss_func = torch.nn.MSELoss()
            loss = loss_func(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= step
        train_loss = train_loss**0.5
        train_loss_list.append(train_loss)
        print('epoch:{} train_loss:{}'.format(epoch, train_loss))
        with torch.no_grad():
            val_res_list = []
            val_labels_list = []
            for val_, val_pack in enumerate(val_loader):
                val_pack = list(map(lambda item: item.numpy(), val_pack))
                val_seqs = torch.Tensor(val_pack[0:-2])
                val_labels = torch.Tensor(val_pack[-2])
                val_hour_feature = torch.Tensor(val_pack[-1])

                if args.cuda:
                    val_seqs = val_seqs.cuda()
                    val_labels = val_labels.cuda()
                    val_hour_feature = val_hour_feature.cuda()
                
                val_out = model.forward(val_seqs, val_hour_feature)
                val_labels = val_labels.view(val_labels.size(0), -1)

                val_res_list.append(val_out)
                val_labels_list.append(val_labels)
            val_res_list = torch.stack(val_res_list)
            val_res_list = val_res_list.reshape(-1, val_res_list.size(-1)).detach().cpu().numpy()
            val_labels_list = torch.stack(val_labels_list)
            val_labels_list = val_labels_list.reshape(-1, val_labels_list.size(-1)).detach().cpu().numpy()

            val_RMSE = np.mean((val_res_list - val_labels_list)**2)**0.5
            print("val RMSE: {}".format(val_RMSE))

            if BEST_VAL is None or BEST_VAL > val_RMSE:
                BEST_VAL = val_RMSE
                snap_shot = {'state_dict': model.state_dict()}
                torch.save(snap_shot, './save/snap/saved_snap.pth.tar')
                print("saved epoch{}".format(epoch))

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig('./save/img/train_loss.png')
        plt.close()

def test(args):
    test_dataset = bikeDataNYC(args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=512, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False)
    
    model = STDI_Net(args)
    model.load_state_dict(torch.load(
        './save/snap/saved_snap.pth.tar', map_location='cpu')['state_dict'])

    if args.cuda:
        model = model.cuda()

    with torch.no_grad():
        test_res_list = []
        test_labels_list = []
        for test_, test_pack in enumerate(test_loader):
            test_pack = list(map(lambda item: item.numpy(), test_pack))
            test_seqs = torch.Tensor(test_pack[0:-2])
            test_labels = torch.Tensor(test_pack[-2])
            test_hour_feature = torch.Tensor(test_pack[-1])

            if args.cuda:
                test_seqs = test_seqs.cuda()
                test_labels = test_labels.cuda()
                test_hour_feature = test_hour_feature.cuda()
            
            test_out = model.forward(test_seqs, test_hour_feature)
            test_labels = test_labels.view(test_labels.size(0), -1)

            test_res_list.append(test_out)
            test_labels_list.append(test_labels)

        test_res_list = torch.stack(test_res_list)
        test_res_list = test_res_list.reshape(-1, test_res_list.size(-1)).detach().cpu().numpy()
        test_labels_list = torch.stack(test_labels_list)
        test_labels_list = test_labels_list.reshape(-1, test_labels_list.size(-1)).detach().cpu().numpy()

        rmse = RMSE(y_pred=test_res_list, y_true=test_labels_list)
        mae = MAE(y_pred=test_res_list, y_true=test_labels_list)
        print("RMSE: {}, MAE: {}".format(rmse, mae))

def RMSE(y_pred, y_true):
    return np.mean((y_true - y_pred)**2)**0.5

def MAE(y_pred, y_true):
    return np.mean(abs(y_pred - y_true))

def setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = p_parse()
    setSeed(args.seed)
    trainSTDINet(args)
    test(args)
