#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:07:10 2019

@author: xingyu
"""

from model.LPRNET import LPRNet, CHARS
from model.mini_STN import STNet
from data.load_data import LPRDataLoader, collate_fn
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torchvision
import matplotlib.pyplot as plt
import math
import time

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 
    inp = inp[:,:,::-1]
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        dataset = LPRDataLoader([args.img_dirs], args.img_size)   
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn) 
        #imgs, labels, lengths = next(iter(dataloader))
        for imgs, labels, lengths in dataloader:
            input_tensor = imgs.cpu()
            transformed_input_tensor = STN(imgs.to(device)).cpu()
        
            in_grid = convert_image(torchvision.utils.make_grid(input_tensor))
            out_grid = convert_image(torchvision.utils.make_grid(transformed_input_tensor))
        
            # Plot the results side-by-side
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')
            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')
            plt.show()

def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        if pre_c != (len(CHARS) - 1):
            no_repeat_blank_label.append(pre_c)
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
        
    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels, pred_labels

def eval(lprnet, STN, dataloader, dataset, device):
    
    lprnet = lprnet.to(device)
    STN = STN.to(device)
    TP = 0
    for imgs, labels, lengths in dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
        imgs, labels = imgs.to(device), labels.to(device)
        transfer = STN(imgs)
        #transfer = imgs
        logits = lprnet(transfer) # torch.Size([batch_size, CHARS length, output length ])
    
        preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
        _, pred_labels = decode(preds, CHARS)  # list of predict output

        start = 0
        for i, length in enumerate(lengths):
            label = labels[start:start+length]
            start += length
            if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                TP += 1
                
            else:
                if i == imgs.shape[0]-1:
                    print('pred: ', end='')
                    for j in range(len(pred_labels[i])):
                        print(CHARS[pred_labels[i][j]], end='')
                    #print('pred labels:', pred_labels[i])
                    print('\nTrue: ', end='')
                    for j in range(len(label.cpu().numpy())):
                        print(CHARS[label.cpu().numpy().astype('int').tolist()[j]], end='')
                    print('\n------------------------')
           
    ACC = TP / len(dataset) 
    
    return ACC
    

if __name__ == '__main__':
    p=2.5#
    r=math.pow( 1.1, p )
    resw = int(r*94)
    resh = int(r*24)
    print('p:',p)
    print('r:',r)
    dir='/content/gdrive/MyDrive/個別研究/車牌/LPRNet3/test_lr_mini_saving_ckpt(1.1_2.5)/test_lr_mini_saving_ckpt(1.1_2.5-2)'#
    parser = argparse.ArgumentParser(description='LPR Evaluation')
    parser.add_argument('--img_size', default=(resw, resh), help='the image size')
    parser.add_argument('--img_dirs', default="data/test2", help='the images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--batch_size', default=1, help='batch size.')
    args = parser.parse_args()
    device = torch.device("cuda:0")#
    # device = torch.device("cpu")#
    print(device)
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate , r=r)
    lprnet.to(device)
    
    # lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
    checkpoint = torch.load(dir+'/lprnet_Iter_150000_model.ckpt')#
    lprnet.load_state_dict(checkpoint['net_state_dict'])
    lprnet.eval() 
    print("LPRNet loaded")
    
    torch.save(lprnet.state_dict(), 'weights/Final2_LPRNet_model.pth')
    
    STN = STNet( r=r , resw = resw , resh = resh)
    STN.to(device)
    # STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    checkpoint = torch.load(dir+'/stn_Iter_150000_model.ckpt')#
    STN.load_state_dict(checkpoint['net_state_dict'])
    STN.eval()
    print("STN loaded")
    
    torch.save(STN.state_dict(), 'weights/Final2_STN_model.pth')
    
    dataset = LPRDataLoader([args.img_dirs], args.img_size)   
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn) 
    print('dataset loaded with length : {}'.format(len(dataset)))
    start_time = time.time()
    t=1
    for i in range(t):
        ACC = eval(lprnet, STN, dataloader, dataset, device)
    time_elapsed = time.time() - start_time  
    print('the accuracy is {:.2f} %'.format(ACC*100))
    print('one Evaluation complete in {:f}s'.format( time_elapsed/(t*25) ))
    
    visualize_stn()

