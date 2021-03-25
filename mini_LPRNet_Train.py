#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:07:10 2019

@author: xingyu
"""
import os
from model.LPRNET import LPRNet, CHARS
from model.mini_STN import STNet
from data.load_data import LPRDataLoader, collate_fn
from Evaluation import eval, decode
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import numpy as np
import argparse
import time
from torchsummary import summary  
import math

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

if __name__ == '__main__':
    ''' batch=24*2
    mini=math.pow( 4, 0.5 )
    p=0.5
    z2 = math.pow( 2, 0.5 )
    r = math.pow( z2 , p )
    c = z2/r'''
    mini=1
    batch=24
    p=3
    r=math.pow( 1.1, p )
    resw = int(r*94/mini)
    resh = int(r*24/mini)
    torch.backends.cudnn.enabled = True
    parser = argparse.ArgumentParser(description='LPR Training')
    parser.add_argument('--img_size', default=(resw, resh), help='the image size')
    parser.add_argument('--img_dirs_train', default="./data/train/", help='the training images path')
    parser.add_argument('--img_dirs_val', default="./data/test/", help='the validation images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')#0.5
    parser.add_argument('--epoch', type=int, default=1600, help='number of epoches for training')#800
    parser.add_argument('--batch_size', default=batch, help='batch size')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ',device)
    print("len CHARS: ",len(CHARS))
    print('pow:',p)
    print('r:',r )
    print('mini:',mini)
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate, r=r)#mini/c
    lprnet.to(device)
    summary(lprnet, (3,resh,resw))
    #lprnet_load=torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage.cuda)
    #lprnet.load_state_dict(lprnet_load['net_state_dict'])
    #lprnet.load_state_dict(torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
    print("LPRNet loaded")
    print('LPRNet params: ',sum(p.numel() for p in lprnet.parameters()))
    STN = STNet( r=r , resw = resw , resh = resh)
    STN.to(device)
    #summary(STN, (3,24,94))
    summary(STN, (3,resh,resw))
    #STN_load=torch.load('weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage.cuda)
    #STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    print("STN loaded")
    print('STN params: ',sum(p.numel() for p in STN.parameters()))
    # raise NameError('stop')
    
    dataset = {'train': LPRDataLoader([args.img_dirs_train], args.img_size, aug_transform=True),
               'val': LPRDataLoader([args.img_dirs_val], args.img_size, aug_transform=False)} ###shuffle
    dataloader = {'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn),
                  'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)}
    print('training dataset loaded with length : {}'.format(len(dataset['train'])))
    print('validation dataset loaded with length : {}'.format(len(dataset['val'])))
    
    # define optimizer & loss
    #optimizer1 = torch.optim.Adam([{'params': lprnet.parameters()}])
    #optimizer = torch.optim.Adam([{'params': STN.parameters()},
    #                               {'params': lprnet.parameters()}])
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    ## save logging and weights
    train_logging_file = 'test_lr_mini_train_logging.txt'
    #raise NameError('stop')
    with open(train_logging_file, 'a') as f:
        f.write("Start--the r is:"+str(r)+'pow is:'+str(p)+'\n')
    f.close()
    validation_logging_file = 'test_lr_mini_validation_logging.txt'
    with open(validation_logging_file, 'a') as f:
        f.write("Start--the r is:"+str(r)+'pow is:'+str(p)+'\n')
    f.close()
    save_dir = 'test_lr_mini_saving_ckpt'
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    
    start_time = time.time()
    total_iters = 0
    #total_iters = lprnet_load['iters']
    best_acc = 0.0
    T_length = 16 # args.lpr_max_len
    learning_rate = 0.001
    optimizer = torch.optim.Adam([{'params': STN.parameters()},
                                   {'params': lprnet.parameters()}], lr=learning_rate)
    print('training kicked off..')
    print('-' * 10) 
    for epoch in range(args.epoch):
        # train model
        lprnet.train()
        STN.train()
        since = time.time()
        for imgs, labels, lengths in dataloader['train']:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
            imgs, labels = imgs.to(device), labels.type(torch.IntTensor).to(device)
            if( (total_iters % 3000 == 0) and (total_iters > 0)):#2000
                print('**** learning rate decay')
                learning_rate = learning_rate*0.915
                optimizer = torch.optim.Adam([{'params': STN.parameters()},
                                        {'params': lprnet.parameters()}], lr=learning_rate)
                                    
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                transfer = STN(imgs)
                logits = lprnet(transfer)
                #logits = lprnet(transfer)  # torch.Size([batch_size, CHARS length, output length ])
                log_probs = logits.permute(2, 0, 1) # for ctc loss: length of output x batch x length of chars
                log_probs = log_probs.log_softmax(2).requires_grad_()      
                input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths) # convert to tuple with length as batch_size 
                loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
                
                loss.backward()
                optimizer.step()
                
                total_iters += 1
                # print train information
                if total_iters % 100 == 0:
                    # current training accuracy             
                    preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
                    _, pred_labels = decode(preds, CHARS)  # list of predict output
                    total = preds.shape[0]
                    start = 0
                    TP = 0
                    for i, length in enumerate(lengths):
                        #if total_iters % 200 == 0:
                           # print('Label: ',labels[start:start+length])
                            #print('pred_labels: ',pred_labels[i])
                        label = labels[start:start+length]
                        start += length
                        if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                            TP += 1
                    
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in  optimizer.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr))
                    with open(train_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr)+'\n')
                    f.close()
                    
                    # save model
            if total_iters % 500 == 0:

                torch.save({
                    'iters': total_iters,
                    'net_state_dict': lprnet.state_dict()},
                    os.path.join(save_dir, 'lprnet_Iter_%06d_model.ckpt' % total_iters))
                
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': STN.state_dict()},
                    os.path.join(save_dir, 'stn_Iter_%06d_model.ckpt' % total_iters))
                    
            # evaluate accuracy
            if total_iters % 500 == 0:
                
                lprnet.eval()
                STN.eval()
                
                ACC = eval(lprnet, STN, dataloader['val'], dataset['val'], device)
                            
                if best_acc <= ACC:
                    best_acc = ACC
                    best_iters = total_iters
                
                print("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, ACC))
                print('--Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters))
                with open(validation_logging_file, 'a') as f:
                    f.write("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, ACC)+'\n')
                    f.write('--Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters)+'\n')
                f.close()
                
                lprnet.train()
                STN.train()
                                
    time_elapsed = time.time() - start_time  
    print('Finally Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    with open(validation_logging_file, 'a') as f:
        f.write("End--the r is:"+str(r)+'--Finally Best Accuracy: {:.4f} in iters: {}'.format(best_acc, best_iters)+'\n')
    f.close()
    with open(train_logging_file, 'a') as f:
        f.write("End--the r is:"+str(r)+'--Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+'\n')
    f.close()