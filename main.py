from __future__ import print_function
import os
import argparse
from numpy.core.fromnumeric import mean
from tensorboardX.writer import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
# from data import ModelNet40
from adv_model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from WIPDataLoader import load_locomotion_loso_data
import torch.nn.functional as F
import PCM
import time 
from sklearn.metrics import confusion_matrix

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp adv_model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def discrepancy(out1, out2):
    """discrepancy loss"""
    out = torch.mean(
        torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))
    return out

def train(args, io):
    
    src_dataset ,trgt_dataset = load_locomotion_loso_data(args.dataroot,args.target_username)
    
    train_ind = np.asarray([i for i in range(len(trgt_dataset)) if i % 10 < 8]).astype(np.int)
    train_ind = np.random.choice(train_ind,size=len(src_dataset))
    val_ind = np.asarray([i for i in range(len(trgt_dataset)) if i % 10 >= 8]).astype(np.int)
    
    # dataloaders for source and target
    source_train_dataloader = DataLoader(src_dataset, num_workers=8, batch_size=args.batch_size,
                                          drop_last=True,shuffle=True)
    target_train_dataloader = DataLoader(trgt_dataset, num_workers=8, batch_size=args.batch_size,
                                          sampler=SubsetRandomSampler(train_ind), drop_last=True)
    target_test_dataloader = DataLoader(trgt_dataset, num_workers=8, batch_size=args.batch_size,
                                         sampler=SubsetRandomSampler(val_ind), drop_last=True)

    device = torch.device("cuda:" + str(args.gpu) if args.cuda else "cpu")

    model = Pct(args).to(device)
    
    # model = nn.DataParallel(model)

    remain_epoch = 50
    params_g = [{'params': v} for k, v in model.named_parameters() if (
        'c1' not in k and 'c2' not in k)]
    optimizer_g = optim.Adam(params_g, lr=args.lr, weight_decay=1e-4)
    lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=args.epochs+remain_epoch)

    optimizer_c = optim.Adam([{'params': model.c1.parameters()}, {'params': model.c2.parameters()}], lr=args.lr,
                             weight_decay=1e-4)
    lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_c, T_max=args.epochs+remain_epoch)
    
    criterion = cal_loss
    best_test_acc = 0
    
    writer = SummaryWriter(log_dir=args.exp_name)

    for epoch in range(args.epochs):
        # scheduler.step()
        lr_schedule_g.step()
        lr_schedule_c.step()
        
        c1_loss,c2_loss,dis_loss = 0.0,0.0,0.0
        
        count = 0.0
        model.train()
        source_train_pred = []
        source_train_true = []
        target_train_pred = []
        target_train_true = []
        
        
        idx = 0
        total_time = 0.0
        for s,t in zip(source_train_dataloader,target_train_dataloader):
            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            
            data,label = s
            data, label = data.to(device), label.to(device).squeeze() 
            data = data.permute(0, 2, 1)
            
            data_t, label_t = t
            data_t = data_t.permute(0, 2, 1)
            data_t ,label_t = data_t.to(device),label_t.to(device).squeeze()
            
            batch_size = data.size()[0]
            
            if args.apply_pcm:
                data, mixup_vals = PCM.mix_shapes(args.mixup_params, data, label)
                pred_s1, pred_s2 = model(data)
                loss_s1 = PCM.calc_loss(pred_s1,mixup_vals,criterion)
                loss_s2 = PCM.calc_loss(pred_s2,mixup_vals,criterion)
                loss_s = loss_s1 + loss_s2
                loss_s.backward(retain_graph=True)
                
            else:
                # classification loss: L_cls
                pred_s1, pred_s2 = model(data)
                loss_s1 = criterion(pred_s1, label)
                loss_s2 = criterion(pred_s2, label)
                loss_s = loss_s1 + loss_s2
                loss_s.backward(retain_graph=True)

            # L_dis
            pred_t1, pred_t2 = model(data_t, constant=1.0, adaptation=True)
            loss_dis = - 1 * discrepancy(pred_t1, pred_t2)
            loss_dis.backward()
            
            optimizer_c.step()
            optimizer_g.step()
            optimizer_c.zero_grad()
            optimizer_g.zero_grad()
            
            # log
            count += batch_size
            c1_loss += loss_s1.item() * batch_size
            c2_loss += loss_s2.item() * batch_size
            dis_loss += loss_dis.item() * batch_size
            
            preds_source_train = ((pred_s1 + pred_s2)/2).max(dim=1)[1]
            source_train_true.append(label.cpu().numpy())
            source_train_pred.append(preds_source_train.detach().cpu().numpy())
            
            preds_target_train = ((pred_t1 + pred_t2)/2).max(dim=1)[1]
            target_train_true.append(label_t.cpu().numpy())
            target_train_pred.append(preds_target_train.detach().cpu().numpy())
            
            idx += 1
            
        print ('train total time is',total_time)
        source_train_true = np.concatenate(source_train_true)
        source_train_pred = np.concatenate(source_train_pred)
        target_train_true = np.concatenate(target_train_true)
        target_train_pred = np.concatenate(target_train_pred)
        
        target_train_macro = metrics.classification_report(target_train_true,target_train_pred,digits=4,output_dict=True)['macro avg']['precision']
        
        outstr = 'Train %d, cls loss: %.6f,dis loss : %.6f, source train acc: %.6f, source train avg acc: %.6f, \
            target train acc: %.6f, target train avg acc: %.6f,' % (epoch,
                                                                                (c1_loss + c2_loss)*1.0 / count,
                                                                                dis_loss / count,
                                                                                metrics.accuracy_score(
                                                                                source_train_true, source_train_pred),
                                                                                metrics.balanced_accuracy_score(
                                                                                source_train_true, source_train_pred),
                                                                                metrics.accuracy_score(
                                                                                target_train_true, target_train_pred),
                                                                                metrics.balanced_accuracy_score(
                                                                                target_train_true, target_train_pred)
                                                                                )
        
        writer.add_scalars('acc/train', {'source ins acc': metrics.accuracy_score(source_train_true, source_train_pred),
                                         'source avg acc': metrics.balanced_accuracy_score(source_train_true, source_train_pred),
                                         'target ins acc': metrics.accuracy_score(target_train_true, target_train_pred),
                                         'target avg acc': metrics.balanced_accuracy_score(target_train_true, target_train_pred),
                                         'target macro acc' : target_train_macro
                                         }, epoch)
        writer.add_scalars('loss/train', {'c1 loss':c1_loss / count, 
                                          'c2 loss':c2_loss / count,
                                          'class loss': (c1_loss + c2_loss ) / count,
                                          'discrepancy loss':dis_loss / count,
                                          },epoch)
        
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_cls_loss = 0.0
        test_dis_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in target_test_dataloader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            y1,y2 = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            logits = (y1 + y2) / 2
            loss_s1 = criterion(y1, label)
            loss_s2 = criterion(y2, label)
            loss = loss_s1 + loss_s2
            dis_loss = discrepancy(pred_t1, pred_t2)
            
            preds_source_train = logits.max(dim=1)[1]
            count += batch_size
            test_cls_loss += loss.item() * batch_size
            test_dis_loss += dis_loss.item() * batch_size
            
            test_true.append(label.cpu().numpy())
            test_pred.append(preds_source_train.detach().cpu().numpy())
        print ('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc  = metrics.classification_report(test_true,test_pred,digits=4,output_dict=True)['macro avg']['precision']
        
        if avg_per_class_acc >= best_test_acc:
            best_test_acc = avg_per_class_acc
            io.cprint('save model at checkpoints/%s/models/model.t7' % args.exp_name)
            target_names = ['standing', 'walking', 'jogging','jumping','squatdown','squating','squat up','forward','backoff']
            # target_names = ['standing', 'walking']
            report = metrics.classification_report(test_true,test_pred,digits=3,target_names=target_names)
            io.cprint(report)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)

        outstr = 'Test %d, cls loss: %.6f,dis loss: %.6f, test acc: %.6f, test avg(macro) acc: %.6f , best avg(macro): %.6f' % (epoch,
                                                                            test_cls_loss*1.0/count,
                                                                            test_dis_loss / count,
                                                                            test_acc,
                                                                            avg_per_class_acc,
                                                                            best_test_acc)
        writer.add_scalars('acc/test', {'target ins acc': test_acc,
                                         'target avg(macro) acc': avg_per_class_acc,
                                         }, epoch)
        io.cprint(outstr)

def test(args, io):
    src_dataset ,trgt_dataset = load_locomotion_loso_data(args.dataroot,args.target_username)
    # train_ind = np.asarray([i for i in range(len(trgt_dataset)) if i % 10 < 8]).astype(np.int)
    # train_ind = np.random.choice(train_ind,size=len(src_dataset))
    val_ind = np.asarray([i for i in range(len(trgt_dataset)) if i % 10 >= 8]).astype(np.int)
    test_loader = DataLoader(trgt_dataset, num_workers=8, batch_size=args.batch_size,
                                         sampler=SubsetRandomSampler(val_ind), drop_last=True)
    

    device = torch.device("cuda:" + str(args.gpu) if args.cuda else "cpu")

    model = Pct(args).to(device)
    # model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    time_list = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1) # (batch,12,18)
        y1,y2 = model(data)
        logits = (y1 + y2)/2
        preds = logits.max(dim=1)[1] 
        
        # confidence,pred_index = nn.Softmax()(logits).max(dim=1)
        # print(confidence.detach().cpu().numpy(),pred_index.cpu().numpy())
        
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    
    # cm = confusion_matrix(test_true, test_pred,normalize='true')
    # np.save('plot/saved_matrixs/' + args.target_username,cm)
    
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    result_report = metrics.classification_report(test_true,test_pred,digits=3,target_names=['standing', 'walking', 'jogging','jumping','squatdown','squating','squat up','forward','backoff'])
    
    io.cprint('test acc: %s\n balanced acc : %s\n%s \n ' % (test_acc,avg_per_class_acc,result_report))

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size', 
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu', '-g', type=int, help='cuda id', default='1')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')

    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--model_path', type=str, default='checkpoints/change_to_your_dir', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--target_username', type=str, default='cuixinyu')
    parser.add_argument('--dataroot',type=str, help='directory of data', default='Locomotion/data/6frame/step3')
    parser.add_argument('--apply_pcm', type=bool, default=False)
    parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
    parser.add_argument('--exp_name', type=str, default='exp/change_to_your_dir', metavar='N',
                        help='Name of the experiment')
    
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # torch.cuda.set_device(args.gpu)
        # torch.cuda.set_device(1)
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
