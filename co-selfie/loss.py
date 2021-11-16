import torch
from torch.functional import Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

# 相似性衡量
def similarity(x,y):
    # DONE 一个一个元素的算
    sim = None
    for i in range(len(x)):
        sim_one = (torch.cosine_similarity(x[i], y,dim=-1)).unsqueeze(0)
        sim_one = F.softmax(sim_one)
        if sim is None:
            sim = sim_one.clone()
        else:
            sim = torch.cat((sim,sim_one),dim=0)
    return sim

# Label Refurbishment
def refurbishment(y, y_s, sim,omega = 0.3,t = 1):
    y_refurb = y.clone()
    y_refurb = y_refurb .detach()

    # 没有经过log的概率矩阵
    y_refurb = omega * y_refurb + \
                (1-omega) * torch.mm(sim, y_s )  #[8,120]*[120,10] -> [8,10]
    
    y_refurb = F.softmax(y_refurb / t)
    
    # 给refurb 归一化
    return y_refurb

# 变换参数
def label_trans(t,num_class):
    y = torch.zeros(len(t), num_class).cuda()
    for i in range(len(t)):
        y[i][t[i]]  =  1
    return y

# 计算熵
def cal_entropy(y):
    k = (-1) * y.size()[-1] * (math.log( y.size()[-1] ))
    # eph_temp = 1e-100
    for x in (y).nonzero():
        y[x[0]][x[1]] = y[x[0]][x[1]] * math.log(y[x[0]][x[1]])
    
    return (-1) * torch.sum(y,dim=1) / k

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not, h1, h2, history_q_1,history_q_2,epoch):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]

   
    
    # selfie
    # 找到clean的数据
    eph = 0.05
    q = 10
    warm_up = 20

    y_1_pred = torch.max(F.softmax(y_1),dim=1)[1].cpu().numpy()
    y_2_pred = torch.max(F.softmax(y_2),dim=1)[1].cpu().numpy()

    # 更新历史矩阵
    # 每次都需要生成计算出现频率的矩阵，这样比较简单
    col = int(epoch % q)
    history_h_1 = np.zeros([y_1.size()[0],y_1.size()[-1]], dtype = int) # 128 * 10
    history_h_2 = np.zeros([y_2.size()[0],y_2.size()[-1]], dtype = int) # 128 * 10
    for i in range(len(y_1_pred)):
        history_q_1[ind[i]][col] = y_1_pred[i]
        history_q_2[ind[i]][col] = y_2_pred[i]
        # history_h = np.zeros(y_1.size()[-1], dtype = int) #10
        for j in range(q):
            history_h_1[i][history_q_1[ind[i]][j]] += 1    
            history_h_2[i][history_q_2[ind[i]][j]] += 1    


    len_refurb_1 = 0
    len_refurb_2 = 0
    t_1 = t.clone()
    t_2 = t.clone()
    if epoch > warm_up - 1:
        # 计算熵
        history_p_1 = cal_entropy(torch.from_numpy(history_h_1 / q)).cuda()
        history_p_ind_1 = (history_p_1 <= eph).nonzero().squeeze().cpu()
        # print(history_p_ind.size())
        # print(history_p_ind[0:5])
        if history_p_ind_1.dim() == 0:
            loss_1_refurb = torch.zeros(1)
            ind_1_update_new = ind_1_update
            # len_refurb_1 = 0
        else:
            # refurb 和 clean 在更新的时候是不重叠的，所以可以在同一个上面进行更新
            for i in history_p_ind_1:
                # tofix
                t_1[i] = np.argsort(history_h_1[i])[-1]
                # print("there ", t[i])
                # if epoch == 20:
                #     print("argsort")
                #     print(history_p_1[i])
                #     print(np.argsort(history_h_1[i])[-1])
                #     print("here")
                    
            loss_1_refurb = F.cross_entropy(y_1[history_p_ind_1], t_1[history_p_ind_1])
            ind_1_update_new = [i.item() for i in ind_1_update if i not in history_p_ind_1]
            len_refurb_1 = len(history_p_ind_1)
        
        history_p_2 = cal_entropy(torch.from_numpy(history_h_2 / q)).cuda()
        history_p_ind_2 = (history_p_2 <= eph).nonzero().squeeze().cpu()
        if history_p_ind_2.dim() == 0:
            loss_2_refurb = torch.zeros(1)
            ind_2_update_new = ind_2_update
            # len_refurb_1 = 0
        else:
            # refurb 和 clean 在更新的时候是不重叠的，所以可以在同一个上面进行更新
            for i in history_p_ind_2:
                t_2[i] = np.argsort(history_h_2[i])[-1]
                # print("there ", t[i])
            
            loss_2_refurb = F.cross_entropy(y_2[history_p_ind_2], t_2[history_p_ind_2])
            ind_2_update_new = [i.item() for i in ind_2_update if i not in history_p_ind_2]
            len_refurb_2 = len(history_p_ind_2)
        
        if history_p_ind_2.dim() == 0:
            loss_1_refurb = torch.zeros(1)
        else:
            loss_1_refurb = F.cross_entropy(y_1[history_p_ind_2], t_1[history_p_ind_2])
           
        
        if history_p_ind_1.dim() == 0:
            loss_2_refurb = torch.zeros(1)
        else:         
            loss_2_refurb = F.cross_entropy(y_2[history_p_ind_1], t_2[history_p_ind_1])
        

        # 用small loss进行更新
        # self
        # print(ind_1_update.size(),history_p.size())
        # print(history_p[0:5])
        
        # ind_1_update_new = [i for i in ind_1_update if i not in history_p_ind]
        num_remember_1 = len(ind_1_update_new) + len_refurb_1
        num_remember_2 = len(ind_2_update_new) + len_refurb_2

        # print(ind_2_update_new[:5])

        loss_1_update = F.cross_entropy(y_1[ind_2_update_new], t_1[ind_2_update_new])
        loss_1_update = (torch.sum(loss_1_update) + torch.sum(loss_1_refurb))/num_remember_2
        loss_2_update = F.cross_entropy(y_2[ind_1_update_new], t_2[ind_1_update_new])
        loss_2_update = (torch.sum(loss_2_update) + torch.sum(loss_2_refurb))/num_remember_1
        # exchange

        # print("here")
    else:
        # exchange
        # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
        # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
        # loss_1_update = torch.sum(loss_1_update)/num_remember
        # loss_2_update = torch.sum(loss_2_update)/num_remember

        # self
        loss_1_update = F.cross_entropy(y_1, t)
        loss_1_update = torch.sum(loss_1_update)/len(t)
        loss_2_update = F.cross_entropy(y_2, t)
        loss_2_update = torch.sum(loss_2_update)/len(t)


        
        
    # 计算（可以用切片）
    
           
    return loss_1_update,loss_2_update, pure_ratio_1, pure_ratio_2, history_q_1, len_refurb_1, history_q_2, len_refurb_2
