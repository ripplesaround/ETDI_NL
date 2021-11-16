import torch
from torch.functional import Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
    return (-1) * torch.sum(torch.mul(y, torch.log(y)),dim=1)

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not, h1, h2):
    loss_1_update = F.cross_entropy(y_1[noise_or_not[ind]], t[noise_or_not[ind]])
    # print(y_1[noise_or_not[ind]])
   
    loss_2_update = F.cross_entropy(y_2, t)

    remember_rate = 1 - forget_rate
    # num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = 0
    pure_ratio_2 = 0

    # loss 2 是全部noisy的，loss 1 只有clean的
    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])

    # loss_2_update = torch.sum(loss_2.data.cpu())/ len(loss_2)

    # loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    # loss_2_update = F.cross_entropy(y_2[ind_2_update], t[ind_2_update])
    # loss_1_update = torch.sum(loss_1_update)/num_remember
    # loss_2_update = torch.sum(loss_2_update)/num_remember

    # # temperture
    # temp= 20 
    # tao = 0.01
    # omega = 0.3
    # KL_criterion = torch.nn.KLDivLoss()
    # # get pos label
    # t_one_hot = label_trans(t, y_1.size()[-1])
    # # 修改大loss的值
    # if(remember_rate < 1):
    #     # 进行采样，保证每一个类都有足够的数据    cnt = np.zeros(num_classes)
    #     h_1_s = h1[ind_1_sorted[:num_remember]]
    #     h_1_b = h1[ind_1_sorted[(num_remember):]]
    #     y_1_s = y_1[ind_1_sorted[:num_remember]]
    #     y_1_b = y_1[ind_1_sorted[(num_remember):]]
    #     h_2_s = h2[ind_2_sorted[:num_remember]]
    #     h_2_b = h2[ind_2_sorted[(num_remember):]]
    #     y_2_s = y_2[ind_2_sorted[:num_remember]]
    #     y_2_b = y_2[ind_2_sorted[(num_remember):]]
    #     # 衡量label的相似性
    #     sim_1 = similarity(h_1_b,h_1_s)    #[8,128]
    #     # y_1_refurb = refurbishment(y_1_b, y_1_s, sim_1)
    #     # 重构的标签是在model1里
    #     y_1_refurb = refurbishment(t_one_hot[ind_1_sorted[num_remember:]], t_one_hot[ind_1_sorted[:num_remember]], sim_1, omega=omega,t=temp)
    #     y_1_refurb = y_1_refurb.detach()
    #     # KL
    #     # distillation_loss_1 = 0
    #     # for i in range(len(y_1_refurb)):
    #     #     distillation_loss_1 += KL_criterion(F.log_softmax(y_1_b / t)[i], y_1_refurb[i])
    #     distillation_loss_1 = KL_criterion(F.log_softmax(y_1_b / temp), y_1_refurb)
    #     # distillation_loss_1 = distillation_loss_1.detach()


    #     # 衡量label的相似性
    #     sim_2 = similarity(h_2_b,h_2_s)    #[8,128]
    #     # sim_1 = similarity(h_1_b,h_1_s)    #[8,128]
    #     # y_2_refurb = refurbishment(y_2_b, y_2_s, sim_2)
    #     y_2_refurb = refurbishment(t_one_hot[ind_2_sorted[num_remember:]], t_one_hot[ind_2_sorted[:num_remember]], sim_2, omega=omega,t=temp)
    #     y_2_refurb = y_2_refurb.detach()
    #     # KL
    #     # distillation_loss_2 = 0
    #     # for i in range(len(y_2_refurb)):
    #     #     distillation_loss_2 += KL_criterion(F.log_softmax(y_2_b / t)[i], y_2_refurb[i]) 
    #     distillation_loss_2 = KL_criterion(F.log_softmax(y_2_b / temp), y_2_refurb)
    #     # distillation_loss_2 = distillation_loss_2.detach()

    #     loss_1_update += (tao*temp*temp*(distillation_loss_1))
    #     loss_2_update += (tao*temp*temp*(distillation_loss_2))

    # # 计算熵的变化
    # entropy_1 = torch.mean(cal_entropy(F.softmax(y_1[ind_1_sorted[:num_remember]],dim=1))).cpu().item()
    # entropy_2 = torch.mean(cal_entropy(F.softmax(y_2[ind_2_sorted[:num_remember]],dim=1))).cpu().item()

           
    return loss_1_update,loss_2_update, pure_ratio_1, pure_ratio_2
