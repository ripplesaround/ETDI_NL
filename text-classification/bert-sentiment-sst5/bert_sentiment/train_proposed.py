import os

import torch
from loguru import logger
from pytorch_transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm
import numpy as np
from .data import SSTDataset
from matplotlib import pyplot as plt
import torch.nn.functional as F

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def similarity(x,y):
    # DONE 一个一个元素的算
    # y = y.transpose(0, 1)
    x = x.view(x.size()[0],-1)
    y = y.view(y.size()[0],-1)
    # print(x.size())
    # print(x[0].size(), y.size())
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sim = None
    for i in range(len(x)):
        sim_one = (torch.cosine_similarity( x[i],y,dim=-1)).unsqueeze(0)
        # print("sim", sim_one.size())
        sim_one = F.softmax(sim_one)
        if sim is None:
            sim = sim_one.clone()
        else:
            sim = torch.cat((sim,sim_one),dim=0)
    # print(sim.size())
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

def loss_coteaching(y_1, y_2, t, forget_rate, h1, h2):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    
    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    loss_1_update = torch.sum(loss_1_update)/num_remember
    loss_2_update = torch.sum(loss_2_update)/num_remember


    temp= 1 
    tao = 0.0001
    omega = 0.5
    KL_criterion = torch.nn.KLDivLoss()
    # get pos label
    t_one_hot = label_trans(t, y_1.size()[-1])
    
    h_1_s = h1[ind_1_sorted[:num_remember]]
    h_1_b = h1[ind_1_sorted[(num_remember):]]
    y_1_s = y_1[ind_1_sorted[:num_remember]]
    y_1_b = y_1[ind_1_sorted[(num_remember):]]
    h_2_s = h2[ind_2_sorted[:num_remember]]
    h_2_b = h2[ind_2_sorted[(num_remember):]]
    y_2_s = y_2[ind_2_sorted[:num_remember]]
    y_2_b = y_2[ind_2_sorted[(num_remember):]]
    sim_1 = similarity(h_1_b,h_1_s)
    y_1_refurb = refurbishment(t_one_hot[ind_1_sorted[num_remember:]], t_one_hot[ind_1_sorted[:num_remember]], sim_1, omega=omega,t=temp)
    y_1_refurb = y_1_refurb.detach()
    distillation_loss_1 = KL_criterion(F.log_softmax(y_1_b / temp), y_1_refurb) / len(y_1_refurb)
    
    sim_2 = similarity(h_2_b,h_2_s)    #[8,128]
    # print(sim_2.size())
    # print(sim_2[0])
    y_2_refurb = refurbishment(t_one_hot[ind_2_sorted[num_remember:]], t_one_hot[ind_2_sorted[:num_remember]], sim_2, omega=omega,t=temp)
    y_2_refurb = y_2_refurb.detach()
    distillation_loss_2 = KL_criterion(F.log_softmax(y_2_b / temp), y_2_refurb) / len(y_2_refurb)

    loss_1_update += (tao*temp*temp*(distillation_loss_1))
    loss_2_update += (tao*temp*temp*(distillation_loss_2))
        
    return loss_1_update, loss_2_update


def co_teaching_train_one_epoch(model1, model2, lossfn, optimizer1, optimizer2, forget_rate, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model1.train()
    model2.train()
    train_loss_1, train_loss_2, train_acc_1, train_acc_2 = 0.0, 0.0, 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.to(device), labels.to(device)


        
        _, logits_1, hidden_states_1 = model1(batch, labels=labels)
        _, logits_2, hidden_states_2 = model2(batch, labels=labels)
        # err_1 = F.cross_entropy(logits_1, labels, reduce = False)
        
        loss_1, loss_2 = loss_coteaching(logits_1, logits_2, labels, forget_rate, hidden_states_1[-1], hidden_states_2[-1])

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        # acc
        train_loss_1 += loss_1.item()
        pred_labels_1 = torch.argmax(logits_1, axis=1)
        train_acc_1 += (pred_labels_1 == labels).sum().item()
    
        train_loss_2 += loss_2.item()
        pred_labels_2 = torch.argmax(logits_2, axis=1)
        train_acc_2 += (pred_labels_2 == labels).sum().item()

    train_loss_1 /= len(dataset)
    train_acc_1 /= len(dataset)
    train_loss_2 /= len(dataset)
    train_acc_2 /= len(dataset)
    return train_loss_1, train_loss_2, train_acc_1, train_acc_2



def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.to(device), labels.to(device)
            logits = model(batch)[0]
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


def train(
    root=True,
    binary=False,
    bert="bert-large-uncased",
    epochs=30,
    batch_size=64,
    save=False,
    noise = ["symmetric", 0.5],
    forget_rate = 0.5,
    num_gradual = 5
):
    print("proposed")
    noise_type ,noise_rate = noise
    print(noise_type, noise_rate)
    trainset = SSTDataset("train", root=root, binary=binary,noise_type=noise_type,noise_rate=noise_rate)
    devset = SSTDataset("dev", root=root, binary=binary)
    testset = SSTDataset("test", root=root, binary=binary)

    config = BertConfig.from_pretrained(bert)
    config.output_hidden_states=True
    if not binary:
        config.num_labels = 5
    model1 = BertForSequenceClassification.from_pretrained(bert, config=config)
    model2 = BertForSequenceClassification.from_pretrained(bert, config=config)

    model1 = model1.to(device)
    model2 = model2.to(device)
    print(model1)
    lossfn = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-5)

    train_acc_list_1 = []
    test_acc_list_1 = []
    train_acc_list_2 = []
    test_acc_list_2 = []

    rate_schedule = np.ones(epochs)*forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate, num_gradual)
   

    for epoch in range(1, epochs):
        train_loss_1, train_loss_2, train_acc_1, train_acc_2 = co_teaching_train_one_epoch(
            model1, model2, lossfn, optimizer1, optimizer2, rate_schedule[epoch],  trainset, batch_size=batch_size
        )
        val_loss_1, val_acc_1 = evaluate_one_epoch(
            model1, lossfn, optimizer1, devset, batch_size=batch_size
        )
        test_loss_1, test_acc_1 = evaluate_one_epoch(
            model1, lossfn, optimizer1, testset, batch_size=batch_size
        )
        val_loss_2, val_acc_2 = evaluate_one_epoch(
            model2, lossfn, optimizer2, devset, batch_size=batch_size
        )
        test_loss_2, test_acc_2 = evaluate_one_epoch(
            model2, lossfn, optimizer2, testset, batch_size=batch_size
        )
        logger.info(f"epoch={epoch}")
        # logger.info(
        #     f"train_loss_1={train_loss_1:.4f}, val_loss_1={val_loss_1:.4f}, test_loss_1={test_loss_1:.4f}"
        # )
        
        logger.info(
            f"train_acc_1={train_acc_1:.3f}, val_acc_1={val_acc_1:.3f}, test_acc_1={test_acc_1:.3f}"
        )
        logger.info(
            f"train_acc_2={train_acc_2:.3f}, val_acc_2={val_acc_2:.3f}, test_acc_2={test_acc_2:.3f}"
        )
        train_acc_list_1.append(train_acc_1)
        test_acc_list_1.append(test_acc_1)
        train_acc_list_2.append(train_acc_2)
        test_acc_list_2.append(test_acc_2)
        if save:
            label = "binary" if binary else "fine"
            nodes = "root" if root else "all"
            torch.save(model1, f"{bert}__{nodes}__{label}__e{epoch}.pickle")


    
    # plot
    xrange = [(i) for i in range(1, epochs)]
    plt.plot(xrange, train_acc_list_1, 'r', label='train')
    plt.plot(xrange, test_acc_list_1, 'skyblue',linestyle='dashed', label='test')
    # plt.plot(xrange, test_clean_acc2_list, 'lightgreen',linestyle='dashed', label='nr=0')
    # plt.plot(xrange, test_noisy_acc2_list, 'orange',linestyle='dashed', label='nr=1')
    plt.legend()
    plt.title('{}_{}'.format(noise_type, noise_rate))
    plt.savefig('./pic/proposed_{}_{}_{}_{}.png'.format("model1",epochs,noise_type, noise_rate))
    plt.clf()

    xrange = [(i) for i in range(1, epochs)]
    plt.plot(xrange, train_acc_list_2, 'r', label='train')
    plt.plot(xrange, test_acc_list_2, 'skyblue',linestyle='dashed', label='test')
    # plt.plot(xrange, test_clean_acc2_list, 'lightgreen',linestyle='dashed', label='nr=0')
    # plt.plot(xrange, test_noisy_acc2_list, 'orange',linestyle='dashed', label='nr=1')
    plt.legend()
    plt.title('{}_{}'.format(noise_type, noise_rate))
    plt.savefig('./pic/proposed_{}_{}_{}_{}.png'.format("model2",epochs,noise_type, noise_rate))
    plt.clf()

    logger.success("Done!")