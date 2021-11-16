import os
import pickle
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from absl import logging
from models.modeling import CNN, TextCNN
from tensorboardX import SummaryWriter
from utils import EarlyStopping
from matplotlib import pyplot as plt
import torch.autograd as autograd
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_step(data_loader, gpu: bool, model, optimizer, criterion):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.

    model = model.train()
    for x, y, y_hat in data_loader:
        # Forward and Backward propagation
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)

        out = model(x)
        model_loss = criterion(out, y_hat)

        # print(out.size())
        # print(y_hat.size())
        # print(model_loss)

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, model


def test_step(data_loader, gpu: bool, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def valid_step(data_loader, gpu: bool, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    for x, y, y_hat in data_loader:
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)

        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def train(FLAGS):
    # load dataset (train)
    train, valid, test = pickle.load(
        open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_{}_{}.pkl'.format(FLAGS.noise_prob, FLAGS.noise_type)),
             'rb'))
    if FLAGS.dataset.__eq__('TREC'):
        vocab = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_emb.pkl'), 'rb'))
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    logging.info('{} dataloader successfully loaded'.format(FLAGS.dataset))

    if FLAGS.dataset.__eq__('TREC'):
        model = TextCNN(vocab=vocab.stoi, num_class=FLAGS.num_class, drop_rate=FLAGS.drop_rate, pre_weight=vocab.vectors)
    else:
        model = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # learning history
    train_acc_list = []
    test_acc_list = []

    early_stopping = EarlyStopping(patience=FLAGS.stop_patience, verbose=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    for e in range(FLAGS.epochs):
        # training step
        train_accuracy, avg_loss, model = train_step(data_loader=train_data_loader,
                                                     gpu=FLAGS.gpu,
                                                     model=model,
                                                     optimizer=optimizer,
                                                     criterion=criterion)

        # testing/valid step
        test_accuracy = test_step(data_loader=test_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model)

        dev_accuracy = valid_step(data_loader=valid_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model)
        # dev_accuracy = 0

        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        logging.info('{} epoch, Train Loss {}, Train accuracy {}, Dev accuracy {}, Test accuracy {}'.format(e + 1,
                                                                                                            avg_loss,
                                                                                                            train_accuracy,
                                                                                                            dev_accuracy,
                                                                                                            test_accuracy))

        
        early_stopping(-dev_accuracy, model, test_acc=test_accuracy)
        if early_stopping.early_stop:
            logging.info('Training stopped! Best accuracy = {}'.format(max(early_stopping.acc_list)))
            break

    # learning curve plot
    xrange = [(i + 1) for i in range(FLAGS.epochs)]
    plt.plot(xrange, train_acc_list, 'b', label='training accuracy')
    plt.plot(xrange, test_acc_list, 'r', label='test accuracy')
    plt.legend()
    plt.title('Learning curve')
    plt.savefig('l_curve.png')

    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)
    torch.save(model.state_dict(),  # save model object before nn.DataParallel
               os.path.join(FLAGS.save_dir,
                            '{}_{}_{}_{}.pt'.format(FLAGS.dataset, FLAGS.model, FLAGS.noise_prob, FLAGS.noise_type)))


def show_big_loss(FLAGS):
    # load dataset (train) and models
    train, valid, test = pickle.load(
        open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_{}_{}.pkl'.format(FLAGS.noise_prob, FLAGS.noise_type)),
             'rb'))
    if FLAGS.dataset.__eq__('TREC'):
        vocab = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_emb.pkl'), 'rb'))
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    logging.info('{} dataloader successfully loaded'.format(FLAGS.dataset))
    if FLAGS.dataset.__eq__('TREC'):
        model = TextCNN(vocab=vocab.stoi, num_class=FLAGS.num_class, drop_rate=FLAGS.drop_rate, pre_weight=vocab.vectors)
    else:
        model = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    path = "checkpoint.pt"
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    
    test_accuracy = test_step(data_loader=test_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model)
    print(test_accuracy)
    
    rt = 0.5
    criterion=nn.CrossEntropyLoss(reduce=False)
    # small loss
    model_epoch_label_pre_before = []
    model_epoch_label_pre_after = []
    cnt_total = 0
    cnt_total1 = 0
    for x, y, y_hat in tqdm(train_data_loader):
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)
        model = model.train()
        out1 = model(x)
        model_loss = criterion(out1, y_hat)
        label_pre_before, label_pre_after,cnt,cnt1 = big_loss(model_loss,x,y,y_hat,rt, out1)
        model_epoch_label_pre_before.append(label_pre_before)
        model_epoch_label_pre_after.append(label_pre_after)
        cnt_total+=cnt
        cnt_total1+=cnt1
    epoch_label_pre_ratio_before = np.mean(np.array(model_epoch_label_pre_before))
    epoch_label_pre_ratio_after = np.mean(np.array(model_epoch_label_pre_after))
    logging.info("big loss epoch_label_pre_ratio_before: {}".format(epoch_label_pre_ratio_before))
    logging.info("big loss epoch_label_pre_ratio_after: {}".format(epoch_label_pre_ratio_after))
    logging.info("cnt: {}".format(cnt_total))
    logging.info("cnt1: {}".format(cnt_total1))

def big_loss(model_loss,x,y,y_hat,rt,out1):
    _, model_big_idx = torch.topk(model_loss, k=int(int(model_loss.size(0)) * rt), largest=True)
    y_hat_check = y_hat[model_big_idx]
    y_check = y[model_big_idx]
    out1_check = out1[model_big_idx]
    pred = F.softmax(out1_check,dim=1)
    label_pre_before = torch.eq(y_hat_check, y_check).sum().item() / y_check.size(0)

    cnt = 0
    cnt_total = 0
    for i in range(len(y_hat_check)):
        if pred[i].max(dim=0)[0].item()>0.5:
            if y_check[i] == int(pred[i].argmax(dim=0).item()) and y_check[i] != y_hat_check[i]:
                cnt+=1
            cnt_total+=1
            y_hat_check[i] = int(pred[i].argmax(dim=0).item())
            


    label_pre_after = torch.eq(y_hat_check, y_check).sum().item() / y_check.size(0)
    return label_pre_before, label_pre_after,cnt, cnt_total

def get_clean_sample(FLAGS):
    # load dataset (train)
    train, valid, test = pickle.load(
        open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_{}_{}.pkl'.format(FLAGS.noise_prob, FLAGS.noise_type)),
             'rb'))
    if FLAGS.dataset.__eq__('TREC'):
        vocab = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_emb.pkl'), 'rb'))
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    logging.info('{} dataloader successfully loaded'.format(FLAGS.dataset))
    if FLAGS.dataset.__eq__('TREC'):
        model = TextCNN(vocab=vocab.stoi, num_class=FLAGS.num_class, drop_rate=FLAGS.drop_rate, pre_weight=vocab.vectors)
    else:
        model = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    path = "checkpoint.pt"
    model.load_state_dict(torch.load(path))
    model = model.to(device)


    test_accuracy = test_step(data_loader=test_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model)
    print(test_accuracy)
    
    rt = 0.3
    # small gradient
    model_epoch_label_pre = []
    for x, y, y_hat in tqdm(train_data_loader):
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)
        label_pre, x_dc, y_hat_dc, y_dc = loss_gradient(model,x,y,y_hat,rt)
        model_epoch_label_pre.append(label_pre)

    epoch_label_pre_ratio = np.mean(np.array(model_epoch_label_pre))
    logging.info("small gradient epoch_label_pre_ratio: {}".format(epoch_label_pre_ratio))

    criterion=nn.CrossEntropyLoss(reduce=False)
    # small loss
    model_epoch_label_pre = []
    for x, y, y_hat in tqdm(train_data_loader):
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)
        model = model.train()
        out1 = model(x)
        model_loss = criterion(out1, y_hat)
        label_pre = small_loss(model_loss,x,y,y_hat,rt)
        model_epoch_label_pre.append(label_pre)
    epoch_label_pre_ratio = np.mean(np.array(model_epoch_label_pre))
    logging.info("small loss epoch_label_pre_ratio: {}".format(epoch_label_pre_ratio))


def small_loss(model_loss,x,y,y_hat,rt):
    _, model_sm_idx = torch.topk(model_loss, k=int(int(model_loss.size(0)) * rt), largest=False)
    y_hat_dc = y_hat[model_sm_idx]
    y_dc = y[model_sm_idx]

    label_pre = torch.eq(y_hat_dc, y_dc).sum().item() / y_dc.size(0)
    return label_pre

def loss_gradient(model,x, y, y_hat, rt,criterion=nn.CrossEntropyLoss(reduce=False)):
    model.eval()
    param = list(model.named_parameters())
    # 这里选-2是直接选最后一层的weight 和bias
    param_influence = []
    for n, p in param[-2:]:
        param_influence.append(p)
    param_shape_tensor = []
    param_size = 0
    for p in param_influence:
        tmp_p = p.clone().detach()
        param_shape_tensor.append(tmp_p)
        param_size += torch.numel(tmp_p)
    # instance-level
    influence_score = []
    for cnt in range(x.size(0)):
        model.zero_grad()
        out = model(x[cnt].unsqueeze(dim=0))
        self_loss = criterion(out, y_hat[cnt].unsqueeze(dim=0))
        self_grads = autograd.grad(self_loss, param_influence)
        self_grads_tensor = [item.view(-1) for item in self_grads]
        self_grads_tensor = torch.cat((self_grads_tensor[0], self_grads_tensor[1]), 0)
        influence_score.append (
            torch.dot(self_grads_tensor, self_grads_tensor).item()
        )
    influence_score = np.array(influence_score)
    
    # get clean sample
    inf_id_sorted = np.argsort(influence_score)
    id_update = inf_id_sorted[:int((rt)*len(influence_score))]
    small_gradient_set = influence_score[id_update]
    # print()
    # print(influence_score[inf_id_sorted[0]], influence_score[inf_id_sorted[-1]])
    # print(y[inf_id_sorted[0]] == y_hat[inf_id_sorted[0]])
    # print(y[inf_id_sorted[-1]] == y_hat[inf_id_sorted[-1]])
    x_dc = x[id_update]
    y_dc = y[id_update]
    y_hat_dc = y_hat[id_update]

     # 计算label准确率
    label_pre = torch.eq(y_hat_dc, y_dc).sum().item() / y_dc.size(0)
    # print(str(label_pre)+" "+str(y_dc.size(0)))

    return label_pre, x_dc, y_hat_dc, y_dc