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
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss_new = (model1_loss_filter * model1_loss).sum()/(int(int(model1_loss.size(0)) * rt))     #考虑每个batch的平均

    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss_new = (model2_loss_filter * model2_loss).sum()/(int(int(model1_loss.size(0)) * rt))

    return model1_loss_new, model2_loss_new, [model1_sm_idx,model2_sm_idx]

def plot_sample_bias(y,y_hat,clean_idx_list):
    model1_sm_idx, model2_sm_idx = clean_idx_list
    num_class = 10
    cnt_num_class = np.zeros(num_class)
    cnt_num_class_hat = np.zeros(num_class)
    for i in model1_sm_idx:
        cnt_num_class[y[i]] += 1
        cnt_num_class_hat[y_hat[i]] += 1
    return cnt_num_class, cnt_num_class_hat
    

def train_step(data_loader, gpu: bool, model_list: list, optimizer1, optimizer2, criterion, rt):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.

    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()
    
    # 记录每一个元素的东西
    num_class = 10
    cnt_num_class_total = np.zeros(num_class)
    cnt_num_class_hat_total = np.zeros(num_class) 
    for x, y, y_hat in tqdm(data_loader):
        # Forward and Backward propagation
        x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)

        out1 = model1(x)
        out2 = model2(x)

        model1_loss = criterion(out1, y_hat)
        model2_loss = criterion(out2, y_hat)
        model1_loss_new, model2_loss_new, clean_idx_list  = co_teaching_loss(model1_loss=model1_loss.clone(), model2_loss=model2_loss.clone(), rt=rt)
        # batch_size
        cnt_num_class, cnt_num_class_hat = plot_sample_bias(y=y, y_hat=y_hat, clean_idx_list=clean_idx_list)
        
        cnt_num_class_total += cnt_num_class
        cnt_num_class_hat_total += cnt_num_class_hat


        # loss exchange
        optimizer1.zero_grad()
        model1_loss_new.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
        optimizer1.step()

        optimizer2.zero_grad()
        model2_loss_new.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer2.step()

        avg_loss += (model1_loss_new.item() + model2_loss_new.item())

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1
    
    

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2], [cnt_num_class_total,cnt_num_class_hat_total]


def test_step(data_loader, gpu: bool, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()

        # logits = model(x)
        # acc = torch.eq(torch.argmax(logits, 1), y)
        # acc = acc.cpu().numpy()
        # acc = np.mean(acc)      #为什么是求平均，不应该是求和么
        # correct += np.sum(acc)
        # total += y.size(0)
        # avg_accuracy += acc
        # global_step += 1
    # return avg_accuracy / global_step
    return float(correct) / float(total)


def valid_step(data_loader, gpu: bool, model):
    model = model.eval()
    global_step = 0
    avg_accuracy = 0.

    correct = 0
    total = 0

    for images, labels,labels_hat in data_loader:
        images = images.to(device)
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()

    # for x, y, y_hat in data_loader:
    #     x, y, y_hat = x.to(device), y.to(device), y_hat.to(device)
    #
    #     logits = model(x)
    #     acc = torch.eq(torch.argmax(logits, 1), y)
    #     acc = acc.cpu().numpy()
    #     correct += np.sum(acc)
    #     total += y.size(0)
    #     acc = np.mean(acc)
    #     avg_accuracy += acc
    #     global_step += 1
    # return avg_accuracy / global_step
    return float(correct) / float(total)

def update_reduce_step(cur_step, num_gradual, tau=0.5):
    return 1.0 - tau * min(cur_step / num_gradual, 1)


def train(FLAGS):
    os.makedirs("./pic",exist_ok=True)

    # load dataset (train)
    train, valid, test = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_{}_{}.pkl'.format(FLAGS.noise_prob, FLAGS.noise_type)), 'rb'))

    if FLAGS.dataset.__eq__('TREC'):
        vocab = pickle.load(open(os.path.join(FLAGS.datapath, FLAGS.dataset + '_emb.pkl'), 'rb'))
    train_data_loader = torch.utils.data.DataLoader(train, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(valid, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4)
    logging.info('{} dataloader successfully loaded'.format(FLAGS.dataset))

    if FLAGS.dataset.__eq__('TREC'):
        model1 = TextCNN(vocab=vocab.stoi, num_class=FLAGS.num_class, drop_rate=FLAGS.drop_rate,
                         pre_weight=vocab.vectors)
        model2 = TextCNN(vocab=vocab.stoi, num_class=FLAGS.num_class, drop_rate=FLAGS.drop_rate,
                         pre_weight=vocab.vectors)
    else:
        model1 = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)
        model2 = CNN(num_class=FLAGS.num_class, dropout_rate=FLAGS.drop_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)

    model1.to(device)
    model2.to(device)

    # learning history
    train_acc_list = []
    test_acc_list = []

    early_stopping = EarlyStopping(patience=FLAGS.stop_patience, verbose=False)
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer1 = optim.Adam(model1.parameters(), lr=FLAGS.lr)
    optimizer2 = optim.Adam(model2.parameters(), lr=FLAGS.lr)
    # optimizer = optim.Adam(chain(model1.parameters(), model2.parameters()), lr=FLAGS.lr)
    
    for e in range(FLAGS.epochs):
        # update reduce step
        rt = update_reduce_step(cur_step=e, num_gradual=FLAGS.num_gradual, tau=FLAGS.tau)

        # training step
        train_accuracy, avg_loss, model_list, cnt_num_class_list = train_step(data_loader=train_data_loader,
                                                          gpu=FLAGS.gpu,
                                                          model_list=[model1, model2],
                                                          optimizer1=optimizer1,
                                                          optimizer2=optimizer2,
                                                          criterion=criterion,
                                                          rt=rt)
        model1, model2 = model_list
        

        cnt_num_class_total, cnt_num_class_hat_total = cnt_num_class_list
        # plot 每一个epoch的选出的small loss的类分布
        xrange = [(i + 1) for i in range(FLAGS.num_class)]
        plt.plot(xrange, cnt_num_class_total, 'b', label='Real Labels')
        plt.plot(xrange, cnt_num_class_hat_total, 'r', label='Dataset Labels')
        plt.legend()
        plt.title('small loss obtained from the class distribution')
        plt.savefig('./pic/{}_{}.png'.format("cifar10",e))
        plt.clf()


        # testing/valid step
        test_accuracy = test_step(data_loader=test_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model1)

        dev_accuracy = valid_step(data_loader=valid_data_loader,
                                  gpu=FLAGS.gpu,
                                  model=model1)

        train_acc_list.append(train_accuracy)
        test_acc_list.append(test_accuracy)

        logging.info(
            '{} epoch, Train Loss {}, Train accuracy {}, Dev accuracy {}, Test accuracy {}, Reduce rate {}'.format(e + 1,
                                                                                                                avg_loss,
                                                                                                                train_accuracy,
                                                                                                                dev_accuracy,
                                                                                                                test_accuracy,
                                                                                                                rt))

        # early_stopping(-dev_accuracy, model1, test_acc=test_accuracy)
        # if early_stopping.early_stop:
        #     logging.info('Training stopped! Best accuracy = {}'.format(max(early_stopping.acc_list)))
        #     break

    # learning curve plot
    xrange = [(i + 1) for i in range(FLAGS.epochs)]
    plt.plot(xrange, train_acc_list, 'b', label='training accuracy')
    plt.plot(xrange, test_acc_list, 'r', label='test accuracy')
    plt.legend()
    plt.title('Learning curve')
    plt.savefig('l_curve.png')

    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)
    torch.save(model1.state_dict(),  # save model object before nn.DataParallel
               os.path.join(FLAGS.save_dir,
                            '{}_{}_{}_{}.pt'.format(FLAGS.dataset, FLAGS.model, FLAGS.noise_prob, FLAGS.noise_type)))
