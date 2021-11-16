from tarfile import XHDTYPE
import torch
from torch.functional import Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def gen_adv_dataset(train_loader,model, test_model=None):
    """
    用于模型生成对抗的数据
    """
    correct = 0
    total = 0
    # print("hello")
    fgsm = FGSM(model=model)
    pgd = PGD(model=model)

    # FGSM
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()    
        images_adv , labels_adv,ind = fgsm.generate(x=images,y=labels,ind = indexes)
        images_adv = Variable(images_adv).cuda() 
        labels_adv = Variable(labels_adv)

        logits,_ = test_model(images_adv)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels_adv.size(0)
        correct += (pred.cpu() == labels_adv.cpu()).sum()
    fgsm_acc = 100*float(correct)/float(total)

    # PGD
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()    
        images_adv , labels_adv,ind = pgd.generate(x=images,y=labels,ind = indexes)
        images_adv = Variable(images_adv).cuda() 
        labels_adv = Variable(labels_adv)

        logits,_ = test_model(images_adv)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels_adv.size(0)
        correct += (pred.cpu() == labels_adv.cpu()).sum()
    pgd_acc = 100*float(correct)/float(total)

   
    return fgsm_acc, pgd_acc
        

class FGSM:
    """
    We use FGSM to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.01, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        self.model.eval()
    def generate(self, x, y, ind, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        # print("generate")
        # xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        xi = Variable(x).cuda().float()
        target = Variable(y).cuda()
        xi.requires_grad=True
        # y.requires_grad=True
        logits,re_1 = self.model(xi)

        # 求出梯度
        loss = F.cross_entropy(logits, target)
        grads = torch.autograd.grad(outputs=logits,inputs=xi,grad_outputs = torch.ones_like(logits))[0]

        # 生成对抗样本
        delta = torch.sign(grads)
        x_adv = xi + self.ep * delta
        
        x_adv = torch.clamp(x_adv,min = xi-self.ep,max=xi+self.ep)
        x_adv = torch.clamp(x_adv,min = 0, max=1)
        logits_adv,_ = self.model(x_adv)
        # print(np.argmax(logits_adv.detach().cpu().numpy(), axis=1))
        # print(y.detach().cpu().numpy())
        idxs = np.where(np.argmax(logits_adv.detach().cpu().numpy(), axis=1) != y.detach().cpu().numpy())[0]
        print("FGSM SUCCESS:", len(idxs))

        x_adv, xi, target, ind_adv = x_adv[idxs], xi[idxs], target[idxs], ind.detach().cpu().numpy()[idxs]
        x_adv = Variable(x_adv)
        target = Variable(target)

        return x_adv, target, ind_adv


class PGD:
    """
    We use PGD to generate a batch of adversarial examples. 
    """
    def __init__(self, model, ep=0.01, step=None, epochs=10, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        if step == None:
            self.step = ep/6
        self.epochs = epochs
        self.model.eval()

    def generate(self, x, y, ind, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """
        # print("generate")
        # xi = x.copy()
        if self.isRand:
            x = x + np.random.uniform(-self.ep * randRate, self.ep * randRate, x.shape)
            x = np.clip(x, 0, 1)
        
        xi = Variable(x).cuda().float()
        x_adv = Variable(x).cuda().float()
        target = Variable(y).cuda()
        xi.requires_grad=True
        x_adv.requires_grad=True
        # y.requires_grad=True
        

        # 求出梯度
        for i in range(self.epochs): 
            logits,re_1 = self.model(x_adv)
            loss = F.cross_entropy(logits, target)
            grads = torch.autograd.grad(outputs=logits,inputs=x_adv,grad_outputs = torch.ones_like(logits))[0]
            delta = torch.sign(grads)
            x_adv = torch.add(x_adv, self.ep * delta)
            x_adv = torch.clamp(x_adv,min = xi-self.ep,max=xi+self.ep)
            x_adv = torch.clamp(x_adv,min = 0, max=1)

        # 测试
        logits_adv,_ = self.model(x_adv)
        # print(np.argmax(logits_adv.detach().cpu().numpy(), axis=1))
        # print(y.detach().cpu().numpy())
        idxs = np.where(np.argmax(logits_adv.detach().cpu().numpy(), axis=1) != y.detach().cpu().numpy())[0]
        print("PGD SUCCESS:", len(idxs))

        x_adv, xi, target, ind_adv = x_adv[idxs], xi[idxs], target[idxs], ind.detach().cpu().numpy()[idxs]
        x_adv = Variable(x_adv)
        target = Variable(target)

        return x_adv, target, ind_adv


