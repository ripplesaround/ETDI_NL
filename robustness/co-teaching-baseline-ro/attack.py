import torch

class FGSM:
    """
    用FGSM生成 adversarial examples ，然后检测模型的鲁棒性
    """
    def __init__(self, model, ep=0.01, isRand=True):
        """
        isRand is set True to improve the attack success rate. 
        """
        self.isRand = isRand
        self.model = model
        self.ep = ep
        
    def generate(self, x, y, randRate=1):
        """
        x: clean inputs, shape of x: [batch_size, width, height, channel] 
        y: ground truth, one hot vectors, shape of y: [batch_size, N_classes] 
        """