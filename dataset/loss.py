import torch
import torch.nn as nn
from torch.autograd import Variable

class CertaintyCrossEntropy(nn.Module):
    def __init__(self, weight=None, n_classes=2, lmbda=0.1, budget=0.3, flooding=-1):
        super(CertaintyCrossEntropy, self).__init__()
        self.weight = weight
        self.n_classes = n_classes
        self.lmbda = lmbda
        self.budget = budget
        self.flooding = flooding
        self.loss = nn.NLLLoss(weight=self.weight)
        self.eps = 1e-12
    
    def encode_onehot(self, labels):
        onehot = torch.FloatTensor(labels.size()[0], self.n_classes)
        labels = labels.data
        if labels.is_cuda:
            onehot = onehot.cuda()
        onehot.zero_()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        return onehot
    
    def forward(self, pred, confidence, labels):
        pred = torch.softmax(pred, dim=1)
        confidence = torch.sigmoid(confidence)
        labels_onehot = self.encode_onehot(labels)

        pred = torch.clamp(pred, 0 + self.eps, 1 - self.eps)
        confidence = torch.clamp(confidence, 0 + self.eps, 1 - self.eps)

        # Randomly set half of the confidences to 1 (i.e. no hints)
        b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).to(pred.device)
        confidence = confidence * b + (1 - b)
        pred = pred * confidence.expand_as(pred) + labels_onehot * (1 - confidence.expand_as(labels_onehot))

        xenrtopy_loss = self.loss(torch.log(pred), labels)
        if self.flooding > 0:
            xenrtopy_loss = (xenrtopy_loss - self.flooding).abs() + self.flooding

        confidence_loss = torch.mean(-torch.log(confidence))
        loss = xenrtopy_loss + (self.lmbda * confidence_loss)

        if self.budget > confidence_loss.item():
            self.lmbda = self.lmbda / 1.01
        elif self.budget <= confidence_loss.item():
            self.lmbda = self.lmbda / 0.99
        return loss