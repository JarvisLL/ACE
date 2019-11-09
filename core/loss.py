import torch
from torch import nn
import torch.nn.functional as F


class MixSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(**kwargs)
        self.ignore_label = ignore_label

    def forward(self, preds, target):
        return dict(loss=F.cross_entropy(preds, target, ignore_index=self.ignore_label))

class MSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__(**kwargs)
        self.mse = torch.nn.MSELoss()
        
    def forward(self, preds, target):
        return dict(loss=self.mse(preds,target))

# TODO: add aux support
class OHEMSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.6, min_kept=256,
                 down_ratio=1, reduction='mean', use_weight=False):
        super(OHEMSoftmaxCrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def base_forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept < num_valid and num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

    def forward(self, preds, target):
        for i, pred in enumerate(preds):
            if i == 0:
                loss = self.base_forward(pred, target)
            else:
                loss = loss + self.base_forward(pred, target)
        return dict(loss=loss)

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=False, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, min_kept=100000, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(min_kept=min_kept, ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self,preds,target):
        #preds, target = tuple(inputs)
        #inputs = tuple(list(preds) + [target])
        #print('input is ',len(*inputs))
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds,target))

weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                            1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                            1.0865, 1.1529, 1.0507]).to('cuda')
        
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2., use_weight=False, size_average=True, ignore_index=-1,soft_target=False):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.soft_target = soft_target
        if use_weight:
            self.nll_loss = nn.NLLLoss(weights,size_average,ignore_index)
        else:
            self.nll_loss = nn.NLLLoss(None,size_average,ignore_index)

    def forward(self, inputs, targets):
        if self.soft_target:
            inputs = inputs / 2
        return dict(loss=self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets))

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num+1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):  # variables
        P = F.softmax(inputs)

        b,c,h,w = inputs.size()
        class_mask = Variable(torch.zeros([b,c+1,h,w]).cuda())
        class_mask.scatter_(1, targets.long(), 1.)
        class_mask = class_mask[:,:-1,:,:]

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # print('alpha',self.alpha.size())
        alpha = self.alpha[targets.data.view(-1)].view_as(targets)
        # print (alpha.size(),class_mask.size(),P.size())
        probs = (P * class_mask).sum(1)  # + 1e-6#.view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            print(1111111111111111111)
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss

class CriterionKD(nn.Module):
    '''
    '''
    def __init__(self,ignore_label=255):
        super(CriterionKD,self).__init__()
        self.ignore_label = ignore_label
        self.criterion_kd = nn.KLDivLoss()
    def forward(self,predict,target,weight=None):

        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1}".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1}".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1}".format(predict.size(3), target.size(3))
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict[target_mask]
        loss = self.criterion_kd(F.log_softmax(predict),F.log_softmax(target))
        return loss

class Cos_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        #x = x.view(1,1,)
        C, width, height = x.size()
        x = x.view(-1,C,width,height)
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention

class CriterionSDcos(nn.Module):
    '''
    structure distillation loss based on graph
    '''
    def __init__(self, ignore_index=255, use_weight=True, pp=1, sp=1):
        super(CriterionSDcos, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.soft_p = sp
        self.pred_p = pp
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds[self.pred_p])
        graph_t = self.attn(soft[self.soft_p])
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph

class CriterionKlDivergence(nn.Module):
    '''
    '''
    def __init__(self):
        super(CriterionKlDivergence,self).__init__()
        self.criterion_kd = nn.KLDivLoss()
    def forward(self,s_feature,t_feature):
        #assert not t_feature.requires_grad
        assert s_feature.dim() == 4
        assert t_feature.dim() == 4
        assert s_feature.size(0) == t_feature.size(0),'{0} vs {1}'.format(s_feature.size(0),t_feature.size(0))
        assert s_feature.size(2) == t_feature.size(2),'{0} vs {1}'.format(s_feature.size(2),t_feature.size(2))
        assert s_feature.size(3) == t_feature.size(3),'{0} vs {1}'.format(s_feature.size(3),t_feature.size(3))
        return dict(loss = self.criterion_kd(F.log_softmax(s_feature),F.softmax(t_feature)))