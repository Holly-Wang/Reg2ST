import torch.nn as nn
import torch
import torch.nn.functional as F

class NT_Xent(nn.Module):
    """NT_Xent loss for simclr."""

    def __init__(self, batch_size, temperature=1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        """Mask correlated samples.
        :param batch_size: batch size of the dataset
        :type batch_size: int
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """Calculate the compare loss."""
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # print(sim)
        # print(self.batch_size)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    
class DiceLoss(nn.Module):
    
    def __init__(self, reduce_zero_label=False):
        super(DiceLoss, self).__init__()
        print("reduce_zero_label:", reduce_zero_label)
        self.reduce_zero_label = reduce_zero_label

    def forward(self, input, target, reduce=True):
        input = torch.sigmoid(input)
        input = input.reshape(-1)
        target = target.reshape(-1).float()
        mask = ~torch.isnan(target)
        if self.reduce_zero_label:
            target = target - 1  # start from zero
        input = input[mask]
        target = target[mask]
        
        a = torch.sum(input * target)
        b = torch.sum(input * input) + 0.001
        c = torch.sum(target * target) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        if reduce:
            loss = torch.mean(loss)

        return loss


class BinaryCrossEntropyLoss(nn.Module):
    
    def __init__(self, weight, reduce_zero_label=False):
        super(BinaryCrossEntropyLoss, self).__init__()
        print("weight:", weight, "reduce_zero_label:", reduce_zero_label)
        if weight is not None:
            self.weight = torch.tensor(weight).cuda()
        else:
            self.weight = None
        self.reduce_zero_label = reduce_zero_label
            
    def forward(self, outputs, targets):
        # if outputs.size(-1) == 1 and len(outputs.shape) > 1:
        #     outputs = outputs.squeeze(-1)
        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)
        mask = ~torch.isnan(targets)
        if self.reduce_zero_label:
            targets = targets - 1  # start from zero
        if self.weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                outputs[mask].float(), targets[mask].float(), self.weight[targets[mask].long()], reduction="sum")
        else:
            loss = F.binary_cross_entropy_with_logits(
                outputs[mask].float(), targets[mask].float(), reduction="sum")
        sample_size = torch.sum(mask.type(torch.int64))
        return loss / sample_size


class CrossEntropyLoss(nn.Module):
    
    def __init__(self, weight, reduce_zero_label=False):
        super(CrossEntropyLoss, self).__init__()
        print("weight:", weight, "reduce_zero_label:", reduce_zero_label)
        if weight is not None:
            self.weight = torch.tensor(weight).cuda()
        else:
            self.weight = None
        self.reduce_zero_label = reduce_zero_label
    
    
    def forward(self, outputs, targets):
        mask = ~torch.isnan(targets)
        if self.reduce_zero_label:
            targets = targets - 1  # start from zero
        if self.weight is not None:
            loss = F.cross_entropy(
                outputs[mask].float(), targets[mask].long(), self.weight, reduction="sum")
        else:
            loss = F.cross_entropy(
                outputs[mask].float(), targets[mask].long(), reduction="sum")
        sample_size = torch.sum(mask.type(torch.int64))
        return loss / sample_size
        
