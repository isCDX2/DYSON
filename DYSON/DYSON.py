import torch.nn as nn
import torch
import random
from torch.nn import functional as F
from generate_etf import design_etf
from project import MLP


def stack_weights(alist, idxs):
    ans = []
    for idx in idxs:
        ans.append(alist[idx])
    return ans


class DYSON(nn.Module):
    def __init__(self, feature_extractor, k_nearest, mem_size, var_enforce, feature_dim=None):
        super(DYSON, self).__init__()
        self.feature_dim = feature_dim
        self.pseudo_anchor_num = 10  # etf 10
        self.feature = feature_extractor
        self.target = None
        self.proto = []
        self.proto_label = []
        self.pseudo_anchor = torch.randn((self.pseudo_anchor_num, self.feature_dim))  # (10++, dim)
        self.ex2 = []  # x^2 sum / bs
        self.ex1 = []  # x sum / bs

        self.cls_num = []
        self.fc_linear = None
        self.mirror = None
        self.mem_img = []
        self.mem_label = []
        self.k_nearest = k_nearest
        self.mem_size = mem_size
        self.var_enforce = var_enforce

    def get_protos(self, index):
        ans = None
        for idx in index:
            if ans is None:
                ans = self.proto[idx].unsqueeze(0)
            else:
                ans = torch.concat((ans, self.proto[idx].unsqueeze(0)), dim=0)
        return ans

    def save_memory(self, total_images, total_labes, labels):
        random.seed(1993)
        for label in labels:
            images = total_images[total_labes == label]
            if images.shape[0] == 0:
                return
            if label not in self.mem_label:
                if len(images) < self.mem_size:
                    self.mem_img.append(images)
                    self.mem_label = self.mem_label + [label]
                else:
                    samp_list = [x for x in range(len(images))]
                    samp_list = random.sample(samp_list, self.mem_size)
                    mem_img = images[samp_list]
                    self.mem_img.append(mem_img)
                    self.mem_label = self.mem_label + [label]
            else:
                l_index = self.mem_label.index(label)
                mem_num_l = len(self.mem_img[l_index])
                if mem_num_l < self.mem_size:
                    if len(images) > self.mem_size - mem_num_l:
                        samp_list = [x for x in range(len(images))]
                        samp_list = random.sample(samp_list, self.mem_size - mem_num_l)
                        mem_img = images[samp_list]
                        self.mem_img[l_index] = torch.concat((self.mem_img[l_index], mem_img), dim=0)
                    else:
                        self.mem_img[l_index] = torch.concat((self.mem_img[l_index], images), dim=0)

    def update_proto(self, feas, label, bs):
        ex2 = feas.T.norm(2, 1)
        ex1 = feas.T.norm(1, 1)
        if label not in self.proto_label:
            self.ex2.append(ex2 / bs)
            self.ex1.append(ex1 / bs)
            self.proto.append(feas.mean(dim=0))
            self.cls_num.append(bs)
            self.proto_label.append(label)
            return 0
        else:
            cls_index = self.proto_label.index(label)
            ori_num = self.cls_num[cls_index]
            self.ex2[cls_index] = (self.ex2[cls_index] ** 2 + (ex2 / ori_num) ** 2).sqrt() * ori_num / (ori_num + bs)
            self.ex1[cls_index] = (self.ex1[cls_index] + ex1 / ori_num) * ori_num / (ori_num + bs)
            self.proto[cls_index] = (self.proto[cls_index] * self.cls_num[cls_index] + feas.mean(dim=0) * bs) / (
                    self.cls_num[cls_index] + bs)
            self.cls_num[cls_index] += bs
            return 1

    def update_FcAndMirror(self, feature_dim, device):
        if self.fc_linear is None:
            self.mirror = MLP(feature_dim, feature_dim, feature_dim)
        # first M pseudo_anchor replace with class proto
        if len(self.proto) < self.pseudo_anchor_num:
            pseudo = self.pseudo_anchor[len(self.proto):, ].to(device)
            designated_mat = torch.concat((torch.stack(self.proto), pseudo), dim=0)
        else:
            designated_mat = torch.stack(self.proto)
        new_etfs = design_etf(designated_mat.T).T
        self.fc_linear = new_etfs.to(device)

    def fussion_forward(self, x, train_label=None):
        self.eval()
        N = torch.tensor(self.cls_num)
        ex2 = torch.stack(self.ex2)
        ex1 = torch.stack(self.ex1)
        N = N.unsqueeze(1).repeat(1, ex2.shape[1]).to(ex2.device)
        rd = (N * ex2 ** 2 - ex1 ** 2).sqrt()
        rd = F.softmax((rd.max(dim=1)[0].unsqueeze(1) - rd), dim=1)
        if len(self.proto) < self.k_nearest + 1:
            x_temp = x.unsqueeze(1).repeat(1, len(self.proto), 1)
            if self.var_enforce:
                x_dists = ((x_temp - torch.stack(self.proto).unsqueeze(0).repeat(x.shape[0], 1,1)) ** 2) * rd.unsqueeze(0).repeat(x.shape[0], 1, 1)
            else:
                x_dists = ((x_temp - torch.stack(self.proto).unsqueeze(0).repeat(x.shape[0], 1, 1)) ** 2)
            topk_conf = x_dists.sum(dim=2)
            topk_conf = topk_conf.sum(dim=1).unsqueeze(1) / topk_conf
            topk_index = torch.arange(start=0, end=len(self.proto), step=1).unsqueeze(0).repeat(x.shape[0], 1)
        else:
            x_temp = x.unsqueeze(1).repeat(1, len(self.proto), 1)
            if self.var_enforce:
                x_dists = ((x_temp - torch.stack(self.proto).unsqueeze(0).repeat(x.shape[0], 1, 1)) ** 2) * rd.unsqueeze(0).repeat(x.shape[0], 1, 1)
            else:
                x_dists = ((x_temp - torch.stack(self.proto).unsqueeze(0).repeat(x.shape[0], 1, 1)) ** 2)
            simi_mat = x_dists.sum(dim=2)
            topk = torch.topk(simi_mat, self.k_nearest, 1, largest=False)
            topk_index = topk.indices
            topk_conf = topk.values
            topk_conf = topk_conf.sum(dim=1).unsqueeze(1) / topk_conf
        output = topk_conf
        temp = torch.max(output, dim=1)[1]
        ans_index = []
        for j in range(temp.shape[0]):
            ans_index.append(topk_index[j][temp[j]])
        predict = [self.proto_label[i] for i in ans_index]
        if train_label is not None:
            return torch.tensor(predict), x_dists, ans_index
        else:
            return torch.tensor(predict)

    def linear_forward(self, feas):
        fc_linears = torch.stack(stack_weights(self.fc_linear, range(len(self.fc_linear))))
        out_put = torch.mm(feas, fc_linears.t())
        return out_put

    def predict_fc_linear(self, feas):
        output = self.linear_forward(feas)
        ans = torch.topk(output, 1, -1)[1]
        predict = [self.proto_label[x] for x in ans]
        return torch.tensor(predict)

    def NC_linear(self, input):
        self.mirror.eval()
        input = self.mirror(input)

        if torch.norm(input) == 0:
            output = torch.matmul(input, self.fc_linear[:len(self.proto)].T)
        else:
            output = torch.matmul((input / torch.norm(input)), self.fc_linear[:len(self.proto)].T)
        ans = torch.topk(output, 1, -1)[1]
        predict = [self.proto_label[x] for x in ans]

        return torch.tensor(predict)

    def forward(self, input):
        x = self.feature(input)
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        predict = self.NC_linear(x)
        predict2 = self.fussion_forward(x)
        return predict, predict2
