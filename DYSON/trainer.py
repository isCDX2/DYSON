import random

import PIL.Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from DYSON import DYSON
from iDATA import iCIFAR100, iCIFAR10, iCUB, iCORE50
from gaussin_schedule import gaussian_schedule, split_way_schedule


def stack_mems(mem_list):
    ans = None
    labels = None
    for i_label, mem in enumerate(mem_list):
        if ans is None:
            ans = mem
            labels = torch.ones(mem.shape[0]) * i_label
        else:
            ans = torch.concat((ans, mem), dim=0)
            labels = torch.concat((labels, torch.ones(mem.shape[0]) * i_label), dim=0)
    return ans, labels


def sample_i(i, x, y=None):
    index = torch.LongTensor(random.sample(range(x.shape[0]), i))
    x1 = torch.index_select(x, 0, index.to(x.device))
    if y is not None:
        y1 = torch.index_select(y, 0, index)
        return x1, y1
    else:
        return x1


class Trainer:
    def __init__(self, args, file_name, feature_extractor, device, feature_dim):
        self.file_name = file_name
        self.args = args
        self.learning_rate = args.learning_rate
        self.model = DYSON(feature_extractor, args.k_nearest, args.mem_size,
                           args.var_enforce, feature_dim)

        self.prototype = None
        self.class_label = None
        self.device = device

        self.get_data(args.data_name)
        self.train_loader = None
        self.test_loader = None
        if args.schedule_type == 'gaussian':
            self.schedule = gaussian_schedule(args.total_nc)
        else:
            self.schedule = split_way_schedule(args.split_num, args.total_nc, args.batch_size,
                                               self.train_dataset.data.shape[0])
        if args.data_name == 'core50':
            self.schedule[0]['label_set'] = self.schedule[0]['label_set'] + self.schedule[1]['label_set']
            self.schedule[0]['n_batches'] *= 2
            del self.schedule[1]

        print('init end')

    def get_data(self, data_name):
        if data_name == 'cifar100':
            self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                       transforms.Resize((128, 128), interpolation=PIL.Image.CUBIC),
                                                       transforms.RandomHorizontalFlip(p=0.5),
                                                       transforms.ColorJitter(brightness=0.24705882352941178),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                            (0.2675, 0.2565, 0.2761))])
            self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Resize((128, 128), interpolation=PIL.Image.CUBIC),
                                                      transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                           (0.2675, 0.2565, 0.2761))])
            self.train_dataset = iCIFAR100('./dataset', transform=self.train_transform, download=True)
            self.test_dataset = iCIFAR100('./dataset', test_transform=self.test_transform, train=False, download=True)
        elif data_name == 'cifar10':
            self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                       transforms.Resize((128, 128), interpolation=PIL.Image.CUBIC),
                                                       transforms.RandomHorizontalFlip(p=0.5),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                            (0.24205776, 0.23828046, 0.25874835))])
            self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Resize((128, 128), interpolation=PIL.Image.CUBIC),
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                           (0.24205776, 0.23828046, 0.25874835))])
            self.train_dataset = iCIFAR10('./dataset', transform=self.train_transform, download=True)
            self.test_dataset = iCIFAR10('./dataset', test_transform=self.test_transform, train=False, download=True)
        elif data_name == 'cub200':
            self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                                       transforms.ColorJitter(brightness=0.24705882352941178),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.48560741861744905, 0.49941626449353244,
                                                                             0.43237713785804116],
                                                                            [0.2321024260764962, 0.22770540015765814,
                                                                             0.2665100547329813])])
            self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize(
                                                          [0.4862169586881995, 0.4998156522834164, 0.4311430419332438],
                                                          [0.23264268069040475, 0.22781080253662814,
                                                           0.26667253517177186])])
            self.train_dataset = iCUB('./dataset/CUB_200_2011', transform=self.train_transform)
            self.test_dataset = iCUB('./dataset/CUB_200_2011', test_transform=self.test_transform, train=False)
        elif data_name == 'core50':
            self.train_transform = transforms.Compose([transforms.CenterCrop((100, 100)),
                                                       transforms.Resize((128, 128), interpolation=PIL.Image.CUBIC),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                                            [0.229, 0.224, 0.225])])
            self.test_transform = transforms.Compose([transforms.CenterCrop((100, 100)),
                                                      transforms.Resize((128, 128), interpolation=PIL.Image.CUBIC),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])])
            self.train_dataset = iCORE50('./dataset/core50-data', transform=self.train_transform)
            self.test_dataset = iCORE50('./dataset/core50-dat', test_transform=self.test_transform, train=False)
        else:
            print('wrong data name')

    def pseudo_select(self, mode, bs, total_number):
        pseudo_number = 0
        if mode == 'follow_bs':
            pseudo_number = bs
        elif mode == 'all':
            pseudo_number = total_number
        elif mode == 'customize':
            pseudo_number = int(bs * self.args.customize_rate)
        else:
            print('wrong mode')
        return pseudo_number

    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        train_targets = self.train_dataset.targets
        test_targets = self.test_dataset.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        print(100 * '#')
        print(self.class_order)

        self.train_dataset.targets = self.map_new_class_index(train_targets, self.class_order)
        self.test_dataset.targets = self.map_new_class_index(test_targets, self.class_order)

    def beforeTrain(self):
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, filter_label):
        if len(filter_label) > 0:
            self.train_dataset.getGaussinTrain(filter_label)

        temp_label = []
        for l in filter_label:
            if l not in self.model.proto_label:
                temp_label.append(l)
        self.test_dataset.getGaussinTest(self.model.proto_label + temp_label)  # cyj

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now([0, classes])  # cyj
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self):
        acc_list = []
        for i, epoch in enumerate(self.schedule):
            train_labe = epoch['label_set']
            train_batch = epoch['n_batches'] * self.args.lowdata_rate
            self.train_loader, self.test_loader = self._get_train_and_test_dataloader(train_labe)
            self.model.feature.eval()
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)
                with torch.no_grad():
                    feas = self.model.feature(images)
                    if len(feas.shape) > 2:
                        feas = feas.view(feas.shape[0], -1)
                if step > train_batch and self.args.schedule_type == 'gaussian':
                    break
                if self.args.mem_size != 0:
                    self.model.save_memory(feas, target, train_labe)
                target_increase = 1
                for l_i in train_labe:
                    feas_i = feas[target == l_i]
                    if feas_i.shape[0] != 0:
                        target_increase *= self.model.update_proto(feas_i, l_i, feas_i.shape[0])
                if target_increase == 0:
                    with torch.no_grad():
                        self.model.update_FcAndMirror(feas.shape[1], self.device)
                        self.model.to(self.device)
                    self.proto_only_training()
                else:
                    self.model.to(self.device)

                self.model.mirror.train()
                opt1 = torch.optim.Adam(self.model.mirror.parameters(), lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
                lr_scheluder = torch.optim.lr_scheduler.StepLR(optimizer=opt1, step_size=6, gamma=0.1)
                for _ in range(self.args.epoch):
                    opt1.zero_grad()
                    fc_loss = 0
                    for l_i in train_labe:
                        feas_i = feas[target == l_i]
                        if feas_i.shape[0] != 0:
                            l_i_loss = self._compute_loss(feas_i, l_i)
                            if torch.isnan(l_i_loss).any():
                                continue
                            else:
                                fc_loss += l_i_loss

                    if self.args.mem_size != 0:
                        mem_loss = self._compute_mem_loss()
                        fc_loss += mem_loss

                    fc_loss.backward()
                    opt1.step()
                    lr_scheluder.step()

            if i % self.args.print_freq == 0:
                accuracy, accuracy2 = self._test(self.test_loader)
                acc_list.append(accuracy)
                print('test_session:%d, accuracy:%.5f, NCM acc:%.5f' % (i, accuracy, accuracy2))
        print('average acc:' + str(torch.tensor(acc_list).mean()))

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        correct2 = 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                predicts, predicts2 = self.model(imgs)
            correct += (predicts.cpu() == labels.cpu()).sum()
            correct2 += (predicts2.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        accuracy2 = correct2.item() / total
        return accuracy, accuracy2

    def pass_loss(self, feas):
        N = torch.tensor(self.model.cls_num)
        ex2 = torch.stack(self.model.ex2)
        ex1 = torch.stack(self.model.ex1)
        N = N.unsqueeze(1).repeat(1, ex2.shape[1]).to(ex2.device)
        rd = (N * ex2 ** 2 - ex1 ** 2).sqrt().mean(dim=-1)
        proto_aug = []
        aug_label = []
        total_num = len(self.model.proto_label)
        index = list(range(total_num))
        for _ in range(int(feas.shape[0] * self.args.pass_rate)):
            np.random.shuffle(index)
            rand = np.random.normal(0, 1, feas.shape[1])
            rand = torch.from_numpy(rand).to(feas.device)
            temp = self.model.proto[index[0]] + rand * rd[index[0]]
            temp_label = index[0]
            proto_aug.append(temp)
            aug_label.append(temp_label)
        if len(proto_aug) != 0:
            proto_aug = torch.stack(proto_aug).float()
            aug_label = torch.tensor(aug_label).to(feas.device).long()
            soft_feat_aug = self.model.linear_forward(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.args.temp, aug_label)
        else:
            loss_protoAug = 0
        return loss_protoAug

    def NC_loss(self, feas, l_i):
        N = torch.tensor(self.model.cls_num)
        ex2 = torch.stack(self.model.ex2)
        ex1 = torch.stack(self.model.ex1)
        N = N.unsqueeze(1).repeat(1, ex2.shape[1]).to(ex2.device)
        rd = (N * ex2 ** 2 - ex1 ** 2 + 1e-6).sqrt()
        proto_aug = []
        aug_label = []
        total_num = len(self.model.proto_label)
        index = list(range(total_num))
        for _ in range(int(feas.shape[0] * self.args.pseudo_rate)):
            np.random.shuffle(index)
            rand = np.random.normal(0, 1, feas.shape[1])
            rand = torch.from_numpy(rand).to(feas.device)
            temp = self.model.proto[index[0]] + rand * (rd[index[0]])
            temp_label = index[0]
            proto_aug.append(temp)
            aug_label.append(temp_label)
        if len(proto_aug) != 0:
            proto_aug = torch.stack(proto_aug).float()
            aug_label = torch.tensor(aug_label).to(feas.device).long()

        index = self.model.proto_label.index(l_i)
        etf_now = self.model.fc_linear[index]
        old_etf = self.model.fc_linear[:len(self.model.proto)]
        protos = torch.stack(self.model.proto)
        fp = torch.concat((feas, protos), dim=0)
        fpp = torch.concat((fp, proto_aug), dim=0)  # (feature, protos, proto_avg)
        mirrored = self.model.mirror(fpp)
        etf_fp = torch.concat((etf_now.unsqueeze(0).repeat(feas.shape[0], 1), old_etf), dim=0)
        aligned = torch.concat((etf_fp, self.model.fc_linear[aug_label]), dim=0)
        if torch.norm(mirrored) == 0:
            loss = ((mirrored * aligned - 1) ** 2) / 2
        else:
            loss = (((mirrored / torch.norm(mirrored)) * aligned - 1) ** 2) / 2
        feas_align_loss = loss[:feas.shape[0]]
        proto_align_loss = loss[feas.shape[0]:feas.shape[0] + len(self.model.proto_label)]
        pseudo_align_loss = loss[feas.shape[0] + len(self.model.proto_label):]
        if self.args.pseudo_weight != 0:
            loss = feas_align_loss.mean() \
                   + proto_align_loss.mean() * self.args.proto_align_weight \
                   + pseudo_align_loss.mean() * self.args.pseudo_align_weight
        else:
            loss = feas_align_loss.mean() + proto_align_loss.mean() * self.args.proto_align_weight
        return loss

    def _compute_loss(self, feas, l_i):
        if self.args.pseudo_weight != 0:
            loss = self.NC_loss(feas, l_i)
        elif self.args.pass_pseudo != 0:
            out_put = self.model.linear_forward(feas)
            targets = torch.ones(out_put.shape[0]) * self.model.proto_label.index(l_i)
            targets = targets.to(out_put.device)
            loss = torch.nn.CrossEntropyLoss()(out_put, targets.long())
            loss_pass = self.pass_loss(feas)
            loss = loss + loss_pass
        else:
            out_put = self.model.linear_forward(feas)
            targets = torch.ones(out_put.shape[0]) * self.model.proto_label.index(l_i)
            targets = targets.to(out_put.device)
            loss = torch.nn.CrossEntropyLoss()(out_put, targets.long())
        return loss

    def _compute_mem_loss(self):
        mem_loss = 0
        if self.args.mem_size != 0:
            mem_feas, mem_target = stack_mems(self.model.mem_img)
            mirrored = self.model.mirror(mem_feas)
            aligned = self.model.fc_linear[mem_target.long()]
            if torch.norm(mirrored) == 0:
                mem_loss = ((mirrored * aligned - 1) ** 2) / 2
            else:
                mem_loss = (((mirrored / torch.norm(mirrored)) * aligned - 1) ** 2) / 2
        return mem_loss.mean()

    def proto_only_training(self):
        protos = torch.stack(self.model.proto).to(self.device)
        etfs = self.model.fc_linear[:len(self.model.proto)]
        opt = torch.optim.Adam(self.model.mirror.parameters(), lr=self.args.learning_rate * 0.2,
                               weight_decay=self.args.weight_decay)
        for _ in range(10):
            opt.zero_grad()
            loss = 0
            mirrored = self.model.mirror(protos)
            if torch.norm(mirrored) == 0:
                loss += (((mirrored * etfs - 1) ** 2) / 2).mean()
            else:
                loss += ((((mirrored / torch.norm(mirrored)) * etfs - 1) ** 2) / 2).mean()
            loss.backward()
            opt.step()

    def afterTrain(self):
        self.test_loader = self._get_test_dataloader(self.args.total_nc)
        accuracy, accuracy2 = self._test(self.test_loader)
        print('final accuracy:%.5f, NCM acc:%.5f' % (accuracy, accuracy2))
