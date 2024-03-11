import os
import PIL.Image as Image

import cv2
import numpy
import numpy as np

from torch.utils.data import Dataset
import time
from torchvision import transforms


class CUB(Dataset):

    def __init__(self, path, train=True, transform=None, target_transform=None):

        self.root = path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        self.first_trans = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.RandomCrop((256, 256), padding=4)])
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id

        self.data_id = []
        if self.train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
            if os.path.exists('cub_data_train.npy') and os.path.exists('cub_labels_train.npy'):
                self.data = np.load('cub_data_train.npy')
                self.targets = np.load('cub_labels_train.npy')
                self.targets = self.targets.tolist()
            else:
                self.pretreat(True)
            print('cub_train')

        if not self.train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)
            if os.path.exists('cub_data_test.npy') and os.path.exists('cub_labels_test.npy'):
                self.data = np.load('cub_data_test.npy')
                self.targets = np.load('cub_labels_test.npy')
                self.targets = self.targets.tolist()
            else:
                self.pretreat(False)
            print('cub_test')

    def pretreat(self, train):  # when first run on cub200, use this to save dataset to npy file
        print("When running cub200 for the first time, loading data takes more time")
        self.data = None
        self.targets = []
        start_time = time.time()
        point = 0
        for idx in self.data_id:
            point += 1
            if point % 100 == 0:
                print(point)
            path = self.root + '/images/' + self.images_path[idx]
            label = int(self.class_ids[idx]) - 1
            self.targets.append(label)
            img = Image.open(path).convert('RGB')
            img = numpy.array(self.first_trans(img))
            if self.data is None:
                self.data = img[numpy.newaxis, :]
            else:
                self.data = numpy.concatenate((self.data, img[numpy.newaxis, :]), axis=0)
        if train:
            np.save('cub_data_train.npy', self.data)
            labels = np.array(self.targets)
            np.save('cub_labels_train.npy', labels)
            end_time = time.time()
            print('dataload_time:%s sec' % (start_time - end_time))
        else:
            np.save('cub_data_test.npy', self.data)
            labels = np.array(self.targets)
            np.save('cub_labels_test.npy', labels)
            end_time = time.time()
            print('dataload_time:%s sec' % (start_time - end_time))

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, 'images', path))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]
