import os

import cv2
import numpy
import time
import random
from PIL import Image

from torch.utils.data import Dataset


class CoRe(Dataset):

    def __init__(self, path, train=True, transform=None, target_transform=None):

        self.root = path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.imgs_path = None
        self.targets = None
        if train:
            self.data = numpy.load('./core50-data/data_train.npy')
            self.targets = numpy.load('./core50-data/label_train.npy').tolist()
        else:
            self.data = numpy.load('./core50-data/data_test.npy')
            self.targets = numpy.load('./core50-data/label_test.npy').tolist()
        print('core50')

    def __len__(self):
        return len(self.imgs_path)

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


def traversal_files(path):
    labels_train = []
    labels_test = []
    files_train = []
    files_test = []
    for item in os.scandir(path):
        if item.is_dir():
            if item.name[1:] in ['3', '7', '10']:
                for item2 in os.scandir(item.path):
                    for file in os.scandir(item2.path):
                        files_test.append(file.path)
                        labels_test.append(int(item2.name[1:]) - 1)
            else:
                for item2 in os.scandir(item.path):
                    for file in os.scandir(item2.path):
                        files_train.append(file.path)
                        labels_train.append(int(item2.name[1:]) - 1)


    data = []
    idx = 0
    batch = 0
    time1 = time.time()

    for file_path in files_train:
        if idx % 5000 == 0 and idx != 0:
            save_path = './core50-data/data' + str(batch) + '_train.npy'
            data = numpy.array(data)
            numpy.save(save_path, data)
            data = []
            batch += 1
            time2 = time.time()
            print(str(idx) + 'train images have been loaded')
            print('use time:' + str(time2 - time1) + 'seconds')
            time1 = time2
        idx += 1
        img = Image.open(file_path).convert('RGB')
        img = numpy.array(img)
        # if data is None:
        #     data = img[numpy.newaxis, :]
        # else:
        #     data = numpy.concatenate((data, img[numpy.newaxis, :]), axis=0)
        data.append(img)
    if len(data) != 0:
        save_path_f_train = './core50-data/data' + str(batch) + '_train.npy'
        numpy.save(save_path_f_train, data)
    save_path_l_train = './core50-data/label_train.npy'
    labels = numpy.array(labels_train)
    numpy.save(save_path_l_train, labels)

    data = []
    idx = 0
    batch = 0
    time1 = time.time()

    for file_path in files_test:
        if idx % 5000 == 0 and idx != 0:
            save_path = './core50-data/data' + str(batch) + '_test.npy'
            data = numpy.array(data)
            numpy.save(save_path, data)
            data = []
            batch += 1
            time2 = time.time()
            print(str(idx) + 'test images have been loaded')
            print('use time:' + str(time2 - time1) + 'seconds')
            time1 = time2
        idx += 1
        img = Image.open(file_path).convert('RGB')
        img = numpy.array(img)
        # if data is None:
        #     data = img[numpy.newaxis, :]
        # else:
        #     data = numpy.concatenate((data, img[numpy.newaxis, :]), axis=0)
        data.append(img)
    if len(data) != 0:
        save_path_f_test = './core50-data/data' + str(batch) + '_test.npy'
        numpy.save(save_path_f_test, data)
    save_path_l_test = './core50-data/label_test.npy'
    labels = numpy.array(labels_test)
    numpy.save(save_path_l_test, labels)


if __name__ == "__main__":
    traversal_files(r'./dataset/core50')
    ans_train = None
    for i in range(24):
        batch = numpy.load('./core50-data/data' + str(i) + '_train.npy')
        print(i)
        if ans_train is None:
            ans_train = batch
        else:
            ans_train = numpy.concatenate((ans_train, batch), axis=0)
    numpy.save('./core50-data/data_train.npy', ans_train)
    del ans_train
    ans_test = None
    for i in range(9):
        batch = numpy.load('./core50-data/data' + str(i) + '_test.npy')
        print(i)
        if ans_test is None:
            ans_test = batch
        else:
            ans_test = numpy.concatenate((ans_test, batch), axis=0)
    numpy.save('./core50-data/data_test.npy', ans_test)
