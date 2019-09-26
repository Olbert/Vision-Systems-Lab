# @title Default title text
import json
import os
import xml.etree.ElementTree as ET
import numpy as np

import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class CudaVisionDataset(Dataset):

    def __init__(self, dir_path, no_of_classes=4,
                 channel_lut=None, blob_rad=8, root = False):
        """
        :param dir_path:
        :param no_of_classes:
        :param channel_lut:
        :param blob_rad: Try to keep odd
        """
        super(CudaVisionDataset, self).__init__()
        if root:
            self.img_paths, self.annot_paths, self.annot_formats = read_files_root(dir_path)
        else:
            self.img_paths, self.annot_paths, self.annot_formats = read_files(dir_path)
        self.no_of_classes = no_of_classes
        if channel_lut is None:
            channel_lut = {'Head': 0, 'Hand': 1, 'Leg': 2, 'Trunk': 3}
        self.channel_lut = channel_lut
        self.blob_rad = blob_rad

    def __getitem__(self, index):
        # print(index)
        img = cv2.imread(self.img_paths[index], 1)
        # print(img.shape) # Shape is l x w x c
        old_ratio = [img.shape[0], img.shape[1]]
        img = cv2.resize(img, dsize=(640, 480))
        l, w, c = img.shape
        l = l / 4
        w = w / 4
        # print(img.shape)
        # cv2.imshow('image', img)
        # cv2.waitKey()
        annot = parse_annotations(self.annot_paths[index], self.annot_formats[index])

        targets = np.zeros((self.no_of_classes, int(l), int(w)), dtype='float')
        for k in annot.keys():
            # print(annot[k])
            for p in annot[k]:
                targets[self.channel_lut[k], int(p[0] * (480 / old_ratio[0]) * 0.25), int(
                    p[1] * (640 / old_ratio[1]) * 0.25)] = 1.0

        # print(targets.shape)
        targets = torch.Tensor(targets)
        # downsampler = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        # targets = downsampler(targets).numpy()
        # print(targets.shape)

        gaussian_2d = define_2d_gaussian(rad=self.blob_rad)

        target_coords_downsampled = np.where(targets == 1)
        # print(np.where(targets==1))
        for i in range(target_coords_downsampled[0].shape[0]):
            targets[target_coords_downsampled[0][i]] = centre_and_place(targets[target_coords_downsampled[0][i]],
                                                                        gaussian_2d, self.blob_rad,
                                                                        (target_coords_downsampled[1][i],
                                                                         target_coords_downsampled[2][i]))

        img = np.moveaxis(img, 2, 0)  # cv2 images are l X w X c

        img = torch.Tensor(img)
        targets = torch.Tensor(targets)

        return img, targets

    def __len__(self):
        return len(self.img_paths)


def centre_and_place(arr, g, rad, coords):
    lt = int(rad / 2)
    rt = int(rad / 2) + 1
    if rad % 2 == 0:
        rt -= 1
    # print(max(0,coords[0]-lt), min(arr.shape[0],coords[0]+rt),
    #     max(0,coords[1]-lt), min(arr.shape[1],coords[1]+rt))

    #     if arr[max(0,coords[0]-lt): min(arr.shape[0],coords[0]+rt),
    #     max(0,coords[1]-lt): min(arr.shape[1],coords[1]+rt)].shape != (8,8):
    #       print(arr[max(0,coords[0]-lt): min(arr.shape[0],coords[0]+rt),
    #     max(0,coords[1]-lt): min(arr.shape[1],coords[1]+rt)])

    x = abs(max(0, coords[0] - lt) - min(arr.shape[0], coords[0] + rt))
    y = abs(max(0, coords[1] - lt) - min(arr.shape[1], coords[1] + rt))
    #     print(g.shape, x, y)

    arr[max(0, coords[0] - lt): min(arr.shape[0], coords[0] + rt),
    max(0, coords[1] - lt): min(arr.shape[1], coords[1] + rt)] = torch.Tensor(g[0:x, 0:y])

    return arr


def define_2d_gaussian(rad=5, mu=0.0, sigma=8.0):
    x, y = np.meshgrid(np.linspace(-1, 1, rad), np.linspace(-1, 1, rad))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    #     if g.shape != (8,8):
    #       print(g.shape)
    #       print(x)
    #       print(y)
    return g

def parse_annotations(fname, annot_format='.json'):
    if annot_format == 'json':
        with open(fname) as f:
            data = json.load(f)

        data = data['shapes']
        dpoints = {'Head': np.array([]), 'Hand': np.array([]), 'Leg': np.array([]), 'Trunk': np.array([])}
        for d in data:
            # print(d)
            label = d['label']
            bbox_coords = np.array(d['points'])
            mid = (bbox_coords[0] + bbox_coords[1]) / 2
            # print(bbox_coords, mid)
            dpoints[label] = np.append(dpoints[label], mid[::-1], axis=0)

        for k in dpoints.keys():
            dpoints[k] = dpoints[k].reshape((-1, 2))

        return dpoints

    elif annot_format == 'xml':
        xml_data = ET.parse(fname).getroot()

        dpoints = {'Head': np.array([]), 'Hand': np.array([]), 'Leg': np.array([]), 'Trunk': np.array([])}
        for group in xml_data.findall('object'):
            bndvalues = group.find('bndbox')
            label = group.find('name').text
            if label == 'Body':
                label = 'Trunk'
            elif label == 'Foot':
                label = 'Leg'
            bbox_coords = np.array([[int(bndvalues.find('xmin').text), int(bndvalues.find('ymin').text)],
                                   [int(bndvalues.find('xmax').text), int(bndvalues.find('ymax').text)]])
            mid = (bbox_coords[0] + bbox_coords[1]) / 2

            dpoints[label] = np.append(dpoints[label], mid[::-1], axis=0)

        for k in dpoints.keys():
            dpoints[k] = dpoints[k].reshape((-1, 2))

        return dpoints
def read_files_root(dir_path, img_format='.jpg'):
    img_paths, annot_paths, annot_formats =[],[],[]
    for r, dirs, f in os.walk(dir_path):
        for dir in dirs:

            img, annot, formats = read_files(dir_path +r'/'+ dir)
            img_paths +=img
            annot_paths+=annot
            annot_formats+=formats
        pass
    return img_paths, annot_paths, annot_formats




def read_files(dir_path, img_format='.jpg'):
    img_paths = []
    annot_paths = []
    annot_formats = []
    if os.path.isdir(dir_path + "/Images"):
        img_dir_path = "\\Images"
        print("Folder exists. Reading..")
    elif os.path.isdir(dir_path + "/images"):
        img_dir_path = "\\images"
        print("Folder exists. Reading..")
    elif os.path.isdir(dir_path + "/image"):
        img_dir_path = "\\image"
        print("Folder exists. Reading..")
    else:
        print('Image folder error in {}')
        print (dir_path)
        return [],[],[]

    for r, _, f in os.walk(dir_path + img_dir_path):
        for file in f:
            if img_format in file:
                img_paths.append(os.path.join(r, file))

    if len(img_paths) == 0:
        print("No Images in given path available. Check directory or format.")

    del_index = []
    for index, f in enumerate(img_paths):
        f_split = f.split("\\")
        if os.path.isdir(dir_path + "/json"):
            f_split[-1] = f_split[-1].replace(img_format, ".json")
            f_split[-2] = "json"
            annot_formats.append("json")
        elif os.path.isdir(dir_path + "/xml"):
            f_split[-1] = f_split[-1].replace(img_format, ".xml")
            f_split[-2] = "xml"
            annot_formats.append("xml")
        elif os.path.isdir(dir_path + "/annotations_recent"):
            f_split[-1] = f_split[-1].replace(img_format, ".xml")
            f_split[-2] = "annotations_recent"
            annot_formats.append("xml")

        else:
            print("Annotation folder not found")
            break
        # print(f_split)
        annot_path =  os.path.join(*f_split)

        if os.path.exists(annot_path):
            annot_paths.append(annot_path)
        else:
            print(annot_path)
            print("{} does not exist. Please verify.")
            exit(1)
            del_index.append(index)

    return img_paths, annot_paths, annot_formats


if __name__ == "__main__":
    # read_files('./data/Images')
    # parse_annotations('./igus Humanoid Open Platform 331.json')

    dataset = CudaVisionDataset(r'.\data',root = True)
    # dataset = CudaVisionDataset('./data/')
    # for i in enumerate(dataset):
    #     print(0)
