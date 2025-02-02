import sys

sys.path.append("E:/Vision-Systems-Lab/Project9/")

from model import *
from utils import *
from train import *
#from google.colab.patches import cv2_imshow


import numpy as np
from torch.utils.data import dataloader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import time
import random

import matplotlib
import matplotlib.pyplot as plt
torch.manual_seed(0)

def acc(dataloader, itr, tresh=4, gray_thresh=0.4):
    """
    Calculate accuracy of predictions from model for dataloader.
    :param gray_thresh:
    :param tresh:
    :param dataloader: dataloader to evaluate
    :return:
    """
    acc = 0.0
    true_y = []
    pred_y = []
    total = 0.0
    model.eval()
    f_p = np.zeros(4)  # False Positive
    f_n = np.zeros(4)  # False Negative
    true = np.zeros(4)
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            # y = F.pad(y, (4, 5, 7, 6, 0, 0, 0, 0), mode='constant', value=0)

            preds = model(x).cpu().numpy()

            for b_id in range(dataloader.batch_size):
                acc_chan = np.zeros(preds.shape[1])

                for chan in range(preds.shape[1]):

                    # Erosion
                    kernel = np.ones((3, 3), np.uint8)
                    (_, preds_thresh) = cv2.threshold(preds[b_id, chan], gray_thresh, 255, 0)
                    preds_erosion = cv2.erode(preds_thresh, kernel, iterations=1)

                    # Dilation
                    preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)

                    image, contours_p, _ = cv2.findContours(preds_dilation.astype(np.uint8), cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                    contours_poly = [None] * len(contours_p)
                    boundRect_p = [None] * len(contours_p)
                    for i, c in enumerate(contours_p):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        boundRect_p[i] = cv2.boundingRect(contours_poly[i])

                    image, contours_t, _ = cv2.findContours(np.array((y.cpu())[0, chan] * 255).astype(np.uint8),
                                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_poly = [None] * len(contours_t)
                    boundRect_t = [None] * len(contours_t)
                    for i, c in enumerate(contours_t):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        boundRect_t[i] = cv2.boundingRect(contours_poly[i])

                    used = np.zeros(len(boundRect_t))
                    for i in range(len(boundRect_p)):

                        found = -1

                        for k in range(len(boundRect_t)):
                            x_t = min(boundRect_t[k][0], boundRect_t[k][1]) + abs(
                                (boundRect_t[k][0] - boundRect_t[k][1])) / 2
                            y_t = min(boundRect_t[k][2], boundRect_t[k][3]) + abs(
                                (boundRect_t[k][2] - boundRect_t[k][3])) / 2

                            x_p = min(boundRect_p[i][0], boundRect_p[i][1]) + abs(
                                (boundRect_p[i][0] - boundRect_p[i][1])) / 2
                            y_p = min(boundRect_p[i][2], boundRect_p[i][3]) + abs(
                                (boundRect_p[i][2] - boundRect_p[i][3])) / 2

                            if (
                                    abs(x_t - x_p) < tresh and
                                    abs(y_t - y_p) < tresh):
                                found = k
                                true[chan] += 1
                                # break

                        if found == -1:
                            f_p[chan] += 1
                        else:
                            used[found] = 1
                    f_n[chan] += np.count_nonzero(used == 0)
                    # acc_chan[chan] = (true + 0.001) / ((true + f_n + f_p) + 0.001)

                # acc += acc_chan.sum() / acc_chan.size
                # total += 1

        acc = np.average(true) / (np.average(true) + np.average(f_n) + np.average(f_p))
    return true_y, pred_y, acc, true, f_p, f_n

def picture(dataloader, itr):
    acc = 0.0
    true_y = []
    pred_y = []
    total = 0.0
    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dataloader):
            if (batch_id == 1):
                x = x.cuda()
                y = y.cuda()

                drawing_t = x[0].cpu().numpy().astype('uint8')
                drawing_p = x[0].cpu().numpy().astype('uint8')
                drawing_t = np.moveaxis(drawing_t, 0, 2).copy()
                drawing_p = np.moveaxis(drawing_p, 0, 2).copy()
                # cv2.imshow('before_1',drawing_t)


                for chan in range(4):
                    preds = np.array(model(x).cpu()[0][chan])
                    targets = np.array(y.cpu()[0][chan])

                    # (thresh, preds) = cv2.threshold(preds, 0.4, 255, 0)

                    kernel = np.ones((3, 3), np.uint8)
                    # # Erosion
                    #
                    # (_, preds_thresh) = cv2.threshold(preds, 0.4, 255, 0)
                    # preds_erosion = cv2.erode(preds_thresh, kernel, iterations=1)
                    #
                    # # Dilation
                    # preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)

                    # Contour Detection


                    image, contours_p, _ = cv2.findContours((preds).astype(np.uint8), cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                    contours_poly = [None] * len(contours_p)
                    boundRect_p = [None] * len(contours_p)
                    centers_p = [None] * len(contours_p)
                    radius_p = [None] * len(contours_p)

                    for i, c in enumerate(contours_p):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        centers_p[i], radius_p[i] = cv2.minEnclosingCircle(contours_poly[i])

                    for i in range(len(boundRect_p)):
                        cv2.circle(drawing_p, (int(centers_p[i][0] * 4), int(centers_p[i][1] * 4)), int(8), (30, 255, 255), -1)




                    image, contours_t, _ = cv2.findContours(np.array((y.cpu())[0, chan] * 255).astype(np.uint8),
                                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    contours_poly = [None] * len(contours_t)
                    boundRect_t = [None] * len(contours_t)
                    centers_t = [None] * len(contours_t)
                    radius_t = [None] * len(contours_t)

                    for i, c in enumerate(contours_t):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        centers_t[i], radius_t[i] = cv2.minEnclosingCircle(contours_poly[i])

                    for i in range(len(boundRect_t)):
                        cv2.circle(drawing_p, (int(centers_t[i][0] * 4), int(centers_t[i][1] * 4)), int(6), (255, 0, 0),
                                   -1)

                    for i in range(len(centers_p)):
                        print(itr, chan, centers_t[i][0] - centers_p[i][0], centers_t[i][1] - centers_p[i][1])

                    # drawing_t.convertTo(result8u,CV_8U);
                #cv2.imshow('1',drawing_t)
                #cv2.imshow('2',drawing_p)
                cv2.imwrite(str(itr)+".jpg", drawing_p)
                #time.sleep(2)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                pass
batch_size = 1
model = Resnet18NimbroNet().cuda()
dataset = CudaVisionDataset(dir_path='./data/train')  # (image, target) set
for i in range(5):
    #train_split, valid_split, test_split = random_split(dataset, [1,1,3])
    train_split, valid_split, test_split = random_split(dataset, [int(len(dataset) * 0.7),
                                                                  int(len(dataset) * 0.1),
                                                                  int(len(dataset) - int(len(dataset) * 0.7) - int(
                                                                      len(dataset) * 0.1))])
    print('Data: ', [int(len(dataset) * 0.7),
                     int(len(dataset) * 0.1),
                     int(len(dataset) - int(len(dataset) * 0.7) - int(len(dataset) * 0.1))])
    train_dataloader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_split, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_split, batch_size=2, shuffle=True)
    model.load_state_dict(torch.load('model_igus.pth'))
    picture(test_dataloader,i)

    print(acc(valid_dataloader,1, gray_thresh = 0.2))
