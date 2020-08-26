#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 14:03
# @Author  : CHT
# @Blog    : https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Site    : 
# @File    : transfer_A_C_0825.py
# @Function:  A->C
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
from collections import *
import params
import time
from utils import *
from torchvision.utils import save_image

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


def train(discriminator, classifier, dataloader_src, dataloader_tgt, train_epochs, domain_label, loss_weight):
    """
    :param domain_label:
    :param train_epochs:
    :param discriminator:
    :param classifier:
    :param dataloader_src:
    :param dataloader_tgt:
    """

    discriminator.train()
    classifier.train()
    loss_d = torch.nn.BCELoss()
    loss_c = torch.nn.CrossEntropyLoss()
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=3000)
    optimizer_c = OptimWithSheduler(optim.Adam(classifier.parameters(), weight_decay=5e-4, lr=1e-5),
                                      scheduler)
    optimizer_d = OptimWithSheduler(optim.Adam(discriminator.parameters(), weight_decay=5e-4, lr=1e-5),
                                                scheduler)

    # 各种loss和距离，保存下来画图
    sample_distance_src = []
    sample_distance_tgt = []

    discriminator_loss = []
    discriminator_f_loss = []
    classifier_loss = []
    classifier_f_loss = []

    epochs = []

    len_dataloader = min(len(dataloader_src), len(dataloader_tgt))
    for epoch in range(train_epochs[0]):
        start = time.time()
        data_zip = enumerate(zip(dataloader_src, dataloader_tgt))
        for st, ((imgs_src, labels_src), (imgs_tgt, labels_tgt)) in data_zip:
            # adjust_lr(optimizer_c, step, optimizer_c.param_groups[0]['lr'])
            # adjust_lr(optimizer_d, step, optimizer_d.param_groups[0]['lr'])

            # =========================generate transferable examples
            feature_fooling_src = Variable(
                imgs_src.type(FloatTensor), requires_grad=True).reshape(imgs_src.shape[0], -1)
            labels_src = Variable(labels_src.type(LongTensor))
            feature_fooling_src0 = feature_fooling_src.detach()  # 保留原始图片数据，方便计算

            feature_fooling_tgt = Variable(
                imgs_tgt.type(FloatTensor), requires_grad=True).reshape(imgs_tgt.shape[0], -1)
            feature_fooling_tgt0 = feature_fooling_tgt.detach()  # 保留原始图片数据，方便计算

            for i_t in range(train_epochs[1]):
                #   Target Domain
                # 更新 feature_fooling_tgt
                discriminator.zero_grad()
                classifier.zero_grad()

                scores = discriminator(feature_fooling_tgt)
                loss_d_ = loss_d(1 - scores, torch.ones_like(scores)) - 0.1 * torch.sum(
                    (feature_fooling_tgt - feature_fooling_tgt0) * (feature_fooling_tgt - feature_fooling_tgt0))
                feature_fooling_tgt.retain_grad()
                loss_d_.backward()
                # get grad
                g = feature_fooling_tgt.grad
                feature_fooling_tgt = feature_fooling_tgt + 2*g
                # optimizer_d.step()
                # 得到更新后的图像
                feature_fooling_tgt = Variable(feature_fooling_tgt, requires_grad=True)

            for i_s in range(train_epochs[2]):
                #   Source Domain -Discriminator
                # 更新 feature_fooling_src
                discriminator.zero_grad()
                classifier.zero_grad()

                scores = discriminator(feature_fooling_src)
                loss_d_ = loss_d(scores, torch.ones_like(scores)) - 0.1 * torch.sum(
                    (feature_fooling_src - feature_fooling_src0) * (feature_fooling_src - feature_fooling_src0))
                feature_fooling_src.retain_grad()
                loss_d_.backward()

                gss = feature_fooling_src.grad
                feature_fooling_src = feature_fooling_src + 2*gss

                # optimizer_d.step()
                feature_fooling_src = Variable(feature_fooling_src, requires_grad=True)

            for i_c in range(train_epochs[3]):
                # source domain -Classifier
                # 更新后的feature_fooling_src 要能被classifier正确分类
                discriminator.zero_grad()
                classifier.zero_grad()

                pred = classifier.forward(feature_fooling_src)
                loss_c_ = loss_c(pred, labels_src) - 0.1 * torch.sum(
                    (feature_fooling_src - feature_fooling_src0) * (feature_fooling_src - feature_fooling_src0))
                loss_c_.backward()
                gs = feature_fooling_src.grad
                feature_fooling_src = feature_fooling_src + 3 * gs

                feature_fooling_src = Variable(feature_fooling_src, requires_grad=True)

            # 前向传播
            feature_src = Variable(
                imgs_src.type(FloatTensor), requires_grad=False).reshape(imgs_src.shape[0], -1)
            labels_src = Variable(labels_src.type(LongTensor))
            feature_tgt = Variable(
                imgs_tgt.type(FloatTensor), requires_grad=False).reshape(imgs_tgt.shape[0], -1)
            labels_tgt = Variable(labels_tgt.type(LongTensor))

            # classifier output
            predict_prob_src = classifier(feature_src)
            predict_prob_tgt = classifier(feature_tgt)

            # discriminator output
            domain_src = discriminator(feature_src)
            domain_tgt = discriminator(feature_tgt)
            domain_f_tgt = discriminator(feature_fooling_tgt)
            domain_f_src = discriminator(feature_fooling_src)

            # 计算loss  domain_src
            domain_label_src = Variable(FloatTensor(imgs_src.size(0), 1).fill_(domain_label[0]))
            domain_label_tgt = Variable(FloatTensor(imgs_tgt.size(0), 1).fill_(domain_label[1]))
            domain_label_f_src = Variable(FloatTensor(imgs_src.size(0), 1).fill_(domain_label[2]))
            domain_label_f_tgt = Variable(FloatTensor(imgs_tgt.size(0), 1).fill_(domain_label[3]))
            dloss_f = (loss_d(domain_f_src.detach(), domain_label_f_src) +
                       loss_d(domain_f_tgt.detach(), domain_label_f_tgt))
            dloss = loss_d(domain_src, domain_label_src) + \
                    loss_d(domain_tgt, domain_label_tgt)

            # loss_c_src = loss_c(predict_prob_src, labels_src) + loss_c(predict_prob_tgt, labels_tgt)
            loss_c_src = loss_c(predict_prob_src, labels_src)
            entropy = entropy_loss(predict_prob_tgt)

            # 这是更新后生成的图片的损失
            predict_prob_f_src = classifier(feature_fooling_src)
            predict_prob_f_tgt = classifier(feature_fooling_tgt)
            dis = torch.sum((predict_prob_f_tgt - predict_prob_tgt) *
                            (predict_prob_f_tgt - predict_prob_tgt))
            loss_c_f_src = loss_c(predict_prob_f_src, labels_src)

            with OptimizerManager([optimizer_c, optimizer_d]):
                loss = loss_weight[0] * loss_c_src + loss_weight[1] * dloss + loss_weight[2] * dloss_f + \
                       loss_weight[3] * loss_c_f_src + loss_weight[4] * dis + loss_weight[5] * entropy
                loss.backward()

            if st % 200 == 0:
                # 这里也把source domain准确率输出来看一下。
                # target domain的准确率先不输出，因为现在效果还不是很好。
                predict_prob_src, predict_prob_tgt = classifier(feature_src), classifier(feature_tgt)
                pred_src, pred_tgt = predict_prob_src.data.max(1)[1], predict_prob_tgt.data.max(1)[1]
                acc_src, acc_tgt = pred_src.eq(labels_src.data).cpu().sum(), pred_tgt.eq(labels_tgt.data).cpu().sum()
                print(
                    "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [C loss: src:{:.3f}, f_src:{:.3f}] "
                    "[D loss src:{:.3f} f_src:{:.3f}] [Acc src:{:.2%} tgt:{:.2%}]"
                    "[dis:{:.3f} entropy:{:.3f}]"
                        .format(epoch, train_epochs[0],
                                st, len_dataloader,
                                loss_c_src.item(), loss_c_f_src.item(),
                                dloss.item(), dloss_f.item(),
                                int(acc_src) / 100, int(acc_tgt) / 100,
                                dis, entropy)
                )

        if epoch % 50 == 0:
            acc_src = 0
            for (imgs_src, labels_src) in dataloader_src:
                feature_src = Variable(imgs_src.type(FloatTensor)).reshape(imgs_src.shape[0], -1)
                labels_src = Variable(labels_src.type(LongTensor))

                predict_prob_src = classifier(feature_src)
                pred_src = predict_prob_src.data.max(1)[1]
                acc_src += pred_src.eq(labels_src.data).cpu().sum()
            print('epoch={}, src_acc={}'.format(epoch, int(acc_src) / len(dataloader_src.dataset)))

        # 保存参数文件和distance,loss等信息。
        if epoch % 100 == 0 and epoch != 0:
            state = {'classifier': classifier.state_dict(), 'discriminator': discriminator.state_dict()}
            torch.save(state, '../pth/classifier_discriminator_{}.pth'.format(epoch))
    state = {'sample_distance_src': sample_distance_src, 'sample_distance_tgt': sample_distance_tgt,
             'discriminator_loss': discriminator_loss, 'discriminator_f_loss': discriminator_f_loss,
             'classifier_loss': classifier_loss, 'classifier_f_loss': classifier_f_loss,
             'epochs': epochs}
    torch.save(state, '../pth/figure.pth')


def train_generate_samples(discriminator, classifier, dataloader_src, dataloader_tgt, train_epochs, domain_label,
                           loss_weight, save_samples_epoch):
    """
    目的为了生成图片数据，其他功能暂时不理会。
    迭代100次后开始生成图片
    """
    os.makedirs('../generate_samples/A_C', exist_ok=True)

    discriminator.train()
    classifier.train()
    loss_d = torch.nn.BCELoss()
    loss_c = torch.nn.CrossEntropyLoss()
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=3000)
    optimizer_c = OptimWithSheduler(optim.Adam(classifier.parameters(), weight_decay=5e-4, lr=1e-5),
                                      scheduler)
    optimizer_d = OptimWithSheduler(optim.Adam(discriminator.parameters(), weight_decay=5e-4, lr=1e-5),
                                                scheduler)

    len_dataloader = min(len(dataloader_src), len(dataloader_tgt))
    for epoch in range(train_epochs[0]):
        start = time.time()
        data_zip = enumerate(zip(dataloader_src, dataloader_tgt))
        for st, ((imgs_src, labels_src), (imgs_tgt, labels_tgt)) in data_zip:
            # adjust_lr(optimizer_c, step, optimizer_c.param_groups[0]['lr'])
            # adjust_lr(optimizer_d, step, optimizer_d.param_groups[0]['lr'])

            # =========================generate transferable examples
            feature_fooling_src = Variable(
                imgs_src.type(FloatTensor), requires_grad=True).reshape(imgs_src.shape[0], -1)
            labels_src = Variable(labels_src.type(LongTensor))
            feature_fooling_src0 = feature_fooling_src.detach()  # 保留原始图片数据，方便计算

            feature_fooling_tgt = Variable(
                imgs_tgt.type(FloatTensor), requires_grad=True).reshape(imgs_tgt.shape[0], -1)
            feature_fooling_tgt0 = feature_fooling_tgt.detach()  # 保留原始图片数据，方便计算

            for i_t in range(train_epochs[1]):
                #   Target Domain
                # 更新 feature_fooling_tgt
                discriminator.zero_grad()
                classifier.zero_grad()

                scores = discriminator(feature_fooling_tgt)
                loss_d_ = loss_d(1 - scores, torch.ones_like(scores)) - 0.1 * torch.sum(
                    (feature_fooling_tgt - feature_fooling_tgt0) * (feature_fooling_tgt - feature_fooling_tgt0))
                feature_fooling_tgt.retain_grad()
                loss_d_.backward()
                # get grad
                g = feature_fooling_tgt.grad
                feature_fooling_tgt = feature_fooling_tgt + 2*g
                # optimizer_d.step()
                # 得到更新后的图像
                feature_fooling_tgt = Variable(feature_fooling_tgt, requires_grad=True)
            tgt_imgs_f = feature_fooling_tgt.reshape(feature_fooling_tgt.shape[0], 3, params.imgs_size, params.imgs_size)

            for i_s in range(train_epochs[2]):
                #   Source Domain -Discriminator
                # 更新 feature_fooling_src
                discriminator.zero_grad()
                classifier.zero_grad()

                scores = discriminator(feature_fooling_src)
                loss_d_ = loss_d(scores, torch.ones_like(scores)) - 0.1 * torch.sum(
                    (feature_fooling_src - feature_fooling_src0) * (feature_fooling_src - feature_fooling_src0))
                feature_fooling_src.retain_grad()
                loss_d_.backward()

                gss = feature_fooling_src.grad
                feature_fooling_src = feature_fooling_src + 2*gss

                # optimizer_d.step()
                feature_fooling_src = Variable(feature_fooling_src, requires_grad=True)
            src_imgs_c = feature_fooling_src.reshape(feature_fooling_src.shape[0], 3, params.imgs_size, params.imgs_size)

            for i_c in range(train_epochs[3]):
                # source domain -Classifier
                # 更新后的feature_fooling_src 要能被classifier正确分类
                discriminator.zero_grad()
                classifier.zero_grad()

                pred = classifier.forward(feature_fooling_src)
                loss_c_ = loss_c(pred, labels_src) - 0.1 * torch.sum(
                    (feature_fooling_src - feature_fooling_src0) * (feature_fooling_src - feature_fooling_src0))
                loss_c_.backward()
                gs = feature_fooling_src.grad
                feature_fooling_src = feature_fooling_src + 3 * gs

                feature_fooling_src = Variable(feature_fooling_src, requires_grad=True)
            src_imgs_d = feature_fooling_src.reshape(feature_fooling_src.shape[0], 3, params.imgs_size, params.imgs_size)

            # 前向传播
            feature_src = Variable(
                imgs_src.type(FloatTensor), requires_grad=False).reshape(imgs_src.shape[0], -1)
            labels_src = Variable(labels_src.type(LongTensor))
            feature_tgt = Variable(
                imgs_tgt.type(FloatTensor), requires_grad=False).reshape(imgs_tgt.shape[0], -1)
            labels_tgt = Variable(labels_tgt.type(LongTensor))

            # classifier output
            predict_prob_src = classifier(feature_src)
            predict_prob_tgt = classifier(feature_tgt)

            # discriminator output
            domain_src = discriminator(feature_src)
            domain_tgt = discriminator(feature_tgt)
            domain_f_tgt = discriminator(feature_fooling_tgt)
            domain_f_src = discriminator(feature_fooling_src)

            # 计算loss  domain_src
            domain_label_src = Variable(FloatTensor(imgs_src.size(0), 1).fill_(domain_label[0]))
            domain_label_tgt = Variable(FloatTensor(imgs_tgt.size(0), 1).fill_(domain_label[1]))
            domain_label_f_src = Variable(FloatTensor(imgs_src.size(0), 1).fill_(domain_label[2]))
            domain_label_f_tgt = Variable(FloatTensor(imgs_tgt.size(0), 1).fill_(domain_label[3]))
            dloss_f = (loss_d(domain_f_src.detach(), domain_label_f_src) +
                       loss_d(domain_f_tgt.detach(), domain_label_f_tgt))
            dloss = loss_d(domain_src, domain_label_src) + \
                    loss_d(domain_tgt, domain_label_tgt)

            # loss_c_src = loss_c(predict_prob_src, labels_src) + loss_c(predict_prob_tgt, labels_tgt)
            loss_c_src = loss_c(predict_prob_src, labels_src)
            entropy = entropy_loss(predict_prob_tgt)

            # 这是更新后生成的图片的损失
            predict_prob_f_src = classifier(feature_fooling_src)
            predict_prob_f_tgt = classifier(feature_fooling_tgt)
            dis = torch.sum((predict_prob_f_tgt - predict_prob_tgt) *
                            (predict_prob_f_tgt - predict_prob_tgt))
            loss_c_f_src = loss_c(predict_prob_f_src, labels_src)

            with OptimizerManager([optimizer_c, optimizer_d]):
                loss = loss_weight[0] * loss_c_src + loss_weight[1] * dloss + loss_weight[2] * dloss_f + \
                       loss_weight[3] * loss_c_f_src + loss_weight[4] * dis + loss_weight[5] * entropy
                loss.backward()

        # if (epoch+1) % 10 == 0:
        #     # 这里也把source domain准确率输出来看一下。
        #     # target domain的准确率先不输出，因为现在效果还不是很好。
        #     predict_prob_src, predict_prob_tgt = classifier(feature_src), classifier(feature_tgt)
        #     pred_src, pred_tgt = predict_prob_src.data.max(1)[1], predict_prob_tgt.data.max(1)[1]
        #     acc_src, acc_tgt = pred_src.eq(labels_src.data).cpu().sum(), pred_tgt.eq(labels_tgt.data).cpu().sum()
        #     print(
        #         "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [C loss: src:{:.3f}, f_src:{:.3f}] "
        #         "[D loss src:{:.3f} f_src:{:.3f}] [Acc src:{:.2%} tgt:{:.2%}]"
        #         "[dis:{:.3f} entropy:{:.3f}]"
        #             .format(epoch, train_epochs[0],
        #                     st, len_dataloader,
        #                     loss_c_src.item(), loss_c_f_src.item(),
        #                     dloss.item(), dloss_f.item(),
        #                     int(acc_src) / 100, int(acc_tgt) / 100,
        #                     dis, entropy)
        #     )

        if epoch >= save_samples_epoch:
            # Save samples
            # 总共五个部分: src_imgs, src_f_imgs_d, src_f_imgs_c, tgt_imgs, tgt_f_imgs
            src_imgs = feature_src.reshape(feature_src.shape[0], 3, params.imgs_size, params.imgs_size)
            tgt_imgs = feature_tgt.reshape(feature_tgt.shape[0], 3, params.imgs_size, params.imgs_size)
            generate_imgs = torch.cat((src_imgs[10:30], src_imgs_d[10:30], src_imgs_c[10:30],
                                       tgt_imgs[10:30], tgt_imgs_f[10:30]), 0)
            save_image(generate_imgs, '../generate_samples/A_C/{}.png'.format(epoch), nrow=20, normalize=True)
            print('epoch = {} Save samples!'.format(epoch))


def evaluate(classifier, dataloader_src, dataloader_tgt):
    # 只需用到dataloader tgt

    classifier.eval()
    acc_src = acc_tgt = 0
    for (imgs_tgt, labels_tgt) in dataloader_tgt:
        feature_tgt = Variable(imgs_tgt.type(FloatTensor).expand(
            imgs_tgt.shape[0], 3, params.imgs_size, params.imgs_size), requires_grad=False).reshape(imgs_tgt.shape[0], -1)
        labels_tgt = Variable(labels_tgt.type(LongTensor))

        predict_prob_tgt = classifier(feature_tgt)
        pred_tgt = predict_prob_tgt.data.max(1)[1]
        acc_tgt += pred_tgt.eq(labels_tgt.data).cpu().sum()

    for (imgs_src, labels_src) in dataloader_src:

        feature_src = Variable(imgs_src.type(FloatTensor)).reshape(imgs_src.shape[0], -1)
        labels_src = Variable(labels_src.type(LongTensor))

        predict_prob_src = classifier(feature_src)
        pred_src = predict_prob_src.data.max(1)[1]
        acc_src += pred_src.eq(labels_src.data).cpu().sum()
    acc_src = int(acc_src) / len(dataloader_src.dataset)
    acc_tgt = int(acc_tgt) / len(dataloader_tgt.dataset)
    print("Src Accuracy = {:2%}, Tgt Accuracy = {:2%}".format(acc_src, acc_tgt))


def train_classifier(classifier, dataloader_src):
    # 单独测试一下分类器效果
    optimizer = optim.Adam(classifier.parameters(), lr=params.learning_rate,
                             betas=(params.beta1, params.beta2))
    loss_c = nn.CrossEntropyLoss()
    for epoch in range(params.classifier_epochs):
        data_zip = enumerate(dataloader_src)
        for step, (imgs_src, labels_src) in data_zip:
            # adjust_lr(optimizer, step, optimizer.param_groups[0]['lr'])
            feature_src = Variable(imgs_src.type(FloatTensor), requires_grad=False).reshape(imgs_src.shape[0],
                                                                                                   -1)
            labels_src = Variable(labels_src.type(LongTensor))
            # feature_tgt = Variable(imgs_tgt.type(FloatTensor), requires_grad=True).reshape(imgs_tgt.shape[0],
            #                                                                                -1)
            # labels_tgt = Variable(labels_tgt.type(LongTensor))
            # with OptimizerManager([optimizer]):
            #     loss = loss_c(classifier(feature_src), labels_src)
            #     loss.backward()
            optimizer.zero_grad()
            loss = loss_c(classifier(feature_src), labels_src)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            predict_prob_src = classifier(feature_src)
            pred_src = predict_prob_src.data.max(1)[1]
            acc_src = pred_src.eq(labels_src.data).cpu().sum()
            print('acc:{:.3%}'.format(int(acc_src) / imgs_src.shape[0]))
    torch.save(classifier.state_dict(), '../pth/classifier_src.pth')


if __name__ == '__main__':
    from models import networks
    from Images import data_preprocess

    from torch.backends import cudnn

    torch.backends.cudnn.benchmark = True

    # get dataloader
    amazon_path = os.path.join(params.imgs_root_path, r'amazon\images')
    dslr_path = os.path.join(params.imgs_root_path, r'dslr\images')
    webcam_path = os.path.join(params.imgs_root_path, r'webcam\images')
    caltech_path = os.path.join(params.imgs_root_path, r'Clatech\clatech')

    # 不使用target domain
    amazon_dataloader = data_preprocess.get_dataloader(amazon_path, params.images_name)
    dslr_dataloader = data_preprocess.get_dataloader(dslr_path, params.images_name)
    webcam_dataloader = data_preprocess.get_dataloader(webcam_path, params.images_name)
    caltech_dataloader = data_preprocess.get_dataloader(caltech_path, params.images_name)
    # 目标域带标签的
    # amazon_dataloader, dslr_dataloader = data_preprocess.get_src_tgt_dataloader(amazon_path, dslr_path, params.images_name)

    # 初始化网络
    classifier = networks.Classifier(3*params.imgs_size*params.imgs_size, len(params.images_name)).cuda()
    discriminator = networks.LargeDiscriminator(3*params.imgs_size*params.imgs_size).cuda()

    # 定义训练的次数：总迭代次数，tgt_discriminator, src_discriminator, src_classifier
    train_epochs = [150, 20, 20, 20]
    # domain label
    domain_label = [1.0, 0.0, 1.0, 0.0]
    # 各种loss的权重  loss_c_src + dloss + dloss_f + loss_c_f_src + dis + entropy
    loss_weight = [1, 0.5, 0.5, 1.0, 1.0, 0.1]
    save_samples_epoch = 0

    train(discriminator, classifier, amazon_dataloader, dslr_dataloader, train_epochs, domain_label, loss_weight)
    # train_generate_samples(discriminator, classifier, amazon_dataloader, caltech_dataloader, train_epochs, domain_label,
    #                        loss_weight, save_samples_epoch)




