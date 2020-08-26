#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 13:59
# @Author  : CHT
# @Blog    : https://www.zhihu.com/people/xia-gan-yi-dan-chen-hao-tian
# @Site    : 
# @File    : transfer_820_1.py
# @Function: 主要关注learning rate
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


def train(discriminator, classifier, dataloader_src, dataloader_tgt, train_epochs, domain_label):
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

    # optimizer_c = optim.Adam(classifier.parameters(), lr=params.learning_rate,
    #                          betas=(params.beta1, params.beta2), weight_decay=5e-4)
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=params.learning_rate,
    #                          betas=(params.beta1, params.beta2), weight_decay=5e-4)
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
        for step1, ((imgs_src, labels_src), (imgs_tgt, labels_tgt)) in data_zip:
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
                #   Source Domain
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
                #  上面都是 discriminator，接下来是classifier,也是针对source domain的，source domain由三部分组成。
                # 更新后的feature_fooling_src 要能被classifier正确分类
                discriminator.zero_grad()
                classifier.zero_grad()

                pred = classifier.forward(feature_fooling_src)
                loss_c_ = loss_c(pred, labels_src) - 0.1 * torch.sum(
                    (feature_fooling_src - feature_fooling_src0) * (feature_fooling_src - feature_fooling_src0))
                loss_c_.backward()
                gs = feature_fooling_src.grad
                feature_fooling_src = feature_fooling_src + 3 * gs

                # optimizer_c.step()
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
            # dloss_f = loss_d(domain_f_src.detach(), torch.ones_like(domain_f_src)) +\
            #           loss_d(1 - domain_f_tgt.detach(), torch.ones_like(domain_f_tgt))
            # dloss = loss_d(domain_src, torch.ones_like(domain_src)) + \
            #         loss_d(1 - domain_tgt, torch.ones_like(domain_tgt))
            dloss_f = (loss_d(domain_f_src.detach(), domain_label_f_src) +
                       loss_d(domain_f_tgt.detach(), domain_label_f_tgt))
            dloss = loss_d(domain_src, domain_label_src) + \
                    loss_d(domain_tgt, domain_label_tgt)

            # loss_c_src = loss_c(predict_prob_src, labels_src) + loss_c(predict_prob_tgt, labels_tgt)
            loss_c_src = loss_c(predict_prob_src, labels_src)
            # 这里有个entropy的计算，是更新后面的Optimizer的，这里略过先。
            # target domain只用计算entropy，不计算class.
            # 所以说target domain的信息也用上了。
            entropy = entropy_loss(predict_prob_tgt)

            # 这是更新后生成的图片的损失
            predict_prob_f_src = classifier(feature_fooling_src)
            predict_prob_f_tgt = classifier(feature_fooling_tgt)
            dis = torch.sum((predict_prob_f_tgt - predict_prob_tgt) *
                            (predict_prob_f_tgt - predict_prob_tgt))
            loss_c_f_src = loss_c(predict_prob_f_src, labels_src)

            with OptimizerManager([optimizer_c, optimizer_d]):
                loss = loss_c_src + 0.5 * dloss + 0.5 * dloss_f + loss_c_f_src + dis + 0.1 * entropy
                loss.backward()

            if step1 % 200 == 0:
                # 这里也把准确率输出来看一下.
                predict_prob_src, predict_prob_tgt = classifier(feature_src), classifier(feature_tgt)
                pred_src, pred_tgt = predict_prob_src.data.max(1)[1], predict_prob_tgt.data.max(1)[1]
                acc_src, acc_tgt = pred_src.eq(labels_src.data).cpu().sum(), pred_tgt.eq(labels_tgt.data).cpu().sum()
                print(
                    "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [C loss: src:{:.3f}, f_src:{:.3f}] "
                    "[D loss src:{:.3f} f_src:{:.3f}] [Acc src:{:.2%} tgt:{:.2%}]"
                    "[dis:{:.3f} entropy:{:.3f}]"
                        .format(epoch, train_epochs[0],
                                step1, len_dataloader,
                                loss_c_src.item(), loss_c_f_src.item(),
                                dloss.item(), dloss_f.item(),
                                int(acc_src) / 100, int(acc_tgt) / 100,
                                dis, entropy)
                )

        if epoch % 5 == 0:
            acc_src = 0
            for (imgs_src, labels_src) in dataloader_src:
                feature_src = Variable(imgs_src.type(FloatTensor)).reshape(imgs_src.shape[0], -1)
                labels_src = Variable(labels_src.type(LongTensor))

                predict_prob_src = classifier(feature_src)
                pred_src = predict_prob_src.data.max(1)[1]
                acc_src += pred_src.eq(labels_src.data).cpu().sum()
            print('epoch={}, acc={}'.format(epoch, int(acc_src) / len(dataloader_src.dataset)))

            # Save samples
        if epoch >= 1000:
            # Save samples
            # 总共五个部分: src_imgs, src_f_imgs_d, src_f_imgs_c, tgt_imgs, tgt_f_imgs
            # tgt_imgs_f = feature_fooling_tgt.reshape(feature_fooling_tgt.shape[0], 3, params.imgs_size,
            #                                          params.imgs_size)
            src_imgs = feature_src.reshape(feature_src.shape[0], 3, params.imgs_size, params.imgs_size)
            tgt_imgs = feature_tgt.reshape(feature_tgt.shape[0], 3, params.imgs_size, params.imgs_size)
            generate_imgs = torch.cat((src_imgs[10:30], src_imgs_d[10:30], src_imgs_c[10:30],
                                       tgt_imgs[10:30], tgt_imgs_f[10:30]), 0)
            print(generate_imgs.shape)
            save_image(generate_imgs, '../generate_samples/{}.png'.format(epoch), nrow=20, normalize=True)

            # Distance
            dis_src = torch.sum(
                (feature_fooling_src - feature_fooling_src0) * (feature_fooling_src - feature_fooling_src0))
            dis_tgt = torch.sum(
                (feature_fooling_tgt - feature_fooling_tgt0) * (feature_fooling_tgt - feature_fooling_tgt0))
            sample_distance_src.append(dis_src.item())
            sample_distance_tgt.append(dis_tgt.item())
            # Loss
            discriminator_loss.append(dloss.item())
            discriminator_f_loss.append(dloss_f.item())
            classifier_loss.append(loss_c_src.item())
            classifier_f_loss.append(loss_c_f_src.item())

            epochs.append(epoch)

        print('{:.3f}s'.format(time.time() - start))

        if epoch % 100 == 0 and epoch != 0:
            state = {'classifier': classifier.state_dict(), 'discriminator': discriminator.state_dict()}
            torch.save(state, '../pth/classifier_discriminator_{}.pth'.format(epoch))
    state = {'sample_distance_src': sample_distance_src, 'sample_distance_tgt': sample_distance_tgt,
             'discriminator_loss': discriminator_loss, 'discriminator_f_loss': discriminator_f_loss,
             'classifier_loss': classifier_loss, 'classifier_f_loss': classifier_f_loss,
             'epochs': epochs}
    torch.save(state, '../pth/figure.pth')


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
    import utils
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
    # dslr_dataloader = data_preprocess.get_dataloader(dslr_path, params.images_name)
    caltech_dataloader = data_preprocess.get_dataloader(caltech_path, params.images_name)

    # 目标域带标签的
    # amazon_dataloader, dslr_dataloader = data_preprocess.get_src_tgt_dataloader(amazon_path, dslr_path, params.images_name)
    # 数据均衡后的dataloader
    # amazon_dataloader, dslr_dataloader = data_preprocess.data_balance_dataloader(amazon_path, dslr_path, params.images_name)

    # 初始化网络
    classifier = networks.Classifier(3*params.imgs_size*params.imgs_size, len(params.images_name)).cuda()
    discriminator = networks.LargeDiscriminator(3*params.imgs_size*params.imgs_size).cuda()

    # if os.path.exists('../pth/classifier_src.pth'):
    #     classifier.load_state_dict(torch.load('../pth/classifier_src.pth'))
    # else:
    #     train_classifier(classifier, dslr_dataloader)

    # 定义训练的次数：总迭代次数，tgt_discriminator, src_discriminator, src_classifier
    train_epochs = [1000, 20, 20, 20]
    domain_label = [1, 0.0, 1, 0]
    print(len(amazon_dataloader.dataset), len(caltech_dataloader.dataset))
    train(discriminator, classifier, amazon_dataloader, caltech_dataloader, train_epochs, domain_label)
    # classifier.load_state_dict(torch.load('./pth/net.pth')['classifier'])
    # evaluate(classifier, svhn_data_loader, mnist_data_loader)




