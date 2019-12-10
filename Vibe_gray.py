import cv2
import os
import random
import numpy as np

# 用于8邻域的选取
global c_xoff, c_yoff
c_xoff = [-1, 0, 1, -1, 1, -1, 0, 1]
c_yoff = [1, 1, 1, 0, 0, -1, -1, -1]



# 初始化样本集
# 输入参数：图像，样本数
def initial_background(img, sample_num):
    # 行数
    height = img.shape[0]
    # 列数
    width = img.shape[1]
    # 创建样本集，灰度图像
    samples = np.zeros((img.shape[0], img.shape[1], sample_num))
    for i in range(height):
        for j in range(width):
            # 每个点找N个样本
            for h in range(sample_num):
                # 随机从邻域中选择样本
                rand = random.randint(0, 7)
                x = i + c_xoff[rand]
                y = j + c_yoff[rand]
                # 防止越界
                if (x < 0): x = 0
                if (y < 0): y = 0
                if (x >= height): x = height - 1
                if (y >= width): y = width - 1
                # 存储样本
                samples[i][j][h] = img[x][y]
    return samples


# 对前景/背景的分类
# 输入参数：图像，样本，样本数，匹配半径，最小匹配个数，更新因子
def classify(img, samples, sample_num, radius, min_matches):
    height = img.shape[0]
    width = img.shape[1]
    # 用于保存欧式距离
    distance = np.zeros((height, width, sample_num))
    # 计算欧式距离
    for i in range(sample_num):
        distance[:, :, i] = np.abs(img - samples[:, :, i])
    # 计算匹配个数
    count = np.where(distance < radius, 1, 0)
    count = np.sum(count, axis=-1)
    # 分割蒙版
    fg_mask = np.where(count >= min_matches, 0, 255).astype(np.uint8)
    return fg_mask


# 更新样本集
# 输入参数：当前帧，样本集，分割蒙版，样本数，更新因子
def update_background(img, samples, fg_mask, sample_num, update_factor):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            # 若为背景，则更新样本集
            if not fg_mask[i, j]:
                # 更新当前像素样本集
                rand = random.randint(0, update_factor - 1)
                if (rand == 0):
                    # 随机选择样本
                    rand = random.randint(0, sample_num - 1)
                    samples[i][j][rand] = img[i][j]
                # 更新邻域像素样本集
                rand = random.randint(0, update_factor - 1)
                if (rand == 0):
                    # 随机选择邻域像素
                    rand = random.randint(0, 7)
                    x = i + c_xoff[rand]
                    y = j + c_yoff[rand]
                    # 防止越界
                    if (x < 0): x = 0
                    if (y < 0): y = 0
                    if (x >= height): x = height - 1
                    if (y >= width): y = width - 1
                    # 随机选择样本
                    rand = random.randint(0, sample_num - 1)
                    samples[x][y][rand] = img[i][j]
    return samples


# 数据集的根目录
rootDir = '/Users/mac/数据集/dataset2012/PETS2006/input'
# 对所有文件排序
lists = sorted(os.listdir(rootDir))
# 是否初始化的标识位
flag = False
# 每个像素点的样本个数
default_sample_num = 20
# 匹配半径
default_radius = 20
# 最小匹配样本数
default_min_matches = 2
# 更新因子
default_update_factor = 16
for list in lists:
    path = os.path.join(rootDir, list)
    # 当前帧(灰度图像)
    current = cv2.imread(path, 0)
    if (flag == False):
        samples = initial_background(current, default_sample_num)
        flag = True
        continue
    # 对前景/背景的分类
    fg_mask = classify(current, samples, default_sample_num, default_radius, default_min_matches)
    # 更新样本集
    samples = update_background(current, samples, fg_mask, default_sample_num, default_update_factor)
    cv2.imshow('input', current)
    cv2.imshow('foreground', fg_mask)
    # 保存二值化图片
    cv2.imwrite("/Users/mac/数据集/dataset2012/PETS2006/output/" + list.replace("in", "out"), fg_mask)
    cv2.waitKey(1)
