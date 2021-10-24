# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# authot: Mr.Song  time:2021/9/7
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

# image_path = glob('./inputs/coronary/images/*')
# mask_path = glob('./inputs/coronary/masks/*')
image_path = './inputs/vessel/img'
mask_path = './inputs/vessel/mask'
image_size = 256

def main():
    os.makedirs('inputs/vessel_%d/images' % image_size, exist_ok=True)
    os.makedirs('inputs/vessel_%d/masks' % image_size, exist_ok=True)
    name = []
    for root, dirs, files in os.walk('./inputs/vessel/img'):
        # print(root)
        # print(dirs)
        for file in files:
            # print file.decode('gbk')    #文件名中有中文字符时转码
            if os.path.splitext(file)[1] == '.png':
                # t = os.path.splitext(file)[0]
                # print(t)  # 打印所有py格式的文件名
                name.append(file)  # 将所有的文件名添加到L列表中
    for i in tqdm(range(len(name))):
        img = cv2.imread(os.path.join(image_path, name[i]))
        # cv2.imshow('a', img)
        # cv2.waitKey(0)
        # mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.imread(os.path.join(mask_path, name[i]))
        # mask[mask_] = 1
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.resize(img, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size))
        cv2.imwrite(os.path.join('inputs/vessel_%d/images' % image_size,
                                 name[i]), img)
        cv2.imwrite(os.path.join('inputs/vessel_%d/masks' % image_size,
                                 name[i]), mask)


if __name__ == '__main__':
    main()
