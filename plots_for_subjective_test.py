#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pathlib import Path

from skimage.measure import compare_ssim as ssim
from skimage import feature    # Canny Edge 사용하기 위함

import torch
from scipy.ndimage import zoom

import scipy.io
from copy import deepcopy

from collections import defaultdict


# In[2]:


# (2, 6) subplot을 만드는 함수
def plot_img_compare(imgs, img_list): 
    '''
    imgs: list of 2d arrays (images)
    img_list: list of strings (name of the images)
    '''
    plt.figure(figsize=(50, 17))
    n = 12
    
    for i, k in enumerate(img_list):
        ax = plt.subplot(2, 6, i+1)
        plt.axis('on')
        img = imgs[i]
        im = plt.imshow(img, vmin=1.33, vmax=1.4, cmap='viridis')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        cb = plt.colorbar(im, cax=cax)
        ax.set_title(k, fontsize=15)

    plt.show()
    plt.close()


# In[3]:


def brenner_grad(img):
    return np.sum((img[:-1, :] - img[1:, :])**2)

# xy 평면 슬라이드 64개 중에서 어느것에 초점이 맞았는지 알려주는 함수
def focus_finder(img_list): 
#     print(len(img_list))
    focused = []
    for img in img_list:
        brenner_list = []
        for i in range(64):
            img_slice = np.squeeze(img[i])
#             print(np.shape(img_slice))
            brenner_list.append(brenner_grad(img_slice))
#         print(brenner_list)
        focused.append(np.argmax(brenner_list))
    return focused

# print(focus_finder(input_cropped_original))


# In[4]:


##########
# 동민오빠 #
##########

# list로 받은 이미지 전부 imshow로 한 줄에 띄워주는 함수
def plot_img(imgs, img_list):
    '''
    imgs: list of 2d arrays (images)
    img_list: list of strings (name of the images)
    '''
    plt.figure(figsize=(30, 10))
    n = len(imgs)
    
    for i, k in enumerate(img_list):
        ax = plt.subplot(1, n, i+1)
        plt.axis('on')
        img = imgs[i]
        im = plt.imshow(img, vmin=1.33, vmax=1.4, cmap='viridis')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        cb = plt.colorbar(im, cax=cax)
        ax.set_title(k, fontsize=15)

    plt.show()
    plt.close()

# list로 받은 image의 Canny edge detector output을 내고, input mask와 output mask의 차이를 더해 출력
def plot_edge(reference, imgs, img_list):
    '''
    reference: input image
    imgs: list of 2d arrays (images)
    img_list: list of strings (name of the images)
    '''
    plt.figure(figsize=(30,10))
    n = len(imgs)
    edge_val_arr = []
    reference_edge = 1 - feature.canny(norm(reference), sigma=3)
    for i, k in enumerate(img_list):
        ax = plt.subplot(1, n, i+1)
        plt.axis('off')
        img = imgs[i]
        img = feature.canny(norm(img), sigma=3)
        im = plt.imshow(img, cmap='viridis')
        divider = make_axes_locatable(ax)
        ax.set_title("Edge (sigma=3) ({})".format(k), fontsize=15)
        edge_val = np.sum(img * reference_edge)
        edge_val_arr.append(edge_val)
    plt.show()
    plt.close()
    print(edge_val_arr)

    #canny edge수치를 구하기 위해 return 값 추가
    return edge_val_arr
    
def norm(img):
    # accounting for TCF ver 1.2
    if img.dtype == 'float64':
        img = img.astype('float32')
        # img /= 10000.
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std #, mean, std

# Reference image와의 차영상 (method noise) 출력
def plot_diff(reference, imgs, img_list):
    
    '''
    reference: input image
    imgs: list of 2d arrays (images)
    img_list: list of strings (name of the images)
    '''
    plt.figure(figsize=(30, 10))
    n = len(imgs)
    for i, k in enumerate(img_list):
        ax = plt.subplot(1, n, i+1)
        plt.axis('off')
        img = imgs[i]
        im = plt.imshow(img-reference, cmap='viridis') #, vmin=-0.1, vmax=0.1, cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.03)
        cb = plt.colorbar(im, cax=cax)
        ax.set_title("method noise ({})".format(k), fontsize=15)

    plt.show()
    plt.close()



# list로 받은 image들의 특정 patch 확대 및 해당 patch에서의 RI value histogram 비교
def plot_zoom(imgs, img_list, p1=(0, 0), p2=(64, 64), range=(1.33, 1.35), additional_line=True):
    '''
    imgs: list of 2d arrays (images)
    img_list: list of strings (name of the images)
    p1: start point
    p2: end point
    '''
    n = len(imgs)
    for i, k in enumerate(img_list):
        img = imgs[i]
        fig = plt.figure(figsize=(30, 10))
        
        # Draw original image (1st plot)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title(k, fontsize=15)
        plt.axis('on')
        ax1.imshow(img, vmin=1.33, vmax=1.4, cmap='viridis')
        
        # Draw red rectangles indicating the selected region on the first plot
        (x, y) = img.shape
        (x1, y1) = p1
        (x2, y2) = p2
        bx = (x1, x2, x2, x1, x1)
        by = (y1, y1, y2, y2, y1)
        ax1.plot(by, bx, '-r', linewidth=1.5)
        
        # Draw image patch (2nd plot)
        img_patch = img[x1:x2, y1:y2]
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("image patch from ({}, {}) to ({}, {})".format(x1, y1, x2, y2), fontsize=15)
        plt.axis('on')
        ax2.imshow(img_patch, vmin=1.33, vmax=1.4, cmap='viridis')
        
        # Draw patch histogram (3rd plot)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("patch histogram", fontsize=15)
        ax3.hist(img_patch.flatten(), bins=200, range=range, density='True', histtype='bar', color=(0, 0, 0))
        
        # Draw median line on the histogram
        median_x = np.median(img_patch)
        p05_x = np.percentile(img_patch, 5)
        p95_x = np.percentile(img_patch, 95)
        if additional_line is True:
            ax3.axvline(x=median_x, linewidth=1.5, color='b')
            ax3.axvline(x=p05_x, linewidth=1.5, color='b', linestyle='--')
            ax3.axvline(x=p95_x, linewidth=1.5, color='b', linestyle='--')
            ax3.axvline(x=1.337, linewidth=1, color='r')
        plt.show()
        plt.close()


# In[5]:


################################################################
# 크기와 데이터타입이 제각각인 입력값들을 출력값과 똑같은 형식으로 맞춰주는 함수들 #
################################################################

def _center_crop(img):
    z_cropped_size, y_cropped_size, x_cropped_size = 64, 256, 256
    z_center, y_center, x_center = np.array(img.shape) // 2
    cropped = img[z_center - z_cropped_size // 2: z_center + z_cropped_size // 2,
                         y_center - y_cropped_size // 2: y_center + y_cropped_size // 2,
                         x_center - x_cropped_size // 2: x_center + x_cropped_size // 2]
    return cropped

def _center_crop_512(img):
    print("working")
    z_cropped_size, y_cropped_size, x_cropped_size = 64, 512, 512
    z_center, y_center, x_center = np.array(img.shape) // 2
    cropped = img[z_center - z_cropped_size // 2: z_center + z_cropped_size // 2,
                         y_center - y_cropped_size // 2: y_center + y_cropped_size // 2,
                         x_center - x_cropped_size // 2: x_center + x_cropped_size // 2]
    return cropped

def _to_float32(img):
    # accounting for TCF ver 1.2
    if img.dtype == 'uint16':
        img = np.true_divide(img, 10000.)
        # at this point, img.dtype == float64
    if img.dtype == 'float64':  # DO NOT change this to elif. Intention: uint16-->float64-->float32
        img = img.astype('float32')
        # img /= 10000.
    return img


# In[17]:


################################################################
# 입력값들: 디렉토리, 이름 리스트, 3d raw 리스트, 3d cropped 리스트 등    #
################################################################
# 기존 샘플

# 5번서버 input_dir = Path("/home/user/jaeun/dip/dataset/tcf/nih3t3/dip")
# 3번서버:
input_dir = Path("/home/user/jaeun/dip/dataset/tcf/nih3t3/dip/")
input_names = sorted([path.stem for path in list(input_dir.rglob("*.TCF"))])
print(input_names)
# input_3d_list = [np.array(h5py.File(Path(str(input_dir / input_name) + ".TCF"))['Data/3D/000000']) for input_name in input_names]
# input_cropped = [zoom(_to_float32(_center_crop_512(img)), [1, 0.5, 0.5]) for img in input_3d_list]

input_3d_dict = {input_name : zoom(_to_float32(_center_crop_512(np.array(h5py.File(Path(str(input_dir / input_name) + ".TCF"))['Data/3D/000000']))), [1, 0.5, 0.5])               for input_name in input_names}

# print(input_3d_list)


# In[18]:


output_dict = defaultdict(dict)
for key, item in input_3d_dict.items():
    output_dict[key]["input"] = input_3d_dict[key]


# In[19]:


# output dict가 잘 만들어졌는지 확인
for key, item in output_dict.items():
    print(key)
    print(np.shape(item["input"]))
    plt.imshow(np.max(item["input"], axis=0))
    plt.show()


# In[29]:


#######################
#     BM3D 출력값       #
#######################

bm3d_results = scipy.io.loadmat('bm3d_result.mat')


# In[30]:


print(bm3d_results.keys())


# In[36]:


print(input_names)
date = []
for name in input_names:
    date.append(name.split(".")[0])
print(date)


# In[39]:


for i, name in enumerate(input_names):
    output_dict[name]["slice"] = bm3d_results["slice_5_{}".format(date[i])]
    output_dict[name]["volume_Rice"] = bm3d_results["volume_Rice_{}".format(date[i])]


# In[40]:


#######################
#      DIP 출력값       #
#######################

################################################################
# 출력값들: 디렉토리, 이름 리스트, 3d raw 리스트, 3d cropped 리스트 등    #
################################################################
import os

def append_dip_results(output_dict, output_dirs, label):
    for output_dir in output_dirs:
#         print(output_dir)
        os.listdir(output_dir)
        input_name = Path(output_dir).stem
#         print(input_name)
        output_paths = sorted(list(Path(output_dir).glob("*.hdf")))
#         print(output_paths)
        for output_path in output_paths:
            if output_path.stem in ["0300", "0800", "1300", "1800"]: # epochs to visualize
#                 print(output_path.stem)
                output = h5py.File(output_path, "r")["/output/"]
                output_dict[input_name]["{}_{}".format(label, output_path.stem)] = output
            
# 3D-DIP 세포에 대해서 처음했을때, 5번서버 output_base = Path("/home/user/jaeun/dip/dip_denoised")
output_base_tv_zoom = Path("/home/user/jaeun/dip/tv_1_zoom") # !! 주목 !! 여기를 수정해서 확인할 실험결과를 정할 수 있다
output_dirs_tv_zoom = [str(output_base_tv_zoom)+"/"+input_name+str(".TCF") for input_name in input_names]

append_dip_results(output_dict, output_dirs_tv_zoom, "tv_zoom")
        
output_base_zoom = Path("/home/user/jaeun/dip/zoom") # !! 주목 !! 여기를 수정해서 확인할 실험결과를 정할 수 있다
output_dirs_zoom = [str(output_base_zoom)+"/"+input_name+str(".TCF") for input_name in input_names]

append_dip_results(output_dict, output_dirs_zoom, "zoom")


# In[41]:


def duplicateChecker(imgs):
    sums = sorted([np.sum(img) for img in imgs])
    for i in range(len(sums)-1):
        if(sums[i]==sums[i+1]):
            print("Duplicate detected?!")
            return
    print("no duplicate.")
    return


# In[42]:


for key, results in output_dict.items():
    print(key)
    for key, item in results.items():
        print(key)
        print(np.shape(item))


# In[47]:


check = []
for key, item in output_dict[input_names[0]].items():
    print(key)
    check.append(item)
    plt.imshow(np.max(item, axis=0))
    plt.show()


# In[49]:





# In[ ]:




