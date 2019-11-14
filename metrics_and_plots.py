#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cv2

from pathlib import Path
from PIL import Image

from skimage.measure import compare_ssim as ssim
from skimage import feature    # Canny Edge 사용하기 위함

import torch


# In[55]:


def norm(img):
    # accounting for TCF ver 1.2
    if img.dtype == 'float64':
        img = img.astype('float32')
        # img /= 10000.
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / std #, mean, std

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

# list로 받은 image들의 정해진 x값 혹은 y값의 RI value plot
def plot_line(imgs, img_list, axis=0, coord=128):
    '''
    imgs: list of 2d arrays (images)
    img_list: list of strings (name of the images)
    axis: direction (0: horizontal, 1: vertical)
    coord: position
    '''
    
    n = len(imgs)
    plt.figure(figsize=(5, 5))
    plt.axis('on')
    plt.imshow(imgs[0], vmin=1.337, vmax=1.39, cmap='viridis')

    if axis==0:
        plt.axhline(y=coord, linewidth=1.5, color='r')
        plt.show()
        plt.close()
        plt.figure(figsize=(15, 10))
        for i, k in enumerate(img_list):
            plt.plot(imgs[i][coord, :], label=k)
        plt.xlabel("y position")
        
    elif axis==1:
        plt.axvline(x=coord, linewidth=1.5, color='r')
        plt.show()
        plt.close()
        plt.figure(figsize=(15, 10))
        for i, k in enumerate(img_list):
            plt.plot(imgs[i][:, coord], label=k)
        plt.xlabel("x position")
        
    plt.ylabel("RI value")
    plt.legend()
    plt.show()
    plt.close()



def imshow_path(imagepath):
    list_image = sorted(imagepath.rglob('*.hdf'))
    for i, path in enumerate(list_image):
        print(path.name)
        with h5py.File(path, 'r') as f:
            img = f['ri'][()]
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            im = ax.imshow(img, vmin=1.33, vmax=1.4, cmap='viridis')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.03)
            plt.colorbar(im, cax=cax, label='RI')
            
            plt.suptitle(path.stem, fontsize=15)
            plt.tight_layout()
            plt.show()
        plt.close()

def imshow_single(img, vmin=1.33, vmax=1.4):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    im = ax.imshow(img)#, vmin=vmin, vmax=vmax, cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.03)
    plt.colorbar(im, cax=cax, label='RI')
    plt.tight_layout()
    plt.show()
    plt.close()

def brenner_grad(img):
    return np.sum((img[:-1, :] - img[1:, :])**2)

def brenner_map(img, patchsize):
    x, y = img.shape
    for i in range(x//patchsize):
        for j in range(y//patchsize):
            img[i*patchsize:(i+1)*patchsize, j*patchsize:(j+1)*patchsize]             = brenner_grad(img[i*patchsize:(i+1)*patchsize, j*patchsize:(j+1)*patchsize])
    return img

def brenner_nonpatch(img, patchsize):
    x, y = img.shape
    print(img.shape)
    img = np.pad(img, (patchsize//2, patchsize//2), 'reflect')
    nimg = np.zeros(img.shape)
    print(img.shape)
    for i in range(patchsize//2, x+patchsize//2):
        for j in range(patchsize//2, y+patchsize//2):
            nimg[i, j] = brenner_grad(img[i-patchsize//2:i+patchsize//2, j-patchsize//2:j+patchsize//2])
    return nimg[patchsize:x+patchsize, patchsize:y+patchsize]

def cnr(img1, img2):
    return abs(np.mean(img1) - np.mean(img2)) / (np.std(img1)**2 + np.std(img2)**2)**0.5

def cnr(img, bg_p1, bg_p2, sp_p1, sp_p2):
    '''
    Constrast to Noise Ratio
    '''
    (bg_x1, bg_y1) = bg_p1
    (bg_x2, bg_y2) = bg_p2
    (sp_x1, sp_y1) = sp_p1
    (sp_x2, sp_y2) = sp_p2
    
    bg_patch = img[bg_x1:bg_x2, bg_y1:bg_y2]
    sp_patch = img[sp_x1:sp_x2, sp_y1:sp_y2]
    
    return abs(np.mean(bg_patch) - np.mean(sp_patch)) / (np.std(bg_patch)**2 + np.std(sp_patch)**2)**0.5

def snr(img, bg_p1, bg_p2, sp_p1, sp_p2):
    '''
    Signal to Noise Ratio
    return: background patch, sample patch의 peak to peak value ratio
    meaning: background value의 peak to peak value가 낮아질수록 (=shotnoise가 없어질수록) 값이 커짐.
            --> 즉, 출력값이 높을수록 denoise 잘 됐다는 의미
            --> 단 edge 정보에는 대응하지 못함
    '''
    (bg_x1, bg_y1) = bg_p1
    (bg_x2, bg_y2) = bg_p2
    (sp_x1, sp_y1) = sp_p1
    (sp_x2, sp_y2) = sp_p2
    
    bg_patch = img[bg_x1:bg_x2, bg_y1:bg_y2]
    sp_patch = img[sp_x1:sp_x2, sp_y1:sp_y2]
    
    return (np.max(sp_patch) - np.min(sp_patch)) / (np.max(bg_patch) - np.min(bg_patch))

def show_metrics(img, bg_p1, bg_p2, sp_p1, sp_p2):
    snr_ = snr(img, bg_p1, bg_p2, sp_p1, sp_p2)
    cnr_ = cnr(img, bg_p1, bg_p2, sp_p1, sp_p2)
    br_sp_ = brenner_grad(img[sp_p1[0]:sp_p2[0], sp_p1[1]:sp_p2[1]])
    br_bg_ = brenner_grad(img[bg_p1[0]:bg_p2[0], bg_p1[1]:bg_p2[1]])
    print("SNR: ", snr_)
    print("CNR: ", cnr_)
    print("Brenner Gradient of the edge: ", br_sp_)
    print("Brenner Gradient of the background: ", br_bg_)
    return snr_, cnr_, br_sp_, br_bg_


# In[40]:


## Analyzer lists

# NN-regularized bead list
path_nnbead_dip_base = Path('/data3/denoise/dip_output/bead/0924_dip_base/20180726.150404.579.SiO2_5um-001_NN_MIP.hdf')
path_nnbead_dip_tv01 = Path('/data3/denoise/dip_output/bead/0924_dip_tv01/20180726.150404.579.SiO2_5um-001_NN_MIP.hdf')
path_nnbead_dip_ssim05 = Path('/data3/denoise/dip_output/bead/0924_dip_ssim05/20180726.150404.579.SiO2_5um-001_NN_MIP.hdf')
path_nnbead_dip_tv001_ssim05 = Path('/data3/denoise/dip_output/bead/0924_dip_tv001_ssim05/20180726.150404.579.SiO2_5um-001_NN_MIP.hdf')
list_nnbead_dip_base = sorted(path_nnbead_dip_base.rglob('*.hdf'), key=lambda x: int(x.stem))
list_nnbead_dip_tv01 = sorted(path_nnbead_dip_tv01.rglob('*.hdf'), key=lambda x: int(x.stem))
list_nnbead_dip_ssim05 = sorted(path_nnbead_dip_ssim05.rglob('*.hdf'), key=lambda x: int(x.stem))
list_nnbead_dip_tv001_ssim05 = sorted(path_nnbead_dip_tv001_ssim05.rglob('*.hdf'), key=lambda x: int(x.stem))
list_nnbead_dip_zip = list(zip(list_nnbead_dip_base, list_nnbead_dip_tv01, list_nnbead_dip_ssim05, list_nnbead_dip_tv001_ssim05))

# Raw bead list
path_rawbead_dip_base = Path('/data3/denoise/dip_output/bead/0924_dip_base/20180726.150404.579.SiO2_5um-001_raw_MIP.hdf')
path_rawbead_dip_tv01 = Path('/data3/denoise/dip_output/bead/0924_dip_tv01/20180726.150404.579.SiO2_5um-001_raw_MIP.hdf')
path_rawbead_dip_ssim05 = Path('/data3/denoise/dip_output/bead/0924_dip_ssim05/20180726.150404.579.SiO2_5um-001_raw_MIP.hdf')
path_rawbead_dip_tv001_ssim05 = Path('/data3/denoise/dip_output/bead/0924_dip_tv001_ssim05/20180726.150404.579.SiO2_5um-001_raw_MIP.hdf')
list_rawbead_dip_base = sorted(path_rawbead_dip_base.rglob('*.hdf'), key=lambda x: int(x.stem))
list_rawbead_dip_tv01 = sorted(path_rawbead_dip_tv01.rglob('*.hdf'), key=lambda x: int(x.stem))
list_rawbead_dip_ssim05 = sorted(path_rawbead_dip_ssim05.rglob('*.hdf'), key=lambda x: int(x.stem))
list_rawbead_dip_tv001_ssim05 = sorted(path_rawbead_dip_tv001_ssim05.rglob('*.hdf'), key=lambda x: int(x.stem))
list_rawbead_dip_zip = list(zip(list_rawbead_dip_base, list_rawbead_dip_tv01, list_rawbead_dip_ssim05, list_rawbead_dip_tv001_ssim05))



# In[ ]:


# Bead output view
for i, path in enumerate(list_nnbead_dip_zip):
    if i in [2, 3, 5, 10, 20, 30, 39]:
        print("{}th epoch".format(path[0].stem))
        _input = []
        _output = []

        for j, hdf in enumerate(path):
            with h5py.File(hdf, 'r') as f:
                if j in [0, 1]:
                    _input.append(f['input'][()])
                else:
                    _input.append(f['target'][()])
                _output.append(f['output'][()])

        plt_list = [_input[0]] + _output
#         plt_title_list = ["input", "DIP (MSE)", "DIP (MSE, SSIM*0.5)", "DIP (MSE, SSIM*0.5, TV*0.01)"]
        plt_title_list = ["input", "DIP (MSE)", "DIP (MSE, TV*0.1)", "DIP (MSE, SSIM*0.5)", "DIP (MSE, SSIM*0.5, TV*0.01)"]
        print("image")   # 영상 비교
        plot_img(plt_list, plt_title_list)

        print("difference")   # 차영상 비교
        plot_diff(_input[0], plt_list, plt_title_list)
        
        print("edge")    # Canny Edge detector
        plot_edge(_input[0], plt_list, plt_title_list)
        
        print("background histogram")    # 배경 histogram 변화 비교
        plot_zoom(plt_list, plt_title_list, p1=(0, 0), p2=(64, 64), range=(1.335, 1.35))

        print("boundary zoomup")   # 디테일 비교 (gradient 큰 곳)
        (x1, y1) = (96, 96)
        (x2, y2) = (160, 160)
        plot_zoom(plt_list, plt_title_list, p1=(x1, y1), p2=(x2, y2), range=(1.33, 1.42), additional_line=False)

        print("midline value plot")   # RI값 변화 확인
        plot_line(plt_list, plt_title_list, axis=0, coord=128)

        for i, img in enumerate(plt_list):
            print("{}'s metric".format(plt_title_list[i]))
            show_metrics(img, bg_p1=(0, 0), bg_p2=(64, 64), sp_p1=(96, 96), sp_p2=(160, 160))


# In[52]:


# Cell output view

# NIH3T3 list
output_path = '/data3/denoise/dip_output/nih3t3'
sample_name='20181114.133609.514.Default-001_mip.hdf'
path_nih3t3_dip_base = Path('{}/0924_dip_base/{}'.format(output_path, sample_name))
path_nih3t3_dip_ssim05 = Path('{}/0924_dip_ssim05/{}'.format(output_path, sample_name))
path_nih3t3_dip_tv001_ssim05 = Path('{}/0924_dip_tv001_ssim05/{}'.format(output_path, sample_name))
path_nih3t3_dip_tv01 = Path('{}/0924_dip_tv01/{}'.format(output_path, sample_name))

list_nih3t3_dip_base = sorted(path_nih3t3_dip_base.rglob('*.hdf'), key=lambda x: int(x.stem))
list_nih3t3_dip_ssim05 = sorted(path_nih3t3_dip_ssim05.rglob('*.hdf'), key=lambda x: int(x.stem))
list_nih3t3_dip_tv001_ssim05 = sorted(path_nih3t3_dip_tv001_ssim05.rglob('*.hdf'), key=lambda x: int(x.stem))
list_nih3t3_dip_tv01 = sorted(path_nih3t3_dip_tv01.rglob('*.hdf'), key=lambda x: int(x.stem))

list_nih3t3_dip_zip = list(zip(list_nih3t3_dip_base, list_nih3t3_dip_ssim05, list_nih3t3_dip_tv001_ssim05, list_nih3t3_dip_tv01))


# In[52]:


for i, path in enumerate(list_nih3t3_dip_zip):
    if i in [2, 3, 5, 10, 20, 30, 39]:
        print("{}th epoch".format(path[0].stem))
        _input = []
        _output = []

        for j, hdf in enumerate(path):
            with h5py.File(hdf, 'r') as f:
                _input.append(f['target'][()])
                _output.append(f['output'][()])

        plt_list = [_input[0]] + _output
        plt_title_list = ["input", "DIP (MSE)", "DIP (MSE, SSIM*0.5)", "DIP (MSE, SSIM*0.5, TV*0.01)", "DIP (MSE, TV*0.1)"]
        print("image")   # 영상 비교
        plot_img(plt_list, plt_title_list)

        print("difference")   # 차영상 비교
        plot_diff(_input[0], plt_list, plt_title_list)
        
        print("edge")    # Canny Edge detector
        plot_edge(_input[0], plt_list, plt_title_list)
        
        print("background histogram")    # 배경 histogram 변화 비교
        plot_zoom(plt_list, plt_title_list, p1=(0, 0), p2=(64, 64), range=(1.335, 1.35))

        print("boundary zoomup")   # 디테일 비교 (gradient 큰 곳)
        (x1, y1) = (100, 320)
        (x2, y2) = (164, 384)
        plot_zoom(plt_list, plt_title_list, p1=(x1, y1), p2=(x2, y2), range=(1.33, 1.42), additional_line=False)

        print("midline value plot")   # RI값 변화 확인
        plot_line(plt_list, plt_title_list, axis=0, coord=256)

        for i, img in enumerate(plt_list):
            print("{}'s metric".format(plt_title_list[i]))
            show_metrics(img, bg_p1=(0, 0), bg_p2=(64, 64), sp_p1=(100, 320), sp_p2=(164, 384))


# In[56]:


metric_arr = []
for i, path in enumerate(list_nih3t3_dip_base):
    with h5py.File(path, 'r') as f:
        _input = f['target'][()]
        _output = f['output'][()]
    snr_, cnr_, br_sp_, br_bg_ = show_metrics(_output, bg_p1=(0, 0), bg_p2=(64, 64), sp_p1=(100, 320), sp_p2=(164, 384))
    metric_arr.append([snr_, cnr_, br_sp_, br_bg_])


# In[59]:


print(metric_arr[0][0])


# In[ ]:




