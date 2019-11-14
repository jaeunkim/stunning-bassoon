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
        im = plt.imshow(img, vmin=1.32, vmax=1.4, cmap='viridis')
        
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


# In[6]:


################################################################
# 입력값들: 디렉토리, 이름 리스트, 3d raw 리스트, 3d cropped 리스트 등    #
################################################################
# 기존 샘플

# 5번서버 input_dir = Path("/home/user/jaeun/dip/dataset/tcf/nih3t3/dip")
# 3번서버:
input_dir = Path("/data1/jaeun/dip/dataset/garbage/")
input_names = sorted([path.stem for path in list(input_dir.rglob("*.TCF"))])
print(input_names)
input_3d_list = [np.array(h5py.File(Path(str(input_dir / input_name) + ".TCF"))['Data/3D/000000']) for input_name in input_names]
# input_cropped = [zoom(_to_float32(_center_crop_512(img)), [1, 0.5, 0.5]) for img in input_3d_list]

input_3d_dict = {input_name : zoom(_to_float32(_center_crop_512(np.array(h5py.File(Path(str(input_dir / input_name) + ".TCF"))['Data/3D/000000']))), [1, 0.5, 0.5])               for input_name in input_names}

# print(input_3d_list)


# In[9]:


slice_to_plot = focus_finder(input_3d_list)
print(slice_to_plot)


# In[10]:


output_dict = defaultdict(dict)
for key, item in input_3d_dict.items():
    output_dict[key]["input"] = input_3d_dict[key]


# In[11]:


for key, item in output_dict.items():
    print(key)
    print(np.shape(item["input"]))
    plt.imshow(np.max(item["input"], axis=0))
    plt.show()


# In[12]:


# 안중요한셀. 확인용 

for img in input_cropped_original:
    plt.imshow(np.max(img, axis=0))
    print(np.shape(img))


# In[13]:


#######################
#     BM3D 출력값       #
#######################

bm3d_options = ["target", "st_1", "st_01", "st_001", "st_0001", "st_00001"]
slice_5_garbage = scipy.io.loadmat('slice_5_garbage_fixed.mat')
volume_Rice_garbage = scipy.io.loadmat('volume_Rice_garbage_fixed.mat')
volume_Gauss_garbage = scipy.io.loadmat('volume_Gauss_garbage_fixed.mat')


# In[14]:


print(np.shape(slice_5_garbage['slice_5_garbage'][0][0]))
print(np.shape(volume_Rice_garbage['volume_Rice_garbage'][0][0]))
print(np.shape(volume_Gauss_garbage['volume_Gauss_garbage'][0][0]))


# In[15]:


#######################
#      DIP 출력값       #
#######################

################################################################
# 출력값들: 디렉토리, 이름 리스트, 3d raw 리스트, 3d cropped 리스트 등    #
################################################################

# 3D-DIP 세포에 대해서 처음했을때, 5번서버 output_base = Path("/home/user/jaeun/dip/dip_denoised")
output_base = Path("/home/user/jaeun/dip/tv_1_zoom/") # !! 주목 !! 여기를 수정해서 확인할 실험결과를 정할 수 있다
output_dirs = [str(output_base)+"/"+input_name+str(".TCF") for input_name in input_names]

output_dict = {}
for output_dir in output_dirs:
    input_name = Path(output_dir).stem
    print(input_name)
    output_paths = sorted(list(Path(output_dir).glob("*.hdf")))
    output_list = []
    for output_path in output_paths:
        if output_path.stem in ["0300", "0800", "1300", "1800"]: # epochs to visualize
#             print(output_path.stem)
            output = h5py.File(output_path, "r")["/output/"]
            output_list.append(output)
    output_dict[input_name] = output_list


# In[16]:


#######################
#      DIP 출력값       #
#######################

################################################################
# 출력값들: 디렉토리, 이름 리스트, 3d raw 리스트, 3d cropped 리스트 등    #
################################################################

# 3D-DIP 세포에 대해서 처음했을때, 5번서버 output_base = Path("/home/user/jaeun/dip/dip_denoised")
output_base_tv_zoom = Path("/data1/jaeun/dip/tv_1_zoom_garbage/") # !! 주목 !! 여기를 수정해서 확인할 실험결과를 정할 수 있다
output_dirs_tv_zoom = [str(output_base_tv_zoom)+"/"+input_name+str(".TCF") for input_name in input_names]

print(output_dirs_tv_zoom)

output_base_zoom = Path("/data1/jaeun/dip/zoom_garbage/") # !! 주목 !! 여기를 수정해서 확인할 실험결과를 정할 수 있다
output_dirs_zoom = [str(output_base_zoom)+"/"+input_name+str(".TCF") for input_name in input_names]


# In[31]:


output_dict_tv_zoom = {}
for output_dir in output_dirs_tv_zoom:
    input_name = Path(output_dir).stem
    print(input_name)
    output_paths = sorted(list(Path(output_dir).glob("*.hdf")))
    output_list = []
    for output_path in output_paths:
        if output_path.stem in ["800"]: # epochs to visualize
#             print(output_path.stem)
            output = h5py.File(output_path, "r")["/output/"]
            output_list.append(output)
    output_dict_tv_zoom[input_name] = output_list
    
for key in output_dict_tv_zoom.keys():
    print(np.shape(output_dict_tv_zoom[key]))
    
output_dict_zoom = {}
for output_dir in output_dirs_zoom:
    input_name = Path(output_dir).stem
    print(input_name)
    output_paths = sorted(list(Path(output_dir).glob("*.hdf")))
    output_list = []
    for output_path in output_paths:
        if output_path.stem in ["800"]: # epochs to visualize
#             print(output_path.stem)
            output = h5py.File(output_path, "r")["/output/"]
            output_list.append(output)
    output_dict_zoom[input_name] = output_list


# In[32]:


#name order editing
name_dict = defaultdict(dict)
num_list = []
for i in range(12):
    num = input_names[i][16:19]
    print(num)
    name_dict[num] = i
    num_list.append(num)
#     name_dict[input_names]

sorted_num_list = sorted(num_list)

print(sorted_num_list)

dummy = [np.zeros((64, 256, 256), dtype='float32')]
print(len(dummy))

s5g_list = dummy*12
vGg_list = dummy*12
vRg_list = dummy*12

for i in range(12):
    dest_idx = name_dict[sorted_num_list[i]]
    print(dest_idx)
    s5g_list[dest_idx] = slice_5_garbage['slice_5_garbage'][0][i]
    vGg_list[dest_idx] = volume_Gauss_garbage['volume_Gauss_garbage'][0][i]
    vRg_list[dest_idx] = volume_Rice_garbage['volume_Rice_garbage'][0][i]
    


# In[33]:





# In[56]:


#garbage image plotting (middle slice)

for i in range(len(input_names)):
    print(i," out of ",len(input_names))
    my_name = input_names[i]
    print(my_name)
    imgs_to_plot = []
    imgs_to_plot.append(np.squeeze(input_3d_dict[my_name][32]))  # 입력값
    
    print(len(imgs_to_plot))
    for img in output_dict_zoom[my_name]:  # zoom only
        imgs_to_plot.append(np.squeeze(img[32]))
    print(len(imgs_to_plot))
    for img in output_dict_tv_zoom[my_name]:  # TV + zoom
        imgs_to_plot.append(np.squeeze(img[32]))
    print(len(imgs_to_plot))
    imgs_to_plot.append(s5g_list[i][32])
    print(len(imgs_to_plot))
    imgs_to_plot.append(vGg_list[i][32])
    print(len(imgs_to_plot))
    imgs_to_plot.append(vRg_list[i][32])
    
    print(len(imgs_to_plot))
    
    name_list = ["garbage_{}[{}]".format(input_names[i], k) for k in range(len(imgs_to_plot))]
    
    duplicateChecker(imgs_to_plot)
    plot_img_compare(imgs_to_plot, name_list)


# In[33]:


#garbage image plotting (MIP)

for i in range(len(input_names)):
    print(i," out of ",len(input_names))
    my_name = input_names[i]
    print(my_name)
    imgs_to_plot = []
    imgs_to_plot.append(np.squeeze(np.max(input_3d_dict[my_name], 0)))  # 입력값
    
    print(len(imgs_to_plot))
    for img in output_dict_zoom[my_name]:  # zoom only
        imgs_to_plot.append(np.squeeze(np.max(img, 0)))
    print(len(imgs_to_plot))
    for img in output_dict_tv_zoom[my_name]:  # TV + zoom
        imgs_to_plot.append(np.squeeze(np.max(img, 0)))
    print(len(imgs_to_plot))
    imgs_to_plot.append(np.max(s5g_list[i], 0))
    print(len(imgs_to_plot))
    imgs_to_plot.append(np.max(vGg_list[i], 0))
    print(len(imgs_to_plot))
    imgs_to_plot.append(np.max(vRg_list[i], 0))
    
    print(len(imgs_to_plot))
    
    name_list = ["garbage_{}[{}]".format(input_names[i], k) for k in range(len(imgs_to_plot))]
    
    duplicateChecker(imgs_to_plot)
    plot_img_compare(imgs_to_plot, name_list)


# In[26]:


for i in range(len(input_names)):
    print(i)
    print(input_names[i])
#     print(input_names_original[i][:8])
    print("slice " + str(slice_to_plot[i]))
    
    imgs_to_plot = []
    imgs_to_plot.append(np.squeeze(input_cropped_new[i][slice_to_plot[i]]))  # 입력값
    
    for img in output_dict_tv_zoom[input_names_new[i]]:  # TV + zoom
        imgs_to_plot.append(np.squeeze(img[slice_to_plot[i]]))
    
    for img in output_dict_zoom[input_names_new[i]]:  # zoom only
        imgs_to_plot.append(np.squeeze(img[slice_to_plot[i]]))
        
    imgs_to_plot.append(bm3d_results['slice_5_new_{}'.format(input_names_new[i][:8])][slice_to_plot[i]])
    imgs_to_plot.append(bm3d_results['volume_Gauss_new_{}'.format(input_names_new[i][:8])][slice_to_plot[i]])
    imgs_to_plot.append(bm3d_results['volume_Rice_new_{}'.format(input_names_new[i][:8])][slice_to_plot[i]])
    
    name_list = ["new_{} [{}]".format(input_names_new[i][:8], k+1) for k in np.arange(12)]
    
    duplicateChecker(imgs_to_plot)
    plot_img_compare(imgs_to_plot, name_list)


# In[49]:


for i in range(len(input_names_new)):
    print(i)
    print(input_names_new[i])
#     print(input_names_original[i][:8])
    
    imgs_to_plot = []
    imgs_to_plot.append(zoom(np.squeeze(input_cropped_new[i][:,128,:]), (0.5, 1)))  # 입력값
    
    for img in output_dict_tv_zoom[input_names_new[i]]:  # TV + zoom
        imgs_to_plot.append(zoom(np.squeeze(img[:, 128, :]), (0.5, 1))
    
    for img in output_dict_zoom[input_names_new[i]]:  # zoom only
        imgs_to_plot.append(zoom(np.squeeze(img[:, 128, :]), (0.5, 1)))
        
    imgs_to_plot.append(zoom(bm3d_results['slice_5_new_{}'.format(input_names_new[i][:8])][:, 128, :]), [0.5, 1])
    imgs_to_plot.append(zoom(bm3d_results['volume_Gauss_new_{}'.format(input_names_new[i][:8])][:, 128, :]), [0.5, 1])
    imgs_to_plot.append(zoom(bm3d_results['volume_Rice_new_{}'.format(input_names_new[i][:8])][:, 128, :]), [0.5, 1])
    
    name_list = ["new_{} [{}]".format(input_names_new[i][:8], k+1) for k in np.arange(12)]
    
    duplicateChecker(imgs_to_plot)
    plot_img_compare(imgs_to_plot, name_list)


# In[20]:


def duplicateChecker(imgs):
    sums = sorted([np.sum(img) for img in imgs])
    for i in range(len(sums)-1):
        if(sums[i]==sums[i+1]):
            print("Duplicate?!")
            return
    print("no duplicate.")
    return
    


# In[ ]:


duplicateChecker(imgs_to_plot[:3]*4)


# In[ ]:


imgs_to_plot.append(bm3d_results['slice_5_20181114'][34])
imgs_to_plot.append(bm3d_results['volume_Gauss_20181114'][34])
imgs_to_plot.append(bm3d_results['volume_Rice_20181114'][34])


# In[ ]:


for imgs in imgs_to_plot:
#     print(imgs.dtype)
    print(np.sum(imgs))


# In[ ]:


name_list = ["original_20181114 [{}]".format(i+1) for i in np.arange(12)]
plot_img_compare(imgs_to_plot, name_list)


# In[ ]:


st_auto = bm3d_results['st_auto']
st_1 = bm3d_results['st_1']
st_01 = bm3d_results['st_01']
st_001 = bm3d_results['st_001']
st_0001 = bm3d_results['st_0001']
st_00001 = bm3d_results['st_00001']


# In[ ]:


print(bm3d_results.keys())


# In[ ]:


bm3d_comparison = [input_cropped[0], st_1, st_01, st_001, st_0001, st_00001]


# In[ ]:


#### BM3D MIP plot ###
imgs_to_plot = []
for img_3d in bm3d_comparison:
    imgs_to_plot.append(img_3d[32,:,:])

plot_img(imgs_to_plot, bm3d_options)


# In[ ]:


#### BM3D Slice별 plot ###
imgs_to_plot = []
for img_3d in bm3d_comparison:
    imgs_to_plot.append(img_3d[32])

plot_img(imgs_to_plot, bm3d_options)


# In[ ]:


imgs_to_plot = []
imgs_to_plot.append(input_cropped[0])  # 플랏의 첫번째 그림은 target 이다 (눈으로 쉽게 비교하기 위해)
for item in output_dict["20181114.133609.514.Default-001"]:
    imgs_to_plot.append(np.array(item))
    
# 가장 밝은 슬라이스 찾는 부분
brightest_z = []
z_slices = [np.sum(np.sum(img, axis = 1), axis = 1) for img in input_cropped]
for img in z_slices:
    brightest_z.append(np.argmax(img))
print(brightest_z)


# In[ ]:


##############################
# MIP가 아닌 slice별 max image #
##############################

i = 0
for key, value in output_dict.items():
    print(key)
    imgs_to_plot = []
    imgs_to_plot.append(input_cropped[i])  # 플랏의 첫번째 그림은 target 이다 (눈으로 쉽게 비교하기 위해)
    for item in output_dict[key]:
        imgs_to_plot.append(np.array(item))

    ### 여기서 plot_img 호출   ###
    plot_img([img_3d[brightest_z[i]] for img_3d in imgs_to_plot], names_to_plot)
    i = i+1


# In[ ]:


names_to_plot = range(0, 2000, 50)

ref = np.max(_center_crop(input_3d_list[0]), axis = 0)
results = [np.max(_center_crop(img_3d), axis=0) for img_3d in output_dict["20181114.133609.514.Default-001"]]
edge_list = plot_edge(ref, results, names_to_plot)
edge_score = [np.sum(arr) for arr in edge_list]


# In[ ]:


X = range(100, 2000, 50)
Y = edge_score[2:]

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
line = slope*X+intercept

plt.plot(X, Y,'o', X, line)
plt.xlabel("epoch")
plt.ylabel("canny edge score")
plt.title("Canny edge")
plt.show()


# In[ ]:


############################
# 옆에서 본 모습 출력            #
############################

i = 0
for key, value in output_dict.items():
    print(key)
    imgs_to_plot = []
    imgs_to_plot.append(input_cropped[i])
    i = i+1
    for item in output_dict[key]:
        imgs_to_plot.append(np.array(item))
    plot_img([np.max(img_3d, axis=1) for img_3d in imgs_to_plot], names_to_plot)


# In[ ]:


# 기타: 플랏 한 줄만 그리기

imgs_to_plot = []
imgs_to_plot.append(input_cropped[0])  # 플랏의 첫번째 그림은 target 이다 (눈으로 쉽게 비교하기 위해)
for item in output_dict["20181114.133609.514.Default-001"]: 
    imgs_to_plot.append(np.array(item))


# In[ ]:


plot_img(imgs_to_plot, )


# In[ ]:


# 기타: 배경 denoise 성능 확인

imgs_to_plot = []
print(input_names[2])
imgs_to_plot.append(input_cropped[2])  # 플랏의 첫번째 그림은 target 이다 (눈으로 쉽게 비교하기 위해)
for item in output_dict["20190307.140627.606.Default-036"]: 
    imgs_to_plot.append(np.array(item))

imgs_to_plot = [np.max(img, axis=0)[-70: -20, -70: -20] for img in imgs_to_plot]
print(len(imgs_to_plot))
#plot_img(imgs_to_plot, names_to_plot)
print(len(range(0, 2000, 50)))
background_mean = [] 
background_std = []
for background in imgs_to_plot:
    background_mean.append(np.mean(background))
    background_std.append(np.std(background))

print(background_mean)


# In[ ]:


from scipy import stats


# In[ ]:


X = range(100, 2000, 50)
Y = background_std[2:]

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
line = slope*X+intercept

plt.plot(X, Y,'o', X, line)
plt.xlabel("epoch")
plt.ylabel("std of RI")
plt.title("Background denoising: std")
plt.show()


# In[ ]:


plt.plot([0, 150, 300, 800, 1300, 1800], background_mean)
plt.xlabel("epoch")
plt.ylabel("mean RI")
plt.title("Background denoising: mean value")


# In[ ]:


background_std = []
for background in imgs_to_plot:
    background_std.append(np.std(background))

plt.plot(range(0, 1950, 50), background_std)
plt.xlabel("epoch")
plt.ylabel("std of RI")
plt.title("Background denoising: std")


# In[ ]:




