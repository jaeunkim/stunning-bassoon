#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
import h5py
import scipy.ndimage
import scipy.io


# In[2]:


def _center_crop(img):
    z_cropped_size, y_cropped_size, x_cropped_size = 64, 512, 512
    z_center, y_center, x_center = np.array(img.shape) // 2
    cropped = img[z_center - z_cropped_size // 2: z_center + z_cropped_size // 2,
                         y_center - y_cropped_size // 2: y_center + y_cropped_size // 2,
                         x_center - x_cropped_size // 2: x_center + x_cropped_size // 2]
    return cropped


# In[3]:


raw_list = sorted(list(Path("/data1/jaeun/dip/dataset/g2").rglob("*.TCF")))


# In[4]:


for path in raw_list:
    stem = path.stem
    print(stem)
    print(stem[16:19])


# In[5]:


names = []
imgs = []
for path in raw_list:
    with h5py.File(path, 'r') as hf:
        print(hf)
        raw_3d = hf["Data/3D/000000"]
        print(raw_3d.dtype)
        if raw_3d.dtype=='uint16':
            raw_3d = np.true_divide(raw_3d, 10000.)
        #atthispoint,img.dtype==float64
        if raw_3d.dtype=='float64': #DONOTchangethistoelif.Intention:uint16-->float64-->float32
            raw_3d=raw_3d.astype('float32')
        #img/=10000.
#         print(raw_3d.dtype)
#         print(np.max(raw_3d))
#         print(np.min(raw_3d))
#         print(raw_3d.shape)
        cropped = _center_crop(raw_3d)
#         print(cropped.shape)
#         print(cropped.dtype)
        zoomed = scipy.ndimage.zoom(cropped, [1, 0.5, 0.5])
        imgs.append(zoomed)
#         zoomed_matlab = matlab.double(zoomed)
#         print(type(zoomed_matlab))
#         print(zoomed.shape)
#         print(zoomed.dtype)
        stem = path.stem
        names.append(stem[16:19])
#         print(path.stem)
#         print('{}'.format(path.stem))
#         print(stem[:8])
#         print(type(zoomed))

print(names)
for img in imgs:
    print(np.shape(img))
#         scipy.io.savemat('/home/user/jaeun/dip/dataset/to_compare/more_samples.mat', {'new_{}'.format(stem[:8]): zoomed})


# In[6]:


save_dict = {}
for i in range(len(names)):
    save_dict["g2_{}".format(names[i])] = imgs[i]


# In[7]:


scipy.io.savemat('/data1/jaeun/dip/g2.mat', save_dict)


# In[8]:


for key, item in save_dict.items():
    print(key)
    plt.show()


# In[23]:


print(save_dict)


# 으아아ㅏㅏ아아아ㅏ아아아아ㅏ아아아ㅏ아ㅏㅏㅏ아ㅏㅏ아ㅏㅏ아ㅏㅏ아ㅏ아ㅏㅏ아ㅏㅏ아아ㅏ아아아ㅏㅏ아ㅏㅏ아ㅏㅏㅏ아ㅏㅏㅏ!!!!!

# In[37]:


for name in names:
    print("[volume_Gauss_new_{}, sigmaMat] = bm4d(new_{}, 'Gauss', 5);".format(name, name))
    print("[volume_Rice_new_{}, sigmaMat] = bm4d(new_{}, 'Rice', 5);".format(name, name))


# In[36]:





# In[ ]:




