#!/usr/bin/env python
# coding: utf-8

# In[12]:


import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# In[13]:


garbage_paths = sorted(Path('/data1/jaeun/dip/dataset/garbage').rglob("*.TCF"))
print(garbage_paths)


# In[14]:


with h5py.File(garbage_paths[0], 'r') as hf:
    print(list(hf.keys()))


# In[7]:


with h5py.File(nih3t3_paths[0], 'r') as hf:
    print(list(hf.keys()npp))


# In[18]:


# hf = h5py.File(nih3t3_paths[0], 'r')
# np.shape(hf['Data/3D/000000'])

# plt.imshow(hf['Data/3D/000000'], )
# hf.close()

z_cropped_size, y_cropped_size, x_cropped_size = 64, 256, 256
img_dict = {}

with h5py.File(nih3t3_paths[0], 'r') as hf:
    img = hf['target']
    print(img.shape)
    
for path in bead_to_denoise:
    with h5py.File(path, 'r') as hf:
        img = hf['target']
        print(img.shape)
        plt.imshow(np.max(img, axis=0))
        plt.show()
    
# with h5py.File(nih3t3_paths[0], 'r') as hf:
#     img = hf['Data/3D/000000']
#     projection = np.max(img, axis=2)
#     plt.imshow(projection)


# In[9]:


for path in nih3t3_paths:
    with h5py.File(path, 'r') as hf:
        img = hf['Data/3D/000000']
        if img.dtype
        z_center, y_center, x_center = np.array(img.shape)//2
        center_cropped = img[z_center-z_cropped_size//2:z_center+z_cropped_size//2, 
                             y_center-y_cropped_size//2:y_center+y_cropped_size//2,
                             x_center-x_cropped_size//2:x_center+x_cropped_size//2] 
        
        fname = path.stem
        img_dict[fname] = center_cropped

print(img_dict)


# In[58]:


x, y, z = np.array((1, 2, 3))/2
print(x, y, z)


# In[35]:


print(hf.keys())


# In[ ]:




