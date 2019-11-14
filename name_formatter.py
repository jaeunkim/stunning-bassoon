#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import os


# In[ ]:


parent = Path("/home/user/jaeun/dip/dip_denoised/")


# In[ ]:


results = list(parent.rglob("*.hdf"))


# In[ ]:


print(results)


# In[ ]:


for path in results:
    fname = path.stem
    if len(fname) == 3:
#         print(str(path)[:-7]+"0"+fname+".hdf")
#         print(fname)
        Path.rename(path, str(path)[:-7]+"0"+fname+".hdf")
#         Path.rename(str(path)[:-7]+"0000.hdf")


# In[ ]:




