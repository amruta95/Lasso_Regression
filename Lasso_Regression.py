# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


# In[12]:


x = np.random.normal(0,10,80*250)
x = np.reshape(x,(80,250))

weights=np.zeros((80,1))

weights[0][0]=10
weights[1][0]=-10
weights[2][0]=10
weights[3][0]=-10
weights[4][0]=10
weights[5][0]=-10
weights[6][0]=-10
weights[7][0]=10
weights[8][0]=-10
weights[9][0]=10

b = np.zeros(250)

noise=np.random.normal(0, 10, 250)
# noise= np.reshape(noise,(250,1))

y =  np.transpose(x).dot(weights) + b + noise

lmbda=2*np.amax(abs(x.dot(y-np.sum(y)/float(250))))
print lmbda
temp = x.copy()
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        temp[i][j] **=2

ak = 2*np.sum(temp,axis=1)
print ak.shape


# In[13]:


r = y - np.transpose(x).dot(weights) - b
r.shape


# In[14]:


d = 80
n= 250
m=1

p_list = []
r_list = []
l_list = []

while m<=10:
    loss_prev = float("inf")
    loss = 0
    while (loss_prev - loss)> 0.001:
        b_old = b.copy().reshape(-1)
        r = y - np.transpose(x).dot(weights) - b

        b = np.full(n, (np.sum(r+b))/float(n))

        r = r + b_old - b

        for k in range(d):
            wk = weights[k]
            wk_old = wk

            ck = 2*(np.transpose(x[k]).dot((r + (x[k]) * wk)))

            ck = ck.item(0)

            if ck < -lmbda:
                wk = (ck+lmbda)/float(ak[k])
            elif ck > lmbda:
                wk = (ck-lmbda)/float(ak[k])
            else:
                wk = 0

            r = r + x[k]*(wk_old-wk)

            weights[k] = wk



        loss_prev = loss
        if(loss_prev==0): loss_prev = float("inf")


        w2 = weights.reshape(80,1)

        t1 = x.T.dot(w2)

        t = (t1+b-y).sum()
        loss  = np.sum((t**2)) + lmbda*np.sum(abs(w2))
        print loss
        print "RMSE   ::::   ", np.sqrt(np.sum(((y-(np.transpose(x).dot(weights)+b))**2))/n)
    lmbda/=2
    l_list.append(lmbda)
    print "Lambda --- ",lmbda
    nz = np.count_nonzero(weights[0:10])
    if not float(np.count_nonzero(weights)):
        p = 1
    else:
        p = nz/float(np.count_nonzero(weights))
    r = nz/float(10)
    p_list.append(p)
    r_list.append(r)
    print "Precision",p
    print "Recall",r
    m=m+1


# In[16]:


plt.plot(p_list,l_list)
plt.xlabel('Precision values')
plt.ylabel('Lambda values')
plt.show()
plt.plot(r_list,l_list)
plt.xlabel('Recall values')
plt.ylabel('Lambda values')
plt.show()
