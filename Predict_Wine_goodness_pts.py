# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


train_data = pd.read_csv('C:/Users/Tejas/Desktop/ML/trainData.txt',sep=' ',header=None)
train_labels = pd.read_csv('C:/Users/Tejas/Desktop/ML/trainLabels.txt',sep=' ',header=None)
test_data = pd.read_csv('C:/Users/Tejas/Desktop/ML/testData.txt',sep=' ',header=None)
validation_data = pd.read_csv('C:/Users/Tejas/Desktop/ML/valData.txt',sep=' ',header=None)
validation_labels = pd.read_csv('C:/Users/Tejas/Desktop/ML/valLabels.txt',sep=' ',header=None)
features = pd.read_csv('C:/Users/Tejas/Desktop/ML/featureTypes.txt',sep=' ',header=None)

data_frame = pd.DataFrame(data=train_data)
data_val_frame=pd.DataFrame(data=validation_data)
data_test_frame = pd.DataFrame(data=test_data)

x=sparse.csr_matrix((data_frame[2], (data_frame[1]-1, data_frame[0]-1)))
x_validation=sparse.csr_matrix((data_val_frame[2], (data_val_frame[1]-1, data_val_frame[0]-1)))
x_test = sparse.csr_matrix((data_test_frame[2], (data_test_frame[1]-1, data_test_frame[0]-1)))

weights = np.random.rand(3000)
t = x.copy()
t.data**= 2
ak = 2 * np.sum(t,axis=1)
print ak.shape

d = x.shape[0]
n = x.shape[1]

print x.shape


# In[5]:


b = np.zeros(n)

y = np.array(train_labels[0])

val_labels = np.array(validation_labels[0])

lmbda=2*np.amax(abs(x.dot(y-np.sum(y)/float(n))))

rmse_old = float("inf")
rmse_new = 0
rmse_old_train = float("inf")
rmse_new_train = 0

count=0
lmbda_list = []
rmse_list = []
rmse_list_train = []
non_zeros = []
while rmse_old>rmse_new:
    loss_prev = float("inf")
    loss = 0
    while abs(loss-loss_prev) > 0.01:


        b_old = b.copy()
        r = y - np.transpose(x).dot(weights) - b
        b = np.full(n, (np.sum(r+b))/float(n))
        r = r + b_old - b
        #print r.shape
        r = r.reshape(n,1)
        #print r.shape

        for k in range(d):
            wk = weights[k]
            wk_old = wk
            ck = 2*x[k].dot((r + np.transpose(x[k]) * weights[k]))
            ck = ck.item(0)
            if ck < -lmbda:
                wk = (ck+lmbda)/float(ak[k])
            elif ck > lmbda:
                wk = (ck-lmbda)/float(ak[k])
            else:
                wk = 0

            r = r + np.transpose(x[k])*(wk_old-wk)
            weights[k] = wk

        loss_prev = loss
        loss  = np.sum((np.transpose(x).dot(weights) + b - y)**2) + (lmbda*np.sum(abs(weights)))
        #print "Loss --- ",loss, ' ' , loss_prev
    if count==0:
        rmse_old=float("inf")
        rmse_old_train = float("inf")
    else:
        rmse_old = rmse_new
        rmse_old_train = rmse_new_train

    count=count+1
    rmse_new = np.sqrt(np.sum(((val_labels-(np.transpose(x_validation).dot(weights)+b))**2))/n)
    rmse_new_train = np.sqrt(np.sum(((y-(np.transpose(x).dot(weights)+b))**2))/n)
 #   print rmse_old, ' ', rmse_new
    lmbda/=2
    print "Lambda --- ",lmbda, "RMSE VAL-- ",rmse_new, "RMSE TRAIN--", rmse_new_train
    lmbda_list.append(lmbda)
    rmse_list.append(rmse_new)
    rmse_list_train.append(rmse_new_train)
    print "Nonzero : ",np.count_nonzero(weights)
    non_zeros.append(np.count_nonzero(weights))

lmbda*=2


# In[3]:


weights = weights.reshape(d,1)
b_new = []


index_list = []
for i in range(x_test.shape[1]):
    index_list.append(i+1)
    b_new.append(b[0])

#b_new = np.array(b_new)
y_pred = np.transpose(x_test).dot(weights) + b_new

y_pred = np.array(y_pred.T)[0]



index_list = np.array(index_list)

#data = np.array([index_list,y_pred])
#print data
#print index_list.shape,y_pred.shape
#print data.shape
np.savetxt('C:/Users/Tejas/Desktop/np4.csv',zip(index_list,y_pred),fmt=['%d','%.2f'],delimiter=',',header='instanceId,Prediction')


# In[6]:


#fig = plt.figure()
plt.plot(lmbda_list,rmse_list)
plt.xlabel('Lambda')
plt.ylabel('RMSE_Validation')
plt.show()
plt.plot(lmbda_list,rmse_list_train)
plt.xlabel('Lambda')
plt.ylabel('RMSE_Training')

plt.show()
plt.plot(non_zeros,lmbda_list)
plt.xlabel('Lambda')
plt.ylabel('Non-Zeros')

plt.show()


# In[9]:


indices_top10 = []
indices_bottom10 = []
#print weights
indices_top10 = weights.argsort()[-10:][::-1]
indices_bottom10 = weights.argsort()[:10]
#print features[0]
for index in indices_top10:
    print features[0][index]

print "\n"
for index in indices_bottom10:
    print features[0][index]