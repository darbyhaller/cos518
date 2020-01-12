import os
import numpy as np
import matplotlib.pyplot as plt
from SVM import SVM
from PIL import Image
from sklearn import svm as s
import time

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    

def load_CIFAR10():
    x_t=[]
    y_t=[]
    for i in range(1,6):
        path_train=os.path.join('cifar-10-batches-py','data_batch_%d'%(i))
        data_dict=unpickle(path_train)
        x=data_dict[b'data'].astype('float')
        y=np.array(data_dict[b'labels'])
        
        x_t.append(x)
        y_t.append(y)
        
    x_train=np.concatenate(x_t)
    y_train=np.concatenate(y_t)
    
    path_test=os.path.join('cifar-10-batches-py','test_batch')
    data_dict=unpickle(path_test)
    x_test=data_dict[b'data'].astype('float')
    y_test=np.array(data_dict[b'labels'])
    
    return x_train,y_train,x_test,y_test
    
def data_processing():
    x_train,y_train,x_test,y_test=load_CIFAR10()
    
    num_train=10000
    num_test=1000
    num_val=1000
    num_check=100
    
    x_tr=x_train[0:num_train]
    y_tr=y_train[0:num_train]

    x_val=x_train[num_train:(num_train+num_val)]
    y_val=y_train[num_train:(num_train+num_val)]

    x_te=x_test[0:num_test]
    y_te=y_test[0:num_test]

    mask=np.random.choice(num_train,num_check,replace=False)
    x_check=x_tr[mask]
    y_check=y_tr[mask]

    mean_img=np.mean(x_tr,axis=0)
    
    x_tr+=-mean_img
    x_val+=-mean_img
    x_te+=-mean_img
    x_check+=-mean_img
    
    x_tr=np.hstack((x_tr,np.ones((x_tr.shape[0],1))))
    x_val=np.hstack((x_val,np.ones((x_val.shape[0],1))))
    x_te=np.hstack((x_te,np.ones((x_te.shape[0],1))))
    x_check=np.hstack((x_check,np.ones((x_check.shape[0],1))))
    
    return x_tr,y_tr,x_val,y_val,x_te,y_te,x_check,y_check

def VisualizeWeights(best_W):
    w=best_W[:-1,:]
    w=w.T
    w=np.reshape(w,[10,3,32,32])
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes=len(classes)
    plt.figure(figsize=(12,8))
    for i in range(num_classes):
        plt.subplot(2,5,i+1)
        x=w[i]
        minw,maxw=np.min(x),np.max(x)
        wimg=(255*(x.squeeze()-minw)/(maxw-minw)).astype('uint8')
        
        r=Image.fromarray(wimg[0])
        g=Image.fromarray(wimg[1])
        b=Image.fromarray(wimg[2])
        wimg=Image.merge("RGB",(r,g,b))
        plt.imshow(wimg)
        plt.axis('off')
        plt.title(classes[i])
    
    
    
    
if __name__ == '__main__':
    x_train,y_train,x_val,y_val,x_test,y_test,x_check,y_check= data_processing()
    
    start=time.process_time()
    learning_rate=[7e-6,1e-7,3e-7]
    regularization_strength=[1e4,3e4,5e4,7e4,1e5,3e5,5e5]

    max_acc=-1.0
    for lr in learning_rate:
        for rs in regularization_strength:
            svm=SVM()
            history_loss=svm.train(x_train,y_train,reg=rs,learning_rate=lr,num_iters=2000)
            y_pre=svm.predict(x_val)
            acc=np.mean(y_pre==y_val)
            
            if(acc>max_acc):
                max_acc=acc
                best_learning_rate=lr
                best_regularization_strength=rs
                best_svm=svm
                
            print("learning_rate=%e,regularization_strength=%e,val_accury=%f"%(lr,rs,acc))
    print("max_accuracy=%f,best_learning_rate=%e,best_regularization_strength=%e"%(max_acc,best_learning_rate,best_regularization_strength))
    end=time.process_time()

    y_pre=best_svm.predict(x_test)
    acc=np.mean(y_pre==y_test)
    print('The test accuracy with self-realized svm is:%f'%(acc))
    print("\nProgram time of self-realized svm is:%ss"%(str(end-start)))
    
    VisualizeWeights(best_svm.W)
    
    start=time.process_time()
    lin_clf = s.LinearSVC()
    lin_clf.fit(x_train,y_train)
    y_pre=lin_clf.predict(x_test)
    acc=np.mean(y_pre==y_test)
    print("The test accuracy with svm.LinearSVC is:%f"%(acc))
    end=time.process_time()
    print("Program time of svm.LinearSVC is:%ss"%(str(end-start)))
