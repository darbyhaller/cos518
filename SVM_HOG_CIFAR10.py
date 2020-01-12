import numpy as np
from PIL import Image
from skimage import feature as ft
from SVM_CIFAR10 import load_CIFAR10
from SVM import SVM
import time
from sklearn import svm as s

def hog_extraction(data,size=8):
    num=data.shape[0]
    data=data.astype('uint8')
    data1_hogfeature=[]
    for i in range(num):
        x=data[i]
        r=Image.fromarray(x[0])
        g=Image.fromarray(x[1])
        b=Image.fromarray(x[2])
        
        img=Image.merge("RGB",(r,g,b))
        gray=img.convert('L')
        gray_array=np.array(gray)
        
        hogfeature=ft.hog(gray_array,pixels_per_cell=(size,size))

        data1_hogfeature.append(hogfeature)

    data_hogfeature=np.reshape(np.concatenate(data1_hogfeature),[num,-1])
    return data_hogfeature

    
if __name__=='__main__':
    
    start=time.process_time()
    x_tr,y_tr,x_te,y_te=load_CIFAR10()
    
    num_train=10000
    num_test=1000
    num_val=1000
    
    x_train=x_tr[0:num_train].reshape(num_train,3,32,32)
    y_train=y_tr[0:num_train]
    
    x_val=x_tr[num_train:(num_train+num_val)].reshape(num_val,3,32,32)
    y_val=y_tr[num_train:(num_train+num_val)]
    
    x_test=x_te[0:num_test].reshape(num_test,3,32,32)
    y_test=y_te[0:num_test]

    cellsize=[2,4,6,8,10]
    learning_rate=[1e-5,5e-5,1e-6,7e-6,1e-7,3e-7,1e-8]
    regularization_strength=[1e1,1e2,1e3,1e4,5e4,1e5,3e5,1e6,1e7]
    
    max_acc=-1.0
    for cs in cellsize:
        hog_train=hog_extraction(x_train,size=cs)
        hog_val=hog_extraction(x_val,size=cs)
        
        for lr in learning_rate:
            for rs in regularization_strength:
                svm=SVM()
                history_loss=svm.train(hog_train,y_train,reg=rs,learning_rate=lr,num_iters=2000)
                y_pre=svm.predict(hog_val)
                acc=np.mean(y_pre==y_val)
                
                if(acc>max_acc):
                    max_acc=acc
                    best_learning_rate=lr
                    best_regularization_strength=rs
                    best_cellsize=cs
                    best_svm=svm
                    
                print("cellsize=%d,learning_rate=%e,regularization_strength=%e,val_accury=%f"%(cs,lr,rs,acc))
    print("max_accuracy=%f,best_cellsize=%d,best_learning_rate=%e,best_regularization_strength=%e"%(max_acc,best_cellsize,best_learning_rate,best_regularization_strength))

    
    hog_test=hog_extraction(x_test,size=best_cellsize)
    y_pre=best_svm.predict(hog_test)
    acc=np.mean(y_pre==y_test)
    print('The test accuracy with self-realized svm is:%f'%(acc))

    end=time.process_time()
    print('Program time with self-realized svm is:%ss'%(str(end-start)))

    print('\n\n')
    
    start=time.clock()
    max_acc=-1.0
    for cs in cellsize:
        hog_train=hog_extraction(x_train,size=cs)
        hog_val=hog_extraction(x_val,size=cs)
        
        lin_clf = s.LinearSVC()
        lin_clf.fit(hog_train,y_train)
        y_pre=lin_clf.predict(hog_val)
        acc=np.mean(y_pre==y_val)
        
        if(acc>max_acc):
            max_acc=acc
            best_cellsize=cs
            best_s=lin_clf
        print("cellsize=%d,accuray=%f"%(cs,acc))
    print("max_accuray=%f,best_cellsize=%d"%(max_acc,best_cellsize))
        
    hog_test=hog_extraction(x_test,size=best_cellsize)
    y_pre=best_s.predict(hog_test)
    acc=np.mean(y_pre==y_test)
    print("The test accuracy with svm.LinearSVC is:%f"%(acc))
    end=time.clock()
    print("\nProgram time of svm.LinearSVC is:%ss"%(str(end-start)))






































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
