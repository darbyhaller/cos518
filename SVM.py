import numpy as np
import multiprocessing

class SVM(object):
    def __init__(self):
        self.W=None

    def svm_loss_vectorized(self, tuple):
        x, y, reg = tuple
        loss=0.0
        dW=np.zeros(self.W.shape)
    
        num_train=x.shape[0]
        scores=x.dot(self.W)
        margin=scores-scores[np.arange(num_train),y].reshape(num_train,1)+1
        margin[np.arange(num_train),y]=0.0
        margin=(margin>0)*margin
        loss+=margin.sum()/num_train
        loss+=0.5*reg*np.sum(self.W*self.W)
    
        margin=(margin>0)*1
        row_sum=np.sum(margin,axis=1)
        margin[np.arange(num_train),y]=-row_sum
        dW=x.T.dot(margin)/num_train+reg*self.W
    
        return loss,dW
    
    
    def train(self,x,y,reg=1e-5,learning_rate=1e-3,num_iters=100,batch_size=200,verbose=False):
        
        num_train,dim=x.shape
        num_class=np.max(y)+1
        if self.W is None:
            self.W=0.005*np.random.randn(dim,num_class)
        
        batch_x=None
        batch_y=None
        history_loss=[]
        n = 4
        p = multiprocessing.Pool(processes=n)
        for i in range(num_iters):
            mask=np.random.choice(num_train,batch_size,replace=False)
            batch_x=x[mask]
            batch_y=y[mask]
            
            batches = [(batch_x[int(i*batch_x.shape[0]/n): int((i+1)*batch_x.shape[0]/n)] \
                , batch_y[int(i*batch_y.shape[0]/n): int((i+1)*batch_y.shape[0]/n)] \
                , reg) for i in range(n)]
            gradsAndLosses = p.map(self.svm_loss_vectorized, batches)
            temp = [i[1] for i in gradsAndLosses]
            grad = np.mean(temp, axis=0)
            temp2 = [i[0] for i in gradsAndLosses]
            loss = np.mean(temp2, axis=0)
            print(loss)
            self.W+=-learning_rate*grad
            
            history_loss.append(loss)
            
            if verbose==True and i%100==0:
                print("iteratons:%d/%d,loss:%f"%(i,num_iters,loss))
            
        return history_loss
        
    def predict(self,x):
        y_pre=np.zeros(x.shape[0])
        scores=x.dot(self.W)
        y_pre=np.argmax(scores,axis=1)
        
        return y_pre
