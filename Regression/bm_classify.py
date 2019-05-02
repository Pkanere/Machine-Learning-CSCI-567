import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    #print(X)
    
    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
    
    for i in range(N):
        if(y[i]==0):
            y[i]=-1
                    
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        #w = np.zeros(D)
        #b = 0
        ############################################
        '''
        for iteration in range(max_iterations):
            total_delta=np.zeros(D)
            total_bias=0
            z=np.matmul(X,w)+b
            for i in range(N):
                
                if(y[i]*z[i]<=0):
                    total_delta=np.add(total_delta,y[i] * X[i])
                    total_bias+=y[i]
             
                
            w=np.add(w,step_size*(1/N)*total_delta)
            b=np.add(b,step_size*(1/N)*total_bias)
        '''
        
        for iteration in range(max_iterations):
                   
                z=np.matmul(X,w)+b
                for i in range(N):
                    
                    if(y[i]*z[i]<=0):
                        w= w + step_size*(1/N)*y[i] * X[i]
                        b= b + step_size*(1/N)*y[i]
            
            


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        #w = np.zeros(D)
        #b = 0
        ############################################
        
        for iteration in range(max_iterations):
            z = np.multiply(y,np.matmul(X,w)+b)
            
            val = np.exp(-1*z)*sigmoid(z)
            
            inc = step_size * (1/N) * val * y
            #print(inc)
            #print(z.shape,val.shape,increase.shape,X.T.shape)
            w= w + np.matmul(X.T,inc)
            #rint(inc,np.sum(inc))
            b=b+np.sum(inc)
            '''
            z = np.multiply(y,np.matmul(X,w)+b)
            
            val = np.exp(-1*z)*sigmoid(z)
            
            for i in range(N):
                increase = step_size*(1/N)* val[i] * y[i]
                #print(increase.shape, X[i].shape)
                w= w + increase * X[i]
                b= b + increase
            '''
            
                
        
        
    else:
        raise "Loss Function is undefined."
    #print(w,b)
    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    
    ############################################
    value = 1/(1+np.exp(-1*z))
               
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        
        ############################################

        
        preds = np.zeros(N)
        
        
        z=np.matmul(X,w) + b
        preds=sigmoid(z)
        
        for i in range(N):    
            
            if(preds[i]>=0.5):
                preds[i]=1
            else:
                preds[i]=0
       
    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        ############################################
        preds = np.zeros(N)
        preds=np.matmul(X,w) + b
        
        for i in range(N):    
             
            if(preds[i]>=0):
                preds[i]=1
            else:
                preds[i]=0
        
    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        #w = np.zeros((C, D))
        #b = np.zeros(C)
        ############################################
        for iteration in range(max_iterations):
   
            index = np.random.choice(N)
    
            val=np.matmul(w,X[index])+(b)
            val=val-max(val)
            val=np.exp(val)
            sm=np.sum(val)
    
            val=val/sm
    
            val[y[index]]-=1
            
            b=b-step_size*val
            w=w-step_size*np.matmul(val[np.newaxis].T,X[index][np.newaxis])

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        for iteration in range(max_iterations):
            val=np.matmul(w,X.T)+b[:,None]
    
            val=val-np.max(val,axis=0) #val c*n
            val=np.exp(val)
            sm=np.sum(val,axis=0)
    
            val=val/sm
    
            for i in range(N):
                val[y[i]][i]-=1
            
            b=b-step_size*(1/N)*np.sum(val,axis=1)
            w=w-step_size*(1/N)*np.matmul(val,X)
        
    
    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
   
    ############################################
    preds = np.zeros(N)
    pred_val=np.matmul(w,X.T) + b[:,None]
    preds = np.argmax(pred_val, axis=0)
    
    assert preds.shape == (N,)
    return preds




        