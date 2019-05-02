import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    l=0
    for x in branches:
        l+=sum(x)
    weighted = 0
    import numpy as np
    for i in range(len(branches)):
        total=sum(branches[i])
        entropy=0
        for j in range(len(branches[i])):
            if(branches[i][j]==0):
                entropy+=0
            else:
                entropy+=-1*(branches[i][j]/total)*np.log2((branches[i][j]/total))
                       
        weighted+=(total/l)*entropy
    gain =  (S - weighted) # figure out what to pass to entropy set
    return gain


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    
    y_predict = decisionTree.predict(X_test)
    c_max=0
    for i in range(len(y_predict)):
                if(y_predict[i]==y_test[i]):
                    c_max+=1
   
    temp=[]
    lt=[]
    temp.append(decisionTree.root_node) #append root
    lt.append(decisionTree.root_node)
    #append all nodes to lt
    while(len(temp)):
        node = temp.pop(0)
        if(len(node.children)):
                
            for child in node.children:
                child.parent = node
                temp.append(child)
                lt.append(child)
    
    update_tree=False
    while(len(lt)):
        node = lt.pop(-1)
        if(len(node.children)):
                temp_children = node.children
                node.splittable=False
                node.children=[]
            
                #finding accuracy
                c=0
                y_predict = decisionTree.predict(X_test)
                ###print("pre",y_predict)
                ###print("actual", y_test)
                for i in range(len(y_predict)):
                    if(y_predict[i]==y_test[i]):
                        c+=1
                ###print("acc",c)
                
                ###print(c_max)
                if(c>c_max):
                    c_max=c
                    update_tree=True
                    
                #restore tree if accuracy decreased
                
                if(update_tree):
                    ###print("UPDATES")
                    node.splittable=False
                    node.children=[]
                else:
                    node.children = temp_children
                    node.splittable=True  
            
                update_tree=False
                
'''            for child in node.children:
                temp_children = node.children
                temp_cls_max=node.cls_max
                
                node.splittable=False
                node.children=[]
                node.cls_max=child.cls_max
                
                c=0
                y_predict = decisionTree.predict(X_test)
                
                for i in range(len(y_predict)):
                    if(y_predict[i]==y_test[i]):
                        c+=1
      
                if(c>c_max):
                    c_max=c
                    output = node.cls_max
                    update_tree=True
                    ret_pre = y_predict
                  
                #restore tree if accuracy decreased
                node.children = temp_children
                node.cls_max = temp_cls_max
                node.splittable=True
            
            if(update_tree):
                node.splittable=False
                node.children=[]
                #node.cls_max=output
            
            update_tree=False
                
            
        else:
            #replacing parent by its leaf
            pass
    
    #raise NotImplementedError

'''

# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    tp,fp,fn=0,0,0
    for x in range(len(real_labels)):
        tp += 1 if real_labels[x]==1 and predicted_labels[x]==1 else 0
        fp += 1 if real_labels[x]==0 and predicted_labels[x]==1 else 0
        fn += 1 if real_labels[x]==1 and predicted_labels[x]==0 else 0
        #print(tp,fp,fn)
    if tp+fp==0 or tp+fn==0 or tp==0:
        return 0
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    f1_score=(2*p*r)/(p+r)
    return(f1_score)
    #raise NotImplementedError

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    dis=0
    #print(point1,point2)
    for x in range(len(point1)):        
        dis += (point1[x] - point2[x])**2
    return (dis**0.5)
    


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    dis=0
    for x in range(len(point1)):
        dis += point1[x]*point2[x]
    return dis
    #raise NotImplementedError


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    dis=0
    for x in range(len(point1)):
        dis += (point1[x] - point2[x])**2
    dist = -np.e**(-0.5*dis)
    return dist
    #raise NotImplementedError


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    dis,len_train,len_test=0,0,0
    for x in range(len(point1)):
        dis += point1[x]*point2[x]
        len_train += point1[x]**2 
        len_test += point2[x]**2 
    denominator = (len_train**0.5)*(len_test**0.5) 
    return (1 - dis/denominator)
    #raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    #raise NotImplementedError
            
            best_valid_score = -1 
            best_k=0
            best_function=''
           
            if(len(ytrain)<30):
                p=len(ytrain)
            else:
                p=30
            for name,dst in distance_funcs.items():                
                
                for k in range(1,p,2):
                    model=KNN(k=k,distance_function=dst)
                    model.train(Xtrain,ytrain)                   
                    yval_predict = model.predict(Xval)
                    valid_f1_score = f1_score(yval,yval_predict)
                    
                    
            #Dont change any print statement
                    
                    if(valid_f1_score>best_valid_score):
                        best_valid_score=valid_f1_score
                        best_k=k
                        best_model=model
                        best_function=name
                       
            return best_model, best_k,  best_function


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    #raise NotImplementedError
        
        best_valid_score = -1 
        best_k=0
        best_function=''
        best_scaler=''
        
        if(len(ytrain)<30):
                p=len(ytrain)
        else:
                p=30
                
        for scaling_name,scale_func in scaling_classes.items():
            
                scaler = scale_func()
                train_features_scaled = scaler(Xtrain)
                val_features_scaled = scaler(Xval)

                for name,dst in distance_funcs.items():
                    
                    for k in range(1,p,2):
                        model=KNN(k=k,distance_function=dst)
                        model.train(train_features_scaled,ytrain)                        
                        yval_predict = model.predict(val_features_scaled)
                        valid_f1_score = f1_score(yval,yval_predict)
               
                        if(valid_f1_score>best_valid_score):
                            best_valid_score=valid_f1_score
                            best_k=k
                            best_model=model
                            best_function=name
                            best_scaler=scaling_name        
                
        return best_model, best_k,  best_function, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        output=[]
        for feature in features:
            dis=0
            for x in feature:
                dis+=x*x
            if(dis==0): 
                output.append(feature)
            else:
                normalized_feature = []
                denominator = dis**0.5
                for x in feature:
                    normalized_feature.append(x/denominator)
                output.append(normalized_feature)
        return output
        #raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.minimum_feature, self.maximum_feature= [], []
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if self.minimum_feature==[] or self.maximum_feature==[]:
            
            for j in range(len(features[0])):
                a=min(features[i][j] for i in range(len(features)))
                self.minimum_feature.append(a)
                b=max(features[i][j] for i in range(len(features)))
                self.maximum_feature.append(b)
          
        
        output=[]
        for feature in features:
            normalized_feature =[]
            for i in range(len(feature)):
                value = (feature[i]-self.minimum_feature[i])/(self.maximum_feature[i]-self.minimum_feature[i]) if self.maximum_feature[i]-self.minimum_feature[i]!=0 else 0
                normalized_feature.append(value)
            output.append(normalized_feature)
        return output
        #raise NotImplementedError





