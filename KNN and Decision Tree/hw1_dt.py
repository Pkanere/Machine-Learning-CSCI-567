import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
       
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size
        
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)        
        if self.root_node.splittable:           
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        
        y_pred = []       
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
 
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
            #print("False")
        else:
            self.splittable = True
            #print("True")
        
        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        #raise NotImplementedError
        #calculating parent entropy
        label_count= np.array(np.unique(self.labels,return_counts=True)).T
        total=np.sum([int(label_count[i][1]) for i in range(len(label_count))])
        entropy_parent=np.sum([-1*(int(label_count[i][1])/total)*np.log2((int(label_count[i][1])/total)) for i in range(len(label_count))])
        
        #print("entropy parent", entropy_parent)
       
        #maximum gain
        max_gain = -1
        for j in range(len(self.features[0])):
            
                split_attr=[self.features[i][j] for i in range(len(self.features))]           
                split_unique = np.unique(split_attr)
                unique_labels = np.unique(self.labels)
                branches = np.zeros((len(split_unique), len(unique_labels)),dtype='int')
                #print(branches)

                for i in range(len(split_attr)):
                   
                    attr_index = list(split_unique).index(split_attr[i])
                    class_index = list(unique_labels).index(self.labels[i])
                    branches[attr_index][class_index]+=1
                    
                gain=Util.Information_Gain(entropy_parent,branches)
                #print(gain)
                
                if(gain>max_gain):
                    max_gain=gain
                    self.dim_split = j 
                    self.feature_uniq_split = split_unique
                
                if(gain==max_gain):
                    if(len(split_unique) > len(self.feature_uniq_split)):                        
                        self.dim_split = j 
                        self.feature_uniq_split = split_unique
        
        #print("node",self.dim_split,self.feature_uniq_split)
        
        #split child
        
        for f in self.feature_uniq_split:
       
            sub_data=[]
            sub_label=[]    
            
            for j in range(len(self.features)):
                if(self.features[j][self.dim_split]==f):
                    sub_feature = self.features[j][:self.dim_split]+self.features[j][self.dim_split+1:]
                    sub_data.append(sub_feature)
                    sub_label.append(self.labels[j])
                 
            child=TreeNode(sub_data, sub_label, self.num_cls) #what is num_cls
            
            if(all(child.features[0][j] == None for j in range(len(child.features[0])))):
                child.splittable=False
          
            self.children.append(child)
            
        for child in self.children:
            if child.splittable:
                child.split()
        return

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        #raise NotImplementedError
        if self.splittable:
            child_value = feature[self.dim_split]
            sub_data = feature[:self.dim_split]+feature[self.dim_split+1:]
            
            flag=0
            for i in range(len(self.feature_uniq_split)):         
                if(self.feature_uniq_split[i] == child_value):
                           child_index = i
                           flag = 1
            if(flag==0):
                #print(self.cls_max)
                return self.cls_max
            
            return self.children[child_index].predict(sub_data)
        else:
            #print (self.cls_max)
            return self.cls_max
