import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    #raise Exception(        'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    centers = [generator.randint(n)]
    #print(centers.shape, x.shape)
    while(n_cluster>1):
        dist = np.min(np.sum((x - x[centers][:,np.newaxis])**2,axis=2),axis=0)
        #centers = np.append(centers,np.argmax(dist/sum(dist)))
        centers.append(np.argmax(dist/sum(dist)))
        n_cluster-=1
    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        # self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)
        #self.centers.tolist()
        centroids = x[self.centers]
        #print("centroids", centroids)
        
        J=10**10
        n=len(x)
        iterations=0
        while(iterations<=self.max_iter):
            #print(iterations)
            
            dist = np.sum((x - centroids[:,np.newaxis])**2,axis=2)
            r_ik = np.argmin(dist,axis=0)
            J_new = np.sum(np.min(dist,axis=0))
            
            #print("Jnew",J_new)
            
            if(abs(J-J_new)<=self.e):
                break
            else:
                J=J_new
                centroids = np.zeros((self.n_cluster,D))
                #compute new centers
                for k in range(self.n_cluster):
                    new_centers = np.mean(x[r_ik==k],axis=0)         
                    centroids[k] = new_centers
            y=r_ik
            iterations +=1
            
    
            
        self.max_iter = iterations+1
        #centroids = centroids.tolist()
        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception( 'Implement fit function in KMeans class')
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception('Implement fit function in KMeansClassifier class')
        self.centers = centroid_func(N, self.n_cluster, x, self.generator)
        
        
        #self.centers.tolist()
        centroids = x[self.centers]
        #print("centroids", centroids)
        
        J=10**10
        
        iterations=1
        while(iterations<=self.max_iter):
            #print(iterations)
            
            dist = np.sum((x - centroids[:,np.newaxis])**2,axis=2)
            r_ik = np.argmin(dist,axis=0)
            J_new = np.sum(np.min(dist,axis=0))
            
            #print("Jnew",J_new)
            
            if(abs(J-J_new)<=self.e):
                break
            else:
                J=J_new
                centroids = np.zeros((self.n_cluster,D))
                #compute new centers
                for k in range(self.n_cluster):
                    new_centers = np.mean(x[r_ik==k],axis=0)         
                    centroids[k] = new_centers
            membership=r_ik
            iterations +=1
            
        centroid_labels = np.zeros(self.n_cluster)
        for c in range(self.n_cluster):
            dic = {}
            for i in range(N):
                if(membership[i]==c):
                    if(y[i] in dic):
                        dic[y[i]]+=1
                    else:
                        dic[y[i]]=1
            if(len(dic) ==0):
                centroid_labels[c] = 0
            else:
                centroid_labels[c] = max(dic, key=dic.get)

       
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        distance = np.zeros((N,self.n_cluster))
        for i in range(N):
            for c in range(self.n_cluster):
                distance[i][c] = sum([(a-b)**2 for a,b in zip(x[i],self.centroids[c])])
        #print(distance[0])
        labels = np.argmin(distance,axis = 1) 
        
        #raise Exception(    'Implement predict function in KMeansClassifier class')
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        #return np.array(labels)
        return self.centroid_labels[labels]

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    N = image.shape[0]
    M = image.shape[1]
    
    d=len(code_vectors)
    
    new_im = np.zeros((N,M,3))
    #distance = np.zeros((N*M,self.n_cluster))
    for i in range(N):
        for j in range(N):
            dist = sum([(a-b)**2 for a,b in zip(image[i][j],code_vectors[0])])
            index = 0
            
            for c in range(1,d):
                distance = sum([(a-b)**2 for a,b in zip(image[i][j],code_vectors[c])])
        
                if(distance<dist):
                    dist=distance
                    index = c
            new_im[i][j] = code_vectors[index]
            
    #raise Exception(             'Implement transform_image function')
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

