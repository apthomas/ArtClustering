import csv
import numpy as np
import os

from os import listdir
from os.path import isfile, join
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import matplotlib.cm as cm
import random
from numpy import genfromtxt


# returns the feature set in a numpy ndarray
def loadCSV(filename):
    stuff = np.asarray(np.loadtxt(open(filename, 'rb'), delimiter=','))
    return stuff


# returns list of artist names
def getArtists(directory):
    return [name for name in os.listdir(directory)]


# loads all image files into memory
def loadImages():
    image_files = [f for f in listdir('../artworks_ordered_50') if f.endswith('.png')]
    images = []
    for f in image_files:
        images.append(mpimg.imread(os.path.join('../artworks_ordered_50', f)))
    return images

        
# convert color image to grayscale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


# creates a feature matrix using raw pixel values from all images, one image per row
def loadPixelFeatures():
    images = loadImages()
    X = []
    for img in images:
        img = rgb2gray(img)
        img = img.flatten()
        X.append(img)
    return np.array(X)


def ml_compute_eigenvectors_SVD(X,m):
    left, s, right = np.linalg.svd(np.matrix(X))    
    U = np.matrix.getA(right)    
    return (U[0:m])


#Colour function: helper function for plot_2D_clusters
def clr_function(labels):
    colors = []
    for i in range(len(labels)):
        if(labels[i] == 0):
            color = 'red'
        elif(labels[i] == 1):
             color = 'blue'
        elif(labels[i] == 2):
            color = 'green'
        elif(labels[i] == 3):
            color = 'yellow'
        elif(labels[i] == 4):
            color = 'orange'
        elif(labels[i] == 5):
            color = 'purple'
        elif(labels[i] == 6):
            color = 'greenyellow'
        elif(labels[i] == 7):
            color = 'brown'
        elif(labels[i] == 8):
            color = 'pink'
        elif(labels[i] == 9):
            color = 'silver'
        else:
            color = 'black'                
        colors.append(color)
    return colors


#Plot clusters of points in 2D
def plot_2D_clusters(X, clusterAssignments, cluster_centers):    
    
    points = X
    labels = clusterAssignments
    centers = cluster_centers
            
#    points = X.tolist()
#    labels = clusterAssignments.tolist()
#    centers = cluster_centers.tolist()
                                            
    N = len(points)
    K = len(centers)
    x_cors = []
    y_cors = []
    for i in range(N):
        x_cors.append( points[i][0] )
        y_cors.append( points[i][1] )
            
    plt.scatter(x_cors[0:N], y_cors[0:N], color = clr_function(labels[0:N]))                    
    plt.title('2D toy-data clustering visualization')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')    

    x_centers = [0]* K
    y_centers = [0]* K    
    for j in range(K):
        x_centers[j] = centers[j][0]
        y_centers[j] = centers[j][1]
        
    plt.scatter(x_centers, y_centers, color = 'black', marker = ',')
    plt.grid(True)
    plt.show()
    return


#Plot original and reconstructed points in 2D 
def plot_pca(X_original, X_recon):
    x_orig = []
    y_orig = []
    x_cors = []
    y_cors = []
    for i in range(len(X_original)):
        x_orig.append( X_original[i][0] )
        y_orig.append( X_original[i][1] )        
        x_cors.append( X_recon[i][0] )
        y_cors.append( X_recon[i][1] )                
    plt.title('2D toy-data PCA visualization')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')   
    plt.axis('equal')   #Suggestion: Try removing this command and see what happens!
                            
    plt.scatter(x_orig, y_orig, color = 'red' )    
    plt.scatter(x_cors, y_cors, color = 'green', marker = ',')        
    plt.grid(True)    
    plt.show()
    return            


# display paintings by artist, one artist per matplotlib figure
def plotArtworks():
    artists = getArtists('../selected_subset')
    figure_count = 0
    for artist in artists:
        artist_dir = os.path.join('../', 'selected_subset', artist)
        image_files = [f for f in listdir(artist_dir) if f.endswith('.png')]
        print image_files
        n_row = math.floor(math.sqrt(len(image_files)))
        n_col = math.ceil(len(image_files)/n_row)
        fig = plt.figure(figure_count)
        fig.canvas.set_window_title(artist)
        for i in range(len(image_files)):
            plt.subplot(n_row, n_col,i+1)
            plt.axis('off')
            plt.imshow(mpimg.imread(os.path.join(artist_dir, image_files[i])))
        figure_count += 1
    plt.show()

# creates a dictionary mapping cluster label to indices of X that belong to that cluster
def create_cluster_dict(cluster_labels):
    clusters = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in clusters.keys():
            clusters[cluster_labels[i]] = [i]
        else:
            clusters[cluster_labels[i]].append(i)
    return clusters


# plots clusters of images
def plotClusters(cluster_labels):
    ordered_artist_dir = os.path.join('../', 'artworks_ordered_50')
    image_files = [f for f in listdir(ordered_artist_dir) if f.endswith('.png')]
    clusters = create_cluster_dict(cluster_labels)
    figure_count = 0
    for key in clusters.keys():
        n_row = math.floor(math.sqrt(len(clusters[key])))
        n_col = math.ceil(len(clusters[key])/n_row)
        fig = plt.figure(figure_count)
        fig.canvas.set_window_title(str(key))
        for i in range(len(clusters[key])):
            plt.subplot(n_row, n_col, i+1)
            plt.axis('off')
            plt.imshow(mpimg.imread(os.path.join(ordered_artist_dir, image_files[clusters[key][i]])))
        figure_count += 1
    plt.show()


# displays images specified in labeled.csv after reconstruction (grayscale)
# Input:
    # matrix of pixel values, one image per row
# Output:
    # plot of the selected images in labeled.csv
def plotGallery(reconstruction):
    ordered_artist_dir = os.path.join('../', 'artworks_ordered_50')
    image_files = [f for f in listdir(ordered_artist_dir) if f.endswith('.png')]
    indices = loadCSV('labeled.csv')
    num_images = len(indices)
    n_row = math.floor(math.sqrt(num_images))
    n_col = math.ceil(num_images/n_row)
    for i in range(indices.shape[0]):
        plt.subplot(10, 5, i+1)
        plt.axis('off')
        img = np.reshape(reconstruction[int(indices[i])-1], (50,50))
        plt.imshow(img, cmap=cm.gray)
    plt.show()


# computes the majority vote for each cluster
# Input:
    # list of labels of each row in the feature matrix
    # fraction of data points that are labeled, defaults to 1 (all points have labels)
# Output:
    # a dictionary, with (key, value) = (cluster_label, majority)
def majorityVote(cluster_labels, labeled = 100):
    artist_labels = loadCSV('artist_labels_' + str(labeled) + '.csv')
    clusters = create_cluster_dict(cluster_labels)
    majorities = {} 
    for key in clusters.keys():
        votes = []
        for i in range(len(clusters[key])):
            label = artist_labels[clusters[key][i]]
            if label != -1:
                votes.append(label)
        if len(votes) == 0:
            votes.append(-1)
        votes = np.array(votes)
        majorities[key] = stats.mode(votes)[0][0]
    return majorities


# returns the total number of classification errors, comparing the majority vote label to true label
def computeClusterPurity(cluster_labels, majorities=None):
    if majorities == None:
        majorities = majorityVote(cluster_labels)
    artist_labels = loadCSV('artist_labels.csv')
    clusters = create_cluster_dict(cluster_labels)
    errors = 0 
    for key in clusters.keys():
        majority = majorities[key]
        for i in range(len(clusters[key])):
            if artist_labels[clusters[key][i]] != majority:
                errors += 1
    return 1-(float(errors)/float(len(cluster_labels)))


# computes the majority vote for each cluster
# Input:
    # list of labels of each row in the feature matrix
    # fraction of data you have labeled (valid inputs: 5, 15, 25, 50, 75, 100)
# Output:
    # classification accuracy
def classifyUnlabeledData(cluster_labels, labeled):
    majorities = majorityVote(cluster_labels, labeled)
    acc = computeClusterPurity(cluster_labels, majorities)
    return acc

# computes the maximum pairwise distance within a cluster
def intraclusterDist(cluster_values):
    max_dist = 0.0 
    for i in range(len(cluster_values)):
        for j in range(len(cluster_values)):
            dist = np.linalg.norm(cluster_values[i]-cluster_values[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


# helper function for Dunn Index
def index_to_values(indices, dataset):
    output = []
    for index in indices:
        output.append(dataset[index])
    return np.matrix(output)


# computes the Dunn Index, as specified in the project description
# Input:
    # cluster_centers - list of cluster centroids
    # cluster_labels - list of labels of each row in feature matrix
    # features - feature matrix 
# Output:
    # dunn index (float)
def computeDunnIndex(cluster_centers, cluster_labels, features):  
    clusters = create_cluster_dict(cluster_labels)
    index = float('inf')  
    max_intra_dist = 0.0
    # find maximum intracluster distance across all clusters
    for i in range(len(cluster_centers)):
        cluster_values = index_to_values(clusters[i], features)
        intracluster_d = float(intraclusterDist(cluster_values))
        if intracluster_d > max_intra_dist:
            max_intra_dist = intracluster_d

    # perform minimization of ratio
    for i in range(len(cluster_centers)):
        inner_min = float('inf')
        for j in range(len(cluster_centers)):
            if i != j:
                intercluster_d = float(np.linalg.norm(cluster_centers[i]-cluster_centers[j]))
                ratio = intercluster_d/max_intra_dist
                if ratio < inner_min:
                    inner_min = ratio
        if inner_min < index:
            index = inner_min
    return index

#helper function for init_medoids_plus
def weighted_choice(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i

#helper function for init_medoids_plus
def square_distance(point1, point2):
    value = 0.0    
    for i in range(0,len(point1)):
        value += (point1[i] - point2[i])**2    
    return value

#Function for generating initial centers uniformly at random (without replacement) from the data
def init_medoids(X, K):        
    points = X
    N = len(points)
    indices = range(0,N)
    centers = list([0]*K)
        
    for j in range(0,K):            
        temp = random.randrange(0,N-j)       
        centers[j] = indices[temp]
        del indices[temp]    

    medoids = []        
                        
    for j in range(0,K):
        medoids.append(points[centers[j]])
    
    return medoids


#Function for generating initial centers according to the KMeans++ initializer
def init_medoids_plus(X, K):        
    points = X
    N = len(points)
    indices = range(0,N)
    medoids = []
    
    for j in range(0,K):            
        if(j == 0):
            temp = random.randrange(0,N)       
            medoids.append(points[indices[temp]])
            del indices[temp]
            continue
        
        weights = []
        for i in range(len(indices)):
            weights.append(square_distance(medoids[0],points[indices[i]]))        
        
        
        for i in range(len(indices)):
            for c in medoids:
                if(square_distance(c, points[indices[i]]) < weights[i]):
                    weights[i] = square_distance(c, points[indices[i]])
                    
        temp = weighted_choice(weights)
        medoids.append(points[indices[temp]])    
        del indices[temp]
         
    return medoids


#Function for generating initial centers according to the KMeans++ initializer #This is a faster version 
# def init_medoids_plus(X, K): 
#     points = X 
#     N = len(points) 
#     indices = range(0,N) 
#     medoids = [] 
#     weights = [] 
#     #initialize first medoid 
#     temp = random.randrange(0,N) 
#     medoids.append(list(points[indices[temp]])) 
#     del indices[temp] 
#     for i in range(len(indices)): 
#         weights.append(square_distance(medoids[0],points[indices[i]])) 
#     for j in range(0,K): 
#         if(j == 0): 
#             continue
#         for i in range(len(indices)): 
#             c = medoids[j-1] 
#             if(square_distance(c, points[indices[i]]) < weights[i]): 
#                 weights[i] = square_distance(c, points[indices[i]]) 
#     temp = weighted_choice(weights) 
#     medoids.append(list(points[indices[temp]])) 
#     del indices[temp] 
#     del weights[temp] 
#     # return np.array(medoids)
#     return medoids


def ml_split(X):
    # X, an n x d Numpy array of n data points, each with d features.

    # returns 
    # X centered is an n x d Numpy array, such that X centered[i; j] = X[i; j] - mean[j] .
    # mean is a d x 1 Numpy array,
    mean = np.mean(X, axis=0)
    
    X_centered =  np.zeros((len(X), len(X[0])))

    for j in range(0, len(X_centered)):
        for i in range(0, len(X_centered[0])):
            X_centered[j][i] = X[j][i] - mean[i]
 
    return (X_centered, mean)
    
    
def ml_compute_eigenvectors(X,m):
    cov_matrix = np.cov(X.transpose())
    
    # cov_matrix = np.dot(X,X.transpose())
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_matrix)
    
    eig_val_cov = eig_val_cov.transpose()
    eig_vec_cov = eig_vec_cov.transpose()

    idx = np.argsort(eig_val_cov)[::-1]
    eig_val_cov = eig_val_cov[idx]
    eig_vec_cov = eig_vec_cov[:, idx]
    topEV=np.zeros(shape=(m, len(X.transpose())))
    for i in range(0,m):
        # print eig_vec_cov[i]
        topEV[i] = eig_vec_cov[i]

    return topEV


def ml_pca(X, U):
    #   X, an n x d Numpy array of n data points, each with d features.
    #   U, a m x d matrix whose rows are the top m eigenvectors of XTX, in descending order of eigenvalues

    # returns E is an n x m matrix, whose rows represent low-dimensional feature vectors
    U_T = U.transpose()
    # print len(X.transpose())
    # print len(X.transpose()[0])
    # print len(U_T)
    # print len(U_T[0])
    # print "pca"
    E = np.dot(X,U_T)
    
    return E


def ml_reconstruct(U, E, mean):
    # E, an n x m Numpy array of n data points, each with m featuresself.
    # U is a m x d matrix whose rows are the top m eigenvectors of XTX.
    # mean is a vector which is the mean of the data.

    # returns X recon is a n x d Numpy array of n reconstructed data points
    U_T = U.transpose()
    # U_T_inverse = np.linalg.ing(U_T)
    # print (len(E))
    # print len(E[0])
    # print len(U)
    # print len(U[0])
    X_centered = E.dot(U)
    X_recon =  np.zeros((len(X_centered), len(X_centered[0])))
    for j in range(0, len(X_centered)):
        for i in range(0, len(X_centered[0])):
            X_recon[j][i] = X_centered[j][i] + mean[i]
    return X_recon


def ml_k_means(X, k, init):
    #   X, an n x d Numpy array of n data points, each with d features.
    #   k, the number of desired clusters
    #   init, a k x d Numpy array of k data points, each with d features, which are the initial
    # guesses for the centroids.

    # returns centroids is a k x d Numpy array of k points, each with d features, indicating the final
    # centroids.
    # clusterAssignments is an n x1 Numpy array of integers, each between 0 and k -1,
    # indicating which cluster the input points are assigned to.
    iterations = 0
    done=False
    init = np.array(init)
    while (done!=True):
        clusterAssignments=[]
        centroids= []
        # assign data points to cluster
        for i in range(0, len(X)):
            minDist = np.linalg.norm(X[i]-init[0])
            lowestCluster=0
            for j in range(0, len(init)):
                dist = np.linalg.norm(X[i]-init[j])
                if (dist<minDist):
                    minDist=dist
                    lowestCluster=j
            clusterAssignments.append(lowestCluster)
        # reevaluate means
        for a in range(0,k):
            arr = []
            for b in range(0, len(clusterAssignments)):
                if(clusterAssignments[b]==a):
                    arr.append(X[b])
            arr = np.array(arr)
            mean = np.mean(arr, axis=0)
            centroids.append(mean)
        if (np.array_equal(init,centroids)):
            done = True
        init = centroids
        iterations = iterations+1
    return (centroids, clusterAssignments)
        

def ml_k_medoids(X, k, init):    
    iterations = 0
    done=False
    init = np.array(init)
    while (done!=True):
        clusterAssignments=[]
        centroids= []
        mediods=[]
        # assign data points to cluster
        for i in range(0, len(X)):
            minDist = np.linalg.norm(X[i]-init[0])
            lowestCluster=0
            for j in range(0, len(init)):
                dist = np.linalg.norm(X[i]-init[j])
                if (dist<minDist):
                    minDist=dist
                    lowestCluster=j
            clusterAssignments.append(lowestCluster)

        for a in range(0,k):
            lowestDist=100000000000000000000000000000000000000000000000000000000000000000000000.0 #arbitrarily high number
            mediodPoint=-1
            for b in range(0, len(clusterAssignments)):
                totalDist=0
                if (clusterAssignments[b]==a):
                    for c in range(0, len(clusterAssignments)):
                        if (clusterAssignments[b] ==clusterAssignments[c]):
                            totalDist = totalDist+np.linalg.norm(X[b]-X[c])
                    if (totalDist < lowestDist):
                        lowestDist = totalDist
                        mediodPoint = b
            mediods.append(X[mediodPoint])
        if (np.array_equal(init,mediods)):
            print iterations
            done = True
        init = mediods
        iterations = iterations+1
    return (mediods, clusterAssignments)


# data = genfromtxt('toy_pca_data.csv', delimiter=',')
# # data = loadPixelFeatures()\
# X_c,mean = ml_split(data)
# U= ml_compute_eigenvectors(X_c, 1)

# E = ml_pca(X_c, U)
# reconstructedData = ml_reconstruct(U, E,mean)
# plot_pca(data, reconstructedData)
# # reconstructedData = reconstructedData.transpose()
# plotGallery(reconstructedData)

def runK_means(k,tcd):
    array = tcd[0:k]
    return array

# k=4
# toyClusterData = genfromtxt('toy_cluster_data.csv', delimiter=',')
# output = ml_k_means(toyClusterData,k, runK_means(k, toyClusterData))
# plot_2D_clusters(toyClusterData, output[1], output[0])
    
# plotArtworks()

# Question 6
# data = loadPixelFeatures()
# data = loadCSV("gist_features.csv")
data = loadCSV("deep_features.csv")

def run_q_6(m, k,data,typeD, portion):
    total = 0
    X_c,mean = ml_split(data)
    eigs= ml_compute_eigenvectors_SVD(X_c.transpose(), m)
    E = ml_pca(X_c, eigs)

    # reconstructedData = ml_reconstruct(eigs, E,mean)
    output = ml_k_means(E,k, init_medoids_plus(E,k))
    
    # print "type is: "+typeD
    # print "m is: "+str(m)+" and k is: "+str(k)
    print "portion of dataset: "+str(portion)+"% and k is: "+str(k)

    # print "Dunn Index: "+str(computeDunnIndex(output[0], output[1], E))
    # print "Cluster Purity: "+ str(computeClusterPurity(output[1]))
    # plotClusters(output[1])
    total = total + classifyUnlabeledData(output[1], portion)
    output = ml_k_means(E,k, init_medoids_plus(E,k))

    total = total + classifyUnlabeledData(output[1], portion)
    output = ml_k_means(E,k, init_medoids_plus(E,k))

    total = total + classifyUnlabeledData(output[1], portion)
    print "accuracy is: "+str(total/3.0)

# run_q_6(200,70,data,"CNN", 5)
# run_q_6(200,50,data,"CNN", 15)
# run_q_6(200,50,data,"CNN", 25)
# run_q_6(200,50,data,"CNN", 50)
# run_q_6(200,70,data,"CNN", 75)
# run_q_6(200,70,data,"CNN", 100)