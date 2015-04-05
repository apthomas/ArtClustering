import numpy as np
from numpy import genfromtxt
import project2_students_final as p2

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
	
	return (X_centered.transpose(), mean)



def ml_compute_eigenvectors(X, m):
	cov_matrix = np.cov(X)
	
	# print len(X)
	# print len(X[0])
	# cov_matrix = np.dot(X,X.transpose())
	eig_val_cov, eig_vec_cov = np.linalg.eig(cov_matrix)
	
	# print len(eig_vec_cov)
	# print len(eig_vec_cov[0])
	# print "haha"
	eig_val_cov = eig_val_cov.transpose()
	eig_vec_cov = eig_vec_cov.transpose()

	idx = np.argsort(eig_val_cov)[::-1]
	eig_val_cov = eig_val_cov[idx]
	eig_vec_cov = eig_vec_cov[:, idx]
	topEV=np.zeros(shape=(m, len(X)))
	for i in range(0,m):
		# print eig_vec_cov[i]
		topEV[i] = eig_vec_cov[i]
	return topEV


	# top_eigenvectors = np.zeros((m, len(X[0])))
	# for i in range(0, m):
	# 	for j in range(0, len(X[0])):
	# 		top_eigenvectors[i][j] = eig_vec_cov[i][j]
	# return top_eigenvectors

def ml_pca(X, U):
	# 	X, an n x d Numpy array of n data points, each with d features.
	# 	U, a m x d matrix whose rows are the top m eigenvectors of XTX, in descending order of eigenvalues

	# returns E is an n x m matrix, whose rows represent low-dimensional feature vectors
	U_T = U.transpose()
	E = np.dot(X.transpose(),U_T)
	return E


# print pca
def ml_reconstruct(E, U, mean):
	# E, an n x m Numpy array of n data points, each with m featuresself.
	# U is a m x d matrix whose rows are the top m eigenvectors of XTX.
	# mean is a vector which is the mean of the data.

	# returns X recon is a n x d Numpy array of n reconstructed data points
	U_T = U.transpose()
	# U_T_inverse = np.linalg.ing(U_T)
	X_centered = E.dot(U)
	X_recon =  np.zeros((len(X_centered), len(X_centered[0])))
	for j in range(0, len(X_centered)):
		for i in range(0, len(X_centered[0])):
			X_recon[j][i] = X_centered[j][i] + mean[i]
	return X_recon

# print "overall, is it X?/?"
# X = [[1, 7],[3, 4]]
# X = np.array(X)
# val =  ml_split(X)
# eigs= ml_compute_eigenvectors(val[0], 2)
# pca = ml_pca(val[0], eigs)
# print ml_reconstruct(pca, eigs,val[1])

	
# A = [[0,1],[-2,3]]
# print ml_compute_eigenvectors(A

# toyData = genfromtxt('toy_pca_data.csv', delimiter=',')
data = p2.loadPixelFeatures()

X_c,mean = ml_split(data)
# eigs= ml_compute_eigenvectors(X_c, 1)

print eigs
pca = ml_pca(X_c, eigs)
reconstructedData = ml_reconstruct(pca, eigs,mean)
p2.plot_pca(toyData, reconstructedData)



def ml_k_means(X, k, init):
	# 	X, an n x d Numpy array of n data points, each with d features.
	# 	k, the number of desired clusters
	#	init, a k x d Numpy array of k data points, each with d features, which are the initial
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
			print iterations
			done = True
		init = centroids
		iterations = iterations+1
	return (centroids, clusterAssignments)


def ml_k_mediods(X, k, init):
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



def runK_means(k,tcd):
	array = tcd[0:k]
	return array

k=2
toyClusterData = genfromtxt('toy_cluster_data.csv', delimiter=',')
output = ml_k_means(toyClusterData,k, runK_means(k, toyClusterData))
p2.plot_2D_clusters(toyClusterData, output[1], output[0])
# # p2.plotArtworks()