# ArtClustering
This program enables you to perform k-means and k-mediods clustering on art pieces, while also performing a set of analytics, ie. dunn indexes and cluster purity.

To work, go to project2_students_final.py in the src folder and add the lines.

data = genfromtxt('toy_pca_data.csv', delimiter=',') #takes toy data and splits it up
# data = loadPixelFeatures() #can use data from actual pictures instead
X_c,mean = ml_split(data) #determine mean of each dimension and centered values of initial matrix
U= ml_compute_eigenvectors_SVD(X_c, m) # determine eignenvectors and as many, m, as you want dimensions in reconstructed Data
E = ml_pca(X_c, U) #perform pca for dimension reduction

output = ml_k_means(E,k, init_medoids_plus(E,k)) # example to perform k means clustering.

reconstructedData = ml_reconstruct(U, E,mean) # reconstruct the data, lower dimensional, to compare to initial dataset.
plot_pca(data, reconstructedData) # use the numerous plot functions to display the data.
