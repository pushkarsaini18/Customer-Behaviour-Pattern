# Clustering-Project

Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters.

# Types of clustering algorithms

### Hierarchical clustering
Hierarchical clustering involves creating clusters that have a predetermined ordering from top to bottom. For example, all files and folders on the hard disk are organized in a hierarchy. There are two types of hierarchical clustering, Divisive and Agglomerative

### K Means Clustering
K means is an iterative clustering algorithm that aims to find local maxima in each iteration. This algorithm works in these 5 steps :

Step 1: Specify the desired number of clusters K : Let us choose k=2 for these 5 data points in 2-D space.

Step 2: Randomly assign each data point to a cluster.

Step 3: Compute cluster centroids.

Step 4: Re-assign each point to the closest cluster centroid.

Step 5: Re-compute cluster centroids.

Repeat steps 4 and 5 until no improvements are possible : Similarly, we’ll repeat the 4th and 5th steps until we’ll reach global optima. When there will be no further switching of data points between two clusters for two successive repeats. It will mark the termination of the algorithm if not explicitly mentioned.

### Dbscan clustering
A cluster includes core points that are neighbors (i.e. reachable from one another) and all the border points of these core points. The required condition to form a cluster is to have at least one core point. Although very unlikely, we may have a cluster with only one core point and its border points.


# Project description
Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors, and concerns of different types of customers. Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.


# Project overview

### Week 1: Perform EDA
1. Check for unwanted columns, null values, replacing null values, duplicates etc.
2. Perform label encoding 
3. Uni-variate analysis with considering relationships with other variables.
4. Bi-variate analysis without considering relationships with other variables
5. Scaling and Normalization


### Week 2: Perform Clustering

##### --> Hierarchical Clustering

##### --> Kmeans clustering

##### --> DBSCAN Clustering


### Week 3: Model building

1. Feature Engineering

2. Basic Transformations

3. Data Pre- Processing

4. Model building 


### Week 4: Model Deployment Using Streamlit

1. Model Building

2. Creating a python script

3. Create front-end: Python

4. Model Deploy

