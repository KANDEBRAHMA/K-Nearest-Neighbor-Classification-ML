# K-Nearest-Neighbor Classification-ML
 Implemented KNN in Python

### K-Nearest Neighbors Classification

In machine learning world, K-Nearest Neighbors Classification model has prominent role.
My approach to this problem is there are two different metrics used to calculate distance for the features of the dataset.
    * Manhattan distance
        * Manhattan distance is calculated as the sum of the absolute differences between the two vectors.
    * Eucledian Distance
        * Eucledian Distance is calculated from the Cartesian vectors of the datapoints using the Pythagorean theorem.
    
* Fitting the train dataset
    * I have encorporated data features along with respective target class for better access later.

* Predicting the target class for test dataset:
    * For each datapoint in test:
        * I have calculated distance based on the metrics provided for all datapoints in the train dataset.
        * From all the distances we got, taking the k-neighbors with least distance along with their target class.
            * If the distance parameter is uniform, we rank them uniformly and return most occurred target class among them and predict that this datapoint assign to that particular target class.
            * If the distance paramter is distance, we assigns weights proportional to the inverse of the distance from the test sample to each neighbor.
                * After the assigning weights in above fashion, we sum of all the weights for a class variable and assign the datapoint to target class that has highest weight.