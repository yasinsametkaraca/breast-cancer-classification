# breast-cancer-classification

Wisconsin Diagnostic Breast Cancer (WDBC)
Abstract: Diagnostic Wisconsin Breast Cancer Database
Data Set Characteristics:  	Multivariate	Number of Instances:	569
Attribute Characteristics:	Real	Number of Attributes:	32
Associated Tasks:	Classification	Missing Values?	No

Predict whether the cancer is benign or malignant.
Problem: Cancer is one of the leading causes of human death in the world and has caused the death of approximately 9.6 million people in 2018. Breast cancer is the most common cause of cancer deaths in women. However, breast cancer is a type of cancer that can be treated when diagnosed early. The aim of this study is to identify cancer early in life by using machine learning methods.
Predicting field 2, diagnosis: B = benign, M = malignant. Sets are linearly separable using all 30 input features. Best predictive accuracy obtained using one separating plane in the 3-D space of Worst Area, Worst Smoothness and Mean Texture. 
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  They describe characteristics of the cell nuclei present in the image. A few of the images can be found at http://www.cs.wisc.edu/~street/images/
Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree.  Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes. The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34]. This database is also available through the UW CS ftp server:
-ftp ftp.cs.wisc.edu
-cd math-prog/cpo-dataset/machine-learn/WDBC/


Number of instances: 569
Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

Attribute information
1) ID number
2) Diagnosis (M = malignant, B = benign) 
3-32
Ten real-valued features are computed for each cell nucleus:
	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)
Several of the papers listed above contain detailed descriptions ofhow these features are computed. 
The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.  For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius. 
All feature values are recoded with four significant digits.
Missing attribute values: none
Class distribution: 357 benign, 212 malignant
RangeIndex: 569 entries, 0 to 568
dtypes: float64(30), int32(1)

