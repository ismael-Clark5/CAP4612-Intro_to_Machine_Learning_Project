<div style="center">
<h1> Feature Selection Using LASSO, Recursive Feature Elimination, and SelectFromModel </h1>

 <p2> Ismael Clark Antonela Radas Hector Ramirez

 CAP4612-Introduction to Machine Learning

 Florida International University
 </p2>

</div>
<div style="page-break-after: always;"></div>

# Abstract
This paper covers the use of several supervised learning algorithms in a  feature selection task on a dataset. This dataset contains normalized long non-coding RNA (lncRNA) expression profiles of 12 cancer types. Each algorithm yields a set of features which are then used by three different classifiers. To check the performance of the classification (for each set of features), the results are evaluated using different performance metrics. 

<div style="page-break-after: always;"></div>

# Introduction
Feature selection is the process of selecting a subset of relevant features (variables, predictors) for use in model construction. Among the most common applications of this technique are: 
 - simplification of models to make them easier to interpret by researchers/users,
 - shorter training times,
 - to avoid the curse of dimensionality,
 - improve data's compatibility with a learning model class,
 - encode inherent symmetries present in the input space.

Data contains some features that are either redundant or irrelevant and can thus be removed without incurring much loss of information, which is the central idea in feature selection. Redundant and irrelevant are two distinct notions, since one relevant feature may be redundant in the presence of another relevant feature with which it is strongly correlated. 

Another technique that shares the general goal of reducing the number of features with feature selection, but that differs in its implementation is feature extraction. Feature extraction creates new features from functions of the original features, whereas feature selection returns a subset of the features. 

Among the different feature selection techniques figure: 

- Unsupervised Algorithms:
    - Multi-cluster feature selection (MCFS)
     - Unsupervised discriminative feature selection (UDFS)
    - Autoencoder
- Supervised Algorithms:
    - LASSO
    - SelectFromModel (python package)
    - Recursive Feature Elimination

The algorithms from the supervised learning category were used during this project.
 
# Feature Selection Models

## LASSO
Least Absolute Shrinkage and Selection Operator (Lasso)
LASSO is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical model. LASSO was originally formulated for linear regression models. However, it is easily extended to other statistical models including generalized linear models, generalized estimating equations, proportional hazards models, and M-estimators.

For its implementation, LASSO’s cost function is defined as:

<!-- $$
  {1 \over 2N_{training} } {\sum ^{N_{training}} _{i=1} (y_{real} ^{i} - y_{predict} ^i)} = \alpha {\sum _{j=1} ^{n} | \alpha _{j} |}
$$ --> 

<div align="center"><img src="svg/CuraKlP9yk.svg"></div>

where $\alpha _{j}$ is the coefficient of the j-th feature. The final term is called l1 penalty and α is a hyperparameter that tunes the intensity of this penalty term. The higher the coefficient of a feature, the higher the value of the cost function. This model seeks to optimize the cost function reducing the value of the coefficient. The values must be scaled to achieve meaningful results. 

While implementing the model, using the LassoCV library from the sklearn.linear_model package. Also,  <!-- $a_i=i-1+0.1$ --> <img src="svg/n7rvM00GHn.svg"> | <!-- $0=0.01\space and\space i \in [0,10)$ --> <img  src="svg/ID7bLmmnR1.svg">. Each one of these values of alpha yielded a set of features which were stored for later analysis.

## Recursive Feature Elimination (RFE)

Recursive feature elimination is a feature selection algorithm that selects features based on how they affect the performance of a particular model. It reduces model complexity by removing features one by one until the optimal number of features is left. The algorithm can wrap around any model, and it produces the best possible set of features that gives the highest performance.

For its implementation, the RFE class  was used, taken from the sklearn.feature_selection package. Logistic Regression was used as the estimator,  as required by the class. An estimator is the model whose performance will be evaluated for a set of features returned from the algorithm. 

As specified in the project instructions, the number of feature to select <!-- $(n)$ --> <img src="svg/zoYo6oVbqz.svg">  by the algorithm in each iteration is defined as 
<p>
<!-- $n_i=n_{i-1}$ --> <img src="svg/0CBI9wtYAp.svg">|<!-- $n_0=20$ --> <img src="svg/NAK0HMNjd7.svg"> and <!-- $i\epsilon[0,5)$ --> <img src="svg/3VUhepUYee.svg"> </p>

## SelectFromModel (SFM)
SelectFromModel is a python class that acts as a meta converter that selects features based on importance weights. The post-fit evaluator must have feature_importances_ or coef_ attributes. [Sources: 3, 9]. Once set up, the method needs to be adapted for the training set and with the get_support attribute the  importance of features can be controlled.
Like the RFE, SelectFromModel from Scikit-Learn is based on a Machine Learning Model estimation for selecting the features. The differences are that SelectFromModel feature selection is based on the importance attribute (often is coef_ or feature_importances_ but it could be any callable) threshold. By default, the threshold is the mean.
As specified in the project instructions, the number of feature to select <!-- $(n)$ --> <img src="svg/UyTcryxF27.svg">  by the algorithm in each iteration is defined as
<p>
 <!-- $n_i=n_{i -1}$ --> <img src="svg/IhiBA4saXC.svg"> | <!-- $n_0=20$ --> <img src="svg/BldswAeciU.svg"> and <!-- $i\epsilon[0,5)$ --> <img src="svg/NC9Y4jXTFt.svg"> </p>

# Classifiers 
## K-Nearest Neighbors (KNN)
A supervised learning and model classification algorithm that helps us find which class a new input belongs to when the k nearest neighbors are selected and the distance between them is calculated. KNN, also known as K-Nearest Neighbor, is a supervised learning and model classification algorithm that helps us determine which class a new input belongs to (test value) when k nearest neighbors are selected and the distance between them. The k-nearest neighbors (KNN) algorithm is a data classification technique for estimating the likelihood that a data point will become a member of a given group, based on the group to which the nearest data points belong. The k-nearest neighbors (KNN) algorithm is a simple and easy-to-implement supervised machine learning algorithm that can be used to solve classification and regression problems. 
    
Supervised machine learning (as opposed to unsupervised machine learning) relies on labeled inputs to learn a function that produces appropriate output when new untagged data is provided. The unsupervised machine learning algorithm uses untagged inputs, in other words, no teacher (shortcut) tells the child (computer) when he is right or when he made a mistake so that he can correct himself. 

## Support Vector Machine (SVM)

In its basic form, linear partitioning, the SVM tries to find a line that maximizes the separation between the dataset in two 2D spatial point classes. SVM classifies data by finding the best hyperplane that separates all data points of one class from those of another class. The best hyperplane for an SVM is the hyperplane with the largest gap between the two classes. In the SVM algorithm, we are trying to maximize the headroom between data points and hyperplane. 
The distance between the vector and the hyperplane is called the margin. The distance from the vector to the hyperplane is called the field, which is the distance between the line and the closest point in the class. The hyperplane (line) passes through the maximum field, which is the maximum distance between the data points of the two classes. Then the classification is completed by finding the hyperplane that best distinguishes the two classes. 

## Random Forest (RF)
While an estimate of the prediction accuracy is used for the importance of the permutation, in random forests this Gini admixture can be used to measure the informativeness of a function in a model. Swap importance is a measure that tracks the accuracy of a forecast in which variables are randomly swapped from over-the-counter samples. Tree-based algorithms typically use mean for continuous functions or mode for categorical functions when predicting training samples in the regions to which they belong. 
    
They can also make predictions with high accuracy, stability, and easy interpretation. Therefore, every data scientist should learn these algorithms and use them in their machine learning projects. 
       
To facilitate the systematic and standardized refinement of somatic variants based on cancer sequencing data, random forest (RF) models and a deep learning (DL) approach have been used to demonstrate that these machine learning methods can provide classification efficiency. The classes are high and the same in all refinements of the variants (Ainscough et al., 2018). A machine learning approach called Cerebro improved the accuracy of invoking verified somatic mutations in tumor samples and outperformed many other methods for detecting somatic mutations (Wood et al., 2018). The potential for CNNs to use crude nucleotide sequences to classify cancer mutations was initially explored by using tag coding, hot coding, and embedding to preprocess DNA information. 

# Results and Evaluation
## Lasso
### t-SNE
<img src='./images/LASSO TSNE 1.png'/>

<img src='./images/LASSO TSNE 2.png'/>

<img src='./images/LASSO TSNE 3.png'/>

### Performance Matrics 

<img src='./images/LASSO-KNN-Performance-Metrics.png'/>

<img src='./images/LASSO-SVM-Performance-Metrics.png'/>

<img src='./images/LASSO-RF-Performance-Metrics.png'/>

### Confusion Matrix

<img src='./images/LASSO-KNN-Confusion-Matrix-1.png'/> 

<img src='./images/LASSO-KNN-Confusion-Matrix-2.png'/> 

<img src='./images/LASSO-KNN-Confusion-Matrix-3.png'/> 

<img src='./images/LASSO-SVM-Confusion-Matrix-1.png'/> 

<img src='./images/LASSO-SVM-Confusion-Matrix-2.png'/>

<img src='./images/LASSO-SVM-Confusion-Matrix-3.png'/> 

<img src='./images/LASSO-RF-Confusion-Matrix-1.png'/> 

<img src='./images/LASSO-RF-Confusion-Matrix-2.png'/> 

<img src='./images/LASSO-RF-Confusion-Matrix-3.png'/> 

### ROC Curves 

<img src='./images/LASSO-KNN-ROC-1.png'/>

<img src='./images/LASSO-KNN-ROC-2.png'/>

<img src='./images/LASSO-KNN-ROC-3.png'/>

<img src='./images/LASSO-SVM-ROC-1.png'/>

<img src='./images/LASSO-SVM-ROC-2.png'/>

<img src='./images/LASSO-SVM-ROC-3.png'/>

<img src='./images/LASSO-RF-ROC-1.png'/>

<img src='./images/LASSO-RF-ROC-2.png'/>

<img src='./images/LASSO-RF-ROC-3.png'/>


## Recursive Feature Elimination (RFE)

### t-SNE 
<img src='./images/RFE TSNE.png'>

### Performance Metrics 

<img src='./images/RFE-KNN-Performance-Metrics.png' />

<img src="./images/RFE-SVM-Performance-Matrics.png" />

<img src="./images/RFE-RF-Performance-Metrics.png" />

### Confusion Matrix 

<img src="./images/RFE-KNN-Confusion-Matrix.png"/>

<img src="./images/RFE-SVM-Confusion-Matrix.png" />

<img src='./images/RFE-RF-Confusion-Matrix.png' />

## SelectFromModel 

# Conclusion

# References	
- http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html 
- https://www.analyticsvidhya.com/blog/2020/05/7-scikit-learn-hacks-tips-tricks/ 
- https://analyticsindiamag.com/guide-to-dimensionality-reduction-with-recursive-feature-elimination/ 
- https://www.titanwolf.org/Network/q/8f9189c7-fb76-4628-9707-822e8074ebfc/x 
- https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
- https://learn.g2.com/k-nearest-neighbor
- https://www.kdnuggets.com/2020/04/introduction-k-nearest-neighbour-algorithm-using-examples.html
		


