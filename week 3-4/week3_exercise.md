# ISLP Conceptual Exercise Questions

## Question 1
 Proving the Equivalence of the Logistic Function and Logit Representation in Logistic Regression


## Question 2

The Bayes classifier is a fundamental concept in statistical classification that assigns an observation to the class with the highest posterior probability, given the observation. When the observations in the \(k\)th class are assumed to be drawn from a normal distribution \(N(\mu_k, \sigma^2)\), with a common variance \(\sigma^2\) across classes but different means \(\mu_k\), prove that the Bayes classifier can be shown to assign an observation to the class for which the discriminant function is maximized.



## Question 3

This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a class-specific mean vector and a class specific covariance matrix. We consider the simple case where \(p = 1\); i.e. there is only one feature. Suppose that we have \(K\) classes, and that if an observation belongs to the \(kth\) class then \(X\) comes from a one-dimensional normal distribution, \(X \sim N(\mu_k, \sigma^2_k)\). Recall that the density function for the one-dimensional normal distribution is given in (4.16). Prove that in this case, the Bayes classifier is not linear. Argue that it is in fact quadratic.
   - *Hint:* For this problem, you should follow the arguments laid out in Section 4.4.1, but without making the assumption that \(\sigma^2_1 = \dots = \sigma^2_K\).



## Question 4
When the number of features \(p\) is large, there tends to be a deterioration in the performance of KNN and other local approaches that perform prediction using only observations that are near the test observation for which a prediction must be made. This phenomenon is known as the curse of dimensionality, and it ties into the fact that non-parametric approaches often perform poorly when \(p\) is large. We will now investigate this curse.

### (a)

Suppose that we have a set of observations, each with measurements on \(p = 1\) feature, \(X\). We assume that \(X\) is uniformly (evenly) distributed on \([0, 1]\). Associated with each observation is a response value. Suppose that we wish to predict a test observation’s response using only observations that are within 10% of the range of \(X\) closest to that test observation. For instance, in order to predict the response for a test observation with \(X = 0.6\), we will use observations in the range \([0.55, 0.65]\). On average, what fraction of the available observations will we use to make the prediction?

### (b)

Now suppose that we have a set of observations, each with measurements on \(p = 2\) features, \(X_1\) and \(X_2\). We assume that \((X_1, X_2)\) are uniformly distributed on \([0, 1] \times [0, 1]\). We wish to predict a test observation’s response using only observations that are within 10% of the range of \(X_1\) and within 10% of the range of \(X_2\) closest to that test observation. For instance, in order to predict the response for a test observation with \(X_1 = 0.6\) and \(X_2 = 0.35\), we will use observations in the range \([0.55, 0.65]\) for \(X_1\) and in the range \([0.3, 0.4]\) for \(X_2\). On average, what fraction of the available observations will we use to make the prediction?

### (c)

Suppose we have a set of observations on \(p = 100\) features, with each feature uniformly distributed from 0 to 1. To predict a test observation's response, we use observations within the 10% of each feature's range closest to the test observation. 


### (d)

Using the answers from parts (a)–(c), we can argue that a significant drawback of KNN (K-Nearest Neighbors) when \(p\) is large is the sparsity of the data in high-dimensional space. As \(p\) increases, the fraction of observations considered "near" any given test observation decreases exponentially, making it challenging to find a sufficient number of close neighbors for reliable prediction.

### (e)

Suppose we wish to make a prediction for a test observation by creating a \(p\)-dimensional hypercube centered around the test observation that contains, on average, 10% of the training observations. 


## Questions 5

### (a) 
If the Bayes decision boundary is linear, do we expect LDA or QDA to perform better on the training set? On the test set?

### (b) 
If the Bayes decision boundary is non-linear, do we expect LDA or QDA to perform better on the training set? On the test set?

### (c)
In general, as the sample size \(n\) increases, do we expect the test prediction accuracy of QDA relative to LDA to improve, decline, or be unchanged? Why?

### (d)
True or False: Even if the Bayes decision boundary for a given problem is linear, we will probably achieve a superior test error rate using QDA rather than LDA because QDA is flexible enough to model a linear decision boundary. Justify your answer.


## Questions 6

Given the scenario where a dataset is split into equally-sized training and test sets, and two classification methods are applied with the following outcomes:

1. **Logistic Regression** yields:
   - A 20% error rate on the training data.
   - A 30% error rate on the test data.
2. **1-Nearest Neighbors (K=1)** results in:
   - An average error rate of 18% across both the training and test datasets.

Based on these results, which method should we prefer to use for classifcation of new observations? Why?

## Questions 7

Suppose you wish to classify an observation \(X \in \mathbb{R}\) into apples and oranges. You fit a logistic regression model and find that:

\[ \Pr(Y = \text{orange} | X = x) = \frac{\exp(\hat{\beta}_0 + \hat{\beta}_1x)}{1 + \exp(\hat{\beta}_0 + \hat{\beta}_1x)} \]

Your friend fits a logistic regression model to the same data using the softmax formulation, and finds that:

\[ \Pr(Y = \text{orange} | X = x) = \frac{\exp(\hat{\alpha}_{\text{orange}0} + \hat{\alpha}_{\text{orange}1}x)}{\exp(\hat{\alpha}_{\text{orange}0} + \hat{\alpha}_{\text{orange}1}x) + \exp(\hat{\alpha}_{\text{apple}0} + \hat{\alpha}_{\text{apple}1}x)} \]


### (a) 
What is the log odds of orange versus apple in your model?

### (b) 
What is the log odds of orange versus apple in your friend’s model?

### (c) 
Suppose that in your model, \(\hat{\beta}_0 = 2\) and \(\hat{\beta}_1 = -1\). What are the coefficient estimates in your friend’s model? Be as specific as possible.

### (d) 
Now suppose that you and your friend fit the same two models on a different data set. This time, your friend gets the coefficient estimates \(\hat{\alpha}_{\text{orange}0} = 1.2\), \(\hat{\alpha}_{\text{orange}1} = -2\), \(\hat{\alpha}_{\text{apple}0} = 3\), \(\hat{\alpha}_{\text{apple}1} = 0.6\). What are the coefficient estimates in your model?

### (e) 
Finally, suppose you apply both models from (d) to a data set with 2,000 test observations. What fraction of the time do you expect the predicted class labels from your model to agree with those from your friend’s model? Explain your answer.


## Question 8

When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable. In contrast, Linear Discriminant Analysis (LDA) does not suffer from this problem. Why?



## Question 9

When the sample size \(n\) is small and the distribution of the predictor \(X\) is approximately normal in each of the classes, Linear Discriminant Analysis (LDA) tends to be more stable than logistic regression. Why?



## Low-Dimensional Views with LDA for Multiclass Responses

Linear Discriminant Analysis (LDA) is a technique that is particularly useful for providing low-dimensional views of data, especially in scenarios involving more than two response classes. This capability is a direct result of LDA's mathematical formulation and its approach to dimensionality reduction and class separation.

#### Foundation of LDA

LDA seeks to find linear combinations of the features that best separate the classes. For a dataset with \(p\) features and \(K\) classes, LDA aims to project the data onto a lower-dimensional space (with up to \(K-1\) dimensions) that maximizes the separation between these classes. This is achieved by maximizing the ratio of the between-class variance to the within-class variance in this lower-dimensional space.

#### Mechanism for Low-Dimensional Views

##### Dimensionality Reduction

- **Projection**: LDA projects the high-dimensional data onto a new space defined by the linear discriminants, which are directions in the feature space that provide the best class separation.
- **Reduced Dimensions**: The maximum number of linear discriminants (or axes) is \(K-1\), where \(K\) is the number of classes. This means that even for data with a large number of features, LDA can reduce the dimensionality to at most \(K-1\) dimensions, facilitating visualization and analysis.

##### Visualization and Interpretation

- **Visualization**: By projecting the data onto the first two or three linear discriminants, LDA allows for easy visualization of the data in two or three dimensions. This can be particularly helpful in understanding how well-separated the classes are in the reduced-dimensional space.
- **Interpretation**: The directions of the linear discriminants can also provide insights into which features are most important for distinguishing between classes. This interpretability is a key advantage of LDA, as it not only reduces dimensionality but also highlights features relevant to class separation.

#### Advantages in Multiclass Scenarios

- **Direct Multiclass Handling**: Unlike some methods that require adaptations to handle multiclass scenarios, LDA naturally extends to more than two classes, directly incorporating multiclass separation into its formulation.
- **Class-Specific Projections**: The dimensionality reduction in LDA is guided by the goal of class separation, which means that the reduced-dimensional views are specifically optimized to highlight differences between the classes, unlike other methods like PCA which do not consider class labels.

#### Conclusion

LDA's ability to provide low-dimensional views of data when handling more than two response classes stems from its focus on finding the most discriminative directions for class separation. This not only aids in visualization and interpretation by reducing the complexity of the data but also ensures that the reduced dimensions are meaningful in terms of class differentiation. The result is a powerful tool for both exploratory data analysis and model building in multiclass classification problems.
