# ISLP Conceptual Exercise Questions

## Question 1
Describe the null hypotheses to which the p-values given in Table 3.4 correspond. Explain what conclusions you can draw based on these p-values. Your explanation should be phrased in terms of sales, TV, radio, and newspaper, rather than in terms of the coefficients of the linear model.

Table 3.4:
| Predictor  | Coefficient | Standard Error | t-statistic | p-value |
|------------|-------------|----------------|-------------|---------|
| Intercept  | 2.939       | 0.3119         | 9.42        | <0.0001 |
| TV         | 0.046       | 0.0014         | 32.81       | <0.0001 |
| Radio      | 0.189       | 0.0086         | 21.89       | <0.0001 |
| Newspaper  | -0.001      | 0.0059         | -0.18       | 0.8599  |

## Answer 1
Based on the p-values, TV and Radio are significant with their p-values smaller than 0.0001 meaning that the null hypothethsis can be rejected. In other words, there's less than 0.0001 chance that TV/Radio is not effective in sales. While, Newspaper is not significant with p-values being vary large (0.8599) meaning that the null hypothethsis can not be rejected. In order words, there's 0.8599 chance that newspaper is not effective in sales. Thus, TV and radio has a significant impact on the sales whereas Newspaper doesn't. 

## Question 2
Carefully explain the differences between the KNN classifier and KNN regression methods.

## Answer 2

KNN Classifier:

In KNN classification, we aim to assign a class label to a new data point based on the class labels of its nearest neighbors. The class labels are categorical or discrete variables.
The algorithm determines the class label of the new data point by taking a majority vote among its K nearest neighbors. For example, if K=5 and three of the nearest neighbors belong to class A and two belong to class B, the new data point would be assigned to class A.
The distance metric used to measure the proximity between data points is typically Euclidean distance, but other distance metrics can also be applied.
KNN classifier is a non-parametric algorithm, meaning it does not make any assumptions about the underlying data distribution.
The algorithm can handle multi-class classification problems and can adapt to complex decision boundaries.

KNN Regression:

In KNN regression, we aim to predict a continuous or numerical value for a new data point based on the values of its nearest neighbors. The target variable is continuous.
Instead of taking a majority vote, KNN regression predicts the value of the new data point by averaging or taking the median of the target values of its K nearest neighbors.
Similar to KNN classification, the distance metric used is typically Euclidean distance, although other distance metrics can also be utilized.
KNN regression, like KNN classification, is non-parametric, making no assumptions about the underlying data distribution.
KNN regression is capable of capturing non-linear relationships between the features and the target variable, which can be advantageous for certain regression problems.
It's important to note that the choice of the value for K in KNN significantly influences the performance of both the classifier and the regression model. A small value of K may result in a more flexible model, prone to overfitting, while a large value of K may lead to a smoother but potentially less accurate model.

In summary, the main difference between KNN classifier and KNN regression lies in their output. The classifier predicts discrete class labels based on the majority vote of the K nearest neighbors, while the regression model predicts continuous values by averaging or taking the median of the target values of the K nearest neighbors.

## Question 3
Suppose we have a data set with five predictors: $X_1$ = GPA, $X_2$ = IQ, $X_3$ = Level (1 for College and 0 for High School), $X_4$ = Interaction between GPA and IQ, and $X_5$ = Interaction between GPA and Level. The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model and get $\hat{β}_0 = 50$, $\hat{β}_1 = 20$, $\hat{β}_2 = 0.07$, $\hat{β}_3 = 35$, $\hat{β}_4 = 0.01$, $\hat{β}_5 = -10$.

(a) Which answer is correct, and why?
    i. For a fixed value of IQ and GPA, high school graduates earn more, on average, than college graduates.
    ii. For a fixed value of IQ and GPA, college graduates earn more, on average, than high school graduates.
    iii. For a fixed value of IQ and GPA, high school graduates earn more, on average, than college graduates provided that the GPA is high enough.
    iv. For a fixed value of IQ and GPA, college graduates earn more, on average, than high school graduates provided that the GPA is high enough.

(b) Predict the salary of a college graduate with IQ of 110 and a GPA of 4.0.

(c) True or false: Since the coefficient for the GPA/IQ interaction term is very small, there is very little evidence of an interaction effect. Justify your answer.

## Answer 3

(a) iii 
The opinion i and ii are wrong because $X_5$ is an interaction term between GPA and Level. When level term changes, the interaction term will change hence the salary will change. Thus, we should consider the changing of the $X_3$ and the interaction term $X_5$. 
The regression model can be represented with two functions, one for colleage and the other one for high school level. For high school, since $X_3$ is 0 and IQ and GPA are fixed, we only need to consider the intercept term. For college, we need to consider intercept, $X_3$ and $X_5$ that are equal to 1. Thus, if we want salary of high school level greater than college level, we need to have the equation 50 > 50 + 35 - 10*(GPA) resulting GPA > 3.5.

(b) 50+10*4 + 0.07*110 + 35 + 0.01*4*110 - 10*4

(c) False. The coefficient of GPA/IQ interaction term represents one unit change of GPA/IQ interaction terms to 0.01 change in thousands of dollars salary change. It doesn't represent the effectiveness of the interaction. The evidence of effectivenss should be represented by p-value. 


## Question 4
I collect a set of data (n = 100 observations) containing a single predictor and a quantitative response. I then fit a linear regression model to the data, as well as a separate cubic regression, i.e., $Y = β_0 + β_1X + β_2X^2 + β_3X^3 + $".

(a) Suppose that the true relationship between X and Y is linear, i.e., $Y = β_0 + β_1X + $". Consider the training residual sum of squares (RSS) for the linear regression and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

(b) Answer (a) using test rather than training RSS.

(c) Suppose that the true relationship between X and Y is not linear, but we don't know how far it is from linear. Consider the training RSS for the linear regression and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

(d) Answer (c) using test rather than training RSS.


## Answer 4

(a) Given the true relationship is linear, we would expect the training RSS for the linear regression model to be lower than the training RSS for the cubic regression model because the linear regression model is the correct model that captures the true linear relationship between X and Y. Whereas the cubic regression model includes additional higher-order terms ($X^2$ and $X^3$) that are unnecessary and may introduce more noise into the model. Thus, the cubic regression model is likely to have a higher training RSS compared to the linear regression model.

(b) Given the true relationship is linear, we would expect the testing RSS for the linear regression to be be lower than cubic regression model because the linear regression model represents the true relationship between X and Y, while the cubic regression model introduces additional complexity. With the complexicity increases, the model will tend to fit the tranining data too well but not generalized enough for unseen data (testing data) thus leading a higher testing RSS. 

(c) Given the true relationship is not linear, it is hard to predict whether the training RSS for the linear regression model will be lower than the training RSS for the cubic regression model. The linear regression model assumes a linear relationship and may not capture the true underlying non-linear pattern, resulting in a high training RSS. On the other hand, the cubic regression model is more flexible and has the potential to capture non-linear patterns, which could lead to a lower training RSS if the true relationship is indeed non-linear. Therefore, without knowing the true extent of non-linearity, it is challenging to determine which model will have a lower training RSS.


(d) Similarly to the training RSS, the test RSS is hard to predict whether the linear regression model or the cubic regression model will have a lower test RSS when the true relationship between X and Y is not linear. The linear regression model may have a higher test RSS due to its inability to capture the non-linear nature of the relationship, while the cubic regression model has the potential to better approximate the true relationship if it is indeed non-linear. Therefore, without knowledge of the true extent of non-linearity, it is hard to determine which model will have a lower test RSS.

## Question 5
Consider the fitted values that result from performing linear regression without an intercept. In this setting, the ith fitted value takes the form:
$$\hat{y}_i = x_i\hat{β}$$
where

$$\hat{β} = (\sum_{i=1}^{n} x_i y_i)/(\sum_{i'=1}^{n} x_{i'}^{2})$$

Show that we can write:
$$\hat{y_i} = \sum_{i'=1}^{n} a_{i'} y_{i'}$$
What is $a_{i'}$?
Note: We interpret this result by saying that the fitted values from linear regression are linear combinations of the response values.

## Answer 5

In linear regression without an intercept, the fitted value for the ith observation, denoted as $\hat{y}_i$, is given by:
$$\hat{y}_i = x_i\hat{\beta}$$

To find $\hat{\beta}$, let's substitute the formula for $\hat{\beta}$ into the expression for $\hat{y}_i$:  
$$\hat{y}_i = x_i \left(\frac{\sum_{i'=1}^{n} x_{i'} y_{i'}}{\sum_{i''=1}^{n} x_{i''}^{2}}\right)$$

Now, let's simplify this expression:  
$$\hat{y}_i = \frac{x_i}{\sum_{i''=1}^{n} x_{i''}^{2}} \sum_{i'=1}^{n} x_{i'} y_{i'}$$

We can observe that the term $\frac{x_i}{\sum_{i''=1}^{n} x_{i''}^{2}}$ is a constant for each observation, which we can denote as $a_{i'}$. Therefore, we can rewrite the expression for $\hat{y}i$ as:  
$$\hat{y}_i = \sum_{i'=1}^{n} a_{i'} y_{i'}$$

In this formulation, $a_{i'} = \frac{x_i}{\sum_{i'=1}^{n} x_{i'}^{2}}$ represents the weight or coefficient assigned to each response value $y_{i'}$ in the linear combination.

To interpret this result, we can say that the fitted values obtained from linear regression without an intercept are linear combinations of the response values. Each fitted value $\hat{y}_i$ is calculated by taking a weighted sum of the response values $y_{i'}$, where the weights $a_{i'}$ depend on the predictor variable values $x_i$ and the sum of squares of the predictor variable $\sum_{i'=1}^{n} x_{i'}^{2}$.

This interpretation elucidates the relationship between the predictor variable and the response variable, indicating that the fitted values are determined by combining the response values in a linear manner, with the weights determined by the predictor variable values.

## Question 6
Using $\hat{β_1} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_{i}- \bar{x})^{2}}$, $\hat{β_0} = \bar{y} - \hat{β_1} \bar{x}$, argue that in the case of simple linear regression, the least squares line always passes through the point ($\bar{x}$, $\bar{y}$).


## Answer 6
In simple linear regression, we have the equation for the least squares line as:

$$\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$$

where $\hat{\beta_0}$ is the intercept and $\hat{\beta_1}$ is the slope of the line.

To find the estimated coefficients, $\hat{\beta_0}$ and $\hat{\beta_1}$, we use the following formulas:

$$\hat{\beta_1} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_{i}- \bar{x})^{2}}$$
$$\hat{\beta_0} = \bar{y} - \hat{\beta_1} \bar{x}$$

Now, let's substitute the formula for $\hat{\beta_0}$ into the equation for the least squares line:

$$\hat{y} = (\bar{y} - \hat{\beta_1} \bar{x}) + \hat{\beta_1}x$$

Simplifying this expression, we get:

$$\hat{y} = \bar{y} + \hat{\beta_1}(x - \bar{x})$$

Notice that the expression $(x - \bar{x})$ is the deviation of the predictor variable from its mean. Similarly, $\hat{\beta_1}$ is the coefficient that represents the change in the response variable for a one-unit change in the predictor variable. Since $\bar{x}$ represents the average or mean of all the predictor variable values, the term $(x - \bar{x})$ represents the deviation of each individual predictor variable value from the mean.

Therefore, when we substitute the values into the equation, we find that the least squares line can be written as:

$$\hat{y} = \bar{y} + \hat{\beta_1}(x - \bar{x})$$

Now, let's consider the point ($\bar{x}$, $\bar{y}$). Plugging in $\bar{x}$ into the equation, we have:

$$\hat{y} = \bar{y} + \hat{\beta_1}(\bar{x} - \bar{x})$$

We can see that the term $(\bar{x} - \bar{x})$ is zero. Therefore, the equation simplifies to:

$$\hat{y} = \bar{y}$$

This demonstrates that when we substitute $\bar{x}$ into the equation for the least squares line, the resulting predicted value $\hat{y}$ is equal to $\bar{y}$. Hence, the least squares line always passes through the point ($\bar{x}$, $\bar{y}$).

This result holds because the estimated intercept, $\hat{\beta_0}$, is chosen to ensure that the line passes through the point represented by the mean values of the predictor variable ($\bar{x}$) and the response variable ($\bar{y}$).


## Question 7
It is claimed in the text that in the case of simple linear regression of Y onto X, the $R^2$ statistic (3.17) is equal to the square of the correlation between X and Y (3.18). Prove that this is the case. For simplicity, you may assume that $\bar{x}$ = $\bar{y}$ = 0.


## Answer 7
In simple linear regression, the coefficient of determination $R^2$ measures the proportion of the total variation in the response variable Y that can be explained by the linear relationship with the predictor variable X.

The formula for $R^2$ is given by equation (3.17):

$$R^2 = \frac{\text{SSR}}{\text{SST}}$$

where SSR represents the sum of squares of the regression (explained variation) and SST represents the total sum of squares (total variation).

Now, let's consider the correlation between X and Y, denoted as $\rho_{XY}$. In the case where $\bar{x} = \bar{y} = 0$, the formula for the correlation coefficient is given by equation (3.18):

$$\rho_{XY} = \frac{\sum_{i=1}^{n}x_iy_i}{\sqrt{\sum_{i=1}^{n}x_i^2 \sum_{i=1}^{n}y_i^2}}$$

To prove that $R^2$ is equal to the square of $\rho_{XY}$, we need to show that SSR is equal to $\rho_{XY}^2$ times SST.

In simple linear regression, SSR measures the variation in the response variable Y that is explained by the linear relationship with X. It is given by:

$$\text{SSR} = \sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2$$

where $\hat{y}_i$ represents the predicted value of Y for the ith observation.

Using the assumption that $\bar{x} = \bar{y} = 0$, the least squares line simplifies to:

$$\hat{y}_i = \hat{\beta}_1x_i$$

Substituting this expression into the formula for SSR, we have:

$$\text{SSR} = \sum_{i=1}^{n}(\hat{\beta}_1x_i - \bar{y})^2$$

Since $\bar{y} = 0$, we can simplify further:

$$\text{SSR} = \sum_{i=1}^{n}(\hat{\beta}_1x_i)^2$$

Expanding the squared term, we get:

$$\text{SSR} = \sum_{i=1}^{n}\hat{\beta}_1^2x_i^2$$

Now, let's consider SST, the total sum of squares. It measures the total variation in the response variable Y. It is given by:

$$\text{SST} = \sum_{i=1}^{n}(y_i - \bar{y})^2$$

Again, using the assumption that $\bar{x} = \bar{y} = 0$, we have:

$$\text{SST} = \sum_{i=1}^{n}y_i^2$$

Now, let's express $\sum_{i=1}^{n}x_i^2$ in terms of $\sum_{i=1}^{n}y_i^2$ using the correlation coefficient $\rho_{XY}$:

$$\sum_{i=1}^{n}x_i^2 = \rho_{XY}^2\sum_{i=1}^{n}y_i^2$$

Substituting this into the formula for SSR, we have:

$$\text{SSR} = \rho_{XY}^2\sum_{i=1}^{n}y_i^2$$

Finally, dividing SSR by SST, we get:

$$\frac{\text{SSR}}{\text{SST}} = \frac{\rho_{XY}^2\sum_{i=1}^{n}y_i^2}{\sum_{i=1}^{n}y_i^2}$$

Canceling out the common terms, we are left with:

$$\frac{\text{SSR}}{\text{SST}} = \rho_{XY}^2$$

This proves that in the case of simple linear regression, when assuming $\bar{x} = \bar{y} = 0$, the coefficient of determination $R^2$ is equal to the square of the correlation coefficient $\rho_{XY}$, as stated in equations (3.17) and (3.18).

It's important to note that this proof assumes the assumption of $\bar{x} = \bar{y} = 0$. In practice, the proof holds even if the means are non-zero, but theassumption simplifies the calculations. Additionally, this proof is specific to simple linear regression and may not hold for more complex regression models.

I hope this explanation clarifies the relationship between the coefficient of determination $R^2$ and the correlation coefficient $\rho_{XY}$ in simple linear regression. 