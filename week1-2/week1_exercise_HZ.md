# Conceptual Exercise Questions

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


## Question 7
It is claimed in the text that in the case of simple linear regression of Y onto X, the $R^2$ statistic (3.17) is equal to the square of the correlation between X and Y (3.18). Prove that this is the case. For simplicity, you may assume that $\bar{x}$ = $\bar{y}$ = 0.