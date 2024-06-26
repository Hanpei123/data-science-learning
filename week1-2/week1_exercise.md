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

## Question 4
I collect a set of data (n = 100 observations) containing a single predictor and a quantitative response. I then fit a linear regression model to the data, as well as a separate cubic regression, i.e., $Y = β_0 + β_1X + β_2X^2 + β_3X^3 + $".

(a) Suppose that the true relationship between X and Y is linear, i.e., $Y = β_0 + β_1X + $". Consider the training residual sum of squares (RSS) for the linear regression and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

(b) Answer (a) using test rather than training RSS.

(c) Suppose that the true relationship between X and Y is not linear, but we don't know how far it is from linear. Consider the training RSS for the linear regression and also the training RSS for the cubic regression. Would we expect one to be lower than the other, would we expect them to be the same, or is there not enough information to tell? Justify your answer.

(d) Answer (c) using test rather than training RSS.

## Question 5
Consider the fitted values that result from performing linear regression without an intercept. In this setting, the ith fitted value takes the form:
$$\hat{y}_i = x_i \hat{β}$$
where

$$\hat{β} = (\sum_{i=1}^{n} x_i y_i)/(\sum_{i'=1}^{n} x_{i'}^{2})$$

Show that we can write:
$$\hat{y_i} = \sum_{i'=1}^{n} a_{i'} y_{i'}$$
What is $a_{i'}$?
Note: We interpret this result by saying that the fitted values from linear regression are linear combinations of the response values.

## Question 6
Using (3.4), argue that in the case of simple linear regression, the least squares line always passes through the point ($\bar{x}$, $\bar{y}$).

## Question 7
It is claimed in the text that in the case of simple linear regression of Y onto X, the $R^2$ statistic (3.17) is equal to the square of the correlation between X and Y (3.18). Prove that this is the case. For simplicity, you may assume that $\bar{x}$ = $\bar{y}$ = 0.


# Exercise for Linear Regression

## Question 1
Describe the impact of violating the normality assumption in linear regression on hypothesis testing and confidence intervals.


## Question 2
How does multicollinearity affect the interpretation of coefficient estimates in linear regression? Provide an example.


## Question 3
In linear regression, the coefficient can be interpreted as the average change in the dependent variable for each unit change in the independent variable, making it meaningful to consider it as an average measure. Explain this in details.


## Question 4
Explain the inflation of the Variable inflation factor? What term inflated and how?