# Resampling Methods

## Conceptual

### Question 1

> Prove that $\alpha$ given by $$\alpha = \frac{\sigma^2_Y - \sigma_{XY}}{\sigma^2_X + \sigma^2_Y - 2\sigma_{XY}}$$ does indeed minimize $Var(\alpha X + (1 - \alpha)Y)$.


Remember that: 

$$
Var(aX) = a^2Var(X), \\ 
\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X,Y), \\
\mathrm{Cov}(aX, bY) = ab\mathrm{Cov}(X, Y)
$$

If we define $\sigma^2_X = \mathrm{Var}(X)$, $\sigma^2_Y = \mathrm{Var}(Y)$ and
$\sigma_{XY} = \mathrm{Cov}(X, Y)$

\begin{align}
Var(\alpha X + (1 - \alpha)Y) 
  &= \alpha^2\sigma^2_X + (1-\alpha)^2\sigma^2_Y + 2\alpha(1 - \alpha)\sigma_{XY} \\
  &= \alpha^2\sigma^2_X + \sigma^2_Y - 2\alpha\sigma^2_Y + \alpha^2\sigma^2_Y + 
     2\alpha\sigma_{XY} - 2\alpha^2\sigma_{XY} 
\end{align}

Now we want to find when the rate of change of this function is 0 with respect
to $\alpha$, so we compute the partial derivative, set to 0 and solve.

$$
\frac{\partial}{\partial{\alpha}} = 
  2\alpha\sigma^2_X - 2\sigma^2_Y + 2\alpha\sigma^2_Y + 2\sigma_{XY} - 4\alpha\sigma_{XY} = 0
$$

Moving $\alpha$ terms to the same side:

$$
\alpha\sigma^2_X + \alpha\sigma^2_Y - 2\alpha\sigma_{XY} = \sigma^2_Y - \sigma_{XY}
$$

$$
\alpha = \frac{\sigma^2_Y - \sigma_{XY}}{\sigma^2_X + \sigma^2_Y - 2\sigma_{XY}}
$$

We should also show that this is a minimum, so that the second partial 
derivative wrt $\alpha$ is $>= 0$.

\begin{align}
\frac{\partial^2}{\partial{\alpha^2}} 
  &=  2\sigma^2_X + 2\sigma^2_Y - 4\sigma_{XY} \\
  &=  2(\sigma^2_X + \sigma^2_Y - 2\sigma_{XY}) \\
  &= 2\mathrm{Var}(X - Y)
\end{align}

Since variance is positive, then this must be positive.

### Question 2

> We will now derive the probability that a given observation is part of a bootstrap sample. Suppose that we obtain a bootstrap sample from a set of n observations.
>
> a. What is the probability that the first bootstrap observation is _not_ the
>    $j$th observation from the original sample? Justify your answer.

The probability is $1 - 1/n$, since each observation has an equal chance of being selected.

> b. What is the probability that the second bootstrap observation is _not_ the $j$th observation from the original sample?

Since each bootstrap observation is a random sample, this probability is the same ($1 - 1/n$).

> c. Argue that the probability that the $j$th observation is _not_ in the bootstrap sample is $(1 - 1/n)^n$.

For the $j$th observation to not be in the sample, it would have to _not_ be picked for each of $n$ positions, so not picked for $1, 2, ..., n$, thus the probability is $(1 - 1/n)^n$ 

> d. When $n = 5$, what is the probability that the $j$th observation is in the bootstrap sample?

```{r}
n <- 5
1 - (1 - 1 / n)^n
```

$p = 0.67$

> e. When $n = 100$, what is the probability that the $j$th observation is in the bootstrap sample?

```{r}
n <- 100
1 - (1 - 1 / n)^n
```

$p = 0.64$

> f. When $n = 10,000$, what is the probability that the $j$th observation is in the bootstrap sample?

```{r}
n <- 100000
1 - (1 - 1 / n)^n
```

$p = 0.63$

> g. Create a plot that displays, for each integer value of $n$ from 1 to 100,000, the probability that the $j$th observation is in the bootstrap sample. Comment on what you observe.

```{r}
x <- sapply(1:100000, function(n) 1 - (1 - 1 / n)^n)
plot(x, log = "x", type = "o")
```

The probability rapidly approaches 0.63 with increasing $n$.

Note that $$e^x = \lim_{x \to \inf} \left(1 + \frac{x}{n}\right)^n,$$ so with $x = -1$, we
can see that our limit is $1 - e^{-1} = 1 - 1/e$.

> h. We will now investigate numerically the probability that a bootstrap sample of size $n = 100$ contains the $j$th observation. Here $j = 4$. We repeatedly create bootstrap samples, and each time we record whether or not the fourth observation is contained in the bootstrap sample.
>    
>    ```r
>    > store <- rep (NA, 10000)
>    > for (i in 1:10000) {
>        store[i] <- sum(sample(1:100, rep = TRUE) == 4) > 0
>    }
>    > mean(store)
>    ```
>    
>    Comment on the results obtained.

```{r}
store <- replicate(10000, sum(sample(1:100, replace = TRUE) == 4) > 0)
mean(store)
```

The probability of including $4$ when resampling numbers $1...100$ is close to
$1 - (1 - 1/100)^{100}$.

### Question 3

> 3. We now review $k$-fold cross-validation.
>
> a. Explain how $k$-fold cross-validation is implemented.

Divide the dataset into $k$ segments, or folds. Train the model on $k-1$ folds and test it on the remaining fold. Repeat this $k$ times, each time with a different fold as the test set, and average the test scores.


> b. What are the advantages and disadvantages of $k$-fold cross-validation
>    relative to:
>    i. The validation set approach?
>    ii. LOOCV?

- **Compared to the Validation Set Approach:**
  - Advantages: More reliable estimate of model performance.
  - Disadvantages: More computationally intensive.
- **Compared to LOOCV:**
  - Advantages: Less computationally intensive for large datasets.
  - Disadvantages: Might introduce bias if $k$ is too small.

### Question 4

> Suppose that we use some statistical learning method to make a prediction for the response $Y$ for a particular value of the predictor $X$. Carefully describe how we might estimate the standard deviation of our prediction.


Use the bootstrap method to estimate the standard deviation of a prediction for a particular value of $X$. This involves sampling from the dataset with replacement, fitting the model, and predicting for $X$ multiple times. The standard deviation of these predictions estimates the standard deviation of the original prediction.



# Linear Model Selection and Regularization

## Conceptual

### Question 1

> We perform best subset, forward stepwise, and backward stepwise selection on a single data set. For each approach, we obtain $p + 1$ models, containing $0, 1, 2, ..., p$ predictors. Explain your answers:
>
> a. Which of the three models with $k$ predictors has the smallest *training* RSS?

The model with $k$ predictors that has the smallest *training* Residual Sum of Squares (RSS) is typically obtained through **best subset selection**. Here's why:

- **Best Subset Selection** systematically searches for the best model by considering all possible subsets of the predictors. This means it evaluates all combinations of predictors of size $k$ and selects the model that has the lowest RSS on the training data. Because it exhaustively searches all possible models, it is guaranteed to find the model with the lowest RSS for a given number of predictors $k$.

- **Forward Stepwise Selection** starts with no predictors and sequentially adds the predictor that provides the most significant improvement to the model. While this approach is computationally more efficient than best subset selection, it does not evaluate all possible models of size $k$. Therefore, it might miss the model that minimally reduces the RSS for a given $k$.

- **Backward Stepwise Selection** starts with all predictors and sequentially removes the least significant predictor at each step. Similar to forward stepwise selection, it does not consider all possible subsets of size $k$, potentially overlooking the optimal model with the lowest RSS for a given number of predictors $k$.

In summary, **best subset selection** is the method that, for any given $k$, is most likely to yield the model with the smallest training RSS because it evaluates all possible combinations of $k$ predictors.


> b. Which of the three models with $k$ predictors has the smallest *test* RSS?

The model with $k$ predictors that has the smallest *test* Residual Sum of Squares (RSS) cannot be determined a priori to consistently be best subset, forward stepwise, or backward stepwise selection. The reason is that the performance of these models on test data depends on various factors, including the dataset's characteristics and the specific predictors involved. Here's a brief overview of how each method might perform in terms of test RSS:

- **Best Subset Selection** might overfit the training data, especially for larger $k$, because it selects the model that fits the training data best among all possible subsets of predictors. While this approach is exhaustive and can potentially find the model with the lowest training RSS, it does not guarantee that this model will also have the lowest test RSS due to overfitting.

- **Forward Stepwise Selection** and **Backward Stepwise Selection** are more constrained in their search for models, which might help in preventing overfitting to some extent. By adding or removing one predictor at a time based on its contribution to the model's performance, these methods might inadvertently select a model that generalizes better to unseen data, potentially leading to a lower test RSS compared to the best subset selection in some cases.

- The effectiveness of **Regularization** (not mentioned in the original question but relevant for comparison) techniques like Lasso or Ridge Regression in reducing test RSS highlights the importance of balancing model complexity and training data fit. These methods penalize the inclusion of less significant predictors and can often result in models that generalize better than those selected purely based on training RSS.

In practice, the model with the smallest test RSS is determined empirically through cross-validation or a similar approach that evaluates model performance on unseen data. It's also important to note that minimizing test RSS is not the only criterion for model selection; interpretability and the underlying assumptions about the data and model should also be considered.



> c. True or False:
>    i. The predictors in the $k$-variable model identified by forward stepwise are a subset of the predictors in the ($k+1$)-variable model identified by forward stepwise selection.

Yes, the statement is true. In forward stepwise selection, the method starts with no predictors and sequentially adds one predictor at a time that most improves the model according to a specified criterion (usually a reduction in RSS or an improvement in R-squared).

Therefore, by design:

- The model with $k$ predictors, identified through forward stepwise selection, will contain a specific set of $k$ predictors.
- When moving to a $k+1$ model, forward stepwise selection adds one more predictor to the existing $k$ predictors, making the $k+1$ model include all the predictors from the $k$-predictor model plus one additional predictor.

This process ensures that the predictors in the $k$-variable model are always a subset of the predictors in the $(k+1)$-variable model identified by forward stepwise selection.


>    ii. The predictors in the $k$-variable model identified by backward stepwise are a subset of the predictors in the $(k+1)$-variable model identified by backward stepwise selection.

Actually, the statement should be reversed for clarity and accuracy. In backward stepwise selection, the method starts with all predictors and sequentially removes the least significant predictor at each step. Therefore:

- The model with $(k+1)$ predictors, identified through backward stepwise selection, will contain a specific set of $(k+1)$ predictors.
- When moving to a $k$ model by removing one predictor, the $k$-variable model will include all the predictors from the $(k+1)$-predictor model except the one that was removed.

This means that the predictors in the $k$-variable model are indeed a subset of the predictors in the $(k+1)$-variable model identified by backward stepwise selection, but the direction of selection (from all predictors down to fewer) is the opposite of forward stepwise selection.


>    iii. The predictors in the $k$-variable model identified by backward stepwise are a subset of the predictors in the $(k+1)$-variable model identified by forward stepwise selection.


The statement is generally **false**. The relationship between the predictors in the $k$-variable model identified by backward stepwise selection and the predictors in the $(k+1)$-variable model identified by forward stepwise selection is not as straightforward as being subsets of one another. Here's why:

- **Backward Stepwise Selection** starts with all available predictors and sequentially removes the least significant predictor based on a certain criterion (like p-value, AIC, BIC, or RSS) until $k$ predictors remain. The selection of predictors to remove at each step is based on the model's performance with all the current predictors.

- **Forward Stepwise Selection** begins with no predictors and adds one predictor at a time that most improves the model based on the same or similar criteria until $k+1$ predictors are included. The selection of predictors to add at each step depends on the model's performance with the currently included predictors.

Because these two methods start from opposite ends (all predictors vs. no predictors) and follow different paths through the predictor space (removing vs. adding), the set of predictors in a model identified by backward stepwise selection is not necessarily a subset of the predictors in a model identified by forward stepwise selection, especially when comparing models with different numbers of predictors ($k$ vs. $k+1$).

The predictors selected at each step are contingent upon the predictors already in the model and how they interact with the remaining predictors not yet chosen (in the case of forward selection) or those being considered for removal (in the case of backward selection). Therefore, the specific predictors in a model of size $k$ identified by one method are not guaranteed to be a subset of those in a model of size $k+1$ identified by the other method.


>    iv. The predictors in the $k$-variable model identified by forward stepwise are a subset of the predictors in the $(k+1)$-variable model identified by backward stepwise selection.

This statement is **generally false** for similar reasons as the previous one. The relationship between the predictors in the $k$-variable model identified by forward stepwise selection and the predictors in the $(k+1)$-variable model identified by backward stepwise selection does not inherently ensure that one set of predictors is a subset of the other. 

>    v. The predictors in the $k$-variable model identified by best subset are a subset of the predictors in the $(k+1)$-variable model identified by best subset selection.

This statement is **false**. Best subset selection operates differently from both forward stepwise and backward stepwise selection. Here's the key distinction:

- **Best Subset Selection** evaluates all possible combinations of predictors for each model size, from 0 predictors up to the maximum number of available predictors, $p$. For each model size $k$, it selects the combination of $k$ predictors that results in the best model according to a chosen criterion (like lowest RSS, highest $R^2$, etc.).

Given this approach:

1. **Independent Selection for Each Model Size:** The best subset of $k$ predictors is chosen independently of the best subset of $k+1$ predictors. The selection process for each model size does not build upon or reduce from the selection of another model size. Instead, it independently evaluates all combinations of predictors for that specific model size.

2. **No Guaranteed Subset Relationship:** Because each model size is considered independently, the set of $k$ predictors in the best model identified by best subset selection is not guaranteed to be a subset of the $k+1$ predictors in the best model for $k+1$ predictors. The best $k$-predictor model could have a completely different set of predictors than the best $k+1$-predictor model, depending on which combinations yield the optimal criterion value for each model size.

In summary, the predictors in a $k$-variable model identified by best subset selection are not necessarily a subset of the predictors in a $(k+1)$-variable model identified by best subset selection, due to the independent evaluation and selection process for each model size.


### Question 2

> For parts (a) through (c), indicate which of i. through iv. is correct.
> Justify your answer.
>
> a. The lasso, relative to least squares, is:
>    i. More flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.
>    ii. More flexible and hence will give improved prediction accuracy when its increase in variance is less than its decrease in bias.
>    iii. Less flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.
>    iv. Less flexible and hence will give improved prediction accuracy when its increase in variance is less than its decrease in bias.

For part (a), the correct statement is:

iii. Less flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.

**Justification:**

- The Lasso (Least Absolute Shrinkage and Selection Operator) is a regularization technique that adds a penalty equal to the absolute value of the magnitude of coefficients to the loss function. This constraint makes the Lasso less flexible compared to ordinary least squares regression, which does not impose any restrictions on the coefficients and can fit the data as closely as possible (at the risk of overfitting).

- The primary effect of reducing model flexibility through regularization (like Lasso) is to increase bias but decrease variance. The rationale is that by constraining the model (e.g., by shrinking coefficients toward zero), we're deliberately introducing some bias (the model will not fit the training data as closely). However, this trade-off is beneficial when it leads to a significant reduction in variance, as it can improve the model's performance on unseen data by preventing overfitting.

- Therefore, the Lasso can give improved prediction accuracy over least squares when the increase in bias (due to its constraints on the model's flexibility) is more than offset by a decrease in variance, leading to better generalization to new data. This aligns with statement iii, which correctly identifies the Lasso as being less flexible and potentially improving prediction accuracy when the trade-off between increased bias and decreased variance is favorable.


> b. Repeat (a) for ridge regression relative to least squares.

For part (b), the correct statement is the same as for the Lasso:

iii. Less flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.

**Justification:**

- Ridge Regression, like Lasso, is a regularization technique. However, instead of adding a penalty equal to the absolute value of the coefficients (L1 penalty), Ridge Regression adds a penalty equal to the square of the magnitude of coefficients (L2 penalty) to the loss function. This penalty term discourages large coefficients by penalizing them more heavily than smaller ones, leading to a more constrained and hence less flexible model compared to ordinary least squares regression.

- The effect of this regularization is to reduce the variance of the model's predictions at the cost of introducing some bias. By shrinking the coefficients towards zero (but not exactly to zero, as Lasso can), Ridge Regression makes the model less sensitive to the training data, which can help in reducing overfitting and improving the model's performance on unseen data.

- The key to improved prediction accuracy with Ridge Regression, relative to least squares, lies in the trade-off between bias and variance. When the increase in bias (due to the model being less flexible and not fitting the training data as closely) is outweighed by the decrease in variance (resulting in more stable and generalizable predictions), Ridge Regression can outperform least squares.

- Therefore, statement iii accurately describes the scenario under which Ridge Regression would provide improved prediction accuracy relative to least squares: when its increase in bias is less than its decrease in variance, leading to a net improvement in model performance on new data.


> c. Repeat (a) for non-linear methods relative to least squares.

For part (c), the correct statement is:

i. More flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.

**Justification:**

- Non-linear methods, by definition, are more flexible than linear methods like least squares regression. This increased flexibility comes from the non-linear methods' ability to capture complex relationships and patterns in the data that linear models cannot. Examples of non-linear methods include decision trees, neural networks, and kernel-based methods.

- The primary characteristic of more flexible models is their ability to reduce bias by fitting the training data more closely. However, this comes at the potential cost of increasing variance, as highly flexible models can become too tailored to the training data, capturing noise as if it were signal, which can lead to overfitting.

- The statement i suggests that non-linear methods will give improved prediction accuracy over least squares when they can achieve a significant reduction in bias without a corresponding large increase in variance. This is because the key to effective modeling is finding the right balance between bias and variance, known as the bias-variance tradeoff. 

- For non-linear methods to outperform least squares in terms of prediction accuracy, they must be able to capture the underlying patterns in the data more accurately than a linear model (thereby reducing bias), without overfitting to the noise in the training data (thereby controlling the increase in variance). When the decrease in bias outweighs the increase in variance, non-linear methods can provide superior prediction accuracy on new, unseen data.

- It's important to note that while non-linear methods have the potential to significantly improve prediction accuracy due to their flexibility, their performance heavily depends on the specific data set and how well the method is tuned (e.g., parameters, model complexity). Proper model selection, regularization, and validation are crucial to harnessing the benefits of non-linear methods while mitigating the risks of overfitting.


### Question 3

> Suppose we estimate the regression coefficients in a linear regression model
> by minimizing:
>
> $$
> \sum_{i=1}^n\left(y_i - \beta_0 - \sum_{j=1}^p\beta_jx_{ij}\right)^2
>   \textrm{subject to} \sum_{j=1}^p|\beta_j| \le s
> $$
>
> for a particular value of $s$. For parts (a) through (e), indicate which of i. through v. is correct. Justify your answer.
>
> a. As we increase $s$ from 0, the training RSS will:
>    i. Increase initially, and then eventually start decreasing in an inverted U shape.
>    ii. Decrease initially, and then eventually start increasing in a U shape.
>    iii. Steadily increase.
>    iv. Steadily decrease.
>    v. Remain constant.

For part (a), the correct statement is:

iv. Steadily decrease.

**Justification:**

- The given constraint, $\sum_{j=1}^p|\beta_j| \le s$, is characteristic of Lasso regression, which imposes an L1 penalty on the coefficients of the regression model. The parameter $s$ controls the strength of the penalty; as $s$ increases, the constraint becomes less restrictive, allowing the absolute values of the coefficients $\beta_j$ to increase.

- When $s = 0$, the constraint is most restrictive, forcing all coefficients $\beta_j$ to be zero (except for the intercept $\beta_0$, which is not subject to the sum constraint). This results in the simplest model, which is likely to have high bias and high training RSS.

- As $s$ increases from 0, the constraint on the sum of the absolute values of the coefficients becomes less restrictive, allowing some or all of the $\beta_j$ coefficients to move away from zero. This enables the model to fit the training data more closely, thereby reducing the training RSS.

- Therefore, as $s$ increases, the model becomes more flexible, capable of capturing more complex relationships in the data, which leads to a steady decrease in training RSS. The training RSS decreases because the model is allowed to fit the data more closely as the penalty becomes less severe with increasing $s$.



> b. Repeat (a) for test RSS.

ii.

**Conceptual Explanation:**

- As $s$ increases from 0, the constraint on the coefficients becomes less restrictive, allowing the model to become more complex and fit the training data more closely. Initially, this reduction in bias leads to an improvement in test RSS due to better model generalization.

- At some point, as $s$ continues to increase, the model may start to overfit the training data, capturing noise as if it were a true signal. This overfitting results in an increase in variance, which can harm the model's performance on unseen test data, leading to an increase in test RSS.

- Therefore, the behavior of test RSS as $s$ increases is characterized by an initial decrease as the model better captures the underlying data structure, followed by a potential increase as overfitting becomes more pronounced. This pattern suggests a U-shaped relationship between test RSS and $s$, where there is an optimal level of $s$ that minimizes test RSS before it starts to increase due to overfitting.


> c. Repeat (a) for variance.

For part (c), the correct statement is:

iii.

**Justification:**

The variance of the model's predictions is directly related to the model's complexity. As $s$ increases, the constraint on the sum of the absolute values of the coefficients $\sum_{j=1}^p|\beta_j| \le s$ becomes less restrictive, allowing the model to become more complex by letting the coefficients $\beta_j$ move away from zero. This increase in model complexity typically leads to an increase in variance because the model becomes more sensitive to fluctuations in the training data.


> d. Repeat (a) for (squared) bias.

For part (d), the correct statement is:

iv. As $s$ increases, the model becomes more flexible so bias will decrease.

Given the usual relationship between model complexity and bias, and considering the regularization context provided, the squared bias is expected to decrease as $s$ increases from 0. This decrease occurs because allowing the coefficients more freedom (less penalty) enables the model to better capture the underlying data patterns, thus reducing the error due to overly simplistic assumptions (high bias).


> e. Repeat (a) for the irreducible error.

For part (e), the correct statement is:

v. Remain constant.

**Justification:**

- Irreducible error, also known as the noise term in the model, represents the part of the error that cannot be reduced by any model due to the inherent variability in the data or the unknown factors that affect the output. It is not a function of the model's complexity, the choice of predictors, or the value of $s$ in regularization.

- Since the irreducible error is determined by the data itself and not by how the model is specified or optimized, changing the value of $s$ in a regularization context (like Lasso or Ridge Regression) does not impact the irreducible error. The irreducible error remains constant regardless of how model parameters are adjusted, including the regularization parameter $s$.

- The concept of irreducible error is important in understanding the limits of predictive modeling. No matter how sophisticated the model is or how optimal the regularization parameter is chosen, the irreducible error sets a lower bound on the error that can be achieved on new data.

Therefore, as $s$ is varied to adjust the model's complexity in the context of regularization, the irreducible error component of the total error remains unaffected and constant.

### Question 4

> Suppose we estimate the regression coefficients in a linear regression model
> by minimizing
>
> $$
> \sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p\beta_jx_{ij}\right)^2 +
>   \lambda\sum_{j=1}^p\beta_j^2
> $$
>
> for a particular value of $\lambda$. For parts (a) through (e), indicate which of i. through v. is correct. Justify your answer.
>
> a. As we increase $\lambda$ from 0, the training RSS will:
>    i. Increase initially, and then eventually start decreasing in an inverted U shape.
>    ii. Decrease initially, and then eventually start increasing in a U shape.
>    iii. Steadily increase.
>    iv. Steadily decrease.
>    v. Remain constant.

For part (a), the correct statement is:

iii. Steadily increase.

**Justification:**

- The given equation represents the cost function of Ridge Regression, which adds a penalty term to the ordinary least squares (OLS) regression. The penalty term is $\lambda\sum_{j=1}^p\beta_j^2$, where $\lambda$ is the regularization parameter that controls the strength of the penalty applied to the size of the coefficients.

- When $\lambda = 0$, the Ridge Regression model reduces to the ordinary least squares regression model, as the penalty term has no effect. In this case, the model is fully flexible within the constraints of being linear and will fit the training data as closely as possible, resulting in the lowest possible training RSS for a linear model given the data.

- As $\lambda$ increases from 0, the penalty on the size of the coefficients becomes more significant. This penalty discourages large values of the coefficients, effectively shrinking them towards zero. The shrinkage increases bias in the model because it is less able to fit the training data closely, leading to an increase in the training RSS.

- The increase in $\lambda$ continues to constrain the model further, making it less flexible and causing the training RSS to steadily increase. The model becomes increasingly biased, and its ability to capture the variability in the training data diminishes, resulting in higher RSS values.

- Therefore, as $\lambda$ increases from 0, the training RSS will steadily increase because the model is increasingly penalized for complexity, leading to higher bias and less fit to the training data.


> b. Repeat (a) for test RSS.

ii. As $\lambda$ increases, flexibility decreases so test RSS will decrease (variance decreases) but will then increase (as bias increases).

**Conceptual Explanation:**

- As $\lambda$ increases from 0, the Ridge Regression model initially may see an improvement in test RSS. This improvement occurs because the penalty term helps to prevent overfitting by shrinking the coefficients, which can lead to a model that generalizes better to unseen data. This phase corresponds to a decrease in test RSS due to the reduction in variance outweighing the increase in bias.

- However, as $\lambda$ continues to increase beyond a certain point, the model becomes too simple, overly penalizing the coefficients, which leads to underfitting. In this phase, the test RSS may start to increase because the model is no longer complex enough to capture the underlying patterns in the data, resulting in an increase in bias that outweighs the benefits of reduced variance.

- Therefore, the behavior of test RSS as $\lambda$ increases is characterized by an initial decrease as the model becomes more regularized and less prone to overfitting, followed by a potential increase as overfitting is mitigated but underfitting begins to occur. This pattern suggests a U-shaped relationship between test RSS and $\lambda$, where there is an optimal level of $\lambda$ that minimizes test RSS before it starts to increase due to excessive penalization and underfitting.

> c. Repeat (a) for variance.

For part (c), the correct statement is:
iv.

Variance should generally decrease as $\lambda$ increases, because the model becomes consistently simpler and less sensitive to the training data.


> d. Repeat (a) for (squared) bias.

iii. Steadily increase.

- The increase in $\lambda$ leads to a steady increase in the squared bias. This is because the model becomes increasingly simplistic, moving further away from the complexity needed to capture the true underlying data generating process. The simplification results in predictions that are, on average, further from the true values, hence an increase in bias.


> e. Repeat (a) for the irreducible error.

v. The irreducible error is unchanged.

### Question 5

> It is well-known that ridge regression tends to give similar coefficient values to correlated variables, whereas the lasso may give quite different coefficient values to correlated variables. We will now explore this property in a very simple setting.
>
> Suppose that $n = 2, p = 2, x_{11} = x_{12}, x_{21} = x_{22}$. Furthermore, suppose that $y_1 + y_2 =0$ and $x_{11} + x_{21} = 0$ and $x_{12} + x_{22} = 0$, so that the estimate for the intercept in a least squares, ridge regression, or lasso model is zero: $\hat{\beta}_0 = 0$.
>
> a. Write out the ridge regression optimization problem in this setting.

In the given setting, the ridge regression optimization problem can be written as follows:

Given that $n = 2$, $p = 2$, with $x_{11} = x_{12}$, $x_{21} = x_{22}$, $y_1 + y_2 = 0$, and $x_{11} + x_{21} = 0$ (which also implies $x_{12} + x_{22} = 0$), and $\hat{\beta}_0 = 0$, the ridge regression optimization problem is to minimize:

$$
\sum_{i=1}^{2} \left(y_i - \beta_1 x_{i1} - \beta_2 x_{i2}\right)^2 + \lambda \left(\beta_1^2 + \beta_2^2\right)
$$

where:
- $y_i$ are the response variables,
- $x_{i1}$ and $x_{i2}$ are the predictor variables,
- $\beta_1$ and $\beta_2$ are the coefficients to be estimated,
- $\lambda$ is the regularization parameter that controls the amount of shrinkage applied to the coefficients.

This formulation directly incorporates the ridge penalty, $\lambda \left(\beta_1^2 + \beta_2^2\right)$, which penalizes the square of the coefficients, effectively shrinking them towards zero depending on the value of $\lambda$. The goal of this optimization problem is to find the values of $\beta_1$ and $\beta_2$ that minimize the penalized residual sum of squares.

> b. Argue that in this setting, the ridge coefficient estimates satisfy $\hat{\beta}_1 = \hat{\beta}_2$

To mathematically prove that $\hat{\beta}_1 = \hat{\beta}_2$ in the given setting for ridge regression, let's start with the optimization problem:

$$
\sum_{i=1}^{2} \left(y_i - \beta_1 x_{i1} - \beta_2 x_{i2}\right)^2 + \lambda \left(\beta_1^2 + \beta_2^2\right)
$$

Given the conditions:
- $x_{11} = x_{12}$, $x_{21} = x_{22}$
- $x_{11} + x_{21} = 0$ and $x_{12} + x_{22} = 0$, which implies $x_{11} = -x_{21}$ and $x_{12} = -x_{22}$
- $y_1 + y_2 = 0$, which implies $y_1 = -y_2$

Substituting the conditions into the optimization problem:

$$
\sum_{i=1}^{2} \left(y_i - \beta_1 x_{i} - \beta_2 x_{i}\right)^2 + \lambda \left(\beta_1^2 + \beta_2^2\right)
$$

Since $x_{i1} = x_{i2} = x_i$, we can simplify the expression:

$$
(y_1 - \beta_1 x_1 - \beta_2 x_1)^2 + (y_2 - \beta_1 x_2 - \beta_2 x_2)^2 + \lambda (\beta_1^2 + \beta_2^2)
$$

Given $x_1 = -x_2$ and $y_1 = -y_2$, let's substitute $x_2 = -x_1$ and $y_2 = -y_1$:

$$
(y_1 - (\beta_1 + \beta_2) x_1)^2 + (-y_1 - (\beta_1 + \beta_2) (-x_1))^2 + \lambda (\beta_1^2 + \beta_2^2)
$$

This simplifies to:

$$
2(y_1 - (\beta_1 + \beta_2) x_1)^2 + \lambda (\beta_1^2 + \beta_2^2)
$$

To minimize this expression with respect to $\beta_1$ and $\beta_2$, we take the partial derivatives and set them to zero.

For $\beta_1$:

$$
\frac{\partial}{\partial \beta_1} \left[ 2(y_1 - (\beta_1 + \beta_2) x_1)^2 + \lambda (\beta_1^2 + \beta_2^2) \right] = 0
$$

For $\beta_2$:

$$
\frac{\partial}{\partial \beta_2} \left[ 2(y_1 - (\beta_1 + \beta_2) x_1)^2 + \lambda (\beta_1^2 + \beta_2^2) \right] = 0
$$

Both partial derivatives will yield equations that are symmetric in $\beta_1$ and $\beta_2$. The symmetry in these equations implies that any solution for $\beta_1$ and $\beta_2$ that minimizes the expression must satisfy $\beta_1 = \beta_2$, as the roles of $\beta_1$ and $\beta_2$ are interchangeable in the context of the optimization problem due to the identical contributions of $x_{i1}$ and $x_{i2}$ to the model.

Thus, mathematically, it's shown that in this specific setting, the ridge regression coefficient estimates must satisfy $\hat{\beta}_1 = \hat{\beta}_2$.


> c. Write out the lasso optimization problem in this setting.

We are trying to minimize:

$$
\sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p\beta_jx_{ij}\right)^2 +
  \lambda\sum_{j=1}^p |\beta_j|
$$

As above (and defining $x_1 = x_{11} = x_{12}$ and $x_2 = x_{21} = x_{22}$) we simplify to

$$
(y_1 - \beta_1x_1 - \beta_2x_1)^2 + 
  (y_2 - \beta_1x_2 - \beta_2x_2)^2 + 
  \lambda|\beta_1| + \lambda|\beta_2|
$$

> d. Argue that in this setting, the lasso coefficients $\hat{\beta}_1$ and $\hat{\beta}_2$ are not unique---in other words, there are many possible solutions to the optimization problem in (c). Describe these solutions.

We will consider the alternate form of the lasso optimization problem

$$
(y_1 - \hat{\beta_1}x_1 - \hat{\beta_2}x_1)^2 + (y_2 - \hat{\beta_1}x_2 - \hat{\beta_2}x_2)^2 \quad \text{subject to} \quad |\hat{\beta_1}| + |\hat{\beta_2}| \le s
$$

Since $x_1 + x_2 = 0$ and $y_1 + y_2 = 0$, this is equivalent to minimising
$2(y_1 - (\hat{\beta_1} + \hat{\beta_2})x_1)^2$
which has a solution when $\hat{\beta_1} + \hat{\beta_2} = y_1/x_1$.
Geometrically, this is a $45^\circ$ backwards sloping line in the 
($\hat{\beta_1}$, $\hat{\beta_2}$) plane.

The constraints $|\hat{\beta_1}| + |\hat{\beta_2}| \le s$ specify a diamond 
shape in the same place, also with lines that are at $45^\circ$ centered at the
origin and which intersect the axes at a distance $s$ from the origin. 

Thus, points along two edges of the diamond
($\hat{\beta_1} + \hat{\beta_2} = s$ and $\hat{\beta_1} + \hat{\beta_2} = -s$) 
become solutions to the lasso optimization problem.

### Question 6

> We will now explore (6.12) and (6.13) further.
>
> a. Consider (6.12) with $p = 1$. For some choice of $y_1$ and $\lambda > 0$, plot (6.12) as a function of $\beta_1$. Your plot should confirm that (6.12) is solved by (6.14).

Equation 6.12 is:

$$
\sum_{j=1}^p(y_j - \beta_j)^2 + \lambda\sum_{j=1}^p\beta_j^2
$$

Equation 6.14 is:

$$
\hat{\beta}_j^R = y_j/(1 + \lambda)
$$

where $\hat{\beta}_j^R$ is the ridge regression estimate.

```{r}
lambda <- 0.7
y <- 1.4
fn <- function(beta) {
  (y - beta)^2 + lambda * beta^2
}
plot(seq(0, 2, 0.01), fn(seq(0, 2, 0.01)), type = "l", xlab = "beta", ylab = "6.12")
abline(v = y / (1 + lambda), lty = 2)
```

> b. Consider (6.13) with $p = 1$. For some choice of $y_1$ and $\lambda > 0$, plot (6.13) as a function of $\beta_1$. Your plot should confirm that
>    (6.13) is solved by (6.15).

Equation 6.13 is:

$$
\sum_{j=1}^p(y_j - \beta_j)^2 + \lambda\sum_{j=1}^p|\beta_j|
$$

Equation 6.15 is:

$$
\hat{\beta}_j^L = \begin{cases}
  y_j - \lambda/2 &\mbox{if } y_j > \lambda/2; \\
  y_j + \lambda/2 &\mbox{if } y_j < -\lambda/2; \\
  0               &\mbox{if } |y_j| \le \lambda/2;
\end{cases}
$$

For $\lambda = 0.7$ and $y = 1.4$, the top case applies.

```{r}
lambda <- 0.7
y <- 1.4
fn <- function(beta) {
  (y - beta)^2 + lambda * abs(beta)
}
plot(seq(0, 2, 0.01), fn(seq(0, 2, 0.01)), type = "l", xlab = "beta", ylab = "6.12")
abline(v = y - lambda / 2, lty = 2)
```

### Question 7

> We will now derive the Bayesian connection to the lasso and ridge regression
> discussed in Section 6.2.2.
>
> a. Suppose that $y_i = \beta_0 + \sum_{j=1}^p x_{ij}\beta_j + \epsilon_i$
>    where $\epsilon_1, ..., \epsilon_n$ are independent and identically
>    distributed from a $N(0, \sigma^2)$ distribution. Write out the likelihood
>    for the data.

\begin{align*}
\mathcal{L} 
  &= \prod_i^n \mathcal{N}(0, \sigma^2) \\
  &= \prod_i^n \frac{1}{\sqrt{2\pi\sigma}}\exp\left(-\frac{\epsilon_i^2}{2\sigma^2}\right) \\
  &= \left(\frac{1}{\sqrt{2\pi\sigma}}\right)^n \exp\left(-\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2\right)
\end{align*}

> b. Assume the following prior for $\beta$: $\beta_1, ..., \beta_p$ are
>    independent and identically distributed according to a double-exponential
>    distribution with mean 0 and common scale parameter b: i.e.
>    $p(\beta) = \frac{1}{2b}\exp(-|\beta|/b)$. Write out the posterior for
>    $\beta$ in this setting.

The posterior can be calculated by multiplying the prior and likelihood
(up to a proportionality constant).

\begin{align*}
p(\beta|X,Y) 
  &\propto \left(\frac{1}{\sqrt{2\pi\sigma}}\right)^n \exp\left(-\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2\right) \prod_j^p\frac{1}{2b}\exp\left(-\frac{|\beta_j|}{b}\right)  \\
  &\propto \frac{1}{2b} \left(\frac{1}{\sqrt{2\pi\sigma}}\right)^n \exp\left(-\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2 -\sum_j^p\frac{|\beta_j|}{b}\right)
\end{align*}

> c. Argue that the lasso estimate is the _mode_ for $\beta$ under this
>    posterior distribution.

Let us find the maximum of the posterior distribution (the mode). Maximizing
the posterior probability is equivalent to maximizing its log which is:

$$
\log(p(\beta|X,Y)) \propto  \log\left[ \frac{1}{2b} \left(\frac{1}{\sqrt{2\pi\sigma}}\right)^n \right ] - \left(\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2 + \sum_j^p\frac{|\beta_j|}{b}\right)
$$

Since, the first term is independent of $\beta$, our solution will be when
we minimize the second term.

\begin{align*}
\DeclareMathOperator*{\argmin}{arg\,min} % Jan Hlavacek
\argmin_\beta \left(\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2 + \sum_j^p\frac{|\beta|}{b}\right)
&= \argmin_\beta \left(\frac{1}{2\sigma^2} \right ) \left( \sum_i^n \epsilon_i^2 +\frac{2\sigma^2}{b}\sum_j^p|\beta_j|\right) \\
&= \argmin_\beta \left( \sum_i^n \epsilon_i^2 +\frac{2\sigma^2}{b}\sum_j^p|\beta_j|\right)
\end{align*}

Note, that $RSS = \sum_i^n \epsilon_i^2$ and if we set $\lambda =
\frac{2\sigma^2}{b}$, the mode corresponds to lasso optimization.
$$
\argmin_\beta RSS + \lambda\sum_j^p|\beta_j|
$$

> d. Now assume the following prior for $\beta$: $\beta_1, ..., \beta_p$ are
>    independent and identically distributed according to a normal distribution
>    with mean zero and variance $c$. Write out the posterior for $\beta$ in
>    this setting.

The posterior is now:

\begin{align*}
p(\beta|X,Y) 
  &\propto \left(\frac{1}{\sqrt{2\pi\sigma}}\right)^n \exp\left(-\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2\right) \prod_j^p\frac{1}{\sqrt{2\pi c}}\exp\left(-\frac{\beta_j^2}{2c}\right)  \\
  &\propto 
   \left(\frac{1}{\sqrt{2\pi\sigma}}\right)^n 
   \left(\frac{1}{\sqrt{2\pi c}}\right)^p
\exp\left(-\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2 - \frac{1}{2c}\sum_j^p\beta_j^2\right)
\end{align*}

> e. Argue that the ridge regression estimate is both the _mode_ and the _mean_
>    for $\beta$ under this posterior distribution.

To show that the ridge estimate is the mode we can again find the maximum by
maximizing the log of the posterior. The log is 

$$
\log{p(\beta|X,Y)}
  \propto 
   \log{\left[\left(\frac{1}{\sqrt{2\pi\sigma}}\right)^n \left(\frac{1}{\sqrt{2\pi c}}\right)^p \right ]}
- \left(\frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2 + \frac{1}{2c}\sum_j^p\beta_j^2 \right)
$$

We can maximize (wrt $\beta$) by ignoring the first term and minimizing the
second term. i.e. we minimize:

$$
\argmin_\beta \left( \frac{1}{2\sigma^2} \sum_i^n \epsilon_i^2 + \frac{1}{2c}\sum_j^p\beta_j^2 \right)\\
= \argmin_\beta \left( \frac{1}{2\sigma^2} \left( \sum_i^n \epsilon_i^2 + \frac{\sigma^2}{c}\sum_j^p\beta_j^2 \right) \right)
$$

As above, if $RSS = \sum_i^n \epsilon_i^2$ and if we set $\lambda =
\frac{\sigma^2}{c}$, we can see that the mode corresponds to ridge optimization.