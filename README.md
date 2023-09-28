# **SUB-COFI**
Study of Unfairness Biases in Regression over Dyadic Data

## Hypothesis
Let a dyadic dataset be defined as a set of tuples of the form (x, y, z), where x and y are entities and z is the target variable. We find that, in models of the State of the Art, for any dyad (x, y), the model's predictions will be biased towards the seen average of the target variable z when x and y are involved.
Expressed mathematically, 
```
E[ f(x, y) ] =~ (E[ z | x ]) + (E[ z | y ]) / 2
```
where f is the model's prediction function, E[.] is the expected value, and z is the target variable.

This means that, for any entity x, the model will make small errors if the value of z to predict is close to the seen average of z for entity x, but will make large errors if the value of z to predict is far from the average of z for entity x. That is, the model is overfitting and becoming biased to the average target values of entities. This is a problem because it can induce unfairness in the model's predictions, as the model only focuses on the average of z for each entity, and not on learning the individual characteristics of each entity to generalise its knowledge. 

Additionally, the usage of RMSE as the evaluation metric for dyadic datasets can be misleading, as it does not take into account the bias of the model towards the average of z for each entity. For example, a model that predicts the average of z for each entity may have a very low RMSE, but will be biased towards the average of z for each entity. This means that, in dyadic datasets, RMSE isn't comprehensive enough to evaluate the performance of a model in terms of this bias, and a complementary metric is needed.

## Datasets

The following table shows basic information about the used datasets:

| Dataset | # Users | # Items | # Ratings | Density |
|---------|---------|---------|-----------|---------|
| [Movielens 1M](https://grouplens.org/datasets/movielens/1m/) | 6,040 | 3,706 | 1,000,209 | 0,044 |
| [Movielens 10M](https://grouplens.org/datasets/movielens/10m/) | 69,878 | 10,677 | 10,000,054 | 0,013 |
| [Douban Monti](https://github.com/fmonti/mgcnn) | 3,000 | 3,000 | 136,066 | 0,015 |
| [Netflix Prize](https://www.kaggle.com/netflix-inc/netflix-prize-data) | 480,189 | 17,770 | 100,480,507 | 0.011 |

In all datasets, the target variable is the rating given by a user to an item. We split the datasets into train, validation and test sets (0.85, 0.05, 0.1). We use the train set to train the models and the validation set to tune the hyperparameters. When then use the joint train and validation sets to train the final model, and evaluate it on the test set.

## Model

To agilize the experimentation, we use a basic Matrix Factorization model, which learns a latent representation for each user and item, and predicts the rating of a user to an item as the dot product of the latent representations of the user and the item. Mathematicall, the model is defined as follows:

```
f(x, y) = <u_x, v_y> + b_x + c_y + d
```

where u_x and v_y are the latent representations of user x and item y, respectively, b_x and c_y are the biases of user x and item y, respectively, and d is a global bias. The model is trained by minimizing the Mean Squared Error (MSE) between the predictions and the true ratings. For inference, model predictions are clipped to the range [1, 5].

## BIAS-AUC

In order to globally evaluate the existence of the aformentioned bias to the average of z for each entity, we propose a new metric called BIAS-AUC. This metric is based on the Area under the Curve paradigm, and is defined as follows:
 - 1) For each sample (x, y, z) in the test set, we compute the model's prediction `z' = f(x, y)`. 
 - 2) We then compute the absolute error between the prediction and the true value, `e = |z - z'|`, as well as the distance between the real z value and the average of z for entities x and y in the trainset, `d = z - (E[ z | x ] + E[ z | y ]) / 2`. As such, we have a tuple `(d, e)` for each sample. 
 - 3) We then sort the tuples by d, and compute the Area under the Curve of the resulting curve. This is the BIAS-AUC metric.
 - 4) In order to normalize the metric, we divide it by the maximum possible value of the metric, defined by the area formed by the points `(min(d), 0), (min(d), max(z)-min(z)), (max(d), max(z)-min(z)), (max(d), 0)`. This normalization is done because the maximum possible value of the metric is dataset-dependent, and we want to be able to compare the metric across datasets. The resulting metric is called Normalized BIAS-AUC (NBIAS-AUC) [0, 1].

## Correction of the bias

In order to demostrate the potential for mitigation of this bias, after and possibly during training, we perform a trivial correction by computing a linear regression of the form `z' = a + b*z + c*E[ z | x ] + d*E[ z | y ]` on a mixed probed set consisting of a 20% of the test set and as many samples of the train set. We then define the "inverse" of the model as `z = (z' - a - c*E[ z | x ] - d*E[ z | y ]) / b`, and use it to correct the predictions of the model in the remainder of the test set. We then compute the BIAS-AUC metric on the corrected predictions, and compare it to the BIAS-AUC metric on the uncorrected predictions.

## Results

The following table shows the results of the experiments:

| Dataset | RMSE | RMSE (corrected) | NBIAS-AUC | NBIAS-AUC (corrected) |
|---------|------|------------------|-----------|------------------------|
| Movielens 1M | 0.837 | 1.145 | 0.341 | 0.264 |
| Movielens 10M | 0.783 | 1.059 | 0.338 | 0.240 |
| Douban Monti | 0.739 | 1.615 | 0.347 | 0.349 |
| Netflix Prize | 0.842 | 1.298 | 0.394 | 0.271 |
