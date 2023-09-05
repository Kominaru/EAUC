import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = 'MF'

train_samples: pd.DataFrame = pd.read_csv(f'outputs/{MODEL_NAME}/train_samples.csv')
test_samples: pd.DataFrame = pd.read_csv(f'outputs/{MODEL_NAME}/test_samples_with_predictions.csv')

# Compute the average rating per user and per movie
user_avg_ratings = train_samples.groupby('user_id')['rating'].mean()
movie_avg_ratings = train_samples.groupby('movie_id')['rating'].mean()

# Add the average ratings to the train and test samples
train_samples['user_avg_rating'] = train_samples['user_id'].map(user_avg_ratings)
train_samples['movie_avg_rating'] = train_samples['movie_id'].map(movie_avg_ratings)

test_samples['user_avg_rating'] = test_samples['user_id'].map(user_avg_ratings)
test_samples['movie_avg_rating'] = test_samples['movie_id'].map(movie_avg_ratings)

# Make a selection of the train and test samples: those from users and movies
# with mean ratings in the range [4, 4.5]

train_samples_selection = train_samples[(train_samples['user_avg_rating'] >= 3) & (train_samples['user_avg_rating'] <= 3.5) & (train_samples['movie_avg_rating'] >= 3) & (train_samples['movie_avg_rating'] <= 3.5)]
test_samples_selection = test_samples[(test_samples['user_avg_rating'] >= 3) & (test_samples['user_avg_rating'] <= 3.5) & (test_samples['movie_avg_rating'] >= 3) & (test_samples['movie_avg_rating'] <= 3.5)]

# Make regression of the predictions on the ratings for the train samples, using linear and polynomial (degree 2) regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

lr = LinearRegression()

# Linear regression
lr.fit(train_samples_selection['rating'].values.reshape(-1, 1), train_samples_selection['pred'].values.reshape(-1, 1))

# Define the inverse of the linear regression function
def inverse_lr(x):

    # y = a*x + b
    # x = (y - b) / a
    return (x - lr.intercept_) / lr.coef_

train_samples_selection['pred_corrected'] = train_samples_selection['pred'].apply(inverse_lr)
test_samples_selection['pred_corrected'] = test_samples_selection['pred'].apply(inverse_lr)

# Clip the corrected predictions to the range [1, 5]
train_samples_selection['pred_corrected'] = train_samples_selection['pred_corrected'].clip(1, 5)
test_samples_selection['pred_corrected'] = test_samples_selection['pred_corrected'].clip(1, 5)

print('Train RMSE (uncorrected): ', mean_squared_error(train_samples_selection['rating'], train_samples_selection['pred'], squared=False))
print('Test RMSE (uncorrected): ', mean_squared_error(test_samples_selection['rating'], test_samples_selection['pred'], squared=False))
print('Train RMSE (corrected): ', mean_squared_error(train_samples_selection['rating'], train_samples_selection['pred_corrected'], squared=False))
print('Test RMSE (corrected): ', mean_squared_error(test_samples_selection['rating'], test_samples_selection['pred_corrected'], squared=False))


# 2x2 grid of plots
# Upper plots are the train samples, lower plots are the test samples
# Left plots are the uncorrected predictions, right plots are the corrected predictions
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, samples in enumerate([train_samples_selection, test_samples_selection]):
    # Group by rating and calculate the mean and standard deviation of the predictions (not the pred_bin)
    df = samples.groupby('rating')['pred'].agg(['mean', 'std']).reset_index()

    # Make sure the dataframe has all the ratings
    df = df.merge(pd.DataFrame({'rating': samples['rating'].unique()}), on='rating', how='outer')

    # Plot the mean and standard deviation of the predictions (circle and area)
    axs[i, 0].plot(df['rating'], df['mean'], color='blue', linewidth=1, marker='o', markersize=3)
    axs[i, 0].fill_between(df['rating'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.2, color='blue')

    # Plot the x=y line
    axs[i, 0].plot([1, samples['rating'].max()], [1, samples['pred'].max()], color='black', linewidth=1)

    # Set the x and y limits
    axs[i, 0].set_xlim([1, samples['rating'].max()])
    axs[i, 0].set_ylim([1,  samples['pred'].max()])

    # Set the x and y labels
    axs[i, 0].set_xlabel('Actual Rating')
    axs[i, 0].set_ylabel('Predicted Rating')

    
    df = samples.groupby('rating')['pred_corrected'].agg(['mean', 'std']).reset_index()

    # Make sure the dataframe has all the ratings
    df = df.merge(pd.DataFrame({'rating': samples['rating'].unique()}), on='rating', how='outer')

    # Make sure the values are floats
    df = df.astype('float')

    # Plot the mean and standard deviation of the predictions (circle and area)
    axs[i, 1].plot(df['rating'], df['mean'], color='blue', linewidth=1, marker='o', markersize=3)
    axs[i, 1].fill_between(df['rating'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.2, color='blue')

    # Plot the x=y line
    axs[i, 1].plot([1, samples['rating'].max()], [1, samples['pred_corrected'].max()], color='black', linewidth=1)

    # Set the x and y limits
    axs[i, 1].set_xlim([1, samples['rating'].max()])
    axs[i, 1].set_ylim([1,  samples['pred_corrected'].max()])

    # Set the x and y labels
    axs[i, 1].set_xlabel('Actual Rating')
    axs[i, 1].set_ylabel('Predicted Rating')

    


plt.tight_layout()
plt.show()