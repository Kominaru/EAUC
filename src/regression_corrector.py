import os
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import torch

MODEL_NAME = "MF"
REGRESSION_MODE: Literal["linear", "poly2", "spline", "linear_bins", "linear_sigmoid"] = "linear"
DATA_PARTITION = "test_probe"  # "train" or "test_probe"
DATASET_NAME = "ml-10m"
EXECUTION = 1


os.makedirs(f"outputs/{DATASET_NAME}/{MODEL_NAME}_correction_{REGRESSION_MODE}", exist_ok=True)

train_samples: pd.DataFrame = pd.read_csv(f"outputs/{DATASET_NAME}/{MODEL_NAME}/train_outputs_{EXECUTION}.csv")
test_samples: pd.DataFrame = pd.read_csv(f"outputs/{DATASET_NAME}/{MODEL_NAME}/test_outputs_{EXECUTION}.csv")

all_samples = pd.concat([train_samples, test_samples])

print(f"==============================")
print(f"Dataset statistics")

print(f"\t#Ratings: {len(all_samples)} | Train: {len(train_samples)} | Test: {len(test_samples)}")
print(f"\t#Users:   {all_samples['user_id'].nunique()}")
print(f"\t#Items:  {all_samples['item_id'].nunique()}")

# Compute the average rating per user and per item
user_avg_ratings = train_samples.groupby("user_id")["rating"].mean()
item_avg_ratings = train_samples.groupby("item_id")["rating"].mean()

# Add the avg. ratings as a column to the train and test samples
train_samples["user_avg_rating"] = train_samples["user_id"].map(user_avg_ratings)
train_samples["item_avg_rating"] = train_samples["item_id"].map(item_avg_ratings)
test_samples["user_avg_rating"] = test_samples["user_id"].map(user_avg_ratings)
test_samples["item_avg_rating"] = test_samples["item_id"].map(item_avg_ratings)

# Filter the train and test samples by the average rating of the user and the item
USER_AVG_RATING_RANGE = (0, 5)
ITEM_AVG_RATING_RANGE = (0, 5)

train_samples = train_samples[
    train_samples["user_avg_rating"].between(*USER_AVG_RATING_RANGE)
    & train_samples["item_avg_rating"].between(*ITEM_AVG_RATING_RANGE)
]
test_samples = test_samples[
    test_samples["user_avg_rating"].between(*USER_AVG_RATING_RANGE)
    & test_samples["item_avg_rating"].between(*ITEM_AVG_RATING_RANGE)
]


def create_test_probe(train_samples, test_samples):
    test_probe = test_samples.sample(frac=1 / 5, random_state=0)
    test_samples = test_samples.drop(test_probe.index)
    train_probe = train_samples.sample(n=len(test_probe), random_state=0)
    test_probe = pd.concat([test_probe, train_probe])
    return test_samples, test_probe


#############################
# Linear regression corrector
#############################

if REGRESSION_MODE == "linear":
    lr = LinearRegression()

    test_samples, test_probe = create_test_probe(train_samples, test_samples)
    print(f"Probe size: {len(test_probe)}")

    # Fit the linear regression on the probe set
    lr.fit(test_probe[["rating", "user_avg_rating", "item_avg_rating"]], test_probe["pred"])

    print(
        f"""
    ==============================
    Linear equation:
            pred = {lr.intercept_:.3f} + {lr.coef_[0]:.3f} * rating + {lr.coef_[1]:.3f} * user_avg_rating + {lr.coef_[2]:.3f} * item_avg_rating"""
    )

    def correct_predictions(sample):
        """
        Return the corrected prediction of a sample using the linear regression by solving the equation for rating.
        Original prediction: pred = a + b * rating + c * user_avg_rating + d * item_avg_rating
        Solveing rating: rating = (pred - a - c * user_avg_rating - d * item_avg_rating) / b

        Params:
            sample: A Pandas Row with the columns "rating", "user_avg_rating", "item_avg_rating" and "pred"
            lr: the linear regression model

        Returns:
            The corrected prediction (predicted rating) clipped to the range [1, 5]
        """

        corrected_pred = np.clip(
            (
                sample["pred"].values
                - lr.intercept_
                - lr.coef_[1] * sample["user_avg_rating"].values
                - lr.coef_[2] * sample["item_avg_rating"].values
            )
            / lr.coef_[0],
            1,
            5,
        )

        return corrected_pred


#############################
# SPLINE REGRESSION
#############################

elif REGRESSION_MODE == "spline":
    train_samples["user_bin"] = (2 * train_samples["user_avg_rating"]).apply(int)
    train_samples["item_bin"] = (2 * train_samples["item_avg_rating"]).apply(int)
    train_samples["ui_bin"] = (train_samples["user_bin"] - 1) * 10 + train_samples["item_bin"] - 1

    print("Max bin:", train_samples["ui_bin"].max())
    print("Min bin:", train_samples["ui_bin"].min())

    test_samples["user_bin"] = (2 * test_samples["user_avg_rating"]).apply(int)
    test_samples["item_bin"] = (2 * test_samples["item_avg_rating"]).apply(int)
    test_samples["ui_bin"] = (train_samples["user_bin"] - 1) * 10 + train_samples["item_bin"] - 1

    KNOTS = 20

    test_samples, test_probe = create_test_probe(train_samples, test_samples)

    splines = []

    for i in range(10 * 10):
        samples = test_probe[test_probe["ui_bin"] == i]
        samples = samples.sort_values("pred")

        x = samples["pred"].values
        y = samples["rating"].values

        if len(x) <= KNOTS:
            splines.append(None)
            continue

        splines.append(scipy.interpolate.UnivariateSpline(x, y, k=3, check_finite=True))

    # Compute the corrected predictions for the train and test sets
    def correct_prediction(samples):
        corrected_pred = samples["pred"].values

        for i in range(10 * 10):
            x = samples[samples["ui_bin"] == i]["pred"].values

            if splines[i] is None:
                continue

            corrected_pred[samples["ui_bin"] == i] = np.clip(splines[i](x), 1, 5)

        return corrected_pred


#############################
# POLYNOMIAL REGRESSION
#############################

elif REGRESSION_MODE == "poly2":
    # Fit a polynomial regression model on the train set

    train_samples["user_bin"] = (2 * train_samples["user_avg_rating"]).apply(int)
    train_samples["item_bin"] = (2 * train_samples["item_avg_rating"]).apply(int)
    train_samples["ui_bin"] = (train_samples["user_bin"] - 1) * 10 + train_samples["item_bin"] - 1

    print("Max bin:", train_samples["ui_bin"].max())
    print("Min bin:", train_samples["ui_bin"].min())

    test_samples["user_bin"] = (2 * test_samples["user_avg_rating"]).apply(int)
    test_samples["item_bin"] = (2 * test_samples["item_avg_rating"]).apply(int)
    test_samples["ui_bin"] = (train_samples["user_bin"] - 1) * 10 + train_samples["item_bin"] - 1

    DEGREE = 2

    test_samples, test_probe = create_test_probe(train_samples, test_samples)

    # For each bin, fit a polynomial regression model on the eest probe set

    sol = np.zeros((10 * 10, DEGREE + 1))

    for i in range(10 * 10):
        x = test_probe[test_probe["ui_bin"] == i]["rating"].values
        y = test_probe[test_probe["ui_bin"] == i]["pred"].values

        if len(x) == 0:
            continue

        z = PolynomialFeatures(DEGREE).fit_transform(x.reshape(-1, 1))

        sol[i] = scipy.linalg.lstsq(z, y)[0]

    # Compute the corrected predictions for the train and test sets
    def correct_prediction(samples):
        """
        Return the corrected prediction of a sample using the polynomial regression by solving the equation for rating.
        Original prediction: pred = a + b * rating + c * rating^2
        Solveing rating: rating = (-b +- sqrt(b^2 - 4ac)) / 2a
        """

        a = sol[samples["ui_bin"].values, 2]
        b = sol[samples["ui_bin"].values, 1]
        c = sol[samples["ui_bin"].values, 0]

        corrected_pred = np.clip((-b + np.sqrt(b**2 - 4 * a * (c - samples["pred"].values))) / (2 * a), 1, 5)

        return corrected_pred


#############################
# STRATIFIED REGRESSION
#############################

elif REGRESSION_MODE == "linear_bins":
    # Bin the samples by the average rating of the user and the item (10 intervals in the range [0, 5]) and convert to categorical integers

    train_samples["user_bin"] = (2 * train_samples["user_avg_rating"]).apply(int)
    train_samples["item_bin"] = (2 * train_samples["item_avg_rating"]).apply(int)
    train_samples["ui_bin"] = train_samples["user_bin"] * 10 + train_samples["item_bin"]

    test_samples["user_bin"] = (2 * test_samples["user_avg_rating"]).apply(int)
    test_samples["item_bin"] = (2 * test_samples["item_avg_rating"]).apply(int)
    test_samples["ui_bin"] = test_samples["user_bin"] * 10 + test_samples["item_bin"]

    # Create a dummy variable for each bin by iterating over the bins and creating a new column for each bin
    for i in range(10 * 10):
        train_samples[f"bin_{i}"] = (train_samples["ui_bin"] == i).apply(int)
        train_samples[f"bin_{i}_r"] = (train_samples["ui_bin"] == i) * train_samples["rating"]
        test_samples[f"bin_{i}"] = (test_samples["ui_bin"] == i).apply(int)
        test_samples[f"bin_{i}_r"] = (test_samples["ui_bin"] == i) * test_samples["rating"]

    lr = LinearRegression(fit_intercept=False)

    test_samples, test_probe = create_test_probe(train_samples, test_samples)

    # Learn a linear regression model using all the bin_i and bin_i_r columns as features, and the pred column as target
    lr.fit(
        test_probe[[f"bin_{i}" for i in range(10 * 10)] + [f"bin_{i}_r" for i in range(10 * 10)]],
        test_probe["pred"],
    )

    def correct_predictions(sample):
        """
        Return the corrected prediction of a sample using the linear regression by solving the equation for rating.
        Original prediction: pred = b_n_r*r + b_n (where n is the bin number)
        Solveing rating: rating = (pred - b_n) / b_n_r
        """

        corrected_pred = np.clip(
            (sample["pred"].values - sample[[f"bin_{i}" for i in range(10 * 10)]].values @ lr.coef_[: 10 * 10])
            / (sample[[f"bin_{i}" for i in range(10 * 10)]].values @ lr.coef_[10 * 10 :]),
            1,
            5,
        )

        return corrected_pred


#############################
# LINEAR WITH SIGMOID
#############################

elif REGRESSION_MODE == "linear_sigmoid":
    # Create a nn.module that takes rating, user_avg_rating, item_avg_rating as input and outputs a corrected rating
    # The model is a linear regression with a sigmoid activation function

    class LinearSigmoidRegression(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x)) * 4 + 1

    test_samples, test_probe = create_test_probe(train_samples, test_samples)

    print(f"Probe size: {len(test_probe)}")

    # Fit the model on the probe set
    model = LinearSigmoidRegression()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(7500):
        optimizer.zero_grad()
        inputs = torch.tensor(test_probe[["pred", "user_avg_rating", "item_avg_rating"]].values, dtype=torch.float)
        outputs = model(inputs)
        target = torch.tensor(test_probe["rating"].values.reshape(-1, 1), dtype=torch.float)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.3f}")

    # Print the learned weights
    print(f"==============================")
    print(f"Learnt equation:")
    print(
        f"\trating = Sig({model.linear.weight[0][0].item():.3f} * pred + {model.linear.weight[0][1].item():.3f} * user_avg_rating + {model.linear.weight[0][2].item():.3f} * item_avg_rating + {model.linear.bias[0]:.3f})*4 +1"
    )

    # Compute the corrected predictions for the train and test sets
    def correct_predictions(samples):
        inputs = torch.tensor(samples[["pred", "user_avg_rating", "item_avg_rating"]].values, dtype=torch.float)
        corrected_pred = np.clip(model(inputs).detach().numpy().reshape(-1), 1, 5)
        return corrected_pred


train_samples["pred"] = correct_predictions(train_samples)
test_samples["pred"] = correct_predictions(test_samples)


train_samples = train_samples[["user_id", "item_id", "rating", "pred"]]
test_samples = test_samples[["user_id", "item_id", "rating", "pred"]]

os.makedirs(f"outputs/{MODEL_NAME}_correction_{REGRESSION_MODE}", exist_ok=True)

print("Saving corrected predictions...")

train_samples.to_csv(
    f"outputs/{DATASET_NAME}/{MODEL_NAME}_correction_{REGRESSION_MODE}/train_outputs_{EXECUTION}.csv", index=False
)
test_samples.to_csv(
    f"outputs/{DATASET_NAME}/{MODEL_NAME}_correction_{REGRESSION_MODE}/test_outputs_{EXECUTION}.csv", index=False
)

print(f"==============================")
print(f"Saved corrected Train predictions in outputs/{MODEL_NAME}_correction_{REGRESSION_MODE}/train_samples.csv")
print(
    f"Saved corrected Test predictions in outputs/{MODEL_NAME}_correction_{REGRESSION_MODE}/test_samples_with_predictions.csv"
)

# exit()
