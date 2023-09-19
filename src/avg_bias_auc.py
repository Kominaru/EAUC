# Utility to compute, given a dataframe of samples, the AVG-BIAS-AUC of the model.
# The AVG-BIAS-AUC is the area under the curve where the x-axis is the distance from the rating
# to the average of the average rating of the user and the average rating of the movie, and the
# y-axis is the error of the prediction

from typing import Literal
import pandas as pd
import numpy as np


def compute_avg_bias_auc(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame = None,
    method: Literal["bins", "ordered", "grouped"] = None,
    bins: int = 100,
) -> float:
    """
    Compute the AVG-BIAS-AUC of the model
    - Params:
        - samples (pandas.DataFrame): dataframe containing the samples. Should have the columns 'rating', 'pred', 'user_id', 'movie_id'
        - method (str): method to use to compute the AVG-BIAS-AUC.
            - 'bins': compute the AVG-BIAS-AUC by dividing the x-axis into bins of equal width and computing the mean error in each bin
            - 'ordered': compute the AVG-BIAS-AUC by ordering the samples by the x-axis
            - 'grouped': same as 'ordered', but first the samples are grouped by their x-axis value and the mean error is computed for each group
        - bins (int): number of bins to use if method is 'bins'
    - Returns:
        - AVG-BIAS-AUC
    """

    # If test_samples is None, assume we want to compute the AVG-BIAS-AUC on the train set
    test_samples = train_samples.copy() if test_samples is None else test_samples

    # Compute the average rating per user and movie
    user_avg_ratings = train_samples.groupby("user_id")["rating"].mean()
    movie_avg_ratings = train_samples.groupby("movie_id")["rating"].mean()

    train_samples["user_avg_rating"] = train_samples["user_id"].map(user_avg_ratings)
    train_samples["movie_avg_rating"] = train_samples["movie_id"].map(movie_avg_ratings)

    test_samples["user_avg_rating"] = test_samples["user_id"].map(user_avg_ratings)
    test_samples["movie_avg_rating"] = test_samples["movie_id"].map(movie_avg_ratings)

    # Filter out samples for which the user or movie average rating is not available
    test_samples = test_samples[~test_samples["user_avg_rating"].isna()]
    test_samples = test_samples[~test_samples["movie_avg_rating"].isna()]

    # Compute the rating-avg distances
    test_samples["user_rating_avg_dist"] = (
        test_samples["rating"]
        - (test_samples["user_avg_rating"] + test_samples["movie_avg_rating"]) / 2
    )

    test_samples["error"] = (test_samples["rating"] - test_samples["pred"]).abs()

    # Option 1) Compute the AVG-BIAS-AUC by dividing the x-axis into bins of
    # equal width and computing the mean error in each bin
    if method == "bins":
        bins = np.linspace(
            -(test_samples["rating"].max() - test_samples["rating"].min()),
            +(test_samples["rating"].max() - test_samples["rating"].min()),
            bins,
        )

        errors_by_dist = test_samples.groupby(
            pd.cut(test_samples["user_rating_avg_dist"], bins)
        )["error"].mean()

        xx = errors_by_dist.index.to_series().apply(lambda x: x.mid).values
        yy = errors_by_dist.values

    # Option 2) Compute the AVG-BIAS-AUC by ordering the samples by the x-axis
    elif method == "ordered":
        avg_bias_auc = test_samples.sort_values("user_rating_avg_dist")

        xx = avg_bias_auc["user_rating_avg_dist"].values
        yy = avg_bias_auc["error"].values

    # Option 3) Compute the AVG-BIAS-AUC by grouping the samples by the x-axis
    elif method == "grouped":
        avg_bias_auc = test_samples.groupby("user_rating_avg_dist")["error"].mean()

        xx = avg_bias_auc.index.values
        yy = avg_bias_auc.values

    else:
        raise ValueError(
            f"Method {method} not recognized. Must be one of ['bins', 'ordered', 'grouped']"
        )

    xx = xx[~np.isnan(yy)]  # Filter out NaNs. Otherwise, np.trapz returns NaN
    yy = yy[~np.isnan(yy)]

    # Compute the AVG-BIAS-AUC and normalize to [0, 1]
    avg_bias_auc = np.trapz(yy, xx) / (
        (xx.max() - xx.min())
        * (test_samples["rating"].max() - test_samples["rating"].min())
    )  # The maximum error is the difference between the maximum and minimum rating (4 in MovieLens),
    # and the avg-to-rating distances are taken from the data directly. Therefore, the worst case
    # scenario would be a rectangle of width (xx.max() - xx.min()) and height (test_samples["rating"].max() - test_samples["rating"].min())

    # Note that, in this case, the lower the AVG-BIAS-AUC, the better the model
    return avg_bias_auc
