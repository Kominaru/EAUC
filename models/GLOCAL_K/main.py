"""GLocal-K baseline (Han et al., CIKM 2021): global-local kernel autoencoder for matrix completion.

Adapted from the authors' notebook (https://github.com/usydnlp/Glocal_K) to dyadic regression datasets. The
model uses the TensorFlow 1.x graph API (``tf.contrib``), so it needs a TensorFlow 1.x environment, see
``envs_requirements/requirements_GLOCAL_K.txt``. Predictions of the epoch with the best test RMSE are written
to ``outputs/<dataset>/GLOCAL_K/{train,test}_outputs_<n>.csv``.

Run from the repository root, e.g. ``python models/GLOCAL_K/main.py --dataset ml-1m``.
"""

import argparse
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf

from eauc import save_model_outputs
from eauc.datasets import load_dataset

# Hyperparameters shared by all datasets
N_HID = 500
N_DIM = 5
N_LAYERS = 2
GK_SIZE = 3

# Per-dataset hyperparameters: l2 regularisation, sparse regularisation, L-BFGS iterations and training epochs
# of pre-training (p) and fine-tuning (f), and the scale of the global-kernel dot product
HYPERPARAMETERS = {
    "small": dict(lambda_2=20.0, lambda_s=0.006, iter_p=5, iter_f=5, epoch_p=30, epoch_f=60, dot_scale=1),
    "large": dict(lambda_2=70.0, lambda_s=0.018, iter_p=50, iter_f=10, epoch_p=20, epoch_f=30, dot_scale=0.5),
}
HYPERPARAMETER_SET = {
    "ml-100k": "small",
    "kiva-ml-17": "small",
    "dot_2023": "small",
    "tripadvisor-london": "small",
    "ml-1m": "large",
    "gdsc1": "large",
    "ctrpv2": "large",
}


# ----------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------


def rating_matrices(train: pd.DataFrame, test: pd.DataFrame, n_users: int, n_items: int):
    """Dense ``items x users`` rating matrices and their observation masks, from 0-based id frames."""
    matrices = []
    for df in (train, test):
        ratings = np.zeros((n_items, n_users), dtype="float32")
        ratings[df["item_id"].to_numpy(), df["user_id"].to_numpy()] = df["rating"].to_numpy()
        matrices += [ratings, np.greater(ratings, 1e-12).astype("float32")]
    train_r, train_m, test_r, test_m = matrices
    return train_r, train_m, test_r, test_m


def load_matrices(dataset: str, data_dir: str, frac: float = 0.1):
    """Load ``dataset`` and split it into train and test rating matrices (``frac`` of the ratings for test).

    Returns ``(n_items, n_users, train_r, train_m, test_r, test_m)``.
    """
    df = load_dataset(dataset, data_dir)
    if dataset == "gdsc1":
        df["rating"] = df["rating"] - df["rating"].min()  # keep observed entries distinguishable from the mask
    df = df.sample(frac=1).reset_index(drop=True)
    n_test = int(frac * len(df))
    test, train = df.iloc[:n_test], df.iloc[n_test:]

    # Contiguous 0-based ids over the users and items of the whole dataset
    user_to_index = {user: i for i, user in enumerate(np.unique(df["user_id"]).tolist())}
    item_to_index = {item: i for i, item in enumerate(np.unique(df["item_id"]).tolist())}

    def reindex(part):
        return part.assign(
            user_id=part["user_id"].map(user_to_index).to_numpy(), item_id=part["item_id"].map(item_to_index).to_numpy()
        )

    train_r, train_m, test_r, test_m = rating_matrices(
        reindex(train), reindex(test), len(user_to_index), len(item_to_index)
    )
    print(f"users: {len(user_to_index)} | items: {len(item_to_index)} | train: {len(train)} | test: {len(test)}")
    return len(item_to_index), len(user_to_index), train_r, train_m, test_r, test_m


def samples_frame(ratings: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """The observed entries of a rating matrix as a ``user_id, item_id, rating`` frame."""
    items, users = np.where(mask > 0)
    return pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings[items, users]})


# ----------------------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------------------


def local_kernel(u, v):
    dist = tf.norm(u - v, ord=2, axis=2)
    return tf.maximum(0.0, 1.0 - dist**2)


def kernel_layer(x, n_hid, n_dim, lambda_s, lambda_2, activation=tf.nn.sigmoid, name=""):
    """Local-kernelised dense layer; returns its output and its regularisation term."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable("u", initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0.0, 1e-3))
        v = tf.get_variable("v", initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0.0, 1e-3))
        b = tf.get_variable("b", [n_hid])

    w_hat = local_kernel(u, v)

    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])

    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    W_eff = W * w_hat  # Local kernelised weight matrix
    y = activation(tf.matmul(x, W_eff) + b)
    return y, sparse_reg_term + l2_reg_term


def global_kernel(input, gk_size, dot_scale):
    avg_pooling = tf.reduce_mean(input, axis=1)  # Item (axis=1) based average pooling
    avg_pooling = tf.reshape(avg_pooling, [1, -1])
    n_kernel = avg_pooling.shape[1].value

    conv_kernel = tf.get_variable(
        "conv_kernel", initializer=tf.random.truncated_normal([n_kernel, gk_size**2], stddev=0.1)
    )
    gk = tf.matmul(avg_pooling, conv_kernel) * dot_scale  # Scaled dot product
    return tf.reshape(gk, [gk_size, gk_size, 1, 1])


def global_conv(input, W):
    input = tf.reshape(input, [1, input.shape[0], input.shape[1], 1])
    conv2d = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME"))
    return tf.reshape(conv2d, [conv2d.shape[1], conv2d.shape[2]])


def build_networks(R, train_r, train_m, n_users, hp):
    """Pre-training and fine-tuning graphs. Returns ``(pred_p, optimizer_p, pred_f, optimizer_f)``."""

    def layer(x, name, n_hid=N_HID, activation=tf.nn.sigmoid):
        return kernel_layer(x, n_hid, N_DIM, hp["lambda_s"], hp["lambda_2"], activation, name)

    def optimizer(loss, maxiter):
        return tf.contrib.opt.ScipyOptimizerInterface(
            loss, options={"disp": True, "maxiter": maxiter, "maxcor": 10}, method="L-BFGS-B"
        )

    # Pre-training: local kernelised autoencoder
    y, reg_losses = R, None
    for i in range(N_LAYERS):
        y, reg_loss = layer(y, str(i))
        reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss
    pred_p, reg_loss = layer(y, "out", n_users, tf.identity)
    reg_losses = reg_losses + reg_loss
    loss_p = tf.nn.l2_loss(train_m * (train_r - pred_p)) + reg_losses

    # Fine-tuning: global kernel on top of the pre-trained layers (shared variables)
    y = R
    for i in range(N_LAYERS):
        y, _ = layer(y, str(i))
    y_dash, _ = layer(y, "out", n_users, tf.identity)

    gk = global_kernel(y_dash, GK_SIZE, hp["dot_scale"])  # Global kernel
    y_hat = global_conv(train_r, gk)  # Global kernel-based rating matrix

    reg_losses = None
    for i in range(N_LAYERS):
        y_hat, reg_loss = layer(y_hat, str(i))
        reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss
    pred_f, reg_loss = layer(y_hat, "out", n_users, tf.identity)
    reg_losses = reg_losses + reg_loss
    loss_f = tf.nn.l2_loss(train_m * (train_r - pred_f)) + reg_losses

    return pred_p, optimizer(loss_p, hp["iter_p"]), pred_f, optimizer(loss_f, hp["iter_f"])


# ----------------------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------------------


def ndcg_k(y_hat, y, k):
    score_label = sorted(np.stack([y_hat, y], axis=1).tolist(), key=lambda d: d[0], reverse=True)
    ideal = sorted(score_label, key=lambda d: d[1], reverse=True)

    def dcg(pairs):
        return sum((2 ** s[1] - 1) / np.log2(2 + i) for i, s in enumerate(pairs[:k]))

    return dcg(score_label) / dcg(ideal)


def call_ndcg(y_hat, y):
    """User-wise NDCG over the observed entries of ``y`` (users with fewer than two ratings are skipped)."""
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    for i in range(y.shape[0]):
        y_hat_i, y_i = y_hat[i][np.where(y[i])], y[i][np.where(y[i])]
        if y_i.shape[0] < 2:
            continue
        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])
        num += 1
    return ndcg_sum / num


def masked_errors(pred, ratings, mask):
    """RMSE and MAE over the observed entries."""
    squared = (mask * (pred - ratings) ** 2).sum() / mask.sum()
    absolute = (mask * np.abs(pred - ratings)).sum() / mask.sum()
    return np.sqrt(squared), absolute


# ----------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="ml-1m", choices=sorted(HYPERPARAMETER_SET))
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    hp = HYPERPARAMETERS[HYPERPARAMETER_SET[args.dataset]]
    model_params = {**hp, "n_hid": N_HID, "n_dim": N_DIM, "n_layers": N_LAYERS, "gk_size": GK_SIZE}

    n_m, n_u, train_r, train_m, test_r, test_m = load_matrices(args.dataset, args.data_dir)
    train_samples, test_samples = samples_frame(train_r, train_m), samples_frame(test_r, test_m)
    min_clip, max_clip = train_samples["rating"].min(), train_samples["rating"].max()

    tf.compat.v1.config.experimental.set_visible_devices([], "GPU")  # Disable GPU
    R = tf.placeholder("float", [n_m, n_u])
    pred_p, optimizer_p, pred_f, optimizer_f = build_networks(R, train_r, train_m, n_u, hp)

    best_rmse_ep, best_mae_ep, best_ndcg_ep = 0, 0, 0
    best_rmse, best_mae, best_ndcg = float("inf"), float("inf"), 0
    time_cumulative = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for stage, epochs, optimizer, prediction in (
            ("PRE-TRAINING", hp["epoch_p"], optimizer_p, pred_p),
            ("FINE-TUNING", hp["epoch_f"], optimizer_f, pred_f),
        ):
            for i in range(epochs):
                tic = time()
                optimizer.minimize(sess, feed_dict={R: train_r})
                pre = np.clip(sess.run(prediction, feed_dict={R: train_r}), min_clip, max_clip)
                elapsed = time() - tic
                time_cumulative += elapsed

                test_rmse, test_mae = masked_errors(pre, test_r, test_m)
                train_rmse, train_mae = masked_errors(pre, train_r, train_m)

                print(".-^-._" * 12)
                print(stage)
                print("Epoch:", i + 1, "test rmse:", test_rmse, "train rmse:", train_rmse)

                if stage == "FINE-TUNING":
                    test_ndcg = call_ndcg(pre, test_r)
                    print("Epoch:", i + 1, "test mae:", test_mae, "test ndcg:", test_ndcg)

                    if test_rmse < best_rmse:
                        best_rmse, best_rmse_ep = test_rmse, i + 1
                        print("New best test rmse:", best_rmse)
                        for samples in (train_samples, test_samples):
                            samples["pred"] = pre[samples["item_id"], samples["user_id"]]
                    if test_mae < best_mae:
                        best_mae, best_mae_ep = test_mae, i + 1
                    if best_ndcg < test_ndcg:
                        best_ndcg, best_ndcg_ep = test_ndcg, i + 1

                print("Time:", elapsed, "seconds")
                print("Time cumulative:", time_cumulative, "seconds")
                print(".-^-._" * 12)

    print("Epoch:", best_rmse_ep, " best rmse:", best_rmse)
    print("Epoch:", best_mae_ep, " best mae:", best_mae)
    print("Epoch:", best_ndcg_ep, " best ndcg:", best_ndcg)

    save_model_outputs(train_samples, test_samples, "GLOCAL_K", args.dataset, model_params)


if __name__ == "__main__":
    main()
