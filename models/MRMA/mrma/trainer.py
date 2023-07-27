import os
import time
import math
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '999'
import tensorflow as tf

# Disable tensorflow logging

tf.logging.set_verbosity(tf.logging.FATAL)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.MRMA.common.dataset import DatasetManager
from models.MRMA.common.batch import BatchManager
from .configs import *
from .models import init_models


def _validate(session, models, batch_manager):
    valid_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['i']: batch_manager.valid_data[:, 0],
            models['j']: batch_manager.valid_data[:, 1],
            models['r']: batch_manager.valid_data[:, 2],
        })

    test_rmse, r_hat = session.run(
        [models['rmse'] , models['r_hat']],
        feed_dict={
            models['i']: batch_manager.test_data[:, 0],
            models['j']: batch_manager.test_data[:, 1],
            models['r']: batch_manager.test_data[:, 2],
        })

    return valid_rmse, test_rmse, r_hat


def _train(session, kind, models, batch_manager):
    min_valid_rmse = float("Inf")
    min_valid_iter = 0
    final_test_rmse = float('Inf')

    train_samples = pd.DataFrame({
            'movie_id': batch_manager.train_data[:, 1].astype(np.int32),  # Assuming movie_id is in the second column
            'user_id': batch_manager.train_data[:, 0].astype(np.int32),   # Assuming user_id is in the first column
            'rating': batch_manager.train_data[:, 2],    # Assuming rating is in the third column
        })
    
    test_samples = pd.DataFrame({
            'movie_id': batch_manager.test_data[:, 1].astype(np.int32),  # Assuming movie_id is in the second column
            'user_id': batch_manager.test_data[:, 0].astype(np.int32),   # Assuming user_id is in the first column
            'rating': batch_manager.test_data[:, 2],    # Assuming rating is in the third column
        })


    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs/MRMA'):
        os.makedirs('outputs/MRMA')

    train_samples.to_csv('outputs/MRMA/train_samples.csv', index=False)



    i = batch_manager.train_data[:, 0]
    j = batch_manager.train_data[:, 1]
    r = batch_manager.train_data[:, 2]

    for assign_idx, assign_op in enumerate(models['assign_ops']):
        if assign_idx == 4 or assign_idx == 6:
            continue
        session.run(
            assign_op,
            feed_dict={
                models['i']: i,
                models['j']: j,
                models['r']: r,
            })

    for iter in range(N_ITER):

        for train_op in models['train_ops']:
            for _ in range(1):
                results = session.run(
                    [train_op, models['loss'], models['rmse']],
                    feed_dict={
                        models['i']: i,
                        models['j']: j,
                        models['r']: r,
                    })
                loss, rmse = results[-2], results[-1]
                if math.isnan(loss):
                    raise Exception("NaN found!")

        # results = session.run(
        #     models['train_ops'] + [models['loss'], models['rmse']],
        #     feed_dict={
        #         models['i']: i,
        #         models['j']: j,
        #         models['r']: r,
        #     })
        # loss, rmse = results[-2], results[-1]
        # if math.isnan(loss):
        #     raise Exception("NaN found!")

        for assign_op in models['assign_ops']:
            _, loss, rmse = session.run(
                (assign_op, models['loss'], models['rmse']),
                feed_dict={
                    models['i']: i,
                    models['j']: j,
                    models['r']: r,
                })

            if math.isnan(loss):
                raise Exception("NaN found!")

        # results = session.run(
        #     models['assign_ops'] + [models['loss'], models['rmse']],
        #     feed_dict={
        #         models['i']: i,
        #         models['j']: j,
        #         models['r']: r,
        #     })
        # loss, rmse = results[-2], results[-1]
        # if math.isnan(loss):
        #     raise Exception("NaN found!")

        valid_rmse, test_rmse, r_hat = _validate(session, models, batch_manager)
        if valid_rmse < min_valid_rmse:
            test_samples['pred'] = r_hat
            min_valid_iter = iter
            min_valid_rmse = valid_rmse
            final_test_rmse = test_rmse

        print('>> ITER:', iter)
        print('>>', valid_rmse, test_rmse)
        print('>>', min_valid_iter, min_valid_rmse, final_test_rmse)

        if iter > min_valid_iter + N_EARLY_STOP_ITER:
            test_samples.to_csv('outputs/MRMA/test_samples_with_predictions.csv', index=False)
            break
        else: 
            
            # Go back three lines in the terminal
            print('\033[F\033[F\033[F\r', end='')


def main(kind):

    
    batch_manager = BatchManager(kind)
    models = init_models(batch_manager, kind)

    saver = tf.train.Saver()

    # gpu usage
    use_gpu = False
    if use_gpu:
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        session = tf.Session()

    session.run(tf.global_variables_initializer())

    _train(session, kind, models, batch_manager)

    session.close()


if __name__ == '__main__':
    # kind = DatasetManager.KIND_MOVIELENS_100K

    kind = DatasetManager.KIND_MOVIELENS_1M
    # kind = DatasetManager.KIND_MOVIELENS_10M

    main(kind)
