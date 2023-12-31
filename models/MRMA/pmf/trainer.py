import os
import time
import math

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '999'
import tensorflow as tf
# Disable tensorflow logging

# tf.logging.set_verbosity(tf.logging.FATAL)


import numpy as np
import matplotlib.pyplot as plt

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

    test_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['i']: batch_manager.test_data[:, 0],
            models['j']: batch_manager.test_data[:, 1],
            models['r']: batch_manager.test_data[:, 2],
        })

    return valid_rmse, test_rmse


def _train(session, kind, models, batch_manager, k):
    weight_folder_path = 'models/mrma/pmf/weights/{}/{}'.format(kind, k)
    if not os.path.exists(weight_folder_path):
        os.mkdir(weight_folder_path)

    min_valid_rmse = float("Inf")
    min_valid_iter = 0
    final_test_rmse = float('Inf')

    for iter in range(N_ITER):

        i = batch_manager.train_data[:, 0]
        j = batch_manager.train_data[:, 1]
        r = batch_manager.train_data[:, 2]

        results = session.run(
            [models['train_op'], models['loss'], models['rmse']],
            feed_dict={
                models['i']: i,
                models['j']: j,
                models['r']: r,
            })
        loss, rmse = results[-2], results[-1]
        # print(loss, rmse)
        if math.isnan(loss):
            raise Exception("NaN found!")

        valid_rmse, test_rmse = _validate(session, models, batch_manager)
        if valid_rmse < min_valid_rmse:
            min_valid_iter = iter
            min_valid_rmse = valid_rmse
            final_test_rmse = test_rmse

            U, V = session.run([models['U'], models['V']])
            np.save(os.path.join(weight_folder_path, 'U.npy'), U)
            np.save(os.path.join(weight_folder_path, 'V.npy'), V)

        print('>> ITER:', iter)
        print('>>', valid_rmse, test_rmse)
        print('>>', min_valid_iter, min_valid_rmse, final_test_rmse)

        if iter > min_valid_iter + N_EARLY_STOP_ITER:
            break
        else: 
            # Go back three lines in the terminal
            print('\033[F\033[F\033[F\r', end='')



def main(kind, k):
    batch_manager = BatchManager(kind)
    models = init_models(batch_manager, k)

    # Check GPU device
    device_name = tf.test.gpu_device_name()
    print('>> Found GPU at: {}'.format(device_name))

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    session.run(tf.global_variables_initializer())

    _train(session, kind, models, batch_manager, k)

    session.close()


if __name__ == '__main__':
    for k in range(10, 310, 10):
        print('>> k =', k)
        main("ml-1m", k=k)
