NUM_TREES = 200
NUM_ROUNDS = 10
NUM_USERS = 55540

print("Importing Libraries...")
import time
import torch
import numpy as np
import pandas as pd
import os
from song_spliter import *
from lightgbm import LGBMClassifier, log_evaluation
from joblib import dump

print("Importing Data...")
trainData = torch.load('./data/train.pt')
validateData = torch.load('./data/validate.pt')

print("Importing Data...")
replay_meta = pd.read_pickle('data_loader/-1_df.pkl')
cache_path = 'data_loader/test_cache_all.dat'
cache_size = os.stat(cache_path).st_size
cache_length = cache_size // (23 * 4)  # 21 = (4 [rot] + 3 [pos]) * 3 + 3
cache = np.memmap(cache_path, dtype=np.float32, mode='r', shape=(int(cache_length), 23))
all_data = top_song_filter(replay_meta, 50)


def getClassifyData(data):
    dataX = []
    dataY = []
    for elem in data:
        idx_range = elem[1]
        this_x = cache[idx_range[0]: idx_range[1], :21]
        i = 5
        while i < this_x.shape[0] - 5:
            this_x_calc = this_x[i - 5:i + 5, :21]
            dataX.append(np.concatenate((this_x_calc.max(axis=0),
                                         this_x_calc.min(axis=0),
                                         this_x_calc.mean(axis=0),
                                         np.median(this_x_calc, axis=0),
                                         np.std(this_x_calc,
                                                axis=0),)))
            dataY.append(elem[0])
            i += 10
        print(elem[0])
    dataX = torch.from_numpy(np.array(dataX))
    dataY = torch.tensor(dataY)
    return dataX, dataY


for round in range(NUM_ROUNDS):
    print("Starting Round " + str(round + 1) + "/" + str(NUM_ROUNDS) + "...")

    print("Selecting Data...")
    trainFrame = []
    validateFrame = []
    for id in range(NUM_USERS)[round::NUM_ROUNDS]:
        trainFrame.append((id, all_data[id][0]))
        validateFrame.append((id, all_data[id][1]))

    print("Processing Data...")
    trainX, trainY = getClassifyData(trainFrame)
    validateX, validateY = getClassifyData(validateFrame)

    print("Training Model " + str(round + 1) + "/" + str(NUM_ROUNDS) + "...")
    clf = LGBMClassifier(boosting_type='goss', colsample_bytree=0.6933333333333332,
                         learning_rate=0.1, max_bin=63, max_depth=-1, min_child_weight=7, min_data_in_leaf=20,
                         min_split_gain=0.9473684210526315, n_estimators=NUM_TREES,
                         num_leaves=33, reg_alpha=0.7894736842105263, reg_lambda=0.894736842105263,
                         subsample=1, n_jobs=16, objective='multiclass', device_type='gpu')
    start_time = time.time()
    clf.fit(trainX, trainY.long(),
            eval_set=[(validateX, validateY)],
            eval_metric='multi_error',
            callbacks=[log_evaluation()])
    end_time = time.time()
    print("Training Finished in %s Minutes" % ((end_time - start_time) / 60))

    print("Saving Model " + str(round + 1) + "/" + str(NUM_ROUNDS) + "...")
    dump(clf, './models/layer2/model' + str(round) + '.pkl')
    file = open("./stats/training/layer2/" + str(round) + ".txt", "w")
    file.write(str(end_time - start_time))
    file.close()
