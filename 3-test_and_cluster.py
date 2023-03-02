NOTES_TEST = 50
NOTES_CLUSTER = 50
NUM_USERS = 55540
NUM_ROUNDS = 20

LAYER_1_SIZE = 10
LAYER_2_SIZE = 10

print("Importing Libraries...")
import time
import torch
import math
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier, log_evaluation
from joblib import load

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

print("Importing Data...")
replay_meta = pd.read_pickle('data_loader/-1_df.pkl')
cache_path = 'data_loader/test_cache_all.dat'
cache_size = os.stat(cache_path).st_size
cache_length = cache_size // (23 * 4)  # 21 = (4 [rot] + 3 [pos]) * 3 + 3
cache = np.memmap(cache_path, dtype=np.float32, mode='r', shape=(int(cache_length), 23))
all_data = top_song_filter(replay_meta, 50)
testData = []
clusterData = []
user_num = len(all_data)
for id in range(0, user_num):
    testData.append((id, all_data[id][2]))
    clusterData.append((id, all_data[id][3]))
testX, testY = getClassifyData(testData)
clusterX, clusterY = getClassifyData(clusterData)

print("Importing Models (1/2)...")
clfs1 = []
for i in tqdm(range(LAYER_1_SIZE)):
    clfs1.append(load('./models/layer1/model' + str(i) + '.pkl'))

print("Importing Models (2/2)...")
clfs2 = []
for i in tqdm(range(LAYER_2_SIZE)):
    clfs2.append(load('./models/layer2/model' + str(i) + '.pkl'))

valid = 0
total = 0

for round in range(NUM_ROUNDS):
    print("Starting Round " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")

    mtrxAtest = []
    mtrxBtest = []
    mtrxAcluster = []
    mtrxBcluster = []

    def predictUser(i):
        preds1 = []
        for j in range(LAYER_1_SIZE):
            preds1.append(clfs1[j].predict_proba(testX[50*i:NOTES_TEST+50*i]))
        pred1 = np.hstack(preds1).sum(axis=0)
        mtrxAtest.append(pred1)

        preds2 = []
        for j in range(LAYER_2_SIZE):
            preds2.append(clfs2[j].predict_proba(testX[50*i:NOTES_TEST+50*i]))
        pred2 = np.hstack(preds2).sum(axis=0)
        mtrxBtest.append(pred2)

        preds1C = []
        for j in range(LAYER_1_SIZE):
            preds1C.append(clfs1[j].predict_proba(clusterX[50*i:NOTES_CLUSTER+50*i]))
        pred1C = np.hstack(preds1C).sum(axis=0)
        mtrxAcluster.append(pred1C)

        preds2C = []
        for j in range(LAYER_2_SIZE):
            preds2C.append(clfs2[j].predict_proba(clusterX[50*i:NOTES_CLUSTER+50*i]))
        pred2C = np.hstack(preds2C).sum(axis=0)
        mtrxBcluster.append(pred2C)

        pred = []
        for j in range(NUM_USERS):
            model2 = j % LAYER_2_SIZE
            pos2 = math.floor(j / LAYER_2_SIZE)
            users_per_round2 = NUM_USERS // LAYER_2_SIZE
            pred.append(pred1[j] + pred2[model2*users_per_round2 + pos2] + pred1C[j] + pred2C[model2*users_per_round2 + pos2])
        return np.argmax(pred)

    print("Testing Accuracy...")
    start_time = time.time()
    users_per_round = NUM_USERS // NUM_ROUNDS
    t = tqdm(range(users_per_round*round, users_per_round*(round+1)), desc='0/0 Valid (0%)')
    for i in t:
        pred = predictUser(i)
        if (pred == i): valid += 1
        total += 1
        t.set_description(str(valid) + "/" + str(total) + " Valid (" + str((valid/total)*100) + "%)")
    end_time = time.time()
    print("Finished in %s Minutes" % ((end_time - start_time) / 60))
    file = open("./stats/clustering/" + str(round) + ".txt", "w")
    file.write(str(end_time - start_time))
    file.close()

    print("Saving Results...")
    np.save('./preds/test/layer1/' + str(round), mtrxAtest)
    np.save('./preds/test/layer2/' + str(round), mtrxBtest)
    np.save('./preds/cluster/layer1/' + str(round), mtrxAcluster)
    np.save('./preds/cluster/layer2/' + str(round), mtrxBcluster)
