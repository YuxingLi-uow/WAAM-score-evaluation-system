import pandas as pd
import glob
import numpy as np

# this file is used to upsample the defect data

def save_feature(features, feature_csv='MildsteelFeaturetest.csv'):
    with open(feature_csv, 'ab') as f:  # 将新的到的features append到文件中
        # np.savetxt(f, np.zeros((1,23)), fmt='%.6f', delimiter=',')
        np.savetxt(f, features, fmt='%.6f', delimiter=',')
        print('Data features and scores saved successful to {}'.format(feature_csv))
    f.close()

floder_path = 'Kmeans'
feature_path = glob.iglob(floder_path + '/*.csv')

# feature_path = ['LabelData/feature_layer3 3.csv', 'LabelData/feature_layer3 5.csv']

for path in feature_path:
    feature_array = pd.read_csv(path)
    print('feature array shape', feature_array.shape)
    # label_array = feature_array['label']
    # print('label array shape', label_array.shape)
    labeled_array = feature_array[feature_array['label'] > 0]
    labeled_array = labeled_array.to_numpy()
    # print(labeled_array.shape)

    if labeled_array.shape[0] < 20:
        n = 30
    else:
        n = 20

    for i in range(n):
        labeled_array[:, 1] += 0.01
        print('{}/{} label unsampling of {}'.format(i, n, path))
        save_feature(labeled_array, feature_csv=path)

    # for label in range(1, 6):
    #     tmp_array = labeled_array[labeled_array[:, -1].astype('int') == label]
    #     # print(tmp_array)
    #     if tmp_array.shape[0] < 10:
    #         n = 10
    #     else:
    #         n = 5
    #     for i in range(n):
    #         tmp_array[:, 0] += 0.5
    #         print('{}/{} label {} unsampling of {}'.format(i, n, label, path))
    #         save_feature(tmp_array, feature_csv=path)























