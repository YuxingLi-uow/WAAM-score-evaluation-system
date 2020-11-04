import numpy as np
import pandas as pd
from scipy import signal
import yaml
import glob
import sys
import creme
import pickle
import time
import logging
import math


class CurVolFeatureExtract:
    """
    This class is used for feature extraction for current and voltage raw signals.
    Input raw data collected from the LabVIEW program and this class provides feature extraction functions
    """

    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.data_row, self.data_col = self.data.shape
        self.data_time = self.params['data_time']
        self.window_len = self.params['window_len']
        self.features = np.array([])
        self.idle_stop = 0
        self.idle_start = 0


    def data_idlefound(self):
        idlestop = np.argmax(self.data['cur'] > 10)  # stop idle time and start welding
        idlestart =  np.argmax(self.data['cur'][idlestop:] < 10) # start idle time and stop welding
        # print(idlestop, idlestop + idlestart)
        self.idle_stop = self.params['data_time'] * idlestop
        self.idle_start = self.params['data_time'] * (idlestop + idlestart)
        return idlestop, idlestart

    def data_preprocess(self, column1, column2):
        """
        This method change the column name of the data to 'time', 'cur', 'vol'
        and reindex the data time column from 0 to last row, step as 0.0004
        :return: self.data: the pd frame of the data after rename the column names and reindex
        """
        data_time = self.data_time
        # 重命名data的列名
        self.data.rename(columns={'waveform': 'time', '[0]': column1, '[1]': column2}, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        # 修改时间序列到第一列，将数据改成float类型
        self.data['time'] = np.linspace(0, data_time * self.data_row, num=self.data_row).astype('float32')
        self.data['cur'] = self.data['cur'].astype('float32')
        self.data['vol'] = self.data['vol'].astype('float32')
        # 过滤掉起弧时间，预设是 0.6s
        thres_start, thres_stop = self.data_idlefound()
        self.data.drop(self.data.index[0: 1 + int(self.params['arc start time'] / data_time) + thres_start], 0, inplace=True)
        self.data.drop(self.data.index[-self.data_row + thres_start + thres_stop -int(self.params['arc stop time'] / data_time): -1], 0, inplace=True)
        self.data['time_scale'] = np.linspace(0, 1, num=self.data.shape[0]).astype('float32')
        self.data.reset_index(drop=True, inplace=True)
        self.data_row, self.data_col = self.data.shape
        return self.data


    def data_filter(self, window_length=3, polyorder=1):
        """
        Apply a Savitzky-Golay filter to data.
        :param: window_length: The length of the filter window (i.e. the number of coefficients).
                window_length must be a positive odd integer.
        :param: polyorder: The order of the polynomial used to fit the samples.
                polyorder must be less than window_length.
        :return: data after Savitzky-Golay filter
        """
        if window_length <= polyorder:
            raise ValueError('window length must greater than polyorder')
        if window_length % 2 != 1:
            raise ValueError('window length must be positive odd number')
        self.data['cur'] = signal.savgol_filter(self.data['cur'], window_length, polyorder)
        self.data['vol'] = signal.savgol_filter(self.data['vol'], window_length, polyorder)
        return self.data


    def min_max_scale(self):  # 归一化 data to [0, 1] (use min-max scaling method)
        """
        :return: self.data: the class global data after normalization
        """
        # self.data['time_scale'] = (self.data['time_scale'] - self.data['time_scale'].min()) / (self.data['time_scale'].max()- self.data['time_scale'].min())
        self.data['cur'] = (self.data['cur'] - self.data['cur'].min()) / (self.data['cur'].max()- self.data['cur'].min())
        self.data['vol'] = (self.data['vol'] - self.data['vol'].min()) / (self.data['vol'].max() - self.data['vol'].min())
        return self.data


    def feature_integrate(self, timeIndex, timeScale, cur_feature, vol_feature):
        self.features = np.concatenate((timeIndex, timeScale, cur_feature, vol_feature))
        return self.features


    def window_calculate(self, window_length):
        """
        :param window_length: the length of the window we used for feature extraction
        :return: mask: index mask, which could be used for data slicing
                 num_window: count of windows
        """

        self.window_len = window_length
        num_win = np.ceil(self.data_row / window_length).astype('int')  # 计算number of window
        reminder = self.data_row % window_length  # 计算余数
        mask = np.zeros((num_win, 2))

        if reminder:  # 如果余数存在
            for i, j in enumerate(range(0, self.data_row, window_length)):
                mask[i] = [j, j + window_length]
            else:
                mask[-1, :] = [self.data_row - window_length, self.data_row]  # 最后如果不足window_length，向前取数补足
        else:
            for i, j in enumerate(range(0, self.data.shape[0], window_length)):
                mask[i] = [j, j + window_length]
        return mask.astype('int'), num_win.astype('int')  # 返回所有window的index组成的array


    def data_process(self, startIdx, endIdx, typ='cur', scale_ratio=1.0, peak_width=5, peak_gap_threshold=5):
        """
        Extract features from data

        :param typ: index name of data, 'cur' = current data, 'vol' = voltage data
        :param startIdx: index of start process data
        :param endIdx: index of end process data
        :param scale_ratio: used to scale data mean value as threshold to find peaks
        :param peak_width: peak width in sample, used for peak width threshold
        :param peak_gap_threshold: filter two neighbor peak index
        :return: data_feature: a ndarray of data features, contains:
                 'mean': data mean
                 'std': data std
                 'max': data max
                 'min': data min
                 'peak mean': mean of peak found
                 'peak std': std of peak found
                 'peak count': the number of peak
                 'peak width mean': mean of peak width
                 'peak width std': std of peak width
                 'peak gap mean': mean of peak gap, used for indicating frequency mean, various from 0 to 100
                 'peak gap std': std of peak gap, used for indicating frequency std
        """
        process_data = self.data[typ][startIdx: endIdx]
        data_features = np.zeros(11)  # 11 features in total
        data_features[0] = process_data.mean()  # mean
        peaks_found = signal.find_peaks(process_data, height=scale_ratio * data_features[0], width=peak_width)
        data_features[1] = process_data.std()  # std
        data_features[2] = process_data.max()  # max
        data_features[3] = process_data.min()  # min
        data_features[6] = len(peaks_found[0])  # peak count
        if data_features[6] == 0:
            data_features[4] = 0
            data_features[5] = 0
            data_features[7] = 0
            data_features[8] = 0
            data_features[9] = 0
            data_features[10] = 0
        else:
            data_features[4] = peaks_found[1]['peak_heights'].mean()  # peak mean
            data_features[5] = peaks_found[1]['peak_heights'].std()  # peak std
            data_features[7] = peaks_found[1]['widths'].mean()  # peak width mean
            data_features[8] = peaks_found[1]['widths'].std()  # peak width std

            if data_features[6] > 1:
                data_features[9] = np.diff(peaks_found[0])[np.diff(peaks_found[0]) > peak_gap_threshold].mean()  # peak gap mean
                data_features[10] = np.diff(peaks_found[0])[np.diff(peaks_found[0]) > peak_gap_threshold].std()  # peak gap std
                if math.isnan(data_features[9]):
                    data_features[9] = 0
                if math.isnan(data_features[10]):
                    data_features[10] = 0
            else:
                data_features[9] = 0
                data_features[10] = 0

        return data_features


class CsvFileObserve:
    def __init__(self, configPath, state):
        self.config_path = configPath
        self.csv_file_path = load_params(self.config_path)['feature csv file']
        self.prev_file_num = load_params(self.config_path)['prev data num']
        self.cur_file = glob.glob(self.csv_file_path + '/*.csv')
        self.cur_file_num = self.cur_file.__len__()
        self.state = state

    def Flag(self):
        if self.cur_file_num > self.prev_file_num:
            # print('previous file num', self.prev_file_num)
            # print('current file num', self.cur_file_num)
            logging.debug('previous file num {}'.format(self.prev_file_num))
            logging.debug('current file num {}'.format(self.cur_file_num))
            recursive_num = self.cur_file_num - self.prev_file_num
            if self.state == 'train':
                self.set_prev_num()
            return True, recursive_num
        else:
            recursive_num = 0
            return False, recursive_num

    def set_prev_num(self):
        write_params(self.config_path, 'prev data num', self.cur_file_num)
        # print('Previous file num in config.yaml file is overwrote to {}'.format(self.cur_file_num))
        logging.debug('Previous file num in config.yaml file is overwrote to {}'.format(self.cur_file_num))


class ScoreAnalysis:
    """
    Analyse the score calculated: window data mean and std.
    """
    def __init__(self, score, window_length=10):
        self.array = score
        self.window_length = window_length
        self.data_row, self.data_column = self.array.shape

    def window_calculate(self):
        num_win = np.ceil(self.data_row / self.window_length).astype('int')  # 计算number of window
        reminder = self.data_row % self.window_length  # 计算余数
        mask = np.zeros((num_win, 2))

        if reminder:  # 如果余数存在
            for i, j in enumerate(range(0, self.data_row, self.window_length)):
                mask[i] = [j, j + self.window_length]
            else:
                mask[-1, :] = [self.data_row - self.window_length, self.data_row]  # 最后如果不足window_length，向前取数补足
        else:
            for i, j in enumerate(range(0, self.array.shape[0], self.window_length)):
                mask[i] = [j, j + self.window_length]
        return mask.astype('int'), num_win.astype('int')  # 返回所有window的index组成的array

    def score_analyse(self):
        mask, num_win = self.window_calculate()
        score_feature = np.zeros((num_win, 3))  # time mean std
        for i in range(num_win):
            score_feature[i, 0] = self.array[mask[i, 0], 0]
            score_feature[i, 1] = self.array[mask[i, 0]: mask[i, 1], -1].mean()
            score_feature[i, 2] = self.array[mask[i, 0]: mask[i, 1], -1].std()
        return score_feature



def load_params(filename):
    """
    Read yaml file to load params setting in the welding process.
    :param filename: yaml file path
    :return: dict type of params in yaml file
    """
    with open(filename) as stream:
        params = yaml.safe_load(stream)
    return params


def write_params(filename, var, value):
    with open(filename) as stream:
        params = yaml.load(stream)
        params[var] = value
    with open(filename, 'w') as stream:
        yaml.dump(params, stream)


def get_array_statistical(x):
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    return feature_mean, feature_std


def evaluate_scores(data_array):
    m, s = get_array_statistical(data_array)
    # mean - std < data < mean + std, get score 0.9
    scores = (np.array(data_array < m + s) & np.array(data_array > m - s)) * 0.6
    # mean - 1.5std < data < mean + 1.5std, get score 0.4
    scores += (np.array(data_array < m + 1.5 * s) & np.array(data_array > m - 1.5 * s)) * 0.3
    # data out this range, get score 0.1
    scores += 0.1
    return scores


def save_feature(features, feature_csv='MildsteelFeaturetest.csv'):
    with open(feature_csv, 'ab') as f:  # append features to files
        header = 'date time,time scale,'\
                 'cur mean,cur std,cur max,cur min,cur peak mean,cur peak std,cur peak count,' \
                 'cur peak width mean,cur peak width std,cur gap mean,cur gap std, ' \
                 'vol mean,vol std,vol max,vol min,vol peak mean,vol peak std,vol peak count,' \
                 'vol peak width mean,vol peak width std,vol gap mean,vol gap std,score,label'
        # np.savetxt(f, np.zeros((1,23)), fmt='%.6f', delimiter=',')
        np.savetxt(f, features, fmt='%.6f', header=header, delimiter=',')
        print('Data features and scores saved successful to {}'.format(feature_csv))
    f.close()


def evaluate_prediction(features, predicts):
    defect_class = load_params('defect class config.yaml')
    # feature_counts = dict(pd.value_counts(predicts[:, -1]))
    label_dict = {}
    for i in defect_class.keys():
        if i > 0:
            label_dict[defect_class[i]] = features[predicts[predicts[:, -1] == i][:,0].astype('int'), 0]
    return label_dict


def optim_prediction(time_dict, params, pool_shift_threshold=1):

    wrong_pred = {}
    optim_time = time_dict
    # optimize weld pool shift prediction
    # discontinuous time stamp in weld pool shift is moved to arc unstable
    t = time_dict[params[1]]
    wrong_pred[params[1]] = []
    stamp_gap = np.around(t[1:] - t[:-1], 2)
    stamp_bool = stamp_gap >= pool_shift_threshold
    print(stamp_bool)
    for i in range(len(stamp_bool) - 1):
        if all([stamp_bool[i], stamp_bool[i + 1]]) is True:
            optim_time[params[3]] = np.append(optim_time[params[3]], t[i + 1])
            wrong_pred[params[1]].append(i + 1)  # append wrong predict index to wrong predict dict
            # print(stamp_gap[i + 1])
    if stamp_bool[-1]:
        optim_time[params[3]] = np.append(optim_time[params[3]], t[-1])
        wrong_pred[params[1]].append(len(stamp_bool))  # append wrong predict index to wrong predict dict
    if stamp_bool[0]:
        optim_time[params[3]] = np.append(optim_time[params[3]], t[0])
        wrong_pred[params[1]].append(0)  # append wrong predict index to wrong predict dict
    optim_time[params[1]] = np.delete(optim_time[params[1]], wrong_pred[params[1]])  # New version of np.delete could accept negative index (higher than v1.18.5)
    return optim_time

def feature_extraction(filepath):

    params = load_params('config.yaml')

    cur_vol_data = pd.read_csv(filepath)[4:]
    FeatureExtract = CurVolFeatureExtract(cur_vol_data, params)
    # rename, reindex, drop arc start and stop time
    FeatureExtract.data_preprocess('vol', 'cur')
    print('Data preprocess successful')
    FeatureExtract.data_filter(3, 1)  # data clean use Savitzky-Golay filter
    print('Data noise filter successful')
    cur_vol_data = FeatureExtract.min_max_scale()  # data normalize, min-max-scale
    print('Data normalization successful')

    index_mask, n_win = FeatureExtract.window_calculate(params['window_len'])  # 100 data as a window
    time_gap = params['window_len'] * params['data_time']
    final_features = np.zeros((n_win, 26))   # 1 time + 1 time scale + 22 features + 1 score + 1 label

    for i in range(n_win):
        # print(FeatureExtract.data['time'][i * params['window_len']])
        # time_index = np.array([i * time_gap]) + params['arc start time'] + FeatureExtract.idle_stop
        time_index = np.array([FeatureExtract.data['time'][i * params['window_len']]])
        time_scale = np.array([FeatureExtract.data['time_scale'][i * params['window_len']]])
        cur_features = FeatureExtract.data_process(index_mask[i, 0], index_mask[i, 1], typ='cur', scale_ratio=1.5,
                                                   peak_width=5, peak_gap_threshold=5)  # current features
        vol_features = FeatureExtract.data_process(index_mask[i, 0], index_mask[i, 1], typ='vol', scale_ratio=1.5,
                                                   peak_width=5, peak_gap_threshold=5)  # vol features
        final_features[i, :-2] = FeatureExtract.feature_integrate(time_index, time_scale, cur_features, vol_features)
    else:
        print('Data features extraction successful')

    final_features[:, -2] = np.mean(evaluate_scores(final_features[:, 2:25]), axis=1)
    print('Data scores evaluation successful\n')

    ScoreAnalys = ScoreAnalysis(np.stack((final_features[:, 0], final_features[:, -2]), axis=-1))
    score_analys = ScoreAnalys.score_analyse()

    return final_features, score_analys, FeatureExtract.idle_stop, FeatureExtract.idle_start


def logger_setup(log_file):
    logger = logging.getLogger()
    logger.setLevel('DEBUG')

    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    strmHandler = logging.StreamHandler()  # Used for console print
    strmHandler.setFormatter(formatter)
    fileHandler = logging.FileHandler(log_file)  # Used for file logger save
    fileHandler.setFormatter(formatter)

    logger.addHandler(strmHandler)
    logger.addHandler(fileHandler)


# def run_gui(defect_class, threshold):
#     app = QApplication(sys.argv)
#     mainWindow = DisplayFeature(defect_class, threshold)
#     # sys.exit(app.exec_())
#     mainWindow.show()
#     app.exec_()
#     return mainWindow.rect_region, mainWindow.bad_features, mainWindow.features, mainWindow.displaytext


# def generate_labelData_auto(recursive_time, flag, file_dir, defect_class, params):
#     if flag:
#         for i in range(recursive_time):
#             final_features = feature_extraction(file_dir[i])  # feature extracted here
#             ################################################################################################################
#             # TODO: based on the scores we get in features, return bad data and require manually label (A GUI may needed?) #
#             ROI_region, bad_features_labels = run_gui(defect_class, params['threshold'])
#             print('Label index generated, gui closed')
#
#             if bad_features_labels[:, 2].any() == 0:
#                 print('no data labels modified from table')
#             else:
#                 labels = bad_features_labels[bad_features_labels[:, 2] != 0]
#                 for j, t in enumerate(labels[:, 0]):
#                     labels_index = np.where(final_features[:, 0] == t)
#                     final_features[labels_index, -1] = labels[j, -1]
#                 print('{} data labels modified from table'.format(labels.shape[0]))
#
#             for j in range(len(ROI_region)):
#                 idex_start = np.floor(
#                     (ROI_region[j]['pos'][0] - params['arc start time']) / (params['data_time'] * params['window_len']))
#                 idex_stop = np.ceil(
#                     (ROI_region[j]['pos'][0] + ROI_region[j]['size'][0] - params['arc start time']) / (
#                                 params['data_time'] * params['window_len']))
#                 final_features[int(idex_start): int(idex_stop), -1] = int(ROI_region[j]['class'])
#             print('{} ROI region labels modified'.format(len(ROI_region)))
#             ################################################################################################################
#             save_feature(final_features,
#                          feature_csv=params['feature csv file'] + '/feature_{}'.format(file_dir.rsplit('\\', 1)[-1]))
#
#             print()
#     else:
#         print('No new csv file found')


# def generate_labelData(defect_class, params):
#
#     ################################################################################################################
#     # TODO: based on the scores we get in features, return bad data and require manually label (A GUI may needed?) #
#     ROI_region, bad_features_labels, final_features, file_dir = run_gui(defect_class, params['threshold'])
#     print('Label index generated, gui closed')
#
#     if bad_features_labels[:, 2].any() == 0:
#         print('no data labels modified from table')
#     else:
#         labels = bad_features_labels[bad_features_labels[:, 2] != 0]
#         for j, t in enumerate(labels[:, 0]):
#             labels_index = np.where(final_features[:, 0] == t)
#             final_features[labels_index, -1] = labels[j, -1]
#         print('{} data labels modified from table'.format(labels.shape[0]))
#
#     for j in range(len(ROI_region)):
#         idex_start = np.floor(
#             (ROI_region[j]['pos'][0] - params['arc start time']) / (params['data_time'] * params['window_len']))
#         idex_stop = np.ceil(
#             (ROI_region[j]['pos'][0] + ROI_region[j]['size'][0] - params['arc start time']) / (
#                         params['data_time'] * params['window_len']))
#         final_features[int(idex_start): int(idex_stop), -1] = int(ROI_region[j]['class'])
#     print('{} ROI region labels modified'.format(len(ROI_region)))
#     ################################################################################################################
#     save_feature(final_features,
#                  feature_csv=format(file_dir.rsplit('\\', 2)[0] +
#                                     '/{}'.format(params['feature csv file']) +
#                                     '/feature_{}'.format(file_dir.rsplit('\\', 1)[-1])))
#     print()



def run_incremental_frame(model, recursive_time, file_dir, params, state, array=np.array([0])):
    for i in range(recursive_time):

        if state == 'train':
            # train_csv = params['feature csv file'] + '/feature_{}'.format(file_dir[i].rsplit('\\', 1)[-1])
            train_csv = file_dir[i]
            train_data = np.asarray(pd.read_csv(train_csv))
            train_data = train_data.astype('float32')
            data_stream = creme.stream.iter_array(train_data[:, 1:25], train_data[:, -1])

            logging.info('{}/{} - {} data transferred '.format(i + 1, recursive_time, train_csv))
            logging.info('[INFO] Start training...')
            metric_confusion = creme.metrics.ConfusionMatrix()
            metric_precision = creme.metrics.MacroPrecision()
            metric_recall = creme.metrics.MacroRecall()
            metric_F1 = creme.metrics.MacroF1()

            # loop over the dataset
            for (j, (X, y)) in enumerate(data_stream):
                # make predictions on the current set of features, train the
                # model on the features, and then update our metric
                preds = model.predict_one(X)
                model = model.fit_one(X, y)  # this line should be enabled in train, and disabled in val
                metric_confusion = metric_confusion.update(y, preds)
                metric_precision = metric_precision.update(y, preds)
                metric_recall = metric_recall.update(y, preds)
                metric_F1 = metric_F1.update(y, preds)
                if j % 100 == 0:
                    logging.info("[INFO] update {} - {}, {}, {} \n{}".format(
                        j, metric_precision, metric_recall, metric_F1, metric_confusion))

            logging.info("[INFO] final - {}, {}, {} \n{}\n".format(
                metric_precision, metric_recall, metric_F1, metric_confusion))


        if state == 'val':
            # train_csv = params['feature csv file'] + '/feature_{}'.format(file_dir[i].rsplit('\\', 1)[-1])
            train_csv = file_dir[i]
            train_data = np.asarray(pd.read_csv(train_csv))
            train_data = train_data.astype('float32')
            data_stream = creme.stream.iter_array(train_data[:, 1:25], train_data[:, -1])

            logging.info('{}/{} - {} data transferred '.format(i + 1, recursive_time, train_csv))
            logging.info('[INFO] Start validation...')
            metric_confusion = creme.metrics.ConfusionMatrix()
            metric_precision = creme.metrics.MacroPrecision()
            metric_recall = creme.metrics.MacroRecall()
            metric_F1 = creme.metrics.MacroF1()

            # loop over the dataset
            for (j, (X, y)) in enumerate(data_stream):
                # make predictions on the current set of features, train the
                # model on the features, and then update our metric
                preds = model.predict_one(X)
                # model = model.fit_one(X, y)  # this line should be enabled in train, and disabled in val
                metric_confusion = metric_confusion.update(y, preds)
                metric_precision = metric_precision.update(y, preds)
                metric_recall = metric_recall.update(y, preds)
                metric_F1 = metric_F1.update(y, preds)
                if j % 100 == 0:
                    logging.info("[INFO] update {} - {}, {}, {} \n{}".format(
                        j, metric_precision, metric_recall, metric_F1, metric_confusion))

            logging.info("[INFO] final - {}, {}, {} \n{}\n".format(
                metric_precision, metric_recall, metric_F1, metric_confusion))


        if state == 'test':
            # train_csv = params['feature csv file'] + '/feature_{}'.format(file_dir[i].rsplit('\\', 1)[-1])
            train_csv = file_dir[i]
            # train_data = np.asarray(pd.read_csv(train_csv))
            # train_data = train_data.astype('float32')
            preds_label = np.zeros((array.shape[0], 2))
            preds_prob_dict = np.zeros((array.shape[0], 2))
            data_stream = creme.stream.iter_array(array[:, 1:25], array[:, -1])

            logging.info('{}/{} - {} data transferred '.format(i + 1, recursive_time, train_csv))
            logging.info('[INFO] Start prediction...')

            # loop over the dataset
            for (j, (X, y)) in enumerate(data_stream):
                # make predictions on the current set of features, train the
                # model on the features, and then update our metric
                preds = model.predict_one(X)
                # model = model.fit_one(X, y)  # this line should be enabled in train, and disabled in val
                # preds_prob = model.predict_proba_one(X)
                preds_label[j] = j, preds
                # preds_prob_dict[j] = j, preds_prob
            # print(preds_prob_dict)
            return preds_label

    if state=='train':
        # save model
        with open('model_creme.pkl', 'wb') as f:
            pickle.dump(model, f)
        # save a back up model
        with open('model_creme_{}.pkl'.format(params['prev data num'] + recursive_time), 'wb') as f:
            pickle.dump(model, f)

    # if state == 'test':
    #     return preds_label








