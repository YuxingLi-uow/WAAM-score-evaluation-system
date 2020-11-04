from sklearn.linear_model import SGDClassifier
from utils import *
import os
import argparse
import matplotlib.pyplot as plt


# comment: argument parse setup
parser = argparse.ArgumentParser(description='Run incremental learning for current and voltage model')
parser.add_argument('state', type=str, default='val', help='set state of the model: train or val or test')
parser.add_argument('--file', nargs='+', default='', help='directory of file need to be val or test')
parser.add_argument('--log', type=str, default='learning_log.log', help='log file name, format: xxx.log')
args = parser.parse_args()
# # pass parameters here instead of command line
# args = parser.parse_args(['test', '--file', r'D:\OneDrive - University of Wollongong\STEVENPAN\Vol_Data\NAB1\nab_layer6_3.csv'])

state = args.state
log = args.log
file_dir = [' '.join(args.file)]

# comment: logger
logger_setup(log)
logging.debug('state -- {}'.format(state))

# comment: find number of files need to be processed
params = load_params('config.yaml')
defectClass = load_params('defect class config.yaml')

# comment: After label, feed into incremental learning model
try:
    with open('model_creme.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    if state == 'train':
        model = creme.preprocessing.StandardScaler()
        model |= creme.compat.convert_sklearn_to_creme(
            estimator=SGDClassifier(learning_rate='adaptive', eta0=0.01 ),  # should set some params in SGD
            classes=list(defectClass.keys())
        )
    else:
        # print('No pre-trained model, cannot validate data')
        logging.debug('No pre-trained model, cannot validate data')

if state == 'train':
    CsvObserve = CsvFileObserve('config.yaml', state)
    flag, recursive_time = CsvObserve.Flag()
    file_dir = CsvObserve.cur_file
    file_dir.sort(reverse=True, key=os.path.getmtime)  # sort file from new to old

    # print('\nFile processing:')
    logging.debug('File processing:')
    if recursive_time != 0:
        for i in range(recursive_time):
            # print(file_dir[i], '\n')
            logging.debug('{}\n'.format(file_dir[i]))
    else:
        # print('None')
        logging.debug('None')

    # comment: set state ='train' if you want update model
    for i in range(params['epoch']):
        logging.debug('epoch {}/{}'.format(i, params['epoch']))
        run_incremental_frame(model, recursive_time, file_dir, params, state=state)


# comment: set state='val' if you want use model to validate data and no updates applied in model
if state == 'val':
    run_incremental_frame(model, 1, file_dir, params, state=state)


# comment: set state='test' if you want use model to predict data and no updates applied in model
if state == 'test':
    final_feature, _, _, _ = feature_extraction(file_dir[0])
    preds_label = run_incremental_frame(model, 1, file_dir, params, state=state, array=final_feature)
    # print(final_feature.shape, preds_label.shape)
    predict_time = evaluate_prediction(final_feature, preds_label)
    optim_time = optim_prediction(predict_time, defectClass, pool_shift_threshold=0.2)  # optimize defect prediction results
    for defect, t in optim_time.items():
        # logging.debug('')
        logging.debug('{}'.format('\n'))
        logging.debug('{} time stamp:'.format(defect))
        for stamp in t:
            # logging.debug('\t', '%.4f' % stamp, 's')
            logging.debug('{} {}s'.format('\t', '%.4f' % stamp))
    # plt.plot(final_feature[:, 0], final_feature[:, -2])
    # plt.show()





