from dlcliche.image import *
sys.path.append('..') # app
sys.path.append('../..') # root
from easydict import EasyDict
from app_utils_clf import *
from whale_utils import *
from config import DATA_PATH

weight = 'app_whale_n1_k50_q1_sz224_dense121_epoch200'
calculate_results(weight=weight, SZ=320, get_model_fn=get_densenet121, device=device,
                          train_csv=DATA_PATH+'/train.csv', data_train=DATA_PATH+'/train', data_test=DATA_PATH+'/test')

test_dists = np.load(f'test_dists_{weight}.npy')
np_describe(test_dists)
prepare_submission(weight, test_dists, data_test=DATA_PATH+'/test',
                           classes=get_classes(data=DATA_PATH), new_whale_thresh=-1.85)
