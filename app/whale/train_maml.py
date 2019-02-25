from dlcliche.image import *
sys.path.append('..') # app
sys.path.append('../..') # root
from easydict import EasyDict
from app_utils_clf import *
from whale_utils import *
from config import DATA_PATH
from torch.utils.data import DataLoader
from torch import nn
import argparse
from few_shot.core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from few_shot.maml import meta_gradient_step
from few_shot.models import FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs

# Basic training parameters
args = EasyDict()

args.n = 1 
args.k = 10 
args.q = 1

SZ = 224
RE_SZ = 256

args.inner_train_steps = 1
args.inner_val_steps = 3
args.inner_lr = 0.01
args.meta_lr = 0.001
args.meta_batch_size = 32
args.order = 1
args.epochs = 20 
args.epoch_len = 30 
args.eval_batches = 20

data_train = DATA_PATH+'/train'
data_test  = DATA_PATH+'/test'

args.param_str = f'app_maml_whale_n{args.n}_k{args.k}_q{args.q}'
args.checkpoint_monitor = 'categorical_accuracy'
args.checkpoint_period = 10
args.init_weight = None
print(f'Training {args.param_str}.')

# Data
df = pd.read_csv(DATA_PATH+'/train.csv')
df = df[df.Id != 'new_whale']
ids = df.Id.values
classes = sorted(list(set(ids)))
images = df.Image.values
all_cls2imgs = {cls:images[ids == cls] for cls in classes}

trn_images = [image for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) >= 2]
trn_labels = [_id   for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) >= 2]
val_images = [image for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) == 2]
val_labels = [_id   for image, _id in zip(images, ids) if len(all_cls2imgs[_id]) == 2]

print(f'Samples = {len(trn_images)}, {len(val_images)}')

# Model
feature_model = get_resnet18(device=device, weight_file=args.init_weight)

# Dataloader
background = WhaleImages(data_train, trn_images, trn_labels, re_size=RE_SZ, to_size=SZ)
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)
evaluation = WhaleImages(data_train, val_images, val_labels, re_size=RE_SZ, to_size=SZ, train=False)
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size),
    num_workers=8
)

# Train
train_maml(args,
                num_input_channels=3,
                device=device,
                path='.',
                n_epochs=args.epochs,
                background_taskloader=background_taskloader,
                evaluation_taskloader=evaluation_taskloader,
                fc_layer_size=12544
               )
torch.save(feature_model.state_dict(), f'{args.param_str}_epoch{args.epochs}.pth')
