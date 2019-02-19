from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn

from few_shot.models import get_few_shot_encoder, Flatten
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from torch.utils.data import DataLoader
from torch import nn
import argparse

from few_shot.models import FewShotClassifier
from few_shot.maml import meta_gradient_step

from dlcliche.utils import *
from config import PATH
assert torch.cuda.is_available()
device = torch.device('cuda')


def show_normalized_image(img, ax=None, mono=False):
    if mono:
        img.numpy()[..., np.newaxis]
    np_img = img.numpy().transpose(1, 2, 0)
    lifted = np_img - np.min(np_img)
    ranged = lifted / np.max(lifted)
    show_np_image(ranged, ax=ax)


class MonoTo3ChLayer(nn.Module):
    def __init__(self):
        super(MonoTo3ChLayer, self).__init__()
    def forward(self, x):
        x.unsqueeze_(1)
        return x.repeat(1, 3, 1, 1)


def _get_model(weight_file, device, model_fn, mono):
    base_model = model_fn(pretrained=True)
    feature_model = nn.Sequential(*list(base_model.children())[:-1],
                                  nn.AdaptiveAvgPool2d(1),
                                  Flatten())
    # Load initial weights
    if weight_file is not None:
        feature_model.load_state_dict(torch.load(weight_file))
    # Add mono image input layer at the bottom of feature model
    if mono:
        feature_model = nn.Sequential(MonoTo3ChLayer(), feature_model)
    if device is not None:
        feature_model.to(device)

    feature_model.eval()
    return feature_model


def get_resnet101(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet101, mono=mono)


def get_resnet50(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet50, mono=mono)


def get_resnet34(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet34, mono=mono)


def get_resnet18(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.resnet18, mono=mono)


def get_densenet121(weight_file=None, device=None, mono=False):
    return _get_model(weight_file, device, models.densenet121, mono=mono)


def train_proto_net(args, model, device, n_epochs,
                    background_taskloader,
                    evaluation_taskloader,
                    path='.',
                    lr=3e-3,
                    drop_lr_every=100,
                    evaluation_episodes=100,
                    episodes_per_epoch=100,
                   ):
    # Prepare model
    model.to(device, dtype=torch.float)
    model.train(True)

    # Prepare training etc.
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.NLLLoss().cuda()
    ensure_folder(path + '/models')
    ensure_folder(path + '/logs')

    def lr_schedule(epoch, lr):
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr

    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=path + '/models/'+args.param_str+'_e{epoch:02d}.pth',
            monitor=args.checkpoint_monitor or f'val_{args.n_test}-shot_{args.k_test}-way_acc',
            period=args.checkpoint_period or 100,
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(path + f'/logs/{args.param_str}.csv'),
    ]

    fit(
        model,
        optimizer,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        epoch_metrics=[f'val_{args.n_test}-shot_{args.k_test}-way_acc'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
    )
    
    
import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union

from few_shot.core import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def train_maml(args, device, n_epochs,
                    background_taskloader,
                    evaluation_taskloader,
                    num_input_channels,
                    path='.',
                    fc_layer_size=1600
                   ):
    # Prepare model
    meta_model = FewShotClassifier(num_input_channels, args.k, fc_layer_size).to(device, dtype=torch.double)
    meta_model.train(True)

    # Prepare training etc.
    meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    loss_fn = nn.CrossEntropyLoss().to(device)
    ensure_folder(path + '/models')
    ensure_folder(path + '/logs')
    
    def prepare_meta_batch(n, k, q, meta_batch_size):
        def prepare_meta_batch_(batch):
            x, y = batch
            # Reshape to `meta_batch_size` number of tasks. Each task contains
            # n*k support samples to train the fast model on and q*k query samples to
            # evaluate the fast model on and generate meta-gradients
            x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
            # Move to device
            x = x.double().to(device)
            # Create label
            y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
            return x, y
        return prepare_meta_batch_

    callbacks = [
        EvaluateFewShot(
            eval_fn=meta_gradient_step,
            num_tasks=args.eval_batches,
            n_shot=args.n,
            k_way=args.k,
            q_queries=args.q,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
            # MAML kwargs
            inner_train_steps=args.inner_val_steps,
            inner_lr=args.inner_lr,
            device=device,
            order=args.order,
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/{args.param_str}.pth',
            monitor=f'val_{args.n}-shot_{args.k}-way_acc'
        ),
        ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
        CSVLogger(PATH + f'/logs/{args.param_str}.csv'),
    ]


    fit(
        meta_model,
        meta_optimiser,
        loss_fn,
        epochs=args.epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=meta_gradient_step,
        fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                             'train': True,
                             'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                             'inner_lr': args.inner_lr},
    )

