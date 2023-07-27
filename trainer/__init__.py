import os
from pathlib import Path
from random import randint
import datetime

import torch
import wandb
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins import DDPPlugin    # for pytorch_lightning < 2.0
from pytorch_lightning.strategies import DDPStrategy    # for pytorch_lightning >= 2.0

from util.filesystem_logger import FilesystemLogger


def generate_experiment_name(name, config):
    if config.resume is not None:
        experiment = Path(config.resume).parents[1].name
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{datetime.datetime.now().strftime('%d%m%H%M')}_{name}_{config.experiment}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


def create_trainer(name, config):
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = generate_experiment_name(name, config)
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)

    # noinspection PyUnusedLocal
    logger = WandbLogger(project=f'{name}{config.suffix}',
                         name=config.experiment,
                         id=config.experiment,
                         settings=wandb.Settings(start_method='thread'))
    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment / "checkpoints"),
                                          filename='_{epoch}',
                                          save_top_k=-1,
                                          verbose=False,
                                          every_n_epochs=config.save_epoch)
    gpu_count = torch.cuda.device_count()
    if config.batch_gpu is None:
        config.batch_gpu = config.batch_size
        # config.batch_size *= gpu_count
        config.val_check_interval *= gpu_count
    print(f"batch_size = {config.batch_size} / {config.batch_gpu}")
    assert config.batch_size >= config.batch_gpu and config.batch_size % config.batch_gpu == 0

    if gpu_count > 1:
        trainer = Trainer(
                          # gpus=-1,    # for older pytorch_lightning versions
                          accelerator='ddp',
                          plugins=DDPStrategy(find_unused_parameters=True),
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          #   resume_from_checkpoint=config.resume,    # moved to Trainer.fit() in newer pytorch_lightning versions
                          logger=logger,
                          benchmark=True)
    else:
        trainer = Trainer(
                          # gpus=[0],    # for older pytorch_lightning versions
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          #   resume_from_checkpoint=config.resume,    # moved to Trainer.fit() in newer pytorch_lightning versions
                          logger=logger,
                          benchmark=True)
    return trainer
