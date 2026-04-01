"""Minimal train/val loop with checkpoint saving and W&B integration."""

import random

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cbct_diffusion.training.base_loss import BaseLoss
from cbct_diffusion.training.stats_tracker import StatsTracker


class PyTorchExperiment:
    """Simple training loop.

    Parameters
    ----------
    args : dict
        Hyper-parameter dictionary (logged to W&B).
    train_dataset, test_dataset : Dataset
        PyTorch datasets for training and validation.
    batch_size : int
        Mini-batch size.
    model : nn.Module
        The model to train.
    loss_fn : BaseLoss
        Loss computation object.
    checkpoint_path : str
        Where to save the best / latest checkpoint.
    experiment_name : str
        Used as W&B project/run name.
    with_wandb : bool
        Whether to initialise a W&B run.
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed.
    loss_to_track : str
        Key from ``loss_fn.stats_names`` used for best-checkpoint tracking.
    save_always : bool
        If ``True`` save after every epoch regardless of improvement.
    """

    def __init__(
        self,
        args,
        train_dataset,
        test_dataset,
        batch_size: int,
        model: nn.Module,
        loss_fn: BaseLoss,
        checkpoint_path: str,
        experiment_name: str = "",
        num_workers: int = 0,
        with_wandb: bool = False,
        seed: int = 0,
        loss_to_track: str = "loss",
        save_always: bool = False,
    ):
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.model = model
        self.seed = seed
        self.loss_to_track = loss_to_track
        self.save_always = save_always
        torch.manual_seed(seed)
        random.seed(seed)
        self.loss_fn = loss_fn
        self.checkpoint_path = checkpoint_path
        self.best_val_loss = float("inf")

        if with_wandb and experiment_name:
            wandb.init(
                project=experiment_name,
                name=experiment_name + str(seed),
                config=args,
            )
            wandb.watch(model)
        elif not experiment_name:
            experiment_name = f"exp_{random.randint(0, 100_000)}"
        self.experiment_name = experiment_name

    def train(self, epochs, optimizer, milestones, gamma, scheduler=None):
        """Run the training loop for *epochs* epochs."""
        train_tracker = StatsTracker("Train", self.loss_fn.stats_names)
        test_tracker = StatsTracker("Test", self.loss_fn.stats_names)

        if scheduler is None:
            scheduler = MultiStepLR(
                optimizer,
                milestones=[x * len(self.train_loader.dataset) for x in milestones],
                gamma=gamma,
            )

        for epoch in range(epochs):
            self.model.train()
            iterator = tqdm(self.train_loader)
            for instance in iterator:
                optimizer.zero_grad()
                loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                loss.backward()
                optimizer.step()
                scheduler.step()
                bs = len(instance[0]) if isinstance(instance, (tuple, list)) else len(instance)
                train_tracker.add(loss_dict, bs)
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            train_tracker.log_stats_and_reset()

            self.model.eval()
            with torch.no_grad():
                for instance in tqdm(self.test_loader):
                    loss, loss_dict = self.loss_fn.compute_loss(instance, self.model)
                    bs = len(instance[0]) if isinstance(instance, (tuple, list)) else len(instance)
                    test_tracker.add(loss_dict, bs)
                self.loss_fn.log_epoch_summary(instance, self.model, epoch)

                if self.save_always or test_tracker.get_mean(self.loss_to_track) < self.best_val_loss:
                    self.best_val_loss = test_tracker.get_mean(self.loss_to_track)
                    print("Saving model to", self.checkpoint_path)
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        self.checkpoint_path,
                    )
                    if wandb.run:
                        wandb.save(self.checkpoint_path)
                test_tracker.log_stats_and_reset()
