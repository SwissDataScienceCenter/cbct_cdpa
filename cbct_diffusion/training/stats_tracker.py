"""Epoch-level metric tracker with optional W&B logging."""

import torch
import wandb


class StatsTracker:
    """Tracks running totals of named metrics and logs per-epoch means.

    Parameters
    ----------
    name : str
        Label prefix used when printing / logging (e.g. ``"Train"``).
    stat_names : list[str]
        Names of the tracked scalar statistics.
    """

    def __init__(self, name: str, stat_names: list):
        self.name = name
        self.stats = {
            n: {"total": torch.tensor(0.0), "count": 0} for n in stat_names
        }
        self.current_epoch = 0

    def add(self, stat_value_dict: dict, batch_size: int) -> None:
        """Accumulate a batch of statistics."""
        for stat_name, value in stat_value_dict.items():
            if stat_name not in self.stats:
                raise ValueError(f"Stat name {stat_name} not found!")
            try:
                value = value.item()
            except AttributeError:
                pass
            self.stats[stat_name]["total"] += value * batch_size
            self.stats[stat_name]["count"] += batch_size

    def get_mean(self, stat_name: str) -> float:
        """Return the running mean for *stat_name*."""
        if stat_name not in self.stats:
            raise ValueError(f"Stat name {stat_name} not found!")
        return self.stats[stat_name]["total"] / self.stats[stat_name]["count"]

    def log_stats_and_reset(self) -> None:
        """Print and (optionally) log to W&B, then reset counters."""
        epoch = self.current_epoch
        for stat_name in self.stats:
            mean_stat = self.get_mean(stat_name)
            print(f"[{self.name} Epoch {epoch}] ({stat_name}) Mean: {mean_stat:.2f}")
            if wandb.run:
                wandb.log({f"{self.name}_{stat_name}_mean": mean_stat}, step=epoch)
        self.current_epoch += 1
        for key in self.stats:
            self.stats[key]["total"] = torch.tensor(0.0)
            self.stats[key]["count"] = 0
