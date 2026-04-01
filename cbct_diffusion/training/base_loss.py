"""Abstract loss interface for the training loop."""


class BaseLoss:
    """Base class that every training loss must subclass.

    Parameters
    ----------
    stats_names : list[str]
        Names of the scalar statistics returned by ``compute_loss``.
    """

    def __init__(self, stats_names: list):
        self.stats_names = stats_names

    def log_epoch_summary(self, instance, model, epoch):
        """Called once at the end of each validation epoch (optional hook)."""
        pass

    def compute_loss(self, instance, model):
        """Compute the training loss.

        Returns
        -------
        loss : Tensor
            Scalar loss for back-propagation.
        stats : dict[str, float]
            Named scalars matching ``self.stats_names``.
        """
        raise NotImplementedError
