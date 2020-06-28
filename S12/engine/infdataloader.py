class InfiniteDataLoader:
    """Create infinite loop in a data loader.
    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader object.
        auto_reset (bool, optional): Create an infinite loop data loader.
            (default: True)
    """

    def __init__(self, data_loader, auto_reset=True):
        self.data_loader = data_loader
        self.auto_reset = auto_reset
        self._iterator = iter(data_loader)

    def __next__(self):
        # Get a new set of inputs and labels
        try:
            data, target = next(self._iterator)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            data, target = next(self._iterator)

        return data, target

    def get_batch(self):
        return next(self)