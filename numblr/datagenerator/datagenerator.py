class GeneratorDataSet:
    def __init__(self, inventory,
            data_loader,
            data_encoder,
            target_encoder):
        self._inventory = inventory
        self._data_loader = data_loader
        self._data_encoder = data_encoder
        self._target_encoder = target_encoder

    @property
    def inventory(self):
        return self._inventory
