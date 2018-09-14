import copy
from sklearn.model_selection import train_test_split

class GeneratorDataSet:
    def __init__(self, inventory,
            data_loader=None,
            data_encoder=None,
            target_encoder=None):
        self._inventory = inventory
        self._data_loader = data_loader
        self._data_encoder = data_encoder
        self._target_encoder = target_encoder

    @property
    def inventory(self):
        return self._inventory

    def size(self):
        return len(self.inventory)

    def split(self, validation=0.2, test=0.0):
        if validation < 0.0 or 1.0 < validation:
            raise ValueError("validation ratio must be between 0.0 and 1.0: " + str(validation_ratio))
        if test < 0.0 or 1.0 < test:
            raise ValueError("validation ratio must be between 0.0 and 1.0: " + str(validation_ratio))
        if test + validation > 1.0:
            raise ValueError("test and validation size exceed 1.0: " + str(test_ratio + validation_ratio))

        test_size = int(round(self.size() * test))
        validation_size = int(round(self.size() * validation))

        learning_set, test_set = train_test_split(self.inventory, test_size=test_size)
        training_set, validation_set = train_test_split(learning_set, test_size=validation_size)

        return self._clone_with_inventory(training_set), \
                self._clone_with_inventory(validation_set), \
                self._clone_with_inventory(test_set)

    def _clone_with_inventory(self, inventory):
        if len(inventory) == 0:
            return EMPTY_GEN

        clone = copy.copy(self)
        clone._inventory = inventory

        return clone

EMPTY_GEN = GeneratorDataSet(())
