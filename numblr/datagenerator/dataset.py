import logging
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger()


class GeneratorDataSet:
    def __init__(self, inventory, data_encoder=None, target_encoder=None):
        if not isinstance(inventory, pd.DataFrame):
            raise ValueError("inventory must be a pandas.DataFrame")

        self._inventory = inventory
        self._data_encoder = data_encoder
        self._target_encoder = target_encoder

    @property
    def inventory(self):
        return self._inventory

    @property
    def size(self):
        return len(self.inventory)

    @property
    def data_encoder(self):
        return self._data_encoder

    @property
    def target_encoder(self):
        return self._target_encoder

    def fit_encoders(self):
        self.target_encoder.fit(self.inventory)
        try:
            [ encoder.fit(self.inventory) for encoder in self.data_encoder ]
        except TypeError:
            self.data_encoder.fit(self.inventory)

    def sort(self, columns=['size'], ascending=True, na_position='last'):
        self._inventory.sort_values(by=columns, ascending=ascending, na_position=na_position, inplace=True)

    def shuffle(self, random_state=None):
        self._inventory = self._inventory \
                .sample(frac=1, random_state=random_state) \
                .reset_index(drop=True)

    def split(self, validation=0.2, test=0.0):
        if validation < 0.0 or 1.0 < validation:
            raise ValueError("validation ratio must be between 0.0 and 1.0: " + str(validation))
        if test < 0.0 or 1.0 < test:
            raise ValueError("validation ratio must be between 0.0 and 1.0: " + str(validation))
        if test + validation > 1.0:
            raise ValueError("validation plus test size exceed 1.0: " + str(test_ratio + validation))

        test_size = int(round(self.size * test))
        validation_size = int(round(self.size * validation))

        learning_set, test_set = train_test_split(self.inventory, test_size=test_size)
        training_set, validation_set = train_test_split(learning_set, test_size=validation_size)

        return self._clone_with_inventory(training_set), \
                self._clone_with_inventory(validation_set), \
                self._clone_with_inventory(test_set)

    def data(self):
        data = next(self.data_batches(batch_size=self.size, epochs=1))

        return np.array([x for x in data], copy=False)

    def targets(self):
        targets = next(self.target_batches(batch_size=self.size, epochs=1))

        return np.array([x for x in targets], copy=False)

    def batches(self, batch_size=10, epochs=None, truncate=True):
        self.__validate_batch_size(batch_size, truncate)

        return ( (self._get_batch_data(batch), self._get_batch_targets(batch))
                for batch in self.__inventory_batches(batch_size, epochs, truncate) )

    def data_batches(self, batch_size=10, epochs=None, truncate=True):
        self.__validate_batch_size(batch_size, truncate)

        return ( self._get_batch_data(batch)
                for batch in self.__inventory_batches(batch_size, epochs, truncate) )

    def _get_batch_data(self, batch):
        """Override to customize batch data loading and featurization."""
        try:
            encoders = [ encoder for encoder in self._data_encoder ]
        except:
            encoders = (self._data_encoder,)

        try:
            data_batches = [ encoder.transform_batch(rec for _, rec in batch.iterrows())
                    for encoder in encoders ]
        except AttributeError:
            data_batches = [
                    [ self._get_data(record, encoder) for _, record in batch.iterrows() ]
                    for encoder in encoders ]

        try:
            batches = [ np.array(encoder.finalize_batch(batch))
                    for encoder, batch in zip(encoders, data_batches)]
        except AttributeError:
            batches = [ np.array(batch) for batch in data_batches ]

        return batches if len(batches) > 1 else batches[0]

    def _get_data(self, record, encoder):
        """Override to customize data loading and featurization."""
        try:
            return encoder.transform(record)
        except AttributeError:
            return encoder(record)

    def target_batches(self, batch_size=10, epochs=None, truncate=True):
        """Override to customize batch target creation."""
        self.__validate_batch_size(batch_size, truncate)

        return ( self._get_batch_targets(batch)
                for batch in self.__inventory_batches(batch_size, epochs, truncate) )

    def _get_batch_targets(self, batch):
        """Override to customize target creation."""
        try:
            return np.array(self._target_encoder.transform(batch))
        except AttributeError:
            return np.array(self._target_encoder(batch))

    def __inventory_batches(self, batch_size, epochs, truncate):
        if batch_size < 1:
            batch_size = self.size

        epoch = 0
        while epochs is None or epoch < epochs:
            epoch += 1
            yield from ( self._inventory.iloc[i:i + batch_size]
                    for i in range(0, self.size, batch_size)
                    if i + batch_size <= self.size or not truncate )

        logger.info("Fetched " + str(epochs) + "batches")

    def __validate_batch_size(self, batch_size, truncate):
        if truncate and batch_size > self.size:
            raise ValueError("batch_size larger than data set size: "
                    + str(batch_size) + " > " + str(self.size)
                    + ", use a valid batch_size or the 'truncate=False' option")

    def _clone_with_inventory(self, inventory):
        """Override to control the creation of new instances with modified inventory"""
        clone = copy.copy(self)
        clone._inventory = inventory

        return clone

    def __copy__(self):
        """Override to control cloning of the instance"""
        return GeneratorDataSet(self._inventory, self._data_encoder, self._target_encoder)
