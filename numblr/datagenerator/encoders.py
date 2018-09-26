import logging
logger = logging.getLogger()

import os
import io
from functools import reduce

import numpy as np
import sklearn.preprocessing as preprocessing

try:
    from urllib.parse import urljoin
    import requests as http
except ImportError as e:
    logger.warning("Could not load dependencies for HTTP support", exc_info=True)


class ResourceDataEncoder:
    def __call__(self, record):
        return self.encode(record)

    def normalize(self, inventory):
        pass

    def get_path(self, record):
        pass

    def get_size(self, record):
        pass

    def encode(self, record):
        pass


class FileDataEncoder(ResourceDataEncoder):
    def __init__(self,
            data_encoder=None,
            data_path=None,
            id_mapper=None,
            id='id',
            binary=False):
        if not callable(data_encoder):
            raise ValueError("data_encoder must be a callable" + str(type(data_encoder)))
        if not isinstance(data_path, str):
            raise ValueError("data_path must be a string" + str(type(data_path)))
        if not callable(id_mapper):
            raise ValueError("id_mapper must be a callable" + str(type(id_mapper)))
        if not isinstance(id, str):
            raise ValueError("id must be a string" + str(type(id)))
        if not isinstance(binary, bool):
            raise ValueError("binary must be a boolean" + str(type(binary)))

        self._data_encoder = data_encoder
        self._data_path = data_path
        self._id_mapper = id_mapper
        self._id = id
        self._binary = binary

    def normalize(self, inventory):
        pass

    def get_path(self, record):
        id = record[self._id]

        if self._id_mapper is None and data_path is None:
            return id
        elif self._id_mapper is None:
            return os.path.join(self._data_path, id)
        else:
            return os.path.join(self._data_path, self._id_mapper(id))

    def get_size(self, record):
        return os.path.getsize(self.get_path(record))

    def encode(self, record):
        mode = 'rb' if self._binary else 'r'

        with open(self.get_path(record), mode) as handle:
            return self._encode_data(handle)

    def _encode_data(self, data):
        """Override to customize featurization"""
        return self._data_encoder(data)


class UrlDataEncoder(ResourceDataEncoder):
    def __init__(self,
            data_encoder=None,
            base_url=None,
            id_mapper=None,
            headers=None,
            type = 'text',
            id='id'):
        self._data_encoder = data_encoder
        self._base_url = base_url
        self._headers = headers
        self._id_mapper = id_mapper
        self._id = id
        self._type = type

    def normalize(self, inventory):
        pass

    def get_path(self, record):
        id = record[self._id]

        if self._id_mapper is None and data_path is None:
            return id
        elif self._id_mapper is None:
            return urljoin(self._base_url, id)
        else:
            return urljoin(self._base_url, self._id_mapper(id))

    def get_size(self, record):
        request = http.head(self.get_path(record), headers=self._headers)
        request.raise_for_status()

        return int(request.headers.get('content-length'))

    def encode(self, record):
        request = http.get(self.get_path(record), headers=self._headers)
        request.raise_for_status()

        if self._type == 'text':
            data = request.text
        elif self._type == 'json':
            data = request.json()
        elif self._type == 'binary':
            data = request.content

        return self._encode_data(data)

    def _encode_data(self, data):
        """Override to customize featurization"""
        return self._data_encoder(data)


class IdentityEncoder:
    def fit(self, data):
        pass

    def fit_transform(self, data):
        return data

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class IntToOneHotEncoder:
    def __init__(self, sparse=False, n_values='auto', handle_unknown='ingore'):
        self._encoder = OneHotEncoder(sparse=sparse, n_values=n_values, handle_unknown=handle_unknown)

    def fit(self, data):
        self._encoder.fit(self.__reshape(data))

    def fit_transform(self, data):
        return self._encoder.fit_transform(self.__reshape(data))

    def transform(self, data):
        return self._encoder.transform(self.__reshape(data))

    def inverse_transform(self, data):
        data = np.array(data, copy=False)

        return np.argmax(data, axis=1).flatten()

    def __reshape(self, data):
        return data.reshape(len(data), 1)


class RecordTargetEncoder:
    def __init__(self, encoders=None, target='target'):
        if not isinstance(target, str):
            raise ValueError("target must be a string: " + str(type(target)))

        try:
            any(encoders)
            self._delegates = encoders
        except TypeError:
            self._delegates = (encoders, ) if encoders is not None else IdentityEncoder()

        self._target = target

    def __call__(self, record):
        return self.transform(record)

    def fit(self, records):
        target_data = self.get_target_data(records)
        self.__fit_transform_targets(target_data)

        return self

    def fit_transform(self, records):
        target_data = self.get_target_data(records)

        return self.__fit_transform_targets(target_data)

    def __fit_transform_targets(self, target_data):
        return reduce(lambda x, enc: self.__fit_transform(enc, x), self._delegates, target_data)

    def __fit_transform(self, encoder, data):
        encoder.fit(data)

        return encoder.transform(data)

    def transform(self, records):
        target_data = self.get_target_data(records)

        return reduce(lambda x, enc: enc.transform(x), self._delegates, target_data)

    def inverse_transform(self, target_data):
        target_data = np.array(target_data, copy=False)
        return reduce(lambda x, enc: enc.inverse_transform(x), self._delegates[::-1], target_data)

    def get_target_data(self, records):
        return records if self._target is None else records[self._target]


class LabelRecordEncoder(RecordTargetEncoder):
    def __init__(self, target='target'):
        super(LabelRecordEncoder, self).__init__(LabelEncoder())

    @property
    def classes_(self):
        return self._delegates[0].classes_


class OneHotRecordEncoder(RecordTargetEncoder):
    def __init__(self, target='target'):
        super(OneHotRecordEncoder, self).__init__([LabelEncoder(), IntToOneHotEncoder()])

    @property
    def classes_(self):
        return self._delegates[0].classes_


LabelEncoder = preprocessing.LabelEncoder
OneHotEncoder = preprocessing.OneHotEncoder
