import logging
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from numblr.datagenerator.dataset import GeneratorDataSet
from numblr.datagenerator.encoders import FileDataEncoder, UrlDataEncoder, RecordTargetEncoder


logger = logging.getLogger()


inventory_from_csv = pd.read_csv
inventory_from_records = pd.DataFrame.from_records
inventory_from_dict = pd.DataFrame.from_dict
inventory_from_items = pd.DataFrame.from_items

def enrich_inventory(inventory, resource_encoder, id='id', include_meta={'size': 'size'}):
    try:
        if 'size' in include_meta.keys():
            inventory[include_meta['size']] = inventory.apply(resource_encoder.get_size, axis=1)
        if 'path' in include_meta.keys():
            inventory[include_meta['path']] = inventory.apply(resource_encoder.get_path, axis=1)
    except:
        logger.warning("Failed to enrich inventory with metadata")

    inventory.set_index(id, drop=False, inplace=True, verify_integrity=True)

    return inventory


def generator_for_files(inventory_path, data_path, data_encoder, target_encoder,
        id_mapper=None, id='id', target='target', binary=False):
    file_data_encoder = FileDataEncoder(data_encoder, data_path,
            id=id, id_mapper=id_mapper, binary=binary)
    record_target_encoder = RecordTargetEncoder(target_encoder, target)
    inventory = enrich_inventory(pd.read_csv(inventory_path), file_data_encoder, id)

    data_set = GeneratorDataSet(inventory, file_data_encoder, record_target_encoder)
    data_set.fit_encoders()

    return data_set


def generator_for_urls(inventory_path, base_url,
        data_encoder, target_encoders,
        id='id', target='target'):
    url_data_encoder = UrlDataEncoder(data_encoder, data_path)
    record_target_encoder = RecordTargetEncoder(target_encoders)
    inventory = enrich_inventory(pd.read_csv(inventory_path), url_data_encoder, id)

    data_set = GeneratorDataSet(inventory, file_data_encoder, record_target_encoder)
    data_set.fit_encoders()

    return data_set
