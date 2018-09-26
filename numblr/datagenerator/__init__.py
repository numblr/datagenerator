__version__ = '0.0.1'
__copyright__ = "Copyright 2018, Thomas Baier"

__all__ = ['dataset', 'encoders']

from numblr.datagenerator.factories import (generator_for_files, generator_for_urls,
        inventory_from_csv, inventory_from_records, inventory_from_dict, inventory_from_items)
from numblr.datagenerator.encoders import (LabelEncoder, IntToOneHotEncoder,
        FileDataEncoder, UrlDataEncoder, IdentityEncoder)
