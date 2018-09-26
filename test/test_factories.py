import unittest
from pprint import pprint

from numblr.datagenerator.encoders import LabelEncoder, IntToOneHotEncoder
from numblr.datagenerator.factories import generator_for_files


class TestGeneratorDataSet(unittest.TestCase):
    def setUp(self):
        def data_encoder(file):
            return int(file.readline().split('_')[1])
        def id_mapper(id):
            return id + '.txt'

        self.data_encoder = data_encoder
        self.id_mapper = id_mapper
        self.target_encoder = [LabelEncoder(), IntToOneHotEncoder()]

    def test_inventory(self):
        generator_data_set = generator_for_files('test/resources/inventory.csv', 'test/resources',
                self.data_encoder, self.target_encoder, self.id_mapper)

        generator = generator_data_set.batches(batch_size=4, epochs=1)

        first_batch = next(generator)
        self.assertTrue(isinstance(first_batch, tuple))
        self.assertEqual(len(first_batch), 2)

        first_data, first_target = first_batch
        self.assertEqual(first_data.shape, (4,))
        self.assertEqual(first_target.shape, (4,3))

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 1)

        shapes = { batch[0].shape for batch in batches }
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes.pop(), (4,))

        shapes = { batch[1].shape for batch in batches }
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes.pop(), (4,3))
