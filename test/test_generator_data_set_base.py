import unittest
from pprint import pprint

from numblr.datagenerator.datagenerator import GeneratorDataSet


class TestGeneratorDataSet(unittest.TestCase):

    def setUp(self, size=10, targets=3):
        inventory = [ { 'id': 'id_{}'.format(i), 'target': 'cat_{}'.format(i % targets) }
                for i in range(size) ]

        def data_loader(record):
            return int(record['id'].split('_')[-1]) % targets

        def data_encoder(data):
            encoded = [0] * targets
            encoded[data] = 1

            return tuple(encoded)

        def target_encoder(record):
            return int(record['target'].split('_')[-1])

        self.data_loader = data_loader
        self.data_encoder = data_encoder
        self.target_encoder = target_encoder
        self.data_set = GeneratorDataSet(inventory,
                data_loader,
                data_encoder,
                target_encoder)

    def test_inventory(self):
        self.assertEqual(len(self.data_set.inventory), 10)
        self.assertEqual(self.data_set.inventory[0]['id'], 'id_0')
        self.assertEqual(self.data_set.inventory[0]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory[1]['target'], 'cat_1')
        self.assertEqual(self.data_set.inventory[2]['target'], 'cat_2')
        self.assertEqual(self.data_set.inventory[3]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory[9]['id'], 'id_9')
        self.assertEqual(self.data_set.inventory[9]['target'], 'cat_0')

        record = self.data_set.inventory[0]
        self.assertEqual(self.data_loader(record), 0)
        self.assertEqual(self.target_encoder(record), 0)
        self.assertEqual(self.data_encoder(0), (1, 0, 0))

        record = self.data_set.inventory[1]
        self.assertEqual(self.data_loader(record), 1)
        self.assertEqual(self.target_encoder(record), 1)
        self.assertEqual(self.data_encoder(1), (0, 1, 0))

        record = self.data_set.inventory[2]
        self.assertEqual(self.data_loader(record), 2)
        self.assertEqual(self.target_encoder(record), 2)
        self.assertEqual(self.data_encoder(2), (0, 0, 1))

        record = self.data_set.inventory[3]
        self.assertEqual(self.data_loader(record), 0)
        self.assertEqual(self.target_encoder(record), 0)
        self.assertEqual(self.data_encoder(0), (1, 0, 0))

    def test_size(self):
        self.assertEqual(self.data_set.size(), 10)

    @unittest.skip
    def test_shape(self):
        pass

    def test_split(self):
        training, validation, test = self.data_set \
                .split(validation = 0.2, test = 0.3)

        self.assertEqual(training.size() + validation.size() + test.size(), self.data_set.size())

        self.assertEqual(training.size(), 5)
        self.assertEqual(validation.size(), 2)
        self.assertEqual(test.size(), 3)
