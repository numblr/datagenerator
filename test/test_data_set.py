import unittest
from pprint import pprint

from numblr.datagenerator.dataset import GeneratorDataSet


class TestGeneratorDataSet(unittest.TestCase):
    def setUp(self, size=10, targets=3):
        inventory = [ { 'id': 'id_{}'.format(i), 'target': 'cat_{}'.format(i % targets) }
                for i in range(size) ]

        def data_loader(record):
            return int(record['id'].split('_')[-1]) % targets

        def data_encoder(record):
            data = data_loader(record)

            encoded = [0] * targets
            encoded[data] = 1

            return tuple(encoded)

        def target_encoder(record):
            return int(record['target'].split('_')[-1])

        self.data_loader = data_loader
        self.data_encoder = data_encoder
        self.target_encoder = target_encoder
        self.data_set = GeneratorDataSet(inventory,
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
        self.assertEqual(self.data_encoder(record), (1, 0, 0))

        record = self.data_set.inventory[1]
        self.assertEqual(self.data_loader(record), 1)
        self.assertEqual(self.target_encoder(record), 1)
        self.assertEqual(self.data_encoder(record), (0, 1, 0))

        record = self.data_set.inventory[2]
        self.assertEqual(self.data_loader(record), 2)
        self.assertEqual(self.target_encoder(record), 2)
        self.assertEqual(self.data_encoder(record), (0, 0, 1))

        record = self.data_set.inventory[3]
        self.assertEqual(self.data_loader(record), 0)
        self.assertEqual(self.target_encoder(record), 0)
        self.assertEqual(self.data_encoder(record), (1, 0, 0))

    def test_size(self):
        self.assertEqual(self.data_set.size, 10)

    @unittest.skip
    def test_shape(self):
        pass

    def test_split(self):
        training, validation, test = self.data_set \
                .split(validation = 0.2, test = 0.3)

        self.assertEqual(training.size + validation.size + test.size, self.data_set.size)

        self.assertEqual(training.size, 5)
        self.assertEqual(validation.size, 2)
        self.assertEqual(test.size, 3)

    def test_data_batches(self):
        generator = self.data_set.data_batches(batch_size=4, epochs=1)

        first_batch = next(generator)
        self.assertEqual(first_batch.shape, (4, 3))

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 1)

        shapes = { batch.shape for batch in batches }
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes.pop(), (4, 3))

    def test_target_batches(self):
        generator = self.data_set.target_batches(batch_size=4, epochs=1)

        first_batch = next(generator)
        self.assertEqual(first_batch.shape, (4,))

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 1)

        shapes = { batch.shape for batch in batches }
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes.pop(), (4,))

    def test_batches(self):
        generator = self.data_set.batches(batch_size=4, epochs=1)

        first_batch = next(generator)
        self.assertTrue(isinstance(first_batch, tuple))
        self.assertEqual(len(first_batch), 2)

        first_data, first_target = first_batch
        self.assertEqual(first_data.shape, (4,3))
        self.assertEqual(first_target.shape, (4,))

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 1)

        shapes = { batch[0].shape for batch in batches }
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes.pop(), (4, 3))

    def test_batches_multiple_epochs(self):
        generator = self.data_set.batches(batch_size=5, epochs=5)

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 10)

        shapes = { batch[0].shape for batch in batches }
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes.pop(), (5, 3))

    def test_batch_data_epochs_no_truncate(self):
        generator = self.data_set.batches(batch_size=4, epochs=5, truncate=False)

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 15)

        shapes = sorted({ batch[0].shape for batch in batches })
        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes.pop(), (4, 3))
        self.assertEqual(shapes.pop(), (2, 3))

    def test_batch_data_multiple_encoders(self):
        inventory = [ { 'id': 'id_{}'.format(i), 'target': 'cat_{}'.format(i % 3) }
                for i in range(10) ]

        data_encoder = [
            lambda record: (1,) + (int(record['target'][-1]),),
            lambda record: (2,) + (int(record['target'][-1]),),
            lambda record: (3,) + (int(record['target'][-1]),)
        ]

        def target_encoder(record):
            return int(record['target'].split('_')[-1])

        data_set = GeneratorDataSet(inventory,
                data_encoder,
                target_encoder)

        generator = data_set.data_batches(batch_size=3, epochs=1)

        for batch in generator:
            self.assertTrue(isinstance(batch, list))
            self.assertEqual(len(batch), 3)

            self.assertSequenceEqual(list(batch[0][0]), [1, 0])
            self.assertSequenceEqual(list(batch[0][1]), [1, 1])
            self.assertSequenceEqual(list(batch[0][2]), [1, 2])

            self.assertSequenceEqual(list(batch[1][0]), [2, 0])
            self.assertSequenceEqual(list(batch[1][1]), [2, 1])
            self.assertSequenceEqual(list(batch[1][2]), [2, 2])

            self.assertSequenceEqual(list(batch[2][0]), [3, 0])
            self.assertSequenceEqual(list(batch[2][1]), [3, 1])
            self.assertSequenceEqual(list(batch[2][2]), [3, 2])
