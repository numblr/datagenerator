import unittest
import numpy as np
from numpy.testing import assert_array_equal

from pandas import DataFrame

from numblr.datagenerator.dataset import GeneratorDataSet


class TestGeneratorDataSet(unittest.TestCase):
    def setUp(self, size=10, targets=3):
        inventory = DataFrame.from_records([
                { 'id': 'id_{}'.format(i), 'target': 'cat_{}'.format(i % targets) }
                for i in range(size) ])

        def data_encoder(record):
            """Encode to vectors of length #targets"""
            postition = int(record['id'].split('_')[-1]) % targets

            encoded = [0] * targets
            encoded[postition] = 1

            return tuple(encoded)

        def target_encoder(records):
            return [ int(record['target'].split('_')[-1]) for _, record in records.iterrows() ]

        self.inventory = inventory
        self.data_encoder = data_encoder
        self.target_encoder = target_encoder
        self.data_set = GeneratorDataSet(inventory,
                data_encoder,
                target_encoder)

    def test_inventory(self):
        self.assertEqual(len(self.data_set.inventory), 10)
        self.assertEqual(self.data_set.inventory.iloc[0]['id'], 'id_0')
        self.assertEqual(self.data_set.inventory.iloc[0]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory.iloc[1]['target'], 'cat_1')
        self.assertEqual(self.data_set.inventory.iloc[2]['target'], 'cat_2')
        self.assertEqual(self.data_set.inventory.iloc[3]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory.iloc[9]['id'], 'id_9')
        self.assertEqual(self.data_set.inventory.iloc[9]['target'], 'cat_0')

        record = self.data_set.inventory.iloc[0]
        self.assertEqual(self.data_encoder(record), (1, 0, 0))

        record = self.data_set.inventory.iloc[1]
        self.assertEqual(self.data_encoder(record), (0, 1, 0))

        record = self.data_set.inventory.iloc[2]
        self.assertEqual(self.data_encoder(record), (0, 0, 1))

        record = self.data_set.inventory.iloc[3]
        self.assertEqual(self.data_encoder(record), (1, 0, 0))

        self.assertSequenceEqual(self.target_encoder(self.data_set.inventory),
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

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

    def test_data(self):
        data = self.data_set.data()

        self.assertEqual(data.shape, (10, 3))
        expected = np.array([
            [1 ,0, 0],
            [0 ,1, 0],
            [0 ,0, 1],
            [1 ,0, 0],
            [0 ,1, 0],
            [0 ,0, 1],
            [1 ,0, 0],
            [0 ,1, 0],
            [0 ,0, 1],
            [1 ,0, 0]])
        assert_array_equal(data, expected)

    def test_targets(self):
        targets = self.data_set.targets()

        self.assertEqual(targets.shape, (10,))
        self.assertSequenceEqual(list(targets), [0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    def test_data_batches(self):
        generator = self.data_set.data_batches(batch_size=4, epochs=1)

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape, (4, 3))
        self.assertEqual(batches[1].shape, (4, 3))

        self.assertSequenceEqual(list(batches[0][0]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[0][1]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[0][2]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[0][3]), [1, 0, 0])

        self.assertSequenceEqual(list(batches[1][0]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[1][1]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[1][2]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[1][3]), [0, 1, 0])

    def test_data_batches_encoder_class(self):
        data_encoder = self.data_encoder

        class RecordDataEncoder:
            def transform(self, record):
                return data_encoder(record)

        data_set = GeneratorDataSet(self.inventory,
                RecordDataEncoder(),
                self.target_encoder)
        generator = data_set.data_batches(batch_size=4, epochs=1)

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape, (4, 3))
        self.assertEqual(batches[1].shape, (4, 3))

        self.assertSequenceEqual(list(batches[0][0]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[0][1]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[0][2]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[0][3]), [1, 0, 0])

        self.assertSequenceEqual(list(batches[1][0]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[1][1]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[1][2]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[1][3]), [0, 1, 0])

    def test_data_batches_batch_encoder(self):
        data_encoder = self.data_encoder

        class BatchDataEncoder:
            def transform_batch(self, records):
                return [ data_encoder(record) for record in records ]

        data_set = GeneratorDataSet(self.inventory,
                BatchDataEncoder(),
                self.target_encoder)
        generator = data_set.data_batches(batch_size=4, epochs=1)

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape, (4, 3))
        self.assertEqual(batches[1].shape, (4, 3))

        self.assertSequenceEqual(list(batches[0][0]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[0][1]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[0][2]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[0][3]), [1, 0, 0])

        self.assertSequenceEqual(list(batches[1][0]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[1][1]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[1][2]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[1][3]), [0, 1, 0])

    def test_data_batches_encoder_class_with_finalizer(self):
        data_encoder = self.data_encoder

        class TruncateAndPadDataEncoder:
            def transform(self, record):
                """Encode and truncate vector"""
                encoded = data_encoder(record)
                return encoded[:encoded.index(1)+1]

            def finalize_batch(self, records):
                """Pad encoded vector"""
                return [ rec + (0,) * (3 - len(rec)) for rec in records ]

        data_set = GeneratorDataSet(self.inventory,
                TruncateAndPadDataEncoder(),
                self.target_encoder)
        generator = data_set.data_batches(batch_size=4, epochs=1)

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].shape, (4, 3))
        self.assertEqual(batches[1].shape, (4, 3))

        self.assertSequenceEqual(list(batches[0][0]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[0][1]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[0][2]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[0][3]), [1, 0, 0])

        self.assertSequenceEqual(list(batches[1][0]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[1][1]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[1][2]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[1][3]), [0, 1, 0])


    def test_data_batches_not_truncated(self):
        generator = self.data_set.data_batches(batch_size=4, epochs=1, truncate=False)

        batches = [ batch for batch in generator ]
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].shape, (4, 3))
        self.assertEqual(batches[1].shape, (4, 3))
        self.assertEqual(batches[2].shape, (2, 3))

        self.assertSequenceEqual(list(batches[0][0]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[0][1]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[0][2]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[0][3]), [1, 0, 0])

        self.assertSequenceEqual(list(batches[1][0]), [0, 1, 0])
        self.assertSequenceEqual(list(batches[1][1]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[1][2]), [1, 0, 0])
        self.assertSequenceEqual(list(batches[1][3]), [0, 1, 0])

        self.assertSequenceEqual(list(batches[2][0]), [0, 0, 1])
        self.assertSequenceEqual(list(batches[2][1]), [1, 0, 0])

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
        inventory = DataFrame.from_records([ { 'id': 'id_{}'.format(i), 'target': 'cat_{}'.format(i % 3) }
                for i in range(10) ])

        data_encoder = [
            lambda record: (1,) + (int(record['target'][-1]),),
            lambda record: (2,) + (int(record['target'][-1]),),
            lambda record: (3,) + (int(record['target'][-1]),)
        ]

        def target_encoder(records):
            return [ int(record['target'].split('_')[-1]) for _, record in records.iterrows() ]

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

    def test_sort(self):
        self.data_set.sort(columns='target')

        self.assertEqual(len(self.data_set.inventory), 10)
        self.assertEqual(self.data_set.inventory.iloc[0]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory.iloc[1]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory.iloc[2]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory.iloc[3]['target'], 'cat_0')
        self.assertEqual(self.data_set.inventory.iloc[4]['target'], 'cat_1')
        self.assertEqual(self.data_set.inventory.iloc[5]['target'], 'cat_1')
        self.assertEqual(self.data_set.inventory.iloc[6]['target'], 'cat_1')
        self.assertEqual(self.data_set.inventory.iloc[7]['target'], 'cat_2')
        self.assertEqual(self.data_set.inventory.iloc[8]['target'], 'cat_2')
        self.assertEqual(self.data_set.inventory.iloc[9]['target'], 'cat_2')

        self.assertSequenceEqual(sorted(self.data_set.inventory.iloc[0:4]['id']),
                ['id_0', 'id_3', 'id_6', 'id_9'])
        self.assertSequenceEqual(sorted(self.data_set.inventory.iloc[4:7]['id']),
                ['id_1', 'id_4', 'id_7'])
        self.assertSequenceEqual(sorted(self.data_set.inventory.iloc[7:]['id']),
                ['id_2', 'id_5', 'id_8'])

    def test_shuffle(self):
        self.setUp(size=25, targets=25)

        before = np.array(self.data_set.inventory['id'])

        self.data_set.shuffle()

        after = np.array(self.data_set.inventory['id'])

        self.assertTrue(np.any(np.not_equal(before, after)))
        for _, record in self.data_set.inventory.iterrows():
            id_postfix = int(record['id'].split('_')[-1])
            target_postfix = int(record['target'].split('_')[-1])
            self.assertEqual(id_postfix, target_postfix)

    def test_batches_raises_if_batch_size_too_large(self):
        with self.assertRaises(ValueError):
            self.data_set.batches(batch_size=100)

        with self.assertRaises(ValueError):
            self.data_set.data_batches(batch_size=100)

        with self.assertRaises(ValueError):
            self.data_set.target_batches(batch_size=100)



class TestBatchDataEncoder:
    def __init__(self, id):
        self.id = id

    def transform_batch(self, records):
        pass
