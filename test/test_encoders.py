from pprint import pprint

import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from numblr.datagenerator.encoders import *

class TestFileDataEncoder(unittest.TestCase):
    def setUp(self):
        self.records = pd.DataFrame.from_records([
                { 'id': 'id1' },
                { 'id': 'id2' },
                { 'id': 'id3' }])

        def data_encoder(data):
            return int(data.readline().split('_')[-1])

        self.data_encoder = data_encoder
        self.encoder = FileDataEncoder(self.data_encoder, 'test/resources',
                lambda id: id + '.txt')

    def test_get_path(self):
        self.assertEqual(self.encoder.get_path(self.records.iloc[0]), 'test/resources/id1.txt')
        self.assertEqual(self.encoder.get_path(self.records.iloc[1]), 'test/resources/id2.txt')
        self.assertEqual(self.encoder.get_path(self.records.iloc[2]), 'test/resources/id3.txt')

    def test_encode(self):
        self.assertEqual(self.encoder.encode(self.records.iloc[0]), 1)
        self.assertEqual(self.encoder.encode(self.records.iloc[1]), 2)
        self.assertEqual(self.encoder.encode(self.records.iloc[2]), 3)

    def test_callable(self):
        self.assertEqual(self.encoder(self.records.iloc[0]), 1)
        self.assertEqual(self.encoder(self.records.iloc[1]), 2)
        self.assertEqual(self.encoder(self.records.iloc[2]), 3)

    def test_get_size(self):
        self.assertEqual(self.encoder.get_size(self.records.iloc[0]), 7)
        self.assertEqual(self.encoder.get_size(self.records.iloc[1]), 7)
        self.assertEqual(self.encoder.get_size(self.records.iloc[2]), 7)


class TestUrlDataEncoder(unittest.TestCase):
    def setUp(self):
        self.records = pd.DataFrame.from_records([
                { 'id': 'id1' },
                { 'id': 'id2' },
                { 'id': 'id3' }])

        def data_encoder(data):
            return data

        self.data_encoder = data_encoder
        self.encoder = UrlDataEncoder(self.data_encoder, 'http://google.com',
                lambda id: 'search?q=' + id)

    def test_get_path(self):
        self.assertEqual(self.encoder.get_path(self.records.iloc[0]), 'http://google.com/search?q=id1')
        self.assertEqual(self.encoder.get_path(self.records.iloc[1]), 'http://google.com/search?q=id2')
        self.assertEqual(self.encoder.get_path(self.records.iloc[2]), 'http://google.com/search?q=id3')

    @unittest.skip
    def test_get_size(self):
        self.assertTrue(int(self.encoder.get_size(self.records.iloc[0])) > 0)

    @unittest.skip
    def test_callable(self):
        self.assertTrue(self.encoder(self.records.iloc[0]).lower().startswith("<!doctype html>"))


class TestRecordTargetEncoder(unittest.TestCase):
    def setUp(self):
        self.records = pd.DataFrame.from_records([
                { 'target': 'one' },
                { 'target': 'two' },
                { 'target': 'three' }])

    def test_transform_with_single_encoder(self):
        encoder = LabelRecordEncoder()

        encoder.fit(self.records)

        transformed = encoder.transform(self.records)
        self.assertEqual(len(transformed), 3)
        self.assertEqual(set(transformed), set([0, 1, 2]))

        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[0]])), ['one'])
        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[1]])), ['two'])
        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[2]])), ['three'])

    def test_transform_with_encoder_sequence(self):
        encoder = OneHotRecordEncoder()

        encoder.fit(self.records)

        transformed = encoder.transform(self.records)
        self.assertEqual(len(transformed), 3)
        self.assertEqual(set(transformed.flatten()), set([0, 1]))
        self.assertEqual(abs(np.linalg.det(transformed)), 1)

        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[0]])), ['one'])
        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[1]])), ['two'])
        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[2]])), ['three'])

    def test_fit_transform(self):
        encoder = LabelRecordEncoder()

        transformed = encoder.fit_transform(self.records)
        self.assertEqual(len(transformed), 3)
        self.assertEqual(set(transformed), set([0, 1, 2]))

        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[0]])), ['one'])
        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[1]])), ['two'])
        self.assertEqual(encoder.inverse_transform(encoder.transform(self.records.iloc[[2]])), ['three'])

    def test_inverse_transform_with_single_encoder(self):
        encoder = LabelRecordEncoder()

        encoder.fit(self.records)

        self.assertEqual(set(encoder.classes_), set(['one', 'two', 'three']))
        self.assertEqual(len(encoder.inverse_transform([0, 1, 2])), 3)
        assert_array_equal(encoder.inverse_transform([0, 1, 2]), encoder.classes_)

    def test_inverse_transform_with_encoder_sequence(self):
        encoder = OneHotRecordEncoder()

        encoder.fit(self.records)

        self.assertEqual(set(encoder.classes_), set(['one', 'two', 'three']))
        assert_array_equal(encoder.inverse_transform([[1, 0, 0], [0, 1, 0], [0, 0 ,1]]), encoder.classes_)
