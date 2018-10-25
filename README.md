# Generator Data Set

A library to provide data generators that can be used to train Keras models.

The main class of the library provides is the *GeneratorDataSet* that is based on an inventory of records and provides generators that create batches of data based on the records in the inventory.

A typical inventory would consists of unique file names with an associated target label. During data generation for each file name entry the associated file is loaded and processed into a feature vector. In addition the target label associated with the entry is encoded into some scalar or vector representation. The batches of featurized data and encoded targets are fed to the model for training via a python generator that loops over the inventory in each epoch.

## General concept

The general idea of the library is to operate on the inventory, i.e. a list of records, where a record contains data as key value pairs. The inventory is divided into batches, and each record in a batch is processed by a data encoder to load and encode the data associated with the record, as well as by a target encoder to load and encode the learning target for the record.

![General concept](/doc/diagram.svg)

On top of this very general setup the library provides convenience classes to support the most common use cases and make the creation of data and target encoders as easy as possible.

In particular data encoder skeletons for inventories based on files and URLs  are provided, as well as skeletons to encode labels from the inventory based on standard SciPy data encoders, such as LabelEncoder or one hot encodings.

## Usage

The following sections describe the detailed API that provides most flexibility. The library provides also a couple of convenience methods that cover common use cases:

**[Data from files](#data-from-files)**<br>
**[Data from URLs](#data-from-urls)**

### Inventory

The inventory used to create the *GeneratorDataSet* must be a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

### Encoders

Encoders provide methods to transform a record in the inventory into a feature/target vector representation. Commonly this transformation involves IO operations or data augmentation or both. Data and target encoders must implement one of the following two interfaces:

    class BatchDataEncoder():
        def fit(self, inventory):
            pass

        def transform_batch(self, records):
            raise NotImplementedError()


    class DataEncoder():
        def __call__(self, record):
            return self.transform(record)

        def fit(self, inventory):
            pass

        def transform(self, record):
            raise NotImplementedError()

        def finalize_batch(self, records):
            return records

The fit method is optional in both cases. In the second case it is sufficient for the encoder to be callable, all other methods are optional.

The library provides encoders for common use cases:

#### File based data

For data that is loaded from files it provides a *FileDataEncoder* that will take care of basic file handling and only data transformation from the file handle needs to be implemented.

#### URL based data

For data that is loaded from files it provides a *UrlDataEncoder* that will take care of basic resource loading and only data transformation from the the returned data needs to be implemented.

### Target encoders

Also several target encoders are provided to make the transformation from e.g. labels in the inventory to e.g. integer or one-hot encoding as easy as possible. See the unit tests for examples.

### Creation of a *GeneratorDataSet*

The library provides factory methods for the most common use cases.

#### Data from files

For this case we assume that the data consists of a list of files contained in a data directory. Next to the data files there is an inventory *.csv* file that contains the association of each file to a target category. Then a *GeneratorDataSet* based on the inventory file can be created by

    from numblr.datagenerator import generator_for_files, LabelEncoder

    generator_for_files('inventory.csv', 'my/data/dir', MyDataEncoder(), LabelEncoder())

In the full form

    generator_for_files(('inventory.csv', 'my/data/dir', MyDataEncoder(), LabelEncoder(),
            id_mapper=lambda id: id + '.ext',
            id='file_name',
            target='label',
            binary=False)

#### Data from URLs

    from numblr.datagenerator import generator_for_urls, LabelEncoder

    generator_for_urls('inventory.csv', 'my/data/dir', MyDataEncoder(), LabelEncoder())

In the full form

    generator_for_urls(('inventory.csv', 'my/data/dir', MyDataEncoder(), LabelEncoder(),
        id_mapper=lambda id: id + '.ext',
        id='file_name',
        target='label',
        binary=False)

## Examples

For basic usage see the integration test in */test/test_integration.py* and the
examples in the */examples* directory.

To run the examples the top level directory of this repository must be added to the python path for execution. One way to do this is run the script from within the respective */examples/the_example* directory with the following command:

    > PYTHONPATH=".." python the_example.py

### MNIST example

The example creates an example file based data set based on the MNIST data set provided by *Keras*. The entries in the MNIST data set are exported a text file for each digit and an inventory of the files with the associated digit they represent is created. Based on the inventory a *GeneratorDataSet* is created with a file based data encoder and a one hot target encoder to train a simple DNN with the generators obtained from the *GeneratorDataSet*.

## Tests

Run the unit tests of the library with

    pyhton -m unittest

from the root directory of the repository.

## References

See also *fit_generator* on [Keras models](https://keras.io/models/model/).
