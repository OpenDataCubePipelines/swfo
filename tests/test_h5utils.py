import h5py
import numpy as np
import yaml

from swfo.h5utils import (
    _get_next_md_id,
    write_h5_md
)


def _in_memory_h5(tmp_path):
    """
    Helper function to setup h5 files
    """
    test_file = h5py.File(
        name=tmp_path / 'test_file.h5',
        driver='core',
        backing_store=False
        )

    # Create Datasets

    return test_file


def test_get_next_id(tmp_path):
    h5file = _in_memory_h5(tmp_path)

    # Test first id of empty group
    first_id = _get_next_md_id(h5file, '/')
    assert first_id == 1

    # Test first id of non-existent group
    first_id = _get_next_md_id(h5file, '/metadata')
    assert first_id == 1

    # Test getting next id
    h5file.create_dataset(
        name='/metadata/_1',
        data=np.empty(shape=(1,))
    )

    h5file.visit(print)
    second_id = _get_next_md_id(h5file, '/metadata')
    assert second_id == 2


class TestH5Write:
    def test_simple_dataset(self, tmp_path):
        h5file = _in_memory_h5(tmp_path)

        sample_md = {
            'id': 1,
            'nested_path': { 'key': 'value' }
        }

        write_h5_md(h5file, datasets=sample_md)

        read_dataset = h5file.get('/metadata/_1')

        assert read_dataset is not None
        doc = yaml.load(read_dataset[()][0])
        assert doc == sample_md
        del doc
        del read_dataset

        read_dataset = h5file.get('/metadata/current')
        assert read_dataset is not None
        doc = yaml.load(read_dataset[()][0])

        assert doc == sample_md

    def test_multi_dataset(self, tmp_path):
        h5file = _in_memory_h5(tmp_path)

        sample_md = [
            {'id': 2},
            {'id': 3},
            {'id': 4}
        ]

        dataset_names = ['two', 'three', 'four']

        write_h5_md(h5file, datasets=sample_md, dataset_names=dataset_names)

        for i in range(len(dataset_names)):
            path = '/metadata/{}'.format(dataset_names[i])

            read_dataset = h5file.get(path + '/_1')
            assert read_dataset is not None

            doc = yaml.load(read_dataset[()][0])

            assert doc == sample_md[i]

            assert read_dataset == h5file.get(path + '/current')

            assert (
                h5file['/metadata/current'].virtual_sources()[i].dset_name
                    == path + '/current'
            )

            del doc
            del read_dataset
