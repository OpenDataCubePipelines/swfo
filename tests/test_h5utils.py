from contextlib import contextmanager
from pathlib import Path
import os

import numpy as np
import h5py

from swfo.h5utils import (
    _get_next_md_id,
    atomic_h5_write,
    write_h5_md,
    PUBLIC_NAMESPACE,
    PRIVATE_NAMESPACE,
    METADATA_PTR,
    METADATA_LIST_PTR,
    YAML,
)


@contextmanager
def _in_memory_h5(tmp_path):
    """
    Helper function to setup h5 in memory file
    """
    with h5py.File(
        name=tmp_path / "test_file.h5", driver="core", backing_store=False
    ) as test_file:

        yield test_file


def test_get_next_id(tmp_path):
    """
    * Tests next id resolution for the hdf5 archives
    """
    with _in_memory_h5(tmp_path) as h5file:

        # Test first id of empty group
        first_id = _get_next_md_id(h5file, "/")
        assert first_id == 1

        # Test first id of non-existent group
        first_id = _get_next_md_id(h5file, PRIVATE_NAMESPACE)
        assert first_id == 1

        # Test getting next id
        h5file.create_dataset(
            name=os.path.join("/", PRIVATE_NAMESPACE, "1"), data=np.empty(shape=(1,))
        )

        second_id = _get_next_md_id(h5file, PRIVATE_NAMESPACE)
        assert second_id == 2


class TestH5Write:
    def test_multi_dataset(self, tmp_path):
        """
        Simple tests for writing multiple datasets
        """
        sample_md = [{"id": 2}, {"id": 3}, {"id": 4, "nested_path": {"key": "value"}}]
        dataset_names = ["two", "three", "FOUR"]

        with _in_memory_h5(tmp_path) as h5file:
            write_h5_md(h5file, datasets=sample_md, dataset_names=dataset_names)
            virtual_collection = h5file[
                os.path.join("/", PUBLIC_NAMESPACE, METADATA_LIST_PTR)
            ]

            for i, _ in enumerate(dataset_names):
                path = os.path.join("/", PRIVATE_NAMESPACE, dataset_names[i])

                read_dataset = h5file.get(path + "/1")
                assert read_dataset is not None

                doc = YAML.load(read_dataset[()][0])

                assert doc == sample_md[i]

                public_ref = os.path.join(
                    "/", PUBLIC_NAMESPACE, dataset_names[i], METADATA_PTR
                )
                assert read_dataset == h5file.get(public_ref)

                assert virtual_collection.virtual_sources()[i].dset_name == public_ref

                del doc
                del read_dataset

    def test_dataset_move(self, tmp_path):
        """
        * Tests the consistency of internal links when moving the HDF5 file.
            This test is relevant to the handling of virtual datasets
        """
        h5fp = tmp_path / "test.h5"
        sample_md = [{"id": 2}]
        public_path = os.path.join("/", PUBLIC_NAMESPACE, METADATA_PTR)

        with h5py.File(h5fp, "w") as h5file:
            write_h5_md(h5file, datasets=sample_md, dataset_names=["/"])
            virtual_collection = h5file[
                os.path.join("/", PUBLIC_NAMESPACE, METADATA_LIST_PTR)
            ]

            assert virtual_collection.virtual_sources()[0].dset_name == public_path
            assert sample_md[0] == YAML.load(virtual_collection[()][0])

            del virtual_collection

        h5fp.rename(tmp_path / "test2.h5")
        h5fp = tmp_path / "test2.h5"
        with h5py.File(h5fp, "r") as h5file:
            virtual_collection = h5file[
                os.path.join("/", PUBLIC_NAMESPACE, METADATA_LIST_PTR)
            ]

            assert virtual_collection.virtual_sources()[0].dset_name == public_path
            assert sample_md[0] == YAML.load(virtual_collection[()][0])

            del virtual_collection

    def test_append_to_dataset(self, tmp_path):
        """
        * Tests appending a third dataset to the collection
        * Tests updating metadata documents by appending the new version
            and updating the internal references within the same file handle
        """
        datasets = [{"id": 1}, {"id": 2}, {"id": 3}]
        dataset_names = ["one", "two", "three"]
        with _in_memory_h5(tmp_path) as h5file:
            write_h5_md(
                h5file, datasets=[{"id": "NA"}] * 2, dataset_names=dataset_names[:2]
            )
            write_h5_md(h5file, datasets=datasets, dataset_names=dataset_names)

            for idx, dname in enumerate(dataset_names):
                public_path = os.path.join("/", PUBLIC_NAMESPACE, dname, METADATA_PTR)
                assert datasets[idx] == YAML.load(h5file[public_path][()].item())
                # h5file[h5file[public_path].ref] is deferencing the public path
                assert h5file[h5file[public_path].ref].name == os.path.join(
                    "/", PRIVATE_NAMESPACE, dname, ["2", "2", "1"][idx]
                )

    def test_append_to_existing_dataset(self, tmp_path):
        fname = Path(tmp_path) / "test-append.h5"
        datasets = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        dataset_names = ["one/one", "two/two", "one/three", "four/four"]
        with atomic_h5_write(fname, "a", track_order=True) as dest:
            write_h5_md(
                dest, datasets=[{"id": "NA"}] * 2, dataset_names=dataset_names[:2]
            )
            for dname in dataset_names[:2]:
                dest.create_dataset(dname, data=np.empty(shape=(1,)))

        with atomic_h5_write(fname, "a", track_order=True) as dest:
            write_h5_md(dest, datasets=datasets, dataset_names=dataset_names)
            for dname in dataset_names[2:]:
                dest.create_dataset(dname, data=np.empty(shape=(1,)))

            with h5py.File(fname, "r") as src:
                # assert datasets haven't been written to pre-existing file
                assert len(src.keys()) == 4

        with h5py.File(fname, "r") as h5file:
            for idx, dname in enumerate(dataset_names):
                public_path = os.path.join("/", PUBLIC_NAMESPACE, dname, METADATA_PTR)
                assert datasets[idx] == YAML.load(h5file[public_path][()].item())
                # h5file[h5file[public_path].ref] is deferencing the public path
                assert h5file[h5file[public_path].ref].name == os.path.join(
                    "/", PRIVATE_NAMESPACE, dname, ["2", "2", "1", "1"][idx]
                )
