"""
Utility functions used in the conversion process to hdf5 archives
"""

from typing import Optional, List, Dict, Union
from io import BufferedReader
import uuid
import urllib.parse
import hashlib

from ruamel.yaml import YAML as _YAML
from ruamel.yaml.compat import StringIO
import h5py


FALLBACK_UUID_NAMESPACE = uuid.UUID('c5908e58-7301-4054-9f04-a0fa8cdef63b')
PUBLIC_NAMESPACE = '/METADATA'
PRIVATE_NAMESPACE = '/.METADATA'
METADATA_PTR = 'CURRENT'
METADATA_LIST_PTR = 'CURRENT-LIST'

YAML = _YAML()
VLEN_STRING = h5py.special_dtype(vlen=str)


def _get_next_md_id(h5_group: h5py.Group, group_prefix: str) -> int:
    """
    Returns the next incremental ID for metadata versioning
    """
    ids = [0]

    def _append_id(h5_key):
        ids.append(int(h5_key))

    group = h5_group.get(group_prefix)
    if group:
        group.visit(_append_id)

    return max(ids) + 1


def _write_dataset(
        h5_group: h5py.Group,
        dataset: Dict,
        dataset_path: str = '/',
        track_order: bool = True) -> h5py.Dataset:
    """
    Internal function for writing dataset metadata to a H5 file
    """
    doc_group = '/'.join((PRIVATE_NAMESPACE, dataset_path))
    if not h5_group.get(doc_group):
        h5_group.create_group(doc_group, track_order=track_order)

    doc_id = str(_get_next_md_id(h5_group, doc_group))
    ds = h5_group.create_dataset(
        '/'.join((PRIVATE_NAMESPACE, dataset_path, doc_id)),
        dtype=VLEN_STRING,
        shape=(1,)
    )

    with StringIO() as _buf:
        YAML.dump(dataset, _buf)
        ds[()] = _buf.getvalue()

    public_group = '/'.join((PUBLIC_NAMESPACE, dataset_path))
    public_path = public_group + '/' + METADATA_PTR

    if not h5_group.get(public_group):
        h5_group.create_group(public_group, track_order=track_order)
    if h5_group.get(public_path):
        del h5_group[public_path]

    h5_group[public_path] = h5py.SoftLink(ds.name)

    return h5_group[public_path]


def write_h5_md(
        h5_group: h5py.Group,
        datasets: Union[List[Dict], Dict],
        dataset_names: Optional[List[str]] = None
        ) -> None:
    """
    Appends metadata documents to a hdf5 collection.
    Variable length strings will be written to records in
        /.METADATA/path/to/dataset_name/offset
    A softlink will be created at:
        /METADATA/path/to/dataset_name/CURRENT
    An array of CURRENT metadata docs will be available at:
        /METADATA/CURRENT-LIST
    """

    collection_path = '/'.join((PUBLIC_NAMESPACE, METADATA_LIST_PTR))
    new_metadata_paths = []
    old_metadata_paths = []

    if h5_group.get(collection_path):
        # Collate known metadata references
        old_collection = h5_group[collection_path]
        for i in range(old_collection._dcpl.get_virtual_count()):
            old_metadata_paths.append(old_collection._dcpl.get_virtual_dsetname(i))

    # Create metadata and collate new references
    for i, _ in enumerate(datasets):
        ref = _write_dataset(h5_group, datasets[i], dataset_names[i])
        if ref not in old_metadata_paths:
            new_metadata_paths.append(ref)

    if new_metadata_paths:
        # Extend the virtual layout for new datasets
        virtual_collection = h5py.VirtualLayout(
            shape=(len(old_metadata_paths) + len(new_metadata_paths),),
            dtype=VLEN_STRING)

        ds_cntr = 0
        for ds_cntr, _ in enumerate(old_metadata_paths):
            virtual_collection[ds_cntr] = h5py.VirtualSource(
                path_or_dataset='.',
                name=old_metadata_paths[ds_cntr],
                dtype=VLEN_STRING,
                shape=(1,))

        # Increment counter if a dataset was written
        ds_cntr = ds_cntr + 1 if ds_cntr else 0

        for j, _ in enumerate(new_metadata_paths):
            virtual_collection[ds_cntr+j] = h5py.VirtualSource(
                path_or_dataset='.',
                name=new_metadata_paths[j].name,
                dtype=VLEN_STRING,
                shape=(1,))

        if h5_group.get(collection_path):
            del h5_group[collection_path]

        h5_group.create_virtual_dataset(collection_path, virtual_collection)


def generate_fallback_uuid(
        product_href: str, uuid_namespace: uuid.UUID = FALLBACK_UUID_NAMESPACE,
        **product_params):
    """
    Generates a fallback UUID from the fallback UUID namespace
    """
    return uuid.uuid5(
        uuid_namespace,
        "{}?{}".format(product_href, urllib.parse.urlencode(product_params))
    )


def generate_md5sum(src: BufferedReader, chunk_size=16384):
    """
    Generate a md5sum for the src component.
    Used to help generate fallback uuids
    """
    md5_hash = hashlib.md5()
    for chunk in iter(lambda: src.read(chunk_size), b''):
        md5_hash.update(chunk)

    return md5_hash
