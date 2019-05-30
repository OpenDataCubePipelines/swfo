"""
Utility functions used in the conversion process to hdf5 archives
"""

from typing import Optional, List, Dict, Union
from io import StringIO, BufferedReader
import uuid
import urllib.parse
import hashlib

from ruamel.yaml import YAML
import h5py

VLEN_STRING = h5py.special_dtype(vlen=str)
FALLBACK_UUID_NAMESPACE = uuid.UUID('c5908e58-7301-4054-9f04-a0fa8cdef63b')

YAML_SERIALISER = YAML()
YAML_SERIALISER.indent(mapping=2, sequence=4, offset=2)


def _get_next_md_id(h5_group: h5py.Group, group_prefix: str) -> int:
    """
    Returns the next incremental ID for metadata versioning
    """
    ids = [0]

    def _append_id(h5_key):
        ids.append(int(h5_key[1:]))

    group = h5_group.get(group_prefix)
    if group:
        group.visit(_append_id)

    return max(ids) + 1


def write_h5_md(
        h5_group: h5py.Group,
        datasets: Union[List[Dict], Dict],
        dataset_names: Optional[List[str]] = None) -> None:
    """
    Serialises provided metadata information to yaml format inside hdf5
    archive.
    If a single dataset is provided the metadata is written to /METADATA/_{index}
    and linked to /METADATA/CURRENT

    If multiple datasets are provided and named, they are written to
        /METADATA/{dataset_name}/_{index}
    linked to /METADATA/{dataset_name}/CURRENT and
    consolidated in a virtual dataset under /METADATA/CURRENT
    """
    path_fmt = '/METADATA/{}'

    def _write_dataset(
            h5_group: h5py.Group,
            path: str,
            dataset: Dict) -> h5py.Dataset:
        """
        Internal function for writing dataset metadata to a H5 file
        """
        ds = h5_group.create_dataset(
            path,
            dtype=VLEN_STRING,
            shape=(1,),
        )

        with StringIO() as _buf:
            YAML_SERIALISER.dump(dataset, _buf)
            ds[()] = _buf.getvalue()

        current_path = path.rsplit('/', 1)[0] + '/CURRENT'
        if h5_group.get(current_path):
            del h5_group[current_path]

        h5_group[current_path] = h5py.SoftLink(ds.name)

        return h5_group[current_path]

    if not dataset_names:
        doc_id = _get_next_md_id(h5_group, path_fmt.format(''))
        _write_dataset(
            h5_group,
            path_fmt.format('_' + str(doc_id)),
            datasets
        )
    else:
        collection_layout = h5py.VirtualLayout(
            shape=(len(dataset_names), 1),
            dtype=VLEN_STRING
        )
        for idx, dn in enumerate(dataset_names):
            dataset_path = path_fmt.format(dn)
            doc_id = _get_next_md_id(h5_group, dataset_path)
            current_ref = _write_dataset(
                h5_group,
                dataset_path + '/_' + str(doc_id),
                datasets[idx]
            )
            collection_layout[idx, :] = h5py.VirtualSource(current_ref)

        collection_path = path_fmt.format('CURRENT')
        if h5_group.get(collection_path):
            del h5_group[collection_path]

        h5_group.create_virtual_dataset(collection_path, collection_layout)


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
