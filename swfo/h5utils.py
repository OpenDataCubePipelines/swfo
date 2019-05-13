from typing import Optional, List, Dict, Union
from functools import partial
import yaml

import h5py

VLEN_STRING = h5py.special_dtype(vlen=str)


yaml_dumper = partial(yaml.dump, indent=4, default_flow_style=False)


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
    If a single dataset is provided the metadata is written to /metadata/_{index}
    and linked to /metadata/current

    If multiple datasets are provided and named, they are written to
        /metadata/{dataset_name}/_{index}
    linked to /metadata/{dataset_name}/current and consolidated in a virtual dataset
    under /metadata/current
    """
    path_fmt = '/metadata/{}'

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
        ds[()] = yaml_dumper(dataset)

        current_path = path.rsplit('/', 1)[0] + '/current'
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

        collection_path = path_fmt.format('current')
        if h5_group.get(collection_path):
            del h5_group[collection_path]

        h5_group.create_virtual_dataset(collection_path, collection_layout)
