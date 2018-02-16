'''
===================================
dataset - handlers for raw datasets
===================================

This module contains tools for collecting source datasets and
accessing them in a consistent way.

Classes defined in this module
------------------------------

.. inheritance-diagram::  dataset_tools


Members
-------
'''
from abc import abstractmethod
from collections import ChainMap, defaultdict
from collections.abc import MutableMapping
from contextlib import suppress
from itertools import chain, combinations, product
from urllib.parse import urlparse
import pathlib, re, sys, urllib.request, shutil, zipfile

import numpy as np
import recordlinkage.datasets
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from settings import paths
from misc import PickleMap




class Dataset(PickleMap):
    '''
    Abstract base class for dataset wrappers.  Common functionality:
        * sourcing and extraction (override methods)
        * caching
        * metadata and summary statistics
        * mapping api for database objects
    '''
    def __init__(self, name, cache_subdir=None):
        super().__init__(folder=paths['data_interim'].joinpath(cache_subdir, name))
        self.name = name
    def table_stats(self):
        stats_dict = defaultdict(list)
        for k, v in self.items():
            stats_dict['key'].append(k)
            for colname, getter in [('length', len), 
                                    ('column_count', lambda x: len(x.columns)),
                                    ('type', type),
                                    ]:
                stat = np.nan
                with suppress(Exception):
                    stat = getter(v)
                stats_dict[colname].append(stat)
        return pd.DataFrame(stats_dict).set_index('key').sort_index(axis='columns').sort_index(axis='rows')
    def column_stats(self, reset_indices=True):
        def df_stats(key, value):
            if not(isinstance(value, pd.DataFrame) or (reset_indices and isinstance(value, pd.Index))):
                return
            df = pd.DataFrame(index=value) if isinstance(value, pd.Index) else value
            assert isinstance(df, pd.DataFrame)
            if reset_indices:
                df = df.reset_index()
            stats = pd.DataFrame({'dtype': df.dtypes})
            stats.index.name = 'column'
            stats['key'] = key
            for colname, getter in [('null_count', lambda c: c.isnull().sum()), 
                                    ('distinct_count', lambda c: len(c.unique())), 
                                    ]:
                stats[colname] = df.apply(getter, axis='rows')
            row_count = len(df) or np.nan  # NaN if row_count is zero
            for prefix in ['null', 'distinct']:
                stats['{prefix}_proportion'.format(**locals())] = stats['{prefix}_count'.format(**locals())] / row_count
            return stats.reset_index()
        return pd.concat(df_stats(k, v) for k, v in self.items() if v is not None).set_index(['key', 'column'])
    def extract(self, clear=True):
        if clear:
            self.clear()
        for k, v in self._extracted_items():
            self[k] = v
    def __repr__(self):
        src = getattr(self, 'source_url', self.name)
        return '{self.__class__.__name__}<{src}>'.format(**locals())
    @abstractmethod
    def _extracted_items(self):
        '''
        Names and objects produced by ETL
        
        Yields:
            tuple: (key, value) pairs
        '''
        raise NotImplementedError

class RawDataset(Dataset):
    def __init__(self, name, cache_subdir='raw', **kwargs):
        super().__init__(name=name, cache_subdir=cache_subdir, **kwargs)

class DownloadableDataset(RawDataset):
    '''
    Dataset for a downloaded zip file

    '''
    def __init__(self, source_url, table_defs=None, name=None, **kwargs):
        '''
        Args:
            source_url (str): Source url
            table_defs (list(tuple)): each tuple describes a table as indicated below
            name (str): Override dataset name (default parsed from *source_url*)

        Each tuple in table_defs describes a table in the dataset and must
        contain the following fields:

            * name (str): table name
            * member (str): path within (zip) archive file
            * file_type (str): a pandas-readable file type (eg: "csv")
            * kwargs (dict): keyword arguments for pandas import function (see `pandas.io <http://pandas.pydata.org/pandas-docs/stable/io.html>`_)
            * transformer (callable): transformation to apply to each :class:`pandas.DataFrame` imported (eg: lambda x:x)
        '''
        self.source_url = source_url
        self.table_defs = table_defs
        super().__init__(name = re.sub(r'\..*', '', self.raw_path.stem) if name is None else name, **kwargs)
    @property
    def raw_path(self):
        path = paths['data_raw'] / pathlib.PurePosixPath(urlparse(self.source_url).path).name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    def _extracted_items(self, force=False):
        if force or (not self.raw_path.exists()):
            print('downloading {self.name}'.format(self=self))
            with urllib.request.urlopen(self.source_url) as in_stream, self.raw_path.open('wb') as out_stream:
                shutil.copyfileobj(in_stream, out_stream)
        with zipfile.ZipFile(self.raw_path) as archive:
            for name, member, file_type, kwargs, df_tx in self.table_defs:
                read_method = getattr(pd, 'read_{file_type}'.format(file_type=file_type))
                with archive.open(member, 'r') as in_stream:
                    yield (name, df_tx(read_method(in_stream, **kwargs)))

class recordlinkageDataset(RawDataset):
    '''
    Dataset where source data is supplied in recordlinkage.datasets
    '''
    def __init__(self, name, table_names='df1', **kwargs):
        '''
        Args:
            name (str): dataset name (name or retrieval function is this name prefixed with "load\_")
            table_names (str or list(str)): names of tables returned by retrieval function
        '''
        self.table_names = [table_names] if isinstance(table_names, str) else table_names
        super().__init__(name=name, **kwargs)
    def _extracted_df_items(self):
        frames = getattr(recordlinkage.datasets, 'load_{self.name}'.format(**locals()))()
        if isinstance(frames, pd.DataFrame):
            frames = [frames]
        return zip(self.table_names, frames)
    def _true_mapping_index(self):
        groupings = []
        febrl_record_number_rec = re.compile(r'rec-(\d+)-(?:org|dup-\d+)')
        def febrl_record_number_from_id(rec_id):
            return int(febrl_record_number_rec.match(rec_id).groups()[0])
        for _, df in self._extracted_df_items():
            grouping = defaultdict(set)
            groupings.append(grouping)
            for rec_id in df.index:
                grouping[febrl_record_number_from_id(rec_id)].add(rec_id)
        if len(groupings) == 1:
            pairs = chain.from_iterable(map(lambda x: combinations(x, 2), groupings[0].values()))
        elif len(groupings) == 2:
            g0, g1 = groupings
            pairs = chain.from_iterable(product(ids, g1[k]) for k, ids in g0.items() if k in g1)
        else:
            raise ValueError('{} tables encountered when either 1 or 2 was expected'.format(len(groupings)))
        columns = ['id1', 'id2']
        return pd.DataFrame.from_records(pairs, columns=columns).set_index(columns).index
    def _extracted_items(self):
        yield from self._extracted_df_items()
        yield ('true_mapping', self._true_mapping_index())


class TransformedDataset(Dataset):
    '''
    A :class:`Dataset` that's calculated from another
    '''
    def __init__(self, source_dataset, transforms, cache_subdir='transformed', **kwargs):
        '''
        Args:
            source_dataset (:class:`DataSet`): source dataset
            transforms (mapping (key -> :class:`preprocessing_tools.DataFrameChunkTransformer`)): mapping of dataset objects to transforms to be applied to them
            cache_subdir (str): subdirectory for cache files
        '''
        super().__init__(cache_subdir=cache_subdir, **dict(ChainMap(kwargs, {'name':source_dataset.name})))
        self.source_dataset = source_dataset
        self.transforms = transforms
    def _extracted_items(self):
        # Transformers might be re-used, so fit them all first
        for key, transform in self.transforms.items():
            if transform.requires_fit:
                transform.partial_fit(self.source_dataset[key])
        for key, value in self.source_dataset.items():
            tx_value = self.transforms[key].transform(value) if key in self.transforms else value
            yield (key, tx_value)


class DatasetGroup(dict):
    '''
    Container for :class:`Dataset` (or subclasses) objects.

    This is a subclass of :class:`dict` with methods added for performing
    record linkage operations.  Keys are the dataset names.
    '''
    def __init__(self, datasets):
        '''
        Args:
            datasets (iterable of :class:`Dataset`):
        '''
        self.update({ds.name: ds for ds in datasets})
    def extract_all(self):
        '''
        Run extract on all datasets
        '''
        for dataset in self.values():
            dataset.extract()
    def _combined_component_dfs(self, method_name, force_recalc=False):
        def get_chunks():
            for dataset in self.values():
                chunk = getattr(dataset, method_name)().copy()
                non_index_cols = list(chunk.columns)
                chunk.reset_index(inplace=True)
                index_cols = ['dataset'] + [c for c in chunk.columns if c not in non_index_cols]
                chunk[index_cols[0]] = dataset.name
                chunk.set_index(index_cols, inplace=True)
                yield chunk
        chunks = get_chunks()
        first_chunk = next(chunks)
        result = pd.concat(chain([first_chunk], chunks))
        for colname, dtype in first_chunk.dtypes.items():
            result[colname] = result[colname].astype(dtype)
        return result
    def table_stats(self, force_recalc=False):
        'Combined table statistics for component datasets'
        return self._combined_component_dfs('table_stats', force_recalc=force_recalc)
    def column_stats(self, force_recalc=False):
        'Combined column statistics for component datasets'
        return self._combined_component_dfs('column_stats', force_recalc=force_recalc)
    def __repr__(self):
        datasets_repr = ', '.join(repr(ds) for ds in self.values())
        return '{self.__class__.__name__}([{datasets_repr}])'.format(**locals())
