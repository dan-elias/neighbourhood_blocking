'''
============================
misc - Miscellaneous helpers
============================

Members
-------
'''
from collections.abc import MutableMapping
import pathlib
import pandas as pd


class PickleMap(MutableMapping):
    '''
    Mapping API for a folder of pickle files
    '''
    def __init__(self, folder, cache_file_suffix='.pickle'):
        '''
        Args:
            folder (str or :class:`pathlib.Path`): folder to store pickle files
            cache_file_suffix (str): suffix for file names (default: ".pickle")
        '''
        self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cache_file_suffix = cache_file_suffix
    # Start: MutableMapping abstract methods
    def __getitem__(self, key):
        try:
            return pd.read_pickle(str(self._cache_file(key)))
        except FileNotFoundError:
            raise KeyError(key)
    def __setitem__(self, key, value):
        pd.to_pickle(value, str(self._cache_file(key)))
    def __delitem__(self, key):
        try:
            self._cache_file(key).unlink()
        except FileNotFoundError:
            raise KeyError(key)
    def __iter__(self):
        return (p.stem for p in self._cache_files())
    def __len__(self):
        return len(list(self._cache_files()))
    # End: MutableMapping abstract methods
    def _cache_file(self, key):
        return self.folder.joinpath(key).with_suffix(self.cache_file_suffix)
    def _cache_files(self):
        return self.folder.glob('*{self.cache_file_suffix}'.format(**locals()))
        
        