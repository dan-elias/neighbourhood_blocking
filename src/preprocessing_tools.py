'''
==========================================================
preprocessing - feature extraction from individual records
==========================================================

Classes defined in this module
------------------------------

.. inheritance-diagram::  preprocessing_tools

Members
-------
'''
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import count, chain
import re
import numpy as np
import pandas as pd
import nameparser
from streetaddress import StreetAddressParser
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.pipeline import Pipeline

class ColumnTransformer(metaclass=ABCMeta):
    '''
    ABC for transformations to columns in a :class:`pandas.DataFrame`

    Abstract Methods:
        transform: must be overridden in subclasses.  Transforms a chunk

    Attributes:
        requires_fit (bool): indicates that partial_fit has been overridden
    '''
    @property
    def requires_fit(self):
        return any(hasattr(self, at) for at in ('fit', 'partial_fit'))
    @abstractmethod
    def transform(self, series_chunk):
        '''
        Transform a portion of a :class:`pandas.Series`
        Args:
            series_chunk (:class:`pandas.Series`): series chunk to transform
        Returns:
            (:class:`pandas.Series` or :class:`pandas.DataFrame`): transformed data (possibly
                split into multiple columns)
        '''
        raise NotImplementedError

class SplitColumnTransformer(ColumnTransformer):
    '''
    ABC for :class:`ColumnTransformer` that produces multiple columns

    Returns:
        (:class:`pandas.DataFrame`)
    '''
    def __init__(self, orig_colname='orig', **kwargs):
        '''
        Args:
            orig_colname (str): column name for original value (None to omit)
        '''
        super().__init__(**kwargs)
        self.orig_colname = orig_colname
    @abstractmethod
    def _calculated_columns(self, series_chunk):
        '''
        Must be overwritten in subclasses

        Args:
            series_chunk (:class:`pandas.Series`): source values
        Yields:
            (tuple) (*column name* (str), *column contents* (:class:`pandas.Series`))
        '''
        raise NotImplementedError
    def transform(self, series_chunk):
        col_items = chain([(self.orig_colname, series_chunk)] if self.orig_colname else [],
                          self._calculated_columns(series_chunk))
        result = None
        for name, vals in col_items:
            if result is None:
                result = pd.DataFrame({name: vals})
            else:
                result[name] = vals
        return result

class CompositeColumnTransformer(SplitColumnTransformer):
    '''
    Combination of multiple :class:`ColumnTransformer` objects
    '''
    def __init__(self, components, **kwargs):
        super().__init__(**kwargs)
        self.components = components
    @property
    def requires_fit(self):
        return any(component.requires_fit for component in self.components.values())
    def partial_fit(self, series_chunk):
        for component in self.components.values():
            if component.requires_fit:
                component.partial_fit(series_chunk)
    def _calculated_columns(self, series_chunk):
        for name, component in sorted(self.components.items()):
            part = component.transform(series_chunk)
            if all(hasattr(part, atr) for atr in ('columns', 'iteritems')):
                for part_colname, values in part.iteritems():
                    yield ('{name}_{part_colname}'.format(**locals()), values)
            else:
                yield (name, part)

class DatePartsColumnTransformer(SplitColumnTransformer):
    '''
    :class:`ColumnTransformer` to split datetime into parts

    '''
    _allowed_granularities = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond')
    def __init__(self, datetime_format=None, granularity='day', **kwargs):
        super().__init__(**kwargs)
        self.datetime_format = datetime_format
        if granularity not in self._allowed_granularities:
            raise TypeError('Unrecognized granularity: {}'.format(repr(granularity)))
        self.granularity = granularity
    def _calculated_columns(self, series_chunk):
        if self.datetime_format:
            series_chunk = pd.to_datetime(series_chunk, format=self.datetime_format, errors='coerce')
        for unit in self._allowed_granularities:
            yield (unit, getattr(series_chunk.dt, unit))
            if unit == self.granularity:
                break


class ParsedComponentColumnParser(SplitColumnTransformer):
    '''
    ABC for :class:`ColumnTransformer` for parsed components of a string

    Returns:
        (:class:`pandas.DataFrame`) parsed components
    '''
    def __init__(self, lowercase=False, **kwargs):
        '''
        Args:
            lowercase (bool): Transform components to lowercase
        '''
        super().__init__(**kwargs)
        self.lowercase = lowercase
    @abstractmethod
    def _parsed_columns(self, series_chunk):
        '''
        Args:
            series_chunk (:class:`pandas.Series`): strings to be parsed
        Yields:
            (tuple) (*column name* (str), *column contents* (:class:`pandas.Series`))
        '''
        raise NotImplementedError
    def _calculated_columns(self, series_chunk):
        return self._parsed_columns(series_chunk.fillna(''))
    def transform(self, series_chunk):
        result = super().transform(series_chunk)
        result.fillna(value=np.nan, inplace=True)
        if self.lowercase:
            for col, values in result.iteritems():
                result[col] = values.str.lower()
        return result


class HumanNameColumnParser(ParsedComponentColumnParser):
    '''
    :class:`ParsedComponentColumnParser` for human names
    '''
    def _parsed_columns(self, series_chunk):
        names = series_chunk.map(nameparser.HumanName)
        for colname in ['title', 'first', 'middle', 'last', 'suffix', 'nickname']:
            yield (colname, names.map(lambda x: getattr(x, colname)))

class StreetAddressColumnParser(ParsedComponentColumnParser):
    '''
    :class:`ParsedComponentColumnParser` for street addresses
    '''
    def _parsed_columns(self, series_chunk):
        addresses = series_chunk.map(StreetAddressParser)
        for colname in ['house', 'street_name', 'street_type', 'suite_num', 'suite_type', 'other']:
            yield (colname, addresses.map(lambda x: x[colname]))

class RegexColumnParser(ParsedComponentColumnParser):
    '''
    :class:`ParsedComponentColumnParser` for street addresses
    '''
    def __init__(self, regexs, sort=False, postprocess=None, **kwargs):
        self.regexs = regexs
        self.sort = sort
        self.postprocess = postprocess
        super().__init__(**kwargs)
    def _rec_items(self):
        def as_rec(x):
            return x if hasattr(x, 'findall') else re.compile(x)
        return ((colname, as_rec(val)) for colname, val in sorted(dict(self.regexs).items()))
    def _parsed_columns(self, series_chunk):
        for colname, rec in self._rec_items():
            tx = rec.findall if self.postprocess is None else lambda s: self.postprocess(rec.findall(s))
            yield (colname, series_chunk.map(tx))

class TopicModelTransformer(ColumnTransformer):
    '''
    :class:`ColumnTransformer` to produce topic weights for a text column
    '''
    def __init__(self, n_topics=10, nmf=True, lowercase=True,
                 max_df=0.95, min_df=2, ngram_range=(1,1),
                 stop_words='english', weights=False, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        vectorizer_type = TfidfVectorizer if nmf else CountVectorizer
        model_type = NMF if nmf else LatentDirichletAllocation
        model_kw = {('n_components' if nmf else 'n_topics'): n_topics}
        self.pipeline = Pipeline([('vectorizer', vectorizer_type(lowercase=lowercase,
                                                                 max_df=max_df,
                                                                 min_df=min_df,
                                                                 ngram_range=ngram_range,
                                                                 stop_words=stop_words)),
                                  ('model', model_type(**model_kw)),
                                  ])
    def partial_fit(self, series_chunk):
        if not hasattr(self, '_chunks_to_fit'):
            self._chunks_to_fit = []
        self._chunks_to_fit.append(series_chunk.dropna())
    def transform(self, series_chunk):
        if hasattr(self, '_chunks_to_fit'):
            self.pipeline.fit(chain.from_iterable(self._chunks_to_fit))
            del self._chunks_to_fit
        weights = self.pipeline.transform(series_chunk.fillna(''))
        topic_summary = pd.DataFrame(data=weights, index=series_chunk.index,
                                     columns=['topic_{}'.format(n) for n in range(weights.shape[1])])
        topic_summary['primary_topic'] = topic_summary.apply(np.argmax, axis='columns')
        return topic_summary if self.weights else topic_summary['primary_topic']


class FunctionTransformer(ColumnTransformer):
    '''
    :class:`ColumnTransformer` wrapper for a function
    '''
    def __init__(self, fun, **kwargs):
        super().__init__(**kwargs)
        self.fun = fun
    def transform(self, series_chunk):
        return self.fun(series_chunk)


class CsvElementEnumerator(ColumnTransformer):
    '''
    :class:`ColumnTransformer` to transform elements in csv strings to numeric codes

    Attributes:
        codes (dict): mapping of <sring element> -> numeric code
    '''
    def __init__(self):
        self.codes = {}
        self._code_seq = count()
    def _get_assigned_code(self, element):
        try:
            return self.codes[element]
        except KeyError:
            self.codes[element] = next(self._code_seq)
        return self.codes[element]
    def _parsed_elements(self, collection_repr):
        return (s.strip() for s in collection_repr.split(','))
    def transform(self, chunk_series):
        raise NotImplementedError

class DataFrameTransformer:
    '''
    Transform a columns in a :class:`pandas.DataFrame`
    '''
    def __init__(self, column_transformers=None):
        '''
        Args:
            column_transformers (mapping): mapping of column name
                -> :class:`ColumnTransformer`
        '''
        self.column_transformers = {} if column_transformers is None else column_transformers
        self.colnames_requiring_fit = {colname for colname, ctx in column_transformers.items() if ctx.requires_fit}
    @property
    def requires_fit(self):
        return bool(self.colnames_requiring_fit)
    def partial_fit(self, df):
        for colname in self.colnames_requiring_fit:
            self.column_transformers[colname].partial_fit(df[colname])
        return self
    def fit(self, df):
        '''
        Initialize transformation

        Args:
            df (:class:`pandas.DataFrame`): data to fit to
        '''
        if self.requires_fit:
            return self.partial_fit(df)
    def transform(self, df, inplace=True):
        '''
        Transformed data

        Args:
            df (:class:`pandas.DataFrame`): data to transform
        Returns:
            :class:`pandas.DataFrame`: transformed data
        '''
        if not inplace:
            df = df.copy()
        for colname, ctx in self.column_transformers.items():
            transformed_column = ctx.transform(df[colname])
            if isinstance(transformed_column, pd.Series):
                df[colname] = transformed_column
            elif isinstance(transformed_column, pd.DataFrame):
                del df[colname]
                for tx_colname, tx_values in transformed_column.iteritems():
                    df['{colname}_{tx_colname}'.format(**locals())] = tx_values
            else:
                raise TypeError('column {colname}: Unexpected transformed column type ({typ.__name__})'.format(typ=type(transformed_column), **locals()))
        return df
    def fit_transform(self, df, inplace=True):
        '''
        Convenience method to fit to dataset and transform it

        Args:
            df (:class:`pandas.DataFrame`): data to fit to and to transform
        Returns:
            :class:`pandas.DataFrame`: transformed data
        '''
        return self.fit(df).transform(df, inplace=inplace)
