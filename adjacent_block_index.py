from itertools import product

import numpy as np
import pandas as pd

from recordlinkage.indexing import BlockIndex

def ranking1d(x):
    ndx_by_rank = np.argsort(x)
    rank_by_ndx = np.empty_like(ndx_by_rank)
    rank_by_ndx[ndx_by_rank] = np.arange(len(ndx_by_rank))
    return rank_by_ndx

class AdjacentBlockIndex(BlockIndex):
    '''
    :class:`recordlinkage.indexing.BlockIndex` that 
    includes matches with elements in adjacent blocks
    and null values
    '''
    def __init__(self, *args, max_nulls=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_nulls = max_nulls
    def _link_index(self, df_a, df_b):
        default_on = list(set(df_a.columns) & set(df_b.columns))
        raw_key_columns = [side_on or self.on or default_on for side_on in [self.left_on, self.right_on]]
        if not (all(raw_key_columns) and (len(raw_key_columns[0]) == len(raw_key_columns[1]))):
            raise IndexError('Invalid indexing columns')
        key_dfs = [df[side_on].copy() for df, side_on in zip([df_a, df_b], raw_key_columns)]
        key_columns = ['k_{ndx}'.format(**locals()) for ndx in range(len(key_dfs[0].columns))]
        for df in key_dfs:
            df.columns = key_columns
        def get_key_rank(key_series):
            unique_values = key_series.unique()
            result = pd.Series(ranking1d(unique_values), index=unique_values)
            if np.nan in result:
                result[np.nan] = np.nan
            return result
        key_ranks = {name: get_key_rank(vals) 
                     for name, vals in pd.concat(key_dfs, ignore_index=True).iteritems()}
        index_columns = ['index_{ndx}'.format(**locals()) for ndx in range(len(key_dfs))]
        for index_column, key_df in zip(index_columns, key_dfs):
            key_df[index_column] = key_df.index
            for col in key_columns:
                key_df[col] = key_ranks[col][key_df[col].values].values
        def get_block_df(key_df, df_id):
            blocks = key_df[key_columns].drop_duplicates().reset_index(drop=True)
            blocks.index.name = 'block_id_{df_id}'.format(**locals())
            return blocks
        block_dfs = [get_block_df(key_df, df_id) for df_id, key_df in enumerate(key_dfs)]
        key_dfs = [key_df.join(block_df[key_columns].reset_index().set_index(key_columns), on=key_columns)
                   for key_df, block_df in zip(key_dfs, block_dfs)]
        def df_with_adjusted_nulls(df):
            df = df[df[key_columns].isnull().sum(axis='columns') <= self.max_nulls]
            for col in key_columns:
                df[col].fillna(-2, inplace=True)            
            return df
        def get_adjusted_block_df(block_df_to_adjust, non_null_adjustments):
            rank_increments = [non_null_adjustments+[np.nan] if (self.max_nulls > 0) and (np.nan in key_ranks[col]) else non_null_adjustments
                               for col in key_columns]
            rank_increment_combinations = np.array([incs.flatten() for incs in np.meshgrid(*rank_increments)])
            rank_increment_combinations = rank_increment_combinations[:,(np.sum(np.isnan(rank_increment_combinations), axis=0)<=self.max_nulls)]
            ix_repeated_rows = np.tile(np.arange(len(block_df_to_adjust)), rank_increment_combinations.shape[1])
            result = pd.DataFrame((block_df_to_adjust[key_columns].values[...,np.newaxis] 
                                   + rank_increment_combinations[np.newaxis,...]).transpose([2,0,1]).reshape(-1,len(key_columns)),
                                  columns=key_columns,
                                  index = block_df_to_adjust.index[ix_repeated_rows])
            result = result.join(block_df_to_adjust[[c for c in block_df_to_adjust.columns if c not in result.columns]])
            result = result.reset_index().drop_duplicates().set_index(block_df_to_adjust.index.name)
            result = df_with_adjusted_nulls(result)
            return result
        ext_block_dfs = [get_adjusted_block_df(df, non_null_adjustments=[-1,0,1])
                         for df, adjs in zip(block_dfs, [[-1,0,1], [0]])]
        adj_block_dfs = [df_with_adjusted_nulls(df) for df in block_dfs]
        block_match_components = [df0.reset_index().join(df1.reset_index().set_index(key_columns), on=key_columns, how='inner')
                                  for df0, df1 in zip(ext_block_dfs, reversed(adj_block_dfs))]
        block_matches = pd.concat(block_match_components, ignore_index=True)
        block_matches = block_matches.loc[(block_matches[key_columns]<0).sum(axis='columns').values<=self.max_nulls, :]
        block_id_columns = [df.index.name for df in block_dfs]
        block_matches = block_matches[block_id_columns].drop_duplicates()
        record_index_dfs = [df[[c for c in df.columns if c in index_columns or c in block_id_columns]]
                            for df in key_dfs]
        record_matches = record_index_dfs[0]\
                         .merge(block_matches, on=block_id_columns[0])\
                         .merge(record_index_dfs[1], on=block_id_columns[1])
        record_matches = record_matches[index_columns].drop_duplicates()
        record_matches = record_matches[record_matches.iloc[:,0] != record_matches.iloc[:,1]]
        return record_matches.set_index(index_columns).index.rename([df.index.name for df in key_dfs])
