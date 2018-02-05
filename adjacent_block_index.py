import numpy as np
import pandas as pd

from recordlinkage.indexing import BlockIndex

def ranking1d(x):
    x_by_rank = np.sort(np.unique(x))
    result = np.searchsorted(x_by_rank, x).astype(float)
    result[np.isnan(x)] = np.nan
    return result

class AdjacentBlockIndex(BlockIndex):
    '''
    :class:`recordlinkage.indexing.BlockIndex` that 
    includes matches with elements in adjacent blocks
    and null values
    '''
    def __init__(self, *args, max_nulls=0, ndx_sorting_keys=slice(None), **kwargs):
        super().__init__(*args, **kwargs)
        self.max_nulls = max_nulls
        self.ndx_sorting_keys = slice(0,0) if ndx_sorting_keys is None else ndx_sorting_keys
    def _key_dfs(self, *dfs):
        '''
        Normalized dataframes.  Properties:
        * same indices as originals
        * columns restricted to blocking keys
        * Non-null values of blocking keys replaced with integer ranks
        * column names normalized
        Returns:
            list: key dfs
        '''
        assert 1 <= len(dfs) <= 2
        default_on = [c for c in dfs[0].columns if all(c in df.columns for df in dfs)]
        raw_key_columns = [side_on or self.on or default_on for side_on in [self.left_on, self.right_on]]
        if not (all(raw_key_columns) and (len(raw_key_columns[0]) == len(raw_key_columns[1]))):
            raise IndexError('Invalid blocking keys')
        key_dfs = [df[side_on].copy() for df, side_on in zip(dfs, raw_key_columns)]
        key_columns = ['k_{ndx}'.format(**locals()) for ndx in range(len(key_dfs[0].columns))]
        for df in key_dfs:
            df.columns = key_columns
        for col, vals in pd.concat(key_dfs, ignore_index=True).iteritems():
            unique_vals = vals.unique()
            key_ranks = pd.Series(ranking1d(unique_vals), index=unique_vals)
            for key_df in key_dfs:
                key_df[col] = key_ranks[key_df[col].values].values
        return key_dfs
    def _inclusive_key_df_link_index(self, *key_dfs):
        '''
        link index for dataframe(s) of the form produced by _key_dfs, including "matches" of a record to itself
        Args:
            *key_dfs: 1 or 2 dataframes of the form produced by _key_dfs
        Returns:
            :class:`pandas.MultiIndex`: 
        '''
        assert (1 <= len(key_dfs) <= 2) and all(tuple(df.columns) == tuple(key_dfs[0].columns) for df in key_dfs)
        blocks = pd.concat(key_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
        blocks = blocks[blocks.isnull().sum(axis='columns') <= self.max_nulls].reset_index(drop=True)
        df_unique_block_ids = [blocks.index.values] if len(key_dfs) == 1 else [key_df.merge(blocks.reset_index(), on=list(key_df.columns))['index'].unique() for key_df in key_dfs]
        if blocks.max().max() in [0, np.nan]:
            block_pair_candidates = pd.MultiIndex.from_product([df_unique_block_ids[0], df_unique_block_ids[-1]])
        else:
            coarsened_blocks = np.floor(blocks.copy() / 2)
            block_pair_candidates = self._inclusive_key_df_link_index(*[coarsened_blocks.loc[block_ids, :] for block_ids in df_unique_block_ids])
        rank_difference_components = [blocks.loc[block_pair_candidates.get_level_values(lvl),:].reset_index(drop=True) for lvl in range(2)]
        rank_differences = rank_difference_components[0].copy() - rank_difference_components[1]
        rank_differences.index = block_pair_candidates
        del rank_difference_components
        def key_subset(sorting):
            all_keys = key_dfs[0].columns
            sorting_keys = list(all_keys[self.ndx_sorting_keys])
            return sorting_keys if sorting else [c for c in all_keys if c not in sorting_keys]
        null_count_ok = rank_differences.isnull().sum(axis='columns') <= self.max_nulls
        sorting_keys_match = not(key_subset(sorting=True)) or rank_differences[key_subset(sorting=True)].abs().max(axis='columns').isin([0, 1, np.nan])
        non_sorting_keys_match = not(key_subset(sorting=False)) or rank_differences[key_subset(sorting=False)].abs().max(axis='columns').isin([0, np.nan])
        block_link = rank_differences[null_count_ok & sorting_keys_match & non_sorting_keys_match].index.to_frame().reset_index(drop=True)
        block_link.columns = ['block_id_a', 'block_id_b']
        block_id_lookup = blocks.reset_index(drop=False).set_index(list(blocks.columns)).iloc[:,0]
        block_id_lookup.name = 'block_id'
        block_record_id_mappings = [key_df.join(block_id_lookup, on=block_id_lookup.index.names)[[block_id_lookup.name]].reset_index() for key_df in key_dfs]
        return block_record_id_mappings[0].rename(columns={'index':'index_a'}).merge(block_link, left_on='block_id', right_on='block_id_a')[['index_a', 'block_id_b']].merge(block_record_id_mappings[-1].rename(columns={'index':'index_b'}), left_on='block_id_b', right_on='block_id').set_index(['index_a', 'index_b']).index
    def _dedup_index(self, df_a):
        result = self._inclusive_key_df_link_index(*self._key_dfs(df_a))
        result = result[result.get_level_values(0) < result.get_level_values(1)]
        return result
    def _link_index(self, df_a, df_b):
        return self._inclusive_key_df_link_index(*self._key_dfs(df_a, df_b))