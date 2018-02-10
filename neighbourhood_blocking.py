import numpy as np
import pandas as pd

from recordlinkage.indexing import BlockIndex

def ranking1d(x):
    x_by_rank = np.sort(np.unique(x))
    result = np.searchsorted(x_by_rank, x).astype(float)
    return result

class NeighbourhoodBlockIndex(BlockIndex):
    '''
    :class:`recordlinkage.indexing.BlockIndex` that 
    includes matches with elements in adjacent blocks
    and null values
    '''
    def __init__(self, *args, max_nulls=0, max_non_matches=0, max_rank_differences=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_nulls = max_nulls
        self.max_non_matches = max_non_matches
        self.max_rank_differences = list(max_rank_differences) if hasattr(max_rank_differences, '__iter__') else [max_rank_differences]
    def _get_max_rank_differences(self, key_df):
        n_cols = len(key_df.columns)
        diffs = list(self.max_rank_differences)
        if len(diffs) < n_cols:
            diffs += diffs[-1:] * (n_cols - len(diffs))
        return np.array(diffs[:n_cols])
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
            unique_values = vals.dropna().unique()
            key_ranks = pd.Series(ranking1d(unique_values), index=unique_values)
            for key_df in key_dfs:
                key_rank_values = np.full(shape=len(key_df[col]), fill_value=np.nan)
                ndx_not_null = (~key_df[col].isnull()).values
                key_rank_values[ndx_not_null] = key_ranks[key_df[col].values[ndx_not_null]].values
                key_df[col] = key_rank_values
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
        df_unique_block_ids = [blocks.index.values] if len(key_dfs) == 1 else [key_df.merge(blocks.reset_index(), on=list(key_df.columns))['index'].unique() for key_df in key_dfs]
        if blocks.max().max() in [0, np.nan]:
            block_pair_candidates = pd.MultiIndex.from_product([df_unique_block_ids[0], df_unique_block_ids[-1]])
        else:
            coarsened_blocks = np.floor(blocks.copy() / 2)
            block_pair_candidates = self._inclusive_key_df_link_index(*[coarsened_blocks.loc[block_ids, :] for block_ids in df_unique_block_ids])
        rank_difference_components = [blocks.loc[block_pair_candidates.get_level_values(lvl),:].reset_index(drop=True) for lvl in range(2)]
        abs_rank_difference_excesses = (rank_difference_components[0] - rank_difference_components[1]).abs()
        del rank_difference_components
        abs_rank_difference_excesses.index = block_pair_candidates
        for (col, vals), max_diff in zip(abs_rank_difference_excesses.iteritems(), self._get_max_rank_differences(blocks)):
            abs_rank_difference_excesses[col] = vals - max_diff
        column_match_counts = pd.DataFrame({'wildcard': abs_rank_difference_excesses.isnull().sum(axis='columns')})
        column_match_counts['rank'] = (abs_rank_difference_excesses <= 0).sum(axis='columns')
        n_column_matches_ok = (column_match_counts['rank'] + np.clip(column_match_counts['wildcard'], None, self.max_nulls)) >= (len(abs_rank_difference_excesses.columns) - self.max_non_matches)
        block_link = abs_rank_difference_excesses[n_column_matches_ok].index.to_frame().reset_index(drop=True)
        block_link.columns = ['block_id_a', 'block_id_b']
        def get_block_id_lookup():
            df = blocks.reset_index(drop=False)
            df = df.append(df.max().fillna(0) + 1, ignore_index=True)
            result = df.set_index(list(blocks.columns)).iloc[:,0].iloc[:-1]
            result.name = 'block_id'
            return result
        block_id_lookup = get_block_id_lookup()
        def get_block_record_id_mapping(key_df):
            result = key_df.join(block_id_lookup, on=block_id_lookup.index.names)[[block_id_lookup.name]]
            result.index.name = 'index'
            return result.reset_index()
        block_record_id_mappings = [get_block_record_id_mapping(key_df) for key_df in key_dfs]
        return block_record_id_mappings[0].rename(columns={'index':'index_a'}).merge(block_link, left_on='block_id', right_on='block_id_a')[['index_a', 'block_id_b']].merge(block_record_id_mappings[-1].rename(columns={'index':'index_b'}), left_on='block_id_b', right_on='block_id').set_index(['index_a', 'index_b']).index
    def _dedup_index(self, df_a):
        result = self._inclusive_key_df_link_index(*self._key_dfs(df_a))
        result = result[result.get_level_values(0) < result.get_level_values(1)]
        return result
    def _link_index(self, df_a, df_b):
        return self._inclusive_key_df_link_index(*self._key_dfs(df_a, df_b))