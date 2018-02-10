
# coding: utf-8

# # Index method scalability comparison
# 
# This notebook compares the scalability properties of NeighbourhoodBlockIndex with the BlockIndex and SortedNeighbourhoodIndex (both in the recordlinkage package).  
# 
# 
# 
# Note: it should be run with memory paging disabled.  To do this on Linux, use:
# 
# ```
# sudo watch --interval 500 swapoff -a
# ```
# 
# Also, the kernel frequently gets killed, but the test function can continue from save points it makes as it goes.  A good way of restarting automatically (on Linux) is to export the notebook to a .py file
# (say, scalability.py) and then run:
# 
# ```
# while [ ! -f timings.pickle ]; do python scalability.py ; done
# ```
# 

# In[ ]:


#from collections import ChainMap
from functools import lru_cache, reduce
from itertools import product
from operator import mul
import datetime, inspect, pathlib
import numpy as np
import pandas as pd


from recordlinkage import FullIndex, BlockIndex, SortedNeighbourhoodIndex
from neighbourhood_blocking import NeighbourhoodBlockIndex

from experiment_helpers import sample_dedup_dataset, int_geomspace


# ## Testing function

# In[ ]:


def get_index_timings(index_details, row_counts, column_counts, distinct_entity_instance_counts, granularities, index_length_limit=5e6, index_time_limit=1, result_file=None, save_interval_seconds=60, verbose=False, debug=False, continue_based_on_count=False):
    '''
    Negative granularities are neighbourhood radii
    '''
    if result_file is not None:
        result_file = pathlib.Path(result_file).resolve()
        partial_result_file = result_file.parent.joinpath('_{}'.format(result_file.name))
        latest_keys_file = partial_result_file.parent.joinpath('_{}'.format(partial_result_file.name))
    timings = pd.read_pickle(str(partial_result_file)) if (result_file is not None) and partial_result_file.exists() else pd.DataFrame()
    failed_keys = pd.read_pickle(str(latest_keys_file)) if (result_file is not None) and latest_keys_file.exists() else {}
    if continue_based_on_count and (len(timings)>0) and ('combinations_processed' in timings.columns):
        skip_to_past_combination_number = timings['combinations_processed'].max()
        if pd.isnull(skip_to_past_combination_number):
            skip_to_past_combination_number = -1
    else:
        skip_to_past_combination_number = -1
    last_save_time = datetime.datetime.now()
    def msg(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    def debug_msg(*args, **kwargs):
        if debug:
            print(*args, **kwargs)
    total_combinations = reduce(mul, map(len, [index_details, row_counts, column_counts, distinct_entity_instance_counts, granularities]), 1)
    combinations_processed = progress = 0
    id_cols = ['instances_per_entity', 'granularity', 'rows', 'columns', 'label']
    for instances_per_entity in sorted(distinct_entity_instance_counts):
        msg('Instances per entity: {instances_per_entity}'.format(**locals()))
        @lru_cache(maxsize=1)
        def largest_continuous_dataset():
            msg('Computing continuous dataset')
            return sample_dedup_dataset(rows=max(row_counts), columns=max(column_counts), instances_per_entity=instances_per_entity)
        for granularity in sorted(granularities, reverse=True):
            msg('{progress:.0%}\tGranularity: {granularity}'.format(**locals()))
            @lru_cache(maxsize=1)
            def largest_discrete_dataset():
                msg('\tDiscretizing dataset')
                return np.floor(largest_continuous_dataset() * granularity)
            for rows, columns in product(sorted(row_counts), sorted(column_counts)):
                msg('{progress:.0%}\t\tTable: {rows} x {columns}'.format(**locals()))
                @lru_cache(maxsize=2)
                def dataset(dataset_type):
                    largest_dataset = {'continuous': largest_continuous_dataset, 'discrete':largest_discrete_dataset}[dataset_type]
                    return largest_dataset().iloc[:rows, :columns]
                length_full = rows * (rows-1) // 2
                for label, index_type, index_kwargs in index_details:
                    index_kwargs = dict(index_kwargs)
                    combinations_processed += 1
                    if combinations_processed <= skip_to_past_combination_number:
                        continue
                    progress = combinations_processed / total_combinations
                    _locals = locals()
                    current_keys = {k:_locals[k] for k in id_cols}
                    def omission_contitions():
                        debug_msg('\t\t\t\t{granularity} inapplicable to {label}?'.format(**locals()))
                        yield ((index_type is not SortedNeighbourhoodIndex) and (granularity < 0))
                        debug_msg('\t\t\t\t{granularity} granularity exceeds {rows} rows?'.format(**locals()))
                        yield granularity > rows
                        debug_msg('\t\t\t\t{columns} Too many columns?'.format(**locals()))
                        yield ((index_type is SortedNeighbourhoodIndex) and (columns > 1))
                        debug_msg('\t\t\t\tmax nulls too high?'.format(**locals()))
                        yield index_kwargs.get('max_nulls', columns) > columns
                        if len(timings) > 0:
                            debug_msg('\t\t\t\tthis config already done?'.format(**locals()))
                            yield reduce((lambda a,b: a&b), (timings[c]==_locals[c] for c in id_cols)).any()
                            if index_type is FullIndex:
                                debug_msg('\t\t\t\t{rows} rows already done for {label}?'.format(**locals()))
                                yield (timings[timings['label']==label]['rows'] == rows).any()
                            if index_type is SortedNeighbourhoodIndex:
                                debug_msg('\t\t\t\t{rows} rows {granularity} granularity already done for {label}?'.format(**locals()))
                                yield ((timings[timings['label']==label]['rows'] == rows) & (timings[timings['label']==label]['granularity'] == granularity)).any()
                            comparison_rows = reduce((lambda a,b: a&b), 
                                                     (timings[c]==_locals[c] for c in id_cols if c != 'rows'),
                                                     timings['rows'] <= rows,
                                                    )
                            if comparison_rows.any():
                                comparison_maxima = timings[comparison_rows][['elapsed_seconds', 'index_length', 'memory_overflow']].max()
                                debug_msg('\t\t\t\tsmaller index produced memory overflow?')
                                yield comparison_maxima['memory_overflow'] 
                                if index_time_limit is not None:
                                    debug_msg('\t\t\t\tsmaller index already exceeds time limit?')
                                    yield comparison_maxima['elapsed_seconds'] > index_time_limit
                                if index_length_limit is not None:
                                    debug_msg('\t\t\t\tsmaller index already exceeds length limit?')
                                    yield comparison_maxima['index_length'] > index_length_limit
                    if any(omission_contitions()):
                        debug_msg ('\t\t\t\t\tskipping {label}'.format(**locals()))
                        continue
                    msg('{progress:.0%}\t\t\t{label}'.format(**locals()))
                    stats_dict = {k: _locals[k] for k in (['length_full', 'combinations_processed'] + id_cols)}
                    stats_dict.update(index_kwargs)
                    if index_type is SortedNeighbourhoodIndex:
                        dataset_type = 'continuous'
                        index_kwargs['window'] = 1 + 2 * (int(rows / granularity / 2) if granularity > 0 else -granularity)
                    else:
                        dataset_type = 'discrete'
                        stats_dict.update({k: _locals[k] for k in ['granularity']})
                    if 'on' in inspect.signature(index_type).parameters:
                        index_kwargs['on'] = list(dataset(dataset_type).columns)
                    indexer = index_type(**index_kwargs)
                    memory_overflow = (current_keys == failed_keys)
                    elapsed_seconds = index_length = np.nan
                    if not memory_overflow:
                        if result_file is not None:
                            pd.to_pickle(current_keys, str(latest_keys_file))
                        try:
                            start_time = datetime.datetime.now()
                            index_length = len(indexer.index(dataset(dataset_type)))
                            elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
                        except MemoryError:
                            memory_overflow = True
                    stats_dict.update({'index_length': index_length,
                                       'elapsed_seconds': elapsed_seconds,
                                       'memory_overflow': memory_overflow,
                                      })
                    timings = timings.append(stats_dict, ignore_index=True)
                    if (result_file is not None) and (memory_overflow or ((datetime.datetime.now() - last_save_time).total_seconds() > save_interval_seconds)):
                        timings.to_pickle(str(partial_result_file))
                        last_save_time = datetime.datetime.now()
    timings['reduction_ratio'] = 1 - timings['index_length'] / timings['length_full']
    if result_file is not None:
        timings.to_pickle(str(partial_result_file))
        partial_result_file.rename(result_file)
    progress = 1
    msg('{progress:.0%}\t Done'.format(**locals()))
    return timings


# ## Execution

# In[ ]:


index_details = [#(label, type, additional kwargs),
                 ('Full', FullIndex, {}),
                 ('Standard Blocking',BlockIndex , {}),
                 ('Sorted Neighbourhood', SortedNeighbourhoodIndex, {}),
                 ('Neighbourhood Blocking - Standard Blocking settings',NeighbourhoodBlockIndex , {'max_nulls': 0, 'max_rank_differences':0}),
                 ('Neighbourhood Blocking - no wildcards',NeighbourhoodBlockIndex , {'max_nulls': 0}),
                 ('Neighbourhood Blocking - 1 wildcard',NeighbourhoodBlockIndex , {'max_nulls': 1}),
                 ('Neighbourhood Blocking - 2 wildcards',NeighbourhoodBlockIndex , {'max_nulls': 2}),
                ]

timings = get_index_timings(index_details=index_details,
                        row_counts = int_geomspace(start=10, stop=1000000, num=16),
                        column_counts = 1 + np.arange(10),
                        distinct_entity_instance_counts = int_geomspace(start=1, stop=100, num=5),
                        granularities = (-int_geomspace(start=1, stop=100, num=10)).tolist() + int_geomspace(start=1, stop=1000, num=10).tolist(),
                        index_length_limit = 10e6,
                        index_time_limit = 30,
                        result_file = pathlib.Path('timings.pickle'),
                        verbose = True,
                        debug = False,
                        continue_based_on_count=True,
                        )


# In[ ]:




