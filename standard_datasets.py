
# coding: utf-8

# # Indexing tests on standard datasets

# In[1]:


from collections import OrderedDict
from contextlib import suppress
from itertools import product, islice, chain, combinations
from functools import reduce
import datetime, pathlib, sys

import numpy as np
import pandas as pd

import recordlinkage as rl

import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
#import ipywidgets as ipw


sys.path.insert(0, '../../src')
from dataset_defs import get_preprocessed_datasets as get_datasets
from experiment_helpers import int_geomspace
from neighbourhood_blocking import NeighbourhoodBlockIndex

#sns.set_style()
#%matplotlib inline


# ## Data

# In[2]:


def shortened_dataset_contents(dataset, target_size=1000):
    result = {'df1': dataset['df1'].iloc[:target_size, :]}
    orig_true_mapping = dataset['true_mapping']
    true_mapping_flags = np.in1d(orig_true_mapping.get_level_values(0), result['df1'].index.values)
    if 'df2' in dataset:
        orig_df2 = dataset['df2']
        ndx_df2 = np.in1d(orig_df2.index.values, orig_true_mapping.get_level_values(1)[true_mapping_flags])
        size2 = np.sum(ndx_df2)
        if size2 < target_size <= len(orig_df2):
            ndx_df2[np.arange(len(ndx_df2))[~ndx_df2][:target_size - size2]] = True
        result['df2'] = orig_df2.loc[ndx_df2, :]
    else:
        true_mapping_flags &= np.in1d(orig_true_mapping.get_level_values(1), result['df1'].index.values)
    result['true_mapping'] = orig_true_mapping[true_mapping_flags]
    return result

datasets = get_datasets()
#datasets = {k: shortened_dataset_contents(ds, target_size=100) for k, ds in datasets.items()}


# ## Indexing

# In[3]:


def show_sample_matches(dataset):
    ndx = np.array(dataset['true_mapping'].values[:3].tolist())
    if 'df2' in dataset:
        raise NotImplementedError
    else:
        display.display(dataset['df1'].loc[ndx.flatten(),:])
    
show_sample_matches(datasets['febrl2'])


# ## Helpers

# In[4]:


def rowdicts2DataFrame(rowdicts):
    rowdicts = list(rowdicts)
    columns = {k for row in rowdicts for k in row}
    return pd.DataFrame({c: [row.get(c, np.nan) for row in rowdicts] for c in columns}).sort_index(axis='columns')


# ## Indexing comparison

# In[5]:


from scipy.misc import comb

def rl_confusion_matrix(true_mapping, pred_mapping, *dfs):
    if len(dfs) == 1:
        full_index_size = len(dfs[0]) * (len(dfs[0]) - 1) // 2
        def normalized_dedup_index(mapping):
            if len(mapping) == 0:
                return mapping
            index_names = ['x1', 'x2']
            mapping_values = np.array(mapping.values.tolist())
            directed_pairs = pd.DataFrame(np.vstack([mapping_values, mapping_values[:,::-1]]), columns=['x1', 'x2']).drop_duplicates()
            return directed_pairs[directed_pairs.iloc[:,0] < directed_pairs.iloc[:,1]].set_index(list(directed_pairs.columns)).index
        return rl.confusion_matrix(*[normalized_dedup_index(x) for x in [true_mapping, pred_mapping]], n_pairs=full_index_size)
    elif len(dfs) == 2:
        full_index_size = len(dfs[0]) * len(dfs[1])
        return rl.confusion_matrix(true_mapping, pred_mapping, n_pairs=full_index_size)
    else:
        raise ValueError('Invalid number of dataframes')

def distinct_random_combinations(a, size, p=None):
    used = set()
    np.random.seed(1)
    for _ in range(int(comb(len(a), size))):
        current_combination = np.random.choice(a, size=size, replace=False, p=p)
        prev_n_used = len(used)
        used.add(frozenset(current_combination))
        if len(used) > prev_n_used:
            yield np.sort(current_combination)
        

def dataset_index_stats(dataset_key, column_mapping=None, result_file=None, verbose=False, save_interval_seconds=60, max_elapsed_seconds=30, max_index_size=10e6):
    shared = {}
    if result_file is not None:
        result_file = pathlib.Path(result_file).resolve()
        partial_result_file = result_file.parent.joinpath('_{}'.format(result_file.name))
        latest_keys_file = partial_result_file.parent.joinpath('_{}'.format(partial_result_file.name))
    shared['stats'] = pd.read_pickle(str(partial_result_file)) if (result_file is not None) and partial_result_file.exists() else pd.DataFrame()
    failed_rowdict = pd.read_pickle(str(latest_keys_file)) if (result_file is not None) and latest_keys_file.exists() else {}
    shared['save_time'] = datetime.datetime.now()
    def msg(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    dataset = datasets[dataset_key]
    dfs = [dataset[k] for k in ['df1', 'df2'] if k in dataset]
    unique_counts_0 = dfs[0].apply(lambda s: len(s.unique()))
    unique_count_proportions_0 = (unique_counts_0 / unique_counts_0.sum()).values
    if (len(dfs) == 2) and (column_mapping is None):
        column_mapping = {k:k for k in {c for df in dfs for c in df.columns}}
    left_keys = sorted(c for c in dfs[0].columns if (column_mapping is None) or (c in column_mapping))
    if column_mapping is None:
        column_mapping = {k:k for k in left_keys}
    def get_on_kwargs(left_on):
        left_on = list(left_on)
        return {'on': left_on} if len(dfs) == 1 else {'left_on': left_on, 'right_on':[column_mapping[c] for c in left_on]}
    def add_stats_rowdict(indexer_type, force=False, **indexer_kwargs):
        rowdict = {'dataset': dataset_key,
                   'indexer_class': indexer_type.__name__,
                   'description': '{indexer_type.__name__}{kwargs_repr}'.format(kwargs_repr = ' ({})'.format(', '.join(map(str, (v for _, v in sorted(indexer_kwargs.items()))))) if indexer_kwargs else '', **locals()),
                  }
        def omission_conditions():
            if len(shared['stats']) > 0:
                # this config already done
                yield (set(rowdict) <= set(shared['stats'].columns)) and reduce((lambda a, b: a&b), (shared['stats'][c] == v for c, v in rowdict.items())).any()
                if all(kw in shared['stats'].columns for kw in indexer_kwargs):
                    comparable_failed = reduce(lambda a, b: a&b, (shared['stats'][c] == rowdict[c] for c in ['dataset', 'indexer_class']))                                        & reduce(lambda a, b: a|b,(shared['stats'][k]>v for k, v in [('elapsed_seconds',max_elapsed_seconds), ('index_size', max_index_size)]), shared['stats']['failed'])
                    for on_field in (set(indexer_kwargs) & {'on', 'left_on', 'right_on'} & set(shared['stats'].columns)):
                        comparable_failed &= shared['stats'][on_field].astype(str) == str(indexer_kwargs[on_field])
                    def shorter_index_failed(kwargs_and_directions):
                        def kw_comparison(tpl):
                            kw, direction = tpl
                            a, b = shared['stats'][kw], indexer_kwargs[kw]
                            return a<=b if (direction > 0) else a>=b
                        return reduce(lambda a, b: a&b, map(kw_comparison, kwargs_and_directions.items()), comparable_failed).any()
                    if indexer_type is NeighbourhoodBlockIndex:
                        yield shorter_index_failed(kwargs_and_directions = {'max_nulls': 1, 'max_rank_differences':1, 'max_non_matches':1})
                    elif indexer_type is rl.SortedNeighbourhoodIndex:
                        yield shorter_index_failed(kwargs_and_directions={'window':1})
        if (not force) and any(omission_conditions()):
            return
        rowdict.update(indexer_kwargs)
        failed = (rowdict == failed_rowdict)
        if not failed:
            if result_file is not None:
                pd.to_pickle(rowdict, str(latest_keys_file))
            indexer = indexer_type(**indexer_kwargs)
            start_time = datetime.datetime.now()
            try:
                index = indexer.index(*dfs)
            except MemoryError:
                failed = True
        rowdict['failed'] = failed
        if not failed:
            elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
            confusion_matrix = rl_confusion_matrix(dataset['true_mapping'], index, *dfs)
            rowdict.update({'elapsed_seconds': elapsed_seconds,
                            'index_size': len(index),
                            'reduction_ratio': rl.reduction_ratio(index, dfs),
                            'precision': rl.precision(confusion_matrix),
                            'recall': rl.recall(confusion_matrix),
                           })
        shared['stats'] = shared['stats'].append(pd.DataFrame({k:[v] for k, v in rowdict.items()}), ignore_index=True)
        if (result_file is not None) and (failed or ((datetime.datetime.now() - shared['save_time']).total_seconds() >= save_interval_seconds)):
            shared['stats'].to_pickle(str(partial_result_file))
            shared['save_time'] = datetime.datetime.now()
    #add_stats_rowdict(rl.FullIndex)
    for n_matches in reversed(range(1, len(left_keys)+1)):
        msg('{n_matches} match{es}'.format(es='' if n_matches==1 else 'es', **locals()))
        for left_on in islice(distinct_random_combinations(left_keys, size=n_matches), len(left_keys)):
            msg('\tStandard blocking on {left_on}'.format(**locals()))
            add_stats_rowdict(rl.BlockIndex, **get_on_kwargs(left_on))
        max_half_window = int(max(map(len, dfs))/4)
        half_windows = int_geomspace(1, max_half_window, 2 + max([0, int(np.log(max_half_window))]))
        for half_window in half_windows:
            msg('\tHalf window: {half_window}'.format(**locals()))
#            for max_nulls in range(0, n_matches//2+1, 3):
#                for left_on in islice(distinct_random_combinations(left_keys, size=n_matches), len(left_keys)):
#                    msg('\t\tNeighbourhood Blocking on {left_on},  max_nulls = {max_nulls}'.format(**locals()))
#                    add_stats_rowdict(NeighbourhoodBlockIndex,  max_nulls=max_nulls, max_rank_differences=half_window, **get_on_kwargs(left_on))
            if n_matches == 1:
                msg('\t\tSorted Neighbourhood'.format(**locals()))
                window = int(1 + 2 * half_window)
                for sort_key in left_keys:
                    add_stats_rowdict(rl.SortedNeighbourhoodIndex, window=window, **get_on_kwargs([sort_key]))
            if n_matches == 2:
                for left_on in map(list, combinations(left_keys, 2)):
                    msg('\t\tFields: {left_on}'.format(**locals()))
                    if half_window == half_windows[0]:
                        msg('\t\t\tStandard Blocking')
                        add_stats_rowdict(rl.BlockIndex, **get_on_kwargs(left_on))
                    for max_nulls, max_non_matches in product(range(1), range(2)):
                        msg('\t\tNeighbourhood Blocking on {left_on},  max_nulls = {max_nulls}, max_non_matches={max_non_matches}'.format(**locals()))
                        add_stats_rowdict(NeighbourhoodBlockIndex,  max_nulls=max_nulls, max_non_matches=max_non_matches, max_rank_differences=half_window, **get_on_kwargs(left_on))
    if result_file is not None:
        shared['stats'].to_pickle(str(partial_result_file))
        with suppress(FileNotFoundError):
            latest_keys_file.unlink()
        partial_result_file.rename(result_file)
    return shared['stats']

def calc_or_retrieve_dataset_index_stats(dataset_key, column_mapping=None, cache_path=None, force_overwrite=False, max_elapsed_seconds=30, verbose=False):
    def calculated_value(result_file=None):
        return dataset_index_stats(dataset_key=dataset_key, column_mapping=column_mapping, max_elapsed_seconds=max_elapsed_seconds, verbose=verbose, result_file=result_file)
    if cache_path is None:
        return calculated_value()
    else:
        cache_file = pathlib.Path(cache_path) / 'indexing_stats_{dataset_key}.pickle'.format(**locals())
        if force_overwrite or (not cache_file.exists()):
            return calculated_value(result_file=cache_file)

def plot_dataset_index_stats(indexing_stats):
    plot_kwargs = {'FullIndex': {'label': 'Full', 'marker': 's'},
                   'NeighbourhoodBlockIndex': {'label':'Neighbourhood Blocking', 'marker': '.'},
                   'BlockIndex': {'label': 'Standard Blocking', 'marker': 's', 'facecolor': 'none', 's':100},
                   'SortedNeighbourhoodIndex': {'label': 'Sorted Neighbourhood', 'marker': '+', 's':100},
                  }
    axis_vars = ['reduction_ratio', 'recall']
    plt.figure(figsize=(7,7))
    ax = plt.gca()
    for axis_name, axis_var in zip(['x', 'y'], axis_vars):
        getattr(ax, 'set_{axis_name}label'.format(**locals()))('{} (truncated)'.format(axis_var.replace('_', ' ').title()))
        getattr(ax, 'set_{axis_name}lim'.format(**locals()))(0.5, 1.05)
    for indexer_class, vals in indexing_stats.groupby('indexer_class'):
        ax.scatter(*[vals[col].values for col in axis_vars], color='grey', **plot_kwargs[indexer_class])
    plt.legend(loc='lower left', title='Index Type')
    plt.show()


# In[ ]:


def get_amazon_google_products_column_mapping():
    col_grps = [list(datasets['Amazon-GoogleProducts'][k].columns) for k in ['df1', 'df2']]
    result = {k:k.replace('title', 'name') for k in col_grps[0]}
    result = dict(tpl for tpl in result.items() if tpl[1] in col_grps[1])
    return result

for dataset_key, column_mapping in [
                                    ('febrl1', None),
                                    ('febrl2', None),
                                    ('febrl4', None),
                                    ('febrl3', None),
                                    ('DBLP-ACM', None),
                                    ('DBLP-Scholar', None),
                                    ('Abt-Buy', None),
                                    ('Amazon-GoogleProducts', get_amazon_google_products_column_mapping()),
                                   ]:
    index_stats = calc_or_retrieve_dataset_index_stats(dataset_key=dataset_key, 
                                                       cache_path='.', 
                                                       force_overwrite=False, 
                                                       column_mapping=column_mapping, 
                                                       max_elapsed_seconds=300,
                                                       verbose=True)
#    print(dataset_key)
#    plot_dataset_index_stats(index_stats)


# ## Scratch

# In[23]:



ds = datasets['Abt-Buy']
dfs = [ds[k] for k in ['df1', 'df2'] if k in ds]
#indexer = NeighbourhoodBlockIndex(max_non_matches=0)
#indexer = NeighbourhoodBlockIndex(max_non_matches=2)
#indexer = NeighbourhoodBlockIndex(on=list(dfs[0].columns)[1:][:3], max_non_matches=1)
#indexer = NeighbourhoodBlockIndex(on=['soc_sec_id', 'given_name', 'surname', 'date_of_birth_orig'], max_non_matches=2, max_rank_differences=2)
#indexer = NeighbourhoodBlockIndex(on=['soc_sec_id', 'date_of_birth_orig', 'address_2'], max_non_matches=2, max_rank_differences=2)
#indexer = NeighbourhoodBlockIndex(on=['soc_sec_id', 'postcode'], max_nulls=1, max_rank_differences=100)
indexer = NeighbourhoodBlockIndex(on=['name_orig', 'name_codes_codes'], max_nulls=0, max_rank_differences=[42, 5], max_non_matches=1)
#indexer = rl.SortedNeighbourhoodIndex(on=['soc_sec_id'], window=21)
start_time = datetime.datetime.now()
#ndx = indexer1.index(*dfs) | indexer2.index(*dfs)
ndx = indexer.index(*dfs)
elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
conf_mat = rl_confusion_matrix(ds['true_mapping'], ndx, *dfs)
precision = rl.precision(conf_mat)
recall = rl.recall(conf_mat)
index_size = len(ndx)
reduction_ratio = rl.reduction_ratio(ndx, dfs)
print('Elapsed: {elapsed_seconds:.0f}\tprecision: {precision:.0%}\trecall: {recall:.0%}\treduction: {reduction_ratio:.0%}\t size: {index_size}'.format(**locals()))


# In[22]:


dfs[0].head()


# In[ ]:




