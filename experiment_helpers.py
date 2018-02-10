import numpy as np
import pandas as pd
from recordlinkage import BlockIndex

def int_geomspace(start, stop, num=50, endpoint=True):
    return np.unique(np.logspace(np.log10(start), np.log10(stop), num=num, endpoint=endpoint, dtype=int))

def sample_dedup_dataset(rows=1000, columns=5, instances_per_entity=2, noise_sd=0.01, null_proportion=0.01, include_true_matches=False):
    np.random.seed(1)
    n_entities = (rows - 1) // instances_per_entity + 1
    data = np.random.rand(n_entities, columns, 1) + noise_sd * np.random.randn(n_entities, columns, instances_per_entity)
    data = data.transpose([0,2,1]).flatten()[:rows * columns]
    data[np.random.choice(np.arange(len(data)), int(null_proportion * len(data)))] = np.nan
    data = data.reshape(rows, columns)
    dataset = pd.DataFrame(data = data, 
                           columns = np.core.defchararray.add('c_', np.arange(columns).astype(str)), 
                           index = np.core.defchararray.add('r_', np.arange(rows).astype(str)))
    if include_true_matches:
        true_matches = BlockIndex(on='entity_id').index(pd.DataFrame({'entity_id': np.repeat(np.arange(n_entities), instances_per_entity)[:rows]}, index=dataset.index))
        return (dataset, true_matches)
    else:
        return dataset
    
