'''
============================================
indexing - Initial selection of record pairs
============================================

Members
-------
'''
from collections import OrderedDict
from distutils.version import LooseVersion
from functools import reduce, wraps
from itertools import chain
import numpy as np
import pandas as pd
from scipy import spatial



if LooseVersion(np.__version__) < LooseVersion('1.13'):
    # monkey patch for isin function introduced in version 1.13
    def isin(element, test_elements, assume_unique=False, invert=False):
        '''
        implementation of :func:`numpy.isin` (included from version 1.13)
        '''
        return np.in1d(element.flatten(), test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)
    np.isin = isin
    del isin

def orthogonal_unit_vector(points):
    '''
    Unit vector orthogonal to the (unique) hyperplane passing through *points*
    
    Args:
        points (array-like): one or more groups of points, each group 
        specifying a hyperplane.  
                
    Returns:
        :class:`numpy.ndarray`: orthogonal vectors - same shape as *points* except that second-to-last dimension is omitted

    The dimensions of *points* must follow this pattern:
    
    * optional dimensions before the final two: `matrix stacking <https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.linalg.html#linear-algebra-on-several-matrices-at-once>`_
    * second-to-last dimension: points
    * final dimension: vector space ordinates
    
    The number of points to specify a hyperplane must be the same as the 
    number of dimensions in the space (ie: the last two axes of *points*)
    must have the same length.

    
    Example:
        >>> import numpy as np
        >>> points = np.array([[1,2], [4, 6]])
        >>> orthogonal_vector = orthogonal_unit_vector(points)
        >>> orthogonal_vector
        array([-0.8,  0.6])
        >>> np.allclose((points - points[0]) @ orthogonal_vector, 0)
        True
    '''
    points = np.array(points)
    example_point = points[...,0,:]
    offsets = points[...,1:,:] - example_point[..., np.newaxis, :]
    n_columns = offsets.shape[-1]
    all_colnums = np.arange(n_columns)
    subdeterminants = np.array([np.linalg.det(offsets[...,all_colnums != col_num]) for col_num in range(n_columns)])
    subdet_axis_numbers = list(range(len(subdeterminants.shape)))
    subdet_axis_numbers.append(subdet_axis_numbers.pop(0))
    subdeterminants = subdeterminants.transpose(subdet_axis_numbers)
    signs = (2 *(np.arange(n_columns) % 2) - 1)
    signs = signs.reshape((1,) * (len(subdeterminants.shape) - 2) + (n_columns,))
    result = subdeterminants * signs
    result /= np.linalg.norm(result, axis=-1)[...,np.newaxis]
    return result


class ExtConvexHull(spatial.ConvexHull):
    '''
    :class:`scipy.spatial.ConvexHull` with some additional methods
    '''
    @wraps(spatial.ConvexHull.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._standard_attrs = set(self.__dict__)
        self._cache = {}
    def minimizing_points(self):
        '''
        vertices of :meth:`minimizing_simplices`
        '''
        return np.unique(self.minimizing_simplices().flatten())
    def minimizing_simplices(self):
        '''
        Values from :meth:`scipy.spatial.ConvexHull.simplices` comprising only
        points in :meth:`minimizing_subhull_points`
        '''
        simplices = self.simplices
        cache = self._cache.setdefault('_minimizing_simplices', {})
        if not (cache and np.array_equal(simplices, cache['simplices'])):
            flattened_simplex_points = self.points[np.unique(simplices.flatten())]
            corner = np.min(flattened_simplex_points, axis=0)
            centroid = np.mean(flattened_simplex_points, axis=0)
            simplex_points = self.points[simplices] 
            simplex_orthogonal_vectors = orthogonal_unit_vector(simplex_points)
            simplex_example_points = simplex_points[:,0,:]
            def simplex_orthogonal_distances(point):
                return np.sum((point[np.newaxis, ...] - simplex_example_points) * simplex_orthogonal_vectors, axis=1)
            minimizing_simplices = simplices[simplex_orthogonal_distances(centroid) * simplex_orthogonal_distances(corner) < 0]
            cache.update({'simplices':simplices, 'minimizing_simplices':minimizing_simplices})
        return cache['minimizing_simplices']
    def __reduce__(self):
        return (type(self), # constructor
                (self.points, hasattr(self._qhull, 'add_points')), #args (points, incremental)
                {k:v for k, v in vars(self).items() if k not in self._standard_attrs},  # state
                )
    

class IndexCombiner:
    def __init__(self, component_indices, true_index, cost_perturbation=1):
        self.component_indices = dict(component_indices)
        self.true_index = true_index
        self.cost_perturbation = cost_perturbation
    def index_union(self, keys):
        return reduce(lambda a, b: a|b, map(self.component_indices.__getitem__, keys))
    def cost_hull(self):
        cache_attr = '_cost_hull'
        if not hasattr(self, cache_attr):
            all_keys = set(self.component_indices)
            def cost_point(keys):
                pred_index = self.index_union(keys)
                agreement_index = self.true_index & pred_index
                return pd.Series(OrderedDict([('false_omissions', len(self.true_index) - len(agreement_index)),
                                              ('false_inclusions', len(pred_index) - len(agreement_index))]))
            cost_by_seed_key_set = {ks: cost_point(ks) for ks in (frozenset([k]) for k in all_keys)}
            example_cost = next(iter(cost_by_seed_key_set.values()))
            result = ExtConvexHull(list(chain([s.values for s in cost_by_seed_key_set.values()], [example_cost.values] + self.cost_perturbation*np.eye(len(example_cost)))), incremental=True)
            result.cost_point_labels = example_cost.index.values
            def result_point_index(point):
                return np.argmax(np.all(result.points == [point], axis=1))
            result.key_set_points = {k: result_point_index(point.values) for k, point in cost_by_seed_key_set.items()}
            def changes_minimizing_points(new_keys):
                new_keys = frozenset(new_keys)
                if new_keys in result.key_set_points:
                    return False
                costs = cost_point(new_keys).values
                if costs in result.points:
                    result.key_set_points[new_keys] = result_point_index(costs)
                    return False
                else:
                    point_ndx = result.key_set_points[new_keys] = len(result.points)
                    result.add_points([costs])
                    return (point_ndx in result.minimizing_points())
            def gross_combined_length(keys):
                return sum(len(self.component_indices[k]) for k in keys)
            def check_key_set(keys, skip_first=False):
                keys = frozenset(keys)
                if skip_first or changes_minimizing_points(keys):
                    own_point_index = result.key_set_points[keys]
                    perturbed_key_sets = chain((keys|{k} for k in (all_keys-keys)),
                                               (keys - {k} for k in keys) if len(keys) > 1 else [])
                    for perturbed_key_set in sorted(perturbed_key_sets, key=gross_combined_length):
                        if own_point_index not in result.minimizing_points():
                            break
                        check_key_set(perturbed_key_set)
            for ks in cost_by_seed_key_set:
                check_key_set(ks, skip_first=True)
            setattr(self, cache_attr, result)
        return getattr(self, cache_attr)
    def summary(self):
        hull = self.cost_hull()
        key_sets = list(hull.key_set_points)
        result = pd.DataFrame(hull.points[[hull.key_set_points[ks] for ks in key_sets]], columns=hull.cost_point_labels)
        result.index = [', '.join(sorted(ks)) for ks in key_sets]
        result.index.name = 'key_set_str'
        result['key_set'] = [set(ks) for ks in key_sets]
        minimizing_points = set(hull.minimizing_points())
        result['cost_minimizing'] = [hull.key_set_points[ks] in minimizing_points for ks in key_sets]
        result.sort_values(['cost_minimizing'] + hull.cost_point_labels.tolist(), ascending=False, inplace=True)
        return result 

def block_index_lengths(x, x_link=None):
    '''
    Blocking index lengths for :class:`pandas.DataFrame` columns.
    
    Args:
        x (:class:`pandas.DataFrame`): Table to be linked (if *x_link* is specified) or deduplicated (if *x_link* is missing)
        x_link (:class:`pandas.DataFrame`): Optional second table 
    Returns:
        :class:`pandas.Series`: index lengths by column name
    
    The blocking index types and columns used are summarized in the table below.
    
    +---------------------+---------------+--------------------------------+
    | Parameters supplied | blocking type | columns used                   |
    +=====================+===============+================================+
    | *x* only            | deduplication | all                            |
    +---------------------+---------------+--------------------------------+
    | *x* and *x_link*    | linkage       | all common to *x* and *x_link* |
    +---------------------+---------------+--------------------------------+
    
    Example:
        >>> import pandas as pd
        >>> from recordlinkage import BlockIndex
        >>> my_data = pd.DataFrame({'a': [1,2,1,2], 'b':[3,3,3,2]})
        >>> block_index_lengths(my_data)
        column
        a    2
        b    3
        dtype: int64
        >>> len(BlockIndex(on='a').index(my_data))
        2
        >>> len(BlockIndex(on='b').index(my_data))
        3
    '''
    dedup = (x_link is None)
    dfs = [x] + ([] if dedup else [x_link])
    result = pd.Series()
    result.index.name = 'column'
    for column in sorted(x.columns if dedup else (set(x.columns) & set(x_link.columns))):
        value_counts = [df[column].value_counts() for df in dfs]
        result[column] = ((value_counts[0] * (value_counts[0]-1) / 2) if dedup else (value_counts[0] * value_counts[1])).sum()
    result = result.astype(int)
    return result


#def stepwise_selected_subset(elements, scorer, backwards=False, size_bound=None):
#    '''
#    Stepwise subset selection
#    
#    Select a subset of *elements* attempting to maximize:
#        *scorer* (\<subset of elements\>)
#    
#    Args:
#        elements (iterable): elements available
#        scorer (callable): Callable which takes a single argument (set of values from *elements*) and returns a value to be maximized
#        backwards (bool): If True: start with all *elements* and incrementally remove them (otherwise, start with zero or one element and increase)
#        size_bound (int): Limit on size of set to return (either upper or lower bound depending on *backwards*)
#    Returns:
#        set or None: elements selected (or None if *min_score* not exceeded)
#    '''
#    elements = set(elements)
#    if size_bound is None:
#        size_bound = 0 if backwards else len(elements)
#    def _unchanged_or_improved_subset(initial_subset, initial_score=None):
#        if len(initial_subset) == size_bound:
#            return initial_subset
#        if initial_score is None:
#            return _unchanged_or_improved_subset(initial_subset, scorer(initial_subset))
#        def _adj_subset(elem):
#            return initial_subset - {elem} if backwards else initial_subset | {elem}
#        possible_adjustments = initial_subset if backwards else (elements - initial_subset)
#        scores = pd.Series({elem: scorer(_adj_subset(elem)) for elem in possible_adjustments})
#        max_score = scores.max()
#        if max_score > initial_score:
#            return _unchanged_or_improved_subset(_adj_subset(scores.argmax()), max_score)
#        else:
#            return initial_subset
#    return _unchanged_or_improved_subset(elements if backwards else set())

if __name__ == '__main__':
    import doctest
    doctest.testmod()
