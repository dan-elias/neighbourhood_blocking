'''
=====================================================
evaluation - Statistics on record linkage performance
=====================================================

Members
-------
'''

from functools import lru_cache
import numpy as np


#def record_pair_count(tbl1, tbl2=None):
#    '''
#    Number of possible record pairs
#    
#    Args:
#        tbl1, tbl2 (:class:`collections.abc.Sized`): Collections of records (eg: :class:`pandas.DataFrame`)
#    Returns:
#        int: Number of possible pairs for either record linkage or deduplication
#            (depending on whether or not tbl2 is supplied)
#    '''
#    n1 = len(tbl1)
#    if tbl2 is None:
#        return n1 * (n1 - 1) // 2
#    else:
#        n2 = len(tbl2)
#        return n1 * n2

class LinkageComparison:
    '''
    Linkage performance statistics
    
    Attributes:
        stats (dict): comparison statistics:
            
            * count_true (int): Number of true matches
            * count_predicted (int): Number of predicted matches
            * count_predicted_correct (int): Number of matches in intersection of true and predicted
            * sensitivity (real): Proportion of true matches correctly predicted
            * precision (real): Proportion of predictd matches that were correct
    '''
    def __init__(self, ndx_true, ndx_predicted, description=None, length_1=None, length_2=None):
        '''
        Args:
            ndx_true (:class:`pandas.MultiIndex` or :class:`set`): \True matches
            ndx_predicted (:class:`pandas.MultiIndex` or :class:`set`): Predicted matches
        '''
        self.ndx_true = ndx_true
        self.ndx_predicted = ndx_predicted
        self.description = description
        if length_1:
            self.count_total = (length_1 * length_2) if length_2 else (length_1 * (length_1 - 1) // 2)
        else:
            self.count_total = None
    @property
    @lru_cache(maxsize=1)
    def stats(self):
        result = {'count_true': len(self.ndx_true),
                  'count_predicted': len(self.ndx_predicted),
                  'count_agreement': len(self.ndx_true & self.ndx_predicted)}
        result['sensitivity'] = result['count_agreement'] / (result['count_true'] or np.nan)
        result['precision'] = result['count_agreement'] / (result['count_predicted'] or np.nan)
        if self.description:
            result['description'] = self.description  
        if self.count_total:
            result['count_total'] = self.count_total
            result['predicted_proportion'] = result['count_predicted'] / result['count_total']
            result['false_positive_rate'] = (result['count_predicted'] - result['count_agreement']) / (result['count_total'] - result['count_true'])
        return result
        
