'''
========================================================
classification - identification of matching record pairs
========================================================

Members
-------
'''
from collections import Counter
from functools import reduce, lru_cache
from itertools import chain
from operator import itemgetter, mul

import networkx as nx

class EntitySet:
    '''
    Resolved entities from id mapping
    '''
    def __init__(self, id_pairs, fungible_ids=False):
        '''
        Args:
            id_pairs (iterable): pairs of IDs that refer to the same entity 
            fungible (bool): IDs in first and second position are fungible (ie: from the same mapping)
        '''
        self._id_pairs = id_pairs
        self._fungible_ids = fungible_ids
    def _as_edge(self, id_pair):
        return tuple(id_pair) if self.fungible_ids else tuple(enumerate(id_pair))
    def _as_id_pair(self, edge):
        return edge if self.fungible_ids else (edge[0][1], edge[1][1])
    @property
    @lru_cache(maxsize=1)
    def graph(self):
        '''
        returns:
            (:class:`networkx.Graph`) of relationships between ids
        '''
        result = nx.Graph()
        result.add_edges_from(map(self._as_edge, self._id_pairs))
        return result
    @lru_cache(maxsize=1)
    def entity_graphs(self):
        '''
        returns:
            (list(:class:`networkx.Graph`)) connected components of :attr:`graph`
        '''
        return nx.connected_component_subgraphs(self.graph)
    def is_complete_subgraph(self, subgraph):
        '''
        Check transitive closure
        
        Args:
            subgraph(:class:`networkx.Graph`): subgraph to check
        Returns:
            (bool) Indicator of whether *subgraph* is a complete graph
                (or a complete bipartite graph if ids are not fungible)
        '''
        n_nodes = subgraph.number_of_nodes()
        if n_nodes == 0:
            return True
        if self._fungible_ids:
            return subgraph.number_of_edges() == (n_nodes * (n_nodes-1) // 2)
        else:
            node_type_counts = Counter(map(itemgetter(0), subgraph.nodes()))
            complete_n_edges = reduce(mul, node_type_counts.values(), 1)
            return (subgraph.number_of_edges() == complete_n_edges) and not any(ej[0][0] == ej[1][0] for ej in subgraph.edges())
        assert False, 'should never be reached'
    
    
