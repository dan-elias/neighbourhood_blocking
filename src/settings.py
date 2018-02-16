'''
====================================
settings - Project-specific settings
====================================

Members
-------
'''

import pathlib

paths = {'src': pathlib.Path(__file__).parent.absolute()}
'''
filesystem paths
'''
paths['root'] = paths['src'].parent
paths['data'] = paths['root'] / 'data'
paths['data_raw'] = paths['data'] / 'raw'
paths['data_interim'] = paths['data'] / 'interim'
paths['data_processed'] = paths['data'] / 'processed'
paths['data_external'] = paths['data'] / 'external'
paths['data_cache'] = paths['data'] / 'cache'
