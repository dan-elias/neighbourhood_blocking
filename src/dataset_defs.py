'''
===========================================
dataset_defs: Datasets used in this project
===========================================

This module applies code from :mod:`dataset_tools` and 
:mod:`preprocessing_tools` to define the datasets used in this project:

* :const:`raw_datasets`: as downloaded
* :const:`preprocessed_datasets`: Including any normalization and feature
  extraction (eg: name and address parsing, topic modelling)

The source datasets come from:

* the `recordlinkage.datasets <http://recordlinkage.readthedocs.io/en/latest/ref-datasets.html>`_ package
* the `Database Group <https://dbs.uni-leipzig.de/en/research/projects/object_matching/fever/benchmark_datasets_for_entity_resolution>`_ at the University of Leipzig
* the `North Carolina Voter Registry <http://www.ncsbe.gov/>`_

Members
-------
'''
import re
import pandas as pd
from dataset_tools import (DatasetGroup, DownloadableDataset, recordlinkageDataset,
                           TransformedDataset)
from preprocessing_tools import (DatePartsColumnTransformer, 
                                 RegexColumnParser, TopicModelTransformer, 
                                 CompositeColumnTransformer,
                                 DataFrameTransformer,
#                                 HumanNameColumnParser, 
#                                 StreetAddressColumnParser,
#                                 FunctionTransformer,
                                 )

# Large datasets (for use with spark)
#                             DownloadableDataset(source_url='https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvhis_Statewide.zip',
#                                                 table_defs=[('df1', 'ncvhis_Statewide.txt', 'csv', {'encoding':'latin-1', 'delimiter':'\t'}, lambda c:c),
#                                                            ]),
#                             DownloadableDataset(source_url='https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvoter_Statewide.zip',
#                                                 table_defs=[('df1', 'ncvoter_Statewide.txt', 'csv', {'encoding':'latin-1', 'delimiter':'\t'}, lambda c:c),
#                                                            ]),
#                             DownloadableDataset(source_url='https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvhis_ncvoter_data_format.txt', table_defs=[]),



def get_raw_datasets():
    '''
    Retirms:
        :class:`dataset.DatasetGroup` object containing raw datasets
    '''
    def parse_price_column(df):
        '''
        Column transformer for Amazon-GoogleProducts dataset
    
        Params:
            df (pandas DataFrame): Raw dataframe containing string "price" column
    
        Returns:
            pandas DataFrame: DataFrame with numeric "price" column and
                string "currency" column
        '''
        price_rec = re.compile(r'\s*(?P<amount>\d+\.?\d*)\s*(?P<ccy>[a-z]{3})\s*', re.IGNORECASE)
        def get_groupdict(px_str):
            match = price_rec.fullmatch(px_str)
            return match.groupdict() if match else pd.np.nan
        groupdicts = df['price'].map(get_groupdict, na_action='ignore')
        df['ccy'] = groupdicts.map(lambda d: d.get('ccy', None), na_action='ignore').fillna('').astype(str)
        df['price'] = groupdicts.map(lambda d: float(d['amount']) if 'price' in d else pd.np.nan, na_action='ignore')
        return df
    
    return DatasetGroup([
                         DownloadableDataset(source_url='http://dbs.uni-leipzig.de/file/DBLP-ACM.zip',
                                             table_defs=[('df1', 'DBLP2.csv', 'csv', {'encoding':'latin-1', 'index_col':'id'}, lambda c:c),
                                                         ('df2', 'ACM.csv', 'csv', {'encoding':'latin-1', 'index_col':'id'}, lambda c:c),
                                                         ('true_mapping', 'DBLP-ACM_perfectMapping.csv', 'csv', {'encoding':'latin-1'}, lambda c:c.rename(columns={'idDBLP': 'id1', 'idACM':'id2'}).set_index(['id1', 'id2']).index),
                                                        ]),
                         DownloadableDataset(source_url='http://dbs.uni-leipzig.de/file/DBLP-Scholar.zip',
                                             table_defs=[('df1', 'DBLP1.csv', 'csv', {'encoding':'latin-1', 'index_col':'id'}, lambda c:c),
                                                         ('df2', 'Scholar.csv', 'csv', {'encoding':'latin-1', 'index_col':'id'}, lambda c:c),
                                                         ('true_mapping', 'DBLP-Scholar_perfectMapping.csv', 'csv', {'encoding':'latin-1'}, lambda c:c.rename(columns={'idDBLP': 'id1', 'idScholar':'id2'}).set_index(['id1', 'id2']).index),
                                                        ]),
                         DownloadableDataset(source_url='http://dbs.uni-leipzig.de/file/Amazon-GoogleProducts.zip',
                                             table_defs=[('df1', 'Amazon.csv', 'csv', {'encoding':'latin-1', 'index_col':'id', 'dtype':{'price':str}}, parse_price_column),
                                                         ('df2', 'GoogleProducts.csv', 'csv', {'encoding':'latin-1', 'index_col':'id', 'dtype':{'price':str}}, parse_price_column),
                                                         ('true_mapping', 'Amzon_GoogleProducts_perfectMapping.csv', 'csv', {'encoding':'latin-1'}, lambda c:c.rename(columns={'idAmazon': 'id1', 'idGoogleBase':'id2'}).set_index(['id1', 'id2']).index),
                                                        ]),
                         DownloadableDataset(source_url='http://dbs.uni-leipzig.de/file/Abt-Buy.zip',
                                             table_defs=[('df1', 'Abt.csv', 'csv', {'encoding':'latin-1', 'index_col':'id', 'dtype':{'price':str}}, parse_price_column),
                                                         ('df2', 'Buy.csv', 'csv', {'encoding':'latin-1', 'index_col':'id', 'dtype':{'price':str}}, parse_price_column),
                                                         ('true_mapping', 'abt_buy_perfectMapping.csv', 'csv', {'encoding':'latin-1'}, lambda c:c.rename(columns={'idAbt': 'id1', 'idBuy':'id2'}).set_index(['id1', 'id2']).index),
                                                         ]),
                         recordlinkageDataset('febrl1', table_names='df1'),
                         recordlinkageDataset('febrl2', table_names='df1'),
                         recordlinkageDataset('febrl3', table_names='df1'),
                         recordlinkageDataset('febrl4', table_names=['df1', 'df2']),
                         ])

def get_preprocessed_datasets():
    '''
    Returns:
        :class:`dataset.DatasetGroup` object containing datasets after preprocessing and normalization
    '''
    raw_datasets = get_raw_datasets()
    def febrl_transformer():
        return DataFrameTransformer(column_transformers={'date_of_birth': DatePartsColumnTransformer(datetime_format='%Y%m%d')})
    def product_column_transformer():
        return CompositeColumnTransformer(components={'topics': TopicModelTransformer(n_topics=20),
                                                      'codes': RegexColumnParser(regexs={'codes': r'(?:[A-Z]{5,})|(?:\S*[a-zA-Z\d][^a-zA-Z\d\s]\S*)|(?:\S*[^a-zA-Z\d\s][a-zA-Z\d]\S*)|(?:\S*[^\d\s]\d\S*)|(?:\S*\d[^\d\s]\S*)'}, postprocess=lambda l: '; '.join(sorted(l)))})
    Abt_Buy_transformer = DataFrameTransformer(column_transformers={'name': product_column_transformer(),
                                                                         'description': product_column_transformer()})
    Amazon_GoogleProducts_ct = {'name': product_column_transformer(), 'description':product_column_transformer()}
    def Amazon_GoogleProducts_transformer(name_col):
        return DataFrameTransformer(column_transformers={name_col: Amazon_GoogleProducts_ct['name'],
                                                              'description': Amazon_GoogleProducts_ct['description']})
    transforms = {}
    transforms['Abt-Buy'] = {'df1': Abt_Buy_transformer, 'df2':Abt_Buy_transformer}
    transforms['Amazon-GoogleProducts'] = {'df1': Amazon_GoogleProducts_transformer('title'),
                                             'df2': Amazon_GoogleProducts_transformer('name')}
    transforms.update({name: {k: febrl_transformer() for k in raw_datasets[name].keys() if k.startswith('df')}
                       for name in ['febrl1', 'febrl2', 'febrl3', 'febrl4']
                       if name in raw_datasets})
    preprocessed_datasets = DatasetGroup(TransformedDataset(source_dataset=dataset, transforms=transforms[name], cache_subdir='preprocessed')
                                         for name, dataset in raw_datasets.items() if name in transforms)
    preprocessed_datasets.update({k:v for k, v in raw_datasets.items() if k not in preprocessed_datasets})
    return preprocessed_datasets

