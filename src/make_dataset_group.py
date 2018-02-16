'''
Script to prepare datasets used in this project
'''

import argparse, re

import dataset_defs

group_name_rec = re.compile(r'get_(?P<name>\w+)_datasets')
group_getters = {match.groupdict()['name']:v 
                 for match, v in ((group_name_rec.fullmatch(k), v) for k, v in vars(dataset_defs).items())
                 if match}

def arg_parser():
    description = 'Prepare datasets',
    epilog = 'This script prepares the datasets that are defined in the dataset_defs module'
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('group_name', choices=sorted(group_getters), help='Name of DatasetGroup object defined in the dataset_defs module')
    return parser

if __name__ == '__main__':
    args = arg_parser().parse_args()
    group_getters[args.group_name]().extract_all()
    
