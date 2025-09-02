import os
import logging
import csv

from .mm_pre import MMDataset
from .text_pre import get_t_data
from .__init__ import benchmarks


__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):        
        self.logger = logging.getLogger(args.logger_name)
        bm = benchmarks[args.dataset]
        max_seq_lengths = bm['max_seq_lengths']
        args.text_seq_len = max_seq_lengths['text']
        self.label_list = bm["intent_labels"]
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))  
        args.num_labels = len(self.label_list) 
        self.data = prepare_data(args, self.logger, self.label_list, bm)


def prepare_data(args, logger, label_list, bm):    
    def get_other_hyper(inputs):
        other_hyper = {}
        other_hyper['train'] = {}
        other_hyper['dev'] = {}
        other_hyper['test'] = {}

        for key in inputs.keys():
            if key not in ['text_data', 'train_label_ids', 'dev_label_ids', 'test_label_ids']:
                if 'train' in inputs[key]:
                    other_hyper['train'][key] = inputs[key]['train']
                if 'dev' in inputs[key]:
                    other_hyper['dev'][key] = inputs[key]['dev']
                if 'test' in inputs[key]:
                    other_hyper['test'][key] = inputs[key]['test']

        return other_hyper
      
    data = {}
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    data_path = os.path.join(args.data_path, args.dataset)
    ind_outputs = get_data(args, logger, data_path, bm, label_map)   
    train_label_ids, dev_label_ids, test_label_ids = ind_outputs['train_label_ids'], ind_outputs['dev_label_ids'], ind_outputs['test_label_ids']
    text_data = ind_outputs['text_data']
    ind_other_hyper = get_other_hyper(ind_outputs)

    mm_train_data = MMDataset(train_label_ids, text_data['train'], other_hyper = ind_other_hyper['train'])
    mm_dev_data = MMDataset(dev_label_ids, text_data['dev'], other_hyper = ind_other_hyper['dev'])
    mm_test_data = MMDataset(test_label_ids, text_data['test'], other_hyper = ind_other_hyper['test'])
    data = {'train': mm_train_data, 'dev': mm_dev_data, 'test': mm_test_data}

    return data 


def get_data(args, logger, data_path, bm, label_map):  
    logger.info('Data preparation...')
    text_data_path = data_path
    train_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'train.tsv'))
    dev_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'dev.tsv'))
    test_outputs = get_indexes_annotations(args, bm, label_map, os.path.join(text_data_path, 'test.tsv'))
    args.num_train_examples = len(train_outputs['indexes'])
    
    data_args = {
        'data_path': data_path,
        'text_data_path':text_data_path,
        'train_data_index': train_outputs['indexes'],
        'dev_data_index': dev_outputs['indexes'],
        'test_data_index': test_outputs['indexes'],
        'label_map': label_map
    }
    text_data = get_t_data(args, data_args)
    
    outputs = {
        'train_label_ids': train_outputs['label_ids'],
        'dev_label_ids': dev_outputs['label_ids'],
        'test_label_ids': test_outputs['label_ids']
    }
    outputs['text_data'] = text_data['features']    
    text_data.pop('features')
    outputs.update(text_data)

    return outputs


def get_indexes_annotations(args, bm, label_map, read_file_path):
    with open(read_file_path, 'r') as f:
        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue

            if args.dataset in ['MIntRec2.0']:
                index = '_'.join(['dia' + str(line[0]), 'utt' + str(line[1])])
                label_id = label_map[line[3]]

            elif args.dataset in ['IEMOCAP-DA']:
                index = line[0]
                label_id = label_map[bm['label_maps'][line[2]]]

            indexes.append(index)
            label_ids.append(label_id)
    
    outputs = {
        'indexes': indexes,
        'label_ids': label_ids,
    }
    return outputs