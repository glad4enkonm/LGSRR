import os
import csv
import sys
import torch
import numpy as np
from transformers import BertTokenizer

def get_t_data(args, data_args):
    
    if args.text_backbone.startswith('bert'):
        t_data = get_data(args, data_args)
    else:
        raise Exception('Error: inputs are not supported text backbones.')

    return t_data

def get_data(args, data_args):

    processor = DatasetProcessor(args)
    if 'text_data_path' in data_args:
        data_path = data_args['text_data_path']
    else:
        data_path = data_args['data_path']
    
    outputs = {}
    if 'train_data_index' in data_args:        
        train_examples = processor.get_examples(data_path, 'train') 
        train_feats = get_backbone_feats(args, train_examples, data_args)
        
        dev_examples = processor.get_examples(data_path, 'dev')
        dev_feats = get_backbone_feats(args, dev_examples)

        for key in train_feats.keys():
            tmp_outputs = {}
            tmp_outputs[key] = {
                'train': train_feats[key],
            }
            outputs.update(tmp_outputs)

        for key in dev_feats.keys():
            if key in outputs:
                outputs[key].update({'dev': dev_feats[key]})
            else:
                outputs[key] = {'dev': dev_feats[key]}
    
    if 'test_data_index' in data_args:
        test_examples = processor.get_examples(data_path, 'test')
        test_feats = get_backbone_feats(args, test_examples)

        for key in test_feats.keys():
            if key in outputs:
                outputs[key].update({'test': test_feats[key]})
            else:
                outputs[key] = {'test': test_feats[key]}
            
    return outputs


def read_rank_data(rank_path, train_data_indexes, dataset):
    rank_map = {'Speakers\' Actions': 0, 'Emotions and Facial Expressions': 1, 'Interaction with Others': 2}
    
    tmp = {}
    with open(rank_path, 'r') as f:
        data = csv.reader(f, delimiter="\t")

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            index, tmp_list = None, None
            rank_list = [0, 0, 0]
            
            if dataset in ['MIntRec2.0']:
                index ='_'.join(['dia' + str(line[0]), 'utt' + str(line[1])])
                tmp_list = [rank_map[line[2]], rank_map[line[3]], rank_map[line[4]]]
                for idx, rank in enumerate(tmp_list):
                    rank_list[rank] = idx
                
            elif dataset in ['IEMOCAP-DA']:
                index = line[0]
                tmp_list = [rank_map[line[1]], rank_map[line[2]], rank_map[line[3]]]
                for idx, rank in enumerate(tmp_list):
                    rank_list[rank] = idx

            tmp[index] = rank_list

    rank_data = [np.array(tmp[x]) for x in train_data_indexes]

    return rank_data

def get_backbone_feats(args, examples, data_args=None):
    
    if args.text_backbone.startswith(('bert')):
        tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)

        outputs = convert_examples_to_features(args, examples, tokenizer)   
        features = outputs['features']
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
        outputs['features'] = features_list

        desc_feats = outputs['desc_feats']
        desc_feats = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in desc_feats]
        outputs['desc_feats'] = desc_feats
        
        if data_args:
            rank_data = read_rank_data(args.rank_path, data_args['train_data_index'],args.dataset)
            outputs['rank_data'] = rank_data

        return outputs
    

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, desc=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.desc = desc

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args):
        super(DatasetProcessor).__init__()
        self.args = args

        if args.dataset in ['MIntRec2.0']:
            self.select_id = 2
            self.desc_id = 2
        elif args.dataset in ['IEMOCAP-DA']:
            self.select_id = 1
            self.label_id = 2
            self.desc_id = 1
        
    def get_examples(self, data_dir, mode):
        if mode=='train':
            return self._create_examples(
                lines=self._read_tsv(os.path.join(data_dir, "train.tsv")), 
                set_type="train",
                lines_desc=self._read_tsv(os.path.join(self.args.desc_path,"train.tsv")))
        elif mode=='dev':
            return self._create_examples(
                lines=self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
                set_type="train",
                lines_desc=self._read_tsv(os.path.join(self.args.desc_path,"dev.tsv")))
        elif mode=='test':
            return self._create_examples(
                lines=self._read_tsv(os.path.join(data_dir, "test.tsv")), 
                set_type="test",
                lines_desc=self._read_tsv(os.path.join(self.args.desc_path,"test.tsv")))
        elif mode=='all':
            return self._create_examples(
                lines=self._read_tsv(os.path.join(data_dir, "all.tsv")), 
                set_type="all",
                lines_desc=self._read_tsv(os.path.join(self.args.desc_path,"all.tsv")))

    def _create_examples(self, lines, set_type, lines_desc=None):
        """Creates examples for the training and dev sets."""
        examples = []
        
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.select_id]

            desc = None
            if lines_desc is not None:
                desc = []
                for j in range(self.desc_id, len(lines_desc[i])):
                    desc.append(lines_desc[i][j])

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=None, desc=desc))
        return examples

def convert_examples_to_features(args, examples, tokenizer):

    max_seq_length = args.text_seq_len
        
    outputs = {}
    features = []
    desc_feats = []

    for (ex_index, example) in enumerate(examples):

        data = tokenizer(example.text_a, padding='max_length', max_length=max_seq_length, truncation=True, return_tensors="pt")

        input_ids = data['input_ids'].squeeze(0).tolist()
        input_mask = data['attention_mask'].squeeze(0).tolist()

        segment_ids = [0] * len(input_ids)
        segment_ids = torch.tensor(segment_ids)

        desc_input_ids_list = []
        desc_input_mask_list = []
        desc_segment_ids_list = []

        for i in range(len(example.desc)): 
            desc_tokenized=tokenizer(example.desc[i], padding='max_length', max_length=max_seq_length, truncation=True, return_tensors="pt")
            desc_input_ids = desc_tokenized['input_ids'].squeeze(0)
            desc_input_ids_list.append(desc_input_ids.tolist())
            
            desc_input_mask = desc_tokenized['attention_mask'].squeeze(0)
            desc_input_mask_list.append(desc_input_mask.tolist())

            desc_segment_ids = [0] * len(desc_input_ids)
            desc_segment_ids_list.append(desc_segment_ids)
                        
        features.append(
            InputFeatures(input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids)
                        )
        
        desc_feats.append(
            InputFeatures(input_ids=desc_input_ids_list,
                        input_mask=desc_input_mask_list,
                        segment_ids=desc_segment_ids_list)
                        )
    outputs = {
        'features': features,
        'desc_feats': desc_feats
    }
    
    return outputs