import os
import re
import sys
import json
import random
import numpy as np
import torch
import configparser
from torch.autograd import Variable
from collections import defaultdict
import configparser
USE_CUDA = True

# Add current module's path to sys path
module_path = os.path.expanduser(os.getcwd())
print(module_path)
sys.path.insert(0, module_path)
from .dataset_woz3 import DatasetWoz3

class Exemplars(object):
    """
    exemplars object contain the data and possibly the output distribution vector for each word
    The output distribution of each word serves as distillation
    """

    def __init__(self, data, distillation = None):
        """
        Constructor of exemplars \n
        @param: \n
            data(list): list of dial_idx, turn_idx, text, meta \n
            distillation(torch.Tensor): tensor of distribution tensor at each word for each sent \n
        """
        self.data = data
        self.distillation = distillation
    
    def merge(self, input_exemplars):
        """
        This func merge the input exemplars to current exemplars
        """

        self.data += input_exemplars.data

        if self.distillation is not None:
            self.distillation = torch.cat([self.distillation, \
                                            input_exemplars.distillation])
        
    def size(self):
        """
        Return size of exemplars
        """

        return len(self.data)

    def print(self, task, num = 10):
        """
        Print num sample sentences from exemplars
        @param:
            task(Task)
            num(int): number of exemplars to print
        """

        # Check the quality of selected samples
        for _, _, text, meta in self.data[: num]:

            # Get the feat of current turn
            do_list = task.get_domain_list()
            do_idx = task.getFeatIdx(meta)[0][0]
            do = do_list[do_idx]
            print(f"{do}: {text['delex']}", file = sys.stderr)
            print(f"{do}: {text['delex']}")
            
    def save(self, data_file, distillation_file):
        """
        Save the data and distillation into file
        """

        with open(data_file, "w+") as f:
            json.dump(self.data, f)
        
        # if self.distillation is not None:
        #     torch.save(self.distillation, distillation_file)

    @classmethod
    def load(cls, data_file, distillation_file = None):
        """
        This function load an exemplar object
        """

        with open(data_file, "r") as f:
            data = json.load(f)

        distillation = None
        if distillation_file is not None:
            distillation = torch.load(distillation_file)
            distillation = distillation.cuda()
        return Exemplars(data, distillation)
    
class Task(DatasetWoz3):
    """
    Task container
    """

    def __init__(self, config, percentage=1.0, data = None,
                    exemplars = None, task_name = None):
        """
        Initialize task with DatasetWoz3 as underlying data structure \n
        @param: \n
            config(configparser.ConfigParser): configuration object \n
            percentage(float): percentage of training data to keep \n
            data(dict): dict containing train, validation and test data \n
            exemplars(dict): dict containing exemplars of previous tasks \n
            task_name(str): name of the task
        """
        
        super(Task, self).__init__(config, dataSplit_file = None, percentage = percentage,
                                    data = data, exemplars = exemplars)

        self.task_name = task_name
        self.merged_data_index = {"train": 0,  "valid": 0, "test": 0}
        self.merged_data_indices = {"train": np.random.permutation(len(data["train"]) + (exemplars["train"].size() if exemplars["train"] \
                                                                else 0)),
                                    "valid": np.random.permutation(len(data["valid"]) + (exemplars["valid"].size() if exemplars["valid"] \
                                                                else 0))}
        # Set exemplars for current task
        self.exemplars = exemplars
        
        # Set number of merged batch for training
        self.n_merged_batch = dict()
        self.n_merged_batch["train"] = (len(self.data["train"]) +\
                                    (len(self.exemplars["train"].data) if self.exemplars["train"] is not None else 0)) // self.batch_size
        self.n_merged_batch["valid"] = (len(self.data["valid"]) +\
                                (len(self.exemplars["valid"].data) if self.exemplars["valid"] is not None else 0)) // self.batch_size
        
        # Set number of exemplar batch for testing the correctness of KL-divergence
        self.n_exemplar_batch = dict()
        self.n_exemplar_batch["train"] = (len(self.exemplars["train"].data) if self.exemplars["train"] is not None else 0) // self.batch_size
        self.n_exemplar_batch["valid"] = (len(self.exemplars["valid"].data) if self.exemplars["valid"] is not None else 0) // self.batch_size
        self.exemplar_index = {"train": 0, "valid": 0}
        self.exemplar_lengths = {"train": (len(self.exemplars["train"].data)) if self.exemplars["train"] is not None else 0,
                                "valid": (len(self.exemplars["valid"].data)) if self.exemplars["valid"] is not None else 0}
    
        # Log task info
        print(f"The number of exemplar batch is {self.n_exemplar_batch['train']}, {self.n_exemplar_batch['valid']}", file = sys.stderr)
        print(f"The number of batch is {self.n_merged_batch['train']}, {self.n_merged_batch['valid']}", file = sys.stderr)
        return   

    def reset_merged(self):
        """
        Re-permute data and exemplars
        """
        self.merged_data_index = {"train": 0, "valid": 0, "test": 0}
        self.merged_data_indices = {"train": np.random.permutation(len(self.data["train"]) + (self.exemplars["train"].size() if self.exemplars["train"] is not None \
                                                                else 0)),
                                    "valid": np.random.permutation(len(self.data["valid"]) + (self.exemplars["valid"].size() if self.exemplars["valid"] is not None \
                                                                else 0))}

    def next_merged_batch(self, data_type = "train"):
        """
        This func return a batch selected from both data and exemplars \n
        @param: \n
            data_type(str): type of data \n
        @return: \n
            input_var(torch.FloatTensor): unpadded sentences, shape = (batch_size, length of sent, vocab_size) \n
            label_var(torch.LongTensor): padded sentences, shape = (batch_size, max_length, vocab_size) \n
            feats_var(torch.LongTensor): feature vector representing do/da/dv, shape = (batch_size, do + da + sv size) \n
            lengths(list): lengths of sents, list \n
            refs(list): list of delexicalized sentences \n
            featStrs(list): list of feature strings \n
            sv_indexes(list): list of slot value indices \n
            new_task_num(int): number of samples from new task in the output \n 
            do_label(torch.Tensor): tensor of domain labels \n
            da_label(torch.Tensor): one-hot tensor of dialogue labels \n
            sv_label(torhc.Tensor): one-hot tensor of slot-value labels
        """

        def indexes_from_sentence(sentence, add_eos=False):
            indexes = [self.word2index[word] if word in self.word2index else self.word2index['UNK_token'] for word in sentence.split(' ')]
            if add_eos:
                return indexes + [self.word2index['EOS_token']]
            else:
                return indexes

        # Pad a with the PAD symbol
        def pad_seq(seq, max_length):
            seq += [self.word2index['PAD_token'] for i in range(max_length - len(seq))]
            return seq

		# turn list of word indexes into 1-hot matrix
        def getOneHot(indexes):
            res = []
            for index in indexes:
                hot = [0]*len(self.word2index)
                hot[index] = 1
                res.append(hot)
            return res
        
        # Get distillation or not
        experiment_type = self.config["EXPERIMENT"]["experiment"]

        # Read a batch
        start = self.merged_data_index[data_type]
        end = self.merged_data_index[data_type] + self.batch_size
        self.merged_data_index[data_type] += self.batch_size
        selected_indices = self.merged_data_indices[data_type][start: end]
        new_task_len = len(self.data[data_type])
        new_data_indices = selected_indices[selected_indices < new_task_len]
        exemplar_indices = selected_indices[selected_indices >= new_task_len]
        exemplar_indices = [i - new_task_len for i in exemplar_indices]

        # Get new task's data
        data = [self.data[data_type][i] for i in new_data_indices]

        # Get exemplar's data if possible
        data += [self.exemplars[data_type].data[i] \
                                 for i in exemplar_indices]

        # Get sentences, refs, feats, featStrs
        sentences, refs, feats, featStrs = [], [], [], []
        do_label, da_label, sv_label = [], [], []
        sv_indexes = []

        for dial_idx, turn_idx, text, meta in data:

            text_ori, text_delex = text['ori'], text['delex']
            sentences.append(indexes_from_sentence(text_delex, add_eos=True))
            refs.append(text_delex)

            # get semantic feature
            do_idx, da_idx, sv_idx, featStr = self.getFeatIdx(meta)
            do_cond = [1 if i in do_idx else 0 for i in range(self.do_size)] # domain condition
            da_cond = [1 if i in da_idx else 0 for i in range(self.da_size)] # dial act condition
            sv_cond = [1 if i in sv_idx else 0 for i in range(self.sv_size)] # slot/value condition
            do_label.append(do_idx[0])
            da_label.append(da_cond)
            sv_label.append(sv_cond)
            feats.append(do_cond + da_cond + sv_cond)
            featStrs.append(featStr)

            sv_indexes.append(sv_idx)

        # Pad with 0s to max length
        lengths = [len(s) for s in sentences]
        max_length = 60 if self.config.getboolean("DATA", "sent_max_len") else max(lengths)
        sentences_padded = [pad_seq(s, max_length) for s in sentences]

        # Turn (batch_size, max_len) into (batch_size, max_len, n_vocab)
        sentences = [getOneHot(s) for s in sentences_padded]
        
        input_var = Variable(torch.FloatTensor(sentences))
        label_var = Variable(torch.LongTensor(sentences_padded))
        feats_var = Variable(torch.FloatTensor(feats))
        do_label = Variable(torch.LongTensor(do_label))
        da_label = Variable(torch.FloatTensor(da_label))
        sv_label = Variable(torch.FloatTensor(sv_label))

        new_task_num = len(new_data_indices)

        if USE_CUDA:
            input_var = input_var.cuda()
            label_var = label_var.cuda()
            feats_var = feats_var.cuda()
            do_label = do_label.cuda()
            da_label = da_label.cuda()
            sv_label = sv_label.cuda()


        return input_var, label_var, feats_var, lengths, refs, featStrs, \
                sv_indexes, new_task_num, do_label, da_label, sv_label

    def add_generated_data(self, data):
        """
        @param:
            data(dict): generated data to be added
        """

        # Record data
        for key in self.data.keys():
            self.data[key] += data[key]
        
        # Update n_batch and n_merged_batch
        self.n_batch = dict(zip(self.data.keys(),
                                [len(self.data[key]) // self.batch_size
                                for key in self.data.keys()]))

        self.n_merged_batch["train"] = (len(self.data["train"]) +\
                                    (len(self.exemplars["train"].data) if self.exemplars["train"] is not None else 0)) // self.batch_size
        self.n_merged_batch["valid"] = (len(self.data["valid"]) +\
                                (len(self.exemplars["valid"].data) if self.exemplars["valid"] is not None else 0)) // self.batch_size
        
def generate_task(dataset, task_seq, old_exemplars: dict):
    """
    Generate task from dataset with task id specified in the task sequence
    @param:
        dataset(DatasetWoz3): full dataset \n
        task_seq(list): indices of task (domain / dialogue act) in template \n
        old_exemplars(dict): exemplars of previous tasks dict{train: exemplars, valid: exemplars} \n
    @return:
        task(Task): task with data containing only sentences with task id in the task sequence and exemplars for rehearsal
        task_name(str): domain / dialogue act of the task generated
        voc_index(list): vocabulary indices of the current domain / dialogue act task
    """ 

	# Get data with task specified by task id
    data = dict()
    task_name = set()

    # Initialize vocabulary for current task
    voc = set()

    # TODO: Find a better way to initialize da map
    da_list = dataset.get_dialogue_act_list()
    da_list = sorted(da_list)
    da_map = dict(zip(da_list, np.arange(len(da_list))))

    # da_map = {'Inform':0, 'Select': 1, 'Request': 2, 'Recommend': 3, 'NoOffer': 4, 'OfferBook':5, 'OfferBooked':6, 'Book': 7, 'NoBook': 8}
 
    for dtype in dataset.data.keys():
        # Get data for train/valid/test mode

        data[dtype] = []
        for item in dataset.data[dtype]:

            _, _, text, meta = item
            current_task_id = dataset.getFeatIdx(meta)[dataset.granularity][0]

            # Map dialog act from different domains to one group
            if dataset.granularity == 1:
                current_da = list(meta.keys())[0].split("-")[1]
                if current_da not in da_map.keys():
                    continue
                else:
                    current_task_id = da_map[list(meta.keys())[0].split("-")[1]]

            # Add current item into task if it has required task id
            if current_task_id in task_seq:
                task_name.add(list(meta.keys())[0].split("-")[dataset.granularity])
                data[dtype].append(item)
                voc = voc | set([word for word in text['delex'].split(' ')])

    task_name = "".join(task_name)

	# Create task with data specified
    old_exemplars = dict({"train": None, "valid": None}) if old_exemplars == None else old_exemplars

    task = Task(dataset.config, dataset.percentage, data,
                exemplars = old_exemplars,
                task_name = task_name)


    # Check whether at least one task is captured
    assert len(task_name) > 0

    # Convert vocabulary to index
    voc_index = [dataset.word2index[word] if word in dataset.word2index else dataset.word2index['UNK_token'] for word in voc]
    voc_index += [0,1,2] # append PAD_token SOS_token EOS_token
    
    return task, task_name, set(voc_index)