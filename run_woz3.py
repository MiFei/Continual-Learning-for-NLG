import os
import sys
import copy
import math
import nltk
import time
import random
import argparse
import configparser
import numpy as np
from pathlib import Path
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import torch

from ewc import *
from util import *
from loader.task import Task
from loader.task import generate_task
from loader.dataset_woz3 import DatasetWoz3
from loader.task import Task
from loader.task import generate_task
from loader.task import Exemplars
from model.lm_deep import LM_deep
from model.cvae import CVAE
from construct_exemplar import construct_exemplars
from model.masked_cross_entropy import *


import time 
import json

USE_CUDA = True
UNIQUE_DA = True

if USE_CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def evaluate(config, dataset, model, data_type, beam_search, beam_size, batch_size):
    """
    Evaluate model's performance in valid/test mode \n
    @param: \n
        config(configparser.ConfigParser): configuration \n
        dataset(DatasetWoz3): dataset object \n
        model(nn.Module): model to be evaluated \n
        data_type(str): valid/test mode \n
        beam_search(boolean): whether use beam search or not \n
        beam_size(int): beam size used in beam search \n
    @return: \n
        loss(float) \n
        se(float): slot error \n
        bleu(list): bleu values \n
    """

    t = time.time()
    model.eval()

    total_loss = 0
    countAll = {'total': 0.0, 'redunt': 0.0, 'miss': 0.0}
    generated_sent = {'ref': [], 'gen': []}
    feat2content = {}
    bleu = [0,0,0,0]

    with torch.no_grad():

        # Shrink batch size if test data is less than one batch (this only happens for the last task of CL DA)
        if batch_size > len(dataset.data[data_type]):
            dataset.n_batch[data_type] = 1
            dataset.n_merged_batch[data_type] = 1
            dataset.batch_size = len(dataset.data[data_type])

        if data_type == "valid":
            for i in range(dataset.n_merged_batch[data_type]):

                # Load next batch
                input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, new_task_num, do_label, da_label, sv_label = dataset.next_merged_batch("valid")

                # Feed-forward w/i ground truth as input
                if  model.model_type == "lm":
                    decoded_words, _ = model(input_var, dataset, feats_var, gen=False, beam_search=False, beam_size=1)
                else:
                    # Forward CVAE
                    model.set_prior(True)
                    target_var = input_var.clone()
                    decoded_words, _ = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = dataset)
        
                # Get loss
                loss = model.get_loss(label_var, lengths, do_label = do_label, da_label = da_label, sv_label = sv_label)
                total_loss += loss.data.item()

                # Evaluate slot error in generation mode
                if model.model_type == "lm":
                    decoded_words, _ = model(input_var, dataset, feats_var, gen=True, beam_search=False, beam_size=1)
                else:
                    model.set_prior(True)
                    target_var = input_var.clone()
                    decoded_words, _ = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, gen = True, dataset = dataset)
     
                countBatch, countPerGen = get_slot_error(dataset, decoded_words, refs, sv_indexes, TO_PRINT = False)

                # Accumulate slot error across batches
                for _type in countAll:
                    countAll[_type] += countBatch[_type]

        else: 
            # Enter test mode
            for i in range(dataset.n_batch[data_type]):

                # Load next batch
                input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, _, do_label, da_label, sv_label = dataset.next_batch(data_type)

                # Feed-forward to get loss
                if model.model_type == "lm":
                    decoded_words, _ = model(input_var, dataset, feats_var, gen=False, beam_search=False, beam_size=1)
                else:
                    # Feed forward CVAE
                    model.set_prior(True)
                    target_var = input_var.clone()
                    _ = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, gen = False, dataset = dataset)
   
                # Get loss
                loss = model.get_loss(label_var, lengths, do_label = do_label, da_label = da_label, sv_label = sv_label)
                total_loss += loss.data.item() 

                # Evaluate slot error in generation mode
                if model.model_type == "lm":
                    decoded_words, _ = model(input_var, dataset, feats_var, gen=True, beam_search=False, beam_size=beam_size)
                else:
                    model.set_prior(True)
                    target_var = input_var.clone()
                    decoded_words, _ = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, gen = True, dataset = dataset)
             
                countBatch, countPerGen = get_slot_error(dataset, decoded_words, refs, sv_indexes)

                # Calculate bleu and print generation results to log
                for batch_idx in range(dataset.batch_size):

                    featStr = featStrs[batch_idx]
                    if featStr not in feat2content:
                        feat2content[featStr] = [[], []]  # [ [refs], [gens] ]
                    target = refs[batch_idx]
                    if featStr in feat2content:
                        feat2content[featStr][0].append(target)

                    if config.getboolean("TESTING", "output_log"):
                        print('Feat: {}'.format(featStr))
                        print('Target: {}'.format(target))

                    for beam_idx in range(beam_size):
                        c = countPerGen[batch_idx][beam_idx]
                        s = decoded_words[batch_idx][beam_idx]
                        if featStr in feat2content:
                            feat2content[featStr][1].append(s)
                        if config.getboolean("TESTING", "output_log"):
                            print('Gen{} ({},{},{}): {}'.format(beam_idx, c['redunt'], c['miss'], c['total'], s))
                    if config.getboolean("TESTING", "output_log"):
                        print('-----------------------------------------------------------')

                # Accumulate slot error across batches
                for _type in countAll:
                    countAll[_type] += countBatch[_type]

            # Calculate bleu for test set only, because we only decode for test
            bleu = get_bleu(feat2content)

    if data_type == "valid":
        total_loss /= dataset.n_merged_batch[data_type]
    elif data_type == "exemplar":
        total_loss /= dataset.n_exemplar_batch["train"]
    else:
        total_loss /= dataset.n_batch[data_type]

    se = (countAll['redunt'] + countAll['miss']) / countAll['total'] * 100 if countAll["total"] > 0 else 0
    print('{} Loss: {:.3f} | Slot error: {:.3f} | BLEU4: {:.5f} | Time: {:.1f}'.format(data_type, total_loss, se, bleu[3], time.time()-t))
    print('{} Loss: {:.3f} | Slot error: {:.3f} | BLEU4: {:.5f} | Time: {:.1f}'.format(data_type, total_loss, se, bleu[3], time.time()-t), file=sys.stderr)
    print('redunt: {}, miss: {}, total: {}'.format(countAll['redunt'], countAll['miss'], countAll['total']))
    print('redunt: {}, miss: {}, total: {}'.format(countAll['redunt'], countAll['miss'], countAll['total']), file=sys.stderr)

    return total_loss, se, bleu


def train_epoch(config, args, dataset, model, voc_ratio, backup_model=None, exemplar_task=None):
    """
    Train epoch \n
    @param: \n
        config(configparser.ConfigParser): configuration \n
        args(argparse.ArgumentParser): command line arguments \n
        dataset(DatasetWoz3): dataset \n
        model(nn.Module) \n
        voc_ratio(float): adaptive weight for ewc regularization \n
        backup_model(nn.Module): model to compute distillation in adaptive mode \n
        exemplar_task(Task): data holder of exemplars \n
    @return: \n
        total_loss(float) \n
    """

    # Get config
    model_type = config["MODEL"]["model_type"]
    batch_size = config.getint("DATA", "batch_size")
    experiment_type = config["EXPERIMENT"]["experiment"]

    # Set model to trainable mode
    t = time.time()
    model.train()

    if 'ewc' in experiment_type and exemplar_task:
        # Generate ewc for regularization if experiment_type is ewc and not the first task
        ewc = EWC(model, exemplar_task)

    if 'l2' in experiment_type and exemplar_task:
        # Conduct L2 regularization
        means = {}
        for n, p in model.named_parameters():
            means[n] = variable(copy.deepcopy(p.data))

    total_loss = 0
    n_batch = dataset.n_merged_batch["train"]

    for i in range(n_batch):

        # Get next batch
        input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes, new_task_num, do_label, da_label, sv_label = dataset.next_merged_batch()
        
        """ Feed forward """
        if model_type == "lm":
            _ = model(input_var, dataset, feats_var)
        else:
            # Feed forward CVAE in train mode
            model.global_t += 1                 # Increment KL annealing time stamp
            model.set_prior(False)              # Sample latent feature from output of recognization network
            target_var = input_var.clone()
            _ = model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, dataset = dataset, gen = False)
        
        """ Get loss with/without regularization """
        if 'l2' in experiment_type and exemplar_task:
            # Get loss with L2 regularization
            _ = model.get_loss(label_var, lengths, do_label = do_label, da_label = da_label, sv_label = sv_label) 
            l2_loss = 0
            for n, p in model.named_parameters():
                _loss = (p - means[n]) ** 2
                l2_loss += _loss.sum()
            loss = model.add_ewc_loss(l2_loss, args.l2_weight)

        elif 'ewc' in experiment_type and exemplar_task:
            # Get loss with ewc regularization
            _ = model.get_loss(label_var, lengths, do_label = do_label, da_label = da_label, sv_label = sv_label)
            ewc_loss = ewc.penalty(model)
            if args.adaptive:
                loss = model.add_ewc_loss(ewc_loss, voc_ratio * args.ewc_importance)
            else:
                loss = model.add_ewc_loss(ewc_loss, args.ewc_importance)

        elif 'distillation' in experiment_type and backup_model != None:
            # Get loss with knowledge distillation based regularization

            voc_mask = torch.zeros(label_var.shape)
            backup_model.train()

            if backup_model.model_type == "lm":
                _ = backup_model(input_var, dataset, feats_var)
            else:
                backup_model.set_prior(False)
                target_var = input_var.clone()
                decoded_words, _ = backup_model(input_var, input_lengths = lengths, target_seq = target_var, target_lengths = lengths, conds_seq = feats_var, gen = False, dataset = dataset)
        
            loss = model.get_distillation_loss(label_var, lengths, voc_mask, backup_model.output_prob, new_task_num, distill_New = 'new' in experiment_type, sv_ratio=1, do_label = do_label,
                                                da_label = da_label, sv_label = sv_label)
        else:
            # Get loss without regularization
            loss = model.get_loss(label_var, lengths, do_label = do_label, da_label = da_label, sv_label = sv_label)

        # Update loss
        total_loss += loss.data.item() 

        # Update model
        model.update(config.getfloat('MODEL', 'clip'))

    total_loss /= dataset.n_merged_batch['train']

    print('Train Loss: {:.3f} | Time: {:.1f}'.format(total_loss, time.time()-t))
    print('Train Loss: {:.3f} | Time: {:.1f}'.format(total_loss, time.time()-t), file=sys.stderr)
    return total_loss

def read(config, args, first_task=False):
    """
    Read experiment configuration, Generate dataset and model \n
    @param: \n
        config(configparser.ConfigParser): configuration object \n
        args(argparse.ArgumentParser): command line argument object \n
    @return: \n
        dataset(DatasetWoz3): dataset to use \n
        model(nn.Module) \n
    """
    print('Processing data...', file=sys.stderr)

    # Read settings from config.cfg
    model_type = config["MODEL"]["model_type"]
    decoder_type = config["MODEL"]["dec_type"]
    percentage = config.getfloat("MODEL", "train_percentage")
    data_split = config["DATA"]["data_split"]
    n_layer = config.getint("MODEL", "num_layer")
    hidden_size = config.getint("MODEL", "hidden_size")
    beam_size = config.getint("TESTING", "beam_size")
    experiment_prefix = config["EXPERIMENT"]["experiment_prefix"]+str(args.random_seed)+'/'
    experiment_type = config["EXPERIMENT"]["experiment"]

    # Get model settings for cvae
    if model_type == "cvae":
        latent_size = config.getint("MODEL", "latent_size")
        std = config.getfloat("MODEL", "std")


    if first_task: # Pretrain the mode of the first task using the same
        if model_type == 'lm':
            dropout = 0.25
            lr = 0.005
        else:  # scave
            dropout = 0.25
            lr = 0.002

    else:
        # Read dropout and learning rate
        dropout = args.dropout if "dropout" in experiment_type else 0
        lr = args.lr


    # Add suffix to experiment type to indicate hyper parameter used
    if 'loss' in experiment_type:
        experiment_type = experiment_type + '_' + str(args.sv_len_weight)
    if 'distillation' in experiment_type:
        experiment_type = experiment_type +'_'+str(args._lambda)
    if 'ewc' in experiment_type:
        experiment_type = experiment_type +'_'+str(args.ewc_importance)
    if 'l2' in experiment_type:
        experiment_type = experiment_type +'_'+str(args.l2_weight)
    if 'dropout' in experiment_type:
        experiment_type = experiment_type +'_'+str(args.dropout)

    dataset = DatasetWoz3(config, data_split, percentage=percentage)

    # Get dataset parameter
    d_size = dataset.do_size + dataset.da_size + dataset.sv_size    # len of 1-hot feature vector
    do_size = dataset.do_size                                       # number of domain
    da_size = dataset.da_size                                       # number of dialogue act
    sv_size = dataset.sv_size                                       # number of slot values
    vocab_size = len(dataset.word2index)                            # vocabulary size

    # Construct model path to save/load the model
    model_path = construct_model_path(experiment_prefix, experiment_type, model_type)
    print(f"The model path is {model_path}", file = sys.stderr)
    print(f"The mode is {args.mode}", file = sys.stderr)

    # Initialize model
    if model_type == "lm":
        model = LM_deep(decoder_type, args, vocab_size, vocab_size, hidden_size, d_size, n_layer=n_layer, dropout=dropout, lr=lr)
    elif model_type == "cvae":
        model = CVAE(decoder_type, args, hidden_size, vocab_size, latent_size, d_size, do_size, da_size, sv_size, std, n_layer = n_layer, dropout = dropout, lr = lr)
    
    # Load model if recover/test mode
    if args.mode == "train":
        assert not os.path.isfile(model_path)

    elif args.mode == "recover":
        # Load the model specified by the task suffix for recovering training
        task_suffix = args.recovered_tasks
        model_path = f"{model_path[: len(model_path) - 3]}_{task_suffix}.pt"
        print(f"Recovering from {model_path}", file=sys.stderr)

        state = torch.load(model_path)
        model.load_state_dict(state["model_state_dict"])
        model.solver.load_state_dict(state["optimizer_state_dict"])
        if USE_CUDA:
            model.to(torch.device("cuda"))

    else: 
        # Load the model specified by the task suffix for testing
        task_suffix = args.recovered_tasks
        model_path = f"{model_path[: len(model_path) - 3]}_{task_suffix}.pt"
        print(f"Testing at {model_path}", file=sys.stderr)

        state = torch.load(model_path)
        model.load_state_dict(state["model_state_dict"])
        if args.mode != 'adapt':
            model.eval()

    # Print model info
    print('\n***** MODEL INFO *****')
    print('MODEL TYPE:', model_type)
    print('MODEL PATH:', model_path)
    print('SIZE OF HIDDEN:', hidden_size)
    print('# of LAYER:', n_layer)
    print('SAMPLE/BEAM SIZE:', beam_size)
    print('*************************\n')

    # Move models to GPU
    if USE_CUDA:
        model.cuda()

    return dataset, model


def train(config, args):
    """
    Train model \n
    @param:
        config(configparser.ConfigParser): configuration object \n
        args(argparse.ArgumentParser): command line argument object \n
    """

    # Read setting from config.cfg
    model_type = config["MODEL"]["model_type"]
    n_layer = config.getint("MODEL", "num_layer")
    experiment_prefix = config["EXPERIMENT"]["experiment_prefix"]+str(args.random_seed)+'/'
    experiment_type = config["EXPERIMENT"]["experiment"]
    batch_size = config.getint("DATA", "batch_size")
    total_exemplar_size = config.getint("EXEMPLARS", "exemplar_size")
    fixed_size_exemplar = config.getboolean("EXEMPLARS", "fixed_size_exemplar")
    task_sequence = config["DATA"]["task_seq"]
    task_sequence = [int(tid) for tid in task_sequence.split(",")]

    # Get granularity indicating whether the task is domain level
    # or dialogue act level
    granularity = config.getint("DATA", "granularity")

    # Add suffix to experiment type to indicate hyper parameter used
    if 'loss' in experiment_type:
        experiment_type = experiment_type + '_' + str(args.sv_len_weight)
    if 'distillation' in experiment_type:
        experiment_type = experiment_type +'_'+str(args._lambda)
    if 'ewc' in experiment_type:
        experiment_type = experiment_type +'_'+str(args.ewc_importance)
    if 'l2' in experiment_type:
        experiment_type = experiment_type +'_'+str(args.l2_weight)
    if 'dropout' in experiment_type:
        experiment_type = experiment_type +'_'+str(args.dropout)

    # Construct path to save model
    model_path = construct_model_path(experiment_prefix, experiment_type, model_type)

    # Start training
    epoch = 0
    best_loss = 10000
    print('Start training', file=sys.stderr)

    # Initialize exemplars and Task object to store exemplars
    old_exemplars = None
    exemplar_task = None

    # Set exemplar_path_prefix to store exemplars
    exemplar_path_prefix = construct_exemplar_directory_path(experiment_prefix, experiment_type, model_type)

    # Initialize cumulative set of vocabulary
    cul_voc = set()

    # Initialize holder for adpatively computing exemplars
    backup_tasks = []
    backup_selected_ids = []
    backup_model = None
    total_train_num = 0
    task_exemplars = {} # exemplar of each task before shrinking
    task_size_dict = {}

    """ ------ Incremental training through task sequence ------ """
    for index, task_id in enumerate(task_sequence):

        """ ------ Read settings and Initilization ------ """
        if index == 0:
            dataset, model = read(config, args, first_task=True)
        else:
            dataset, model = read(config, args)
            if experiment_type != "upper_bound": # if upper_bound: train with new initilization gives better perf
                # load the best model from the last task
                model.load_state_dict(state["model_state_dict"])
                model.solver.load_state_dict(state["optimizer_state_dict"])

        # Reset global timestamp of CVAE model
        if config["MODEL"]["model_type"] == "cvae":
            model.global_t = 0

        # Reset loss and epoch for new task
        epoch = 0
        best_loss = 10000

        # Get task for current round
        if experiment_type == "upper_bound":

            # Generate dataset combining all the tasks available so far 
            task, task_name, task_voc = generate_task(dataset, task_sequence[:index+1], old_exemplars)
        else:
            # Train in incremental manner by gererating next available task with old exemplars
            task, task_name, task_voc = generate_task(dataset, [task_id], old_exemplars)

        # Update total training sample count and task-wise training sample count dictionary
        total_train_num += len(task.data["train"])
        task_size_dict[task_id] = len(task.data["train"])

        # Generate Task object containing only the exemplars
        if old_exemplars:
            exemplar_task = DatasetWoz3(dataset.config, data={'train': old_exemplars['train'].data})

        # Compute voc_ratio for adaptive ewc
        if task.exemplars['train']:
            voc_ratio = math.sqrt(float(len(cul_voc)) / len(task_voc - cul_voc))
            print(f"voc Ratio: {voc_ratio} for {task_name}", file = sys.stderr)
            print(f"voc Ratio: {voc_ratio} for {task_name}")
        else:
            voc_ratio = 1

        # Update cummulative vocabulary size
        cul_voc = cul_voc | task_voc

        # Log current task's info
        print(f"Current task: {task_name} has batches: {task.n_merged_batch}", file = sys.stderr)
        print(f"Current task: {task_name} has batches: {task.n_merged_batch}")
        print(f"Current task: {task_name} has {task.exemplars['train'].size() if task.exemplars['train'] else 0} distillation", file = sys.stderr)

        # # Initialize model with best model for previous tasks
        # # if upper_bound by combining all tasks available so far into a dataset for training, reint model every time
        # if index > 0 and experiment_type != "upper_bound":
        #     model = copy.deepcopy(best_model)

        """ ------ Deploy checkpoint ------ """
        checkpoint_task_seq = task_sequence[: index + 1]
        checkpoint_name = "".join([str(i) for i in checkpoint_task_seq])
        checkpoint_path = construct_checkpoint_path(model_path, checkpoint_name)

        """ ------ Train on current task ------ """
        while epoch < config.getint('TRAINING', 'n_epochs'):

            if args.mode == 'recover' and index == 0:
                break

            # Permute utterances of current task
            task.reset_merged()
            print('Task {} Epoch'.format(task_name), epoch, '(n_layer {})'.format(n_layer))
            print('Task {} Epoch'.format(task_name), epoch, '(n_layer {})'.format(n_layer), file=sys.stderr)

            # Obtain training loss of current epoch
            train_loss = train_epoch(config, args, task, model, voc_ratio, backup_model, exemplar_task)

            # Obtain validation loss, slot error and bleu
            loss, se, bleu = evaluate(config, task, model, 'valid', False, 1, task.batch_size)

            # Check condition for early stopping
            if loss < best_loss:

                # Update best model&loss if current loss is less than best loss
                best_loss = loss
                early_stop = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.solver.state_dict()
                }, checkpoint_path)
                print('Best loss: {:.3f}, AND Save model!'.format(loss))
                print('Best loss: {:.3f}, AND Save model!'.format(loss), file=sys.stderr)
            else:
                print("So sad. Loss does not decrease")
                early_stop += 1

            if early_stop == 10:
                break
            epoch += 1

            print('----------------------------------------')
            print('----------------------------------------', file=sys.stderr)

        # load the best model of the current task from early stop to compute exemplar and for evlauation
        best_model = copy.deepcopy(model)
        state = torch.load(checkpoint_path)
        best_model.load_state_dict(state["model_state_dict"])
        best_model.solver.load_state_dict(state["optimizer_state_dict"])

        """ ------ Construct exemplars if required ------ """
        if "exemplar" in experiment_type and "upper_bound" not in experiment_type and "finetune" not in experiment_type and index != len(task_sequence) - 1:

            if USE_CUDA:
                best_model.to(torch.device("cuda"))

            if fixed_size_exemplar:
                # Calculate exemplar to keep for current task if total exemplar size is fixed
                if old_exemplars is not None:
                    exemplar_size = int(total_exemplar_size * (len(task.data["train"]) / total_train_num))
                else:
                    exemplar_size = total_exemplar_size
            else:
                # Set exemplar size to keep for current task to total_exemplar_size if size is not fixed
                exemplar_size = total_exemplar_size
            exemplar_size_dict = dict({"train": min(len(task.data['train']),exemplar_size), "valid": min(len(task.data['valid']),exemplar_size) })

            # Construct exemplars
            exemplars = construct_exemplars(best_model, task, exemplar_size_dict, config, args.sv_len_weight)

            # Pass exemplars to next task and use it for training
            if old_exemplars is None:
                old_exemplars = exemplars
            elif not fixed_size_exemplar:
                old_exemplars["train"].merge(exemplars["train"])
                old_exemplars["valid"].merge(exemplars["valid"])
            else:
                # Reduce the least representative exemplars for old tasks to provide space for new exemplars
                # if fixed_size_exemplar
                old_exemplars = { 'train': Exemplars([],[]), 'valid': Exemplars([],[]) }

                for dtype in ['train', 'valid']:
                    for id, exemp in task_exemplars.items():

                        shrinked_size = int(total_exemplar_size * (task_size_dict[id] / total_train_num))
                        old_exemplars[dtype].data += exemp[dtype].data[: shrinked_size]

                    old_exemplars[dtype].data += exemplars[dtype].data

            task_exemplars[task_id] = exemplars

            if config.getboolean("TRAINING", "save_exemplars"):
                # Save exemplars
                
                for dtype in ["train", "valid"]:

                    data_file, distillation_file = construct_exemplar_path(exemplar_path_prefix, task_name, dtype)
                    exemplars[dtype].save(f"{data_file}", f"{distillation_file}")
                    print(f"Storing exemplars to {data_file}", file = sys.stderr)

                # Log some random exemplars
                exemplars["train"].print(task, num=5)

            # Get average slot value count of exemplars
            sv_count = 0
            for _, _, _, meta in exemplars["train"].data:
                sv_count += len(task.getFeatIdx(meta)[2])
            print(f"The mean sv count is {sv_count / total_exemplar_size}", file = sys.stderr)
            print(f"The mean sv count is {sv_count / total_exemplar_size}")

        """ ------ Evaluate and save current model and results ------ """

        checkpoint_model(config, best_model, dataset, checkpoint_task_seq, checkpoint_path) 

        """ ------ Backup best model for current task ------ """
        if "distillation" in experiment_type:
            backup_model = copy.deepcopy(best_model)
            backup_model.eval()

    return

def test(config, args):
    """
    Evaluate model performance in test mode \n
    @param:
        config(configparser.ConfigParser): configuration object \n
        args(argparse.ArgumentParser): command line argument object \n
    """

    # Read dataset and model to evaluate
    dataset, model = read(config, args)

    # Read settings
    # TODO: Move beam search option to config file
    beam_search = False
    beam_size = config.getint("TESTING", "beam_size")
    task_sequence = args.recovered_tasks # test all tasks specified by recovered_tasks
    task_sequence = [int(tid) for tid in task_sequence.split(",")]

    # Evaluate on tasks
    for tid in task_sequence:
        task, task_name, _ = generate_task(dataset, [tid], old_exemplars = None)
        print(f"\nEvaluating on task {task_name}", file = sys.stderr)
        print(f"\nEvaluating on task {task_name}")
        loss, se, bleu = evaluate(config, task, model, "test", beam_search, beam_size, task.batch_size)

    return

def parse():
    """
    Read command line arguments and configuration file
    Conduct initialization for experiment
    """
    parser = argparse.ArgumentParser(description='Train dialogue generator')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--random_seed' , default=1111 , type=int, help='random seed')
    parser.add_argument("--config_file", default='config/config.cfg', type=str, help="config file")
    parser.add_argument("--sv_len_weight", default=1.0, type=float, help="the weight to consider the number of sv pairs to rank loss")
    parser.add_argument("--T", default=1.0, type=float, help="temperature in softmax for distillation")
    parser.add_argument("--_lambda", default=1.0, type=float, help="weight to interpolate CE loss and distillation loss")
    parser.add_argument("--adaptive", default=False, type=str2bool, help="using adaptive lambda on ewc")
    parser.add_argument("--ewc_importance", default=300000, type=float, help="weight to ewc loss")
    parser.add_argument("--l2_weight", default=0.001, type=float, help="weight to l2 reg")
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("--recovered_tasks", default="0", type=str, help="tasks to recover or test")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    # Initialize directories to hold the results
    if args.mode == "train":
        initialize_dir(config, args)

    # Initialize stdout
    initialize_stdout(config, args)

    return args, config

def checkpoint_model(config, model, dataset, task_seq, file_path):
    """
    This func evaluate current model on all tasks trained until current task
    and store the model and results into file_path\n
    @param:
        config(configparser.ConfigParser): configuration object \n
        model(nn.Module) \n
        dataset(DatasetWoz3) \n
        task_seq(list): task sequence \n
        file_path(str): output file path \n
    """

    print(f"\nCHECKPOINT {task_seq} to {file_path}", file = sys.stderr)
    print(f"\nCHECKPOINT {task_seq} to {file_path}")

    model.eval()

    # Get setting
    beam_size = config.getint("TESTING", "beam_size")
    batch_size = config.getint("DATA", "batch_size")
    perf_dict = defaultdict(lambda: defaultdict(float))

    with torch.no_grad():

        for task_id in task_seq:

            # Get current individual task
            task, task_name, _ = generate_task(dataset, [task_id], old_exemplars = None)

            # Evaluate on current task
            print(f"\nEvaluate on task {task_id}: {task_name}", file = sys.stderr)
            print(f"\nEvaluate on task {task_id}: {task_name}")
            loss, se, bleu = evaluate(config, task, model, "test", beam_search = False, beam_size = beam_size,
                                batch_size = batch_size)

            # Record loss and slot-error
            perf_dict["loss"][task_id] = loss
            perf_dict["se"][task_id] = se
            perf_dict["bleu"][task_id] = bleu

        # Get all tasks available so far
        task, task_name, _ = generate_task(dataset, task_seq, old_exemplars=None)

        # Evaluate on all tasks till current task
        cul_loss, cul_se, cul_bleu = evaluate(config, task, model, "test", beam_search=False, beam_size=beam_size,
                                                batch_size=batch_size)

        # Record loss and slot-error
        perf_dict["cul_loss"] = cul_loss
        perf_dict["cul_se"] = cul_se
        perf_dict["cul_bleu"] = cul_bleu
        print(f"Evaluate on all previous tasks", file = sys.stderr)
        print(f"Evaluate on all previous tasks")
        print("{} Loss: {:.3f} | Slot error: {:.3f} | BLEU4: {:.5f}".format("test", cul_loss, cul_se, cul_bleu[3]), file=sys.stderr)
        print("{} Loss: {:.3f} | Slot error: {:.3f} | BLEU4: {:.5f}".format("test", cul_loss, cul_se, cul_bleu[3]))

        # Save model and performance of each task
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model.solver.state_dict(),
            "perf": dict(perf_dict)
        }, file_path)

    return

if __name__ == '__main__':

    # Parse arguments and configuration
    args, config = parse()

    # Set seed for reproducibility
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # Train from scratch or previous checkpoint
    if args.mode == 'train' or args.mode == 'adapt' or args.mode == 'recover':
        train(config, args)
    # Test
    elif args.mode == 'test':
        test(config, args)
    else:
        print('mode cannot be recognized')
        sys.exit(1)