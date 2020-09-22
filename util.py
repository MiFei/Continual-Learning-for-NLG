import configparser
import shutil
import sys
import time
import math
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def construct_model_path(experiment_prefix, experiment, model_type):
    
    return f"{experiment_prefix}{experiment}/{model_type}/model/{experiment}.pt"

def construct_checkpoint_path(model_path, checkpoint_name):
    """ This func construct the path to store checkpoint model """

    return f"{model_path[: len(model_path) - 3]}_{checkpoint_name}.pt"

def construct_log_path(experiment_prefix, experiment, model_type):

    return f"{experiment_prefix}{experiment}/{model_type}/{experiment}.log"

def construct_res_path(experiment_prefix, experiment, model_type):

    return f"{experiment_prefix}{experiment}/{model_type}/{experiment}.res"

def construct_exemplar_directory_path(experiment_prefix, experiment, model_type):

    return f"{experiment_prefix}{experiment}/{model_type}/exemplars"

def construct_exemplar_path(exemplar_directory, task_name, dtype):

    data_file = f"{exemplar_directory}/{task_name}_{dtype}_data_file.json"
    distillation_file = f"{exemplar_directory}/{task_name}_{dtype}_distillation.pt"

    return [data_file, distillation_file]

def initialize_dir(config,args):
    """
    This func initialize experiment dir, model dir and exemplars dir
    """

    experiment_prefix = config["EXPERIMENT"]["experiment_prefix"]+str(args.random_seed)+'/'
    experiment_type = config["EXPERIMENT"]["experiment"]
    model_type = config["MODEL"]["model_type"]

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

    print(f"The experiment type is {experiment_type}", file = sys.stderr)

    # Create experiment directory
    experiment_dir = f"{experiment_prefix}{experiment_type}/{model_type}"
    shutil.rmtree(experiment_dir, ignore_errors = True)
    Path(experiment_dir).mkdir(parents = True, exist_ok = True)

    # Create model directory
    model_dir = f"{experiment_dir}/model"
    Path(model_dir).mkdir(parents = True, exist_ok = True)

    # Create exemplars directory
    exemplar_dir = construct_exemplar_directory_path(experiment_prefix, experiment_type, model_type)
    Path(exemplar_dir).mkdir(parents = True, exist_ok = True)

    return

def initialize_stdout(config, args):
    """
    This func set the stdout to be log or res based on train or test mode
    """

    experiment_prefix = config["EXPERIMENT"]["experiment_prefix"]+str(args.random_seed)+'/'
    experiment_type = config["EXPERIMENT"]["experiment"]
    model_type = config["MODEL"]["model_type"]

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

    if args.mode == "test":
        out_file = construct_res_path(experiment_prefix, experiment_type, model_type)
    else:
        out_file = construct_log_path(experiment_prefix, experiment_type, model_type)
    
    sys.stdout = open(out_file, "w+")
    return
    

""" Scorer """
def score(feat, gen, template, to_print = False):
    '''
    feat = ['d-a-s-v:Booking-Book-Day-1', 'd-a-s-v:Booking-Book-Name-1', 'd-a-s-v:Booking-Book-Name-2']
    gen = 'xxx slot-booking-book-name xxx slot-booking-book-time'
    '''
    das = [] # e.g. a list of d-a-s-v:Booking-Book-Day
    with open(template) as f:
        for line in f:
            if 'd-a-s-v:' not in line:
                continue
            if '-none' in line or '-?' in line or '-yes' in line or '-no' in line:
                continue
            tok = '-'.join(line.strip().split('-')[:-1])
            if tok not in das:
                das.append(tok)

    if to_print:
        print(f"Generated {gen}", file = sys.stderr)
        print(f"Ground truth {feat}", file = sys.stderr)
        print("---------------------", file = sys.stderr)

    total, redunt, miss = 0, 0, 0
    for _das in das:
        feat_count = 0
        das_order = [ _das+'-'+str(i) for i in range(20) ]

        for _feat in feat:
            if _feat in das_order:
                feat_count += 1
        slot_tok = 'slot-'+_das.split(':')[1].lower()
        _das = _das.replace('d-a-s-v:', '').lower().split('-')

        gen_count = gen.split().count(slot_tok)
        diff_count = gen_count - feat_count
        if diff_count > 0:
            redunt += diff_count
        else:
            miss += -diff_count
        total += feat_count
    return total, redunt, miss

def get_slot_error(dataset, gens, refs, sv_indexes, TO_PRINT = False):
    '''
    Args:
        gens:  (batch_size, beam_size)
        refs:  (batch_size,)
        sv:    (batch_size,)
    Returns:
        count: accumulative slot error of a batch
        countPerGen: slot error for each sample
    '''
    batch_size = len(gens)
    beam_size = len(gens[0])
    assert len(refs) == batch_size and len(sv_indexes) == batch_size

    count = {'total': 0.0, 'redunt': 0.0, 'miss': 0.0}
    countPerGen = [ [] for _ in range(batch_size) ]

    for batch_idx in range(batch_size):

        for beam_idx in range(beam_size):
            felements = [dataset.cardinality[x+dataset.dfs[2]] for x in sv_indexes[batch_idx]]

            # get slot error per sample(beam)
            total, redunt, miss = score(felements, gens[batch_idx][beam_idx], dataset.template, TO_PRINT)

            c = {}
            for a, b in zip(['total', 'redunt', 'miss'], [total, redunt, miss]):
                c[a] = b
                count[a] += b
            countPerGen[batch_idx].append(c)

    return count, countPerGen

def get_bleu(feat2content):
    '''
    :param feat2content: for each featStr -> [[refs], [gens]]
    :return: bleu score 1-4
    '''

    gen_count = 0
    list_of_references, hypotheses = [], []
    for feat in feat2content:
        refs, gens = feat2content[feat]
        gen_count += len(gens)
        refs = [s.split() for s in refs]

        for gen in gens:
            gen = gen.split()
            list_of_references.append(refs)
            hypotheses.append(gen)

    smooth = SmoothingFunction()

    BLEU = []
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.333, 0.333, 0.333, 0), (0.25, 0.25, 0.25, 0.25)]
    for i in range(4):
        t = time.time()
        bleu = corpus_bleu(list_of_references, hypotheses, weights=weights[i],
                           smoothing_function=smooth.method1)
        BLEU.append(bleu)

    print('BLEU 1-4:', BLEU)
    print('BLEU 1-4:', BLEU, file=sys.stderr)

    return BLEU

def compute_voc_ratio(task, previous_voc):
    '''
    compute the voc ratio of voc in previous tasks / new voc in current data; if the ratio is large, we should distill more
    :param task: current task
    :param previous_voc: set of voc in previous tasks
    :return:
    '''
    voc_current_data = set()
    voc_exemplar = set()
    for i in range(task.n_batch['train']):
        refs = task.next_batch('train')[4]
        for sentence in refs:
            voc_current_data = voc_current_data | set([word for word in sentence.split(' ')])

    ratio = float(len(previous_voc)) / len(voc_current_data - previous_voc)

    # return sqrt to shrink the magnitude
    return math.sqrt(ratio)


def str2bool(v):
    """
    Convert string to boolean for argument parsing
    @param:
        v(str): input string
    @return:
        v_bool(boolean): boolean version of input string
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

