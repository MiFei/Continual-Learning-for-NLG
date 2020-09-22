"""
This script splits multi-domain/dialogue act utterances into single-domain/dialogue act utterances
"""
import numpy as np
import re
from collections import defaultdict
import json
import argparse

def split_multi_task(input_data, dial_turn_map, granularity = 1):
    """
    This func split multi domain/dialogue action sentences into single domain/dialogue act sent
    @param input: dict("sent": {"dial_idx": {"turn_idx": {delex": delexicalized sent, "ori": original sent}}},
                        "feat": {"dial_idx": {"turn_idx": {"da": [slot-values]}}})
    @param dial_turn_map: dict("dial_idx": [turn_indices]) valid dial_idx, turn_idx
                            pair for consideration in this data split
    @return dict with single domain/dialogue act sentences only
    """
    
    def print_struct(input_data, output_data):

        if "sent" not in input_data.keys():
            return

        for dial_idx in input_data["sent"].keys():

            # Print input
            print("INPUT:")
            for turn_idx in input_data["sent"][dial_idx]:
                if "delex" not in input_data["sent"][dial_idx][turn_idx].keys():
                    continue
                print(f"\tOri: {input_data['sent'][dial_idx][turn_idx]['ori']}")
                print(f"\tDelex: {input_data['sent'][dial_idx][turn_idx]['delex']}")
                if turn_idx not in input_data["feat"][dial_idx]:
                    continue
                print(f"\tFeat: {list(input_data['feat'][dial_idx][turn_idx].keys())}")     
                print()
            print("+++++++++++++++++++++++++++++++++++")

            # Print output
            print("OUTPUT:")
            for turn_idx in output_data["sent"][dial_idx]:
                print(f"\tOri: {output_data['sent'][dial_idx][turn_idx]['ori']}")
                print(f"\tDelex: {output_data['sent'][dial_idx][turn_idx]['delex']}")
                print(f"\tFeat: {list(output_data['feat'][dial_idx][turn_idx].keys())}")
                print()
            print("----------------------------------\n")

    input = input_data["sent"]
    feat = input_data["feat"]
    output = defaultdict(dict)
    spliter = re.compile("((.*?)(?:(\. |\? |\! )))")

    count = 0
    for dial_idx in input.keys():
        # Create filtered dialogue

        # Initialize new dialogue holder
        new_turn = 1
        new_dial = defaultdict(dict)

        for turn_idx in input[dial_idx]:
            # Create filtered turn

            if turn_idx not in feat[dial_idx] or turn_idx not in dial_turn_map[dial_idx]:
                continue

            # Get current turn's sents' info
            delex = input[dial_idx][turn_idx]["delex"] + " "
            ori = input[dial_idx][turn_idx]["ori"] + " "
            splitted_delex = [item[0] for item in spliter.findall(delex)]
            splitted_origin = [item[0] for item in spliter.findall(ori)]
            if len(splitted_delex) != len(splitted_origin):
                continue
            current_feat = feat[dial_idx][turn_idx]

            # Directly add single feat sent to new_dial
            if (len(feat[dial_idx][turn_idx]) == 1 and granularity == 1) or \
                 (granularity == 0 and len(set([item.split("-")[0] for item in feat[dial_idx][turn_idx]])) == 1):
                print(f"Adding single feat sent {feat[dial_idx][turn_idx]}")
                new_dial["feat"][str(new_turn)] = current_feat
                new_dial["sent"][str(new_turn)] = dict({"ori": ori,
                                                        "delex": delex})
                new_turn += 1
                count += 1
                continue

            # Get task type of each sent
            task_sent_map = defaultdict(list)
            for i, sent in enumerate(splitted_delex):
                tasks = find_task_type(sent, granularity)
                task_sent_map[tasks].append(i)

            # Eliminate sents with no indicator
            if () in task_sent_map.keys():
                del task_sent_map[()]

            # Eliminate sents containing tasks such that at least one sent contain
            # multiple tasks including this task
            # e.g. sent_1(task_1, task_2), sent_2(task_2), sent_3(task_3). Then only sent_3
            # is keeped
            task_multi = set()
            for key in task_sent_map.keys():
                if (len(key) > 1 and granularity == 1) or \
                     (granularity == 0 and len(set([item.split("-")[0] for item in key])) > 1):
                    task_multi = task_multi.union(set(key))
            key_to_remove = []

            for key in task_sent_map.keys():
                if len(task_multi & set(key)) > 0:
                    key_to_remove.append(key)
            for key in key_to_remove:
                del task_sent_map[key]
            
            # Create feat and original sents correponding to filtered sents
            for key in task_sent_map.keys():

                # Create new sent
                sent_task = key[0]
                new_ori = "".join([splitted_origin[i] for i in task_sent_map[key]])[: -1]
                new_delex = "".join([splitted_delex[i] for i in task_sent_map[key]])[: -1]
                new_sent = dict()
                new_sent["ori"] = new_ori
                new_sent["delex"] = new_delex

                if granularity == 1:
                    sent_task = key[0]
                    new_feat = dict()
                    for feat_key in current_feat.keys():
                        if feat_key.lower() == sent_task.lower():
                            # Create new feat containing slot-value corresponding to current sent's da
                            new_feat[feat_key] = current_feat[feat_key]

                            # Add entry to new dialogue
                            new_dial["feat"][str(new_turn)] = new_feat
                            new_dial["sent"][str(new_turn)] = new_sent
                            new_turn += 1
                            count += 1
                            break
                elif granularity == 0:
                    sent_tasks = key
                    new_feat = dict()
                    for task in sent_tasks:
                        for feat_key in current_feat.keys():
                            if feat_key.lower() == task.lower():
                                new_feat[feat_key] = current_feat[feat_key]
                    
                    new_dial["feat"][str(new_turn)] = new_feat
                    new_dial["sent"][str(new_turn)] = new_sent
                    new_turn += 1
                    count += 1
                    print(f"Adding multi feature sent {new_sent}")
                    print(f"THe features are {new_feat.keys()}")
        
        # Add new dialogue into output
        output["feat"][dial_idx] = new_dial["feat"]
        output["sent"][dial_idx] = new_dial["sent"]
    
    print(f"We obtain {count} outputs")
    return output

def find_task_type(sent, granularity = 1):
    """
    Find domain type if granularity == 0
    else find dialogue act type
    """

    finder = re.compile("(?:slot-)(\w*-\w*-)")
    das = set(finder.findall(sent))
    return tuple(set(map(lambda item: item[: len(item) - 1], # Get rid of the last hyphen
                    das)))

if __name__ == "__main__":
    # Filter sents with multi domain/dialogue-act
    # Truncate or split them if possible

    # Load split type
    parser = argparse.ArgumentParser(description = "Split type")
    parser.add_argument("--split_type", default = "do", type = str, help = "type to split")
    args = parser.parse_args()

    # Define task type
    TASK_TYPE = args.split_type
    granularity = 1 if TASK_TYPE == "da" else 0

    # Load data  
    text_file = "./resource/woz3/text.json"
    feat_file = "./resource/woz3/feat.json"
    data_split_file = "./resource/woz3/data_split/allDataSplitRand0925.json"
    with open(text_file, "r") as f:
        text = json.load(f)
    with open(feat_file, "r") as f:
        feat = json.load(f)
    with open(data_split_file, "r") as f:
        data_split = json.load(f)
        print(f"Considering {data_split_file.split('/')[-1].split('.')[0]}")
    input_data = dict({"sent": text, "feat": feat})
    
    # Check text without delex
    text_without_delex_count = 0
    for dial_idx in text.keys():
        for turn_idx in text[dial_idx].keys():
            if "delex" not in text[dial_idx][turn_idx]:
                text_without_delex_count += 1
    print(f"Text without delex count {text_without_delex_count}")

    # Create mapping for dialogues and turns existing in data split file
    dial_turn_map = defaultdict(set)
    for dtype, turns in data_split.items():
        for dial_idx, turn_idx, _ in turns:
            dial_turn_map[dial_idx].add(turn_idx)

    # Truncate input data
    num_keep = len(input_data["sent"].keys())
    keys = list(input_data["sent"].keys())[: num_keep]
    input_data["sent"] = {k: input_data["sent"][k] for k in keys}
    input_data["feat"] = {k: input_data["feat"][k] for k in keys}

    # Split input
    output_data = split_multi_task(input_data, dial_turn_map, granularity = granularity)

    # Calculate statistics
    input_count = 0 
    for _, turns in dial_turn_map.items():
        input_count += len(turns)
    output_count = 0
    for dial_idx in output_data["sent"].keys():
        output_count += len(output_data["sent"][dial_idx])
    
    print(f"Input count is {input_count}")
    print(f"Output count is {output_count}")

    # Establish data split
    train_dial_indices = set([item[0] for item in data_split["train"]])
    valid_dial_indices = set([item[0] for item in data_split["valid"]])
    test_dial_indices = set([item[0] for item in data_split["test"]])
    total_dial_indices = list(text.keys())

    new_data_split = dict({"train": [], "valid": [], "test": []})
    bad_sent_count = 0
    for dial_idx in output_data["feat"].keys():
        _type = "train" if dial_idx in train_dial_indices else (\
                "valid" if dial_idx in valid_dial_indices else 
                "test"
                )

        # Count dialogue not in data split
        if not (dial_idx in train_dial_indices) and \
            not (dial_idx in valid_dial_indices) and \
            not (dial_idx in test_dial_indices):
            bad_sent_count += 1
            continue
        
        # Add turns of current dialogue to current type in new data split
        for turn_idx in output_data["feat"][dial_idx].keys():
            new_data_split[_type].append([dial_idx, turn_idx, "-"])

    # Write data split, text and feat
    new_data_split_file = f"all_unique_{TASK_TYPE}.json"
    new_text_file = text_file[: len(text_file) - 5] + f"_unique_{TASK_TYPE}.json"
    new_feat_file = feat_file[: len(feat_file) - 5] + f"_unique_{TASK_TYPE}.json"

    print(f"The task type is {TASK_TYPE}")
    with open(new_data_split_file, "w+") as f:
        json.dump(new_data_split, f)
        print(f"Dumping to {new_data_split_file}")
    with open(new_text_file, "w+") as f:
        json.dump(output_data["sent"], f)
        print(f"Dumping to {new_text_file}")
    with open(new_feat_file, "w+") as f:
        json.dump(output_data["feat"], f)
        print(f"Dumping to {new_feat_file}")
