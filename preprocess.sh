# This script run split_multi_task.py to prepare the dataset for continual learning
# If the first command line argument is do, then unique-domain utterances are created
# Elif the first command line argument is da, the unique-dialogue-act utterances are created

split_type=$1
echo "Create dataset with unique ${split_type}"

python3 split_multi_task.py --split_type=$split_type