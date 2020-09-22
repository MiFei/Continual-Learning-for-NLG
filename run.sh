mode=$1
config_file=$2
task_seq=0 # only used to specify model to recover or test

if [ -z "${mode}" ];
then 
    mode="train"
fi

if [ -z "${config_file}" ];
then 
    config_file="config/config.cfg"
fi

echo "The mode is ${mode}"

if [ "$mode" == "recover" ] || [ "$mode" == "test" ];
then 
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore run_woz3.py \
        --mode $mode \
        --random_seed 1111 \
        --sv_len_weight 0.5 \
        --adaptive True \
        --ewc_importance 300000 \
        --config_file $config_file \
        --recovered_tasks $task_seq
else
    CUDA_VISIBLE_DEVICES=1 python3 -W ignore run_woz3.py \
        --mode $mode \
        --random_seed 1111 \
        --sv_len_weight 0.5 \
        --adaptive True \
        --ewc_importance 300000 \
        --lr 0.005 \
        --dropout 0 \
        --_lambda 2.0 \
        --config_file $config_file
fi
