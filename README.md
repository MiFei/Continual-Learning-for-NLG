# Continual Learning for Natural Language Generation

## Getting started

The incremental folder contains the dataset, source code and configurations to reproduce our results.


### Prerequisites

The following modules are necessary to run the experiments

* [nltk 3.4.5](http://www.nltk.org/) `conda install -c anaconda nltk`
* [PyTorch 1.3.1](https://pytorch.org/) `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
* [pandas 0.20.3](https://pandas.pydata.org/) `conda install -c anaconda pandas`
* [numpy 1.11.3](https://numpy.org/) `conda instal -c anaconca numpy`


### Code structure
```
.
├── model/                      # The model implementation containing the sclstm layers  
│  ├── lm_deep.py               # SCLSTM basedimplementation of utterance generation
│  ├── cvae.py                  # SCVAE based implementation of utterance generation
│  ├── masked_cross_entropy.py  # Masked cross entropy loss
│  ├── masked_kl_divergence.py  # Masked KL-divergence loss
│  ├── model_util.py            # Some helper functions of model
│  ├── layers/                  # The unique domain data split
│  │   ├── sclstm.py            # SCLSTM network
│  │   ├── encoder.py           # Encoder for SCVAE
│  │   └── decoder_deep.py      # SCLSTM based decoder
├── loader                      # Dataset loader
│  ├── dataset_woz3.py          # Loader of original dataset
│  └── task.py                  # Loader of task and implementation of examplars
├── resource/                   # Data source
│  ├── data_split/*.json        # Data split file
│  ├── feat_*.json              # Feature file containing dialogue acts and slot-values
│  ├── text_*.txt               # Text file with original and delexicalized utterances
│  ├── vacab.txt                # Vocabulary
│  └── template.txt             # The template containing possible domain-action-slot-value
├── ewc.py                      # Compute ewc penalty
├── bleu.py                     # Compute bleu score
├── run_woz3.py                 # Code for starting training, evaluating and testing
├── run.sh                      # Script to run experiment
├── config/*.cfg                # Experiment configuration 
├── experiments/experiment_name # Model and results of experiments
│  ├── model/                   # Models throughout the training of task sequence, notice that the intermediate model.pt contains the test results on all tasks till current task
│  ├── examplars/               # Examplars of each task
│  ├── experiment_name.log      # Training log
│  └── experiment_name.res      # Testing result
├── view_results.ipynb          # Summarize statistics of experiments
└── 
...
```


#### Experiment helper functions
```
.
├── split_multi_task.py                 # Script for generating utterances with unique task (domain/dialogue act), run this only if you need to re-generate feat.json, text.json and data split
├── construct_examplar.py               # Construct exemplars with `herding`, `random` or `prioritized (loss)` based method
└── 
...
```

### Run the code

1. The default datasets have already been available in `resource`. However, to generate dataset from scratch, please run `bash preprocess.sh ${split_type}` to generate single domain text, feature and datasplit json file. The `split_type` can be either `do` for unique-domain utterances or `da` for unique dialogue-act utterances.

2. Create your configuration file, the default is `config/config.cfg`. Please go to `config/` for detailed explanations.

3. Run `bash run.sh ${train/test/recover} ${config_file}` to reproduce our results or conduct extensive experiments.

p.s. Default settings are provided, but hyper-parameters of different methods can be set as comman line arguments to run_woz3.py


### Check the result
- Continual learning results in each task are stored in `experiments/${experiment_name}/*.log`.

- To check aggregated results in our paper, run the notebook `view_results.ipynb`

- The generated utterances after running test mode are stored in `experiments/${experiment_name}/*.res`.

