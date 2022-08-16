# Transformer for CC Paragraph Framing Classification

Model for the multi-class classification of framing in article paragraphs. Built on top of AllenNLP.

## Prerequisites
```
pip install -r requirements.
```


## Commands


#### cc_framing/data/preprocess.py
Script for preprocessing the dataset. Take's path to a file in tsv format, output's the preprocessed data in jsonl format.


| Argument           | Required | Default | Description                                                      |
|:-------------------|:----------|:--------|:-----------------------------------------------------------------|
| --path             | true      |         | path to dataset                                                  |
| --para-frame       | true      |         | which para-framing to process: 1 or 2                            |
| --split            | false     | false   | whether to split data into train/test                            |
| --val              | false     | false   | whether to additionaly split test set into test/val              |  
| --test-size        | false     | 0.2     | size of a test set                                               |
| --drop-rare        | false     | false   | whether to remove rare labels                                    |
| --min-size         | false     | 20      | the minimal number of times a label should appear in the dataset (used together with --drop-rare          |
| --top3-split       | false     | false   | split the dataset into two: one containing top 3 most frequently appearing labels, another - the rest of the labels |
| --no-prep          | false     | false      | to not do preprocessing                                       |



#### train.sh
Script for training a model based on AllenNLP `train` command.

| Argument | Required | Description                                 |
|:---------|:---------|---------------------------------------------|
| -c       | true     | path to file with configuration             |
| -s       | true     | path to directory where model will be saved |
| -t       | true     | path to train dataset                       |
| -v       | true     | path to val dataset                         |
| -e       | true     | path to test/eval dataset                   |
| -r       | false    | recover from checkpoint                     |

#### predict.sh

Script for model evaluation. 
Returns metrics for the whole test set as well as for each of the labels, including accuracy, weighted precision, weighted recall, weighted f1 score, macro precision, macro recall,  macro f1 score.
Saves the predicted labels into a separate file. 

| Argument | Required | Default | Description                                                      |
|:---------|:---------|:--------|:-----------------------------------------------------------------|
| -t       | true     |         | path to test dataset                                             |
| -m       | true     |         | path to tar.gz archive with model                                |
| -p       | true     |         | name of Predictor                                                |
| -c       | false    | 0       | CUDA device                                                      |                                          |
| -b       | false    | 32      | size of a batch with test examples to run simultaneously         |

## WandB

#### Installation

```
$ pip install wandb-allennlp
$ echo wandb_allennlp >> .allennlp_plugins
```

#### Logging 

Add a trainer callback in your config file. Use one of the following based on your AllenNLP version:

```
...,
trainer: {
    type: 'callback',
    callbacks: [
      ...,
      {
        type: 'wandb_allennlp',
        files_to_save: ['config.json'],
        files_to_save_at_end: ['*.tar.gz'],
      },
      ...,
    ],
    ...,
}
...
...
```

Execute the `train.sh` script. Enter wandb credentials when prompted.

#### Hyperparameter Search

1. Create the config as in `./cc_framing/tuning/config-bert.jsonnet`

2. Create a *sweep configuration* file and generate a sweep on the wandb server. Note that the tied parameters that are accepted through environment variables are specified using the prefix `env.` in the sweep config. See example in `./cc_framing/tuning/sweep.yaml`

3. Create the sweep on wandb. E.g.:

```
$ wandb sweep ./cc_framing/tuning/sweep.yaml
```

4. Start the search agent.

```
wandb agent <sweep_id>
```
