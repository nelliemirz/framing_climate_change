# Transformer for CC Paragraph Framing Classification

Model for the multi-class classification of framing in article paragraphs. Built on top of AllenNLP.

## Prerequisites
```
pip install -r requirements.
```

## Commands

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

1. Create the config as in `./cc_framing/tuning/config.jsonnet`

2. Create a *sweep configuration* file and generate a sweep on the wandb server. Note that the tied parameters that are accepted through environment variables are specified using the prefix `env.` in the sweep config. See example in `./cc_framing/tuning/sweep.yaml`

3. Create the sweep on wandb. E.g.:

```
$ wandb sweep ./cc_framing/tuning/sweep.yaml
```

4. Start the search agent.

```
wandb agent <sweep_id>
```
