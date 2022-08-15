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

