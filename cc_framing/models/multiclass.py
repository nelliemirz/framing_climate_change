from typing import Dict, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("multi_class")
class MultiClassClassifier(Model):
    """
    This `Model` implements a basic multi-class text classifier. After embedding the text into a
    text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The resulting
    sequence is pooled using a `Seq2VecEncoder` and then passed to a linear classification layer,
    which projects into the label space. If a `Seq2SeqEncoder` is not provided, we will pass the
    embedded text directly to the `Seq2VecEncoder`.

    Registered as a `Model` with name "multi_class".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`.
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`, optional, (default = `None`)
        Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder will pool its output.
        Otherwise, this encoder will operate directly on the output of the `text_field_embedder`.
        If `None`, defaults to `BagOfEmbeddingsEncoder` with `averaged=True`.
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification
        layer will project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = `"labels"`)
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder = None,
            seq2seq_encoder: Seq2SeqEncoder = None,
            feedforward: Optional[FeedForward] = None,
            dropout: float = None,
            num_labels: int = None,
            label_namespace: str = "labels",
            namespace: str = "tokens",
            initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder

        self._seq2seq_encoder = seq2seq_encoder

        # Default to mean BOW pooler. This performs well and so it serves as a sensible default.
        self._seq2vec_encoder = seq2vec_encoder or BagOfEmbeddingsEncoder(
            text_field_embedder.get_output_dim(), averaged=True
        )

        self._feedforward = feedforward
        if self._feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        self._classification_layer = torch.nn.Linear(in_features=self._classifier_input_dim,
                                                     out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.f1 = FBetaMeasure(average='macro')
        self.f1w = FBetaMeasure(average='weighted')
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(  # type: ignore
            self, tokens: TextFieldTensors, labels: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        labels : `torch.IntTensor`, optional (default = `None`)
            From a `MultiLabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """

        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        probs = torch.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if labels is not None:
            loss = self._loss(logits, labels)
            output_dict["loss"] = loss
            self.f1(logits, labels)
            self.f1w(logits, labels)
            self.accuracy(logits, labels)

        return output_dict

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Thresholds the probabilities, converts indices to string labels, and add `"labels"` key
        to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["labels"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        weighted = self.f1w.get_metric(reset)
        macro = self.f1.get_metric(reset)

        metrics = {
            "accuracy": self.accuracy.get_metric(reset),
            "weighted_precision": weighted["precision"],
            "weighted_recall": weighted["recall"],
            "weighted_fscore": weighted["fscore"],
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_fscore": macro["fscore"],
        }
        return metrics

    default_predictor = "multi_class"

