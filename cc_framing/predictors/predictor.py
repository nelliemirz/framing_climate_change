from typing import List 
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("multi_class")
class MultiClassPredictor(Predictor):
    """
    Predictor for any model that takes in some text and returns a multi-class prediction
    for it. In particular, it can be used with the `MultiClassClassifier(Model)` model.

    Registered as a `Predictor` with name "multi_class".
    """

    def predict(self, text: str) -> JsonDict:
        return self.predict_json({"text": text})
    

    # def _process_output(self, output) -> str:
    #     tokens = output["token_ids"]
    #     text = "".join(tokens).replace("▁", " ").strip()
    #     label = output["labels"]
    #     return text + '\t' + label
    
    # def predict_batch_instance(self, instances: List[Instance]) -> List[str]:
    #     outputs = self._model.forward_on_instances(instances)
    #     return [self._process_output(output) for output in outputs]


    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"text": "..."}`. Runs the underlying model, and adds the
        `"labels"` to the output. Based on:
        https://github.com/allenai/allennlp/blob/master/allennlp/predictors/text_classifier.py
        """
        text = json_dict["text"]
        return self._dataset_reader.text_to_instance(text=text)
