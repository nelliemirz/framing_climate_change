// This should be a registered name in the Transformers library (see https://huggingface.co/models)
// OR a path on disk to a serialized transformer model.
local transformer_model = std.parseJson(std.extVar('transformer_model'));
// Inputs longer than this will be truncated.
// Should not be longer than the max length supported by transformer_model.

local max_length = std.parseJson(std.extVar('max_length'));
local dropout = std.parseJson(std.extVar('dropout'));
local batch_size = std.parseJson(std.extVar('batch_size'));
local lr = std.parseJson(std.extVar('lr'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local grad_norm = std.parseJson(std.extVar('grad_norm'));
local lr_scheduler = std.parseJson(std.extVar('lr_scheduler'));

{"dataset_reader": {
        "type": "json_reader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            // Account for special tokens (e.g. CLS and SEP), otherwise a cryptic error is thrown.
            "max_length": max_length - 2,
        },
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": transformer_model,
            },
        },
    },
    "train_data_path": "../data/frame1_with_rare/train.jsonl",
    "validation_data_path": "../data/frame1_with_rare/test.jsonl",
    "model": {
        "type": "multi_label",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name": transformer_model,
                },
            },
        },
        "dropout": dropout,
    },
    "data_loader": { 
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["tokens"],
            "batch_size" : batch_size,
        },
    },
    
    "trainer": {
        // Set use_amp to true to use automatic mixed-precision during training (if your GPU supports it)
        "use_amp": true,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": lr,
            "weight_decay": 0.0,
            "parameter_groups": [
                # Apply weight decay to pre-trained parameters, exlcuding LayerNorm parameters and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3/tests
                [["(?=.*transformer_model)(?=.*\\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": weight_decay}],
            ],
        },
        "validation_metric": "+weighted_fscore",
        "patience": 10,
        "num_epochs": 50,
        "grad_norm": grad_norm,
        "learning_rate_scheduler": {
            "type": lr_scheduler,
        },

    "callbacks": [
      {
        "type": "wandb_allennlp",
       "files_to_save": ["config.json"]
       },
    ],
   "overrides": {
        "sampler": {
            "type": "weighted_random",
            "weights": [1.76652893, 1.06475716, 1.09475032, 0.9965035, 0.83577713, 0.36663808, 6.47727273, 11.1038961, 0.53237858, 19.43181818, 0.67588933],
            "num_samples": 5,
            "replacement": false
        },
},
},
}
