// This should be a registered name in the Transformers library (see https://huggingface.co/models)
// OR a path on disk to a serialized transformer model.
local transformer_model = "distil-bert-base";
local max_length = 512;
local TRAIN_DATA_PATH = std.extVar("TRAIN_DATA_PATH");
local VAL_DATA_PATH = std.extVar("VAL_DATA_PATH");
local TEST_DATA_PATH = std.extVar("TEST_DATA_PATH");
local dropout = 0.5;
local batch_size = 16;
local lr = 3e-6;
local weight_decay = 0.005;
local num_epochs = 30;
local patience = 5;
local grad_norm = 1.0;


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
    "train_data_path": TRAIN_DATA_PATH,
    "validation_data_path": VAL_DATA_PATH,
    "test_data_path": TEST_DATA_PATH,
    "evaluate_on_test": true,
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
            "weight_decay": weight_decay,
            "parameter_groups": [
                # Apply weight decay to pre-trained parameters, exlcuding LayerNorm parameters and biases
                # See: https://github.com/huggingface/transformers/blob/2184f87003c18ad8a172ecab9a821626522cf8e7/examples/run_ner.py#L105
                # Regex: https://regex101.com/r/ZUyDgR/3/tests
                [["(?=.*transformer_model)(?=.*\\.+)(?!.*(LayerNorm|bias)).*$"], {"weight_decay": 0.1}],
            ],
        },
        "patience": patience,
        "num_epochs": num_epochs,
        "grad_norm": grad_norm,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
    },
}
