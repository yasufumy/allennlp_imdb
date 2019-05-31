{
  "dataset_reader": {
    "type": "imdb",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "tokenizer": {
        "type": "word"
    }
  },
  "train_data_path": "train",
  "test_data_path": "test",
  "evaluate_on_test": true,
  "model": {
    "type": "rnn_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": false
        }
      }
    },
    "seq2vec_encoder": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1
    },
    "dropout": 0.2
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 64
  },

  "trainer": {
    "num_epochs": 10,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
    }
  }
}
