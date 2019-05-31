# A gentle guild to starting your NLP project with AllenNLP

## Requirements

- Pipenv

## Setup

```bash
git clone https://github.com/yasufumy/allennlp_imdb
cd allennlp_imdb
pipenv install
pipenv shell
```

## Usage

There is a post about this repository: Click [this link](https://towardsdatascience.com/allennlp-startup-guide-24ffd773cd5b)

Running on [Colab](https://colab.research.google.com/drive/1rUAnv2AeTUdpk_VMa8cTc8nv9bOKp-Hx)


Running on CPU:

```bash
allennlp train \
    --include-package allennlp_imdb \
    -s /path/to/serialization \
    training_config/base_cpu.jsonnet
```

Running on GPU:

```bash
allennlp train \
    --include-package allennlp_imdb \
    -s /path/to/serialization \
    -o '{"trainer": {"cuda_device": 0}}' \
    training_config/base_cpu.jsonnet
```

Creating your own configuration file:

```
allennlp configure --include-package allennlp_imdb
```
