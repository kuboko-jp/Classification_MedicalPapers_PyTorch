# Medical text classification with PyTorch BERT



This source code and analysis was developed during my participation in [the Systematic Review Work Shop-Peer Support Group (SRWS-PSG)](https://signate.jp/competitions/471) held at [SIGNATE]((https://signate.jp/)).

## PubMedBERT
This program uses [microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) as part of its AI model, and has been granted an [MIT Licence](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/blob/main/LICENSE.md).


## Setup
```Bash
docker build -t srws:latest ./environment
```
```Bash
docker run -it --name srws --gpus all -v $(pwd):/workspace srws:latest /bin/bash
```
