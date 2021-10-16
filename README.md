# Medical text classification with PyTorch BERT


本ソースコード及び分析結果は、[SIGNATE](https://signate.jp/)で開催された[Systematic Review Work Shop-Peer Support Group (SRWS-PSG)](https://signate.jp/competitions/471) に参加した際に開発したものです。<br>
This source code and analysis was developed during my participation in [the Systematic Review Work Shop-Peer Support Group (SRWS-PSG)](https://signate.jp/competitions/471) held at [SIGNATE]((https://signate.jp/)).

## PubMedBERT
本プログラムでは、[microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)を使用しています。<br>
This program uses [microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).


## Setup
```Bash
docker build -t srws:latest ./environment
```
```Bash
docker run -it --name srws --gpus all -v $(pwd):/workspace -p 9994:9994 srws:latest /bin/bash
```
