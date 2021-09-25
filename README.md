# srws_psg_torch

## Setup
```Bash
docker build -t srws:latest ./environment
```
```Bash
docker run -it --name srws --gpus all -v $(pwd):/workspace -p 9994:9994 srws:latest /bin/bash
```
