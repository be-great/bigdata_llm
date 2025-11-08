# LLM

## Methodology
![roject methodology](imgs/flow.png)


## Folder and file structure
```bash
|-setup_env.sh # setup wsl gpu and essential tools
|-init.sh # auto create virtualenv and run scripts
|-pipeline_01_data_process.py # use pyspark to process the bigdata
|-pipeline_02_faiss.py
|-imgs/ # where needed images needed
|-requirements.txt # python requirement
|-data/ # folder of the data
```
## Pipline structure 

1. Data_process = pipeline_01_data_process.py
2. Knowledge base + FAISS GPU index = pipeline_02_faiss.py
3. QLoRA fine-tuning (PEFT + bitsandbytes, GPU) = pipeline_03_qlora.py
4. Local deployment = api_server.py
# concepts :

1. FAISS index: The goal of a FAISS index is to speed up searching by grouping similar data close to each other.
