## RoseNER:Robust Semi-supervised Named Entity Recognition on Insufficient Labeled Data
RoseNER is a named entity recognition framework.
The paper has been accepted in IJCKG 2021.
### Environment
```python
python3.7
pytorch==1.6.0 +
transformers==2.10.0
```
### Data
```python
BC5CDR
NCBI-Diease
```
### PretrainModel
```python
BERT: bert-base-cased
https://huggingface.co/bert-base-cased/tree/main
Local:./bert/torch_bert
```
### Train&Test
```python
bash run.sh
```
