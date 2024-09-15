# SGPT: A Generative Approach for SPARQL Query Generation from Natural Language Questions
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

PyTorch code for the IEEE Access paper: SGPT: A Generative Approach for SPARQL Query Generation from Natural Language Questions [[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9815253).

### ⚙️ Installation (anaconda)
```commandline
conda create -n sgpt -y python=3.8 && source activate sgpt
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 🏋️ Training
In multiple GPU setting, run the following command:
```bash
python -m torch.distributed.launch train.py --dataset lcquad2 --epochs 40
```
For single GPU:
```
python train.py --dataset lcquad2 --epochs 40
```

- valid dataset names: lcquad2, qald9, vquanda
- For using the masked entity in the question use ```--masked``` for both training and evaluation.
- For taking the additional knowledge (entities) into account ````--knowledge````. Only entities are regarded as additional knowledge.


### 🎯 Evaluation
```
python -u eval.py --generate runs/sgpt/lcquad2/ --dataset lcquad2 --generation_params_file config/gpt-2-base/generation_params.json --eval_dataset test  --output_file outputs/predictions_gpt2-base.json
```

### 🎲 Hyper-paramters
Please try the following number of epochs to find the best results: 10,20,30,40 or 70 and the following learning rates: 6e-4 or 6e-5 .

### 📝 Citation
If you use the code, please cite the following paper.
```
@ARTICLE{
      9815253,  
      author={Rony, Md Rashad Al Hasan and Kumar, Uttam and Teucher, Roman and Kovriguina, Liubov and Lehmann, Jens},
      journal={IEEE Access},   
      title={SGPT: A Generative Approach for SPARQL Query Generation From Natural Language Questions},   
      year={2022},  
      volume={10},  
      number={},  
      pages={70712-70723},  
      doi={10.1109/ACCESS.2022.3188714}
    }
```

### 📜 License
[MIT](https://github.com/rashad101/SGPT-SPARQL-query-generation/blob/main/LICENSE.md)

### 📪 Contact
For further information, contact the corresponding author Md Rashad Al Hasan Rony ([email](mailto:rashad.research@gmail.com)).
