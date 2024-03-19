# DposFinder

DposFinder is a transformer-based protein language model for phage-encoded depolymerase prediction.

## Setup

To install DposFinder, follow these steps：

1. Clone the repository:
```
git clone https://github.com/shenyanxiang/DposFinder`
```


2. create a new vertual environment, for example：

```
cd DposFinder
conda env create -f environment.yml
conda activate DposFinder
```

3. Download trained DposFinder model from the link below and put it into `model` directory:

https://tool2-mml.sjtu.edu.cn/DposFinder/model/Final_DposFinder.pt

or you can retrain DposFinder model with optional params by running:
```
python ./code/main.py [--FLAGS]
```
params can be checked by command:
```
python ./code/main.py -h
```

## Usage

Simply use DposFinder to predict depolymerases from protein sequences:
```
python ./code/main.py --mode predict --data_path ./data --test_data test_set.fasta 
```

## Contact

Please contact Yanxiang shen at shenyanxiang@sjtu.edu.cn for questions.