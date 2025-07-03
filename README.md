# DeepMLP

DeepMLP: a proteomics-driven deep learning framework for identifying mis-localized proteins across pan-cancer



## Install dependencies



- Clone the repository.

```shell
git clone https://github.com/wangbing587/DeepMLP.git
```




- The  package is developed based on the Python libraries [torch](https://pytorch.org/get-started/locally/) and [torch-geometric](https://pypi.org/project/torch-geometric/) (*PyTorch Geometric*) framework, and can be run on GPU (recommend) or CPU.

(i)  torch (CPU) 

```shell
pip3 install torch torchvision torchaudio

```

(ii) torch  (GPU)

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

- Install other the required packages.

```shell
pip install -r requirement.txt
```


- Required package list

```shell
pandas>=2.2.2
numpy>=1.26.4
torch-geometric==2.6.1
gseapy>=1.1.9
networkx>=3.2.1
iterative-stratification>=0.1.9
scikit-learn>=1.4.2
```

## Proteomics Data  format

- the Proteomics Data format must be .csv (comma-separated values).
- the first column must be GeneSymbol.
- For a detailed format, refer to  **COAD_Protein.csv**.
- the sample annotations Data format must be .csv (comma-separated values).
- sample annotations,  column = ['Sample_id', 'Type']", where 'Type' must be either 'Normal' or 'Tumor'.
- For a detailed format, refer to  **COAD_an_col.csv**.


## DeepMLP help

the DeepMLP.py in DeepMLP file is a python script (python==3.12.4) for convenience using DeepMLP by Command node

The introduction of the DeepMLP parameters can be achieved by naming them as follows

```shell
cd DeepMLP
python DeepMLP.py -h
```

Output

```shell
$ python DeepMLP.py -h
usage: DeepMLP.py [-h] [-f F] [-a A] [-p P] [-n N] [-e E]

DeepMLP: A proteomics-driven deep learning framework for identifying mis-
localized proteins across pan-cancer.

options:
  -h, --help            show this help message and exit
  -f F, --file F        Path to the input data file. The first column must
                        contain unique GeneSymbols.
  -a A, --an_col A      Path to the sample annotation file. It must contain
                        columns: ['Sample_id', 'Type'].
  -p P, --proportion P  Proportion threshold for selecting loss-type and gain-
                        type MLPs. Default is 0.03.
  -n N, --n_splits N    Number of folds for cross-validation. Default is 5.
  -e E, --epochs E      Number of training epochs. Default is 500.

```





## Run DeepMLP

the DeepMLP.py can be run from the command line interface with the following commands, where -f (dataset file) and -a (sample annotation file) are the two required parameters. An example command is “python DeepMLP.py -f COAD_Protein.csv -a COAD_an_col.csv”, in which “-f” is input dataset file and “-a” is sample annotation file. 

```shell
python DeepMLP.py -f COAD_Protein.csv -a an_col.csv
```



If there are any problems, please contact me.

Bing Wang, E-mail: wangbing587@163.com
