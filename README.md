<h1 align="center"><b>MoleRec</b></h1>

Official implementation for our paper:

*MoleRec: Combinatorial Drug Recommendation with Substructure-Aware Molecular Representation Learning*

Nianzu Yang, Kaipeng Zeng, Qitian Wu, Junchi Yan* (* denotes correspondence)

*Proceedings of the ACM Web Conference 2023* (**TheWebConf (a.k.a. WWW) 2023**)

## Folder Specification

- `data/` folder contains necessary data or scripts for generating data.
  - `drug-atc.csv`, `ndc2atc_level4.csv`, `ndc2rxnorm_mapping.txt`: mapping files for drug code transformation
  - `atc2rxnorm.pkl`: It maps ATC-4 code to rxnorm code and then query to drugbank.
  - `idx2SMILES.pkl`: Drug ID (we use ATC-4 level code to represent drug ID) to drug SMILES string dictionary.
  - `drug-DDI.csv`: A file containing the drug DDI information which is coded by CID.
  - `ddi_mask_H.pkl`:  A mask matrix containing the relations between molecule and substructures. If drug molecule $i$ contains substructure $j$, the $j$-th column of $i$-the row of the matrix is set to 1.
  - `substructure_smiles.pkl`: A list containing the smiles of all the substructures.
  - `ddi_mask_H.py`: The python script responsible for generating `ddi_mask_H.pkl` and `substructure_smiles.pkl`.
  - `processing.py`: The python script responsible for generating `voc_final.pkl`, `records_final.pkl`, `data_final.pkl` and `ddi_A_final.pkl`.    
- `src/` folder contains all the source code.
  - `modules/`: Code for model definition.
  - `utils.py`: Code for metric calculations and some data preparation.
  - `training.py`: Code for the functions used in training and evaluation.
  - `main.py`: Train or evaluate our MoleRec Model.

**Remark:** `data/` only contains part of the data. See the [Data Generation](#data-generation) section for more details.


## Dependency
The `MoleRec.yml` lists all the dependencies of the MoleRec. To quickly set up a environment for our model, use the following command

```shell
conda env create -f MoleRec.yml
```

## Data Generation

The usage of MIMIC-III datasets requires certification, so it's illegal for us to provide the raw data here. Therefore, if you want to have access to MIMIC-III datasets, you have to obtain the certification first and then download it from  [https://physionet.org/content/mimiciii/](https://physionet.org/content/mimiciii/). 

After downloading the MIMIC-III dataset, put the three csv file `PRESCRIPTIONS.csv`, `DIAGNOSES_ICD.csv` and `PROCEDURES_ICD.csv` from the raw data into the `data/` folder and generate the necessary files for training and evaluating apart from the files that we already have provided in thte `data/` folder, using the command as below: 

```shell
cd data
python processing.py
```

For the explanation of each output file, please refer to the [SafeDrug](https://github.com/ycq091044/SafeDrug) repository. Note that in our paper, we follow the same data processing procedure as the SafeDrug
after the commit [c7218d0](https://github.com/ycq091044/SafeDrug/tree/c7218d0976e5ee5588aeaf5bdbc86b338126bba5).

If you want to re-generate `ddi_matrix_H.pkl` and `substructure_smiles.pkl`, use the following command:
```shell
cd data
python ddi_mask_H.py
```
Note that the BRICS decomposition method generates substructures in a random order. Since that `ddi_matrix_H.pkl` and `substructure_smiles.pkl` are effected by this order, if you re-generate these two files, please re-train the model. For convenience, we've already provided the generated result by us in `data/` folder, which can be used for training and evaluating directly.


## Run the Code
We provide two versions of our model. They learn the substructure representations using embedding table and GNNs, respectively. If you want to train or evaluate our model, please change your working directory first via：
```shell
cd src
```

### Embedding Table Version

To train the model, use the following command:

```shell
python main.py --device ${device} --embedding --lr ${learning rate} --dp ${dropout rate} --dim ${dim} --target_ddi ${expected ddi} --coef ${coefficient of annealing weight} --epochs ${epochs}
```

To evaluate a well-trained model, use the following command:
```shell
python main.py --Test --embedding --resume_path ${model_path}
```

We've provide our well-trained model in folder `best_models/`, to evaluate it, use the command
```shell
python main.py --Test --embedding --resume_path ../best_models/embedding_table/MoleRec.model
```

### GNNs Version

This version learns the substructure representation using GNNs, which is more powerful but has more parameters. You can use the following command to train the model:

```shell
python main.py --device ${device} --lr ${learning rate} --dp ${dropout rate} --dim ${dim} --target_ddi ${expected ddi} --coef ${coefficient of annealing weight} --epochs ${epochs}
```

To evaluate a well-trained model, use the following command:
```shell
python main.py --Test --resume_path ${model_path}
```

We also provide a well-trained model weight for this version, which can be evaluated by:
```shell
python main.py --Test --resume_path ../best_models/GNN/MoleRec.model
```

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{yang2023molerec,
      title = {MoleRec: Combinatorial Drug Recommendation with Substructure-Aware Molecular Representation Learning},
      author = {Nianzu Yang and Kaipeng Zeng and Qitian Wu and Junchi Yan},
      booktitle = {Proceedings of the ACM Web Conference 2023},
      year = {2023}
}
```
Welcome to contact us [yangnianzu@sjtu.edu.cn](mailto:yangnianzu@sjtu.edu.cn) or [zengkaipeng@sjtu.edu.cn](mailto:zengkaipeng@sjtu.edu.cn) for any question.

## Acknowledgement
We sincerely thank this repository [SafeDrug](https://github.com/ycq091044/SafeDrug) for its well-implemented pipeline upon which we build our codebase.
