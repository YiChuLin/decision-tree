# Decision Tree
Decision Tree project for Introduction to Machine Learning Course

## Prerequisites
- Linux
- Windows
- Python2.7.15+
(The code is tested on Ubuntu 18.04.3 LTS with python version 2.7.15+, 3.6.8)
## Getting Started
install numpy and matplotlib if not already installed
```
pip install --upgrade numpy, matplotlib
```
clone this repository:
```
git clone https://gitlab.doc.ic.ac.uk/al4419/decision-tree.git
cd decision_tree
```
To train the decision tree with a given dataset, run:
```
python train.py path/to/dataset.txt
```
train.py will output the confusion matrix as well as important measures for both pruned and unpruned tree.
Two visualization figures would be created for the pruned and unpruned tree.
- Pruned.png
- Unpruned.png

Below is the output to result.txt of running
```
python train.py wifi_db/clean_dataset.txt > result.txt
```

```
Performace on the dataset: wifi_db/noisy_dataset.txt
-------------Unpruned Average Performance----------------------
Precision:
        Class 1: 0.808
        Class 2: 0.829
        Class 3: 0.817
        Class 4: 0.823
Recall:
        Class 1: 0.839
        Class 2: 0.811
        Class 3: 0.825
        Class 4: 0.804
F Score:
        Class 1: 0.823
        Class 2: 0.820
        Class 3: 0.821
        Class 4: 0.813
The average accuracy = 0.820
--------------Pruned Average Performance-----------------------
Precision:
        Class 1: 0.886
        Class 2: 0.871
        Class 3: 0.831
        Class 4: 0.857
Recall:
        Class 1: 0.853
        Class 2: 0.841
        Class 3: 0.866
        Class 4: 0.886
F Score:
        Class 1: 0.869
        Class 2: 0.856
        Class 3: 0.848
        Class 4: 0.871
The average accuracy = 0.861
```

The generated figures are as follows:
![Unpruned.png](img/Unpruned.png?raw=true "Unpruned.png")
![Pruned.png](img/Pruned.png?raw=true "Pruned.png")

## Authors
- Arvin Lin (al4419@imperial.ac.uk)
- YiChong Chen
- 
