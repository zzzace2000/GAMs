# The repository for paper "How Interpretable and Trustworthy are GAMs?"

This repo is to reproduce the result of the paper. To only just use the models on your own datasets, please see the simplified repo https://github.com/zzzace2000/GAMs_models/

## Install

We use the following package:

- interpret: https://github.com/interpretml/interpret (we use the modified version in the folder my_interpret/, so no need to install)
- scikit-learn
- xgboost
- rpy2
- R programming language
- R package flam

You can first install the python package via:

`conda create --name <env> --file ./conda_env.txt`

Then install R and R package flam, if you want to run the FLAM model. You can run ``` install.packages('flam') ``` in R console.

## Model Usage

```
from arch import MyXGBClassifier

model = MyXGBClassifier()
model.fit(X, y)

df = model.get_GAM_plot_dataframe()
df.head()
```

## Paper results

We provide the csv summary of model performance and shape graphs for each GAM in this repo for easier reproducibility.
Pre-trained models are too large to keep, but we could provide them upon requests.

## Running the code to reproduce the paper result

You can see run.sh for all the experimental commands. Here we illustrate how to run our code.

### Section 3.2: the test set AUC on the real data

For example, if we want to run an experiment with XGB classifier with tree depth set as 1 and and bagging with 100 times, then we set the model_name as "xgb-d1-o100".

And we can run the Support2 classification datasets, then we set the d_name as "support2cls2".
(We support datasets "adult", "breast", "churn", "credit", "heart", "support2cls2")


```
model_name='xgb-d1-o100'
d_name='support2cls2'

python -u main.py --identifier 091719_datasets --model_name ${model_name} --d_name ${d_name}

# To summarize the graph for 5 runs and cache the graph mean and stdev in results/
python -u summarize.py --identifier 0927 --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name}
```

After running summarize.py, it creates a folder at "results/0927-091719_datasets-df/" with all the GAM dataframe cached under it. To load and visualize it:

```
import pickle
churn_dict = pickle.load(open('results/0927-091719_datasets-df/churn.pkl', 'rb'))

from vis_utils import vis_main_effects
vis_main_effects(churn_dict, model_names=['ebm-o100-i100-q', 'xgb-d1-o100'])

```


### Section 3.3: measuring how GAMs spread the features (l2-ness)

After running 3.2, we run

```
python feature_importances.py --data_path results/091719_datasets.csv --identifier 0210_add
```

It generates a summary file in "results/0210_add-fimp-AddExp-091719_datasets.tsv". Then follow the notebook "051520 Importance per feature by forward selection using test_test set.ipynb".

### Section 3.4: measuring bias and variance tradeoff

Similar to section 3.2, but we set the experiment mode as "bias_var".
So assume you want to run EBM classifier with outer bagging 100, inner bagging 100 and the quantile binning, then we set the model_name as "ebm-o100-i100-q".

```
model_name='ebm-o100-i100-q'
d_name='adult'

python -u main.py --identifier 112819_bias_var --exp_mode bias_var --model_name ${model_name} --d_name ${d_name}
```

Then run the notebooks/051520 Bias and variance tradeoff - get models to final parameters.ipynb

### Section 4.1/4.2: running semi-synthetic-datasets

To run a churn dataset that's generated from ebm-o100-i100-q with random seed 1377, we set the d_name as "ss_churn_b1_r1377_ebm-o100-i100-q".

Then we follow similar commands in section 3.2.

```
model_name='xgb-d1-o100'
d_name='ss_churn_b1_r1377_ebm-o100-i100-q'

python -u main.py --identifier 1108_ss --model_name ${model_name} --d_name ${d_name}

# To summarize the graph for 5 runs and cache the graph mean and stdev in results/
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv
```

Then follow the "notebooks/051520 Semi-synthetic-new.ipynb" to calculate the graph fidelity.


## Citations

If you find the code useful, please cite:
```
@article{chang2020interpretable,
  title={How Interpretable and Trustworthy are GAMs?},
  author={Chang, Chun-Hao and Tan, Sarah and Lengerich, Ben and Goldenberg, Anna and Caruana, Rich},
  journal={arXiv preprint arXiv:2006.06466},
  year={2020}
}
```
