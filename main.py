# TARIMAN, Sienna Ross M.
# CPE215 - D01
# Week 11 - Project

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# CLEANING
# import sympy
from pprint import pprint
from tabulate import tabulate
import numpy as np
import torch, numpy as np, pandas as pd
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)

from IPython.display import display

df = pd.read_csv('train.csv')
# display(df)


print(tabulate(df, headers='keys', tablefmt='psql'))

df.isna().sum()

modes = df.mode().iloc[0]
# print(modes)

df.fillna(modes, inplace=True)
df.isna().sum()

df.describe(include=(np.number))
df['Fare'].hist();
df['LogFare'] = np.log(df['Fare']+1)
df['LogFare'].hist();
pclasses = sorted(df.Pclass.unique())
pclasses
df.describe(include=[object])

df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
# print(df.columns)


added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
# print(df[added_cols].head())

# Create create  independent (predictors) and dependent (target) variables.
from torch import tensor
t_dep = tensor(df.Survived)
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols
t_indep = tensor(df[indep_cols].values, dtype=torch.float)
# print(t_indep)
# print(t_indep.shape)

# Forming the model
torch.manual_seed(442)
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5
# print(coeffs)
t_indep*coeffs
# print(t_indep*coeffs)
vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals
t_indep*coeffs
# print(t_indep*coeffs)
t_indep = t_indep / vals
preds = (t_indep*coeffs).sum(axis=1)
pprint(preds[:10])
loss = torch.abs(preds-t_dep).mean()
# print(loss)
def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()


# Gradient
coeffs.requires_grad_()
loss = calc_loss(coeffs, t_indep, t_dep)
# print(loss)
loss.backward()
# print(coeffs.grad)
loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
# print(coeffs.grad)

loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
with torch.no_grad():
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))

# Training

from fastai.data.transforms import RandomSplitter
trn_split,val_split=RandomSplitter(seed=42)(df)
# print(trn_split)
# print(val_split)
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
print(len(trn_indep))
print(len(val_indep))

# 3 functions
def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()

def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep)
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")

def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()
#

# three functions to train model
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442)
    coeffs = init_coeffs()
    for i in range(epochs): one_epoch(coeffs, lr=lr)
    return coeffs

coeffs = train_model(18, lr=0.2)
#print(coeffs)

def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))

pprint(show_coeffs())

# Accuracy
preds = calc_preds(coeffs, val_indep)
results = val_dep.bool()==(preds>0.5)
print(results[:16])
# Average accuracy
print("Accuracy")
print(results.float().mean())

def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs, val_indep)>0.5)).float().mean()
acc(coeffs)

# Sigmoid
def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))
coeffs = train_model(lr=100)
print(acc(coeffs))
pprint(show_coeffs())