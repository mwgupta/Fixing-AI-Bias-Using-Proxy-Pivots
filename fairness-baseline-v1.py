# IMPORTS ####################################################
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'figure.max_open_warning': 0})

import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from scipy.stats import entropy
from sklearn import metrics

torch.manual_seed(1)
np.random.seed(7)
sns.set(style="white", palette="muted", color_codes=True, context="talk")

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

file1 = open("output/lambda/baseline-v1/output_baseline.txt","w")

# HELPER FUNCTIONS ####################################################

def load_ICU_data(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  .loc[lambda df: df['race'].isin(['White', 'Black'])])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(race=lambda df: (df['race'] == 'White').astype(int),
                 sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['target', 'race', 'sex', 'fnlwgt'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    
    return X, y, Z

def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.minimum(odds, 1/odds) * 100

def plot_distributions(y_true, Z_true, y_pred, name, image_name, Z_pred = None, epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    subplot_df = (
        Z_true
        .assign(race=lambda x: x['race'].map({1: 'white', 0: 'black'}))
        .assign(sex=lambda x: x['sex'].map({1: 'male', 0: 'female'}))
        .assign(y_pred=y_pred)
    )
    _subplot(subplot_df, 'race', ax=axes[0])
    _subplot(subplot_df, 'sex', ax=axes[1])
    _performance_text(fig, y_true, Z_true, y_pred, name, image_name, protected, Z_pred, epoch)
    fig.tight_layout()
    return fig

def _subplot(subplot_df, col, ax):
    for label, df in subplot_df.groupby(col):
        sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)
    ax.set_title(f'Sensitive attribute: {col}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_yticks([])
    ax.set_ylabel('Prediction distribution')
    ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(col))

def _performance_text(fig, y_test, Z_test, y_pred, name, image_name, protected, Z_pred=None, epoch=None):

    file1.writelines(f"{image_name}_{name} \n")
    file1.writelines("\n")
    
    clf_roc_auc = metrics.roc_auc_score(y_test, y_pred)
    clf_accuracy = metrics.accuracy_score(y_test, y_pred > 0.5) * 100
    p_rules = {'race': p_rule(y_pred, Z_test['race']),
               'sex': p_rule(y_pred, Z_test['sex']),}
    fig.text(1.0, 0.65, '\n'.join(["Classifier performance:",
                                   f"- ROC AUC: {clf_roc_auc:.2f}",
                                   f"- Accuracy: {clf_accuracy:.1f}"]),
             fontsize='16')
    fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                 [f"- {attr}: {p_rules[attr]:.0f}%-rule"
                                  for attr in p_rules.keys()]),
             fontsize='16')
    if Z_pred is not None:
        adv_roc_auc = metrics.roc_auc_score(Z_test, Z_pred)
        fig.text(1.0, 0.20, '\n'.join(["Adversary performance:",
                                       f"- ROC AUC: {adv_roc_auc:.2f}"]),
                 fontsize='16')
        
def average(my_list):
    total = 0
    for i in my_list: total += i
    return total/len(my_list)

# LOAD DATA ####################################################

# load ICU data set
X, y, Z = load_ICU_data('data/adult.data')

n_features = X.shape[1]
n_sensitive = Z.shape[1]

# split into train/test set
(X_train, X_test, y_train, y_test,
 Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.5,
                                     stratify=y, random_state=7)

# standardize the data
scaler = StandardScaler().fit(X_train)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), 
                                           columns=df.columns, index=df.index)
X_train = X_train.pipe(scale_df, scaler) 
X_test = X_test.pipe(scale_df, scaler)

class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()

train_data = PandasDataSet(X_train, y_train, Z_train)
test_data = PandasDataSet(X_test, y_test, Z_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)

file1.writelines([f"# training samples: {len(train_data)} \n",f"# batches: {len(train_loader)} \n"])
file1.writelines("\n")

file1.writelines(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes \n")
file1.writelines(f"targets y: {y.shape} samples \n")
file1.writelines(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes \n")
file1.writelines("\n")

# PRETRAIN CLASSIFIER ####################################################

class Classifier(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))

clf = Classifier(n_features=n_features)
clf_criterion = nn.BCELoss()
clf_optimizer = optim.Adam(clf.parameters())

def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y, _ in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        loss = criterion(p_y, y)
        loss.backward()
        optimizer.step()
    return clf


N_CLF_EPOCHS = 2

for epoch in range(N_CLF_EPOCHS):
    clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)

# PRETRAIN ADVERSARY ####################################################

class Adversary(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


def pretrain_adversary(adv, clf, data_loader, optimizer, criterion):
    for x, _, z in data_loader:
        p_y = clf(x).detach()
        adv.zero_grad()
        p_z = adv(p_y)
        loss = (criterion(p_z, z) * lambdas).mean()
        loss.backward()
        optimizer.step()
    return adv


lambdas = torch.Tensor([100, 100])
adv = Adversary(Z_train.shape[1])
adv_criterion = nn.BCELoss(reduce=False)
adv_optimizer = optim.Adam(adv.parameters())


N_ADV_EPOCHS = 5

for epoch in range(N_ADV_EPOCHS):
    pretrain_adversary(adv, clf, train_loader, adv_optimizer, adv_criterion)

# TRAIN AND TEST ####################################################

with torch.no_grad():
    pre_clf_test = clf(test_data.tensors[0])
    pre_adv_test = adv(pre_clf_test)


y_pre_clf = pd.Series(pre_clf_test.data.numpy().ravel(),
                      index=y_test.index)
y_pre_adv = pd.DataFrame(pre_adv_test.numpy(), columns=Z.columns)

fig = plot_distributions(y_test, Z_test, y_pre_clf, y_pre_adv)
fig.savefig('output/lambda/baseline_v1/baseline.png')


def train(clf, adv, data_loader, clf_criterion, adv_criterion,
          clf_optimizer, adv_optimizer, lambdas):
    
    # Train adversary
    adv_losses = []
    for x, y, z in data_loader:
        p_y = clf(x)
        adv.zero_grad()
        p_z = adv(p_y)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        adv_losses.append(loss_adv)
        loss_adv.backward()
        adv_optimizer.step()
 
    # Train classifier on single batch
    clf_losses = []
    entropies = []
    for x, y, z in data_loader:
        pass
    p_y = clf(x)
    p_z = adv(p_y)
    for p_z_i in p_z:
        entropies.append(entropy(p_z_i.detach()))
    clf.zero_grad()
    p_z = adv(p_y)
    loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
    just_clf_loss = clf_criterion(p_y, y)
    clf_loss = just_clf_loss - loss_adv
    clf_losses.append(just_clf_loss)
    clf_loss.backward()
    clf_optimizer.step()
    
    return clf, adv, average(adv_losses), average(clf_losses), average(entropies)

plt_adv_loss = []
plt_clf_loss = []
z_entropies = []

N_EPOCH_COMBINED = 164

for epoch in range(1, N_EPOCH_COMBINED):
    
    clf, adv, adv_loss, clf_loss, z_entropy = train(clf, adv, train_loader, clf_criterion, adv_criterion,
                     clf_optimizer, adv_optimizer, lambdas)

    plt_adv_loss.append(adv_loss)
    plt_clf_loss.append(clf_loss)
    z_entropies.append(z_entropy)
    
    with torch.no_grad():
        clf_pred = clf(test_data.tensors[0])
        adv_pred = adv(clf_pred)

    y_post_clf = pd.Series(clf_pred.numpy().ravel(), index=y_test.index)
    Z_post_adv = pd.DataFrame(adv_pred.numpy(), columns=Z_test.columns)
    
    fig = plot_distributions(y_test, Z_test, y_post_clf, Z_post_adv, epoch)
    display.clear_output(wait=True)

plt.figure()
plt.plot(z_entropies)
plt.savefig('output/lambda/baseline_v1/entropy_fig.png')

plt.figure()
plt.plot(plt_clf_loss)
plt.savefig('output/lambda/baseline_v1/clf_loss_fig.png')

plt.figure()
plt.plot(plt_adv_loss)
plt.savefig('output/lambda/baseline_v1/adv_loss_fig.png')

file1.close()
