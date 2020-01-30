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
from torch.autograd import Variable
from sklearn.cluster import KMeans

import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import statistics
from statistics import mean
from scipy.stats import entropy

import pdb

torch.manual_seed(1)
np.random.seed(7)
sns.set(style="white", palette="muted", color_codes=True, context="talk")

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

n_clusters = 5
file1 = open("output/output_n/output_n_{}/output_c1_{}.txt".format(n_clusters, n_clusters),"w") 

# LOAD DATA ####################################################

def load_ICU_data(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python'))
    
    # targets: 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # x featues: including protected classes
    X = (input_data
         .drop(columns=['target', 'fnlwgt'])
         .fillna('Unknown')
         .pipe(pd.get_dummies))

    # z features:
    Z_race = X[['race_Amer-Indian-Eskimo',
          'race_Asian-Pac-Islander',
          'race_Black',
          'race_Other',
          'race_White']]
    Z_race = Z_race.rename(columns = {'race_Amer-Indian-Eskimo':0,
                              'race_Asian-Pac-Islander':1,
                              'race_Black':2,
                              'race_Other':3,
                              'race_White':4})

    Z_sex = X[['sex_Female','sex_Male']]
    Z_sex = Z_sex.rename(columns = {'sex_Female':0,'sex_Male':1})

    X = X.drop(columns = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other','race_White', 'sex_Female','sex_Male'])

    n_clusters = 2
    Kmean1 = KMeans(n_clusters=n_clusters)
    Kmean1.fit(X)
    b = np.zeros((X.shape[0],n_clusters))
    b[np.arange(X.shape[0]),Kmean1.labels_] = 1
    Z_c1 = pd.DataFrame(b)

    n_clusters2 = 4
    Kmean2 = KMeans(n_clusters=n_clusters2)
    Kmean2.fit(X)
    c = np.zeros((X.shape[0],n_clusters2))
    c[np.arange(X.shape[0]),Kmean2.labels_] = 1
    Z_c2 = pd.DataFrame(c)
    
    return X, y, Z_race, Z_sex, Z_c1, Z_c2

# load ICU data set
X, y, Z_race, Z_sex, Z_c1, Z_c2 = load_ICU_data('data/adult.data')

# split into train/test set
(X_train, X_test, y_train, y_test, _, Z_test_race, _, Z_test_sex, Z_train_c1, Z_test_c1, _, Z_test_c2) = train_test_split(X, y, Z_race, Z_sex, Z_c1, Z_c2,
                                                                                                                          test_size=0.5, stratify=y, random_state=7)
Z_sets = [Z_test_sex, Z_test_race, Z_test_c1, Z_test_c2]
Z_sets_names = ['Z_test_sex', 'Z_test_race', 'Z_test_c1', 'Z_test_c2']

file1.writelines([f"features X: {X.shape[0]} samples, {X.shape[1]} attributes \n",
                  f"targets y: {y.shape} samples \n",
                  f"sensitives Z_train_c1: {Z_train_c1.shape[0]} samples, {Z_train_c1.shape[1]} attributes \n",
                  f"sensitives Z_test_race: {Z_test_race.shape[0]} samples, {Z_test_race.shape[1]} attributes \n",
                  f"sensitives Z_test_sex: {Z_test_sex.shape[0]} samples, {Z_test_sex.shape[1]} attributes \n",
                  f"sensitives Z_test_c1: {Z_test_c1.shape[0]} samples, {Z_test_c1.shape[1]} attributes \n",
                  f"sensitives Z_test_c2: {Z_test_c2.shape[0]} samples, {Z_test_c2.shape[1]} attributes \n"])

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

train_data = PandasDataSet(X_train, y_train, Z_train_c1)
test_data = PandasDataSet(X_test, y_test, Z_test_c1, Z_test_c2, Z_test_race, Z_test_sex)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)

file1.writelines([f"# training samples: {len(train_data)} \n",f"# batches: {len(train_loader)} \n"])
file1.writelines("\n")

# HELPER FUNCTIONS ####################################################
def p_rule(y_pred, protected, Z_test, threshold = 0.5):
    p_rule_out = []
    for i in range(Z_test.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            y_z_i = y_pred[np.array(protected) == i] > threshold if threshold else y_pred[np.array(protected) == i]
            y_z_all = y_pred[np.array(protected) != i] > threshold if threshold else y_pred[np.array(protected) != i]
        odds = y_z_i.mean() / y_z_all.mean()
        p_rule_out.append(min(odds, 1./odds) * 100)
    return  p_rule_out

def plot_distributions(y_true, Z_true, y_pred, name, image_name, Z_pred = None, epoch=None):
    fig, axes = plt.subplots(figsize=(8, 5))
    
    protected = []
    for _, row in Z_true.iterrows():
        protected.append(row[row == 1].index[0])
    
    subplot_df = (
        Z_true
        .assign(protected=protected)
        .assign(y_pred=y_pred))
    _subplot(subplot_df, protected,
             name, ax = axes)
    _performance_text(fig, y_true, Z_true, y_pred, name, image_name, protected, Z_pred, epoch)
    fig.tight_layout()
    return fig

def _subplot(subplot_df, col, name, ax):
    for label, df in subplot_df.groupby(col):
        sns.kdeplot(df['y_pred'].fillna(0), ax=ax, label=df['protected'].iloc[0], shade=True)
    ax.set_title(f'Sensitive attribute: {name}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_yticks([])
    ax.set_ylabel('Prediction distribution')
    ax.set_xlabel(r'$P({{income>50K}}|z_{{{}}})$'.format(name))

def _performance_text(fig, y_test, Z_test, y_pred, name, image_name, protected, Z_pred=None, epoch=None):

    file1.writelines(f"{image_name}_{name} \n")
    file1.writelines("\n")
    
    clf_roc_auc = metrics.roc_auc_score(y_test, y_pred)
    clf_accuracy = metrics.accuracy_score(y_test, y_pred > 0.5) * 100
    p_rule_out = p_rule(y_pred, protected, Z_test)
    for i in range(Z_test.shape[1]):
        file1.writelines(f"Class {i}: {round(p_rule_out[i],2)} \n")
        
    file1.writelines([f"overall_min: {round(np.array(p_rule_out).min(),2)} \n", f"overall_mean: {round(np.array(p_rule_out).mean(),2)} \n"])

    file1.writelines("\n")
    file1.writelines(["Classifier performance: \n",f"- ROC AUC: {clf_roc_auc:.2f} \n", f"- Accuracy: {clf_accuracy:.1f} \n"])
    file1.writelines("\n")
    
    if Z_pred is not None:
        adv_roc_auc = multiclass_roc_auc_score(Z_test, Z_pred)
        file1.writelines(["Adversary performance: \n",f"- ROC AUC: {adv_roc_auc:.2f} \n"])
        file1.writelines("\n")

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(classify(y_pred))
    return metrics.roc_auc_score(y_test, y_pred, average=average)

def classify(y_pred):
    y_pred_out = y_pred
    r = 0
    for _, row in y_pred.iterrows():
        row_max = row.max()
        for c in range(len(row)):
            y_pred_out.at[r,c] = (int(0) if row[c] < row_max else int(1))
        r += 1
    return y_pred_out

def test(image_name):
    i = 0
    for Z_test, Z_test_name in zip(Z_sets, Z_sets_names):
        print(i)
        adv = Adversary(n_features = 1, n_sensitive = Z_test.shape[1])
        
        with torch.no_grad():
            pre_clf_test = clf(test_data.tensors[0])
            pre_adv_test = adv(pre_clf_test)

        y_pre_clf = pd.Series(pre_clf_test.data.numpy().ravel(),
                              index=y_test.index)
        y_pre_adv = pd.DataFrame(pre_adv_test.numpy())
        
        fig = plot_distributions(y_test, Z_test, y_pre_clf, Z_test_name, image_name, y_pre_adv)
        fig.savefig('output/output_n/output_n_{}/{}_{}.png'.format(n_clusters, image_name, Z_test_name))


def average(my_list):
    total = 0
    for i in my_list: total += i
    return float(total)/float(len(my_list))

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

clf = Classifier(n_features=X.shape[1])
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

    def __init__(self, n_features, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x)) 


def pretrain_adversary1(adv, clf, data_loader, optimizer, criterion):
    for x, _, z1 in data_loader:
        p_y = clf(x).detach()
        adv.zero_grad()
        p_z = adv(p_y)
        loss = criterion(p_z, z1).mean() 
        loss.backward()
        optimizer.step()
    return adv

adv1 = Adversary(n_features = 1, n_sensitive = Z_train_c1.shape[1])
adv_criterion = nn.BCELoss(reduce=False)
adv1_optimizer = optim.Adam(adv1.parameters(), lr = 0.02)

N_ADV_EPOCHS = 5

for epoch in range(N_ADV_EPOCHS):
    adv1 = pretrain_adversary1(adv1, clf, train_loader, adv1_optimizer, adv_criterion)

# TRAIN AND TEST ####################################################

# before training: baseline
test('baseline')

def train(clf, adv1, data_loader, clf_criterion, adv_criterion,
          clf_optimizer, adv1_optimizer):
    
    # Train adversary
    adv_losses = []
    for x, y, z1 in data_loader:
        p_y = clf(x)
        adv1.zero_grad()
        p_z1 = adv1(p_y)
        loss_adv = adv_criterion(p_z1, z1).mean()
        adv_losses.append(loss_adv)
        loss_adv.backward()
        adv1_optimizer.step()
 
    # Train classifier on a single batch
    clf_losses = []
    entropies = []
    for x, y, z1 in data_loader:
        pass
    p_y = clf(x)
    p_z1 = adv1(p_y)
    for p_z1_i in p_z1:
        entropies.append(entropy(p_z1_i.detach()))
    clf.zero_grad()
    loss_adv = adv_criterion(p_z1, z1).mean()
    clf_loss = clf_criterion(p_y, y) - loss_adv # minimax loss
    clf_losses.append(clf_loss)
    clf_loss.backward()
    clf_optimizer.step()

    return clf, adv1, average(adv_losses), average(clf_losses), average(entropies)

N_EPOCH_COMBINED = 165

plt_adv_loss = []
plt_clf_loss = []
z_entropies = []

for epoch in range(1, N_EPOCH_COMBINED):
    
    clf, adv1, adv_loss, clf_loss, z_entropy = train(clf, adv1, train_loader, clf_criterion, adv_criterion,
                     clf_optimizer, adv1_optimizer)

    plt_adv_loss.append(adv_loss)
    plt_clf_loss.append(clf_loss)
    z_entropies.append(z_entropy)
    
    with torch.no_grad():
        clf_pred = clf(test_data.tensors[0])
        adv_pred1 = adv1(clf_pred)

    print(epoch)
 
plt.figure()
plt.plot(z_entropies)
plt.savefig('output/output_n/output_n_{}/entropy_fig.png'.format(n_clusters))

plt.figure()
plt.plot(plt_clf_loss, 'bo', plt_adv_loss, 'r+')
plt.savefig('output/output_n/output_n_{}/loss_fig.png'.format(n_clusters))

# after training: results
test('output')

file1.close()
