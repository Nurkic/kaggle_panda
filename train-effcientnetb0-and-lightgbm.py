# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import sys
sys.path = [
    '../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master',
] + sys.path

!pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import model as enet
import albumentations
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm_notebook as tqdm
import os
import copy
import optuna
import featuretools as ft
import lightgbm as lgb
from efficientnet_pytorch import EfficientNet
#model_ft = EfficientNet.from_pretrained('efficientnet-b0')

DEBUG = False
data_dir = 'input/prostate-cancer-grade-assesment'
df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
image_folder = os.path.join(data_dir, 'train_images')

kernel_type = 'how_to_train_effnet_b7_to_get_LB_0.86_kai'

enet_type = 'efficientnet-b0'
fold = 0
tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 2
num_workers = 4
out_dim = 5
init_lr = 3e-4
warmup_factor = 10

warmup_epo = 1
n_epochs = 1 if DEBUG else 30
df_train = df_train.sample(100).reset_index(drop=True) if DEBUG else df_train

device = torch.device('cuda')

print(image_folder)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
df_train['fold'] = -1
for i, (train_idx, valid_idx) in enumerate(skf.split(df_train, df_train['isup_grade'])):
    df_train.loc[valid_idx, 'fold'] = i
print(df_train.head())
pretrained_model = {
    'efficientnet-b0': 'input/efficientnet-b0-08094119.pth'
}
#print(model_ft)

"""
p_temp = 'input/prostate-cancer-grade-assesment/train_images'
files = os.listdir(p_temp)
files2 = [s.replace(".tiff","") for s in files]
print(files2)
df_train_new = pd.DataFrame()
for i in files2:
    df = df_train.copy()
    row = df[df["image_id"]==i]
    df_train_new = df_train_new.append(row)
df_train = df_train_new.reset_index()
print(df_train)"""


class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        #self.extract_features = list(enet.EfficientNet.extract_features())
        self.enet = enet.EfficientNet.from_name(backbone)
        self.enet.load_state_dict(torch.load(pretrained_model[backbone]))
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def extract_features(self, x):
        f = self.enet.extract_features(x)
        return f

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
    


def get_tiles(img, mode=0):
        result = []
        h, w, c = img.shape
        pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
        pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

        img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
        img3 = img2.reshape(
            img2.shape[0] // tile_size,
            tile_size,
            img2.shape[1] // tile_size,
            tile_size,
            3
        )

        img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
        n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
        if len(img3) < n_tiles:
            img3 = np.pad(img3,[[0,n_tiles-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
        idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
        img3 = img3[idxs]
        for i in range(len(img3)):
            result.append({'img':img3[i], 'idx':i})
        return result, n_tiles_with_info >= n_tiles

class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 n_tiles=n_tiles,
                 tile_mode=0,
                 rand=False,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.n_tiles = n_tiles
        self.tile_mode = tile_mode
        self.rand = rand
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        
        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')
        image = skimage.io.MultiImage(tiff_file)[1]
        tiles, OK = get_tiles(image, self.tile_mode)

        if self.rand:
            idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
        else:
            idxes = list(range(self.n_tiles))

        n_row_tiles = int(np.sqrt(self.n_tiles))
        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
        for h in range(n_row_tiles):
            for w in range(n_row_tiles):
                i = h * n_row_tiles + w
    
                if len(tiles) > idxes[i]:
                    this_img = tiles[idxes[i]]['img']
                else:
                    this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                this_img = 255 - this_img
                if self.transform is not None:
                    this_img = self.transform(image=this_img)['image']
                h1 = h * image_size
                w1 = w * image_size
                images[h1:h1+image_size, w1:w1+image_size] = this_img

        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.astype(np.float32)
        images /= 255
        images = images.transpose(2, 0, 1)

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        return torch.tensor(images), torch.tensor(label)

transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])
transforms_val = albumentations.Compose([])


dataset_show = PANDADataset(df_train, image_size, n_tiles, 0, transform=transforms_train)
from pylab import rcParams
rcParams['figure.figsize'] = 20,10
for i in range(2):
    f, axarr = plt.subplots(1,5)
    for p in range(5):
        idx = np.random.randint(0, len(dataset_show))
        img, label = dataset_show[idx]
        axarr[p].imshow(1. - img.transpose(0, 1).transpose(1,2).squeeze())
        axarr[p].set_title(str(sum(label)))

criterion = nn.BCEWithLogitsLoss()

def train_epoch(loader, optimizer):
    mid_train_vals = []
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        
        data, target = data.to(device), target.to(device)
        loss_func = criterion
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        aa = model.extract_features(data).cpu().detach().numpy()
        mid_train_vals.append([aa, target])
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        #mid_train_vals = mid_train_vals
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss, mid_train_vals


def val_epoch(loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            #mid_val_vals = []
            loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()
            LOGITS.append(logits)
            PREDS.append(pred)
            TARGETS.append(target.sum(1))
            #bb = model.extract_features(data).cpu().detach().numpy()
            #mid_val_vals.append([bb, target])
            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = (PREDS == TARGETS).mean() * 100.
    
    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')
    qwk_k = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'karolinska'], df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values, weights='quadratic')
    qwk_r = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'radboud'], df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values, weights='quadratic')
    print('qwk', qwk, 'qwk_k', qwk_k, 'qwk_r', qwk_r)

    if get_output:
        return LOGITS
    else:
        return val_loss, acc, qwk#, mid_val_vals

train_idx = np.where((df_train['fold'] != fold))[0]
valid_idx = np.where((df_train['fold'] == fold))[0]

df_this  = df_train.loc[train_idx]
df_valid = df_train.loc[valid_idx]

dataset_train = PANDADataset(df_this , image_size, n_tiles, transform=transforms_train)
dataset_valid = PANDADataset(df_valid, image_size, n_tiles, transform=transforms_val)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train), num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

model = enetv2(enet_type, out_dim=out_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

print(len(dataset_train), len(dataset_valid))

qwk_max = 0.
best_file = f'{kernel_type}_best_fold{fold}.pth'
for epoch in range(1, n_epochs+1):
    print(time.ctime(), 'Epoch:', epoch)
    scheduler.step(epoch-1)

    train_loss, mid_train_vals = train_epoch(train_loader, optimizer)
    val_loss, acc, qwk = val_epoch(valid_loader)

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, val loss: {np.mean(val_loss):.5f}, acc: {(acc):.5f}, qwk: {(qwk):.5f}'
    print(content)
    with open(f'log_{kernel_type}.txt', 'a') as appender:
        appender.write(content + '\n')

    if qwk > qwk_max:
        print('score2 ({:.6f} --> {:.6f}).  Saving model ...'.format(qwk_max, qwk))
        torch.save(model.state_dict(), best_file)
        qwk_max = qwk

torch.save(model.state_dict(), os.path.join(f'{kernel_type}_final_fold{fold}.pth'))
#print(len(mid_train_vals[4]))
print(mid_train_vals[0][0][0][0].shape, mid_train_vals[0][1][0].shape)


"""for i in range(len(mid_train_vals)):
    if len(mid_train_vals[i]) == 2:"""

train_X = np.array([])
train_y = np.array([])

for i in range(len(mid_train_vals)-1):
    
    t1 = np.array([])
    t2 = np.array([])
    for k in range(len(mid_train_vals[i][0][0])):
        l1 = np.array([])
        l2 = np.array([])
        for j in range(len(mid_train_vals[i][0][0][0][0])):
            s1 = np.mean(mid_train_vals[i][0][0][j])
            
            s2 = np.mean(mid_train_vals[i][0][1][j])
            l1 = np.append(l1, s1)
            l2 = np.append(l2, s2)
        t1 = np.append(t1, l1.mean())
        t2 = np.append(t2, l2.mean())
        
    #print(t1.shape)
    train_X = np.append(train_X, t1.reshape(1, len(t1)))
    train_X = np.append(train_X, t2.reshape(1, len(t2)))
    train_y = np.append(train_y, mid_train_vals[i][1][0].cpu().numpy().sum())
    train_y = np.append(train_y, mid_train_vals[i][1][1].cpu().numpy().sum())

train_X = train_X.reshape(6, len(t1))
print(train_X.shape, train_y.shape)
train_X = pd.DataFrame(train_X)
train_y = pd.Series(train_y, dtype="int")

print(train_X.head())
print(train_y)

class ModelExtractionCallback(object):
    #lightgbm.cv() から学習済みモデルを取り出すためのコールバックに使うクラス

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # _CVBooster の参照を保持する
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # コールバックが呼ばれていないときは例外にする
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # Booster へのプロキシオブジェクトを返す
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # Booster のリストを返す
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # Early stop したときの boosting round を返す
        return self._model.best_iteration

def lgb_custom_metric_qwk_multiclass(preds, data):
    #LightGBM のカスタムメトリックを計算する関数

    #多値分類問題として解いた予測から QWK を計算する
    # 正解ラベル
    y_true = data.get_label()
    # ラベルの数
    num_labels = 6
    # 多値分類問題では本来は二次元配列が一次元配列になって提供される
    reshaped_preds = preds.reshape(num_labels, len(preds) // num_labels)
    # 最尤と判断したクラスを選ぶ　
    y_pred = np.argmax(reshaped_preds, axis=0)  # 最尤と判断したクラスを選ぶ
    # QWK を計算する
    return 'qwk', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True
"""
def objective(trial):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 6,
        'verbosity': -1,
        "seed":42,
        "learning_rate":trial.suggest_loguniform('learning_rate', 0.005, 0.03),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    FOLD_NUM = 4
    kf = KFold(n_splits=FOLD_NUM,
              #shuffle=True,
              random_state=42)
    scores = []
    #feature_importance_df = pd.DataFrame()
    #pred_cv = np.zeros(len(test.index))
    num_round = 10000
    for i, (tdx, vdx) in enumerate(kf.split(train_X, train_y)):
        print(f'Fold : {i}')
        X_train, X_valid, y_train, y_valid = train_X.iloc[tdx], train_X.iloc[vdx], train_y.values[tdx], train_y.values[vdx]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        model = lgb.train(params, lgb_train, num_boost_round=num_round,
                      valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=10, verbose_eval=10000, feval=lgb_custom_metric_qwk_multiclass)
        va_pred = model.predict(X_valid)
        #va_pred[va_pred<0] = 0
        score_ = cohen_kappa_score(y_valid, va_pred, weights='quadratic')
        scores.append(score_)
    return np.mean(scores)
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
# 結果の確認
print('Best trial:')
light_trial = study.best_trial
print('  Value: {}'.format(light_trial.value))
print('  Params: ')
with open("lightgbmparams.txt", "w") as file:
    for key, value in light_trial.params.items():
       print('    "{}": {},'.format(key, value))
       file.write('"{}": {},'.format(key, value))"""

light_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 6,
        'verbosity': -1,
        "seed":42,
        'learning_rate': 0.01}

FOLD_NUM = 2
kf = KFold(n_splits=FOLD_NUM,
           shuffle=True,
           random_state=42)
scores = []
num_round = 10000
from sklearn.model_selection import train_test_split
"""for i, (tdx, vdx) in enumerate(KFold.split(train_X, train_y)):
    print(f'Fold : {i}')
    ######LGB"""
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2, random_state=100)
    # LGB
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid)
model = lgb.train(light_params, lgb_train, num_boost_round=num_round,
                  valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],
                  feval=lgb_custom_metric_qwk_multiclass,
                  early_stopping_rounds=100, verbose_eval=500)
va_pred = model.predict(X_valid).astype(int)
va_pred_true = np.array([])
for i in range(len(va_pred)):
    va_pred_true= np.append(va_pred_true, np.sum(va_pred[i]))
print(va_pred_true)
score = cohen_kappa_score(y_valid, va_pred_true, weights='quadratic')
scores.append(score)
print(np.mean(scores))
import pickle
file = "trained_lgbm.pkl"
pickle.dump(model, open(file, "wb"))

del model




