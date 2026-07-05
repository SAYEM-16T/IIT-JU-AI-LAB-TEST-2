# 🧩 Common sklearn Workflow (Day 5/6/7) — Banglish Note

> Tomar exam kal. Tension nio na. Ei ekta file bhalo kore bujhle Day 5, 6, 7 er supervised part er **90% code tomar already jana hoye jabe**. Cholo shuru kori. 💪

---

## 🎯 Keno EI file ta shobar age porba

3 din er lab code gulo dekhte alada, kintu **bhitore prai puropuri same**. Same skeleton, sudhu chhoto 2-3 ta jinis change hoy.

- Day 5 → **Linear Regression** (USA_Housing.csv, `Price` predict)
- Day 6 → **Logistic Regression** (titanic_train.csv, `Survived` predict)
- Day 7a → **Decision Tree + Random Forest** (kyphosis.csv, `Kyphosis` predict)
- Day 7b → **K-Means** (Live.csv) — ei ta ektu **alada** (unsupervised, no y, no split)

🧠 **Mne rakho:** Ekbar ei "skeleton" ta hatey uthle, exam e tumi sudhu chinta korba — "aj kon **model** ar kon **metric**?" Baki sob same.

---

## 🦴 THE shared supervised skeleton (Day 5/6/7a)

Eta holo shei magic code jeta 3 din e ghure ghure ase. Ekhane structure ta mne rakho:

```python
# 1) Library import kori
import pandas as pd
from sklearn.model_selection import train_test_split

# 2) Dataset porbo (CSV -> DataFrame)
df = pd.read_csv('data.csv')

# 3) X = features (input), y = target (ja predict korbo)
X = df[[...features...]]        # ba: df.drop('target', axis=1)
y = df['target']

# 4) Train/Test bhag kori — model shikhbe train e, test e porikkha hobe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=..., random_state=101)

# 5) Model banai  <-- SUDHU EI LINE ta protidin change hoy
model = SomeModel()

# 6) fit = train korano (model data theke pattern shekhe)
model.fit(X_train, y_train)

# 7) predict = notun (test) data er upor guess korano
pred = model.predict(X_test)

# 8) Evaluate: regression hole MAE/MSE/RMSE, classification hole report/confusion matrix
```

**Ek line e:** CSV porho → X/y banao → train_test_split → model banao → `fit` → `predict` → evaluate. Bas!

⚠️ **Exam tip:** `train_test_split` er output order **mukhosto** rakho — `X_train, X_test, y_train, y_test`. Ulta likhle pura result golmal hoye jabe.

---

## 🔑 "Sudhu 2 ta jinis change hoy"

Puro skeleton same thakche. Din-e-din e sudhu **(1) model constructor line** ar **(2) evaluation metric family** change hoy. Eta table hisebe mne rakho:

| Day | Model line (ei ta change) | Metric family |
|-----|---------------------------|---------------|
| 5 | `LinearRegression()` | MAE / MSE / RMSE (regression) |
| 6 | `LogisticRegression(solver='lbfgs', max_iter=1000)` | classification_report + confusion_matrix |
| 7a | `DecisionTreeClassifier(criterion='entropy', random_state=0)` | classification_report + confusion_matrix |
| 7a | `RandomForestClassifier(n_estimators=100, criterion='entropy')` | classification_report + confusion_matrix |
| 7b | `KMeans(n_clusters=k)` | inertia_ + Elbow method (NO split, NO y) |

🧠 **Mne rakho:** Day 5 continuous number (taka) predict → **regression metric**. Day 6/7a category (0/1) predict → **classification metric**. Day 7b er kono target nei → sudhu `inertia_`.

Copy-paste kore mukhosto korar moto model line gulo (verbatim):

```python
# Day 5
model = LinearRegression()

# Day 6
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Day 7a (single tree)
model = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Day 7a (forest)
model = RandomForestClassifier(n_estimators=100, criterion='entropy')

# Day 7b (alada! no y, no split)
model = KMeans(n_clusters=5, random_state=42)
```

---

## 📦 Import map — kon jinish er jonno kon import

Kon kaj korte gele kon line lekhbe — ei list ta chokher samne rakho:

```python
# Basics (protidin lage)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Train/Test bhag korte
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree                       # tree.plot_tree() visualize er jonno
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Preprocessing tools
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Regression metrics (Day 5)
from sklearn import metrics

# Classification metrics (Day 6/7a)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

⚠️ **Exam tip:** `train_test_split` ashe `sklearn.model_selection` theke, `metrics` ashe `sklearn` / `sklearn.metrics` theke — ei import path gulo prosno e miss korle number kate. Mukhosto rakho.

---

## 🧹 Preprocessing tools table (kon tool + kon din)

Raw data ke model-friendly banate ei tool gulo lage. Ek line kore mne rakho:

| Tool | Ki kore | Kon Day |
|------|---------|---------|
| `pd.get_dummies(col, drop_first=True)` | Categorical → 0/1 dummy columns; `drop_first` dummy trap edai | Day 6 (Sex, Embarked) |
| `LabelEncoder()` | Categorical → integer label (0,1,2,3) | Day 7b (status_type) |
| `StandardScaler()` | Standardize: mean 0, std 1 | Day 7a |
| `MinMaxScaler()` | Scale kore [0, 1] range e ane | Day 7b |
| impute (`fillna` / custom function) | Missing (NaN) value fill kore | Day 6 (Age by Pclass) |

**Scaler er sob-cheye important niom** (Day 7a te ase) — train e `fit_transform`, test e sudhu `transform`:

```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)   # train: fit (shekhe) + transform
X_test  = sc.transform(X_test)        # test: SUDHU transform (no fit!)
```

🧠 **Mne rakho:** Scaler test er upor `fit` korle test data er info leak hoye jay ("data leakage"). Tai test e always sudhu `transform`.

Day 6 dummy variable banano (reference):

```python
sex = pd.get_dummies(train['Sex'], drop_first=True)          # -> 'male' (0/1)
embark = pd.get_dummies(train['Embarked'], drop_first=True)  # -> 'Q','S'
```

Day 7b LabelEncoder + MinMaxScaler (reference):

```python
le = LabelEncoder()
df['status_type'] = le.fit_transform(df['status_type'])   # video/photo/link/status -> 0..3

ms = MinMaxScaler()
y = ms.fit_transform(y)                                    # sob feature 0..1 e
```

---

## ⚠️ Common Mistakes (ei gulo te-i marks kate)

Exam er age ei bhulgulo mathay gethe nao:

- **❌ Test er upor `fit_transform` kora.** Test e always `sc.transform(X_test)`, kokhono `fit_transform` na. Fit korle data leakage.
- **❌ Regression ar classification metric mix kore fela.** Day 5 (number predict) → MAE/MSE/RMSE. Day 6/7a (category predict) → classification_report/confusion_matrix. Uposhomporko metric lagle pura utthor bhul.
- **❌ `random_state` bhule jawa.** `random_state` na dile prottek bar alada split/result ashe, tomar number report er sathe milbe na. Digest e: Day5/6 → `random_state=101`, Day7a tree → `random_state=0`, Day7b → `random_state=42`.
- **❌ K-Means e `y` (target) dewa.** K-Means **unsupervised** — ekhane kono target `y` nei, kono `train_test_split` nei. Sudhu `X` diye `fit`, tarpor `inertia_` ar Elbow method diye K bachai. y ba split khujle bujhba tumi algorithm ta miss korecho.

🧠 **Mne rakho — 1 line summary:** *"Skeleton same, sudhu **model line** ar **metric** change; scaler test e sudhu transform; K-Means e y nei."* Eta mukhosto thakle tumi safe. Best of luck! 🚀
