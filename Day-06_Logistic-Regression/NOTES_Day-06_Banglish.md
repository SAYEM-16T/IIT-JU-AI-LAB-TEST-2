# 🚢 Day 6 — Logistic Regression (Titanic)

> Exam kal? Tension nio na. Ei ek file e Day 6 er sob ache — theory, code, cleaning, evaluation, viva. Bus e boshe ekbar pore felo, hoye jabe. 💪

---

## 🎯 Ek line e: Day 6 e ki?

Day 6 e amra **category predict** kori — Titanic e ke **`Survived` (1)** ar ke **deceased (0)** seta.

- Output continuous number **na** (seta Day 5, Linear Regression).
- Output holo **0 ba 1** → tai eta **Supervised Classification**.
- Algorithm = **Logistic Regression**.

🧠 **Mne rakho:** Day 5 = number predict (Regression), Day 6 = class/label predict (Classification). Dataset e **label/target ache** bole "Supervised".

---

## 📚 Theory: Logistic Regression ki?

**Logistic Regression** ekta **supervised classification** algorithm. Eta discrete/categorical output dey — Yes/No, True/False, **0/1**.

Important kotha: eta thik "0 ba 1" na diye age ekta **probability (0 theke 1 er moddhe)** ber kore, tarpor threshold diye class thik kore.

### Linear vs Logistic — parthokko

| Bishoy | Linear Regression | Logistic Regression |
|---|---|---|
| Task | Regression | Classification |
| Output | Continuous number | Probability → 0/1 |
| Curve | Straight line | "S"-shaped curve |
| Example | House Price | Survived / Not |

- **Form ekhon o similar** (weighted sum of features), kintu Linear soja line boshay; Logistic ekta **"S"-shape (sigmoid) curve** boshay ja duita extreme (0 ba 1) er dike predict kore.

### 🔑 Sigmoid / Logistic function

Sigmoid function er kaj: **je kono real value** (khub boro, khub choto, negative — je kono kichu) ke niye **(0, 1)** range e map kore dey. Manei ekta **probability**.

- Curve ta **S-shape**.
- **Threshold** (usually **0.5**) diye decide kori:
  - probability **> 0.5** → class **1**
  - probability **< 0.5** → class **0**

🧠 **Mne rakho:** Sigmoid = "squash into 0–1". Threshold = "kothay line taanbo". Titanic e default threshold 0.5.

---

## 🗂️ Dataset: Titanic (`titanic_train.csv`)

- **Target = `Survived`** → **1 = beche geche**, **0 = mara geche**.
- Important **features**:

| Feature | Ki bojhay |
|---|---|
| `Pclass` | Ticket class (1/2/3) |
| `Sex` | male / female |
| `Age` | boyosh |
| `Fare` | ticket er dam |
| `SibSp` | sibling/spouse shonge |
| `Parch` | parents/children shonge |
| `Embarked` | kon port theke uthechilo (C/Q/S) |

⚠️ **Exam tip:** EDA te dekha gese — **beshi male die geche**, ar **Pclass 3 (poor class) e mrittu beshi**. Ei intuition viva te kaje lage.

```python
# Library import + dataset load kora
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline

train = pd.read_csv('titanic_train.csv')
train.tail()   # sesh koyekta row dekhi
```

```python
# EDA: kon column e missing data ache seta chart e dekha
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Age ~20% missing (fill kora jabe), Cabin onek beshi missing (drop korbo)

sns.countplot(x='Survived', data=train)                 # koto beche/moreche
sns.countplot(x='Survived', hue='Sex', data=train)      # beshi male moreche
sns.countplot(x='Survived', hue='Pclass', data=train)   # class 3 e moreche beshi
sns.displot(train['Age'].dropna(), kde=False, bins=30)  # Age er distribution
```

---

## 🧹 Data Cleaning (Day 6 er SPECIAL part)

Ei day er asol mojja hocche cleaning. Titanic data noisy — Age missing, Cabin missing, Sex/Embarked text. Model e dewar age egula thik korte hobe.

### 1️⃣ Missing data heatmap
`sns.heatmap(train.isnull(), ...)` → jekhane hlud/light line, sekhane NaN. Ekhane dekhi **Age olpo missing** (fill korbo), **Cabin onek missing** (drop korbo).

### 2️⃣ `impute_age` — Age fill kora (Pclass diye)

Boxplot e dekha jay: **higher class → older passenger**. Tai Age missing thakle randomly na diye, **Pclass onujayi average age** boshai.

```python
sns.boxplot(x='Pclass', y='Age', data=train)  # higher class -> boyosh beshi

# Age missing hole Pclass onujayi typical age boshabo
def impute_age(cols):
    Age = cols[0]; Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1: return 37
        elif Pclass == 2: return 29
        else: return 24
    else:
        return Age

# Age ar Pclass column niye row-wise (axis=1) function apply kori
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
```

**Keno 37/29/24?** — Boxplot e ei class-gulor **median age** ei rokom chilo. Class 1 = boro-loker, boyosh beshi (37); class 2 mid (29); baki (class 3) tulnamulok kom boyosh (24).

🧠 **Mne rakho:** row na fele **smart imputation** — data harate hoy na, ar Pclass er info use kore realistic value boshano hoy.

### 3️⃣ Cabin drop keno?
`Cabin` e **eto beshi NaN** je fill korle bhul info dhukbe. Tai purota column-i baad.

```python
train.drop('Cabin', axis=1, inplace=True)  # Cabin e boro beshi NaN -> drop
train.dropna(inplace=True)                  # baki jei olpo NaN (Embarked) -> row baad
```

- `axis=1` = column drop. `inplace=True` = same DataFrame e change.
- `dropna()` diye baki khub-olpo missing row (jemon Embarked) fele di.

### 4️⃣ `get_dummies` — text ke number banano

Model text ("male", "female", "S") bujhe na. Tai **dummy variable** (0/1 column) banai.

```python
# Sex ar Embarked ke 0/1 dummy column e convert kora
sex = pd.get_dummies(train['Sex'], drop_first=True)          # -> 'male' (0/1) column
embark = pd.get_dummies(train['Embarked'], drop_first=True)  # -> 'Q','S' columns

# Text/useless column baad, tarpor dummy column jog kora
train.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)
# drop_first=True -> "dummy variable trap" (multicollinearity) avoid kore
```

**`drop_first=True` keno? (Dummy Variable Trap)**
- `Sex` er 2 category: male, female. Ekta column `male` (1=male, 0=female) hoilei purota info paoya jay — alada `female` column **redundant**.
- Duita column rakhle tara eke onner theke **fully predictable** (male + female = 1 shobshomoy) → eta **multicollinearity / dummy variable trap**. Model confuse hoy.
- Tai first category drop kore di. **n category → (n−1) column**.

**`Name`, `Ticket` drop keno?** — ei duita motamuti **unique text**, kono pattern nai, predict e help kore na. Tai baad.

**`concat` keno?** — new dummy column-gula (`male`, `Q`, `S`) main `train` DataFrame er sathe jog korte `pd.concat(..., axis=1)`.

⚠️ **Exam tip:** Cleaning er order mne rakho → **heatmap → impute Age → drop Cabin → dropna → get_dummies(drop_first) → drop text cols → concat**.

---

## 🤖 Model: train / split / fit / predict

```python
# Data ke train ar test e vag kora (70% train, 30% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    train.drop('Survived', axis=1), train['Survived'],
    test_size=0.30, random_state=101)
```

- `X` = `train.drop('Survived', axis=1)` → sob feature (target chara).
- `y` = `train['Survived']` → target.
- `test_size=0.30` → 30% test er jonno.
- `random_state=101` → prottekbar **same split** paoar jonno (reproducibility).

```python
# Logistic Regression model banano, train kora, predict kora
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs', max_iter=1000)
logmodel.fit(X_train, y_train)          # train data diye shekha
predictions = logmodel.predict(X_test)  # test data er prediction
```

- `solver='lbfgs'` → optimization algorithm (default, chhoto data te bhalo).
- `max_iter=1000` → convergence er jonno beshi iteration allow (na dile warning ashte pare).

🧠 **Mne rakho:** skeleton ta Day 5 er moto-i — **shudhu model er line ta bodlay** (`LinearRegression()` → `LogisticRegression(...)`).

---

## 📊 Evaluation

### `classification_report`

```python
# Per-class precision/recall/f1 + overall accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

Result:

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| 0 | 0.82 | 0.91 | 0.86 | 163 |
| 1 | 0.84 | 0.68 | 0.75 | 104 |
| **accuracy** | | | **0.82** | 267 |

- **accuracy = 0.82** → 82% test data thik predict.
- class **1 er recall komm (0.68)** → jara sotti beche giyeche tader modde kichu miss korche.

### Individual metric scores

```python
# Alada alada metric ber kora (positive class = 1)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy  = accuracy_score(y_test, predictions)   # 0.8239...
precision = precision_score(y_test, predictions)  # 0.8353...
recall    = recall_score(y_test, predictions)     # 0.6827...
f1        = f1_score(y_test, predictions)          # 0.7513...
```

### `confusion_matrix`

```python
# Predict kotota thik/bhul, tar 2x2 table
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
# array([[149,  14],
#        [ 33,  71]])
```

**Ei matrix ta pori (0 = negative, 1 = positive):**

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | **TN = 149** | **FP = 14** |
| **Actual 1** | **FN = 33** | **TP = 71** |

- **TN = 149** → sotti moreche, model o "moreche" bolese ✅
- **FP = 14** → sotti moreche, kintu model "beche geche" bolese ❌ (false alarm)
- **FN = 33** → sotti beche geche, kintu model "moreche" bolese ❌ (miss kore feleche)
- **TP = 71** → sotti beche geche, model o "beche geche" bolese ✅

**Chek kore dekho:**
- Accuracy = (TP+TN)/total = (71+149)/267 = **0.82** ✅
- Precision (class 1) = TP/(TP+FP) = 71/(71+14) = **0.835** ✅
- Recall (class 1) = TP/(TP+FN) = 71/(71+33) = **0.683** ✅

🧠 **Mne rakho (layout):** `[[TN, FP], [FN, TP]]`. Top row = **actual 0**, bottom row = **actual 1**. Diagonal (TN, TP) = thik; off-diagonal (FP, FN) = bhul.

---

## ⚠️ Exam tip: Minimum reproduce code

Somoy kom? Ei chunk ta mukhosto rakho — cleaning + model + eval sob ache.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv('titanic_train.csv')

# --- cleaning ---
def impute_age(cols):
    Age, Pclass = cols[0], cols[1]
    if pd.isnull(Age):
        return 37 if Pclass == 1 else 29 if Pclass == 2 else 24
    return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)

# --- model ---
X_train, X_test, y_train, y_test = train_test_split(
    train.drop('Survived', axis=1), train['Survived'],
    test_size=0.30, random_state=101)
logmodel = LogisticRegression(solver='lbfgs', max_iter=1000)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

# --- evaluate ---
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

---

## 🎤 Viva Q&A (short + sharp)

**Q: Sigmoid function ki?**
A: Je kono real value ke **(0,1)** range e map kore dey → probability dey. Curve ta **S-shape**.

**Q: Threshold ki?**
A: Probability ke class e vag korar cutoff, usually **0.5**. `>0.5 → 1`, `<0.5 → 0`.

**Q: Precision vs Recall?**
A: **Precision = TP/(TP+FP)** → jader "positive" bollam tader koto % thik. **Recall = TP/(TP+FN)** → sotti positive der koto % dhorte parlam (sensitivity).

**Q: Confusion matrix kivabe pori?**
A: `[[TN, FP],[FN, TP]]`. Top row actual 0, bottom row actual 1. Diagonal = correct. Ekhane TN=149, FP=14, FN=33, TP=71.

**Q: `drop_first=True` keno?**
A: **Dummy variable trap (multicollinearity)** avoid korte. n category → (n−1) column-i jothesto; first ta drop.

**Q: `get_dummies` keno?**
A: Model text bujhe na, tai categorical (Sex, Embarked) ke **0/1 numeric dummy column** e convert korte.

**Q: Linear vs Logistic Regression?**
A: Linear = **regression** (continuous number, straight line). Logistic = **classification** (0/1 probability, S-curve). Form kachakachi, kintu output ar kaj alada.

**Q: impute_age e 37/29/24 keno?**
A: Boxplot e Pclass onujayi **typical (median) age** oi rokom — higher class = boyosh beshi. Age missing hole class onujayi realistic value boshai.

---

## 🔗 Cross-reference (onno file dekho)

- **Metrics er full formula + intuition (accuracy, precision, recall, f1, confusion matrix):** dekho common metrics cheat-sheet → `Day-00_COMMON_All-Days/` folder.
- **Sob din er common viva Q&A:** `Day-00_COMMON_All-Days/` folder er viva file.
- **Linear vs Logistic er full compare + regression metrics (MAE/MSE/RMSE):** `Day-05_Linear-Regression/NOTES_Day-05_Banglish.md`.
- **Onno classification model (Decision Tree, Random Forest) — same classification_report/confusion_matrix use hoy:** `Day-07_Trees-and-Clustering/NOTES_Day-07_Banglish.md`.

---

### ✅ Sesh kotha
Day 6 = **Titanic e Survived 0/1 predict → Logistic Regression**. Cleaning (impute Age, drop Cabin, get_dummies+drop_first) → split (0.30, 101) → fit → predict → **accuracy 0.82**, confusion matrix `[[149,14],[33,71]]`. Eituku confident thakle Day 6 e full marks. Tumi parbe! 🚀
