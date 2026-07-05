# 🗺️ ML Big Picture / Concept Map (Day 5 + 6 + 7 ek jaygay)

Assalamualaikum, tumi eii ek note porlei bujhbe **tinta din er sob algorithm eksathe kivabe fit hoy**. Bus e boshe eta ekbar pore felle exam e "kon algorithm kobe use hoy" type question e tumi never confuse hobe. Cholo shuru kori. 💪

---

## 🌳 ML er 2 boro bhag (sobar age eta mne rakho)

Puro Machine Learning ke prothome **duita bhag** e vebe felo. Pura khela ekta prosno er upore:

> **Amar kache "answer" (target/y) deya ache ki nai?**

- **Answer ache** (dataset e ekta column ache jeta amra predict korte chai) → **Supervised Learning**
- **Answer nai** (kono target column nai, just data er vitore group khujchi) → **Unsupervised Learning**

Ekta simple text tree:

```text
Machine Learning
│
├── Supervised Learning   (target/y ACHE — teacher ache jini answer dekhiye den)
│     │
│     ├── Regression       → output ekta CONTINUOUS number (Price, weight, temp)
│     │        └── Day 5: Linear Regression
│     │
│     └── Classification   → output ekta CATEGORY / label (Yes/No, 0/1)
│              ├── Day 6:  Logistic Regression
│              └── Day 7a: Decision Tree  &  Random Forest
│
└── Unsupervised Learning (target/y NEI — label chara group khoja)
      │
      └── Clustering        → similar jinis ke ek group e fela
               └── Day 7b: K-Means
```

🧠 **Mne rakho:** "Supervised" mane ekjon **teacher** (labelled answer) tomake shikhachhen. "Unsupervised" e kono teacher nai — model nijei pattern khuje ber kore.

---

## 🎯 Supervised er vitore 2 rokom + Unsupervised

### 1) Regression — output ekta *number*
- Output ekta **continuous** value (jekono number, decimal o hote pare).
- Example: house er **Price** koto? (Day 5) — 250000, 310500.75... eirokom.
- Metric: **MAE / MSE / RMSE** (kototuku error, minimize korte chai).

### 2) Classification — output ekta *category*
- Output ekta **discrete label / category** (ginone kora jay emon).
- Example: passenger **Survived** ki na (0/1)? (Day 6), Kyphosis **absent/present**? (Day 7a).
- Metric: **accuracy, precision, recall, f1, confusion matrix**.

### 3) Clustering (Unsupervised) — group khoja, label chara
- Kono answer column nai. Model similar data point gulo ke **cluster** e bhag kore.
- Example: Facebook Live post gulo ke behaviour onujayi group kora (Day 7b).
- Metric: **inertia** + **Elbow Method** (kono accuracy nai, karon kono "correct answer" e nai).

⚠️ **Exam tip:** Regression vs Classification alada korar shortcut — **output number naki category?** Number hole Regression, category hole Classification. Ei ek line diye 90% MCQ solve hoye jay.

---

## 📊 Master Mapping Table (eta mukhosto kore felo)

Eii table tomar puro syllabus ek screen e. Exam er age last minute e ei table ta dekhe nio.

| Day | Algorithm | Type | Task | Dataset | Target (y) |
|----|----|----|----|----|----|
| 5 | Linear Regression | Supervised | Regression | USA_Housing | Price (number) |
| 6 | Logistic Regression | Supervised | Classification | titanic_train | Survived (0/1) |
| 7a | Decision Tree | Supervised | Classification | kyphosis | Kyphosis (0/1) |
| 7a | Random Forest | Supervised | Classification | kyphosis | Kyphosis (0/1) |
| 7b | K-Means | Unsupervised | Clustering | Live.csv | none (nai) |

🧠 **Mne rakho:** ekmatro **Day 7b (K-Means)** e Target = "none". Baki sob din e target column ache, tai baki sob **Supervised**. Ei ekta fact join korle Supervised/Unsupervised bhulbe na.

---

## 🧭 "Ei problem hole kon algorithm" — Decision Guide

Exam e chhoto scenario dibe, tomake bolte hobe kon algorithm. Eii flow ta mathay rakho:

```text
Notun ekta problem elo. Nijeke jiggesh koro:

Q1. Amar kache label/answer (target y) ache?
     │
     ├── NA (label nei, just group/segment khujchi)
     │        → K-Means Clustering            (Unsupervised)  [Day 7b]
     │
     └── HA (target y ache) → Supervised. Q2 te jao.
              │
              Q2. Output ta ki predict korchi?
              │
              ├── Ekta NUMBER (continuous) predict korchi
              │        → Linear Regression                    [Day 5]
              │          (jemon: Price, salary, temperature)
              │
              └── Ekta CATEGORY / Yes-No / label predict korchi
                       → Logistic Regression                  [Day 6]
                       → Decision Tree                        [Day 7a]
                       → Random Forest                        [Day 7a]
                         (jemon: Survived? Kyphosis present?)
```

Choto version mne rakhar jonno:

- **Number predict korte hobe?** → **Linear Regression**
- **Yes/No ba category predict korte hobe, labelled data ache?** → **Logistic Regression / Decision Tree / Random Forest**
- **Kono label nei, similar jinis group korte chai?** → **K-Means**

⚠️ **Exam tip:** Keu jodi jiggesh kore "Titanic e ke Survive korlo predict korbo" — eta 0/1 category, tai **Classification → Logistic Regression**, kokhono Linear na. "House er dam predict" hole number, tai **Linear Regression**. Ei duita gulabe confuse korte chay, sabdhan.

---

## 📚 Key Vocabulary Box (ek line kore, exam er viva te lage)

| Term | Ek line Banglish |
|----|----|
| **feature (X)** | Input column gulo — jegulo diye amra predict kori (jemon Income, Age). |
| **target (y)** | Je column ta amra predict korte chai (jemon Price, Survived). |
| **label** | Classification e target er value/category (jemon 'absent'/'present', 0/1). |
| **training** | Model ke train data diye shekhano — `model.fit(X_train, y_train)`. |
| **testing** | Notun (unseen) test data te model check kora — koto valo perform kore. |
| **model** | Je algorithm object ta pattern shikhe (jemon `LinearRegression()`). |
| **fit** | Model ke data theke shekhano dhap — `.fit()` call kora. |
| **predict** | Shekha model diye notun input er output ber kora — `.predict()`. |
| **overfitting** | Model train data mukhosto kore fele, notun data te kharap kore. |
| **generalization** | Model notun/unseen data teo valo kore — eta i asol goal. |

🧠 **Mne rakho:** `X` = input (features), `y` = output (target). Ei duita ulta korle puro code ulta hoye jabe. `X` boro hater, `y` chhoto hater — code ei convention e lekha.

---

## 🔁 Sob din er ek i skeleton (only 1-2 line change hoy)

Ekta boro secret: **Supervised er sob din er code prai ek** — sudhu model er nam ar metric change hoy.

```python
# Common supervised skeleton — model line ta chhara baki sob same
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df[[ ... features ... ]]     # ba df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=101)

model = SomeModel()              # <-- SUDHU eii line ta din-bhede change hoy
model.fit(X_train, y_train)
pred = model.predict(X_test)
```
> Uporer skeleton: data pora → X/y bhag → train/test split → model fit → predict. Day-bhede sudhu `SomeModel()` ar sheshe metric change.

Day onujayi ki boshbe:

```python
# Day 5 — Regression
model = LinearRegression()
# metric: metrics.mean_absolute_error / mean_squared_error / RMSE

# Day 6 — Classification
model = LogisticRegression(solver='lbfgs', max_iter=1000)
# metric: classification_report / confusion_matrix

# Day 7a — Classification
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model = RandomForestClassifier(n_estimators=100, criterion='entropy')
# metric: classification_report / confusion_matrix
```
> Prottek Day er jonno sudhu constructor ar metric family alada — baki logic same.

K-Means alada, karon eta Unsupervised — **no split, no y**:

```python
# Day 7b — Unsupervised (K-Means): kono train/test split nai, kono y nai
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)                    # sudhu X, kono y nai
pred = kmeans.predict(X)         # each row er cluster number
kmeans.inertia_                  # evaluate: inertia + Elbow Method (accuracy na)
```
> K-Means e y nai bole split o nai; quality maper jonno accuracy na, **inertia** ar **Elbow Method** use kori.

⚠️ **Exam tip:** Keu jodi jiggesh kore "K-Means e train_test_split keno korle na?" — uttor: **Unsupervised bole target y nai, tai train/test split er dorkar nai; inertia diye evaluate kori.**

---

## 🧠 Exam e ashar sombhabona sob theke beshi — 3ta difference

Ei tinta difference bar bar ashe. Table gulo choto rakhlam jate mukhosto kora shoja hoy.

### 1) Supervised vs Unsupervised

| | Supervised | Unsupervised |
|----|----|----|
| Target (y) | Ache | Nai |
| Kaj | Regression / Classification | Clustering |
| Split | train_test_split lage | lage na |
| Example | Day 5, 6, 7a | Day 7b (K-Means) |

### 2) Regression vs Classification (dutai Supervised)

| | Regression | Classification |
|----|----|----|
| Output | Continuous number | Category / label |
| Example | Price (Day 5) | Survived, Kyphosis (Day 6, 7a) |
| Metric | MAE / MSE / RMSE | accuracy, precision, recall, f1 |

### 3) Single Tree vs Ensemble (Forest)

| | Decision Tree | Random Forest |
|----|----|----|
| Ki | Ekta single tree | Onek (100) tree er ensemble |
| Kaj | Ekbar split kore decision | Sob tree er **majority vote** |
| Problem | **Overfit** kore beshi | Averaging kore overfit kome |
| Lab result | Accuracy **0.76** | Accuracy **0.84** (better!) |

🧠 **Mne rakho:** Random Forest = onek Decision Tree eksathe (bagging + random features). Beshi tree er vote nile ekta tree er bhul dhaka pore jay → tai Forest (0.84) single Tree (0.76) ke haraylo eii lab e.

---

## ✅ Last-minute revision (30 second e ekbar poro)

- **2 bhag:** Supervised (y ache) vs Unsupervised (y nai).
- **Supervised er 2 kaj:** Regression (number) + Classification (category).
- **Unsupervised er kaj:** Clustering (K-Means).
- **Number → Linear** | **category/yes-no → Logistic/Tree/Forest** | **label nai, group → K-Means**.
- **Forest > single Tree** (0.84 > 0.76) — karon overfitting kome.
- Sob supervised code **ek i skeleton**, sudhu `model = ...` line ta bodlay.

Tumi parbe, inshallah. Ei map ta mathay thakle exam er jekono "kon algorithm / kon type" question tumi confidently solve korte parbe. Best of luck! 🌟
