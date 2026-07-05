# 🧩 All Algorithms Comparison (Day 5+6+7) — Banglish Cheat Sheet

Tumi bus e boshe ache, exam kalke — chinta koro na. Ei ek note e **5 ta algorithm** ek shathe pashapashi. Ei ekta file mne rakhle whole lab er "kon algorithm konta" clear hoye jabe. Cholo shuru kori 🚀

**5 ta algorithm:** Linear Regression, Logistic Regression, Decision Tree, Random Forest, K-Means.

> 🧠 **Mne rakho:** prothom 4 ta **Supervised** (target `y` ache), sudhu **K-Means Unsupervised** (target nai).

---

## 📊 Big Comparison Table (Part 1 — kaj + model line)

Phone e overflow na kore, tai 2 table e bhag korlam. Ei prothom part e "eta ki kore + kon line" boshe ache.

| Algorithm | Type | Task |
|---|---|---|
| Linear Regression | Supervised | Regression (number) |
| Logistic Regression | Supervised | Classification (0/1) |
| Decision Tree | Supervised | Classification |
| Random Forest | Supervised | Classification |
| K-Means | Unsupervised | Clustering |

**sklearn model line (exact — mukhostho koro):**

| Algorithm | Model line |
|---|---|
| Linear Reg | `LinearRegression()` |
| Logistic Reg | `LogisticRegression(solver='lbfgs', max_iter=1000)` |
| Decision Tree | `DecisionTreeClassifier(criterion='entropy', random_state=0)` |
| Random Forest | `RandomForestClassifier(n_estimators=100, criterion='entropy')` |
| K-Means | `KMeans(n_clusters=5, random_state=42)` |

> ⚠️ **Exam tip:** shared skeleton er moddhe **sudhu ei model line ta change hoy**. Baki `read_csv → X,y → train_test_split → fit → predict` shob same (K-Means chara).

---

## 📊 Big Comparison Table (Part 2 — hyperparam + metric + result)

| Algorithm | Key hyperparam | Metric |
|---|---|---|
| Linear Reg | (default) | MAE / MSE / RMSE |
| Logistic Reg | `solver='lbfgs'`, `max_iter=1000` | accuracy / precision / recall / F1 |
| Decision Tree | `criterion='entropy'`, `random_state=0` | accuracy / confusion matrix |
| Random Forest | `n_estimators=100`, `criterion='entropy'` | accuracy / confusion matrix |
| K-Means | `n_clusters=5`, `random_state=42` | inertia + elbow method |

**Ei lab e amader result (exact numbers):**

| Algorithm | This-lab result |
|---|---|
| Linear Reg | RMSE = **102278.83** (MAE 82288) |
| Logistic Reg | accuracy = **0.82** |
| Decision Tree | accuracy = **0.76** |
| Random Forest | accuracy = **0.84** |
| K-Means | inertia = **96.25**, elbow ~ **K=4** |

> 🧠 **Mne rakho:** kyphosis dataset e Random Forest (0.84) **beat koreche** single Decision Tree (0.76) ke — karon RF onek tree er vote combine kore.

---

## 🃏 Per-Algorithm Mini-Cards

Prottek ta te: **ek line kaj + 1 line pros + 1 line cons.** Chhoto kore mne rakho.

### 📈 Linear Regression
- **Kaj:** continuous number predict kore (jemon house `Price`).
- ✅ **Pros:** simple, fast, coefficient gulo interpret kora easy.
- ❌ **Cons:** dhore ney relationship **linear** — jodi data curved hoy tahole fit kharap.

### 🔵 Logistic Regression
- **Kaj:** category (0/1, Survived/not) predict kore — asole **probability** dey.
- ✅ **Pros:** probability + interpretable, classification e strong baseline.
- ❌ **Cons:** mostly linear boundary; complex non-linear pattern e struggle kore.

### 🌳 Decision Tree
- **Kaj:** flowchart-er moto haan/na question kore leaf e decision dey.
- ✅ **Pros:** khub **easy to interpret**, visualize kora jay, scaling optional.
- ❌ **Cons:** **overfit** kore (tai pruning lage), ekta tree unstable.

### 🌲🌲 Random Forest
- **Kaj:** onek (100 ta) Decision Tree er **majority vote** — ekta ensemble.
- ✅ **Pros:** **accurate & robust**, overfitting kom (bagging + random features).
- ❌ **Cons:** slower + **kom interpretable** (100 ta tree ekshathe dekha kothin).

### 🎯 K-Means
- **Kaj:** label chara data ke K ta cluster e bhag kore (nearest centroid).
- ✅ **Pros:** **label lage na** (unsupervised), simple + fast.
- ❌ **Cons:** tomake **K choose korte hoy** (elbow lage), local optimum e atke jete pare.

---

## 🧠 Frequently-Confused Pairs (clear kore dilam)

### 1️⃣ Linear vs Logistic Regression
- Naam e "Regression" thakleo **Logistic asole Classification**! Eta ekta common trap!
- **Linear** → output ekta **continuous number** (jemon 1250000 taka — illustrative). Straight line fit kore.
- **Logistic** → output ekta **probability (0–1)** → threshold 0.5 diye 0/1 class. **S-shaped sigmoid** curve fit kore.
- Metric o alada: Linear = MAE/RMSE, Logistic = accuracy/precision/recall/F1.

> ⚠️ **Exam tip:** "Logistic Regression regression naki classification?" → answer: **CLASSIFICATION** (sudhu naam e regression).

### 2️⃣ Decision Tree vs Random Forest
- **Decision Tree = ekta gaach.** Random Forest = **onek gaacher jongol** (100 ta tree).
- Single tree easily **overfit** kore; RF onek tree er **vote/average** niye overfitting komay.
- Interpretability: Tree **easy** to read; Forest **kothin** (but beshi accurate).
- Ei lab: DT 0.76 vs RF **0.84** → Forest jity geche.

> 🧠 **Mne rakho:** Random Forest = **Bagging + random feature subset** at each node = ensemble learning.

### 3️⃣ Classification vs Clustering
- **Classification (supervised):** label/target `y` **age thekei ache** (jemon Survived 0/1). Model shekhe kon class.
- **Clustering (unsupervised):** **kono label nai** — model nijei similar point gulo group kore (K-Means).
- Tai K-Means e **train_test_split nai, accuracy nai** — sudhu inertia + elbow diye evaluate.

> ⚠️ **Exam tip:** "supervised vs unsupervised" er ekmatra key difference: **target `y` ache ki nai.** Ache = supervised, nai = unsupervised.

---

## 📦 One Import Line per Algorithm (exact)

Ei import gulo copy-paste level e mukhostho rakho.

```python
# Linear Regression (Day 5)
from sklearn.linear_model import LinearRegression

# Logistic Regression (Day 6)
from sklearn.linear_model import LogisticRegression

# Decision Tree (Day 7a)
from sklearn.tree import DecisionTreeClassifier

# Random Forest (Day 7a)
from sklearn.ensemble import RandomForestClassifier

# K-Means (Day 7b)
from sklearn.cluster import KMeans
```
> Upore: prottek algorithm er jonno kon module theke import — 5 line, ek jhalak e mne pore jabe.

---

## 🏁 Last-Minute Recap (bus theke namar age)

- 4 ta Supervised (Linear, Logistic, Tree, Forest) + 1 ta Unsupervised (K-Means).
- **Regression** = number (Linear). **Classification** = class (Logistic, Tree, Forest). **Clustering** = group (K-Means).
- Model line chara skeleton same: `read_csv → X,y → train_test_split → fit → predict → metric`.
- Numbers: RMSE **102278.83** · Logistic **0.82** · DT **0.76** · RF **0.84** · K-Means inertia **96.25** (elbow **K=4**).

Tumi parbe! Ei table gula chokh bondho korle jate bhese othe — practice koro. Best of luck kalke 💪
