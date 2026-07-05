# 🚌 READ FIRST — Bus Study Plan (Day 5/6/7)

Assalamu alaikum / hi 👋 — tumi ekhon bus e, exam kalke, hate 2 ghonta. Tension nio na. Ei ekta file tomake bujhiye dibe **ki ache** ar **kon order e porle** sob cover hoye jabe. Beshi kichu na — Day 5, 6, 7 er lab. Amra guccha guccha kore porbo. Cholo shuru kori.

> 🧠 **Mne rakho:** ei teen din ta asole **ekta** kajer teen ta version. Ekbar skeleton ta bujhle, baki sob easy.

---

## 🗂️ Ek nazore: ei folder e ki ki ache

Age jene nao kothay ki. Char ta din-folder + ekta common folder (jate 6 ta shared note).

### Day folders

| Folder | Ki ache (short) |
|---|---|
| `Day-05_Linear-Regression` | **Linear Regression** — `USA_Housing.csv` diye house **Price** (continuous number) predict. Supervised · Regression. |
| `Day-06_Logistic-Regression` | **Logistic Regression** — `titanic_train.csv` diye **Survived** (0/1) predict. Supervised · Classification. |
| `Day-07_Trees-and-Clustering` | **Decision Tree + Random Forest** (`kyphosis.csv`) ar **K-Means** (`Live.csv`). Boro din — supervised + unsupervised duita ache. |
| `Day-00_COMMON_All-Days` | Ei common folder — niche er 6 ta shared note ekhane thake. |

### 6 ta common file (`Day-00_COMMON_All-Days/`)

| File | Kaj (one-line) |
|---|---|
| `00_READ-FIRST_Bus-Study-Plan.md` | **Ei file** — entry point, ki porba kon order e. |
| `01_Common-sklearn-Workflow_Banglish.md` | Ekta sklearn **skeleton** ja 3 din-i cover kore (read → split → fit → predict → evaluate). |
| `02_ML-Big-Picture_Map_Banglish.md` | ML er boro chobi — supervised vs unsupervised, kon din kisera sathe connect hoy. |
| `03_Metrics-Cheatsheet_Banglish.md` | Shob metric ek jaygay — MAE/MSE/RMSE, accuracy/precision/recall/F1, confusion matrix, inertia. |
| `04_All-Algorithms-Comparison_Banglish.md` | Shob algorithm er **side-by-side** comparison table. |
| `05_Viva-and-Short-Questions_Banglish.md` | Viva-r common proshno + choto uttor. |

> ⚠️ **Exam tip:** file er number-i tomar porar order. `00 → 01 → 02 …` — confusion hobe na.

---

## 📌 Reading ORDER (2 ghontar bus er jonno)

Ekdom recommended path. Ekhane minute budget ta follow korle 120 min e sob dhuke jabe.

| # | Ki porba | ~Min |
|---|---|---|
| a | **Ei file** (`00_READ-FIRST...`) | 5 |
| b | `02_ML-Big-Picture_Map_Banglish.md` — sob connect korbe | 10 |
| c | `01_Common-sklearn-Workflow_Banglish.md` — 1 skeleton, 3 din cover | 15 |
| d | **Day 5** notes (`Day-05_Linear-Regression/`) | 25 |
| e | **Day 6** notes (`Day-06_Logistic-Regression/`) | 25 |
| f | **Day 7** notes (`Day-07_Trees-and-Clustering/`) — boro | 30 |
| g | `03_Metrics-Cheatsheet_Banglish.md` + `04_All-Algorithms-Comparison_Banglish.md` + `05_Viva-and-Short-Questions_Banglish.md` | baki ~10 |

**Total ≈ 120 min.** ✅

Keno ei order?
- Age **big picture** (b) — tahole prottek din kothay boshe bujhe felba.
- Tarpor **skeleton** (c) — ei ekta code pattern 3 din-e ghure firey ashe, sudhu ekta line palte.
- Erpor din-guccha (d, e, f) — ekhon protita din chena chena lagbe.
- Sheshe metrics + comparison + viva (g) — revision + viva-r jonno finishing touch.

> 🧠 **Mne rakho:** Day 7 sob theke boro (Decision Tree + Random Forest + K-Means — 3 ta jinis). Tai oitar jonno 30 min rakha.

---

## ⏱️ Jodi sudhu 30 min thake (fast-track)

Bus late? Ghum bhanglo late e? Tension na. **Ei 3 ta** por, tahole exam-e ar viva-te lorai kora jabe:

1. **`01_Common-sklearn-Workflow_Banglish.md`** (~12 min) — ek skeleton ja 3 din cover kore. Eta na jene gele kono din-i likhte parba na.
2. **`04_All-Algorithms-Comparison_Banglish.md`** (~10 min) — ek table e kon din kon algorithm, kon type, kon metric. Puro syllabus ek page e.
3. **`05_Viva-and-Short-Questions_Banglish.md`** (~8 min) — common viva proshno guli chokh buliye nao.

> ⚠️ **Exam tip:** 30 min thakle EDA/plot er details skip koro. Skeleton + kon din kon metric — ei duita thakle base cover.

---

## 🎯 3 ta jinis exam-e sob theke important

Egulo mukhosto na, **bujhe** rakho. Prithibir 80% marks ekhane.

### 1️⃣ Shared sklearn skeleton (protita supervised din-i ei ek pattern)

```python
# read → X/y banao → split → model banao → fit → predict → evaluate
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df[[...features...]]        # ba df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=101)

model = SomeModel()            # <-- SUDHU EI LINE-TA prottek din palte
model.fit(X_train, y_train)
pred = model.predict(X_test)
# tarpor: regression -> MAE/MSE/RMSE ; classification -> classification_report/confusion_matrix
```
*Ei skeleton porei bujho: prottek din sudhu `model = ...` line ta ar evaluation metric palay — baki sob same.*

Kon din kon model boshbe:

| Day | `model = ...` |
|---|---|
| 5 | `LinearRegression()` |
| 6 | `LogisticRegression(solver='lbfgs', max_iter=1000)` |
| 7a | `DecisionTreeClassifier(criterion='entropy', random_state=0)` |
| 7a | `RandomForestClassifier(n_estimators=100, criterion='entropy')` |
| 7b | `KMeans(n_clusters=k)` → **exception!** unsupervised, no y, no split |

> ⚠️ **Exam tip:** K-Means-e `train_test_split` **nai**, target `y` **nai**, accuracy **nai**. Eta unsupervised — evaluate koro `inertia_` ar **elbow method** diye.

### 2️⃣ Supervised vs Unsupervised + Regression vs Classification

Ei chena-jana-ta viva-r first proshno hote pare.

- **Supervised** = target `y` ache (uttor jana). Day 5, 6, 7a.
  - **Regression** = output continuous number (e.g. Price). → **Day 5**.
  - **Classification** = output category/label (e.g. 0/1, absent/present). → **Day 6, Day 7a**.
- **Unsupervised** = target `y` **nai** (label chara data). Group khuje ber kore. → **Day 7b (K-Means)**.

> 🧠 **Mne rakho:** "Target column ache?" → thakle **supervised**, na thakle **unsupervised**. Number predict korle **regression**, category predict korle **classification**.

### 3️⃣ Kon metric kon day e

Ei table ta mukhosto koro — exam-e "eta diye keno evaluate korle?" ashe.

| Day | Task | Metric |
|---|---|---|
| 5 | Regression | **MAE, MSE, RMSE** (`sklearn.metrics`) |
| 6 | Classification | **classification_report** + **confusion_matrix** (accuracy/precision/recall/F1) |
| 7a | Classification | same — classification_report + confusion_matrix |
| 7b | Clustering | **inertia_** + **elbow method** (kono accuracy na) |

> ⚠️ **Exam tip:** Regression-e RMSE most popular (y-r same unit e bojha jay). Classification-e class imbalance thakle sudhu accuracy dekho na — precision/recall dekho. Details `03_Metrics-Cheatsheet_Banglish.md`-e.

---

## 💪 Sesh kotha

Bondhu, ei teen din-er puro syllabus asole **ek skeleton + kon din kon model + kon metric** — er beshi kichu na. Ekbar oita mathay boshle, exam ta tomar nijer kore neoya jinis. 

Bus theke nama porjonto ekbar hoyle porei felba — tahole exam hall e sudhu confidence niye dhukba. Tumi parba, easy. Chalo, `02_ML-Big-Picture_Map_Banglish.md` khulo. 🚀
