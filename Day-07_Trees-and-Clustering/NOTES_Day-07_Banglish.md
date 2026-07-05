# 🌳 Day 07 — Decision Tree + Random Forest + K-Means (Banglish Revision)

Assalamu alaikum bondhu! Ei din ta **shob theke boro** — tin ta algorithm ekshathe. Tension nio na, ami step-by-step guide korbo. Bus e boshe ei ekta file porlei Day 7 er full picture matha te boshe jabe. Cholo shuru kori. 💪

Ei din er 3 ta part:

| Part | Algorithm | Type |
|---|---|---|
| A | Decision Tree | Supervised · Classification |
| B | Random Forest | Supervised · Classification |
| C | K-Means | **Unsupervised** · Clustering |

> 🧠 **Mne rakho:** Part A ar B te target `y` **ache** (supervised). Part C te target `y` **NEI** (unsupervised). Eita exam er sob theke boro concept — niche detail e bujhabo.

---
---

# 🌲 PART A — Decision Tree (Supervised Classification)

## 🤔 Decision Tree ki?

Decision Tree ekta **flowchart-er moto** structure. Data ke branch e branch e vaag kore, ar sesh e (leaf node te) ekta decision/outcome dey.

Bhabo tumi ekta "20 questions" khela khelcho — "Age 18 er beshi? haa/na", "Number boro? haa/na" — protita question e data 2 bhage bhag hocche, sesh e answer paa. Thik oi rokom.

⚠️ **Exam tip:** Decision Tree **duitai** kaj kore — **classification** (category predict) ar **regression** (number predict). Ei line ta viva te bolte parle marks!

## 📚 Terminology (viva te ashe)

| Term | Mane ki |
|---|---|
| **Root Node** | Sob theke upore-r node. Ekhan theke pura population divide hoa shuru hoy. |
| **Decision Node** | Root split korar por je node gulo pai. |
| **Leaf / Terminal Node** | Jekhane ar split hoy na — final decision ekhane. |
| **Sub-tree** | Tree-r ekta chhoto section/branch. |
| **Pruning** | Node kete chhoto kora — **overfitting thamano**-r jonno. |

> 🧠 **Mne rakho:** **Pruning = overfitting kombano** (branch kete chhoto kora). Overfitting mane model training data mukhosto kore fele, kintu notun data te fail kore.

## 🔪 criterion — split-er quality kivabe mapi?

Decision Tree protita node e data vaag kore. Kon vaag ta bhalo, seta mape `criterion` diye:

- **`'entropy'`** → **information gain** hisheb kore. Entropy = data koto "mixed"/impure. Split-er por impurity koto kombe, seta max korar chesta.
- **`'gini'`** → Gini impurity. Same kaj, alada formula.

Amader lab e amra **`criterion='entropy'`** use korchi (information gain).

> 🧠 **Mne rakho (viva favourite):** Entropy beshi = data beshi mixed/impure. Split emon chai jate entropy komey, mane **information gain** barey.

## 🩺 Dataset: `kyphosis.csv`

Kyphosis = surgery-r por merudonde ekta baaka obostha. Amra predict korbo surgery-r por Kyphosis **present** naki **absent**.

| Column | Mane |
|---|---|
| `Kyphosis` | **TARGET** — `absent` / `present` |
| `Age` | rogi-r boyosh |
| `Number` | koyta vertebrae involved |
| `Start` | prothom operated vertebra |

Feature = `Age`, `Number`, `Start`. Target = `Kyphosis`.

## 💻 Code — step by step

```python
# Basic library import kori
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline

df = pd.read_csv('kyphosis.csv')   # dataset porlam
df.columns   # Index(['Kyphosis','Age','Number','Start'])
```

```python
# pairplot diye dekhi — hue='Kyphosis' mane absent/present alada rong e dekhabe
sns.pairplot(df, hue='Kyphosis', palette='Set1')
```
Eita EDA — kon feature diye absent/present alada hoy, chokhe dekhar jonno.

```python
# X = feature gulo (Kyphosis baade), y = target
X = df.drop('Kyphosis', axis=1)          # Age, Number, Start
y = df['Kyphosis']
y_numeric = y.map({'absent': 0, 'present': 1})   # target ke 0/1 e convert korlam
```
`map()` diye text label ke number banaisi — model text bujhe na, number lage.

```python
# Train/test split — 30% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.30)
```
`test_size=0.30` mane 70% diye train, 30% diye test.

### ⚖️ Feature Scaling — ar data leakage keno avoid kori

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)   # train e fit_transform
X_test  = sc.transform(X_test)        # test e SUDHU transform
```

- `StandardScaler` shob feature ke **mean 0, std 1** e ene daay (jate boro number chhoto number ke dhakiye na dey).
- **`fit_transform` sudhu train e** — mane mean/std sikhbe **train theke**.
- **`transform` test e** — test er mean/std diye abar fit korle model test er info agei jene felto.

> ⚠️ **Exam tip — Data leakage:** Test e `fit` korle test data-r info train-e "leak" kore jaay → result artificially bhalo dekhabe, kintu real e cheat. Tai **test e sudhu `transform`**, kokhono `fit` na.

### 🌳 Single Decision Tree train

```python
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
# criterion='entropy' -> information gain diye split quality mape
# random_state=0      -> protibar same result (reproducibility)
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
```

### 📊 Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
#          precision  recall  f1   support
#      0     0.94     0.75   0.83    20
#      1     0.44     0.80   0.57     5
#  accuracy                  0.76    25
print(confusion_matrix(y_test, predictions))
# [[15  5]
#  [ 1  4]]
```

**Decision Tree accuracy = 0.76**

Confusion matrix pora: layout `[[TN, FP],[FN, TP]]` (0 = negative/absent, 1 = positive/present):
- TN = 15 (absent-ke thik absent bolche)
- FP = 5, FN = 1, TP = 4

### 🖼️ Tree ta visualize kori

```python
from sklearn import tree
plt.figure(figsize=(20,25))
tree.plot_tree(dtree, feature_names=X.columns, class_names=['absent','present'],
               rounded=True, filled=True, proportion=True)
plt.show()
```
`plot_tree` puro flowchart ta enke dekhay — kon feature e kivabe split holo.

---
---

# 🌲🌲🌲 PART B — Random Forest (Supervised Classification)

## 🤔 Random Forest ki?

Ekta gaach (single Decision Tree) valo, kintu onek gaach ekshathe = **Forest** → aro strong! Random Forest ekta **ensemble** method — mane **onek Decision Tree combine** kore ekta better answer bananoy.

Analogy: ekjon expert er opinion na niye, **100 jon expert er vote** nile decision beshi reliable hoy. 🗳️

## ⚙️ Kivabe kaj kore (viva te full ashe)

1. **Data prep:** labeled data, train/test split.
2. **Build trees:** protita tree ke training data-r ekta **random subset** (ekta **bootstrap sample**) diye banano hoy. Ar protita **node-e** split korar shomoy **random subset of features** consider kora hoy.
3. **Voting:**
   - Classification → tree-der **majority vote** (beshi vote je class pay).
   - Regression → tree-der output-er **average**.
4. **Bagging + Randomness:**
   - **Bagging** = alada alada data subset diye tree banano.
   - **Randomness** = protita node e random feature subset.
   - Ei dutar jonno **overfitting komey**, generalization barey.

> 🧠 **Mne rakho:** **Bootstrap** = replacement soho random sample tola. **Bagging** = Bootstrap AGGregatING — onek sample e onek tree, tarpor result combine.

## ❓ Overfitting keno komey?

Single tree pura training data mukhosto kore fele (overfit). Kintu Random Forest e protita tree alada data + alada feature dekhe, tai keu ekta beshi mistake korleo baki tree-ra vote diye seta thik kore fele. **Average/vote noise ke cancel kore dey.** ✅

**Popular Decision-Tree algorithms:** Random Forest, ID3, C4.5, CART.

## 💻 Code

```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion="entropy")
# n_estimators=100 -> forest e 100 ta decision tree
rfc.fit(X_train, y_train)

estimator = rfc.estimators_[5]   # forest theke 1 ta tree ber kore ana (index 5)
rfc_pred  = rfc.predict(X_test)
```
`n_estimators=100` mane 100 ta tree bananbo. `rfc.estimators_[5]` diye 6-number tree ta alada dekha jay.

```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, rfc_pred))
# [[18  2]
#  [ 2  3]]
print(classification_report(y_test, rfc_pred))
#          precision  recall  f1   support
#      0     0.90     0.90   0.90    20
#      1     0.60     0.60   0.60     5
#  accuracy                  0.84    25
```

**Random Forest accuracy = 0.84**

```python
# Ekta tree ke file e export kora (visualize korar jonno)
from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file='tree.dot', feature_names=X.columns,
                class_names=['absent','present'], rounded=True, proportion=False,
                precision=2, filled=True)
```

## 🏆 Single DT vs Random Forest

| Model | Accuracy |
|---|---|
| Single Decision Tree | **0.76** |
| Random Forest | **0.84** |

> 🧠 **KEY (must remember):** Single Decision Tree **overfit** kore, tai accuracy komm (**0.76**). Random Forest onek tree average kore overfitting kombay, tai **better (0.84)**. Ei tulona viva-r sure-shot question!

---
---

# 🎯 PART C — K-Means Clustering (UNSUPERVISED)

## 🤔 Unsupervised mane ki? (SOB THEKE BORO CONCEPT)

Ekhon tak Day 5, 6, 7A — shob e amader kache **target `y` chhilo** (Price, Survived, Kyphosis). Ei ke bole **supervised** — "teacher" (answer) ache.

**K-Means e target `y` NEI!** Amader kache sudhu data ache, kono label nai. Model nije nije data-r moddhe **groups (clusters)** khuje ber kore. Ei ke bole **unsupervised**.

> 🧠 **KEY difference (exam e HAMMER koro):**
> - Supervised (Day 5/6/7A): `y` **ache** → tai train/test split ache, accuracy/precision diye evaluate.
> - K-Means (unsupervised): `y` **NEI** → tai **NO train/test split, NO accuracy/precision**. Evaluate kori **inertia** ar **elbow method** diye.

## ⚙️ K-Means kivabe kaj kore

- **Centroid-based** clustering. Centroid = ekta cluster-er majkhaner point (center).
- **Input** = koyta cluster chai (K) + dataset.
- **Iterative** — 2 ta step barbar cholte thake:

  1. **Assignment step:** protita point ke tar **nearest centroid** e assign koro (**squared Euclidean distance** diye distance mapa hoy).
  2. **Update step:** protita centroid ke notun kore hishab koro = oi cluster-er sob point-er **mean**.

- Ei dui step **repeat** hoy jotokkhon na point-ra ar cluster change kore (converge). 
- Guaranteed **converge** kore, kintu kokhono kokhono **local optimum** e atke jete pare → tai onekbar run kora hoy (random_state alada diye).

**Applications:** image segmentation, customer segmentation, species clustering, anomaly detection, languages clustering.

## 📐 K kivabe choose kori? — ELBOW METHOD

Problem: K (koyta cluster) to amra jani na! Solution = **Elbow Method**.

- K = 1 theke onek value porjonto K-Means chalai.
- Protita K-er jonno **inertia** (cost) plot kori (inertia vs K graph).
- K barlei inertia kombe. Kintu ekta jaigay improvement **hothat slow** hoye jay — oi bakano jaiga ta = **"elbow"** = optimal K.

Kunhui-r moto bekano point tai elbow. Oikhane bhalo balance — beshi cluster baraleo bishesh labh nai.

## 📉 Inertia ki?

- `inertia_` = **within-cluster sum of squared distances** — protita point tar nearest centroid theke koto dure, tar squared distance-der total.
- **Lower inertia = better fit.** 0 = perfect (kintu practically kokhono paoa jay na).
- Normalized metric na. Beshi dimension e Euclidean distance fule jay (**curse of dimensionality**) → tokhon age **PCA** kore dimension komano jay.

> 🧠 **Mne rakho:** Inertia = "point gulo tader centroid theke total koto dure (squared)". Choto inertia = tight, valo cluster.

## 📱 Dataset: `Live.csv`

Facebook Live sellers in Thailand (UCI). Original shape = **7050 rows × 16 columns**. Onek cleaning lagbe:

1. **`Column1`–`Column4`** → shob **NaN** (faltu) → **drop**. (16 → 12 cols)
2. **`status_id`, `status_published`** → prai unique ID (6997 & 6913 unique) → clustering e faltu → **drop**.
3. **`status_type`** → 4 category: `video`, `photo`, `link`, `status` → number banate **LabelEncoder**.
4. Numeric features: `num_reactions`, `num_comments`, `num_shares`, `num_likes`, `num_loves`, `num_wows`, `num_hahas`, `num_sads`, `num_angrys`.

## 💻 Code

```python
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline

df = pd.read_csv('Live.csv')
df.shape                       # (7050, 16)
df.info(); df.isnull().sum()   # Column1..4 shob 7050 NaN
```

```python
# 1) faltu all-NaN column gulo felo -> 12 cols
df.drop(['Column1','Column2','Column3','Column4'], axis=1, inplace=True)
df['status_type'].unique()     # ['video','photo','link','status']
# 2) unique ID column gulo felo (clustering e kaje lage na)
df.drop(['status_id','status_published'], axis=1, inplace=True)
```

```python
# 3) status_type (text) -> integer label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['status_type'] = le.fit_transform(df['status_type'])
```
`LabelEncoder` category ke 0,1,2,3 e convert kore (video/photo/link/status → number).

```python
# 4) shob feature ke 0..1 range e scale kori
y = df
cols = y.columns
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
y = ms.fit_transform(y)
y = pd.DataFrame(y, columns=[cols])
X = y.values          # <-- unsupervised: NO train/test split, NO target y
```
`MinMaxScaler` shob value ke [0,1] e ane. Khyal koro — **X banaisi, kintu kono `y` target nai, kono split nai**.

### 📐 Elbow method loop (K = 1..10)

```python
from sklearn.cluster import KMeans
clustering_score = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='random', random_state=42)
    kmeans.fit(X)
    clustering_score.append(kmeans.inertia_)   # inertia = within-cluster SS distance

plt.plot(range(1, 11), clustering_score)
plt.scatter(4, clustering_score[3], s=200, c='red', marker='*')  # elbow ~ K=4
plt.title('The Elbow Method'); plt.xlabel('No. of Clusters'); plt.ylabel('Clustering Score')
```
K=1 theke 10 porjonto chalie protita-r inertia store korlam, tarpor plot — elbow khujte.

### 🎯 Final model (5 clusters)

```python
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
pred = kmeans.predict(X)        # protita sample kon cluster e
df['Cluster'] = pd.DataFrame(pred, columns=['cluster'])
df['Cluster'].value_counts()    # 1:4131, 0:2145, 4:334, 2:251, 3:189
```

```python
# model-er parameters
labels1    = kmeans.labels_
centroids1 = kmeans.cluster_centers_
kmeans.inertia_                 # 96.24989550305203
```

**Final inertia = 96.25** (moderately high → ei data te khub perfect fit na, kintu clusters paisi).

> ⚠️ **Exam tip:** K-Means e `predict` er por **accuracy/confusion_matrix nai** — karon target-i nai! Sudhu `value_counts()` (kon cluster e koyta point), `inertia_`, `labels_`, `cluster_centers_` — eigulo dekha hoy.

---
---

# 🧾 Combined Viva Q&A (tinta algorithm mile)

**Q1. Entropy ki?**
Entropy hocche data koto "mixed"/impure tar map. Decision Tree e `criterion='entropy'` diye emon split khuji jate entropy kombe — mane **information gain** barbe. Entropy beshi = beshi impure.

**Q2. Overfitting ki?**
Model training data eto mukhosto kore fele je notun/test data te fail kore. Signs: training accuracy high, test accuracy low. Decision Tree e eta beshi hoy; **pruning** diye ba **Random Forest** diye komano jay.

**Q3. Random Forest keno single Decision Tree theke bhalo?**
Single tree overfit kore (amader lab e **0.76**). Random Forest **onek tree-r vote/average** ney, tai noise cancel hoy, overfitting komey, generalization barey (**0.84**). Ekjon na, 100 jon expert-er vote — beshi reliable.

**Q4. Bagging ki?**
**B**ootstrap **AGG**regat**ING**. Training data theke onek **bootstrap sample** (replacement soho random) niye protita sample e alada tree banano, tarpor tader result combine (vote/average) kora. Random Forest-er core idea.

**Q5. Bootstrap sample ki?**
Original data theke replacement soho (eki row bar bar ashte pare) random subset tola. Protita tree ekta alada bootstrap sample e train hoy.

**Q6. Supervised vs Unsupervised?**
- **Supervised:** target `y` **ache** (Linear/Logistic Regression, Decision Tree, Random Forest). Train/test split + accuracy/MAE diye evaluate.
- **Unsupervised:** target `y` **NEI** (K-Means). Model nije group khuje. Inertia/elbow diye evaluate, no split.

**Q7. Elbow method ki?**
K choose korar technique. K = 1..N porjonto K-Means chalie inertia vs K plot kori. Je K te improvement hothat slow hoy (bekano "elbow" point), setai optimal K.

**Q8. Inertia ki?**
Within-cluster sum of squared distances — protita point tar nearest centroid theke koto dure (squared), tar total. **Lower = better** (tighter cluster). Amader lab e final inertia = **96.25**.

**Q9. K kivabe choose kori?**
**Elbow method** diye (inertia vs K plot-e bekano point). Amader lab e elbow ~K=4 dekhano, kintu final model **K=5** e chalano hoyeche.

**Q10. K-Means-er 2 ta step ki?**
1. **Assignment:** protita point ke nearest centroid e boshao (squared Euclidean distance).
2. **Update:** centroid = oi cluster-er point-der mean. Repeat → converge (local optimum possible).

**Q11. Data leakage keno khottikor + kivabe avoid?**
Test data-r info jodi training-e chole ashe (jemon test e `fit`), result cheat hoye jay. Avoid: **train e `fit_transform`, test e sudhu `transform`**.

**Q12. Decision Tree terminology bolo.**
Root node (upore, shuru), Decision node (split-er por), Leaf/Terminal node (final, ar split nai), Sub-tree (branch), Pruning (overfitting thamate node kata).

---

## ⚡ Last-minute cheat sheet

| Jinish | Value / Fact |
|---|---|
| DT criterion | `'entropy'` (information gain) |
| DT accuracy | **0.76** (overfit) |
| DT confusion | `[[15,5],[1,4]]` |
| RF n_estimators | **100** |
| RF accuracy | **0.84** (better) |
| RF confusion | `[[18,2],[2,3]]` |
| K-Means scaler | MinMaxScaler [0,1] |
| K-Means final K | **5** |
| Final inertia | **96.25** |
| Cluster counts | 1:4131, 0:2145, 4:334, 2:251, 3:189 |

Tumi ei tuku porle Day 7 complete. 3 ta algorithm, 1 ta line e mne rekho:
**"Tree overfit → Forest fix kore (vote). K-Means e y nai → inertia + elbow."** 🌟

Best of luck kalker exam e! Tumi parbe. 🤲
