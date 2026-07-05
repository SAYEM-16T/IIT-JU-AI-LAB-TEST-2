# 🎯 Metrics Master Cheatsheet (Banglish) — Day 5, 6, 7 ek jaygay

Tumi ekhon busy, bus e boshe phone e revise korcho — tension nio na. 😎
Ei ek file e **shob evaluation metric** ache. Kal exam e "kon metric use korbo" prosno ashlei tumi ready.

> 🧠 **Mne rakho:** Metric mane holo "amar model koto bhalo?" — er measure. Kintu **kon type er problem** (regression / classification / clustering) tar upor depend kore kon metric use korbe. Tai age chino **kon din kon type**, tarpor metric.

Quick map (memorize this first):

| Type | Din | Metric family |
|---|---|---|
| Regression | Day 5 | MAE, MSE, RMSE, R² |
| Classification | Day 6, 7a | confusion matrix, Accuracy, Precision, Recall, F1 |
| Clustering | Day 7b | inertia, elbow method |

---

## 📏 1. Regression Metrics (Day 5 — Linear Regression, USA_Housing)

Regression e amra **continuous number** (Price) predict kori. Tai metric measure kore: *actual dam ar predicted dam er moddhe koto gap (error)?*

Notation: `y` = actual value, `ŷ` (y-hat) = predicted value, `n` = koto ta sample.

### Formula gulo (phone-safe, plain text)

```
MAE  = (1/n) * Σ |y - ŷ|
MSE  = (1/n) * Σ (y - ŷ)^2
RMSE = sqrt( (1/n) * Σ (y - ŷ)^2 )   # mane sqrt(MSE)
```

### Banglish meaning

- **MAE (Mean Absolute Error):** protita error er **absolute value** er average. Shobcheye **easy bujhte** — "average e koto taka gap". Sign (+/-) matter kore na.
- **MSE (Mean Squared Error):** error ke **square** kore average. Square korle **boro error gulo beshi punish** hoy (choto error kome jay, boro error fole othe). Kintu unit ta square hoye jay (taka² — bujhte oshubidha).
- **RMSE (Root Mean Squared Error):** MSE er square root. Ei jonno unit abar **original y-unit** (taka) te fire ashe → **interpret kora easy**, ei jonno **shobcheye popular**.

### Kobe konta bhalo?

| Metric | Kobe use korbe |
|---|---|
| MAE | Shob error ke soman gurutto dite chao, outlier er beshi punish chao na |
| MSE | Boro error ke beshi punish korte chao |
| RMSE | Y er unit e interpret korte chao (most common report) |

> 🧠 **Mne rakho:** MAE, MSE, RMSE — tinta i **LOSS function**. Loss mane error → tai **jto KOM, tto BHALO** (minimize koro). Bujhte confuse hoyo na: "higher = better" ei tinta te **NA**.

### Lab er actual number (USA_Housing) — mukhosto kore rakho

```python
from sklearn import metrics
print('MAE:',  metrics.mean_absolute_error(y_test, predictions))   # 82288.22...
print('MSE:',  metrics.mean_squared_error(y_test, predictions))    # 10460958907.20... (≈1.046e10)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # 102278.82...
```

| Metric | Value |
|---|---|
| MAE | ≈ **82288** |
| MSE | ≈ **1.046e10** (10460958907) |
| RMSE | ≈ **102278** |

- Notice: **MSE onek boro** karon ta square (taka²). RMSE (≈1 lakh taka) ta **realistic gap** dey — average e ei porimaan taka model miss korche.

### R² (R-squared) — "goodness of fit"

- R² bole model ta actual data er variation er **koto % explain** korche.
- Range: **0 theke 1**. **Higher = better**, 1.0 = perfect fit.
- Code (notebook e explicitly compute kora hoy nai, kintu standard):

```python
lm.score(X_test, y_test)   # returns R² — 1.0 er kachakachi hole bhalo
```

> ⚠️ **Exam tip:** R² **only** metric jeta "higher better" (0–1). MAE/MSE/RMSE holo "lower better". Guliye felo na.

---

## 📊 2. Classification Metrics (Day 6 Titanic + Day 7a Decision Tree & Random Forest)

Classification e output **category** (0/1, survived/died, absent/present). Tai error mane "kotogula ke thik class e felechi, kotogula bhul".

### 🔲 Confusion Matrix — shob metric er base

sklearn `confusion_matrix(y_test, predictions)` return kore ei 2x2 layout (0 = negative, 1 = positive):

```
                 Predicted 0     Predicted 1
   Actual 0   [    TN      ,        FP     ]   <- upor row = actual 0
   Actual 1   [    FN      ,        TP     ]   <- niche row = actual 1
```

- **TN** (True Negative): 0 chilo, 0 i bolechi ✅
- **FP** (False Positive): 0 chilo, bhul kore 1 bolechi ❌ (false alarm)
- **FN** (False Negative): 1 chilo, bhul kore 0 bolechi ❌ (miss kore felechi)
- **TP** (True Positive): 1 chilo, 1 i bolechi ✅

> 🧠 **Mne rakho:** Diagonal (TN, TP) = **thik**. Off-diagonal (FP, FN) = **bhul**. Thik gulo joto beshi, model tto bhalo.

### Formula gulo

```
Accuracy  = (TP + TN) / total          # overall koto % thik
Precision = TP / (TP + FP)             # jegulo ke 1 bolechi, tar moddhe koto ta shotti 1
Recall    = TP / (TP + FN)             # actual 1 gulor moddhe koto ta dhorte perechi (sensitivity)
F1        = 2 * (P * R) / (P + R)      # precision ar recall er harmonic mean (balance)
```

- **support** = oi class er kotogula true sample test set e chilo (count matro, score na).

### Precision vs Recall — kobe konta important? (viva favourite)

- **Precision important** jokhon **false alarm (FP) costly**. Example: spam filter — bhalo mail ke spam bola kharap. "1 bolle jate shotti 1 hoy."
- **Recall important** jokhon **miss (FN) costly**. Example: cancer/disease detect — rogi ke bhul kore "sustho" bola bipojjonok. "Ekta o positive miss kora jabe na."
- **F1** dorkar jokhon dutar **balance** chao, ba data **imbalanced** (ek class beshi).

> ⚠️ **Exam tip:** Precision = "of predicted positive". Recall = "of actual positive". Denominator alada (FP vs FN) — ei jaygay bhul beshi hoy.

### Lab er actual confusion matrix + accuracy — 3 ta case

**(a) Day 6 — Titanic Logistic Regression** → accuracy **0.82**

```python
confusion_matrix(y_test, predictions)
# array([[149,  14],
#        [ 33,  71]])
```

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | TN=149 | FP=14 |
| **Actual 1** | FN=33 | TP=71 |

**(b) Day 7a — Decision Tree (kyphosis)** → accuracy **0.76**

```python
confusion_matrix(y_test, predictions)
# [[15  5]
#  [ 1  4]]
```

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | TN=15 | FP=5 |
| **Actual 1** | FN=1 | TP=4 |

**(c) Day 7a — Random Forest (kyphosis)** → accuracy **0.84**

```python
confusion_matrix(y_test, rfc_pred)
# [[18  2]
#  [ 2  3]]
```

| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | TN=18 | FP=2 |
| **Actual 1** | FN=2 | TP=3 |

> 🧠 **Mne rakho:** Same kyphosis data te **Random Forest (0.84) > Decision Tree (0.76)**. Karon single DT **overfit** kore; RF onek gula tree er **majority vote** ney, tai beshi accurate ar generalize kore bhalo.

### Ekbare pura report

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))  # precision, recall, f1, support ek shathe
print(confusion_matrix(y_test, predictions))
```

Titanic report ta emon dekhte:

```
             precision  recall  f1-score  support
        0      0.82      0.91     0.86      163
        1      0.84      0.68     0.75      104
  accuracy                        0.82      267
```

---

## 🔵 3. Clustering Metric (Day 7b — K-Means, Live.csv)

Clustering **unsupervised** — **kono target y nai**, tai accuracy/precision **use kora jabe na** (kar shathe milaba? true label i to nai)! Ekhane amra measure kori cluster gula koto "tight".

### Inertia

- **inertia_** = protita point tar **nearest centroid** theke koto dur, tar **squared distance er sum** (within-cluster sum of squared distances).
- **Lower = better** (points centroid er kachakachi = tight cluster). **0 = optimal**.
- Normalized na — tai ekta number "bhalo na kharap" bola shokto, comparison lage.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
kmeans.inertia_          # 96.24989...  (≈ 96.25) — ei lab e high, tai great fit noy
```

> Lab er inertia ≈ **96.25** — motamuti high, mane Live.csv ei clustering te khub tight na.

### 📉 Elbow Method — K (koyta cluster) select kora

- Different K (1 theke 10) er jonno K-Means chalao, protibar **inertia** save koro, tarpor **inertia vs K** plot koro.
- K barle inertia komte thake. Je point e komar **speed hothat slow** hoye jay (graph ta "elbow"/kunui er moto banke) — oita i **optimal K**.

```python
clustering_score = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='random', random_state=42)
    kmeans.fit(X)
    clustering_score.append(kmeans.inertia_)   # protita K er inertia jomao
plt.plot(range(1, 11), clustering_score)       # elbow (banka point) dekhe K choose koro
```

> 🧠 **Mne rakho:** Elbow er por K baralei inertia beshi kome na, sudhu over-complicate hoy. Kunui er jaygay theme jao.

---

## 🧠 4. Kon din kon metric — final quick table

| Day | Algorithm | Task | Metric use korbe |
|---|---|---|---|
| 5 | Linear Regression | Regression | MAE, MSE, RMSE, R² |
| 6 | Logistic Regression | Classification | classification_report, confusion_matrix |
| 7a | Decision Tree / Random Forest | Classification | classification_report, confusion_matrix |
| 7b | K-Means | Clustering | inertia_, elbow method |

### 3-line memory hook

- **Regression** → number predict → gap mapo → **MAE / MSE / RMSE** (kom = bhalo), **R²** (beshi = bhalo).
- **Classification** → category predict → **confusion matrix** theke Accuracy / Precision / Recall / F1.
- **Clustering** → no label → **inertia** (kom = bhalo) + **elbow** diye K.

> ⚠️ **Exam tip:** **KHOBORDAR — regression metric ar classification metric MIX korba na!** Regression e accuracy/precision/confusion matrix chao na (category nai). Classification e MAE/RMSE chao na (number nai). Ei bhul korle full marks chole jabe. 🚫

---

Best of luck kal! 🌟 Ei chart ta mne thakle metric section e tumi confident. Tumi parbe — shanto thako, formula ta plain text e lekho, actual number gula (82288 / 102278 / 0.82 / 0.84 / 96.25) example hishebe likhe dile examiner khushi. 💪
