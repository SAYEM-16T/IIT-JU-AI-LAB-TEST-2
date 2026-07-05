# 🎤 Viva & Short Questions — Day 5/6/7 (Banglish)

Tomar exam kaal, tai eta ekdom last-minute revision. Bus e boshe porar jonno banano — chhoto chhoto answer, sob important prosno ek jaygay.

> 🧠 Mne rakho: Viva te examiner **confidence** dekhe. Answer chhoto rakho (2-4 line), but term gula thik bolo (Linear Regression, overfitting, confusion matrix). Ekta term thik bolle 80% marks emnitei ashe.

Tumi paare. Cholo shuru kori. 💪

---

## 🌍 General ML Concepts

**Q1. Supervised vs Unsupervised learning ki?**
Supervised e data te **label / target (y)** thake — model input theke output predict korte shekhe (Day 5, 6, 7a). Unsupervised e **kono target nai** — model nije data er moddhe group/pattern khuje (Day 7b K-Means).

**Q2. Regression vs Classification?**
Duita e supervised. **Regression** → continuous number predict kore (Price, weight). **Classification** → category/discrete predict kore (Survived 0/1, absent/present).

**Q3. Feature vs Target ki?**
**Feature (X)** = input columns diye amra predict kori (income, age, rooms). **Target (y)** = jeta predict korte chai (Price, Survived). Feature ke "independent variable", target ke "dependent variable"-o bole.

**Q4. Overfitting ki + kivabe komano?**
Overfitting mane model training data **mukhosto** kore fele — train e valo, test/new data te kharap. Komanor upay: beshi data, simpler model, **pruning** (tree), **Random Forest** (many trees average), regularization, ar train/test split diye check.

**Q5. train_test_split keno kori?**
Data ke train + test e bhag kori jate model **notun (unseen)** data te koto valo kaj kore ta jachai kori. Eki data te train + test korle overfitting dhora porbe na.

**Q6. random_state keno dei?**
Split ta **reproducible** korte — proti bar eki rows train/test e jabe. Na dile prottek run e vinno split, vinno result, tulona kora kothin. (Notebook e `random_state=101`.)

**Q7. fit() vs predict() parthokko?**
`fit(X_train, y_train)` → model ke **train/learn** koray (pattern shekhe). `predict(X_test)` → shekha pattern diye **notun output** ber kore. Age fit, tarpor predict.

**Q8. test_size mane ki?**
Total data er koto fraction test set e jabe. Day 5 → `0.4` (40% test), Day 6 → `0.30`, Day 7a → `0.30`.

**Q9. Loss/cost function mane?**
Model koto vul korche tar maap. Amra eta **minimize** korte chai. Regression e MAE/MSE/RMSE, clustering e inertia.

---

## 🏠 Day 5 — Linear Regression

**Q10. Linear Regression ki?**
Ekta **supervised regression** algorithm ja feature gula theke ekta **continuous number** predict kore, ekta best-fit line/plane bosiye. Day 5 e `USA_Housing.csv` theke **Price** predict korechi.

**Q11. Coefficient mane ki?**
Ekta feature 1 unit barle target koto barbe (baki feature fixed rekhe), tar maap. Jemon `Avg. Area Income` coef ≈ **21.53** → income 1 unit barle Price ~21.53 barbe.

**Q12. Intercept mane ki?**
Sob feature = 0 hole predicted Price. Notebook e `lm.intercept_` = **-2640159.79**. (Eta physically meaning nao korte pare, just line er base.)

**Q13. MAE vs MSE vs RMSE?**
| Metric | Formula idea | Bishesh |
|---|---|---|
| MAE | mean of \|error\| | bujhte shohoj |
| MSE | mean of error² | boro error ke beshi shasti |
| RMSE | √MSE | y-unit e, most popular |

Notebook: MAE **82288.22**, MSE **1.046e10**, RMSE **102278.83**. Tinta e loss → **minimize** korte chai.

**Q14. RMSE ke keno beshi use kori?**
RMSE er unit target er unit-er soman (dollar), tai interpret kora shohoj. Ar MSE er moto boro error ke beshi penalize kore.

**Q15. R² (R-squared) ki?**
Goodness-of-fit score: model target er koto % variation explain korte pare (0 → kichu na, 1.0 → perfect). Code e `lm.score(X_test, y_test)`. Beshi = valo.

**Q16. Residual ki, ar eta normal keno chai?**
Residual = **actual − predicted** (`y_test - predictions`). Egula **normal distribution** (bell curve, center 0) hole bujha jay error random — Linear Regression er assumption thik ache, model bhalo fit. Notebook e `sns.distplot` diye check korechi.

**Q17. Address column ke keno drop korlam?**
Oita **text**, model number chai. Tai X e nei nai. Sudhu 5 ta numeric feature diye X banano hoyeche.

---

## 🚢 Day 6 — Logistic Regression

**Q18. Logistic Regression ki?**
Naam e "regression" holeo eta **supervised classification** algorithm — category (0/1, Yes/No) predict kore. Exact 0/1 na diye **probability (0–1)** dey. Day 6 e Titanic e **Survived** predict korechi.

**Q19. Sigmoid function ki?**
Je kono real number ke **(0,1)** range e map kore — mane probability banay. Er graph **S-shaped** curve. Ei jonnoi Logistic Regression straight line na, "S" curve fit kore.

**Q20. Threshold ki?**
Probability ke class e convert korar cut-off, usually **0.5**. prob ≥ 0.5 → class **1**, prob < 0.5 → class **0**.

**Q21. Confusion matrix er 4 ta ghor?**
| | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

Notebook: `[[149, 14], [33, 71]]` → TN=149, FP=14, FN=33, TP=71. Layout: **[[TN, FP],[FN, TP]]**.

**Q22. Precision vs Recall — kobe konta important?**
- **Precision = TP/(TP+FP)** → predicted-positive er moddhe koto shotti positive. Muddo jokhon **false alarm (FP) costly** (spam filter).
- **Recall = TP/(TP+FN)** → actual-positive er koto dhorte parlam. Muddo jokhon **positive miss kora costly** (cancer/disease detection).

**Q23. F1 score keno lage?**
F1 = **2·P·R/(P+R)** — precision o recall er **harmonic mean**. Jokhon duita e balance chai (ba data imbalanced), ekta single number diye judge korte F1 use kori. Notebook e class 1 er F1 ≈ **0.75**.

**Q24. get_dummies + drop_first=True keno?**
`get_dummies` categorical column (Sex, Embarked) ke **0/1 dummy columns** e vange, karon model number chai. `drop_first=True` ekta column bad dey → **dummy variable trap (multicollinearity)** eray. (Sex → shudhu 'male' col thake.)

**Q25. Age missing gula kivabe fill korla?**
Custom `impute_age` function diye **Pclass onujayi fixed age value** boslam: class 1 → 37, class 2 → 29, class 3 → 24. Cabin er onek NaN chilo tai **drop** korlam.

**Q26. accuracy koto pele, ar keno beshi na dekhi?**
Accuracy ≈ **0.82**. Kintu data imbalanced hole accuracy misleading — tai precision/recall/F1 o confusion matrix dekha lage.

---

## 🌳 Day 7A — Decision Tree & Random Forest

**Q27. Decision Tree er node types?**
- **Root Node**: uporer node, ekhan theke split shuru.
- **Decision Node**: split er por paoya internal node.
- **Leaf / Terminal Node**: ar split hoy na — final decision ekhane.
- **Sub-tree**: gaach er ekta branch/section.

**Q28. Entropy vs Gini?**
Duita e split er **quality maape** (criterion). **Entropy** → information gain diye impurity maape. **Gini** → misclassification probability maape (ektu faster). Notebook e `criterion='entropy'` use kora hoyeche.

**Q29. Pruning ki?**
Gaacher extra node/branch **kete fela** jate ta beshi complex na hoy — mane **overfitting komanor** technique.

**Q30. Random Forest ki?**
Ekta **ensemble** method — onek gula Decision Tree combine kore. Classification e trees **majority vote** dey, regression e output er **average**. Notebook e `n_estimators=100` (100 ta tree).

**Q31. Bagging + Bootstrap ki?**
**Bootstrap** = training data theke random subset (replacement soho) tule ekekta tree banano. **Bagging** (Bootstrap Aggregating) = eibhabe vinno vinno subset e onek tree banano, tarpor result combine. Ei randomness overfitting komay.

**Q32. RF keno single DT theke bhalo?**
Ekta DT easily **overfit** kore (high variance). RF onek tree er result average/vote kore, tai stable o general. Notebook e proof: **RF accuracy 0.84 > DT accuracy 0.76**.

**Q33. n_estimators ki?**
Forest e koyta Decision Tree thakbe. Notebook e **100**. Beshi tree → sadharonoto stable, kintu slow.

**Q34. Kyphosis dataset er target/features?**
Target = **Kyphosis** (`absent`/`present`), `.map({'absent':0,'present':1})` diye 0/1 kora hoyeche. Features = **Age, Number, Start**.

**Q35. Day 7a te StandardScaler keno, fit_transform vs transform?**
`StandardScaler` feature gula ke mean 0 / std 1 e ane. **Train e `fit_transform`, test e sudhu `transform`** — jate test er info train scaler e na dhoke (**data leakage** eray).

---

## 🔵 Day 7B — K-Means Clustering

**Q36. K-Means kivabe kaj kore (2 step)?**
1. **Assignment step**: proti point ke tar **nearest centroid** e assign koro (squared Euclidean distance).
2. **Update step**: proti centroid ke oi cluster er point gular **mean** diye update koro.
Ei dui step **repeat** hoy jotokkhon na point ar change hoy / distance minimize hoy. Converge kore, tobe **local optimum** e atke jete pare (tai multiple bar run).

**Q37. Centroid ki?**
Ekta cluster er **majhkhaner point** (center). K-Means "centroid-based" clustering.

**Q38. Inertia ki?**
`kmeans.inertia_` = protita point theke tar nearest centroid porjonto **within-cluster squared distance er sum**. **Kom inertia = valo fit**, 0 = optimal. Notebook e final inertia ≈ **96.25** (high → besh bhalo fit noy).

**Q39. Elbow method ki?**
K = 1..10 er jonno K-Means chaliye **inertia vs K** plot kori. K barale inertia komte thake; je jaygay improvement **hothat slow** hoy (elbow/kohuni), oitai optimal K. Notebook e elbow ~ **K=4** dekhano hoyeche.

**Q40. K kivabe choose korla, final model e koto?**
Elbow method diye. Notebook e elbow ~4 chilo, kintu **final model `n_clusters=5`** use kora hoyeche.

**Q41. K-Means e keno feature scaling (MinMaxScaler)?**
K-Means **distance** er upor chole. Boro-scale feature (num_likes hajaar) chhoto-scale feature ke dabiye dey. Tai `MinMaxScaler` diye sob **0–1** e ane jate shob feature soman gurutto pay.

**Q42. K-Means e train_test_split ba accuracy nai keno?**
Eta **unsupervised** — kono target/label nai, tai split ba accuracy/precision hoy na. Evaluate kori **inertia + elbow method** diye.

**Q43. status_id / status_published keno drop korla?**
Egula ~unique ID (6997 / 6913 unique) — kono useful pattern nai, model ke confuse kore. Column1–4 sob NaN chilo, tai ogulao drop.

---

## 💻 "Code likhte bolle" — thanda mathay

Examiner "sklearn diye ekta model likho" bolle **ei ekta skeleton** mne rakho (detail: `01_Common-sklearn-Workflow_Banglish` file dekho). Proti day e sudhu **model line** ta bodlay:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)          # features
y = df['target']                       # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=101)

model = SomeModel()                    # <-- SUDHU ei line bodlay
model.fit(X_train, y_train)            # train
pred = model.predict(X_test)           # predict
```
*Ei skeleton mukhosto thakle je kono supervised day likhte parba.*

Per-day model line + metric:

```python
# Day 5 — Linear Regression (regression metrics)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, pred))   # MAE
```

```python
# Day 6 — Logistic Regression (classification metrics)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=1000)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, pred))
```

```python
# Day 7a — Decision Tree / Random Forest
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
# OR:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, criterion='entropy')
```

```python
# Day 7b — K-Means (UNSUPERVISED: no y, no split)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
pred = kmeans.predict(X)
print(kmeans.inertia_)                 # evaluate diye inertia
```

> ⚠️ Exam tip: K-Means likhte bolle **train_test_split / y / accuracy likho na** — oita unsupervised. Sudhu X, fit, predict, inertia.

---

## ⚡ Last 5-minute recall table

| Day | Algorithm | Type | Metric | Key number |
|---|---|---|---|---|
| 5 | Linear Regression | Regression | RMSE | 102278.83 |
| 6 | Logistic Regression | Classification | F1/accuracy | acc 0.82 |
| 7a | Decision Tree | Classification | accuracy | 0.76 |
| 7a | Random Forest | Classification | accuracy | 0.84 |
| 7b | K-Means | Clustering | inertia | 96.25 |

> 🧠 Mne rakho: **Supervised = target ache (5,6,7a); Unsupervised = target nai (7b)**. Regression → number, Classification → category, Clustering → group.

Ekhon ekbar chokh bujhe boshe ei answer gula mone koro. Confidence niye jao — tumi ready! 🎯 Best of luck!
