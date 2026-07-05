# 🏠 Day 5 — Linear Regression (USA_Housing)

> ICT-4202 AI Lab · Bus-ride revision · Exam kal — cholo shesh kore feli 💪

---

## 🎯 Ek line e: Day 5 e ki shikhi

Day 5 e amra **Linear Regression** diye **house er Price predict** kori — Price ekta **continuous number** (taka/dollar er moto), tai eta **Supervised Regression** problem.

- **Supervised** = amader kache label/answer (Price) ache training data te.
- **Regression** = output ekta continuous number (0, 1 class na).
- Dataset: `USA_Housing.csv`.

🧠 **Mne rakho:** Regression = "koto?" (how much) predict kora. Classification = "kon category?" predict kora.

---

## 📖 Theory: Linear Regression ki

Linear Regression ekta straight line (best-fit line) diye input feature ar output er relationship ta shikhe.

Line er equation (intuition):

```
Price ≈ b0 + b1·(feature1) + b2·(feature2) + ... + bn·(featuren)
```

- **b0 = intercept** → shob feature 0 hole line ta kothay Y-axis kata (base value).
- **b1, b2, ... = coefficients** → protita feature Price ke koto tane, koto weight.
- Model training mane = emon b0, b1... khuja jate line ta shob point er **shobcheye kache** thake (error minimum).

**Kobe use kori?**
- Jokhon output ekta **continuous number** (price, salary, temperature, weight).
- Jokhon feature ar target er moddhe **motamuti linear (straight-line) relationship** ache.

### Regression vs Classification (exam favourite)

| Bishoy | Regression | Classification |
|---|---|---|
| Output | Continuous number | Category / class |
| Example | House Price | Survived / Died |
| Day | Day 5 | Day 6, 7a |
| Fit kore | Straight line | Boundary / S-curve |

⚠️ **Exam tip:** "Linear Regression continuous value predict kore, tai eta Regression" — ei ek line likhe dile marker khushi.

---

## 🗂️ Dataset: USA_Housing.csv

**5000 rows × 7 columns.** Ekjon real estate agent USA r different city te house er Price predict korte chay.

| Column | Mane ki |
|---|---|
| `Avg. Area Income` | Je city te house, shei elakar manush er average income |
| `Avg. Area House Age` | Oi elakar bari gula average koto purono |
| `Avg. Area Number of Rooms` | Average koyta room |
| `Avg. Area Number of Bedrooms` | Average koyta bedroom |
| `Area Population` | Elakar population |
| `Price` | 🎯 **TARGET** — jeta amra predict korbo |
| `Address` | Text (bari r thikana) — **DROP kori** |

🧠 **Address keno drop kori?** Address ekta **text/string** — model to number niye kaj kore, text theke seedha shikhte pare na. Tai `Address` ba onno text column bad diye shudhu numeric feature rakhi.

---

## 🧑‍💻 Full code walkthrough

### 1) Import + data porhi

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

USAhousing = pd.read_csv('USA_Housing.csv')   # CSV ta DataFrame e load kori
USAhousing.head()        # prothom 5 row dekhi
USAhousing.info()        # 5000 non-null, float64(6) + object(1=Address)
USAhousing.describe()    # mean, std, min, max — quick stats
USAhousing.columns       # column er naam gula
```

Ekhane: `info()` dekhabe 6 ta numeric (float64) column ar 1 ta text (`object` = Address) column.

### 2) EDA — data ke chokhe dekhi

```python
sns.pairplot(USAhousing)                 # shob column er jora-jora scatter — relationship dekhi
sns.distplot(USAhousing['Price'])        # Price er distribution — motamuti NORMAL (bell shape)
sns.heatmap(USAhousing.corr(), annot=True)  # correlation heatmap (number soho)
```

`distplot(Price)` dekhle bujhi Price ekta shundor **bell-shaped / normal** distribution — Linear Regression er jonno bhalo lokkhon.

Ekhon shudhu numeric column niye correlation clearly dekhi:

```python
house = USAhousing[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms',
                    'Avg. Area Number of Bedrooms','Area Population','Price']]
house.corr()
sns.heatmap(house.corr(), cmap="YlGnBu", annot=True)   # rongin heatmap
```

**Price er sathe kon feature koto correlated (mukhosto rakho):**

| Feature | Correlation with Price |
|---|---|
| Avg. Area Income | **0.64** (sobcheye beshi) |
| Avg. Area House Age | 0.45 |
| Area Population | 0.41 |
| Avg. Area Number of Rooms | 0.34 |
| Avg. Area Number of Bedrooms | 0.17 (sobcheye kom) |

⚠️ **Exam tip:** "Price er sathe sobcheye beshi correlated feature = **Avg. Area Income (0.64)**" — eta jiggesh korte pare.

### 3) X (features) ar y (target) banai

```python
X = USAhousing[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms','Area Population']]   # input features
y = USAhousing['Price']                                             # target
```

X e shudhu 5 ta numeric feature (Address bad, Price bad). y = Price.

### 4) Train/Test split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=101)
```

- `test_size=0.4` → data r **40% test**, 60% train.
- `random_state=101` → shuffle ta fix kora, tai protibar **same split** pao (reproducible).

### 5) Model train kori

```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()      # khali model banai
lm.fit(X_train, y_train)     # training data theke best-fit line shikhe
```

`lm.fit()` er por model best b0 (intercept) ar b1..b5 (coefficients) shikhe fele.

---

## 📐 Model evaluation: intercept + coefficients

```python
print(lm.intercept_)     # -2640159.796851625
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
```

**Coefficient table (exact numbers):**

| Feature | Coefficient |
|---|---|
| Avg. Area Income | 21.528276 |
| Avg. Area House Age | 164883.282027 |
| Avg. Area Number of Rooms | 122368.678027 |
| Avg. Area Number of Bedrooms | 2233.801864 |
| Area Population | 15.150420 |

**Coefficient interpretation (khub important):**
- Baki feature fixed rekhe, kono ekta feature **1 unit barle**, Price **oi coefficient poriman barbe**.
- Jemon: Avg. Area Income **1 unit** barle → Price prai **$21.53** barbe (baki gula same rekhe).
- Avg. Area House Age 1 unit barle → Price **$164883** barbe. (Age er weight beshi tai boro impact.)

🧠 **Mne rakho:** Coefficient = shei feature er "weight" / impact. Intercept = shob feature 0 dhorle base Price.

---

## 🔮 Predictions ar plots

```python
predictions = lm.predict(X_test)              # test data te Price predict kori

plt.scatter(y_test, predictions)              # asol Price vs predicted Price
sns.distplot((y_test - predictions), bins=50) # residual (error) er distribution
```

**`scatter(y_test, predictions)` mane ki?**
- X = asol Price, Y = predict kora Price.
- Point gula jodi ekta **straight (45°) line** er moto boshe → prediction asol er kache → **model bhalo**. ✅

**Residual distplot (y_test − predictions) mane ki?**
- Residual = error = asol − predicted.
- Eta jodi **normal (bell shape) around 0** hoy → Linear Regression er assumption thik ache → **model bhalo fit**. ✅
- Jodi baka/skewed hoto → model kichu miss korche bujhte partam.

---

## 📏 Regression metrics: MAE, MSE, RMSE

```python
from sklearn import metrics
print('MAE:',  metrics.mean_absolute_error(y_test, predictions))
print('MSE:',  metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

**Formula + Banglish mane:**

| Metric | Formula | Mane |
|---|---|---|
| **MAE** | (1/n) Σ \|y − ŷ\| | Average absolute error. Sohoj bujhte. |
| **MSE** | (1/n) Σ (y − ŷ)² | Error ke square kore — boro error ke beshi shasti dey. |
| **RMSE** | √MSE | MSE er root — Price er **same unit** e, tai interpret kora sohoj. Most popular. |

**Ei lab er actual results (exact):**

| Metric | Value |
|---|---|
| MAE | 82288.2225191496 |
| MSE | 10460958907.209692 (≈ 1.046e10) |
| RMSE | 102278.82922291246 |

- Tinta e **LOSS function** → amra egulo ke **MINIMIZE** korte chai (joto kom toto bhalo).
- **Kobe konta?**
  - MAE → sohoj, shob error ke shoman gurutto.
  - MSE/RMSE → jokhon boro error gula beshi kharap (outlier ke shasti dite chao).
  - RMSE → answer ta Price er unit e chao (report korar jonno best).

**R² (goodness of fit) — briefly:**

```python
lm.score(X_test, y_test)   # R² score — higher = better, 1.0 = perfect
```

- R² bole model ta target er variance er koto % explain korlo. **Beshi holei bhalo** (max 1.0).
- (Ei notebook e R² explicitly print kora hoy nai, kintu viva te ashte pare — `lm.score()` diye pao.)

📌 **Detail metric cheatsheet:** dekho `..\Day-00_COMMON_All-Days\03_Metrics-Cheatsheet_Banglish.md`.

---

## ⚠️ Exam tip: minimum code to reproduce (skeleton)

Kom shomoy? Ei ~8 line mne rakhle Day 5 puro reproduce korte parba:

```python
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('USA_Housing.csv')
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms','Area Population']]     # Address & Price bad
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression(); lm.fit(X_train, y_train)             # train
predictions = lm.predict(X_test)                              # predict
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

🧠 **Mne rakho — shared skeleton:** Day 5/6/7 e prai same steps (read_csv → X,y → split → fit → predict → evaluate). Shudhu **model er line** ar **metric family** change hoy. Details: `..\Day-00_COMMON_All-Days\01_Common-sklearn-Workflow_Banglish.md`.

---

## 🎤 Viva Q&A (Day 5)

**Q1. Linear Regression ki?**
Ekta supervised regression algorithm ja input feature ar continuous target er moddhe best-fit straight line shikhe Price predict kore.

**Q2. Regression vs Classification?**
Regression → continuous number (Price). Classification → category/class (Survived 0/1). Day 5 regression, Day 6/7a classification.

**Q3. Coefficient mane ki?**
Protita feature er weight. Baki fixed rekhe oi feature 1 unit barle Price koto barbe. Jemon Income er coef 21.53 → 1 unit Income = +$21.53 Price.

**Q4. Intercept (b0) mane ki?**
Shob feature 0 hole predicted base Price. Ei lab e `lm.intercept_ = -2640159.79`.

**Q5. MAE vs RMSE?**
MAE = average absolute error, sohoj, shob error shoman gurutto. RMSE = √MSE, boro error ke beshi shasti dey ebong Price er same unit e thake tai interpret kora sohoj. Ekhane MAE ≈ 82288, RMSE ≈ 102278.

**Q6. Address column keno drop korlam?**
Eta text/string, model shudhu number niye kaj kore — text theke seedha shikhte pare na, tai drop kori.

**Q7. random_state=101 mane ki?**
Train/test split er random shuffle fix kore dey → protibar same split → result reproducible.

**Q8. Price er sathe sobcheye correlated feature konta?**
**Avg. Area Income (0.64)**.

**Q9. scatter(y_test, predictions) e straight line mane ki?**
Predicted Price asol Price er kache → model bhalo fit korche. Residual normal (bell) holeo model bhalo.

---

### ✅ Quick recap card

- Dataset: **USA_Housing.csv** (5000×7), target = **Price**.
- Model: **LinearRegression**, `test_size=0.4`, `random_state=101`.
- Top feature: **Avg. Area Income (0.64)**.
- Intercept: **-2640159.79**.
- MAE **82288** · MSE **1.046e10** · RMSE **102278**.
- Metric gula **minimize** kori; R² **maximize** (1.0 = perfect).

Cross-day boro picture: `..\Day-00_COMMON_All-Days\02_ML-Big-Picture_Map_Banglish.md`. 🚌 Tumi parbe — best of luck kal! 🎯
