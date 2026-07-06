# 📗 Day 5 — Linear Regression (আমার নোট · সহজ বাংলায়)

> **এক ফাইলেই সব:** theory + পূর্ণাঙ্গ code + প্রতিটা metric-এর definition + Address + real result + revision card + viva।
> Technical term গুলো ইচ্ছে করে English-এ রাখা, কারণ exam/viva-তে ওগুলোই লাগবে। অন্য কোনো ফাইল না খুলেও এই নোট থেকেই পুরো Day 5 পড়া যাবে।

---

## 📑 সূচিপত্র

(নিচে section গুলো এই নম্বর অনুযায়ীই সাজানো — নম্বর ধরে scroll করলেই পাবে)

1. Day 5 এক নজরে
2. কীভাবে কাজ করে — line বসানো
3. Least Squares (সবচেয়ে ভালো line)
4. Coefficient ও Intercept
5. Correlation
6. Train / Test Split
7. Evaluation Metrics — সব definition (R², MAE, MSE, RMSE, residual)
8. StatQuest video extra (p-value)
9. Address column কেন বাদ
10. 💻 পূর্ণাঙ্গ কোড + output
11. এক নজরে recap
12. ⭐ শেষ-মুহূর্তের Revision Card
13. Viva প্রশ্ন

---

## ১. Day 5 এক নজরে

তোমার প্রতিবেশী (real estate agent) জানতে চায় — একটা বাড়ির কিছু তথ্য দিলে তার **দাম (Price)** কত হবে?

- **Linear Regression** = কিছু feature দেখে একটা **সংখ্যা (continuous number)** predict করার algorithm। এখানে সংখ্যা = দাম।
- **Type:** Supervised (আগে থেকে ৫০০০ বাড়ির দাম "জানা" আছে) · **Regression** (output একটা number, category নয়)।
- **Dataset:** `USA_Housing.csv` — 5000 row, 7 column, কোনো missing value নেই।

> 🧠 **নিয়ম:** Number predict → **Regression**। হ্যাঁ/না বা category predict → Classification।

---

## ২. কীভাবে কাজ করে? — একটা line বসানো

ছোটবেলার অঙ্ক: **y = mx + c** (সরলরেখা)।
- `x` = feature (যেমন এলাকার আয়), `y` = predict করব (দাম)
- `m` = slope (line কত খাড়া), `c` = intercept (line কোথায় শুরু)

আমাদের ৫টা feature, তাই line-টা আসলে:
> **Price = (m₁ × Income) + (m₂ × House Age) + ... + intercept**

---

## ৩. "সবচেয়ে ভালো line" মানে কী? — Least Squares

- প্রতিটা বাড়ির **আসল দাম** আর **line যে দাম বলছে** — এই দুটোর পার্থক্য = **residual** (error)।
- Model line-টাকে ঘুরিয়ে ঘুরিয়ে এমন জায়গায় বসায় যেখানে **সব residual-এর বর্গের যোগফল (sum of squared residuals) সবচেয়ে কম**। এই পদ্ধতির নাম **Least Squares**।
- **বর্গ (square) কেন?** (ক) +/− error যেন কাটাকাটি হয়ে না যায়, (খ) বড় ভুলকে বেশি শাস্তি দিতে।

> 🔗 `lm.fit(X_train, y_train)` — এই এক লাইনই ভেতরে ভেতরে Least Squares-এর কাজ করে line বসায়।

---

## ৪. Coefficient আর Intercept

Model train হলে প্রতিটা feature-এর একটা **coefficient** (ওই `m`) বের হয়। মানে:
> "বাকি সব ঠিক রেখে, ওই feature ১ ইউনিট বাড়লে দাম কত বাড়বে।"

তোমার lab-এর result:
| জিনিস | মান | মানে |
|---|---|---|
| Avg. Area Income (coef) | ≈ ২১.৫ | আয় ১ বাড়লে দাম ~$21.5 বাড়ে |
| Avg. Area House Age (coef) | ≈ ১,৬৪,৮৮৩ | সবচেয়ে বড় প্রভাব |
| **Intercept** | ≈ −২৬,৪০,১৫৯ | সব feature ০ হলে line-এর base value |

> 🔗 **Code:** `lm.coef_` (coefficient গুলো), `lm.intercept_` (intercept)।

---

## ৫. Correlation — কোন feature দামের সাথে বেশি জড়িত?

`heatmap`/`corr()` দিয়ে দেখা হয় (মান −১ থেকে +১, যত ১-এর কাছে তত শক্তিশালী):
- **Avg. Area Income → 0.64** (সবচেয়ে বেশি) · House Age 0.45 · Population 0.41 · Rooms 0.34 · Bedrooms 0.17

---

## ৬. Train / Test Split — কেন?

- **Train set (৬০%)** → model এখান থেকে শেখে।
- **Test set (৪০%)** → model কখনো দেখেনি এমন data; এখানে যাচাই করি model **সত্যিই শিখেছে নাকি মুখস্থ করেছে**।
- `random_state=101` → প্রতিবার একই split (result মেলানো যায়)।

> 🔗 `train_test_split(X, y, test_size=0.4, random_state=101)`

---

## ৭. Evaluation Metrics — সব definition একসাথে

Prediction (`lm.predict(X_test)`) করার পর model কতটা ভালো তা মাপি।

### 📈 R² (R-squared) — *যত বেশি তত ভালো*
- **Definition:** model, target-এর মোট ওঠানামার (variation) **কত অংশ ব্যাখ্যা করল** তার মাপ।
- **Formula:** `R² = 1 − (SS_res / SS_tot)`
  - `SS_res = Σ(actual − predicted)²` (model যত ভুল করল)
  - `SS_tot = Σ(actual − mean)²` (শুধু গড় দিয়ে predict করলে যত ভুল হতো)
- **Range:** 0–1 (1 = perfect, 0 = গড়ের চেয়ে ভালো কিছু নয়)।
- **Lab:** R² = **0.918** → feature গুলো দামের **~92% variation** ব্যাখ্যা করে (খুব ভালো fit)।

### 📉 MAE — Mean Absolute Error — *যত কম তত ভালো*
- **Definition:** প্রতিটা ভুলের absolute value-এর গড় (সোজা "গড়ে কত ভুল")।
- **Formula:** `MAE = (1/n) × Σ |actual − predicted|`
- Target-এর একই unit (ডলার)। বোঝা সবচেয়ে সহজ; বড় ভুলে বেশি শাস্তি দেয় না।
- **Lab:** ≈ **82,288**।

### 📊 MSE — Mean Squared Error — *যত কম তত ভালো*
- **Definition:** প্রতিটা ভুলের **বর্গের গড়**।
- **Formula:** `MSE = (1/n) × Σ (actual − predicted)²`
- বড় ভুলকে **অনেক বেশি** শাস্তি দেয়; কিন্তু unit = target² (ডলার²) → বোঝা কঠিন।
- **Lab:** ≈ **1.046 × 10¹⁰**।

### 📐 RMSE — Root Mean Squared Error — *যত কম তত ভালো*
- **Definition:** MSE-এর **square root**।
- **Formula:** `RMSE = √[ (1/n) × Σ (actual − predicted)² ]`
- MSE-এর মতোই বড় ভুলে শাস্তি, **কিন্তু** unit আবার target-এর সমান (ডলার) → interpret সহজ। **সবচেয়ে জনপ্রিয়।**
- **Lab:** ≈ **1,02,278** → গড়ে ~১ লাখ ডলার ভুল।

> 🧠 **এক লাইনে:** MAE = সোজা গড় ভুল · MSE = বড় ভুলে শাস্তি (unit²) · RMSE = MSE-এর root, target-এর unit-এ। MAE/MSE/RMSE হলো **loss** (কমাতে চাই); R² হলো fit-এর মাপ (বাড়াতে চাই)।

### 🔔 Residual normal হওয়া — মানে কী?
- **Residual** = প্রতিটা point-এর ভুল = `actual − predicted`।
- **"Residual normal"** = এই ভুলগুলো যদি **normal distribution (bell curve, mean ≈ 0)** মেনে চলে।
- **কেন দরকার:** এটা Linear Regression-এর একটা **assumption**। Residual bell-shape হলে বোঝা যায় model কোনো **pattern মিস করেনি** — ভুলগুলো শুধু random noise।
- **খারাপ লক্ষণ:** residual skewed/বাঁকা হলে বা pattern দেখালে → সম্পর্কটা হয়তো linear নয়।
- **Lab:** `sns.histplot(y_test - predictions)` → bell-shape এসেছিল ✅।

---

## ৮. StatQuest video যা extra শেখায়

Video টা linear regression-কে ৩ ভাগে ভাগ করে — (১) line কীভাবে বসে = **Least Squares** (§৩), (২) line কত ভালো = **R²** (§৭), (৩) নতুন: **p-value**।

- **R²-এর intuition (video-র ভাষায়):** `R² = (Var(mean) − Var(line)) / Var(mean)` — শুধু গড় দিয়ে predict করার তুলনায় line কত % error কমাল।
- **p-value:** কম data হলে R² কপালগুণে বেশি আসতে পারে। p-value বলে R²-টা **statistically significant** কিনা। p-value **ছোট (< 0.05)** হলে সম্পর্কটা আসল।
- ⚠️ তোমার lab-এ p-value আলাদা করে বের করা হয়নি — R², MAE, RMSE-ই main। p-value শুধু বুঝে রাখো।

---

## ৯. Address column — বাদ দিলাম কেন?

`USA_Housing.csv`-এ **Address** = প্রতিটা বাড়ির **ঠিকানা (text)**, যেমন `208 Michael Ferry Apt. 674, Laurabury, NE 37010`।

**`X`-এ বাদ দিলাম কারণ:**
1. এটা **text**, number নয়। ML model শুধু **number** নিয়ে হিসাব করে।
2. প্রতিটা address **unique** — দাম predict করার মতো useful pattern নেই।
3. দিলে **error** দিত: `could not convert string to float: '208 Michael Ferry...'`।

> 🧠 **Rule:** feature (`X`)-এ শুধু **number** column রাখো। Text হয় বাদ দাও, নাহয় number-এ convert করো (Day 6-এ `get_dummies` দিয়ে Sex → 0/1)।

---

## ১০. 💻 পূর্ণাঙ্গ কোড (output সহ)

পুরো Day 5 এই এক ব্লকেই। কমেন্টে (`#`) কী হবে / output কী লেখা আছে।

```python
# ===== ১. Import =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# ===== ২. Data porা o dekha =====
USAhousing = pd.read_csv('day-05__USA_Housing.csv')
USAhousing.head()        # প্রথম ৫ row
USAhousing.info()        # -> 5000 row, 7 column, সব non-null (missing নেই)
USAhousing.describe()    # -> mean/std/min/max (Price mean ≈ 12.3 লাখ)
USAhousing.columns

# ===== ৩. EDA (ছবি) =====
sns.pairplot(USAhousing)                                   # feature-দের জোড়া scatter
sns.histplot(USAhousing['Price'], kde=True)                # Price ~ bell-shape (normal)
sns.heatmap(USAhousing.corr(numeric_only=True), annot=True)# correlation (Income 0.64 সর্বোচ্চ)

# ===== ৪. X (features) ও y (target) =====
X = USAhousing[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms','Area Population']]   # Address বাদ (text)
y = USAhousing['Price']

# ===== ৫. Train/Test split =====
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# ===== ৬. Model train (fit) =====
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)          # -> LinearRegression()   (Least Squares দিয়ে line বসল)

# ===== ৭. Coefficients =====
print(lm.intercept_)              # -> -2640159.7968...
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df
#   Avg. Area Income               21.53
#   Avg. Area House Age        164883.28
#   Avg. Area Number of Rooms  122368.68
#   Avg. Area Number of Bedrooms 2233.80
#   Area Population                15.15

# ===== ৮. Predictions + ছবি =====
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)                    # প্রায় straight line = ভালো prediction
sns.histplot((y_test - predictions), bins=50, kde=True)   # residual bell-shape = ভালো

# ===== ৯. Metrics (মূল evaluation) =====
from sklearn import metrics
print('MAE :', metrics.mean_absolute_error(y_test, predictions))          # -> 82288.22
print('MSE :', metrics.mean_squared_error(y_test, predictions))           # -> 1.046e10
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # -> 102278.83
print('R2  :', metrics.r2_score(y_test, predictions))                     # -> 0.9177
```

> ⚠️ **Lab-এর মূল কোডে ছিল** `sns.distplot(...)` আর `USAhousing.corr()` — কিন্তু নতুন library version-এ ওগুলো **error দেয়**, তাই fix করা হয়েছে:
> `distplot` → **`histplot`** এবং `corr()` → **`corr(numeric_only=True)`**। (exam-এ দুটোই জেনে রেখো।)

---

## ১১. এক নজরে পুরো Day 5

1. **Linear Regression** = line বসিয়ে **number (দাম) predict** → Supervised, Regression।
2. Line বসে **Least Squares** দিয়ে (residual²-এর যোগফল সবচেয়ে কম)।
3. **Coefficient** = feature ১ বাড়লে দাম কত বাড়ে; **Intercept** = base।
4. যাচাই: **R²** (বেশি ভালো) + **MAE/MSE/RMSE** (কম ভালো) + **residual normal**।
5. **Code skeleton:** `read_csv → X,y → train_test_split → LinearRegression → fit → predict → metrics`।

---

## ১২. ⭐ শেষ-মুহূর্তের Revision Card

পরীক্ষার ঠিক আগে শুধু এই বক্সটা এক নজর দেখলেই হবে —

| বিষয় | মনে রাখো |
|---|---|
| **Algorithm** | `LinearRegression()` |
| **Type / Task** | Supervised · Regression (Price predict) |
| **Dataset / Target** | USA_Housing (5000×7) · target = **Price** |
| **Feature (X)** | ৫টা number column (Address ও Price বাদ) |
| **কীভাবে শেখে** | Least Squares (residual²-এর যোগফল min) |
| **Split** | `test_size=0.4, random_state=101` |
| **Metrics** | R² **0.918** · MAE **82,288** · RMSE **1,02,278** |
| **R² vs MAE/RMSE** | R² = কত ভালো (বেশি=ভালো) · MAE/RMSE = কত ভুল (কম=ভালো) |
| **মূল ৩ লাইন কোড** | `lm=LinearRegression()` → `lm.fit(X_train,y_train)` → `lm.predict(X_test)` |

---

## ১৩. ❓ Viva-তে সবচেয়ে বেশি যা আসে

| প্রশ্ন | ছোট উত্তর |
|---|---|
| Linear Regression কী? | feature দেখে continuous number predict করার supervised regression algorithm |
| Least Squares কী? | residual²-এর যোগফল minimize করে best-fit line বসানো |
| Coefficient-এর মানে? | ওই feature ১ বাড়লে target কত বাড়ে (বাকি সব ঠিক রেখে) |
| Intercept কী? | সব feature ০ হলে line-এর base value |
| MAE vs RMSE? | দুটোই কম ভালো; RMSE বড় ভুলে বেশি শাস্তি দেয়, target-এর unit-এ থাকে |
| R² কী? | model কত % variation ব্যাখ্যা করে (0–1, বেশি ভালো) |
| Residual কী? | actual − predicted (প্রতিটা point-এর ভুল); normal হলে model ভালো |
| Address column কেন বাদ? | text + unique, useful pattern নেই, model number চায় |
| `random_state` কেন? | প্রতিবার একই train/test split → result reproducible |

---

> ✅ এই নোটটাই তোমার Day 5-এর সম্পূর্ণ note — theory + code + formula + result + revision + viva সব এখানে।
> চাইলে হাতে-কলমে চালাও: `Practice_Day-05_Linear-Regression.ipynb` · plot/output সহ বিস্তারিত: `README.md`।
> **Best of luck! 💪**
