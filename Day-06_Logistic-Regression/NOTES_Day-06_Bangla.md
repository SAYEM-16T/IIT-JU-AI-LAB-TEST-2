# 📗 Day 6 — Logistic Regression (আমার নোট · সহজ বাংলায়)

> **এক ফাইলেই সব:** theory + পূর্ণাঙ্গ code + সব metric-এর definition + real result + revision card + viva।
> Technical term গুলো ইচ্ছে করে English-এ। অন্য ফাইল না খুলেও এই নোট থেকেই পুরো Day 6 পড়া যাবে।

---

## 📑 সূচিপত্র

(section গুলো এই নম্বর অনুযায়ীই সাজানো — নম্বর ধরে scroll করো)

1. Day 6 এক নজরে
2. Logistic vs Linear — কেন Classification
3. Sigmoid function ও threshold
4. Dataset — Titanic
5. Data Cleaning (impute + drop)
6. Dummy variables (get_dummies)
7. Model train + predict
8. Evaluation — Confusion Matrix ও সব metric
9. 💻 পূর্ণাঙ্গ কোড + output
10. এক নজরে recap
11. ⭐ শেষ-মুহূর্তের Revision Card
12. Viva প্রশ্ন

---

## ১. Day 6 এক নজরে

একটা বাড়ির বদলে এবার প্রশ্ন: **Titanic-এর একজন যাত্রী কি বেঁচেছিল, নাকি মারা গিয়েছিল?**

- **Logistic Regression** = কিছু feature দেখে একটা **category (0/1)** predict করার algorithm। এখানে: **Survived = 1 (বেঁচেছে)** বা **0 (মারা গেছে)**।
- **Type:** Supervised · **Classification** (output একটা category, সংখ্যা নয়)।
- **Dataset:** `titanic_train.csv` — 891 যাত্রী, target = **Survived**।

> 🧠 **নিয়ম:** সংখ্যা predict → Regression (Day 5)। হ্যাঁ/না বা category predict → **Classification** (Day 6)।

---

## ২. Logistic vs Linear — কেন Classification?

| | Linear Regression (Day 5) | Logistic Regression (Day 6) |
|---|---|---|
| Output | continuous number (দাম) | category (0/1) |
| যা বসায় | সরলরেখা (straight line) | **S-shaped curve (sigmoid)** |
| উত্তর | যেকোনো সংখ্যা | **0 থেকে 1 এর মধ্যে probability** |

- নাম-এ "Regression" থাকলেও **Logistic আসলে Classification** algorithm — এটা exam-এর common ফাঁদ! ⚠️
- Linear দিয়ে 0/1 predict করলে সমস্যা হয় (line 0-1 এর বাইরে চলে যায়)। তাই Logistic একটা **S-বক্ররেখা** বসায় যেটা সবসময় 0–1 এর মধ্যে থাকে।

---

## ৩. Sigmoid function ও threshold

- **Sigmoid (Logistic) function** = এমন একটা গাণিতিক function যা **যেকোনো সংখ্যাকে 0 আর 1 এর মধ্যে** নিয়ে আসে → এটাই **probability**।
- আকৃতি = ইংরেজি **"S"**।
- **Threshold (সাধারণত 0.5):** probability **0.5 এর বেশি হলে → class 1**, কম হলে → class 0।
  - যেমন: model বলল 0.8 → "বেঁচেছে (1)"; 0.3 → "মারা গেছে (0)"।

> 🧠 এক লাইনে: Logistic = feature → sigmoid → probability (0–1) → threshold দিয়ে 0/1।

---

## ৪. Dataset — Titanic

| Column | কী |
|---|---|
| **Survived** | 🎯 target — 1 (বেঁচেছে) / 0 (মারা গেছে) |
| Pclass | class (1/2/3) |
| Sex | male/female (text) |
| Age | বয়স (কিছু **missing**) |
| SibSp, Parch | সাথে ভাই-বোন/বাবা-মা-সন্তান সংখ্যা |
| Fare | টিকিটের দাম |
| Cabin | কেবিন (**প্রচুর missing** → বাদ) |
| Embarked | কোন বন্দর (text, 2টা missing) |
| Name, Ticket | text (useless → বাদ) |

---

## ৫. Data Cleaning — এই day-এর আসল অংশ

Model-এ দেওয়ার আগে data পরিষ্কার করতে হয়।

1. **Age-এর missing পূরণ (impute):** boxplot-এ দেখা যায় বড় class = বেশি বয়সী। তাই Pclass অনুযায়ী বসাই — **class 1 → 37, class 2 → 29, class 3 → 24**।
2. **Cabin drop:** এত missing (687/891) যে কাজে লাগবে না → পুরো column বাদ।
3. **`dropna()`:** বাকি missing row (Embarked-এর 2টা) বাদ।

---

## ৬. Dummy variables — text কে number বানানো

Model শুধু **number** বোঝে, text (male/female) বোঝে না।

- **`pd.get_dummies(train['Sex'], drop_first=True)`** → `Sex` কে `male` column-এ ভাঙে (male=1, female=0)।
- **`drop_first=True` কেন?** একটা column বাদ দেয় → **dummy variable trap (multicollinearity)** এড়ায়। (2টা column-এর দরকার নেই, 1টাই যথেষ্ট।)
- `Embarked` → `Q`, `S` column। `Name`, `Ticket` → useless text, drop।
- শেষে `pd.concat(...)` দিয়ে নতুন column গুলো জোড়া দেই।

> 🔎 নতুন pandas-এ dummy column **True/False** দেখায় — মানে-ই **1/0**, model-এর কাছে একই।

---

## ৭. Model train + predict

```python
train_test_split(..., test_size=0.30, random_state=101)   # 70% train / 30% test
LogisticRegression(solver='lbfgs', max_iter=1000)          # model
```
- `solver='lbfgs'` = optimization পদ্ধতি। `max_iter=1000` = বেশি iteration দিলে ভালোভাবে converge করে।
- `.fit(X_train, y_train)` → শেখে; `.predict(X_test)` → 0/1 বলে।

---

## ৮. Evaluation — Confusion Matrix ও সব metric

Classification-এ MAE/RMSE **নয়** — এখানে confusion matrix + precision/recall/F1/accuracy।

### 🔢 Confusion Matrix — layout `[[TN, FP], [FN, TP]]`
তোমার lab result: `[[149, 14], [33, 71]]`

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | **TN = 149** ✅ | FP = 14 ❌ |
| **Actual 1** | FN = 33 ❌ | **TP = 71** ✅ |

- **TN** (True Negative): মরেছে, model-ও বলেছে মরেছে ✅
- **FP** (False Positive): মরেছে, কিন্তু model বলেছে বেঁচেছে ❌
- **FN** (False Negative): বেঁচেছে, কিন্তু model বলেছে মরেছে ❌
- **TP** (True Positive): বেঁচেছে, model-ও ঠিক বলেছে ✅

### 📊 Metric definitions (class 1 = "বেঁচেছে" ধরে)
| Metric | Formula | Lab value | মানে |
|---|---|---|---|
| **Accuracy** | (TP+TN)/মোট = (71+149)/267 | **0.82** | মোট 82% prediction ঠিক |
| **Precision** | TP/(TP+FP) = 71/85 | **0.84** | "বেঁচেছে" বলা গুলোর 84% সত্যিই বেঁচেছিল |
| **Recall** | TP/(TP+FN) = 71/104 | **0.68** | আসল survivor-দের 68% model ধরতে পেরেছে |
| **F1** | 2·P·R/(P+R) | **0.75** | precision ও recall-এর balance (harmonic mean) |
| **support** | — | 163 / 104 | প্রতি class-এ কতগুলো true sample |

> 🧠 **Precision vs Recall কখন?** ভুল "হ্যাঁ" ব্যয়বহুল হলে **Precision** জরুরি; আসল case মিস করা বিপজ্জনক হলে (যেমন রোগ ধরা) **Recall** জরুরি। দুটোর balance = **F1**।
> ⚠️ Data imbalanced হলে শুধু accuracy দেখো না — precision/recall দেখো।

---

## ৯. 💻 পূর্ণাঙ্গ কোড (output সহ)

```python
# ১. Import
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline

# ২. Data
train = pd.read_csv('titanic_train.csv')
train.head()

# ৩. EDA — missing + pattern
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')  # Age, Cabin missing
sns.countplot(x='Survived', hue='Sex', data=train)     # female বেশি বেঁচেছে
sns.countplot(x='Survived', hue='Pclass', data=train)  # class 3 death বেশি

# ৪. Cleaning — Age impute (Pclass অনুযায়ী)
def impute_age(cols):
    Age = cols.iloc[0]; Pclass = cols.iloc[1]
    if pd.isnull(Age):
        if Pclass == 1: return 37
        elif Pclass == 2: return 29
        else: return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)   # অনেক missing → বাদ
train.dropna(inplace=True)                   # বাকি NaN row বাদ

# ৫. Dummy variables (text → 0/1)
sex = pd.get_dummies(train['Sex'], drop_first=True)         # → male
embark = pd.get_dummies(train['Embarked'], drop_first=True) # → Q, S
train.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)

# ৬. X, y + split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    train.drop('Survived', axis=1), train['Survived'], test_size=0.30, random_state=101)

# ৭. Model train + predict
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs', max_iter=1000)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

# ৮. Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))   # accuracy 0.82
print(confusion_matrix(y_test, predictions))         # [[149 14] [33 71]]
```
> ⚠️ Sex/Embarked আগে `get_dummies` না করলে `could not convert string to float: 'male'` error দেবে।

---

## ১০. এক নজরে পুরো Day 6

1. **Logistic Regression** = category (0/1) predict → Supervised **Classification**।
2. **Sigmoid** দিয়ে probability (0–1), **threshold 0.5** দিয়ে 0/1।
3. **Cleaning** (impute Age, drop Cabin, dropna) + **dummy** (text → 0/1) এই day-এর special।
4. যাচাই: **confusion matrix** + accuracy/precision/recall/F1 (MAE/RMSE **নয়**)।
5. **Skeleton:** `read → clean → dummy → X,y → split → LogisticRegression → fit → predict → report`।

---

## ১১. ⭐ শেষ-মুহূর্তের Revision Card

| বিষয় | মনে রাখো |
|---|---|
| **Algorithm** | `LogisticRegression(solver='lbfgs', max_iter=1000)` |
| **Type / Task** | Supervised · **Classification** (Survived 0/1) |
| **Dataset / Target** | Titanic (891 row) · target = **Survived** |
| **Sigmoid** | যেকোনো সংখ্যা → 0–1 probability (S-curve); threshold 0.5 |
| **Cleaning** | Age impute (37/29/24) · Cabin drop · dropna |
| **Text → number** | `get_dummies(drop_first=True)` |
| **Metrics** | confusion matrix `[[149,14],[33,71]]` · accuracy **0.82** · P 0.84 · R 0.68 · F1 0.75 |
| **Confusion layout** | `[[TN, FP], [FN, TP]]` |

---

## ১২. ❓ Viva-তে সবচেয়ে বেশি যা আসে

| প্রশ্ন | ছোট উত্তর |
|---|---|
| Logistic Regression কী? | category (0/1) predict করার supervised classification algorithm |
| Linear vs Logistic? | Linear = number (line); Logistic = category (S-curve, probability) |
| Sigmoid function কী? | যেকোনো সংখ্যাকে 0–1 এ ম্যাপ করে (probability); S-আকৃতি |
| Threshold কী? | probability > 0.5 → class 1, নাহয় 0 |
| Confusion matrix-এর 4 ঘর? | TN, FP, FN, TP |
| Precision vs Recall? | Precision = TP/(TP+FP); Recall = TP/(TP+FN) |
| F1 কী? | precision ও recall-এর harmonic mean (balance) |
| get_dummies + drop_first কেন? | text → 0/1; drop_first দিয়ে dummy variable trap এড়ানো |
| Age missing কীভাবে fill? | impute_age — Pclass অনুযায়ী 37/29/24 |

---

> ✅ এই নোটটাই তোমার Day 6-এর সম্পূর্ণ note। হাতে-কলমে: `Practice_Day-06_Logistic-Regression.ipynb` · plot/output সহ: `README.md`।
> **Best of luck! 💪**
