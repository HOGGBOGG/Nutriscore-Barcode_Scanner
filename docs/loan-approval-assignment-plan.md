# Loan Approval Prediction — 7-Day Assignment Completion Plan

A day-by-day learning + implementation plan to finish the **Loan-approval-prediction**
EDA and Visualization assignment inside a one-week deadline.

---

## 0. Read this first: what you are actually graded on

Your assignment doc has two pages, and they define **two deliverables**, not one:

| Deliverable | What the grader looks for |
|---|---|
| **EDA Report** | 10 named transformation steps, each visibly performed on the data |
| **Visualization Report** | 9 plot families, **each with a written inference** |

Two things follow from this, and they matter more than any code you write:

1. **The rubric is a checklist, not a theme.** Every bullet in the assignment doc should
   appear in your notebook as its own numbered, titled section, in the same order as the doc.
   Graders scan headings. A brilliant notebook with no visible section 6 loses the mark for
   section 6.
2. **This is not a modelling assignment.** The Problem Statement mentions an ML model, but the
   *Approach* and *Outcome* sections only ask for data types, cleaning, EDA and storytelling.
   Do **not** spend your week on scikit-learn model tuning. Build a model only on Day 7 as an
   optional bonus, after everything else is finished.

The single most common way to lose marks here is writing plots with no inference text under
them. Half your visualization marks live in the sentences, not the charts.

---

## 1. Your dataset (assumption — verify in 2 minutes)

This assignment is almost always issued with the Kaggle **`loan_approval_dataset.csv`**:
**4,269 rows × 13 columns**.

| # | Column | Type | Meaning |
|---|---|---|---|
| 1 | `loan_id` | int | Row identifier (drop it — it is not a feature) |
| 2 | `no_of_dependents` | int | Number of dependents (0–5) |
| 3 | `education` | object | ` Graduate` / ` Not Graduate` |
| 4 | `self_employed` | object | ` Yes` / ` No` |
| 5 | `income_annum` | int | Annual income |
| 6 | `loan_amount` | int | Requested loan amount |
| 7 | `loan_term` | int | Term in years (2–20) |
| 8 | `cibil_score` | int | Credit score (300–900) |
| 9 | `residential_assets_value` | int | Can be negative in this file — a real data quality bug |
| 10 | `commercial_assets_value` | int | |
| 11 | `luxury_assets_value` | int | |
| 12 | `bank_asset_value` | int | |
| 13 | `loan_status` | object | ` Approved` / ` Rejected` — **target** |

### Three quirks that will bite you on Day 1

1. **Every column name except `loan_id` has a leading space** (`' education'`, `' loan_status'`).
   `df['education']` raises a `KeyError` until you strip them. This is why the rubric line
   *"Rename column names to meaningful names"* exists — fix it there and the rest of the week is clean.
2. **Categorical values also carry a leading space** (`' Approved'`, not `'Approved'`). Strip those too.
3. **The file has zero nulls and zero duplicate rows.**

Quirk 3 is exactly what the Note on your second page is about:

> *"Please insert some data by your own if existing data is not suffice to demonstrate above
> requirements, so insert some data and apply above transformations."*

Translation: **you must deliberately inject missing values and duplicate rows yourself**, then
clean them. If you skip this you cannot demonstrate `dropna` / `fillna` / `drop_duplicates`, and
you lose three rubric bullets. Plan for it — it is Day 2's main job.

### Verify before you plan (run this first, it takes 2 minutes)

```python
import pandas as pd
df = pd.read_csv('loan_approval_dataset.csv')
print(df.shape)
print(repr(list(df.columns)))     # repr() reveals the hidden leading spaces
print(df.dtypes)
print(df.isnull().sum().sum(), 'total nulls')
print(df.duplicated().sum(), 'duplicate rows')
```

If your shape/columns differ from the table above, keep the plan and just swap column names —
the structure of every step below stays identical.

---

## 2. What to learn — checklist with a time budget

Learn **only** what the rubric needs. Roughly **21 hours ≈ 3 hours/day for 7 days**. Skip any
row you already know and bank the time.

| # | Topic | The exact skill you need | Time |
|---|---|---|---|
| 1 | Environment | Jupyter Notebook via Anaconda, **or** Google Colab (zero install — recommended if you are short on time) | 0.5 h |
| 2 | Python essentials | lists, dicts, functions, `for`, f-strings, slicing | 2 h |
| 3 | NumPy | `np.nan`, `np.random.seed`, `np.random.choice`, `np.percentile`, `np.log1p` | 1 h |
| 4 | pandas — load & inspect | `read_csv`, `head`, `tail`, `shape`, `info`, `dtypes`, `describe`, `columns`, `rename`, `memory_usage`, `nunique`, `value_counts` | 2 h |
| 5 | pandas — missing data | `isnull().sum()`, `fillna(mean/median/mode)`, `dropna(subset=, how=, axis=)` and when each is right | 2 h |
| 6 | pandas — duplicates | `duplicated()`, `drop_duplicates(keep=, subset=)` | 0.5 h |
| 7 | Encoding | `map`/`replace`, `LabelEncoder`, `pd.get_dummies`, **ordinal vs nominal** — know which to use and say why | 2 h |
| 8 | Outliers | IQR rule (Q1 − 1.5·IQR, Q3 + 1.5·IQR), z-score, reading a boxplot, **capping vs dropping** | 2 h |
| 9 | Binning | `pd.cut` (fixed edges) vs `pd.qcut` (equal counts), `labels=` | 1 h |
| 10 | Normalization | `MinMaxScaler`, `StandardScaler`; never scale the target or an ID | 1.5 h |
| 11 | matplotlib | `plt.subplots`, title/xlabel/ylabel, `legend`, `figsize`, `tight_layout`, `savefig` | 2 h |
| 12 | seaborn | `countplot`, `histplot`, `boxplot`, `scatterplot`, `heatmap`, and the `hue=` argument | 2 h |
| 13 | pandas `.plot` | `kind='area'`, `kind='pie'`, `kind='hexbin'` — seaborn has no direct equivalent for these three | 1 h |
| 14 | Correlation | `df.corr(numeric_only=True)`, reading a correlation heatmap, correlation ≠ causation | 1 h |
| 15 | Storytelling | How to write a one-line inference (formula in §6) | 1 h |

**Where to learn it, fastest first:**
- **Kaggle Learn** → *Pandas* and *Data Visualization* micro-courses. Free, browser-based, ~4 h each. This is the highest-value use of your learning time.
- **pandas official docs** → *10 minutes to pandas*, then the *User Guide* pages for Missing Data, Duplicates, Categorical.
- **seaborn tutorial gallery** → copy a chart that looks like what you need, swap in your columns.
- **matplotlib "Pyplot tutorial"** → only the axes/labels/savefig basics; do not go deeper.

Rule for the week: **learn a topic in the morning block, apply it to the assignment in the
afternoon block, the same day.** Do not batch all learning into Days 1–3; you will forget it.

---

## 3. The day-by-day plan

Each day has a *Learn* block, a *Build* block, and a **Done when** test. If a day's "Done when"
fails, cut the stretch items — never carry a broken section forward.

### Day 1 — Setup, first look, and section 1–4 of the EDA report *(~3 h)*

**Learn:** rows 1–4 of the skills table (environment, Python essentials, NumPy basics, pandas inspection).

**Build:**
1. Create the notebook `01_EDA_Report.ipynb`. First cell: imports + `pd.set_option('display.max_columns', None)`.
2. Load the CSV, then **immediately** `df_raw = df.copy()` — you will want the untouched data later.
3. **Rubric 1** — display `df.head()` and `df.tail()`.
4. **Rubric 2** — rename columns: strip whitespace, then give meaningful names.
5. **Rubric 3** — `df.shape`, printed as a sentence: *"The dataset has 4269 rows and 13 columns."*
6. **Rubric 4** — a single summary table of column name, dtype, non-null count, memory size.
7. Also strip the whitespace out of the categorical *values* while you are here.

**Done when:** `df['loan_status'].value_counts()` runs without a `KeyError` and returns
`Approved` / `Rejected` with no leading spaces.

---

### Day 2 — Nulls and duplicates, including the deliberate corruption *(~3 h)*

**Learn:** rows 5–6 (missing data, duplicates).

**Build:**
1. **Rubric 5** — `df.isnull().sum()`. On the clean file this is all zeros. Say so in a markdown
   cell, and quote the assignment's Note as your justification for the next step.
2. Add a clearly titled section: **"Deliberate injection of missing values and duplicates
   (as permitted by the assignment note)."** Set `np.random.seed(42)` so your run is reproducible,
   then punch `np.nan` into roughly 3–5% of the cells of about four columns — pick a mix:
   - a numeric column that is roughly symmetric → you will impute with **mean**
   - a numeric column that is skewed or has outliers → impute with **median**
   - a categorical column → impute with **mode**
   - a low-importance column → you will **`dropna`** these rows, to show that approach too

   Then `pd.concat([df, df.sample(20, random_state=42)])` to create duplicate rows.
3. Re-run `isnull().sum()` and `duplicated().sum()` to prove the corruption landed.
4. **Rubric 6** — clean it, and **use all three approaches** the rubric names, one per column, with
   a markdown line under each saying *why* that choice fits that column (mean is pulled by outliers;
   median is robust; mode is the only option for categories; dropping is right when the column is
   unimportant and the loss is small).
5. **Rubric 7** — `df.duplicated().sum()` → `df.drop_duplicates(inplace=True)` → confirm the shape
   dropped by exactly the number you injected.

**Done when:** `df.isnull().sum().sum() == 0`, `df.duplicated().sum() == 0`, and every imputation
choice has a sentence of justification. **This is the day that carries the most rubric weight — do not skip it.**

---

### Day 3 — Encoding, outliers, binning, normalization *(~3.5 h — the heaviest day)*

**Learn:** rows 7–10 (encoding, outliers, binning, scaling).

**Build:**
1. **Rubric 8 — Encode categorical data.** Three columns need it, and the point is to show you know
   *which* method fits *which* variable:
   - `loan_status` (target, binary) → `map({'Approved': 1, 'Rejected': 0})`
   - `education` (arguably ordinal) → `map({'Not Graduate': 0, 'Graduate': 1})`
   - `self_employed` (nominal binary) → `map` or `get_dummies(drop_first=True)`

   Write one line explaining ordinal vs nominal, and why one-hot is the safe default for a nominal
   variable with 3+ levels. Keep the original text columns as `*_original` so your plots can still
   use readable labels.
2. **Rubric 9 — Handle outliers.**
   - Draw boxplots of the numeric columns *before* treatment (these double as your Day 5 box plots).
   - Compute IQR bounds in a small loop, print how many outliers each column has.
   - Deal with the negative `residential_assets_value` values explicitly — a negative asset value is
     a genuine data error, and catching it is exactly the judgement the marker wants to see.
   - **Cap** (winsorize) rather than delete, and justify it: with only ~4,000 rows, dropping rows
     throws away real applicants; capping keeps them and limits their leverage. Show the boxplot
     again after capping.
3. **Rubric 10 — Binning / Normalization.** Do both, they are two separate skills:
   - **Binning:** `pd.cut` on `cibil_score` → `Poor / Fair / Good / Excellent`; `pd.qcut` on
     `income_annum` → four equal-sized income quartiles. These new columns make Day 4–5 plots far
     more readable, which is the real reason to bin.
   - **Normalization:** `MinMaxScaler` on the continuous numeric columns into new `*_scaled`
     columns. Explicitly exclude `loan_id` and the target, and say in a markdown cell why scaling
     an ID or a label is meaningless.
4. Save the cleaned frame: `df.to_csv('loan_approval_cleaned.csv', index=False)`. Day 4 starts from this file.

**Done when:** `df.describe()` shows scaled columns in [0, 1], no negative assets remain, and your
binned columns show sensible `value_counts()`.

---

### Day 4 — Visualization report, part 1 *(~3 h)*

**Learn:** rows 11–13 (matplotlib, seaborn, pandas `.plot`).

**Build** — start `02_Visualization_Report.ipynb`, load the cleaned CSV, and produce the first five
plot families from §7's pairing table: **basic/line, bar, histogram, box, area**.

Set a house style once at the top (`sns.set_theme(style='whitegrid')`, a fixed `figsize`) so all
your charts look like one report. Every chart gets a title, both axis labels, and — immediately
below it — a markdown cell headed **Inference** (see §6).

**Done when:** five plot families exist, each with an inference paragraph, and every chart is
readable at 100% zoom without squinting.

---

### Day 5 — Visualization report, part 2 + correlation *(~3 h)*

**Learn:** row 14 (correlation) plus anything from Day 4 you had to look up twice.

**Build:** the remaining four families — **scatter, hexagonal bin, pie, heatmap**.
- The heatmap is your headline chart. Use `df.corr(numeric_only=True)` with
  `annot=True, cmap='coolwarm', fmt='.2f'`, and read `cibil_score` vs `loan_status` off it directly.
- Add a second heatmap from `pd.crosstab` (e.g. cibil band × loan status) if you want an easy extra mark.
- `savefig` every chart into an `images/` folder as you go — you will need them for the write-up
  and they take 30 seconds now versus an hour on Day 6.

**Done when:** all nine plot families from the assignment doc are ticked off, each with an inference.

---

### Day 6 — Storytelling, polish, and the write-up *(~3 h)*

**Learn:** row 15 (how to write inferences) — then apply it to everything.

**Build:**
1. Write the **Executive Summary** at the *top* of the notebook (write it last, place it first).
   Five to seven bullets, in plain business English, each naming the number that backs it.
2. Write the **Story** section at the bottom: the narrative arc *"who gets a loan at ABC bank, and
   what actually decides it."* Order your evidence — lead with the strongest driver, end with the
   variables that turned out not to matter. A finding that something *doesn't* matter is a real
   finding; say it confidently.
3. Add a short **Recommendations** block (3–4 bullets: what the bank should do, what data is missing,
   what you would model next). This is what "ring side view of making sense with data" in the
   Outcome section is asking for.
4. Restart the kernel and **Run All**. Fix anything that breaks out of order. A notebook that only
   runs in the order you happened to click cells is a notebook that fails on the grader's machine.
5. Clean up: delete dead cells, number every section to match the rubric, spell-check the markdown.

**Done when:** Kernel → Restart & Run All completes top to bottom with zero errors.

---

### Day 7 — Buffer, export, submit *(~2 h + slack)*

This day is deliberately mostly empty. Something will slip earlier in the week; this is where it lands.

1. Re-check your notebook against §9's submission checklist, bullet by bullet, against the
   original assignment doc — not against this plan.
2. Export: **File → Download as → HTML** (and PDF if your course wants it). Keep the `.ipynb` too.
3. Push to GitHub with a README that states the problem, the dataset, how to run it, and your top
   three findings.
4. **Only if everything above is done:** the bonus model — `train_test_split`, a
   `DecisionTreeClassifier` or `LogisticRegression`, accuracy + confusion matrix, in a clearly
   marked "Bonus" section. Ten lines, no tuning. It costs you 30 minutes and directly answers the
   Problem Statement's mention of an ML system.

---

## 4. Notebook skeleton (mirror the rubric exactly)

```
00. Executive Summary                      <- written last, placed first
01. Imports & Setup
02. Load the dataset — head(5) & tail(5)             [rubric 1]
03. Rename columns to meaningful names               [rubric 2]
04. Dataset dimensions — rows & columns              [rubric 3]
05. Column names, data types & size                  [rubric 4]
06. Null value report                                [rubric 5]
07. Deliberate injection of nulls + duplicates       [assignment Note]
08. Handling nulls — dropna / mean / median / mode   [rubric 6]
09. Removing duplicate records                       [rubric 7]
10. Encoding categorical data                        [rubric 8]
11. Outlier detection & treatment                    [rubric 9]
12. Binning & Normalization                          [rubric 10]
13. Visualisations
    13.1 Basic (line) plot        13.6 Scatter plot
    13.2 Bar plot                 13.7 Hexagonal bin plot
    13.3 Histogram                13.8 Pie plot
    13.4 Box plot                 13.9 Heatmap
    13.5 Area plot
14. Story: what decides a loan at ABC bank
15. Recommendations & next steps
16. (Bonus) Baseline model
```

Use markdown headings (`##`) for these, not comments. The heading *is* the evidence that you did the step.

---

## 5. Code cheat sheet — one line per rubric bullet

Keep this open while you work. These are the calls you need; the learning in §2 is about
understanding *why* each one is the right choice.

```python
# --- 1. Load + top/bottom 5 -------------------------------------------------
df = pd.read_csv('loan_approval_dataset.csv');  df.head();  df.tail()

# --- 2. Rename columns ------------------------------------------------------
df.columns = df.columns.str.strip()                     # kill the leading spaces
df = df.rename(columns={'no_of_dependents': 'dependents',
                        'income_annum': 'annual_income',
                        'cibil_score': 'credit_score'})
for c in df.select_dtypes('object'):                    # strip the VALUES too
    df[c] = df[c].str.strip()

# --- 3. Rows & columns ------------------------------------------------------
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# --- 4. Name, dtype, size ---------------------------------------------------
pd.DataFrame({'dtype': df.dtypes,
              'non_null': df.notnull().sum(),
              'nunique': df.nunique(),
              'bytes': df.memory_usage(deep=True).drop('Index')})

# --- 5. Null report ---------------------------------------------------------
df.isnull().sum()[lambda s: s > 0]

# --- Note: inject nulls + duplicates so 6 & 7 have something to clean -------
np.random.seed(42)
for col, frac in [('annual_income', .04), ('loan_amount', .04),
                  ('self_employed', .03), ('loan_term', .02)]:
    idx = df.sample(frac=frac, random_state=42).index
    df.loc[idx, col] = np.nan
df = pd.concat([df, df.sample(20, random_state=42)], ignore_index=True)

# --- 6. Handle nulls — one technique per column, each justified -------------
df['annual_income'] = df['annual_income'].fillna(df['annual_income'].mean())    # symmetric
df['loan_amount']   = df['loan_amount'].fillna(df['loan_amount'].median())      # skewed
df['self_employed'] = df['self_employed'].fillna(df['self_employed'].mode()[0]) # categorical
df = df.dropna(subset=['loan_term'])                                            # few rows, low cost

# --- 7. Duplicates ----------------------------------------------------------
before = df.shape[0]; df = df.drop_duplicates(); print(before - df.shape[0], 'removed')

# --- 8. Encoding ------------------------------------------------------------
df['loan_status_enc'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
df['education_enc']   = df['education'].map({'Not Graduate': 0, 'Graduate': 1})
df = pd.get_dummies(df, columns=['self_employed'], drop_first=True)

# --- 9. Outliers — IQR detect, then cap -------------------------------------
def iqr_bounds(s):
    q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ['annual_income', 'loan_amount', 'residential_assets_value']:
    lo, hi = iqr_bounds(df[col])
    print(col, 'outliers:', ((df[col] < lo) | (df[col] > hi)).sum())
    df[col] = df[col].clip(lo, hi)                      # cap, don't delete

# --- 10. Binning & normalization -------------------------------------------
df['credit_band'] = pd.cut(df['credit_score'], bins=[299, 500, 650, 750, 900],
                           labels=['Poor', 'Fair', 'Good', 'Excellent'])
df['income_quartile'] = pd.qcut(df['annual_income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

from sklearn.preprocessing import MinMaxScaler
num = ['annual_income', 'loan_amount', 'credit_score', 'loan_term']
df[[c + '_scaled' for c in num]] = MinMaxScaler().fit_transform(df[num])
```

---

## 6. How to write an inference (this is where the marks are)

Under every chart, write **two to four sentences** using this shape:

> **What** the chart shows (the pattern, named) →
> **Number** that proves it (a value, a ratio, a gap) →
> **So what** for ABC bank (the decision it should change).

Weak: *"The heatmap shows correlations between the variables."*

Strong: *"Credit score is by far the strongest correlate of approval (r ≈ 0.77), while annual income
is close to zero. Applicants below roughly 550 are rejected almost without exception, regardless of
how much they earn. For ABC bank this means credit score should be the first gate in the approval
workflow, and a high income should not be treated as compensating for a poor score."*

The second version does three things the first doesn't: names the pattern, cites the number,
and states the business consequence.

**Findings you will probably land on — verify each against your own output before you write it down:**
- `cibil_score` dominates approval; the split is sharp, not gradual, around the 500–600 region.
- `income_annum` and `loan_amount` are very strongly correlated (loan size tracks income), so they
  carry overlapping information.
- `education` and `self_employed` barely move the approval rate — a genuinely useful negative result,
  and worth stating plainly.
- The four asset columns move together and are largely a proxy for income.
- `loan_term` has a milder, non-obvious relationship — look at it before you claim a direction.

Do not copy these as conclusions. Run the numbers, and if your data disagrees, **your data wins**.

---

## 7. Which columns to use for each required plot

| # | Plot type | Use these columns | What the inference should be about |
|---|---|---|---|
| 1 | **Basic / line** | Mean `loan_amount` by `loan_term` | Whether longer terms come with bigger loans |
| 2 | **Bar** | `education` / `self_employed` × approval rate | Whether profile categories shift approval at all |
| 3 | **Histogram** | `cibil_score`, `annual_income`, `loan_amount` | Shape, skew, and where the mass of applicants sits |
| 4 | **Box** | `cibil_score` grouped by `loan_status` | The approval threshold, plus outlier visibility |
| 5 | **Area** | The four asset columns stacked across `income_quartile` | How the asset mix changes as income rises |
| 6 | **Scatter** | `annual_income` vs `loan_amount`, `hue='loan_status'` | The income–loan relationship, and that status doesn't separate on these two axes |
| 7 | **Hexbin** | Same two columns as the scatter | Where applicants actually concentrate — the point is that hexbin reveals density the scatter hides through overplotting; say that |
| 8 | **Pie** | `loan_status` share; asset-type share | Class balance — check this before trusting any accuracy number later |
| 9 | **Heatmap** | `df.corr(numeric_only=True)` | The headline: what predicts approval and what is redundant |

A hexbin next to a scatter of the *same two columns* is a free extra insight: you get to write about
overplotting and density, which is exactly the kind of observation that separates a good report from
a checklist.

---

## 8. Mistakes that quietly cost marks

1. **Plots with no inference text.** The single biggest loss. Nine charts, nine written inferences.
2. **Not injecting nulls/duplicates**, then claiming you "handled" them. The rubric explicitly asks
   for `dropna`/`fillna` *and* duplicate removal; a clean file gives you nothing to show.
3. **Using one imputation method everywhere.** The rubric says *"different approaches like delete,
   imputing using mean/mode/median."* Use all of them, one per column, each justified.
4. **Deleting outliers by default.** Explain your choice. Capping is usually right here; either way,
   the justification is the mark.
5. **Scaling the ID or the target.** Meaningless, and a grader will spot it instantly.
6. **A notebook that won't Restart & Run All.** Test this before submitting, not after.
7. **Charts with no titles or axis labels.** Cheap marks, routinely thrown away.
8. **Building a model instead of doing EDA.** Read the Approach section again — it lists data types,
   cleaning, EDA and storytelling. Not model tuning.
9. **Leaving the leading spaces in column names**, then fighting `KeyError`s all week. Fix it on Day 1.
10. **Starting the visualizations on Day 6.** They are half the assignment. They start Day 4.

---

## 9. Submission checklist

Tick these against the **original assignment doc**, not against this plan:

- [ ] Top 5 and bottom 5 rows displayed
- [ ] Columns renamed to meaningful names
- [ ] Row and column count stated
- [ ] Column name, dtype and size table
- [ ] Null-value report shown
- [ ] Nulls handled with **delete + mean + median + mode**, each justified
- [ ] Duplicates found and removed, with counts before/after
- [ ] Categorical data encoded, ordinal vs nominal explained
- [ ] Outliers detected (IQR) and treated, with before/after boxplots
- [ ] Binning **and** normalization both performed
- [ ] All 9 plot types present
- [ ] Every plot has a title, axis labels, and a written inference
- [ ] Executive summary at the top, story + recommendations at the bottom
- [ ] Restart & Run All passes with zero errors
- [ ] Exported to HTML/PDF, pushed to GitHub with a README

---

## 10. If you finish early

In priority order:
1. **Baseline model** — logistic regression + decision tree, accuracy and confusion matrix, ten lines.
   Directly addresses the Problem Statement's ML framing.
2. **Feature importance** from the tree — it will almost certainly confirm your credit-score finding,
   which makes for a strong closing slide.
3. **A bivariate deep-dive** on the sharpest relationship you found (approval rate by credit band,
   as a clean bar chart with the rate printed on each bar).
4. **Polish the README** into something you would put on a CV. This assignment is a genuinely
   presentable portfolio piece if the story section is written well.

---

### The one-sentence version

Fix the column names on Day 1, spend Day 2 deliberately breaking and then repairing the data,
get all ten cleaning steps done by end of Day 3, spend Days 4–5 making nine charts that each come
with a written inference, and use Day 6 to turn those inferences into a story — leaving Day 7 as
the buffer you will almost certainly need.
