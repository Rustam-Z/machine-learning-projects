```py
# To output full DF in Kaggle
pd.set_option("display.precision", 3)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_rows", 25) 
```
```py
# pivot_table() vs groupby(), the below lines are the same
pd.pivot_table(df, index=["a"], columns=["b"], values=["c"], aggfunc=np.sum)
df.groupby(['a','b'])['c'].sum()
```
```py
# Correlations between features
all_data_corr = all_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
all_data_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
all_data_corr.drop(all_data_corr.iloc[1::2].index, inplace=True)
all_data_corr_nd = all_data_corr.drop(all_data_corr[all_data_corr['Correlation Coefficient'] == 1.0].index)

corr = all_data_corr_nd['Correlation Coefficient'] > 0.1
all_data_corr_nd[corr]
```
```py
# Aggregate using one or more operations over the specified axis
# agg()-can be applied to multiple groups together
df.agg(['sum', 'min'])
df_all.groupby(['Sex', 'Pclass']).agg(lambda x:x.value_counts().index[0])['Embarked'] 

# Apply a function along an axis of the DataFrame
# apply()-cannot be applied to multiple groups together 
df.apply(np.sqrt)
df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
```
```py
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
```
```py
# Using KMeans clustering as preprocessing for MNIST dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)
pipeline_score = pipeline.score(X_test, y_test)
pipeline_score

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
grid_clf.best_params_
grid_clf.score(X_test, y_test)  
```