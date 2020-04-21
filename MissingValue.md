Count the missing value sum:
  df.isnull().sum()
  
if the size of data is hugh and the missing data is small -> we can drop the missing data
  df[df['cols']== np.nan] or df.dropna()
Like Jeremy said, there is no general solution for handling missing data since it depends on
its cause of missingness. I would summarize the adavantages and drawbacks as follow :

## Discarding each line containing a missing value is the simplest solution but there are two dangers. 
The first one is that you may lose all your data if there is a high rate of missingness. Secondly, you may create a bias
in your data : for instance, if you ask people their revenue and people with higher wages refuse more frequently to disclose
it, then you will restrict your dataset to people with modest revenue without being aware of it.
## Replacing by the mean is a good solution to get a filled dataset and being able to process it. Though you might have the same
problems as (1), and you also modify the variance. Then you have imputation which is a very broad subject. You will get a more coherent dataset but it is more complicated to 
preprocess your data. There are a diversity of methods, some of them rely on the EM algorithm to find your unknown values.
There are a lot of packages in R to do this (mice, missMDA, Amelia, etc…) but I don't know if there are such packages in 
Python.
## If you want to go fast I would advice you to replace it by the mean. If you want to go a bit further, I have worked with 
Professor Julie Josse about methods based on the PCA of the data :

* You first replace the missing data by the mean
* You perform the PCA on the centered data and you replace the missing values by their projection on the subspace you got
You iterate until convergence
* I'm not sure I'm really clear, but if you have a look on the internet you might find very interesting articles to handle 
this issue.
## function
* Imputing the missing values
```
def cat_imputation(column, value):
    houseprice.loc[houseprice[column].isnull(),column] = value
```
* Looking at categorical values
```
def cat_exploration(column):
    return houseprice[column].value_counts()
```

