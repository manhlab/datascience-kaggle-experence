# KFLOD + 
'''
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

param_grid = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(x_train,y_train):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = x_train[train_index],x_train[test_index]
     ytr,yvl = y_train[train_index],y_train[test_index]
     model = GridSearchCV(XGBRegressor(), param_grid, cv=10, scoring= 'neg_mean_squared_error',iid=True)
     model.fit(xtr, ytr)
     print (model.best_params_)
     pred=model.predict(xvl)
     print('accuracy_score',mean_squared_error(yvl,pred))
     i+=1
'''
