#----------------------------------------------------------------------------------
# 4 ways to select features:
# 1. Univariate Selection
# 2. Recursive Feature Elimination
# 3. Principle Component Analysis
# 4. Feature Importance
#----------------------------------------------------------------------------------
import pandas as pd
import numpy as np

data = pd.read_csv("file_path")
list(data)
y = data['y']
x = data.drop(['tags_to_be_dropped'], axis=1)
#----------------------------------------------------------------------------------
# Univariate Selection
# - works by selecting the best features based on univariate statistical tests
# - chi -> input x must be non-negative
# - mutual information -> discrete target variable
#----------------------------------------------------------------------------------
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif # ANOVA
from sklearn.feature_selection import mutual_info_classif   # mutual information

SelectKBest(score_func=mutual_info_classif, k=20).fit_transform(x, y)

#----------------------------------------------------------------------------------
# Recursive Feature Elimination
#----------------------------------------------------------------------------------
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=1, step=1)   # step: the number of features to remove at each iteration
#selector = RFECV(estimator, min_features_to_select=20,step=1, cv=3)
selector = selector.fit(x, y)
x.columns[selector.support_]    # list of selected features

selector.ranking_
ranked_features = pd.DataFrame(columns=['rank', 'feature'])
for i, feature in enumerate(x.columns):
    ranked_features = ranked_features.append(pd.DataFrame([[selector.ranking_[i], feature]], columns=['rank', 'feature']))
ranked_features
ranked_features.to_csv("./feature_selection/feature_ranking_by_rfe_vif41.csv")
#----------------------------------------------------------------------------------
# Principal Component Analysis
#----------------------------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(x)

print(pca.explained_variance_ratio_)    # Percentage of variance explained by each of the selected components.
print(pca.singular_values_)             # the 2-norms of the n_components variables in the lower-dimensional space
components = pd.DataFrame(pca.components_,columns=x.columns)    # how components are linearly related with each feature
print(components)
components.to_csv("./feature_selection/pca_components_relations_scaled_vif10.csv")

x_pca = pca.transform(x)
#----------------------------------------------------------------------------------
# Feature Importance
#----------------------------------------------------------------------------------
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)

# Set a minimum threshold of 0.25
sfm = SelectFromModel(ridge)
sfm.fit(x, y)

sfm.transform(x).shape[1]   # the number of selected features
x.columns[sfm.get_support()]    # selected features

#------------------------------------------- Tree-based feature selection
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=10)
rf.fit(x,y)
rf.feature_importances_
x.columns[np.where(rf.feature_importances_>0)]

rf_imp = pd.DataFrame(rf.feature_importances_)
rf_imp = pd.concat([rf_imp, pd.DataFrame(x.columns)], axis=1)
rf_imp.to_csv("./feature_selection/rf_feature_importance_scaled_smoothed.csv")

sfm = SelectFromModel(rf, prefit=True)
sfm.transform(x).shape[1]   # the number of selected features
x.columns[sfm.get_support()]    # selected features
#----------------------------------------------------------------------------------
# https://github.com/WillKoehrsen/feature-selector/blob/master/Feature%20Selector%20Usage.ipynb
# 1. Features with a high percentage of missing values
# 2. Collinear (highly correlated) features
# 3. Features with zero importance in a tree-based model
# 4. Features with low importance
# 5. Features with a single unique value
#----------------------------------------------------------------------------------
# "https://github.com/WillKoehrsen/feature-selector.git"
# from feature_selector import FeatureSelector as fs

