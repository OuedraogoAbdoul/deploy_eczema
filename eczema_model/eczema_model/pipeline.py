from regression_model.config import config
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

from regression_model.get_data import read_data
from regression_model.preprosessing import data_preprocessors as pp

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor



transformer = FeatureUnion(
  transformer_list=[
    ('features', SimpleImputer(strategy='mean')),
    ('indicators', MissingIndicator())])


btc_price_pipeline = make_pipeline(transformer, StandardScaler(), DecisionTreeRegressor(random_state=0))


#  Pipeline(
#     [
#         (
#             "fill missing data",
#             pp.MissingDataHandler(),
#         ), 
#         # (
#         #     "select features",
#         #     pp.BestFeaturesSelected(variables=config.DROPFEATURES),
#         # ),

#         (
#             "StandardScaler",StandardScaler(),
#         ),

#         # (
#         #     "Linear_model", Lasso(alpha=0.005, random_state=0)
#         # ),
#         (
#             "Linear_model", DecisionTreeRegressor(random_state=0)
#         )
#     ]
# )