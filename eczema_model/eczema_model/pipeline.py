from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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