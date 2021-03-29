import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


csv_path = r"C:\Users\KI PC\OneDrive\Documents\GitHub\Bigmart-Sales-Data\train_v9rqX0R.csv"
sales_data = pd.read_csv(csv_path)

#Numerical Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sales_data_num_attr = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]

class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X):
        return self
    def transform(self, X):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy = "median")
        X = X.copy()
        #Imputing Data
        print(X[:, self.cols].shape)

        if len(self.cols) == 1:
            #create a duplicate of the same column so we get 2d array so we can use simple imputer
            #at the end the duplicate column will be discarded anyways
            col_data_to_impute = np.concatenate([X[:, self.cols],X[:, self.cols]], axis = 1)
        else:
            col_data_to_impute = X[:, self.cols]

        imputed_data = imputer.fit_transform(col_data_to_impute) #input needs to be 2d array
        for c in range(0, len(self.cols)):
            X[:, self.cols[c]] = imputed_data[:, c]
        
        return X

class StandardScalerCustom(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X):
        return self
    def transform(self, X):
        from sklearn.preprocessing import StandardScaler
        standard_scaler = StandardScaler()
        #Scaling Data
        if len(self.cols) == 1:
            #create a duplicate of the same column so we get 2d array so we can use simple imputer
            #at the end the duplicate column will be discarded anyways
            col_data_to_impute = np.concatenate([X[:, self.cols],X[:, self.cols]], axis = 1)
        else:
            col_data_to_impute = X[:, self.cols]

        scaled_data = standard_scaler.fit_transform(col_data_to_impute) 
        for c in range(0, len(self.cols)):
            X[:, self.cols[c]] = scaled_data[:, c]
        
        return X

class OutletAgeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, col_num, current_year):
        self.col_num = col_num
        self.current_year = current_year
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        outlet_age = self.current_year - X[:, self.col_num]
        return np.c_[X[:, :self.col_num], outlet_age, X[:, self.col_num + 1:]]

#Input Scaling Columns
scaled_cols = [sales_data.columns.get_loc("Item_Weight"), 
               sales_data.columns.get_loc("Item_Visibility"),
               sales_data.columns.get_loc("Item_MRP"),
               sales_data.columns.get_loc("Outlet_Establishment_Year")
              ]

num_pipeline = Pipeline([
    ('imputer', SimpleImputerCustom([sales_data.columns.get_loc("Item_Weight")])),
    ('outlet_age_adder', OutletAgeAdder(sales_data.columns.get_loc("Outlet_Establishment_Year"), 2013)),
    ('std_scaler', StandardScalerCustom(scaled_cols))
    ])


# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(missing_values=np.nan, strategy="median")),
#     ('outlet_age_adder', OutletAgeAdder(2013)),
#     ('std_scaler', StandardScaler()),
#     ])


#Categorical Transformers
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#Tansforming Item_Identifier
class ItemIdTransformV1(BaseEstimator, TransformerMixin):
    def __init__(self, col_num):
        self.col_num = col_num

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        id_transform = np.zeros(shape = (X.shape[0], 1)).astype(str)
        X = X.copy().astype(str)
        for row in range(0, X.shape[0]):
            id_transform[row] = X[row, self.col_num][0:2]
        return np.c_[X[:, :self.col_num], id_transform, X[:, self.col_num + 1:]]

class ItemIdTransformV2(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        id_transform = np.zeros(shape = (X.shape[0], 1)).astype(str)
        X = X.copy().astype(str)
        for row in range(0, X.shape[0]):
            id_transform[row] = X[row][0:2]
        return id_transform

class ItemFatContentTrnsfrmV1(BaseEstimator, TransformerMixin): #Not Changing Outlet Size
    def __init__(self, col_num, replcm_dict):
        self.col_num = col_num
        self.replcm_dict = replcm_dict
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #Tips from here:https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
        fat_content_trnsfrm = X.copy()[:, self.col_num]
        print(type(X))
        for k, v in self.replcm_dict.items():
            fat_content_trnsfrm[fat_content_trnsfrm == k] = v
        return np.c_[X[:, :self.col_num], fat_content_trnsfrm.astype(int), X[:, self.col_num + 1:]]


class ItemFatContentTrnsfrmV2(BaseEstimator, TransformerMixin):
    def __init__(self, replcm_dict):
        self.replcm_dict = replcm_dict
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #Tips from here:https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
        fat_content_trnsfrm = X.copy()
        for k, v in self.replcm_dict.items():
            fat_content_trnsfrm[fat_content_trnsfrm == k] = v
        return np.c_[fat_content_trnsfrm.astype(int)]

class OutletSizeImputeV1(BaseEstimator, TransformerMixin):
    def __init__(self, col_num, replcm_val):
        self.col_num = col_num
        self.replcm_val = replcm_val
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #Solution Link: https://www.codegrepper.com/code-examples/python/numpy+ndarray+fillna
        #Could not use the solution above because of mixed type in ndarray
        #Convert all to Str
        outlet_size_trnsfrm = X.copy()[:, self.col_num].astype(str)
        outlet_size_trnsfrm[outlet_size_trnsfrm == 'nan'] = self.replcm_val
        return np.c_[X[:, :self.col_num], outlet_size_trnsfrm, X[:, self.col_num + 1:]]

class OutletSizeImputeV2(BaseEstimator, TransformerMixin):
    def __init__(self, replcm_val):
        self.replcm_val = replcm_val
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #Solution Link: https://www.codegrepper.com/code-examples/python/numpy+ndarray+fillna
        #Could not use the solution above because of mixed type in ndarray

        #Convert all to Str
        outlet_size_trnsfrm = X.copy().astype(str)
        outlet_size_trnsfrm[outlet_size_trnsfrm == 'nan'] = self.replcm_val
        return np.c_[outlet_size_trnsfrm]

class OrdinalEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, cols, categories):
        self.cols = cols
        self.categories = categories

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        from sklearn.preprocessing import OrdinalEncoder
        X = X.copy().astype(object) #Need object dtype to use ordinal encoding
        ordinal_encoder = OrdinalEncoder(categories = self.categories)
        cols_ord_encoded = ordinal_encoder.fit_transform(X[:, self.cols])
        #Replace existing columns which are to be encoded with encoded values
        for c in range(0, len(self.cols)):
            # print(c)
            # print(X[:, self.cols[c]])
            # print(cols_ord_encoded[:, c])
            # print(X[0])
            X[:, self.cols[c]] = cols_ord_encoded[:, c] 
            # print(X[0])
        return X

class OneHotEncodingCustom(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X): 
        from sklearn.preprocessing import OneHotEncoder
        X = X.copy()
        one_hot_encoder = OneHotEncoder()
        #one hot encoding data
        one_hot_encoded_data = one_hot_encoder.fit_transform(X[:, self.cols]).toarray()

        #Dropping columns that was encoded
        X = np.delete(X, self.cols, axis = 1)

        #Append the one hot encoded data and return the data
        testX = np.c_[X, one_hot_encoded_data]
        return np.c_[X, one_hot_encoded_data]
        

#column Numbers
item_id_column = sales_data.columns.get_loc("Item_Identifier")
item_fat_content_column = sales_data.columns.get_loc("Item_Fat_Content")
outlet_size_column = sales_data.columns.get_loc("Outlet_Size")
outlet_loc_type_column = sales_data.columns.get_loc("Outlet_Location_Type")


item_fat_content_replcm_dict = {'Low Fat': 1, 'Regular': 0, 'LF': 1, 'reg': 0, 'low fat' : 1}
#Categorical Identifier
#Small will be encoded as 0, Medium as 1 and High as 2
outlet_size_ord_catgs = ['Small', 'Medium', 'High']
#Tier 1 will be encoded as 0, Tier 2 as 1 and Tier 3 as 2
outlet_location_type_ord_catgs = ['Tier 1', 'Tier 2', 'Tier 3']

one_hot_coding_cols = [sales_data.columns.get_loc("Item_Identifier"), 
                       sales_data.columns.get_loc("Item_Type"),
                       sales_data.columns.get_loc("Outlet_Identifier"),
                       sales_data.columns.get_loc("Outlet_Location_Type"),
                       sales_data.columns.get_loc("Outlet_Type")
                      ]


cat_pipeline = Pipeline([
    ('item_id_transform', ItemIdTransformV1(item_id_column)),
    ("item_fat_cont_transfrm", ItemFatContentTrnsfrmV1(item_fat_content_column, item_fat_content_replcm_dict)),
    ("outlet_size_imp", OutletSizeImputeV1(outlet_size_column, "Small")),
    ("ordinal_encoding", OrdinalEncoderCustom([outlet_size_column, outlet_loc_type_column], [outlet_size_ord_catgs, outlet_location_type_ord_catgs])),
    ("one_hot_encoding", OneHotEncodingCustom(one_hot_coding_cols))
    ])

full_pipeline = Pipeline([('num', num_pipeline),
                          ('cat', cat_pipeline)
                         ])

#Columns are: 
# Numerical Columns: ITEM_WEIGHT, ITEM_VISIBILITY, ITEM_MRP, OUTLET_AGE, 
# Categorical Columns: ITEM_FAT_CONTENT_ENC, OUTLET_SIZE, ITEM_LOCATION_TYPE, ITEM_IDENTIFIER_CATGS, ITEM_TYPE_CATGS, OUTLET_IDENTIFIER_CATGS, OUTLET_TYPE_CATGs
#
# had to buld a pipeline for categorical values unlike 


sales_data_train = sales_data.copy().drop("Item_Outlet_Sales", axis=1)
#sales_data_prepared = cat_pipeline.fit_transform(sales_data_train.values)
sales_data_prepared = full_pipeline.fit_transform(sales_data_train.values)

print('stop')

