#Generic Data Science Pipelins Import
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

#Imports for Numerical Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Read Data
csv_path = r"C:\Users\KI PC\OneDrive\Documents\GitHub\Bigmart-Sales-Data\train_v9rqX0R.csv"
sales_data = pd.read_csv(csv_path)

#Numerical Categories Preprocessing Pipeline
sales_data_num_attr = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year"]

#Input Scaling Columns
scaled_cols = [sales_data.columns.get_loc("Item_Weight"), 
               sales_data.columns.get_loc("Item_Visibility"),
               sales_data.columns.get_loc("Item_MRP"),
               sales_data.columns.get_loc("Outlet_Establishment_Year")
              ]

#Note that all column numbers are 0 based
class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X):
        return self
    def transform(self, X):
        imputer = SimpleImputer(missing_values=np.nan, strategy = "median")
        X = X.copy()
        #Imputing Data
        #print(X[:, self.cols].shape)

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

class ConvertFloat(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        X_float = np.empty(shape = (X.shape[0], X.shape[1]), dtype = float)
        for col in range(0, X.shape[1]):
            X_float[:, col] = X[:, col].astype(float)
        
        return X_float

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


#Categorical Columns Preprocessing Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#Tansforming Item_Identifier
class ItemIdTransform(BaseEstimator, TransformerMixin):
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

class ItemIdTransformV1(BaseEstimator, TransformerMixin):
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

class ItemFatContentTrnsfrm(BaseEstimator, TransformerMixin): #Not Changing Outlet Size
    def __init__(self, col_num, replcm_dict):
        self.col_num = col_num
        self.replcm_dict = replcm_dict
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #Tips from here:https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
        fat_content_trnsfrm = X.copy()[:, self.col_num]
        #print(type(X))
        for k, v in self.replcm_dict.items():
            fat_content_trnsfrm[fat_content_trnsfrm == k] = v
        return np.c_[X[:, :self.col_num], fat_content_trnsfrm, X[:, self.col_num + 1:]]


class ItemFatContentTrnsfrmV1(BaseEstimator, TransformerMixin):
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

class OutletSizeImpute(BaseEstimator, TransformerMixin):
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

class OutletSizeImputeV1(BaseEstimator, TransformerMixin):
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
    def __init__(self, cols_names, cols_idx):
        self.cols_names = cols_names
        self.cols_idx = cols_idx
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X): 
        from sklearn.preprocessing import OneHotEncoder
        X = X.copy()
        one_hot_encoder = OneHotEncoder()
        #one hot encoding data
        one_hot_encoded_data = one_hot_encoder.fit_transform(X[:, self.cols_idx]).toarray()
        
        #Get the column header for one-hot encoding; one hot encoder categories always gives list of list
        #Set it such that each column have the following format: Col_Name: Catg1, Col_Name: Catg2, and so on
        global one_hot_coding_cols_catgs
        one_hot_coding_cols_catgs = [] #reset the variable holding one hot encoding categories just in case
        for col_idx in range(0, len(self.cols_idx)):
            for catgs in one_hot_encoder.categories_[col_idx]:
                one_hot_coding_cols_catgs.append(self.cols_names[col_idx] + ": " + catgs)

        #Dropping columns that was encoded
        X = np.delete(X, self.cols_idx, axis = 1)

        #Append the one hot encoded data and return the data
        testX = np.c_[X, one_hot_encoded_data]
        return np.c_[X, one_hot_encoded_data]
        

#Replacement dictionary for Item_Fat_Content
item_fat_content_replcm_dict = {'Low Fat': 1, 'Regular': 0, 'LF': 1, 'reg': 0, 'low fat' : 1}

#Categorical Identifier

##Small will be encoded as 0, Medium as 1 and High as 2
outlet_size_ord_catgs = ['Small', 'Medium', 'High']
##Tier 1 will be encoded as 0, Tier 2 as 1 and Tier 3 as 2
outlet_location_type_ord_catgs = ['Tier 1', 'Tier 2', 'Tier 3']


one_hot_coding_cols_names = ["Item_Identifier", "Item_Type", "Outlet_Identifier", "Outlet_Type"]

one_hot_coding_cols_idx = [sales_data.columns.get_loc("Item_Identifier"), 
                       sales_data.columns.get_loc("Item_Type"),
                       sales_data.columns.get_loc("Outlet_Identifier"),
                       sales_data.columns.get_loc("Outlet_Type")
                      ]

one_hot_coding_cols_catgs = []

cat_pipeline = Pipeline([
    ('item_id_transform', ItemIdTransform(sales_data.columns.get_loc("Item_Identifier"))),
    ("item_fat_cont_transfrm", ItemFatContentTrnsfrm(sales_data.columns.get_loc("Item_Fat_Content"), item_fat_content_replcm_dict)),
    ("outlet_size_imp", OutletSizeImpute(sales_data.columns.get_loc("Outlet_Size"), "Small")),
    ("ordinal_encoding", OrdinalEncoderCustom([sales_data.columns.get_loc("Outlet_Size"), sales_data.columns.get_loc("Outlet_Location_Type")], [outlet_size_ord_catgs, outlet_location_type_ord_catgs])),
    ("one_hot_encoding", OneHotEncodingCustom(one_hot_coding_cols_names, one_hot_coding_cols_idx))
    ])

#mistake in outlet_location_Type. It should have been to something else for ordinal_encoding

#After data proprocessing, Columns are: 
# Numerical Columns: ITEM_WEIGHT, ITEM_VISIBILITY, ITEM_MRP, OUTLET_AGE, 
# Categorical Columns: ITEM_FAT_CONTENT_ENC, OUTLET_SIZE, ITEM_LOCATION_TYPE, ITEM_IDENTIFIER_CATGS, ITEM_TYPE_CATGS, OUTLET_IDENTIFIER_CATGS, OUTLET_TYPE_CATGs
preprocessing_full_pipeline = Pipeline([('num', num_pipeline),
                          ('cat', cat_pipeline),
                          ('convert_float', ConvertFloat())
                         ])

#Return four dataframe
def create_train_test_set(df, label_col_name):
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(sales_data, test_size=0.2, random_state=42)
    df_train = train_set.drop(label_col_name, axis = 1)
    df_train_labels = train_set[label_col_name].copy()
    df_test = test_set.drop(label_col_name, axis = 1)
    df_test_labels = test_set[label_col_name].copy()

    return df_train, df_train_labels, df_test, df_test_labels


#Create test and training data
sales_data_train, sales_data_train_labels, sales_data_test, sales_data_test_labels = create_train_test_set(sales_data, "Item_Outlet_Sales")

sales_data_prepared = preprocessing_full_pipeline.fit_transform(sales_data_train.values)



print('stop')

