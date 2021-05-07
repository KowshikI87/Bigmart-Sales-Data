'''
Some general notes: 
    - Note that all column numbers are 0 based
    - When talking about ndarray, we are talking about 1D/2D array primarily
    - We would usually want our custom transformers to work seamless with Scikit_learn functionalities (such as pipelines).
      Thus, all the transformers here have three method: fit(), transform() and fit_transform() and their use is quite similar
      to Scikit Learn's transformers.
'''

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


class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    '''
    Fill in missing values in numerical column(s) using the median strategy

    Usage is similar to that of SimpleImputer from sklearn.impute. 
    
    When creating the instance object, pass in the column indices for which you want to use the imputer for

    Returns an ndarray where the imputed column(s) indices remain unchanged
    '''
    def __init__(self, cols_idx):
        '''
        Argument Names:
            cols_idx: list of column indices (0 based)
        '''
        self.cols_idx = cols_idx
    def fit(self, X):
        return self
    def transform(self, X):
        imputer = SimpleImputer(missing_values=np.nan, strategy = "median")
        X = X.copy()
        #Imputing Data
        #print(X[:, self.cols].shape)

        if len(self.cols_idx) == 1:
            #create a duplicate of the same column so we get 2d array so we can use simple imputer
            #at the end the duplicate column will be discarded anyways
            col_data_to_impute = np.concatenate([X[:, self.cols_idx],X[:, self.cols_idx]], axis = 1)
        else:
            col_data_to_impute = X[:, self.cols_idx]

        imputed_data = imputer.fit_transform(col_data_to_impute) #input needs to be 2d array
        for c in range(0, len(self.cols_idx)):
            X[:, self.cols_idx[c]] = imputed_data[:, c]
        return X

class StandardScalerCustom(BaseEstimator, TransformerMixin):
    '''
    Scales numerical column(s) using StandardScaler from sklearn.preprocessing . Usage is same as in using StandardScaler. 

    Returns an ndarray where the scaled column(s) indices remain unchanged
    '''
    def __init__(self, cols_idx):
        '''
        Argument Names:
            cols_idx: list of column indices (0 based) that are to be scaled
        '''
        self.cols_idx = cols_idx
    def fit(self, X):
        return self
    def transform(self, X):
        standard_scaler = StandardScaler()
        #Scaling Data
        if len(self.cols_idx) == 1:
            #create a duplicate of the same column so we get 2d array so we can use simple imputer
            #at the end the duplicate column will be discarded anyways
            col_data_to_impute = np.concatenate([X[:, self.cols_idx],X[:, self.cols_idx]], axis = 1)
        else:
            col_data_to_impute = X[:, self.cols_idx]

        scaled_data = standard_scaler.fit_transform(col_data_to_impute) 
        for c in range(0, len(self.cols_idx)):
            X[:, self.cols_idx[c]] = scaled_data[:, c]
        
        return X

class OutletAgeAdder(BaseEstimator, TransformerMixin):
    '''
    Modifies the Outlet_Establishment_Age to Outlet_Age
    '''
    def __init__(self, col_idx, current_year):
        '''
        Argument Names:
            cols_idx: index of the column containing "Outlet_Establishment_Age" (0 based)
            current_year: Current Year

        Returns an ndarray where the indices of ndarray passed in remain unchanged
        '''
        self.col_idx = col_idx
        self.current_year = current_year
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        outlet_age = self.current_year - X[:, self.col_idx]
        return np.c_[X[:, :self.col_idx], outlet_age, X[:, self.col_idx + 1:]]

class ConvertFloat(BaseEstimator, TransformerMixin):
    '''
    Converts the datatype of an ndarray to float. This transformer should only be used
    after making sure that all the columns can be casted to float

    Returns an ndarray where the indices of ndarray passed in remain unchanged

    '''
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


#Categorical Columns Preprocessing Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#Tansforming Item_Identifier
class ItemIdTransform(BaseEstimator, TransformerMixin):
    '''
    Transforms the "Item_Identifier" column such that only the first two charcter of the Item_Identifier is left

    Returns an ndarray where the indices of ndarray passed in remain unchanged 
    '''
    def __init__(self, col_idx):
        '''
        Argument Names:
            cols_idx: index of the column containing "Item_Identifier"

        Returns an ndarray where the indices of ndarray passed in remain unchanged
        '''
        self.col_idx = col_idx

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        id_transform = np.zeros(shape = (X.shape[0], 1)).astype(str)
        X = X.copy().astype(str)
        for row in range(0, X.shape[0]):
            id_transform[row] = X[row, self.col_idx][0:2]
        return np.c_[X[:, :self.col_idx], id_transform, X[:, self.col_idx + 1:]]

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

class ItemFatContentTrnsfrm(BaseEstimator, TransformerMixin): 
    '''
    Transforms Item_Fat_Content such that it only contains 1 if the item is low fat or 0 otherwise.
    '''
    def __init__(self, col_idx, replcm_dict):
        '''
        Argument Names:
            cols_idx: index of the column containing "Item_Fat_Content"
            replcm_dict: contains key: value pair where key contains one of the category currently present in "Item_Fat_Content" and 
                value is 0 or 1; 1 if that category correspond to low fat item or 0 otherwise

        Returns an ndarray where the indices of ndarray passed in remain unchanged
        '''
        self.col_idx = col_idx
        self.replcm_dict = replcm_dict
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #Tips from here:https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
        fat_content_trnsfrm = X.copy()[:, self.col_idx]
        #print(type(X))
        for k, v in self.replcm_dict.items():
            fat_content_trnsfrm[fat_content_trnsfrm == k] = v
        return np.c_[X[:, :self.col_idx], fat_content_trnsfrm, X[:, self.col_idx + 1:]]


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
    '''
        Imputes missing value of Outlet_Size with a replacement value

        Returns an ndarray where the indices of ndarray passed in remain unchanged
    '''
    def __init__(self, col_num, replcm_val):
        '''
        Argument Names:
            cols_idx: index of the column containing "Outlet_Size"
            replcm_val: the value to replace missing values with
        '''
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
    '''
    Encodes ordinal data. Usage is similar to that of OrdinalEncoder from sklearn.preprocessing

    Returns an ndarray where the encoded column(s) indices remain unchanged
    '''
    def __init__(self, cols, categories):
        '''
        Argument Names:
            cols_idx: indices of the categorical columns to encode
            categories: the categories argument for OrdinalEncoder
        '''
        self.cols = cols
        self.categories = categories

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
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
    '''
    One hot encodes categorical data. Usage is similar to that of OneHotEncoder from sklearn.preprocessing

    Returns an ndarray where the encoded column(s) are at the end of the array
    '''
    def __init__(self, cols_names, cols_idx):
        self.cols_names = cols_names
        self.cols_idx = cols_idx
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X): 
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

#Will be used to contain the one hot encoded column names
one_hot_coding_cols_catgs = []

cat_pipeline = Pipeline([
    ('item_id_transform', ItemIdTransform(sales_data.columns.get_loc("Item_Identifier"))),
    ("item_fat_cont_transfrm", ItemFatContentTrnsfrm(sales_data.columns.get_loc("Item_Fat_Content"), item_fat_content_replcm_dict)),
    ("outlet_size_imp", OutletSizeImpute(sales_data.columns.get_loc("Outlet_Size"), "Small")),
    ("ordinal_encoding", OrdinalEncoderCustom([sales_data.columns.get_loc("Outlet_Size"), sales_data.columns.get_loc("Outlet_Location_Type")], [outlet_size_ord_catgs, outlet_location_type_ord_catgs])),
    ("one_hot_encoding", OneHotEncodingCustom(one_hot_coding_cols_names, one_hot_coding_cols_idx))
    ])


preprocessing_full_pipeline = Pipeline([('num', num_pipeline),
                          ('cat', cat_pipeline),
                          ('convert_float', ConvertFloat())
                         ])

#Return four dataframe
def create_train_test_set(df, label_col_name):
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    df_train = train_set.drop(label_col_name, axis = 1)
    df_train_labels = train_set[label_col_name].copy()
    df_test = test_set.drop(label_col_name, axis = 1)
    df_test_labels = test_set[label_col_name].copy()

    return df_train, df_train_labels, df_test, df_test_labels

#After transformation, the columns are:
#attributes = ["Item_Weight", "Item_Fat_Content", "Item_Visibility", "Item_MRP", "Outlet_Age", "Outlet_Size", "Outlet_Location_Type"] + .one_hot_coding_cols_catgs

if __name__ == "__main__":
    #Testing transformation test
    sales_data_train, sales_data_train_labels, sales_data_test, sales_data_test_labels = create_train_test_set(sales_data, "Item_Outlet_Sales")
    sales_data_prepared = preprocessing_full_pipeline.fit_transform(sales_data_train.values)

    #Output submission if model exists
    import os
    final_model_path = r"C:\Users\KI PC\OneDrive\Documents\GitHub\Bigmart-Sales-Data\final_model.pkl"
    if os.path.exists(final_model_path):
        import joblib
        #load the model
        final_model = joblib.load(final_model_path)
        #load data and preprocess
        test_data_path = r"C:\Users\KI PC\OneDrive\Documents\GitHub\Bigmart-Sales-Data\test_AbJTz2l.csv"
        sales_data_comp = pd.read_csv(test_data_path)
        sales_data_comp_prepared = preprocessing_full_pipeline.fit_transform(sales_data_comp.values)
        #predict and output data
        outlet_sales_comp = final_model.predict(sales_data_comp_prepared)
        comp_submission_export = sales_data_comp[["Item_Identifier", "Outlet_Identifier"]]
        comp_submission_export["Item_Outlet_Sales"] = outlet_sales_comp
        submission_path = r"C:\Users\KI PC\OneDrive\Documents\GitHub\Bigmart-Sales-Data\submission.csv"
        comp_submission_export.to_csv(submission_path, index = False)