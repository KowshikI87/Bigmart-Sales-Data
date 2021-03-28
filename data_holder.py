Item_Identifier              8523 non-null object
Item_Weight                  7060 non-null float64
Item_Fat_Content             8523 non-null object
Item_Visibility              8523 non-null float64
Item_Type                    8523 non-null object
Item_MRP                     8523 non-null float64
Outlet_Identifier            8523 non-null object
Outlet_Establishment_Year    8523 non-null int64
Outlet_Size                  6113 non-null object
Outlet_Location_Type         8523 non-null object
Outlet_Type                  8523 non-null object
Item_Outlet_Sales            8523 non-null float64
dtypes: float64(4), int64(1), object(7)
memory usage: 799.2+ KB



Item_Identifier              8523 non-null object ---> Get first two letter only in a new column called: 
Item_Weight                  7060 non-null float64 ---> Impute based on Item_Identifier and then scale
Item_Fat_Content             8523 non-null object ---> 1 if Low Fat, 0 otherwise
Item_Visibility              8523 non-null float64 ---> untouched
Item_Type                    8523 non-null object ----> 1 hot encoding
Item_MRP                     8523 non-null float64 ---> scale feature
Outlet_Identifier            8523 non-null object ---> 1 hot encoding
Outlet_Establishment_Year    8523 non-null int64 ---> transform to Time till established
Outlet_Size                  6113 non-null object ---> ordinal encoding
Outlet_Location_Type         8523 non-null object ---> 1 hot encoding
Outlet_Type                  8523 non-null object ----> 1 hot encoding
Item_Outlet_Sales            8523 non-null float64 ---> Scale feature?


sales_data_train_num 
sales_data_num_attr



sales_data_train_cat
sales_data_cat_attr

There are couple of transformations to do here. First:

#Item_Identifier: Drop this column and replace it with a column which just takes the first two letter each Item_Identifier. We then need to use one hot encoding for each of those categories (total of 3 categories)

#Item_Fat_Content: As stated before, this column is used to keep track of whether an item is low fat or not. We can replace it with a simple binary field where 1 corresponds to Low Fat and 0 corresponds to non low fat food

Item_Type: One hot encoding

Outlet_Identifier: One hot encoding

#Outlet_Size: Figure out a impute strategy and then once imptuation is done, use ordinal encoding

Outlet_Location_Type: ordinal encoding

Outlet_Type: one hot encoding

