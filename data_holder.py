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

