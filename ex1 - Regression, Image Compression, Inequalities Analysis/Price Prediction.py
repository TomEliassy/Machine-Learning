import pandas as pd
import numpy as np
import scipy as sc
from matplotlib.pylab import *

DATE, PRICE, BEDROOMS, BATHROOMS, SQFT_LOT, FLOORS, WATERFRONT,\
VIEW, CONDITION, GRADE, SQFT_ABOVE, SQFT_BASE, YEAR_BUILT,\
YEAR_RENOVATION,ZIP_CODE, LAT, LONG, SQFT_LIVING_15, SQFT_LOT_15 = range(19)

#-------------------------------------------Pre-processing--------------------------------
data_set = pd.read_csv("kc_house_data.csv")


def perform_preprocessing (data_set):
    # drop any nan
    data_set.dropna(axis=0, how="any", inplace=True)
    # ID doesn't matter and the sqft_living is linear dependence
    data_set.drop(labels=["id", "sqft_living" , "lat" , "long" ], axis=1, inplace=True)


    # Omits the suffix of the dates col.
    for idx,row in data_set.iterrows():
        data_set.at[idx, 'date'] = int(row['date'][:8])


    # First, check that there are only numbers in the data
    for idx,row in data_set.iterrows():
            # df.at['C', 'x'] = 10
        for cell in row:
            if not is_number(cell):
                data_set.drop(idx,axis=0, inplace=True)
                break
        if not check_validity_of_house(row):
            data_set.drop(idx,axis=0, inplace=True)

    # Update the year of renovated for lines with the value 0 to their built year
    data_set.loc[data_set['yr_renovated'] == 0 , 'yr_renovated'] = data_set['yr_built']

    # Add a col. of 1-s as mentioned in class in order to use the inner-product format
    data_set.insert(loc=0, column="1", value=1)

    # Categorial features
    pd.get_dummies(data_set, columns=['zipcode'])

    # Casting all the values to float
    data_set['date'] = data_set['date'].astype(int , copy = False)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Pre-processing: delete unreasonable data
def check_validity_of_house (house_detailes):
    #Date - checking
    if (int(house_detailes[DATE]) < 0 ) or (int(house_detailes[DATE]) > 20180426 ):
        return False

    # Price - checking
    if house_detailes[PRICE] <= 0:
        return False

    # Bedrooms - checking
    if house_detailes[BEDROOMS] < 0:
        return False

    # Bathrooms - checking
    if (house_detailes[BATHROOMS] < 0):
        return False

    # sqft - checking
    # if (house_detailes[SQFT_LOT] < 0) or (house_detailes[SQFT_LIVING_15] < 120):
        #return False

    # Floors - checking
    if house_detailes[FLOORS] < 1:
        return False

    # Waterfront - checking
    if house_detailes[WATERFRONT] < 0:
        return False

    # View - checking
    if house_detailes[VIEW] < 0:
        return False

    # Condition - checking
    if house_detailes[CONDITION] < 1:
        return False

    # Grade - checking
    if house_detailes[GRADE] < 1:
        return False

    # sqft_above, sqft_basement - checking

    if house_detailes[SQFT_ABOVE] <120 or (house_detailes[SQFT_BASE] <0):
        return False

    return True


perform_preprocessing(data_set)


#---------------------------------Calculate Predictor-------------------------------------
def predictor_calc (data_matrix , prices):
    prices_mat = prices.as_matrix()
    matrix = data_matrix.as_matrix()
    pseudo_inverse = np.linalg.pinv(matrix.T)
    mult = dot(pseudo_inverse.T , prices_mat)
    return mult


def perform_linear_reg (data_set):
    tst_err_lst , lrn_err_lst = [] , []
    for i in range(1 , 100):
        percentage = i / 100 # calculate the current percentage

        # create the train set & the tst set
        train_set = data_set.sample(frac = percentage)
        train_set_indexes = train_set.index
        test_set = data_set.drop(train_set_indexes)

        # actual prices
        train_prices = train_set["price"]
        test_prices = test_set["price"]

        # omit prices from tables
        train_set.drop("price", 1, inplace=True)
        test_set.drop("price",1, inplace=True)

        # create predictor
        predict = predictor_calc(train_set , train_prices)

        # produce the errors rate
        predicted_train_prices = dot(train_set.as_matrix() , predict)
        predicted_test_prices = dot(test_set.as_matrix() , predict)

        train_error_rate = (np.linalg.norm(train_prices - predicted_train_prices)**2)/len(predicted_train_prices)
        test_error_rate = (np.linalg.norm(test_prices - predicted_test_prices)**2)/len(predicted_test_prices)

        lrn_err_lst.append(train_error_rate)
        tst_err_lst.append(test_error_rate)
    return tst_err_lst , lrn_err_lst

test_errors , train_errors = perform_linear_reg(data_set)


figure(1)
title("Train results Vs. Test results")
plot(test_errors, label="test MSE")
plot(train_errors, label="train MSE")
legend(loc='upper right')
xlabel("percent of size")
ylabel("MSE")
show()
data_set.to_csv("t.csv",sep=",")
# Gets dummy data regarding the zipcode attribute

data_set.to_csv("t.csv",sep=",")

