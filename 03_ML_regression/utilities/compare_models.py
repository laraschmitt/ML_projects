"""
Functions to explore model performance of different machine learning algorithms.
"""

from statsmodels.api import OLS, add_constant
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
print("I am being executed!")


def scale_all_columns(df):
    df = df.copy()
    scaler = StandardScaler()
    for i in list(df.columns):
        df[i] = scaler.fit_transform(df[[i]])
    return df


def scale_one_col(df, column_name):
    df = df.copy()

    scaler = StandardScaler()

    df[column_name] = scaler.fit_transform(df[[column_name]])

    return df


def polynomial_single(df, column_name, depth):

    df = df.copy()

    df.reset_index(inplace=True)

    pt = PolynomialFeatures(degree=depth)

    Xpoly = df[[column_name]]
    p_features = pt.fit_transform(Xpoly)

    poly_df = pd.DataFrame(p_features, columns=[
                           column_name + s for s in pt.get_feature_names()])

    df = df.join(poly_df.iloc[:, 1:])
    df.drop(columns=column_name, inplace=True)

    df.set_index(["index"], inplace=True)

    return df


def poly_interaction(df, column_name_1, column_name_2, depth):

    df = df.copy()

    df.reset_index(inplace=True)

    pt = PolynomialFeatures(degree=depth)

    Xpoly = df[[column_name_1, column_name_2]]
    p_features = pt.fit_transform(Xpoly)

    poly_df = pd.DataFrame(p_features, columns=[
                           column_name_1 + "-" + column_name_2 + s for s in pt.get_feature_names()])

    df = df.join(poly_df.iloc[:, 1:])
    df.drop(columns=[column_name_1, column_name_2], inplace=True)

    df.set_index(["index"], inplace=True)

    return df


def one_hot_enc(df, column_name):

    df = df.copy()

    df[column_name] = column_name + "-" + df[column_name].astype(str)

    binary_sex = pd.get_dummies(df[column_name])
    df = df.join(binary_sex.iloc[:, :-1])

    df = df.drop(columns=column_name)

    return df


def mse(p, a):
    return (p-a)**2/1


def rmse(p, a):
    return np.sqrt((p-a)**2/1)


def rmsle(p, a):
    return np.sqrt(mean_squared_log_error(p, a))


def RandForestReg(Xtr, ytr, Xtest, ytest, mindepth, maxdepth):

    """Random Forest Regressor with search for best depth between mindepth and maxdepth"""

    rfr_best_depth = 2
    rfr_best_test = 0

    for i in range(mindepth, maxdepth + 1):
        depth = i
        mrf = RandomForestRegressor(max_depth=depth)
        mrf.fit(Xtr, ytr)
        rfr_test_score = mrf.score(Xtest, ytest)

        print(depth, rfr_test_score)

        if rfr_test_score > rfr_best_test:
            rfr_best_test = rfr_test_score
            rfr_best_depth = depth

    print('RandForReg, best depth between ' + str(mindepth) + ' and ' +
          str(maxdepth) + " is " + str(rfr_best_depth))

    mrf = RandomForestRegressor(max_depth=rfr_best_depth)
    mrf.fit(Xtr, ytr)

    m_score = mrf.score(Xtr, ytr)
    print("RandForest Train Score", (m_score * 100).round(2))
    c_val_score = cross_val_score(mrf, Xtr, ytr, cv=5)
    print("cross-validation score", (c_val_score * 100).round(2))
    print("cross-validation Average", (c_val_score.mean() * 100).round(2))

    mtest_score = mrf.score(Xtest, ytest)
    print("\nRandForest Test Score", (mtest_score * 100).round(2))

    y_pred = mrf.predict(Xtest)

    print("RandForestReg on Test\nMean Squarred Error: ",
          mse(y_pred, ytest).mean())
    print("Root Mean Squarred Error: ", rmse(y_pred, ytest).mean())
    print("Root Mean Squarred Log Error: ", rmsle(y_pred, ytest))


def LinReg(Xtr, ytr, Xtest, ytest):
    mlr = LinearRegression(normalize=True)
    mlr.fit(Xtr, ytr)

    print("LinearRegression Coefficients", mlr.coef_)
    print("LinearRegression Intercept", mlr.intercept_)
    print("LinearRegression Slope", mlr.coef_[0])

    print("\nLinearRegression Train Score",
          mlr.score(Xtr, ytr))
    c_val_score = cross_val_score(mlr, Xtr, ytr, cv=5)
    print("cross-validation score", c_val_score)
    print("cross-validation Average", c_val_score.mean())

    mtest_score = mlr.score(Xtest, ytest)
    print("\nLinearRegression Test Score", mtest_score)

    y_pred = mlr.predict(Xtest)
    y_pred[y_pred < 0] = 0

    print("LinearReg on Test\nMean Squarred Error: ",
          mse(y_pred, ytest).mean())
    print("Root Mean Squarred Error: ", rmse(y_pred, ytest).mean())
    print("Root Mean Squarred Log Error: ", rmsle(y_pred, ytest))


def RandForestClassif(Xtr, ytr, Xtest, ytest, n_estimators, mindepth, maxdepth):

    """Random Forest Classifier with search for best depth between mindepth and maxdepth"""


    rfr_best_depth = 2
    rfr_best_test = 0

    for i in range(mindepth, maxdepth + 1):
        depth = i
        mrf = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth)
        mrf.fit(Xtr, ytr)
        rfr_test_score = mrf.score(Xtest, ytest)

        print(depth, rfr_test_score)

        if rfr_test_score > rfr_best_test:
            rfr_best_test = rfr_test_score
            rfr_best_depth = depth

    print("RandForClass, best depth between " + str(mindepth) + ' and ' +
          str(maxdepth) + " is " + str(rfr_best_depth))


    mrf = RandomForestClassifier(max_depth=rfr_best_depth)
    mrf.fit(Xtr, ytr)

    m_score = mrf.score(Xtr, ytr)
    print("RandForestClass Train Score", (m_score * 100).round(2))
    c_val_score = cross_val_score(mrf, Xtr, ytr, cv=5)
    print("cross-validation score", (c_val_score * 100).round(2))
    print("cross-validation Average", (c_val_score.mean() * 100).round(2))

    mtest_score = mrf.score(Xtest, ytest)
    print("\nRandForestClass Test Score", (mtest_score * 100).round(2))

    y_pred = mrf.predict(Xtest)

    print("RandForestClass on Test\nMean Squarred Error: ",
          mse(y_pred, ytest).mean())
    print("Root Mean Squarred Error: ", rmse(y_pred, ytest).mean())
    print("Root Mean Squarred Log Error: ", rmsle(y_pred, ytest))


def GradientBoostReg(Xtr, ytr, Xtest, ytest, mindepth, maxdepth):

    """Gradient Boosting Regressor with search for best depth between mindepth and maxdepth"""

    rfr_best_depth = 2
    rfr_best_test = 0

    for i in range(mindepth, maxdepth + 1):
        depth = i
        mrf = GradientBoostingRegressor(max_depth=depth)
        mrf.fit(Xtr, ytr)
        rfr_test_score = mrf.score(Xtest, ytest)

        print(depth, rfr_test_score)

        if rfr_test_score > rfr_best_test:
            rfr_best_test = rfr_test_score
            rfr_best_depth = depth

    print("GradientBoostingRegressor, best depth between 1 and " +
          str(maxdepth) + " is " + str(rfr_best_depth))

    mrf = RandomForestClassifier(max_depth=rfr_best_depth)
    mrf.fit(Xtr, ytr)

    m_score = mrf.score(Xtr, ytr)
    print("GradientBoostingRegressor Train Score", (m_score * 100).round(2))
    c_val_score = cross_val_score(mrf, Xtr, ytr, cv=5)
    print("cross-validation score", (c_val_score * 100).round(2))
    print("cross-validation Average", (c_val_score.mean() * 100).round(2))

    mtest_score = mrf.score(Xtest, ytest)
    print("\nGradientBoostingRegressor Test Score", (mtest_score * 100).round(2))

    y_pred = mrf.predict(Xtest)

    print("GradientBoostingRegressor on Test\nMean Squarred Error: ",
          mse(y_pred, ytest).mean())
    print("Root Mean Squarred Error: ", rmse(y_pred, ytest).mean())
    print("Root Mean Squarred Log Error: ", rmsle(y_pred, ytest))


def SuppVecReg(Xtr, ytr, Xtest, ytest):

    """Support Vector Regressor with different kernels"""

    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

    svr_rbf.fit(Xtr, ytr)
    svr_lin.fit(Xtr, ytr)
    svr_poly.fit(Xtr, ytr)


    svr_rbf_score = svr_rbf.score(Xtr, ytr)
    svr_lin_score = svr_lin.score(Xtr, ytr)
    svr_poly_score = svr_poly.score(Xtr, ytr)

    print("svr_rbf_score Train Score", (svr_rbf_score * 100).round(2))
    print("svr_lin_score Train Score", (svr_lin_score * 100).round(2))
    print("svr_poly_score Train Score", (svr_poly_score * 100).round(2))


    #c_val_score = cross_val_score(mrf, Xtr, ytr, cv=5)
    #print("cross-validation score", (c_val_score * 100).round(2))
    #print("cross-validation Average", (c_val_score.mean() * 100).round(2))

    svr_rbf_score = svr_rbf.score(Xtest, ytest)
    svr_lin_score = svr_lin.score(Xtest, ytest)
    svr_poly_score = svr_poly.score(Xtest, ytest)
    print("\svr_rbf Test Score", (svr_rbf_score * 100).round(2))
    print("\svr_lin Test Score", (svr_lin_score * 100).round(2))
    print("\svr_poly Test Score", (svr_poly_score * 100).round(2))


    y_pred_svr_rbf = svr_rbf.predict(Xtest)
    y_pred_svr_lin = svr_lin.predict(Xtest)
    y_pred_svr_poly = svr_poly.predict(Xtest)

    print("svr_rbf on Test\nMean Squarred Error: ",
          mse(y_pred_svr_rbf, ytest).mean())
    print("svr_lin on Test\nMean Squarred Error: ",
          mse(y_pred_svr_lin, ytest).mean())
    print("svr_poly on Test\nMean Squarred Error: ",
          mse(y_pred_svr_poly, ytest).mean())

    print("Root Mean Squarred Error rbf: ", rmse(y_pred_svr_rbf, ytest).mean())
    print("Root Mean Squarred Error lin: ", rmse(y_pred_svr_lin, ytest).mean())
    print("Root Mean Squarred Error poly: ", rmse(y_pred_svr_poly, ytest).mean())


    print("Root Mean Squarred Log Error rbf: ", rmsle(y_pred_svr_rbf, ytest))
    print("Root Mean Squarred Log Error lin: ", rmsle(y_pred_svr_lin, ytest))
    print("Root Mean Squarred Log Error poly: ", rmsle(y_pred_svr_poly, ytest))


