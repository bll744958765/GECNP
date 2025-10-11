import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from math import sqrt
from sklearn.preprocessing import StandardScaler
from pykrige.ok import OrdinaryKriging
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import csv
import time
start = time.perf_counter()
time.sleep(2)
for xuhao  in range (10):
    data_name1 = {0: './train0.csv', 1: './train1.csv', 2: './train2.csv', 3: './train3.csv', 4: './train4.csv',
                  5: './train5.csv', 6: './train6.csv', 7: './train7.csv', 8: './train8.csv', 9: './train9.csv',
                  10: './train10.csv', 11: './train11.csv', 12: './train12.csv', 13: './train13.csv', 14: './train14.csv'}
    data_name2 = {0: './valid0.csv', 1: './valid1.csv', 2: './valid2.csv', 3: './valid3.csv', 4: './valid4.csv',
                  5: './valid5.csv', 6: './valid6.csv', 7: './valid7.csv', 8: './valid8.csv', 9: './valid9.csv',
                  10: './valid10.csv', 11: './valid11.csv', 12: './valid12.csv', 13: './valid13.csv', 14: './valid14.csv'}

    train_data = pd.read_csv(data_name1[xuhao])
    valid_data = pd.read_csv(data_name2[xuhao])

    # train_data = pd.read_csv('F:\CNP_Variance\california_housing\california_housing/train0.csv')
    # valid_data = pd.read_csv('F:\CNP_Variance\california_housing\california_housing/valid0.csv')
    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    train_x = train_data[:, 1:3]
    train_y = train_data[:, 9]
    valid1_x = valid_data[:, 1:3]
    valid1_y = valid_data[:, 9]
    train_id = train_data[:, 0]
    valid1_id = valid_data[:, 0]
    x=torch.tensor(train_x)
    y = torch.tensor(train_y)
    valid_x=torch.tensor(valid1_x)
    valid_y = torch.tensor(valid1_y).float().squeeze(-1)
    train_lon = x[:, 0]
    train_lat = x[:, 1]
    # x = torch.FloatTensor(scaler.fit_transform(train_x)).float()
    # valid_x = torch.FloatTensor(scaler.transform(valid1_x)).float()
    m=0
    t=200
    for tt in range(93):

        valid_x=valid1_x [m:m+t,:]
        valid_y=valid1_y [m:m+t]
        valid_id=valid1_id[m:m+t]
        m=m+t
        print ("tt:",tt)
        OK = OrdinaryKriging(train_lon, train_lat, y, variogram_model='gaussian', nlags=20)  # 0.1349
        z1, ss1 = OK.execute('grid', valid_x[:,0], valid_x[:,1])
        # print("valid_id",valid_id,"z1",z1)
        # print("ss1",ss1)
        ok_z = []
        sigma=[]
        for i in range(len(valid_y)):
            for j in range(len(valid_y)):
                if i == j:
                    thickness = z1[i][j]
                    variance=ss1[i][j]
                    ok_z.append(thickness)
                    sigma.append(variance)


        # ok_val_index = valid_id.argsort()
        # ok_valid_id = valid_id[ok_val_index]
        # ok_valid_y = valid_y[ok_val_index]
        # z=torch.tensor(z)
        # ok_z = z[ok_val_index]

        ok_value = pd.DataFrame({'id': valid_id, 'SValue':ok_z,"variance":sigma})
        ok_value.to_csv('F:\CNP_Variance\california_housing\california_housing/ok2_{}.csv'.format(xuhao),header=False, index=False,mode="a+")

    # #预测 ok_heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    data_frame = pd.read_csv(r'F:\SMANP\data code\SMANP_revoir/ok2.csv')


    # flights = data_frame.pivot("CMP", "Line", "pred")
    # f, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(flights, fmt="d",cmap="RdBu_r",ax=ax)
    # plt.title('Reservoir thickness predicted by Ordinary Kriging',size=15)
    # # plt.savefig(r'C:\Users\baoll\Desktop\Revised Manuscript with no Changes Marked/ok_heatmap1.jpg',dpi=500)
    # plt.show()

    # flights1 = data_frame.pivot("CMP", "Line", "var")
    # f, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(flights1, fmt="d",cmap="RdBu_r",ax=ax,vmin=0,vmax=120)
    # plt.title('Uncertainty  quantified by Ordinary Kriging ',size=15)
    # # plt.savefig(r'C:/Users/baoll/Desktop/Revised Manuscript with no Changes Marked/ok_uncertainty.jpg',dpi=500)
    # plt.show()

    end = time.perf_counter()
    print(str(end - start))
for i in range(10):
    data = pd.read_csv(r'./ok2_{}.csv'.format(i))
    data=np.array(data)

    ok_pred=data[:,1]
    variance=data[:,2]
    true=data[:,3]


    valid_r2 = 1 - ((np.sum((true - ok_pred) ** 2)) / np.sum((true - true.mean()) ** 2))
    valid_mse = (np.sum((true - ok_pred) ** 2)) / len(true)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae = (np.sum(np.absolute(true - ok_pred))) / len(true)
    corr = np.corrcoef(true,ok_pred)
    print("ID:", xuhao, "valid_MAE:", round(valid_mae, 4), "valid_MSE:", round(valid_mse, 4), " valid_RMSE:", round(valid_rmse, 4),
          " valid_R-square:", round(valid_r2.item(), 4), "average_var:", round(np.mean(variance), 4))
