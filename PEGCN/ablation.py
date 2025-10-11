import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import torch
from math import sqrt
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
start = time.perf_counter()
time.sleep(2)

# def data_load():
train_data = pd.read_csv('F:\SMANP\data code\SMANP_revoir/train.csv')
valid_data = pd.read_csv('F:\SMANP\data code\SMANP_revoir/valid.csv')
train_data = np.array(train_data)
valid_data = np.array(valid_data)
train_x = train_data[:, 1:15]
train_y = train_data[:, (0, 15)]
valid1_x = valid_data[:, 1:15]
valid_y = valid_data[:, (0, 15)]
train_id = train_data[:, 0]
valid_id = valid_data[:, 0]

# standardization
scaler = StandardScaler()  # standand
# new_data = torch.FloatTensor(scaler.fit_transform(new1_data)).to(device).float()
y = torch.tensor(train_y)
valid_y = torch.tensor(valid_y).float().squeeze(-1)
x = torch.FloatTensor(scaler.fit_transform(train_x)).float()
valid_x = torch.FloatTensor(scaler.transform(valid1_x)).float()
    # return x, y, valid_x, valid_y


# x, y, valid_x, valid = data_load()
valid_id = valid_y[:, 0]
valid_y = valid_y[:, 1]
y = y[:, 1]
# mlp = MLPRegressor(hidden_layer_sizes=(128, 128, 128, 128, 128), activation="relu",
#                    solver='adam', alpha=0.0001,
#                    batch_size='auto', learning_rate="constant",
#                    learning_rate_init=0.001,
#                    max_iter=1200, tol=1e-4, shuffle=True)

# mlp.fit(x, y)
# mlp_K_pred = mlp.predict(valid_x)
# mlp_score = r2_score(valid_y, mlp_K_pred)
# print(mlp_score)
# mlp_val_index = valid_id.argsort()
# mlp_valid_id = valid_id[mlp_val_index]
# mlp_pred_y = mlp_K_pred[mlp_val_index]
# mlp_valid_y = valid_y[mlp_val_index]
# valid_mse = (torch.sum((mlp_valid_y - mlp_pred_y) ** 2)) / len(mlp_valid_y)
# valid_rmse = sqrt(valid_mse)
# valid_mae = (torch.sum(torch.absolute(mlp_valid_y - mlp_pred_y))) / len(valid_y)
# valid_r2 = 1 - ((torch.sum((mlp_valid_y - mlp_pred_y) ** 2)) / torch.sum(
#     (mlp_valid_y - mlp_valid_y.mean()) ** 2))

# corr= np.corrcoef(mlp_valid_y, mlp_pred_y)
# corr=torch.tensor(corr)
# print(corr[0,1])
# mlp_C=(2*corr[0,1]*mlp_pred_y.std()*mlp_valid_y.std())/(mlp_valid_y.var()+mlp_pred_y.var()+(mlp_valid_y.mean()-mlp_pred_y.mean())**2)
# print("mlp: \t mae: {:.4f},mse: {:.4f}, rmse: {:.4f},r2: {:.4f},CCC: {:.4f}".format(
#     valid_mae, valid_mse, valid_rmse, valid_r2,mlp_C))
# c = np.linspace(0, len(mlp_pred_y), len(mlp_pred_y))
# plt.scatter(c, mlp_pred_y, color="b", marker="x", label="true_value")
# plt.scatter(c, mlp_valid_y, color="r", marker="x", label="true_value")
# plt.legend()

#
# mixed_kernel = C(1.0, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4))  # 径向基径向基Radial basis function,简称RBF
# gp = GaussianProcessRegressor(kernel=mixed_kernel, n_restarts_optimizer=50, alpha=5)
# gp.fit(x, y)
# gp_K_pred,gp_std = gp.predict(valid_x,return_std=True)
# # print(gp_std)
# gp_score = r2_score(valid_y, gp_K_pred)
# # print(gp_score)
# gp_val_index = valid_id.argsort()
# gp_valid_id = valid_id[gp_val_index]
# gp_pred_y = gp_K_pred[gp_val_index]
# gp_valid_y = valid_y[gp_val_index]
# gp_st=gp_std[gp_val_index]
# gp_valid_mse = (torch.sum((valid_y - gp_pred_y) ** 2)) / len(gp_valid_y)
# gp_valid_rmse = sqrt(gp_valid_mse)
# gp_valid_mae = (torch.sum(torch.absolute(gp_valid_y - gp_pred_y))) / len(valid_y)
# gp_valid_r2 = 1 - ((torch.sum((gp_valid_y - gp_pred_y) ** 2)) / torch.sum(
#     (valid_y - valid_y.mean()) ** 2))

# corr= np.corrcoef(gp_valid_y, gp_pred_y)
# corr=torch.tensor(corr)
# # print(corr[0,1])
# C=(2*corr[0,1]*gp_pred_y.std()*gp_valid_y.std())/(gp_valid_y.var()+gp_pred_y.var()+(gp_valid_y.mean()-gp_pred_y.mean())**2)

# print("gp: \t mae: {:.4f},mse: {:.4f}, rmse: {:.4f},r2: {:.4f},CCC: {:.4f},var:{:.4f}".format(
#     gp_valid_mae, gp_valid_mse, gp_valid_rmse, gp_valid_r2,C,gp_st.mean()))
# c = np.linspace(0, len(gp_pred_y), len(gp_pred_y))
# import scipy.stats as st
# low_CI_bound, high_CI_bound = st.t.interval(0.95, len(gp_pred_y) - 1,
#                                             loc=gp_pred_y,
#                                             scale=gp_std)

# plt.scatter(c, gp_pred_y, color="b", marker="x", label="gp_pred_value")
# plt.scatter(c, gp_valid_y, color="r", marker="x", label="gp_true_value")
# # plt.fill_between(c,gp_pred_y- gp_std, gp_pred_y+gp_std, alpha=0.2, facecolor="r",
# #                      interpolate=True)
# plt.fill_between(c, low_CI_bound, high_CI_bound, alpha=0.2,facecolor="r",
#                 label='confidence interval')
# plt.legend()
# plt.show()

# from sklearn.linear_model import LinearRegression
# linreg = LinearRegression()
# model = linreg.fit(x, y)
# print('模型参数:')
# print(model)
# linreg.pred = linreg.predict(valid_x)

# linreg_score = r2_score(valid_y, linreg.pred)
# print(linreg_score)
# linreg_val_index = valid_id.argsort()
# linreg_valid_id = valid_id[linreg_val_index]
# linreg_pred_y = linreg.pred[linreg_val_index]
# linreg_valid_y = valid_y[linreg_val_index]

# corr= np.corrcoef(linreg_valid_y, linreg_pred_y)
# corr=torch.tensor(corr)
# C=(2*corr[0,1]*linreg_pred_y.std()*linreg_valid_y.std())/(linreg_valid_y.var()+linreg_pred_y.var()+(linreg_valid_y.mean()-linreg_pred_y.mean())**2)

# linreg_valid_mse = (torch.sum((valid_y - linreg_pred_y) ** 2)) / len(linreg_valid_y)
# linreg_valid_rmse = sqrt(linreg_valid_mse)
# linreg_valid_mae = (torch.sum(torch.absolute(linreg_valid_y - linreg_pred_y))) / len(valid_y)
# linreg_valid_r2 = 1 - ((torch.sum((linreg_valid_y - linreg_pred_y) ** 2)) / torch.sum(
#     (valid_y - valid_y.mean()) ** 2))
# print("linreg: \t mae: {:.4f},mse: {:.4f}, rmse: {:.4f},r2: {:.4f},CCC: {:.4f}".format(
#     linreg_valid_mae, linreg_valid_mse, linreg_valid_rmse, linreg_valid_r2,C))
# c = np.linspace(0, len(linreg_pred_y), len(linreg_pred_y))
# import scipy.stats as st

# plt.scatter(c, linreg_pred_y, color="b", marker="x", label="gp_pred_value")
# plt.scatter(c, linreg_valid_y, color="r", marker="x", label="gp_true_value")

# plt.legend()
# plt.show()

# from pykrige.ok import OrdinaryKriging
# train_lon = x[:, 0]
# train_lat = x[:, 1]
# # OK = OrdinaryKriging(train_lon, train_lat, y, variogram_model='spherical',nlags=6) #0.4725
# # OK = OrdinaryKriging(train_lon, train_lat, y, variogram_model='linear',nlags=6) #0.3901
# # OK = OrdinaryKriging(train_lon, train_lat, y, variogram_model='exponential',nlags=6) #0.5723
# OK = OrdinaryKriging(train_lon, train_lat, y, variogram_model='hole-effect',nlags=8)  # 0.6088
# # OK = OrdinaryKriging(train_lon, train_lat, y, variogram_model='gaussian', nlags=20)  # 0.1349
# z1, ss1 = OK.execute('grid', valid_x[:,0], valid_x[:,1])
# z = []
# ss=[]
# for i in range(len(valid_y)):
#     for j in range(len(valid_y)):
#         if i == j:
#             l = z1[i][j]
#             s=ss1[i][j]
#             z.append(l)
#             ss.append(s)
# # print('z:', len(z), z)

# ok_val_index = valid_id.argsort()
# ok_valid_id = valid_id[ok_val_index]
# ok_valid_y = valid_y[ok_val_index]
# ss=torch.tensor(ss)
# z=torch.tensor(z)
# ok_z = z[ok_val_index]
# ok_ss=ss[ok_val_index]
# ok_valid_mse = (torch.sum((ok_valid_y - ok_z) ** 2)) / len(ok_z)  # 残差平方
# ok_valid_rmse = sqrt(ok_valid_mse)  # 残差平方开根号
# ok_valid_mae = (torch.sum(torch.absolute(ok_valid_y - ok_z))) / len(ok_valid_y)  # 残差绝对值
# ok_valid_r2 = 1 - ((torch.sum((ok_valid_y - ok_z) ** 2))/torch.sum((ok_valid_y-ok_valid_y.mean())**2)) # R2均方误差/方差np.var(valid_y)

# corr= np.corrcoef(ok_valid_y, ok_z)
# corr=torch.tensor(corr)
# C=(2*corr[0,1]*ok_z.std()*ok_valid_y.std())/(ok_valid_y.var()+ok_z.var()+(ok_valid_y.mean()-ok_z.mean())**2)


# ok_value = pd.DataFrame({'id': ok_valid_id, 'line': valid_data[:,1],'CMP': valid_data[:,2], 'SValue':ok_valid_y, 'validation': ok_z,
#                              'noise':ok_valid_y-ok_z})
# ok_value.to_csv('C:/Users/baoll/Desktop/SANP/ANP_revoir/ok.csv', index=False)
# print("OK: \t mae: {:.4f},mse: {:.4f}, rmse: {:.4f},r2: {:.4f},CCC: {:.4f},var:{:.4f}".format(
#     ok_valid_mae, ok_valid_mse, ok_valid_rmse, ok_valid_r2, C, ok_ss.mean()))
# c = np.linspace(0, len(valid_y), len(valid_y))
# plt.scatter(c, ok_z, color="b", marker="x", label="ok_pred_value")
# plt.scatter(c, ok_valid_y, color="r", marker="x", label="ok_true_value")
# plt.legend()
# # plt.show()

end = time.perf_counter()
print (str(end-start))