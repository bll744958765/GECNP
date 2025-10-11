# -*- coding : utf-8 -*-

from dataloader_anp import DatasetGP, DatasetGP_test, data_load
from model_anp import SpatialNeuralProcess, Criterion
#from tensorboardX import SummaryWriter
import torch as torch
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from train_configs import train_runner, val_runner
from math import sqrt
import argparse
from dataloader_anp import split, set_seed
import random
from datetime import datetime
import time
import warnings
from pathlib import Path

start = time.perf_counter()
time.sleep(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")


def main(args):
    xuhao = args.xuhao
    set_seed(args.random_state)
    n_epoches = args.epochs
    n_tasks = args.task
    batch_size = args.batch_size
    start_lr = args.lr
    num_hidden = args.num_hidden
    dataset = args.dataset
    model_name = args.model_name
    encMoe = args.encMoe
    decMoe = args.decMoe
    # moran = args.moran
    path = args.path
    lam = args.lambd
    split(args.xuhao, args.random_state, args.dataset)
    x, y, valid_x, valid_y = data_load(1, args.dataset)
    n_context_min = args.n_context_min
    n_target_max = y.shape[0]
    n_context_max = n_target_max * 0.8
    x_size = x.shape[1]
    y_size = y.shape[1]
    args.checkpoint_path = Path(args.checkpoint_path)

    # Tensorboard and logging
    test_ = dataset + '-' + model_name + '-' + '-emb' + str(num_hidden)
    test_ = test_ + "-lr" + str(start_lr) + "-ep" + str(n_epoches) + "-xuhao" + str(args.xuhao)
    if encMoe:
        if args.laplace:
            if args.Similarity1:
                test_ = test_ + "_encMoe" + "_laplace" + "_Similar"
            else:
                test_ = test_ + "_encMoe" + "_laplace" + "_linear"
        else:
            if args.Similarity1:
                test_ = test_ + "_encMoe" + "_eulid" + "_Similar"
            else:
                test_ = test_ + "_encMoe" + "_eulid" + "_linear"
    if decMoe:
        test_ = test_ + "_decMoe"
    # if moran:
    test_ = test_ + "_lam" + str(lam)
    print(args.resume)
    path11 = Path(path + 'trained/{}/checkpoint/checkpoint_0.pth.tar'.format(args.checkpoint_path))
    print(path11.exists())
    print(Path(args.checkpoint_path).exists())
    if args.resume and path11.exists():
        print(f"Loading checkpoint from {args.checkpoint_path}")
        saved_file = Path(path + 'trained/{}'.format(args.checkpoint_path))
        print(saved_file)


    else:

        saved_file = "{}_{}{}_{}{}".format(test_,
                                           datetime.now().strftime("%h"),
                                           datetime.now().strftime("%d"),
                                           datetime.now().strftime("%H"),
                                           datetime.now().strftime("%M"),
                                           )

    if not os.path.exists(path + "trained/{}/result".format(saved_file)):
        os.makedirs(path + "/trained/{}/result".format(saved_file))
    for xuhao in range(xuhao):
        model = SpatialNeuralProcess(x_size=x_size, y_size=y_size, num_hidden=num_hidden,
                                     encMoe=args.encMoe, decMoe=args.decMoe, laplace=args.laplace, Similarity1=args.Similarity1).to(device)
        criterion = Criterion(lam=lam)
        optimizer = optim.Adam(model.parameters(), lr=start_lr)

        trainset = DatasetGP(n_tasks=n_tasks, xuhao=xuhao, dest=args.dataset, batch_size=batch_size,
                             n_context_min=n_context_min, n_context_max=n_context_max, n_target_max=n_target_max)
        testset = DatasetGP_test(n_tasks=n_tasks, xuhao=xuhao, dest=args.dataset, batch_size=batch_size)
        model.train()
        epoch_train_loss = []
        epoch_valid_loss = []
        lo = np.inf

        start_epoch = 0

        if args.resume and path11.exists():
            path1 = path + 'trained/{}/checkpoint/checkpoint_0.pth.tar'.format(saved_file)
            print(f"Loading checkpoint from {path1}")
            checkpoint = torch.load(path1)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1



        for epoch in range(start_epoch, n_epoches):
            trainloader = DataLoader(trainset, shuffle=True)
            testloader = DataLoader(testset, shuffle=True)
            # adjust_learning_rate(optimizer, start_lr,epoch+1)
            mean_y, var_y, target_id, target_y, context_id, loss, train_mse = train_runner(
                model, trainloader, criterion, optimizer)

            val_pred_y, val_var_y, val_target_id, val_target_y, val_loss, valid_mse = val_runner(
                model, testloader, criterion)

            if lo >= valid_mse:
                lo = valid_mse
                train_mse = (torch.sum((target_y - mean_y) ** 2)) / len(target_y)
                train_rmse = sqrt(train_mse)
                train_mae = (torch.sum(torch.absolute(target_y - mean_y))) / len(target_y)
                train_mape = torch.mean(torch.abs((target_y - mean_y) / target_y)) * 100
                train_r2 = 1 - ((torch.sum((target_y - mean_y) ** 2)) / torch.sum((target_y - target_y.mean()) ** 2))
                valid_mse = (torch.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
                valid_rmse = sqrt(valid_mse)
                valid_mae = (torch.sum(torch.absolute(val_target_y - val_pred_y))) / len(val_target_y)
                valid_mape = torch.mean(torch.abs((val_target_y - val_pred_y) / val_target_y)) * 100
                valid_r2 = 1 - ((torch.sum((val_target_y - val_pred_y) ** 2)) / torch.sum((val_target_y - val_target_y.mean()) ** 2))
                print(
                    "ID: {} \t Train Epoch: {} \t Lr:{:.4f},train loss: {:.4f},train_mae: {:.4f},train_mse: {:.4f}, train_rmse: {:.4f},train_mape: {:.4f},train_r2: {:.4f}".format(
                        xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], loss, train_mae, train_mse, train_rmse, train_mape, train_r2))
                print(
                    "ID: {} \t Valid Epoch: {} \t Lr:{:.4f},valid loss: {:.4f},valid_mae: {:.4f},valid_mse: {:.4f}, valid_rmse: {:.4f},valid_mape: {:.4f},valid_r2: {:.4f}".format(
                        xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], val_loss, valid_mae, valid_mse, valid_rmse, valid_mape, valid_r2))

                save_path = path + "/trained/{}/checkpoint".format(saved_file)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           path + '/trained/{}/checkpoint/checkpoint_{}.pth.tar'.format(saved_file, xuhao))
            if epoch % 30 == 0:
                print(
                    "ID: {} \t Train Epoch: {} \t Lr:{:.4f},train loss: {:.4f},train_mse: {:.4f}".format(
                        xuhao, epoch, optimizer.state_dict()['param_groups'][0]['lr'], loss, train_mse))
                print(
                    "ID: {} \t Valid Epoch: {} \t Lr:{:.4f},valid loss: {:.4f},valid_mse: {:.4f}".format(
                        xuhao, epoch, optimizer.state_dict()['param_groups'][0]['lr'], val_loss, valid_mse))

            epoch_train_loss.append(loss)
            epoch_valid_loss.append(val_loss)

        # epoch loss
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        epoch = np.linspace(1, n_epoches, n_epoches)
        plt.plot(epoch, epoch_train_loss, 'blue', label='Train loss')
        plt.plot(epoch, epoch_valid_loss, 'r', label='Valid loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(path + '/trained/{}/result/loss_{}.jpg'.format(saved_file, xuhao))
        plt.close()
        loss_value = pd.DataFrame({'epoch_train_loss': epoch_train_loss, 'epoch_valid_loss': epoch_valid_loss})
        loss_value.to_csv(path + '/trained/{}/result/loss_{}.csv'.format(saved_file, xuhao), index=False)

        c = np.linspace(0, len(val_target_y), len(val_target_y))
        plt.scatter(c, val_target_y.detach().cpu().numpy(), color="b", marker="x", label="true_value", vmin=0, vmax=50)
        plt.scatter(c, val_pred_y.detach().cpu().numpy(), color="r", marker="o", label="predict_value", vmin=0, vmax=50)
        plt.fill_between(c, (val_pred_y - val_var_y).detach().cpu().numpy(), (val_pred_y + val_var_y).detach().cpu().numpy(), alpha=0.2, facecolor="r",
                         interpolate=True)
        plt.legend()
        plt.xlabel('Well-ID')
        plt.ylabel('Reservoir thickness')
        plt.savefig(path + '/trained/{}/result/predict_{}.jpg'.format(saved_file, xuhao), bbox_inches='tight')
        plt.grid()
        plt.close()

    end = time.perf_counter()
    print(str(end - start))

    # inference stage
    print('Inference stage')
    list1 = []
    list0 = []
    model = SpatialNeuralProcess(x_size=x_size, y_size=y_size, num_hidden=num_hidden,
                                 encMoe=args.encMoe, decMoe=args.decMoe, laplace=args.laplace, Similarity1=args.Similarity1)
    model = model.to(device)
    for xuhao in range(args.xuhao):
        dataset = DatasetGP_test(n_tasks=n_tasks, xuhao=xuhao, dest=args.dataset, batch_size=batch_size)
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        state_dict = torch.load(path + "trained/{}/checkpoint/checkpoint_{}.pth.tar".format(saved_file, xuhao))
        # state_dict = torch.load('./checkpoint_anp/checkpoint_{}.pth.tar'.format(xuhao))
        model.load_state_dict(state_dict=state_dict['model'])
        model.eval()
        criterion = Criterion(lam)

        val_pred_y, val_var_y, val_target_id, val_target_y, val_loss, valid_mse = val_runner(model, test_loader, criterion)

        val_target_y = val_target_y.cpu().detach().numpy()
        val_pred_y = val_pred_y.cpu().detach().numpy()
        val_var_y = val_var_y.cpu().detach().numpy()
        val_target_id = val_target_id.cpu().detach().numpy()
        valid_r2 = 1 - ((np.sum((val_target_y - val_pred_y) ** 2)) / np.sum((val_target_y - val_target_y.mean()) ** 2))
        valid_mse = (np.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
        valid_rmse = np.sqrt(valid_mse)
        valid_mae = (np.sum(np.absolute(val_target_y - val_pred_y))) / len(val_target_y)
        valid_mape = np.mean(np.abs((val_target_y - val_pred_y) / val_target_y)) * 100

        corr = np.corrcoef(val_target_y, val_pred_y)
        C = (2 * corr[0, 1] * np.std(val_pred_y) * np.std(val_target_y)) / (np.var(val_target_y) + np.var(val_pred_y) + (val_target_y.mean() - val_pred_y.mean()) ** 2)
        corr_t_p = np.corrcoef(val_target_y, val_pred_y)
        corr_t_v = np.corrcoef(val_target_y, val_var_y)
        corr_p_v = np.corrcoef(val_pred_y, val_var_y)
        corr_r_v = np.corrcoef(val_pred_y - val_target_y, val_var_y)

        prediction = pd.DataFrame(
            {"id": np.array(val_target_id), "true": np.array(val_target_y), "pred": np.array(val_pred_y),
             "cha": np.array(val_target_y) - np.array(val_pred_y), 'var_y': np.array(val_var_y)})
        prediction.to_csv(path + '/trained/{}/result/prediction_val_{}.csv'.format(saved_file, xuhao), index=False)

        print("ID:", xuhao, "valid_MAE:", round(valid_mae, 4), "valid_MSE:", round(valid_mse, 4), " valid_RMSE:", round(valid_rmse, 4),
              "valid_mape:", round(valid_mape.item(), 4), " valid_R-square:", round(valid_r2.item(), 4), "CCC:", round(C, 4), "average_var:", round(np.mean(val_var_y), 4),
              'corr_t_p:', round(corr_t_p[0, 1], 4), "corr_t_v:", round(corr_t_v[0, 1], 4), 'corr_p_v:', round(corr_p_v[0, 1], 4), "corr_r_v:", round(corr_r_v[0, 1], 4)
              , 'val_loss:', val_loss)

        list0 = [round(valid_mae, 4), round(valid_mse, 4), round(valid_rmse, 4), round(valid_mape, 4), round(valid_r2, 4), round(C, 4), round(np.mean(val_var_y), 4),
                 round(corr_t_p[0, 1], 4), round(corr_t_v[0, 1], 4), round(corr_p_v[0, 1], 4), round(corr_r_v[0, 1], 4), val_loss]
        list1.append(list0)

        with open(path + "/trained/{}/result/train_notes.txt".format(saved_file), 'a+') as f:
            # Include any experiment notes here:
            f.write("Experiment notes: .... \n")
            f.write("MODEL_DATA: {}\n".format(test_))
            f.write("mae: {} mse: {} rmse: {} mape: {} r2: {}  val_loss: {}\n\n".format(
                round(valid_mae, 4), round(valid_mse, 4), round(valid_rmse, 4), round(valid_mape, 4), round(valid_r2, 4), val_loss
            ))

    # print('list:', list1)
    print(np.mean(list1, axis=0))
    print(np.std(list1, axis=0))

    with open(path + "/trained/{}/result/train_notes.txt".format(saved_file), 'a+') as f:
        # Include any experiment notes here:
        f.write("final  mae, mse, rmse, mape, r2, C, val_var_y,corr_t_p, corr_t_v, corr_p_v, corr_r_v, val_loss \n")
        f.write("mean: {}\n std: {} \n".format(np.mean(list1, axis=0), np.std(list1, axis=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MoeNP Regression on Spatial Data")
    parser.add_argument('-seed', '--random_state', type=int, default=42)
    parser.add_argument('-xuhao', '--xuhao', type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300, help="number of training epochs")
    parser.add_argument("--task", type=int, default=30, help="number of task")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
    parser.add_argument("--num_hidden", type=int, default=32, help="hidden_dim")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    # parser.add_argument('-EuclideanExpert1', '--EuclideanExpert1', type=bool, default=True)
    parser.add_argument('-laplace', '--laplace', type=bool, default=False)
    parser.add_argument('-Similarity1', '--Similarity1', type=bool, default=True)
    parser.add_argument('-encMoe', '--encMoe', type=bool, default=True)
    parser.add_argument('-decMoe', '--decMoe', type=bool, default=True)
    # parser.add_argument('-moran', '--moran', type=bool, default=True)
    parser.add_argument("--lambd", type=float, default=1.0, help="lambda")
    parser.add_argument('-m', '--model_name', type=str, default='MoxNP', choices=['MoxNP', 'NP'])
    parser.add_argument('-d', '--dataset', type=str, default='cali',
                        choices=['cali', 'Chengdu_housing', 'generation','generation1','generation3', 'temperature'])
    parser.add_argument('-n_context_min', '--n_context_min', type=int, default=3)
    parser.add_argument('-p', '--path', type=str, default='./')
    parser.add_argument('--checkpoint_path', type=str, default='Chengdu_housing-MoxNP--emb128-lr0.001-ep300-xuhao10_encMoe_eulid_Similar_decMoe_lam0.1_Nov07_0951', help='Path to save checkpoint')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume training from the last checkpoint if available')

    args = parser.parse_args()
    main(args)
