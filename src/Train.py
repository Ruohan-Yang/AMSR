import os
import shutil
import torch
from tqdm import trange

def run_model(train_loader, valid_loader, test_loader, model, args, device, log):
    model_path = 'save/'
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    best_valid_metric = 0
    best_valid_dir = ''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in trange(args.epochs, desc='Epoch'):

        model.train()
        for data in train_loader:
            data = data[0]
            whole_loss = model.loss(data,device)
            whole_loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        model.eval()
        valid_acc, valid_pre, valid_f1, valid_auc = model.metrics_eval(valid_loader, device)
        # write_infor = "Valid acc:{:.4f}, Valid pre:{:.4f}, Valid f1:{:.4f}, Valid auc:{:.4f} ".format(valid_acc, valid_pre, valid_f1, valid_auc)
        # print(write_infor)

        # Updated the best AUC, Save the best model
        if valid_auc > best_valid_metric:
            best_valid_metric = valid_auc
            best_valid_dir = model_path + 'model' + str(epoch) + '.pth'
            torch.save(model, best_valid_dir)

    # loading the best model
    model.eval()
    print('Load best model ...')
    write_infor = 'Best model: ' + best_valid_dir
    print(write_infor)
    model = torch.load(best_valid_dir)

    acc, pre, f1, auc = model.metrics_eval(test_loader, device)
    write_infor = "Test acc:{:.4f}, pre:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(acc, pre, f1, auc)
    print(write_infor)
    log.write('\n' + write_infor)

