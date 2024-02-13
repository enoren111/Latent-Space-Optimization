import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
from torch import optim
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='ResNet50_Stride',
                        help='VGG / InceptionV3/ ResNet50/ ResNet50_Stride')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='./../')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--eval_freq_iter', type=int, default=120)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--last_stride', type=int, default=1)

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    model = FGSBIR_Model(hp)
    model.to(device)

    model.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)
    resnet = torchvision.models.resnet50(pretrained=False)
    state_dict = torch.load(r"resnet50-19c8e357.pth")
    resnet.load_state_dict(state_dict)
    new_state_dict = resnet.state_dict()
    op = model.state_dict()

    print(len(new_state_dict.keys()))
    print(len(op.keys()))

    for new_state_dict_num, new_state_dict_value in enumerate(new_state_dict.values()):
        for op_num, op_key in enumerate(op.keys()):
            if op_num == new_state_dict_num and op_num <= 317:
                op[op_key] = new_state_dict_value
    model.load_state_dict(op)

    step_count, top1, top10 = -1, 0, 0

    writer = SummaryWriter(log_dir='./log/stride_new')
    for i_epoch in range(hp.max_epoch):
        if 0 <= i_epoch < 10:
            model.sample_train_params = model.sample_embedding_network.parameters()
            model.optimizer = optim.Adam(model.sample_train_params, 0.0001 * (i_epoch + 1) / 10)
        if 10 <= i_epoch < 40:
            model.sample_train_params = model.sample_embedding_network.parameters()
            model.optimizer = optim.Adam(model.sample_train_params, 0.0001)
        if 40 <= i_epoch < 70:
            model.sample_train_params = model.sample_embedding_network.parameters()
            model.optimizer = optim.Adam(model.sample_train_params, 0.00001)
        if 70 <= i_epoch < 120:
            model.sample_train_params = model.sample_embedding_network.parameters()
            model.optimizer = optim.Adam(model.sample_train_params, 0.000001)
        batch_loss = []
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)
            batch_loss.append(loss)

            if step_count % hp.print_freq_iter == 0:
                print(
                    'Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                    (i_epoch, step_count, loss, top1, top10, time.time() - start))

            if step_count % hp.eval_freq_iter == 0:
                writer.add_scalar(tag='TrainLoss', scalar_value=sum(batch_loss) / len(batch_loss),
                                  global_step=i_epoch)  # 【关键代码2】
                with torch.no_grad():
                    top1_eval, top10_eval = model.evaluate(dataloader_Test)
                    print('results : ', top1_eval, ' / ', top10_eval)
                    writer.add_scalars(main_tag='acc_test', tag_scalar_dict={'acc_1': top1_eval,
                                                                             'acc_10': top10_eval},
                                       global_step=step_count)  # 【关键代码3】

                if top1_eval > top1:
                    torch.save(model.state_dict(),
                               hp.backbone_name + '_no_warmup_' + hp.dataset_name + '_model_best.pth')
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')
                    writer.add_scalars(main_tag='acc_update', tag_scalar_dict={'acc_1_up': top1,
                                                                               'acc_10_up': top10},
                                       global_step=step_count)
    writer.close()
