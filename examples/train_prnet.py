import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
import wandb

from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ops.transform_functions import DCPTransform

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
    sys.path.append(os.path.join(BASE_DIR, os.pardir))
    os.chdir(os.path.join(BASE_DIR, os.pardir))

from learning3d.models import PRNet
# from learning3d.data_utils import #RegistrationData, ModelNet40Data

COLOR_IDS = [0, 13]


def build_wandb_point_cloud(ptss, colors):
    stacked_ptss = []
    for pts, color_id in zip(ptss, colors):
        pts_colored = np.zeros(shape=(pts.shape[0], pts.shape[1] + 1))
        pts_colored[:, :-1] = pts
        pts_colored[:, -1] = color_id
        stacked_ptss.append(pts_colored)
    return np.vstack(stacked_ptss)


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp train_dcp.py checkpoints' + '/' + args.exp_name + '/' + 'train.py.backup')


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def get_transformations(igt):
    R_ba = igt[:, 0:3, 0:3]  # Ps = R_ba * Pt
    translation_ba = igt[:, 0:3, 3].unsqueeze(2)  # Ps = Pt + t_ba
    R_ab = R_ba.permute(0, 2, 1)  # Pt = R_ab * Ps
    translation_ab = -torch.bmm(R_ab, translation_ba)  # Pt = Ps + t_ab
    return R_ab, translation_ab, R_ba, translation_ba


def test_one_epoch(device, model, test_loader, submit_pts=False):
    model.eval()
    test_loss = 0.0
    count = 0
    for i, (src, dst, R_ab, t_ab) in enumerate(tqdm(test_loader)):
        src = src.type(torch.FloatTensor).to(device)
        dst = dst.type(torch.FloatTensor).to(device)
        R_ab = R_ab.type(torch.FloatTensor).to(device)
        t_ab = t_ab.type(torch.FloatTensor).to(device)

        output = model(src, dst, R_ab, t_ab)
        loss_val = output['loss']

        wandb.log({
            'test/loss': output['loss'].detach().cpu().numpy(),
            'test/loss_mse': output['loss_mse'].detach().cpu().numpy(),
        })

        if True:# and submit_pts:
            for idx in range(src.shape[0]):
                target_point_cloud = dst[idx]
                tmp = src[idx]
                predicted_point_cloud = output['transformed_source'][idx]
                point_cloud = build_wandb_point_cloud([target_point_cloud.detach().cpu().numpy(),
                                                       predicted_point_cloud.detach().cpu().numpy(),
                                                       # tmp.detach().cpu().numpy()
                                                       ],
                                                      colors=COLOR_IDS)
                wandb.log({
                    "point_clouds_{:d}/result".format(i): wandb.Object3D(point_cloud)
                })

                point_cloud = build_wandb_point_cloud([target_point_cloud.detach().cpu().numpy(),
                                                       # predicted_point_cloud.detach().cpu().numpy(),
                                                       tmp.detach().cpu().numpy()
                                                       ],
                                                      colors=COLOR_IDS)
                wandb.log({
                    "point_clouds_{:d}/orig".format(i): wandb.Object3D(point_cloud)
                })

        test_loss += loss_val.item()
        count += 1

    test_loss = float(test_loss) / count
    return test_loss


def test(args, model, test_loader, textio):
    test_loss = test_one_epoch(args.device, model, test_loader, True)
    textio.cprint('Validation Loss: %f' % (test_loss))


def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    pred = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt = data
        transformations = get_transformations(igt)
        transformations = [t.to(device) for t in transformations]
        R_ab, translation_ab, R_ba, translation_ba = transformations

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        output = model(template, source, R_ab, translation_ab.squeeze(2))
        loss_val = output['loss']
        wandb.log({
            'train/loss': output['loss'].detach().cpu().numpy(),
            'train/loss_mse': output['loss_mse'].detach().cpu().numpy(),
        })

        # forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1

    train_loss = float(train_loss) / count
    return train_loss


def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001, momentum=0.9)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train_one_epoch(args.device, model, train_loader, optimizer)
        test_loss = test_one_epoch(args.device, model, test_loader, epoch == 0)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': best_test_loss,
                    'optimizer': optimizer.state_dict(), }
            torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
            torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (args.exp_name))
        # torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/best_ptnet_model.t7' % (args.exp_name))

        torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
        torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (args.exp_name))
        # torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/ptnet_model.t7' % (args.exp_name))

        boardio.add_scalar('Train Loss', train_loss, epoch + 1)
        boardio.add_scalar('Test Loss', test_loss, epoch + 1)
        boardio.add_scalar('Best Test Loss', best_test_loss, epoch + 1)

        textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f' % (
            epoch + 1, train_loss, test_loss, best_test_loss))


def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_prnet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset')  # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--emb_dims', default=512, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--num_iterations', default=3, type=int,
                        help='Number of Iterations')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--test_batch_size', default=16, type=int,
                        metavar='N', help='test mini-batch size (default: 16)')
    parser.add_argument('--epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--input_file', nargs="+", default=['data/pc.npy'], type=str)

    args = parser.parse_args()
    return args

class RegistrationData(Dataset):
    def __init__(self, pathes, partial_source=False, partial_template=False,
                 noise=False, additional_params={}):
        super(RegistrationData, self).__init__()

        self.partial_template = partial_template
        self.partial_source = partial_source
        self.noise = noise
        self.additional_params = additional_params
        self.pathes = pathes

        self.transforms = DCPTransform(angle_range=45, translation_range=1)

    def __len__(self):
        return 12

    def __getitem__(self, index):
        path = self.pathes[0]
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            src = torch.from_numpy(data[index]['pts1'])
            dst = torch.from_numpy(data[index]['pts2'])

            R = torch.from_numpy(data[index]['T'][:3, :3])
            t = torch.from_numpy(data[index]['T'][:3, -1])
        return src, dst, R, t

def main():
    run = wandb.init(project="prnet_normalized_hack1_15", reinit=True)
    args = options()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    # testset = RegistrationData('PRNet', ModelNet40Data(train=False), partial_source=True, partial_template=True)
    testset = RegistrationData(args.input_file, partial_source=True, partial_template=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False,
                             num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create PointNet Model.
    model = PRNet(num_iters=15,
                  emb_nn='dgcnn',
                  attention='transformer',
                  head='svd',
                  emb_dims=512,
                  num_subsampled_points=768,
                  cycle_consistency_loss=0.1,
                  feature_alignment_loss=0.1,
                  discount_factor=0.9)

    model = model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    test(args, model, test_loader, textio)

    run.finish()


if __name__ == '__main__':
    main()
