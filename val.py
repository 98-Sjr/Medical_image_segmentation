import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.models as models
from archs import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
"""
需要指定参数：--name 
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="vessel_256_DoubleUnet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = archs.__dict__[config['arch']](3,
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])
    if config['arch'] == 'U_Net':
        model = archs.__dict__[config['arch']](config['input_channels'], config['output_channels'])
    elif config['arch'] == 'NestedUNet':
        model = archs.__dict__[config['arch']](config['output_channels'], config['input_channels'], config['deep_supervision'])
    elif config['arch'] == 'R2U_Net' :
        model = archs.__dict__[config['arch']](config['input_channels'], config['output_channels'], t=2)
    elif config['arch'] == 'AttU_Net' :
        model = archs.__dict__[config['arch']](config['input_channels'], config['output_channels'])
    elif config['arch'] == 'R2AttU_Net' :
        model = archs.__dict__[config['arch']](config['input_channels'], config['output_channels'], t=2)
    elif config['arch'] == 'DeepLabV3Plus':
        model = archs.__dict__[config['arch']](n_classes=3, n_blocks=[3, 4, 23, 3],
                                               atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=16, )
    elif config['arch'] == 'DoubleUnet':
        model = archs.__dict__[config['arch']](models.vgg19_bn(), config['input_channels'], config['output_channels'])
    else:
        raise IndexError

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.eval()
    # model.train()

    val_transform = Compose([
        albumentations.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    test_img = glob('./test/*')
    for name in test_img:
        img = cv2.imread(name)
        # 数据增强
        augmented = val_transform(image=img)  # 这个包比较方便，能把mask也一并做掉
        img = augmented['image']  # 参考https://github.com/albumentations-team/albumentations
        # 开始模型
        img = torch.from_numpy(cv2.resize(img, (int(config['input_h']), int(config['input_w']))).astype('float32') / 255, )
        img = torch.unsqueeze(img, dim=0).permute(0, 3, 1, 2).repeat(config['batch_size'], 1, 1, 1).cuda()
        out = model(img)
        out = torch.sigmoid(out)
        out = out[0, :, :, :].squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
        cv2.imwrite('./test/out_of_%s_'% config['arch'] + os.path.basename(name), out.astype('uint8'))

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    
    plot_examples(input, target, model, num_examples=3)
    
    torch.cuda.empty_cache()

def plot_examples(datax, datay, model,num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
