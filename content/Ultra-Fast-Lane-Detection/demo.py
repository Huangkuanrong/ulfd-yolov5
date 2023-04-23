import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

# 指定测试的配置信息
backbone = '18'  # 骨干网络
dataset = '11'  # 数据集类型
griding_num = 100  # 网格数
model_path = 'content/Ultra-Fast-Lane-Detection/tusimple_18.pth'
img_path = 'content/Ultra-Fast-Lane-Detection/TUSIMPLEROOT/img'
output_path = 'content/result/ufld'


import torch
from PIL import Image
import os
import numpy as np
import cv2
import glob
import datetime


def loader_func(path):
    return Image.open(path)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, path, img_transform=None):
        super(TestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        # self.list：储存测试图片的相对路径 clips/0601/1494452577507766671/20.jpg\n

    def __getitem__(self, index):
        name = glob.glob('%s/*.jpg'%self.path)[index]
        img = loader_func(name)

        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, name

    def __len__(self):
        return len(self.list)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # 加速

    # args, cfg = merge_config()   # 用终端指定配置信息
    dist_print('start testing...')
    assert backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if dataset == 'CULane':
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        # raise NotImplementedError
        cls_num_per_lane = 56

    net = parsingNet(pretrained=False, backbone=backbone, cls_dim=(griding_num + 1, cls_num_per_lane, 4),
                                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(model_path, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    # 图像格式统一：(288, 800)，图像张量，归一化
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(img_path,os.path.join(img_path, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]

        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(img_path,os.path.join(img_path, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:  # 自定义数据集
        # raise NotImplementedError
        datasets = TestDataset(img_path, img_transform=img_transforms)
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor

    for dataset in zip(datasets): # splits：图片列表 datasets：统一格式之后的数据集
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)  # 加载数据集
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # print(split[:-3]+'avi')
        # vout = cv2.VideoWriter(split[:-3]+'avi', fourcc, 30.0, (img_w, img_h))  # 保存结果为视频文件
        for i, data in enumerate(tqdm.tqdm(loader)):  # 进度条显示进度
            imgs, names = data  # imgs:图像张量，图像相对路径：
            imgs = imgs.cuda()  # 使用GPU
            with torch.no_grad():  # 测试代码不计算梯度
                out = net(imgs)  # 模型预测 输出张量：[1,101,56,4]

            col_sample = np.linspace(0, 800 - 1, griding_num)
            col_sample_w = col_sample[1] - col_sample[0]


            out_j = out[0].data.cpu().numpy()  # 数据类型转换成numpy [101,56,4]
            out_j = out_j[:, ::-1, :]  # 将第二维度倒着取[101,56,4]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # [100,56,4]softmax 计算（概率映射到0-1之间且沿着维度0概率总和=1）
            idx = np.arange(griding_num) + 1  # 产生 1-100
            idx = idx.reshape(-1, 1, 1)  # [100,1,1]
            loc = np.sum(prob * idx, axis=0)  # [56,4]
            out_j = np.argmax(out_j, axis=0)  # 返回最大值的索引
            loc[out_j == griding_num] = 0  # 若最大值的索引=griding_num，归零
            out_j = loc  # [56,4]

            # import pdb; pdb.set_trace()
            coordination_save = []
            # img_path = os.path.join(img_path,names[0])
            img_path = img_path + "/road.jpg"
            vis = cv2.imread(img_path)  # 读取图像 [720,1280,3]
            cv2.imshow('img',vis)
            for i in range(out_j.shape[1]):  # 遍历列
                if np.sum(out_j[:, i] != 0) > 2:  # 非0单元格的数量大于2
                    sum1 = np.sum(out_j[:, i] != 0)
                    for k in range(out_j.shape[0]):  # 遍历行
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
                            
                            coordination_save.append(ppp)
            
            # Get the current time
            current_time = datetime.datetime.now()

            # Convert the time to a string in a desired format
            time_string = current_time.strftime("%Y%m%d_%H%M%S")
            
            save_file_basic = output_path + "/result_" + time_string
            save_file_txt = output_path + "/result_" + time_string + ".txt"
            save_file_jpg = output_path + "/result_" + time_string + ".jpg"
            
            # 保存检测结果图
            with open(save_file_txt, 'w', encoding='utf-8') as f:
                f.write('\n'.join(f'{tup[0]} {tup[1]}' for tup in coordination_save))
            
            with open(output_path + "/result.txt", 'w', encoding='utf-8') as f:
                f.write(save_file_basic)

            cv2.imwrite(save_file_jpg, vis)