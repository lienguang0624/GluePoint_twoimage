import glob
import numpy as np
import os
import time
from torch import nn
import cv2
import torch
from copy import deepcopy
from pathlib import Path

# 添加参数 键入权重，默认使用superpoint_v1.pth
weights_path = 'models\\weights\\superpoint_v1.pth'
# 如果指定了图像目录，则glob匹配
img_glob = '*.png'
# 如果输入是电影或目录，则要跳过的图像
skip = 1
# 显示额外的调试输出
show_extra = False
H = 120
W = 160
# 缩放输出可视化的因素
display_scale = 5
# 点轨迹的最小长度
min_length = 2
# 点轨迹的最大长度
max_length = 5
# 非最大抑制（NMS）距离
nms_dist = 4
# 检测器置信度阈值
conf_thresh = 0.015
# 描述符匹配阈值
nn_thresh = 0.7
# OpenCV网络摄像头视频捕获ID，通常为0或1
camid = 0
# 使用cuda GPU加快网络处理速度
cuda = True
# 不要在屏幕上显示图像。 如果远程运行很有用
no_display = False
# 将输出帧保存到目录
write = True
# 写入输出框架的目录
write_dir = 'tracker_outputs/'
# 如果输入是电影或目录，则要跳过的图像(步长)
skip = 1
# 由SuperGlue执行的Sinkhorn迭代次数
sinkhorn_iterations = 20
# SuperGlue匹配阈值
match_threshold = 0.2
# superglue 权重，可选项'indoor', 'outdoor'
superglue = 'indoor'
# 显示检测到的关键点
show_keypoints = True


# 字体参数可视化
font = cv2.FONT_HERSHEY_DUPLEX
font_clr = (255, 255, 255)
font_pt = (4, 12)
font_sc = 0.4

class SuperPointNet(torch.nn.Module):
  """ Pytorch definition of SuperPoint Network. """
  def __init__(self):
    super(SuperPointNet, self).__init__()
    self.relu = torch.nn.ReLU(inplace=True)
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
    # 共享的编码器。
    self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
    self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
    self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
    self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
    self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
    self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
    self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
    self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
    # 探测器的头。
    self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
    # 描述符。
    self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
    self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    """ 联合计算未处理点和描述符张量的前向传递。
    Input
      x: 形似N x 1 x H x W的Pytorch张量。
    Output
      semi: 输出点形制为N×65×H/8×W/8的Pytorch张量。
      desc: 输出描述符形状为N x 256 x H/8 x W/8的pytorch张量。
    """
    # Shared Encoder.
    x = self.relu(self.conv1a(x))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))
    # Detector Head.
    cPa = self.relu(self.convPa(x))
    semi = self.convPb(cPa)
    # Descriptor Head.
    cDa = self.relu(self.convDa(x))
    desc = self.convDb(cDa)
    dn = torch.norm(desc, p=2, dim=1) # 归一化.
    desc = desc.div(torch.unsqueeze(dn, 1)) # 除以范数使之归一化
    return semi, desc

class SuperPointFrontend(object):
  """ 包装pytorch网络，以帮助前后图像处理。 """
  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):
    self.name = 'SuperPoint'
    self.cuda = cuda
    self.nms_dist = nms_dist# 非最大抑制（NMS）距离
    self.conf_thresh = conf_thresh# 检测器置信度阈值
    self.nn_thresh = nn_thresh # L2描述符距离良好匹配。
    self.cell = 8 # 每个输出单元的大小。保持固定的。
    self.border_remove = 4 # 移除边界附近的点。

    # Load the network in inference mode.
    self.net = SuperPointNet()
    if cuda:
      # Train on GPU, deploy on GPU.
      self.net.load_state_dict(torch.load(weights_path))
      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage))
    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):
    """
    运行一个更快的近似非最大抑制numpy角落形状:
      3xN [x_i,y_i,conf_i]^T

    Algo摘要:创建一个网格大小的HxW。每个角的位置赋值为1，其余为0。遍历所有的1并将它们转换为-1或0。通过将附近的值设置为0来抑制点。

    网格值图例：
    -1 : 保持.
     0 : 空或抑制.
     1 : 待处理(转换为保留或supressed)

    注意:NMS的第一轮将指向整数，因此NMS距离可能不是完全的dist_thresh。它还假设点在图像边界内。

    Inputs
      in_corners - 3xN numpy数组与角落[x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh -要压制的距离，以无穷大范数距离度量。
    Returns
      nmsed_corners - 3xN numpy矩阵与幸存的角落。
      nmsed_inds - N长度numpy向量与幸存的角落指标。
    """
    grid = np.zeros((H, W)).astype(int)  # 跟踪NMS数据。
    inds = np.zeros((H, W)).astype(int)  # 存储点的索引。
    # 按置信度排序并四舍五入到最接近的整数。
    inds1 = np.argsort(-in_corners[2, :]) # inds1是按置信度排序后返回的位置序列，大小为（390，）
    corners = in_corners[:, inds1] # 按上述序列对角点进行排序，得到的corners还是(3,390)
    rcorners = corners[:2, :].round().astype(int)  # 将特征点去掉置信度并将浮点转换为整数(2,390)
    # 检查0或1个角的边缘情况。
    if rcorners.shape[1] == 0:
      return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
      return out, np.zeros((1)).astype(int)
    # 初始化网格。
    for i, rc in enumerate(rcorners.T):# i为序号，rc为坐标
      grid[rcorners[1, i], rcorners[0, i]] = 1 # 存疑
      inds[rcorners[1, i], rcorners[0, i]] = i
    # 填充网格的边界，这样我们就可以在边界附近设置NMS点。
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # 遍历从高到低的conf点，抑制邻居。
    count = 0
    for i, rc in enumerate(rcorners.T):
      # 占顶部和左侧填充。
      pt = (rc[0] + pad, rc[1] + pad)
      if grid[pt[1], pt[0]] == 1:  # 如果尚未压制。
        grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # 获取所有尚存的-1，并返回剩余角的排序数组。
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx] # 获得所有留下来的特征点的序列数
    out = corners[:, inds_keep]
    values = out[-1, :]# 所有角点作为角点的可能性
    inds2 = np.argsort(-values)# 排序
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):
    """ 处理一个小图像，以提取点和描述符.
    Input
      img - HxW numpy float32输入图像，范围为 [0,1].
    Output
      corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      desc -  对应单元标准化描述符的256xN numpy数组.
      heatmap - HxW numpy heatmap 点置信度范围[0, 1].
      """

    assert img.ndim == 2, '图片必须为灰度.'
    assert img.dtype == np.float32, '图片必须为float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    # inp <class 'torch.Tensor'> torch.Size([1, 1, 120, 160])
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # 网络正向传递
    outs = self.net.forward(inp)
    # semi <class 'torch.Tensor'> torch.Size([1, 65, 15, 20])
    # coarse_desc <class 'torch.Tensor'> torch.Size([1, 256, 15, 20])
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # 对特征点进行处理
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    # nodust (64, 15, 20)
    nodust = dense[:-1, :, :]
    # 重塑形状以获得完整分辨率的热图。
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    # nodust (15, 20, 64)
    nodust = nodust.transpose(1, 2, 0)
    # heatmap (15, 20, 8, 8)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    # heatmap (15, 8, 20, 8)
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    # heatmap (120, 160)
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # 只有大于置信度阈值的才能留下来,返回所在位置坐标
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]# 每一列三行，分别是x坐标，y坐标，属于特征点的概率
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # 使用非极大值抑制
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # 按置信度按递减排列
    # 移除边界上的目标
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]# pts (3,53)
    # --- Process descriptor.
    # coarse_desc  torch.Size([1, 256, 15, 20])
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # 使用2D点位置插入描述符图。
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1. # 转换为-1到1之间
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous() # 转换为（53,2）
      samp_pts = samp_pts.view(1, 1, -1, 2)# 转换为torch.Size([1, 1, 53, 2])
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts) # desc torch.Size([1, 256, 1, 53])
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :] #np.linalg.norm 求范数
    return pts, desc, heatmap

def create_output(write):
    # 根据需要创建输出目录。
    if write:
        print('==> Will write outputs to %s' % write_dir)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)


def nn_match_two_way(desc1, desc2, nn_thresh):
    """
    对两组描述符执行双向最近邻居匹配，以使描述符A-> B的NN匹配必须等于B-> A的NN匹配。

    Inputs:
      desc1 - N个对应的M维描述符的NxM numpy矩阵。
      desc2 - N个对应的M维描述符的NxM numpy矩阵。
      nn_thresh - 可选的描述符距离，低于此距离是一个很好的匹配。

    Returns:
      matches - L个匹配的3xL numpy数组,其中L <= N，and每列 i 是两个描述符的匹配，d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # 计算L2距离。 容易，因为向量是单位归一化的。
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1)) # (375,412)
    #获取NN索引和分数。
    idx = np.argmin(dmat, axis=1)# 每列里最小的数的所在位置，（375，）
    scores = dmat[np.arange(dmat.shape[0]), idx] # 获得对应idx所在位置的数值，（375，）
    # NN匹配的阈值。
    keep = scores < nn_thresh
    # 确认双向都是最佳匹配。
    idx2 = np.argmin(dmat, axis=0)# 每行里最小的数的所在位置，（375，）
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # 获取剩下的点的索引。
    m_idx1 = np.arange(desc1.shape[1])[keep]# 可匹配的点的序列
    m_idx2 = idx# 可匹配的点的序列，与上面相对应
    # 填充最终的3xN匹配数据结构。
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

def read_image(impath, img_size):
    """ 读取图像为灰度并调整为img_size。
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    grayim = cv2.imread(impath, 0)
    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

def MLP(channels: list, do_bn=True): #MLP([3, 32, 64, 128, 256, 256])
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class KeypointEncoder(nn.Module):
    """ 使用MLP对视觉外观和位置进行联合编码"""
    '''
    encoder中是这样的
        Sequential(
      (0): Conv1d(3, 32, kernel_size=(1,), stride=(1,))
      (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv1d(32, 64, kernel_size=(1,), stride=(1,))
      (4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): Conv1d(64, 128, kernel_size=(1,), stride=(1,))
      (7): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): ReLU()
      (9): Conv1d(128, 256, kernel_size=(1,), stride=(1,))
      (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): ReLU()
      (12): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
    )
    Loaded SuperGlue model ("indoor" weights) 
    '''
    def __init__(self, feature_dim, layers): # feature_dim 256 layers layers [32, 64, 128, 256]
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim]) #MLP([3, 32, 64, 128, 256, 256]) MLP这个函数就是根据输入构建全连接编码器
        nn.init.constant_(self.encoder[-1].bias, 0.0) # self.encoder[-1]是Conv1d(256, 256, kernel_size=(1,), stride=(1,))  对self.encoder[-1]的偏置初始化为0

    def forward(self, kpts, scores):#kpts (1,339,2) scores (1,339)
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]#unsqueeze()函数在目标位置添加一个维度 .transpose改变索引顺序 score从[1,418]变为[1,1,418],kpts从([1,428,2])变为([1,2,418])
        return self.encoder(torch.cat(inputs, dim=1))# 横向进行拼接,从之前的（1,2,339）与（1,1,339）合成为（1,3,339）return返回的是([1, 256, 339]) 正好对应3为输入，256为输出

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads #dim=256/4=64
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):# 输入分别是256和4
        super().__init__()
        '''
        (attn): MultiHeadedAttention(
          (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          )
        )
        '''
        self.attn = MultiHeadedAttention(num_heads, feature_dim)# 对MultiHeadedAttention进行初始化，并命名为attn
        '''
        (mlp): Sequential(
          (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
          (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
        )
        '''
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    '''
    (0): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
          (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
        )
      )
    '''
    def __init__(self, feature_dim: int, layer_names: list):# 看看是怎么进行初始化的 feature_dim描述子位数（256）layer_names是['self', 'cross'] * 9
        super().__init__()

        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):# 两个([1, 256, 339])作为输入
        for layer, name in zip(self.layers, self.names): # 总共有18组layers  ['self', 'cross'] * 9 每一组layer都是上面···里的
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1) # delta0 [1, 256, 339]
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1# desc0 [1, 256, 339]

def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ 在log-space中执行Sinkhorn归一化以保证稳定性"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    # scores([1, 339, 350])
    # alpha tensor(2.3457, requires_grad=True)
    # iters 100
    """ 在log-space中执行可微的最优传输以保证稳定性"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    # ms tensor(339.)
    # ns tensor(350.)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)# torch.Size([1, 339, 1]) 内容全是 2.3
    bins1 = alpha.expand(b, 1, n)# torch.Size([1, 1, 350])
    alpha = alpha.expand(b, 1, 1)# torch.Size([1, 1, 1])
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)# torch.Size([1, 340, 351])
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # 概率乘以M+N torch.Size([1, 340, 351])
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    给定两组关键点和位置，我们通过:
      1. 关键点编码(归一化+视觉特征与位置融合)
      2. 具有多个自关注层和交叉关注层的图神经网络
      3. 最终投影层
      4. 最优传输层(匈牙利可微匹配算法)
      5. 基于互斥性和匹配阈值的阈值矩阵

    对应id使用-1表示不匹配的点。

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'models/weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(path))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """对一对关键字和描述符运行SuperGlue"""
        desc0, desc1 = data['descriptors_left'], data['descriptors_right']
        kpts0, kpts1 = data['keypoints_left'], data['keypoints_right']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # 坐标居中且归一化，不知道要干啥.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint 多层感知机 将特征点坐标与分数作为输入
        # 将特征点位置(1,339,2)与作为特征点可能性(1,339)进行合成得到（1,3,339）送入kenc进行卷积得到([1, 256, 339])，
        # 与desc([1, 256, 339])合并在一起进行描述
        desc0 = desc0 + self.kenc(kpts0, data['scores0']) # (1,256,339) + kenc((1,339,2),(1,339)) kenc((1,339,2),(1,339))得到([1, 256, 339])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # 多层转移网络.
        desc0, desc1 = self.gnn(desc0, desc1)# 两个([1, 256, 339])作为输入 返回两个([1, 256, 339])

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)# 分别对各自的desc进行处理，都是256进去，256出来

        # 计算匹配描述符距离。
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)# 由torch.Size([1, 256, 339]) torch.Size([1, 256, 350])变为torch.Size([1, 339, 350])
        scores = scores / self.config['descriptor_dim']**.5
        # 运行最优传输。.
        scores = log_optimal_transport(
            # scores torch.Size([1, 339, 350])
            # self.bin_score tensor(2.3457, requires_grad=True)
            # iters 100
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations']) # 返回值 torch.Size([1, 340, 351])

        # 获取得分高于“match_threshold”的匹配
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

def match_two_img(match_num):
    '''
    Abstract:
    Match keypoint and draw on the board
    Input:
    1.match_num:the number of keypoint needed match
    '''
    i = 0
    for situation_array_left,situation_array_right,score in match.T:
        i += 1
        if(i == match_num):# 设置预计匹配特征点数目
            break
        situation_array_left = [int(situation_array_left)] #可以绘制的特征点的索引
        situation_left = pts_left.T[situation_array_left] #可以绘制的点的坐标和heatmap数值(1,3)

        situation_array_right = [int(situation_array_right)] #可以绘制的特征点的索引
        situation_right = pts_right.T[situation_array_right] #可以绘制的点的坐标和heatmap数值(1,3)

        point_left = (int(situation_left[0][0]), int(situation_left[0][1]))
        point_right = (int(situation_right[0][0]) + 640, int(situation_right[0][1]))

        cv2.circle(imgs, point_left, 1, (0, 255, 0), -1, lineType=16)# 画点
        cv2.circle(imgs, point_right, 1, (0, 255, 0), -1, lineType=16)  # 画点
        cv2.line(imgs, point_left, point_right, (0, 0, 255))

def image2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


if __name__ == '__main__':

    print('==> Loading iamge.')

    img_left = read_image("assets\\freiburg_sequence\\1.png",[480,640])
    img_right = read_image("assets\\freiburg_sequence\\2.png",[480,640])
    # 创建SuperPoint神经网络并载入预先训练好的权值。
    print('==> Loading pre-trained network.')
    fe = SuperPointFrontend(weights_path=weights_path,
                            nms_dist=nms_dist,
                            conf_thresh=conf_thresh,
                            nn_thresh=nn_thresh,
                            cuda=cuda)
    print('==> Successfully loaded pre-trained network.')

    create_output(write)

    pts_left, desc_left, heatmap_left = fe.run(img_left) #desc (256, 375) pts (3, 375) 放大了之后 375原来是59，随图像W，H的改变而改变
    pts_right, desc_right, heatmap_right = fe.run(img_right)  # desc (256, 375) pts (3, 375) 放大了之后 375原来是59，随图像W，H的改变而改变

    # # 单纯使用score进行匹配
    # match = nn_match_two_way(desc_left, desc_right, nn_thresh) #返回[3,n],[0,n]是一图中留下的特征点的序列，[1,n]是二图中留下的特征点的序列，[2,n]是匹配相似度的概率
    # match = match.T[np.lexsort(match)].T # 按match中的score降序排列

    tensor_img_left = image2tensor(img_left,'cuda')
    tensor_img_right = image2tensor(img_right, 'cuda')

    data = {}
    data['keypoints_left'] = torch.Tensor(pts_left[0:2].T).unsqueeze(0)
    data['scores0'] = torch.Tensor(pts_left[2]).unsqueeze(0)
    data['descriptors_left'] = torch.Tensor(desc_left).unsqueeze(0)
    data['image0'] = tensor_img_left
    data['image1'] = tensor_img_right
    data['keypoints_right'] = torch.Tensor(pts_right[0:2].T).unsqueeze(0)
    data['scores1'] = torch.Tensor(pts_right[2]).unsqueeze(0)
    data['descriptors_right'] = torch.Tensor(desc_right).unsqueeze(0)

    config = {
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    superglue = SuperGlue(config.get('superglue', {}))
    pred = superglue.forward(data)

    kpts0 = data['keypoints_left'][0].cpu().numpy()
    kpts1 = data['keypoints_right'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    #confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    out_img_left = (np.dstack((img_left, img_left, img_left)) * 255.).astype('uint8')
    out_img_right = (np.dstack((img_right, img_right, img_right)) * 255.).astype('uint8')
    imgs = np.hstack([out_img_left, out_img_right]) # 将两幅图像在一个窗口中显示

    #out = make_matching_plot_fast(img_left, img_right, kpts0, kpts1, mkpts0, mkpts1, path=None, show_keypoints=show_keypoints)
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        cv2.line(imgs, (x0, y0), (x1 + img_left.shape[1], y1),
                 color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # 将行端点显示为圆
        cv2.circle(imgs, (x0, y0), 2, (255,0,0), -1, lineType=cv2.LINE_AA)
        cv2.circle(imgs, (x1 + img_left.shape[1], y1), 2, (255,0,0), -1,lineType=cv2.LINE_AA)





    # # 我的匹配方案
    # match_two_img(250)

    cv2.imshow('imgs',imgs)
    cv2.waitKey(-1)



