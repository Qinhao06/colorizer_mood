import os.path

import torch
import logging
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from colorizers import siggraph17, load_img
from colorizers.util import preprocess_img, postprocess_tens, preprocess_img_ab

# 定义一个训练函数，通过微调实现不同情绪化模型
if __name__ == '__main__':
    # 加载初始模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    colorizer_siggraph17 = siggraph17(pretrained=True).to(device)
    colorizer_siggraph17.train()
    # 加载 logger
    logging.basicConfig(filename='training.log', level=logging.INFO, filemode='w')
    n = 1000
    optimizer = torch.optim.SGD(colorizer_siggraph17.parameters(), lr=0.01)  # 优化器
    criterion = nn.CrossEntropyLoss().to(device) # 损失函数
    for i in tqdm(range(n), ncols=100):
        if os.path.exists('img_test/romantic/' + str(i + 1) + '.jpeg'):
            img = load_img('img_test/romantic/' + str(i + 1) + '.jpeg')
        else:
            img = load_img('img_test/romantic/' + str(i + 1) + '.jpg')

        tens_l_orig, tens_l_rs = preprocess_img(img, HW=(256, 256))
        tens_l_rs_a, tens_l_rs_b = preprocess_img_ab(img, HW=(256, 256))
        tens_l_rs = tens_l_rs.to(device)
        model_ab = colorizer_siggraph17(tens_l_rs)
        tens_l_rs_ab = torch.cat([tens_l_rs_a, tens_l_rs_b], dim=1).to(device)
        loss = criterion(model_ab, tens_l_rs_ab) * 10 / model_ab.numel()
        logging.info(str(i) + ":" + str(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(colorizer_siggraph17, 'model_romantic.pth')
    img = load_img('img_test/romantic/10.jpeg')
    colorizer_siggraph17.eval()
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    tens_l_rs = tens_l_rs.to(device)
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    # out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # 保存模型
    plt.imsave('siggraph17.jpg', out_img_siggraph17)
    plt.imsave('img_bw.jpg', img_bw)
