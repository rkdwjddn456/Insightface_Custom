import os
import argparse
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from backbones import get_model
import matplotlib.pyplot as plt

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(400, 400, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (400, 400))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    return feat

def plot_images_and_similarity(img_list, test_img, similarity_values, save_path):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        axs[i, 0].imshow(cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2RGB))
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image {i+1}')

        axs[i, 1].imshow(cv2.cvtColor(cv2.imread(test_img), cv2.COLOR_BGR2RGB))
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Test Image')

        axs[i, 2].text(0.5, 0.5, f'Cosine Similarity: {similarity_values[i]:.4f}', size='large', ha='center', va='center', fontweight='bold')

    axs[0, 2].axis('off')
    axs[1, 2].axis('off')
    axs[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()

    img_dir = r"/database/kjw/our_pic/crop_image"
    img_path = r"/database/kjw/our_pic/test_image/predict1/crops/face/image0.jpg"
    save_path = r"/database/kjw/our_pic/save_pic.png"

    feat_list = []

    for i in range(1, 4):
        current_path = os.path.join(img_dir, f'predict{i}', 'crops', 'face')
        feat = inference(args.weight, args.network, os.path.join(current_path, 'image0.jpg'))
        feat_list.append(feat)

    feat_test = inference(args.weight, args.network, img_path)
    similarity_values = [cosine_similarity(feat_test, feat)[0, 0] for feat in feat_list]
    img_list = [os.path.join(img_dir, f'predict{i}', 'crops', 'face', 'image0.jpg') for i in range(1, 4)]
    plot_images_and_similarity(img_list, img_path, similarity_values, save_path)