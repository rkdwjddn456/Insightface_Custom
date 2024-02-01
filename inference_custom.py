import argparse
import os
import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from backbones import get_model

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    return feat

def compute_cosine_similarity(features1, features2):
    features1_flat = features1.reshape(features1.shape[0], -1)
    features2_flat = features2.reshape(features2.shape[0], -1)
    
    similarities = cosine_similarity(features1_flat, features2_flat)
    return similarities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()

    img1_dir = r"/database/kjw/lg_ree/save_crop_image"
    img2_dir = r"/home/jsh/bcw/yolov8n/ultralytics_crop2"
    
    img1_features = []
    img2_features = []

    for i in range(1, 21):
        current_path = os.path.join(img1_dir, f'predict{i}', 'crops', 'face')
        img1_features.append(inference(args.weight, args.network, os.path.join(current_path, 'image0.jpg')))
    
    similarities = []

    for i in range(1, 65):
        current_path = os.path.join(img2_dir, f'{i}.png')
        img2_features.append(inference(args.weight, args.network, current_path))
        
        similarity = compute_cosine_similarity(np.array(img1_features), np.array(img2_features[-1:]))
        similarities.append(similarity.squeeze().tolist())

    similarities = np.array(similarities)

    avg_similarities = np.mean(similarities, axis=1)
    max_similarities = np.max(similarities, axis=1)
    
    top5_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :5]
    top5_values = np.take_along_axis(similarities, top5_indices, axis=1)
    top5_max_similarities= np.mean(top5_values, axis=1)

    print(avg_similarities)
    print(max_similarities)
    print(top5_max_similarities)
        
# 등록이미지 20개 하나의 feature로 평균
    # img1_features_mean = np.mean(img1_features, axis=0)
    
    # similarities_mean = []  # 리스트로 초기화

    # for i in range(1, 65):
    #     current_path = os.path.join(img2_dir, f'{i}.png')
    #     img2_features.append(inference(args.weight, args.network, current_path))
        
    #     similarity_mean = compute_cosine_similarity(np.array(img1_features_mean), np.array(img2_features[-1:]))
    #     similarities_mean.append(similarity_mean.squeeze().tolist())

    # similarities_mean = np.array(similarities_mean)
    # print(similarities_mean)