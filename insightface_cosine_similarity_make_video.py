import os
import argparse
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    args = parser.parse_args()

    img_dir1 = r"/database/kjw/lg_ree/save_crop_image"  # start path
    img_dir2 = r"/database/kjw/ree1/save_crop"  # test path
    video_output_path = r'/database/kjw/lg_video/output_video.avi'
    feat1_list = []
    feat2_list = []

    similarities = []

    fps = 1.0
    video_size = (1280, 720)

    img_list1 = []
    current_paths_1 = []
    for i in range(1, 21):
        current_path_1 = os.path.join(img_dir1, f'predict{i}', 'crops', 'face')
        current_paths_1.append(current_path_1)
        img_path = os.path.join(current_path_1, 'image0.jpg')
        img_list1.append(cv2.imread(img_path))

    img_list2 = []
    current_paths_2 = []
    for i in range(1, 362):
        current_path_2 = os.path.join(img_dir2, f"predict{i}", "crops", "face")
        current_paths_2.append(current_path_2)
        for folder_path, _, file_names in os.walk(current_path_2):
            for file_name in file_names:
                if file_name.endswith(".jpg"):
                    file_path = os.path.join(folder_path, file_name)
                    img_list2.append(file_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, video_size)

    for (current_path_1, current_path_2, img1) in zip(current_paths_1, current_paths_2, img_list1):
        feat1 = inference(args.weight, args.network, os.path.join(current_path_1, 'image0.jpg'))
        feat1_list.append(feat1)

        for img_path in img_list2:
            feat2 = inference(args.weight, args.network, img_path)  
            feat2_list.append(feat2)

            similarity = cosine_similarity(feat1, feat2)

            img2_path = img_path
            img2 = cv2.imread(img2_path)
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            font_size = 0.8
            font_thickness = 1
            font_color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            text_position = (10, 30)
            text = f"Similarity: {similarity[0][0]:.4f}"

            img2 = cv2.putText(img2, text, text_position, font, font_size, font_color, font_thickness)

            result_image = np.concatenate((img1, img2), axis=1)
            result_image = cv2.resize(result_image, video_size)

            video_writer.write(result_image)

    video_writer.release()
    cv2.destroyAllWindows()