import argparse
import cv2
import os
import glob
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc, precision_recall_curve

## 모델바꿀떄마다 바꿔야하는거 a. model_name b. name(이미지 이름) c. folder_name(폴더 이름)

def compute_cosine_similarity(features1, features2):
    features1_flat = features1.reshape(features1.shape[0], -1)
    features2_flat = features2.reshape(features2.shape[0], -1)
    
    similarities = cosine_similarity(features1_flat, features2_flat)
    return similarities

def main_evaluation_metrics(model_name, folder_name):
    # a
    f1_score20=[]
    roc_auc20=[]

    for i in range(1, 21): # 20명 다 돌릴꺼 
        img1_dir = f"/database/LGE_face/real_test1/00_{i:02}_db"   
        img2_dir = r"/database/LGE_face/real_test1/test"
        onnx_model = model_name
            
        session = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # b
        name = f"mbf_kd_ep1_top5avg_{i}db.png"

        # c
        def inference(onnx_model, img_path):
            if img_path is None:
                img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
            
            else:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (112, 112))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0).astype(np.float32)
            img /= 255.0
            img -= 0.5
            img /= 0.5

            output = session.run([output_name], {input_name: img})

            return output

        img1_features = []
        img2_features = []
        real_similarities = []

        o_dir= f'/database/LGE_face/real_test1/test/{i:02}'
        
        jpg_file_list = []
        
        o_number = 0
        x_number = 0
        index=1

        ans = {}
        path=[]

        # 이미지 파일들의 경로와 파일명을 리스트에 저장합니다.
        for root, dirs, files in os.walk(img2_dir):
            if os.path.basename(root) in ['02_110cm_robot', '04_150cm_robot', '06_200cm_robot','08_250cm_robot']: 
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        current_path = os.path.join(root, file)
                        path.append(current_path)

                        if root.startswith(o_dir):
                        # 파일이 img_dir에서 가져온 것이면 1로 표시
                            ans[current_path] = 1
                            o_number += 1
                        else:
                        # 그렇지 않으면 0으로 표시
                            ans[current_path] = 0
                            x_number +=1

                    jpg_file_list.append(f"{os.path.join(root, file)},{index}")
                    index += 1

        content = "\n".join(jpg_file_list)
        memo_file_path = "jpg_file_list.txt"

        with open(memo_file_path, 'w') as file:
            file.write(content)

        # 이미지 그룹 1에 대한 특징 벡터 가져오기
        for i in range(1, 21):
            current_path = glob.glob(os.path.join(img1_dir, '*.png'))[i-1] 
            img1_features.append(inference(onnx_model, current_path)[0][0])

        p_ans = []

        for i in path:
            current_path = i
            p_ans.append(ans[current_path])
            img2_features.append(inference(onnx_model, current_path)[0][0])

        # 현재 이미지 그룹 2 이미지의 유사도 배열 초기화  
        similarity_1 = compute_cosine_similarity(np.array([img1_features])[0], np.array([img2_features])[0])       
        
        top_5_indices = np.argsort(similarity_1,axis = 0)[-5:].T

        # Top 5 이미지 특징 벡터 및 평균 계산
        img1_features_t5 = [np.mean([img1_features[a1],img1_features[a2],img1_features[a3],img1_features[a4],img1_features[a5]], axis=0) for a1,a2,a3,a4,a5 in top_5_indices]
        
        # 이미지 그룹 1의 Top 3 평균 특징 벡터와 이미지 그룹 2 간의 유사도 계산
        for i,img2_feature in enumerate(img2_features):
            similarity = compute_cosine_similarity(np.array([img1_features_t5])[:,i], np.array([img2_feature]))
            real_similarities.append(similarity)   

        real_similarities = np.array(real_similarities)
        
        rounded_values = np.round(real_similarities, decimals=3)
        flattened_results = rounded_values.flatten().tolist()

        y_true = np.array(p_ans)
        
        y_scores = np.array(flattened_results)
        
        positive_scores = y_scores[y_true == 1]
        negative_scores = y_scores[y_true == 0]

        kde_positive = gaussian_kde(positive_scores)
        kde_negative = gaussian_kde(negative_scores)

        x_range = np.linspace(0, 1, 1000)

        density_positive = kde_positive(x_range)
        density_negative = kde_negative(x_range)

        # roc draw!
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        roc_auc_round=round(roc_auc,3)
        roc_auc20.append(roc_auc_round)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)  

        f1_scores = 2 * (precision * recall) / (precision + recall)

        max_f1_score = max(f1_scores)
        roc_max_f1_score = round(max_f1_score, 3)
        f1_score20.append(roc_max_f1_score)

        best_threshold = thresholds[np.argmax(f1_scores)]
        
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        plt.axhline(y=max_f1_score, color='r', linestyle='-', label='Max F1 Score(= %0.2f)' % max_f1_score)
        plt.axvline(x=recall[np.argmax(f1_scores)], color='g', linestyle='--', label='Best Threshold for Max F1 Score(= %0.2f)' % best_threshold)
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(x_range, density_positive, color='blue', label='Y(1)')
        plt.plot(x_range, density_negative, color='red', label='N(0)')
        plt.axvline(x=best_threshold, color='g', linestyle='--', label='Best Threshold for Max F1 Score(= %0.2f)' % best_threshold)

        plt.xlabel('cos similarity')
        plt.ylabel('density from kde')
        plt.title(f'mbf_top5avg_{i}db')
        plt.legend()

        plt.tight_layout()
        #plt.save()
        # plt.show() 

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        plt.savefig('{}/{}'.format(folder_name,name))
        plt.close()

    print('f1_score_20 list:', f1_score20)
    print('roc_auc20 list: ',roc_auc20)

    average_f1_score = np.mean(f1_score20)
    print("mean Max F1 Score:", average_f1_score)

    average_roc_auc = np.mean(roc_auc20)

    print("mean ROC Area:", average_roc_auc)

    return average_f1_score, average_roc_auc