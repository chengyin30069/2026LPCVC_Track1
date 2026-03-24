import torch
from torchvision.datasets import CocoCaptions
from transformers import CLIPTokenizer
import numpy as np
from tqdm import tqdm

EVA_BATCH = 256

def get_batch_recall(text_pair, image_list, clip_model, st_id, device, batch_size=32):
    tokens_pair = []
    output_image = []
    
    # inference text
    with torch.no_grad():
        for i in range(0, len(text_pair), batch_size):
            tokens_batch = torch.cat([text_pair[j][0] for j in range(i, min(i + batch_size, len(text_pair)))], dim=0)
            text_features = clip_model.encode_text(tokens_batch)
            text_features /= text_features.norm(dim=1, keepdim=True)
            for j in range(i, min(i + batch_size, len(text_pair))):
                tokens_pair.append((text_features[j - i].cpu().numpy(), text_pair[j][1]))

    with torch.no_grad():
        for i in range(0, len(image_list), batch_size):
            img_batch = torch.cat(image_list[i:i + batch_size], dim=0)
            image_features = clip_model.encode_image(img_batch)
            image_features /= image_features.norm(dim=1, keepdim=True)
            output_image.extend(image_features.cpu().numpy())
    
    recall_top10 = 0

    output_image = np.array(output_image)
    output_text = np.array([x[0] for x in tokens_pair]).T
    text_tensor = torch.from_numpy(output_text).to(device)
    image_tensor = torch.from_numpy(output_image).to(device)
    sim_matrix = torch.matmul(image_tensor, text_tensor).cpu().numpy()

    for i in range(len(output_image)):
        simm_pair = [[sim_matrix[i, j], text_img_id] for j, (_, text_img_id) in enumerate(tokens_pair)]
        simm_pair.sort(key=lambda x: x[0], reverse=True)
        top_k = simm_pair[:10]
        cnt = 0
        for _, text_img_id in top_k:
            if text_img_id == i + st_id:
                cnt += 1
        recall_top10 += cnt / 10
    return recall_top10 / len(output_image)

def evaluate_coco(clip_model, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    clip_model.to(device)
    clip_model.eval()
    image_path = './coco2017/val2017'
    cap_json = "./coco2017/annotations/captions_val2017.json"
    ds = CocoCaptions(root=image_path, annFile=cap_json)

    image_list = []
    text_pair = []

    batch_cnt = 0

    recall = 0
    for st_id in tqdm(range(0, len(ds), EVA_BATCH), desc="Evaluating"):
        if(st_id + EVA_BATCH > len(ds)):
            break
            
        batch_cnt += 1
        image_list = []
        text_pair = []
        for i in range(st_id, st_id + EVA_BATCH):
            
            img, caps = ds[i]
            img = np.array(img.convert('RGB').resize((224, 224)))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
            img_tensor = torch.from_numpy(img).to(torch.float32).to(device)
            image_list.append(img_tensor)
            for prompt in caps:
                tokens = tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                )["input_ids"].to(torch.int32).to(device)
                text_pair.append([tokens, i])
        recall += get_batch_recall(text_pair, image_list, clip_model, st_id, device, batch_size)
    
    print("Final Recall@10: {:.4f}".format(recall / batch_cnt))
    
    