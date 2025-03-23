import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

from config import CS_DJ_parser


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    file_list = os.listdir(args.jailbreak_folder_path)
    jailbreak_question_list = []

    for file in file_list:
        with open(os.path.join(args.jailbreak_folder_path, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        jailbreak_question_list.extend([item['instruction'] for item in data])

    with open(os.path.join(args.save_embeding_path, f'map_seed_{args.seed}_num_{args.num_images}.json'), 'r', encoding='utf-8') as f:
        embeding_data = json.load(f)

    image_embeddings = [item['img_emb'] for item in embeding_data]
    image_paths = [item['img_path'] for item in embeding_data]

    image_embeddings = torch.tensor(image_embeddings).to(device)
    model = SentenceTransformer('clip-ViT-L-14').to(device)

    results = {}
    for jailbreak_question in tqdm(jailbreak_question_list, desc="Processing questions"):
        max_distance_embedding_list = []
        selected_image_list = []

        text_emb = model.encode(jailbreak_question, convert_to_tensor=True).to(device)
        max_distance_embedding_list.append(text_emb)

        for _ in range(15):
            combined_emb = torch.vstack(max_distance_embedding_list)
            cos_scores = util.cos_sim(combined_emb, image_embeddings).mean(dim=0) 

            min_score, min_index = torch.min(cos_scores, dim=0)
            selected_image_list.append(image_paths[int(min_index)])
            max_distance_embedding_list.append(image_embeddings[min_index])

        results[jailbreak_question] = selected_image_list

    if not os.path.exists(args.save_map_path):
        os.makedirs(args.save_map_path)
    with open(os.path.join(args.save_map_path, f'distraction_image_map_seed_{args.seed}_num_{args.num_images}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {os.path.join(args.save_map_path, f'distraction_image_map_seed_{args.seed}_num_{args.num_images}.json')}")

if __name__ == "__main__":

    parser = CS_DJ_parser()
    args = parser.parse_args()

    main(args)
