import argparse
import os
import random
import json
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import CS_DJ_parser


def main(src_dir, seed, num_images, save_path):

    random.seed(seed)

    img_list = os.listdir(src_dir)

    random.shuffle(img_list)
    selected_imgs_path = img_list[:num_images]

    model = SentenceTransformer('clip-ViT-L-14')

    image_embedding_list = []
    for img_path in tqdm(selected_imgs_path, desc="Processing images"):
        full_img_path = os.path.join(src_dir, img_path)
        try:
            img = Image.open(full_img_path)
            img_emb = model.encode(img)
            image_embedding_list.append({
                "img_path": full_img_path,
                "img_emb": img_emb.tolist()
            })
        except Exception as e:
            print(f"Error processing {full_img_path}: {e}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = os.path.join(save_path, f'map_seed_{seed}_num_{num_images}.json')
    with open(output_file, 'w') as f:
        json.dump(image_embedding_list, f, indent=4)
    print(f"Image embeddings saved to {output_file}")


if __name__ == "__main__":
    
    parser = CS_DJ_parser()
    args = parser.parse_args()

    main(args.src_dir, args.seed, args.num_images, args.save_embeding_path)
