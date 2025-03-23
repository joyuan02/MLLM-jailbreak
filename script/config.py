import argparse

def CS_DJ_parser():
    
    parser = argparse.ArgumentParser(description="CS-DJ Configuration")
    parser.add_argument("--src_dir", type=str, default='./llava_images', help="Image directory")
    parser.add_argument("--save_embeding_path", type=str, default="./image_embedding", help="Path to save the embedding.")
    parser.add_argument("--save_map_path", type=str, default="./image_map", help="Path to save the jailbreak image map.")
    parser.add_argument("--jailbreak_folder_path", type=str, default='./instructions', help="Jialbreak raw query set.")
    parser.add_argument("--jailbreak_response_save_path", type=str, default='./results', help="Path to save the jailbreak response.")
    parser.add_argument("--distraction_image_save_path", type=str, default='./distraction_images', help="Path to save the jailbreak image.")
    parser.add_argument("--strategy_name", type=str, default='CS-DJ_best_method', help="Strategy name.")
    
    # main set up
    parser.add_argument("--seed", type=int, default=0, help="Random seed for selecting images.")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to select.")
    parser.add_argument("--object_model", type=str, default='gpt-4o-mini', choices=['gpt-4o-mini', 'gpt-4o', 'gpt-4-vision-preview'], help="Jialbreak object model.")
    parser.add_argument("--api_key", type=str, default='<your api key>', help="Openai API key.")
      
    return parser