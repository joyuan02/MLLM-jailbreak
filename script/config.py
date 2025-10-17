import argparse
import os

def CS_DJ_parser():
    parser = argparse.ArgumentParser(description="CS-DJ Configuration")

    parser.add_argument("--src_dir", type=str, default=os.getenv("SRC_DIR", "./llava_images"),
                        help="Image directory")
    parser.add_argument("--save_embeding_path", type=str, default=os.getenv("SAVE_EMBEDING_PATH", "./image_embedding"),
                        help="Path to save the embedding.")
    parser.add_argument("--save_map_path", type=str, default=os.getenv("SAVE_MAP_PATH", "./image_map"),
                        help="Path to save the jailbreak image map.")
    parser.add_argument("--jailbreak_folder_path", type=str, default=os.getenv("JAILBREAK_FOLDER_PATH", "./instructions"),
                        help="Jailbreak raw query set.")
    parser.add_argument("--jailbreak_response_save_path", type=str, default=os.getenv("JAILBREAK_RESPONSE_SAVE_PATH", "./results"),
                        help="Path to save the jailbreak response.")
    parser.add_argument("--distraction_image_save_path", type=str, default=os.getenv("DISTRACTION_IMAGE_SAVE_PATH", "./distraction_images"),
                        help="Path to save the jailbreak image.")
    parser.add_argument("--strategy_name", type=str, default=os.getenv("STRATEGY_NAME", "CS-DJ_best_method"),
                        help="Strategy name.")

    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")), help="Random seed")
    parser.add_argument("--num_images", type=int, default=int(os.getenv("NUM_IMAGES", "100")),
                        help="Number of images to select.")
    parser.add_argument("--object_model", type=str, default=os.getenv("OBJECT_MODEL", "gpt-4o"),
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-vision-preview"],
                        help="Jailbreak object model.")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""),
                        help="OpenAI API key.")

    return parser
