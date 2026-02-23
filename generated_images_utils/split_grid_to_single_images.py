import os
from PIL import Image

def split_images_into_tiles(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tile_id = 33
    
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

    if not files:
        print("No PNG files found in the input directory.")
        return

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        
        try:
            with Image.open(input_path) as img:
                width, height = img.size
                
                tile_width = width // 3
                tile_height = height // 3
                
                for row in range(3):
                    for col in range(3):
                        left = col * tile_width
                        upper = row * tile_height
                        
                        right = width if col == 2 else (col + 1) * tile_width
                        lower = height if row == 2 else (row + 1) * tile_height
                        
                        tile = img.crop((left, upper, right, lower))
                        output_filename = f"{tile_id}.png"
                        tile.save(os.path.join(output_dir, output_filename))
                        
                        tile_id += 1
                        
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nTask completed! Total tiles created: {tile_id - 1}")

INPUT_FOLDER = "../generated"
OUTPUT_FOLDER = "../generated"

split_images_into_tiles(INPUT_FOLDER, OUTPUT_FOLDER)