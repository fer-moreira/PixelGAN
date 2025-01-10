import os
from PIL import Image

def split_sprite_sheet():
    # Get input from user
    sprite_sheet_path = input("Enter the path to your sprite sheet: ")
    sprite_width = int(input("Enter the width of a single sprite: "))
    sprite_height = int(input("Enter the height of a single sprite: "))
    output_dir = input("Enter the output directory path (or press Enter for current directory): ").strip()
    
    # Use current directory if no output path specified
    if not output_dir:
        output_dir = "sprites_output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open the sprite sheet
        with Image.open(sprite_sheet_path) as sprite_sheet:
            # Get the full sprite sheet dimensions
            sheet_width, sheet_height = sprite_sheet.size
            
            # Calculate number of rows and columns
            num_columns = sheet_width // sprite_width
            num_rows = sheet_height // sprite_height
            
            print(f"\nFound {num_columns * num_rows} sprites in {num_rows} rows and {num_columns} columns")
            
            # Counter for naming files
            sprite_count = 0
            
            # Iterate through each row and column
            for row in range(num_rows):
                for col in range(num_columns):
                    # Calculate coordinates for cropping
                    left = col * sprite_width
                    upper = row * sprite_height
                    right = left + sprite_width
                    lower = upper + sprite_height
                    
                    # Crop the sprite
                    sprite = sprite_sheet.crop((left, upper, right, lower))
                    
                    # Save the sprite
                    output_path = os.path.join(output_dir, f"sprite_{sprite_count}.png")
                    sprite.save(output_path, "PNG")
                    print(f"Saved sprite {sprite_count} to {output_path}")
                    
                    sprite_count += 1
            
            print(f"\nSuccessfully extracted {sprite_count} sprites to {output_dir}")
            
    except FileNotFoundError:
        print("Error: Sprite sheet file not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Sprite Sheet Splitter")
    print("====================")
    split_sprite_sheet()