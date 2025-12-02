# generate_synthetic_tags.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
import os
from codecarbon import EmissionsTracker
import json
from datetime import datetime

class SyntheticTagGenerator:
    """
    Generate synthetic clothing tag images based on real examples
    """
    
    def __init__(self):
        # Common tag materials/compositions (based on your images)
        self.materials = [
            # Single material
            "100% Cotton", "100% Polyester", "100% Acrylic", "100% Wool",
            "100% Cashmere", "100% Silk", "100% Linen", "100% Nylon",
            
            # Two materials
            "60% Cotton 40% Polyester", "80% Cotton 20% Polyester",
            "95% Cotton 5% Elastane", "93% Lyocell 7% Spandex",
            "50% Merino Wool 50% Acrylic", "70% Viscose 30% Linen",
            "80% Polyester 20% Nylon", "97% Polyester 3% Spandex",
            
            # Three materials
            "48% Viscose 31% Polyester 18% Nylon 3% Elastane",
            "60% Cotton 35% Polyester 5% Spandex",
            "70% Rayon 25% Nylon 5% Spandex",
            
            # Multi-part (Body/Lining)
            "Body: 100% Polyester Lining: 100% Polyester",
            "Body: 53% Cotton 47% Polyester Rib: 49% Cotton 49% Polyester 2% Spandex",
            "80% Silk 20% Nylon CA 23725",
        ]
        
        # Countries
        self.countries = [
            "China", "Vietnam", "Bangladesh", "Guatemala", "India", "USA",
            "Cambodia", "Turkey", "Indonesia", "Pakistan", "Honduras", "Canada",
            "Sri Lanka", "El Salvador", "Mexico", "Peru", "Madagascar"
        ]
        
        # Care instructions
        self.care_instructions = [
            "Machine wash cold, tumble dry low",
            "Machine wash cold, do not bleach, tumble dry low",
            "Hand wash cold, do not bleach, lay flat to dry",
            "Hand wash cold separately, non-chlorine bleach only",
            "Machine wash cold, gentle cycle, with like colors, non-chlorine bleach only, tumble dry low, warm iron as needed",
            "Machine wash cold, do not bleach, tumble dry low, cool iron",
            "Hand wash cold, reshape while damp, dry flat, cool iron only if needed or dry clean",
            "Dry clean only",
            "Machine wash cold separately, gentle cycle, do not bleach, tumble dry low, heat cool iron",
        ]
        
        # Tag backgrounds (fabric-like textures)
        self.bg_types = ['cream', 'white', 'light_gray', 'beige', 'off_white']
        
        # Text colors
        self.text_colors = [
            (40, 40, 40),      # Dark gray
            (0, 0, 0),         # Black
            (50, 50, 50),      # Medium gray
            (255, 255, 255),   # White (for dark backgrounds)
        ]
    
    def create_fabric_texture(self, width, height, bg_type='cream'):
        """Create realistic fabric texture background"""
        
        # Base color based on type
        if bg_type == 'cream':
            base_color = random.randint(230, 245)
        elif bg_type == 'white':
            base_color = random.randint(240, 255)
        elif bg_type == 'light_gray':
            base_color = random.randint(200, 220)
        elif bg_type == 'beige':
            base_color = random.randint(220, 235)
        elif bg_type == 'off_white':
            base_color = random.randint(235, 250)
        else:
            base_color = 230
        
        # Create base
        background = np.ones((height, width, 3), dtype=np.int16) * base_color
        
        # Add fabric weave texture
        weave_size = random.choice([2, 3, 4])
        for y in range(0, height, weave_size):
            for x in range(0, width, weave_size):
                variation = random.randint(-5, 5)
                # Ensure we don't go out of bounds
                y_end = min(y + weave_size, height)
                x_end = min(x + weave_size, width)
                background[y:y_end, x:x_end] += variation

        # Clip to valid range and convert to uint8
        background = np.clip(background, 0, 255).astype(np.uint8)
        
        # Add noise for realistic texture
        noise = np.random.normal(0, 3, (height, width, 3))
        background = background.astype(np.int16) + noise.astype(np.int16)
        background = np.clip(background, 0, 255).astype(np.uint8)
        
        # Slight blur to make it look more fabric-like
        background = cv2.GaussianBlur(background, (3, 3), 0)
        
        return background
    
    def get_font(self, size):
        """Get font - tries system fonts"""
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        # Fallback to default
        return ImageFont.load_default()
    
    def generate_tag(self, output_path=None):
        """Generate a single synthetic tag image"""
        
        # Random dimensions (based on real tags)
        width = random.randint(600, 1000)
        height = random.randint(800, 1400)
        
        # Create background
        bg_type = random.choice(self.bg_types)
        background = self.create_fabric_texture(width, height, bg_type)
        
        # Convert to PIL for text
        img = Image.fromarray(background)
        draw = ImageDraw.Draw(img)
        
        # Select content
        country = random.choice(self.countries)
        material = random.choice(self.materials)
        care = random.choice(self.care_instructions)
        
        # Font sizes
        large_font = self.get_font(random.randint(35, 50))
        medium_font = self.get_font(random.randint(30, 40))
        small_font = self.get_font(random.randint(25, 35))
        
        # Text color (dark for light backgrounds, light for dark)
        if bg_type in ['cream', 'white', 'beige', 'off_white', 'light_gray']:
            text_color = random.choice([(40, 40, 40), (0, 0, 0), (50, 50, 50)])
        else:
            text_color = (255, 255, 255)
        
        # Layout
        y_pos = random.randint(60, 100)
        x_margin = random.randint(40, 80)
        
        # 1. Country (multilingual)
        draw.text((x_margin, y_pos), f"Made In {country}", 
                 fill=text_color, font=large_font)
        y_pos += random.randint(45, 60)
        
        draw.text((x_margin, y_pos), f"Fabriqué En {country}", 
                 fill=text_color, font=small_font)
        y_pos += random.randint(40, 55)
        
        # Sometimes add more languages
        if random.random() > 0.5:
            draw.text((x_margin, y_pos), f"Hecho en {country}", 
                     fill=text_color, font=small_font)
            y_pos += random.randint(40, 55)
        
        y_pos += random.randint(30, 50)
        
        # 2. Material composition
        # Split material into lines if it's multi-part
        material_lines = material.split(' Body:')
        if len(material_lines) > 1:
            draw.text((x_margin, y_pos), material_lines[0].strip(), 
                     fill=text_color, font=medium_font)
            y_pos += random.randint(40, 50)
            draw.text((x_margin, y_pos), 'Body:' + material_lines[1], 
                     fill=text_color, font=medium_font)
        else:
            draw.text((x_margin, y_pos), material, 
                     fill=text_color, font=medium_font)
        
        y_pos += random.randint(50, 70)
        
        # Sometimes add "Exclusive of decoration"
        if random.random() > 0.7:
            draw.text((x_margin, y_pos), "Exclusive of decoration", 
                     fill=text_color, font=small_font)
            y_pos += random.randint(35, 45)
        
        y_pos += random.randint(30, 50)
        
        # 3. Care instructions
        care_lines = care.split(',')
        for line in care_lines:
            line = line.strip()
            if line:
                draw.text((x_margin, y_pos), line.upper(), 
                         fill=text_color, font=medium_font)
                y_pos += random.randint(40, 50)
        
        # Sometimes add RN number
        if random.random() > 0.5:
            y_pos += random.randint(20, 40)
            rn_number = random.randint(10000, 999999)
            draw.text((x_margin, y_pos), f"RN {rn_number}", 
                     fill=text_color, font=small_font)
        
        # Save
        if output_path:
            img.save(output_path, quality=95)
        
        return np.array(img), {
            'country': country,
            'material': material,
            'care': care
        }
    
    def generate_dataset(self, num_samples=500, output_folder='synthetic_tags'):
        """Generate a dataset of synthetic tags"""
        
        os.makedirs(f'{output_folder}/images', exist_ok=True)
        os.makedirs(f'{output_folder}/labels', exist_ok=True)
        
        # Initialize emissions tracker
        tracker = EmissionsTracker(
            project_name="synthetic_tag_generation",
            output_dir="emissions",
            output_file="generation_emissions.csv"
        )
        
        tracker.start()
        start_time = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"GENERATING SYNTHETIC TAG IMAGES")
        print(f"{'='*60}")
        print(f"Target: {num_samples} images")
        print(f"Tracking emissions...")
        print(f"{'='*60}\n")
        
        for i in range(num_samples):
            img_path = f'{output_folder}/images/synthetic_{i:04d}.jpg'
            txt_path = f'{output_folder}/labels/synthetic_{i:04d}.txt'
            
            img, labels = self.generate_tag(img_path)
            
            # Save labels
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"COUNTRY: {labels['country']}\n")
                f.write(f"MATERIAL: {labels['material']}\n")
                f.write(f"CARE: {labels['care']}\n")
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_samples}")
        
        # Stop tracking and get emissions
        emissions = tracker.stop()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save detailed emissions report
        emissions_report = {
            'step': 'synthetic_generation',
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'num_images': num_samples,
            'emissions_kg_co2': emissions,
            'emissions_per_image_g_co2': (emissions * 1000) / num_samples if num_samples > 0 else 0,
            'energy_consumed_kwh': emissions / 0.475,  # Approximate conversion
        }
        
        # Save to JSON
        os.makedirs('emissions', exist_ok=True)
        with open('emissions/generation_report.json', 'w') as f:
            json.dump(emissions_report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Generated {num_samples} synthetic tags!")
        print(f"  Images: {output_folder}/images/")
        print(f"  Labels: {output_folder}/labels/")
        print(f"\n{'='*60}")
        print(f"EMISSIONS REPORT")
        print(f"{'='*60}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"CO2 Emissions: {emissions*1000:.2f} g CO2")
        print(f"Per image: {(emissions*1000)/num_samples:.4f} g CO2")
        print(f"Energy: {emissions/0.475:.4f} kWh")
        print(f"Equivalent to:")
        print(f"  - {emissions*2.5:.2f} miles driven")
        print(f"  - {emissions*120:.0f} smartphone charges")
        print(f"\nDetailed logs: emissions/")
        print(f"{'='*60}")

if __name__ == "__main__":
    generator = SyntheticTagGenerator()
    generator.generate_dataset(num_samples=500, output_folder='synthetic_tags')