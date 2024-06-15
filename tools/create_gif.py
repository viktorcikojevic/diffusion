import os
from PIL import Image, ImageDraw, ImageFont
import glob

def create_gif(folder_path, output_path, total_duration=200, end_pause=3000):
    # Get all sample images
    sample_paths = sorted(glob.glob(os.path.join(folder_path, 'sample_*_step_*.png')))
    
    # Extract the number of samples and steps from file names
    samples = set()
    steps = set()
    
    for sample_path in sample_paths:
        basename = os.path.basename(sample_path)
        sample_num = int(basename.split('_')[1])
        step_num = int(basename.split('_')[3].split('.')[0])
        samples.add(sample_num)
        steps.add(step_num)
        
    n_samples = len(samples)
    n_steps = len(steps)
    
    # Determine the size of each image from the first image
    first_image_path = sample_paths[0]
    with Image.open(first_image_path) as img:
        img_width, img_height = img.size
    
    # Restrict to the last 400 steps
    steps = sorted(list(steps))[-800:][::20]
    n_steps = len(steps)
    
    # Create a list of frames for the GIF
    frames = []
    
    for step in steps:
        frame = Image.new('RGB', (n_samples * img_width, img_height))  # Create a new frame for each step
        for sample in sorted(samples):
            img_path = os.path.join(folder_path, f'sample_{sample}_step_{step}.png')
            img = Image.open(img_path)
            frame.paste(img, (sample * img_width, 0))  # Position the image in the frame
        frames.append(frame)
    
    # Calculate frame duration to make the entire GIF last 5 seconds
    frame_duration = total_duration // n_steps
    durations = [frame_duration] * n_steps
    
    # Adjust the last frame's duration to include the end pause
    durations[-1] += end_pause
    
    # Save as GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)

# Usage
folder_path = 'samples/all_samples'
output_path = 'samples/evolution.gif'
create_gif(folder_path, output_path)
