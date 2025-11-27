import os
import shutil
import datetime
import json
import subprocess
from pathlib import Path
import argparse

def count_images(folder):
    exts = ['.jpg', '.png', '.jpeg']
    return len([f for f in Path(folder).glob('*') if f.suffix.lower() in exts])

def move_images(src_folder, dst_folder):
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    for img_file in Path(src_folder).glob('*'):
        if img_file.is_file():
            shutil.move(str(img_file), str(Path(dst_folder) / img_file.name))

def clear_folder(folder):
    for f in Path(folder).glob('*'):
        if f.is_file():
            f.unlink()

def main(temp_data_dir, local_dataset_dir, min_images, finetune_script, finetune_args, status_json):
    temp_normal = Path(temp_data_dir) / 'normal'
    temp_faulty = Path(temp_data_dir) / 'faulty'
    img_count_normal = count_images(temp_normal)
    img_count_faulty = count_images(temp_faulty)
    print(f"Found {img_count_normal} normal and {img_count_faulty} faulty images in temp_data")
    if img_count_normal < min_images:
        print(f"Not enough normal images to trigger finetuning (need {min_images}, found {img_count_normal}). Exiting.")
        return
    # Step 2: Move images to Local_Dataset/YYYY_MM/normal and faulty
    now = datetime.datetime.now()
    folder_name = f"{now.month:02d}_{now.year}"
    target_folder = Path(local_dataset_dir) / folder_name
    target_normal = target_folder / 'normal'
    target_faulty = target_folder / 'faulty'
    move_images(temp_normal, target_normal)
    move_images(temp_faulty, target_faulty)
    print(f"Moved {img_count_normal} normal images to {target_normal}")
    print(f"Moved {img_count_faulty} faulty images to {target_faulty}")
    # Step 3: Run finetune script
    cmd = ['python', finetune_script] + finetune_args
    print(f"Running finetune: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    # Step 4: Handle cleanup based on finetune result
    if result.returncode == 0:
        # Finetune succeeded, clear temp data
        clear_folder(temp_normal)
        clear_folder(temp_faulty)
        print("Finetuning successful. Temp data cleared.")
    else:
        # Finetune failed, delete new Local_Dataset folder
        print("Finetuning failed or did not happen. Deleting new Local_Dataset folder.")
        try:
            shutil.rmtree(target_folder)
            print(f"Deleted folder: {target_folder}")
        except Exception as e:
            print(f"Failed to delete folder {target_folder}: {e}")
    # Step 5: Write status JSON
    status = {
        'last_finetune': now.strftime('%Y-%m-%d %H:%M:%S'),
        'finetune_folder': str(target_normal.parent),
        'finetune_images_normal': img_count_normal,
        'finetune_images_faulty': img_count_faulty,
        'finetune_log': str(Path(finetune_args[finetune_args.index('--output-dir')+1]) / 'finetune_log.json') if '--output-dir' in finetune_args else None
    }
    with open(status_json, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"Finetune status written to {status_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automate finetuning workflow.')
    parser.add_argument('--temp-data', type=str, default='Finetune_data/temp_data', help='Path to temp_data folder')
    parser.add_argument('--local-dataset', type=str, default='Finetune_data/Local_Dataset', help='Path to Local_Dataset folder')
    parser.add_argument('--min-images', type=int, default=6, help='Minimum images required to trigger finetuning')
    parser.add_argument('--finetune-script', type=str, default='ML_analysis/finetune.py', help='Path to finetune script')
    parser.add_argument('--finetune-args', nargs='+', default=['--feedback-data', 'Finetune_data/Local_Dataset', '--weights', 'ML_analysis/models/best_model.pth', '--output-dir', 'Finetune_data/output'], help='Arguments for finetune script')
    parser.add_argument('--status-json', type=str, default='Finetune_data/finetune_status.json', help='Path to status JSON file')
    args = parser.parse_args()
    main(
        temp_data_dir=args.temp_data,
        local_dataset_dir=args.local_dataset,
        min_images=args.min_images,
        finetune_script=args.finetune_script,
        finetune_args=args.finetune_args,
        status_json=args.status_json
    )
