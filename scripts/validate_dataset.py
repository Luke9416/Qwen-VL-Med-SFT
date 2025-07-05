#!/usr/bin/env python3
"""
数据集验证工具
检查训练和验证数据的格式和完整性
"""

import json
import os
import argparse
from pathlib import Path

def validate_dataset(data_path, image_folder=""):
    """验证数据集格式"""
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            conversations = data['data']
        elif isinstance(data, list):
            conversations = data
        else:
            print(f"❌ Invalid data format in {data_path}")
            return False
        
        print(f"✅ Loaded {len(conversations)} conversations from {data_path}")
        
        # 检查格式
        missing_images = 0
        for i, conv in enumerate(conversations):
            if 'image' in conv:
                image_files = conv['image'] if isinstance(conv['image'], list) else [conv['image']]
                for img_file in image_files:
                    if image_folder:
                        full_path = os.path.join(image_folder, img_file)
                    else:
                        full_path = img_file
                    
                    if not os.path.exists(full_path):
                        missing_images += 1
        
        if missing_images > 0:
            print(f"⚠️  {missing_images} image files not found")
        else:
            print("✅ All referenced images found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating {data_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--eval_data", default="")
    parser.add_argument("--image_folder", default="")
    
    args = parser.parse_args()
    
    print("📊 Dataset Validation")
    print("=" * 40)
    
    # 验证训练数据
    print("\n🔍 Validating training data...")
    train_valid = validate_dataset(args.train_data, args.image_folder)
    
    # 验证评估数据
    eval_valid = True
    if args.eval_data and args.eval_data.strip():
        print("\n🔍 Validating evaluation data...")
        eval_valid = validate_dataset(args.eval_data, args.image_folder)
    
    print("\n" + "=" * 40)
    if train_valid and eval_valid:
        print("✅ All datasets validated successfully!")
    else:
        print("❌ Dataset validation failed!")

if __name__ == "__main__":
    main()
