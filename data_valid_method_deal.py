import json
import os
import argparse
from PIL import Image
import torch
from transformers import AutoProcessor
from tqdm import tqdm
from collections import defaultdict
import traceback


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")

def check_image_valid(image_path, image_folder=None):
    """检查图片是否存在且可以打开"""
    try:
        # 处理路径
        if not os.path.exists(image_path):
            if image_folder and not image_path.startswith("http"):
                image_path = os.path.join(image_folder, image_path)
        
        if not os.path.exists(image_path):
            return False, f"文件不存在: {image_path}"
        
        # 尝试打开图片
        with Image.open(image_path) as img:
            img.verify()  # 验证图片完整性
        
        # 再次打开以确保可以正常读取
        with Image.open(image_path) as img:
            _ = img.size  # 尝试读取基本属性
        with Image.open(image_path) as img:
            # print(f"无法进行图片转换 to_rgb")
            to_rgb(img) # 尝试读取基本属性
            
        return True, "OK"
    except Exception as e:
        # import pdb;pdb.set_trace()
        return False, f"无法打开图片: {str(e)}"

def estimate_tokens(text, tokenizer):
    """估算文本的token数量"""
    try:
        tokens = tokenizer(text, add_special_tokens=False)
        return len(tokens['input_ids'])
    except:
        # 粗略估算：中文约1.5字符/token，英文约3-4字符/token
        return len(text) // 3

def check_conversation_format(conversations):
    """检查对话格式是否符合llava_to_openai的预期格式"""
    # 验证role字段的转换兼容性
    for i, conv in enumerate(conversations):
        from_field = conv.get('from', '')
        # llava格式使用'human'和'gpt'，openai格式使用'user'和'assistant'
        if from_field not in ['human', 'user', 'gpt', 'assistant']:
            return False, f"对话索引{i}的from字段'{from_field}'不是有效值"
    return True, "OK"

def validate_sample(sample, processor, image_folder=None, image_token="<image>"):
    """验证单个样本并返回统计信息"""
    result = {
        'valid': True,
        'errors': [],
        'stats': {
            'text_tokens': 0,
            'has_image': False,
            'image_path': None,
            'conversation_turns': 0
        }
    }
    
    # 检查基本结构
    if 'conversations' not in sample:
        result['valid'] = False
        result['errors'].append("缺少conversations字段")
        return result
    
    if not isinstance(sample['conversations'], list):
        result['valid'] = False
        result['errors'].append("conversations不是列表")
        return result
    
    # 检查对话数量（应该是偶数，user/assistant成对）
    if len(sample['conversations']) % 2 != 0:
        result['valid'] = False
        result['errors'].append(f"对话轮数为奇数: {len(sample['conversations'])}")
    
    result['stats']['conversation_turns'] = len(sample['conversations']) // 2
    
    # 检查对话格式
    format_valid, format_msg = check_conversation_format(sample['conversations'])
    if not format_valid:
        result['valid'] = False
        result['errors'].append(format_msg)
    
    # 验证对话配对（按照dataloader的逻辑）
    for j in range(0, len(sample['conversations']), 2):
        try:
            user_input = sample['conversations'][j]
            if j + 1 >= len(sample['conversations']):
                result['valid'] = False
                result['errors'].append(f"对话索引{j}缺少对应的回复")
                break
            
            gpt_response = sample['conversations'][j + 1]
            
            # 验证user_input
            if not isinstance(user_input, dict):
                result['valid'] = False
                result['errors'].append(f"对话索引{j}不是字典格式")
                continue
            
            if user_input.get('from') not in ['human', 'user']:
                result['valid'] = False
                result['errors'].append(f"对话索引{j}的from字段应为'human'或'user'，实际为'{user_input.get('from')}'")
            
            # 验证gpt_response
            if not isinstance(gpt_response, dict):
                result['valid'] = False
                result['errors'].append(f"对话索引{j+1}不是字典格式")
                continue
                
            if gpt_response.get('from') not in ['gpt', 'assistant']:
                result['valid'] = False
                result['errors'].append(f"对话索引{j+1}的from字段应为'gpt'或'assistant'，实际为'{gpt_response.get('from')}'")
            
            # 检查必需的value字段
            if 'value' not in user_input:
                result['valid'] = False
                result['errors'].append(f"对话索引{j}缺少value字段")
            
            if 'value' not in gpt_response:
                result['valid'] = False
                result['errors'].append(f"对话索引{j+1}缺少value字段")
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"处理对话配对时出错: {str(e)}")
    
    # 检查图片
    if 'image' in sample:
        result['stats']['has_image'] = True
        image_files = sample['image']
        if isinstance(image_files, str):
            image_files = [image_files]
        
        for image_file in image_files:
            result['stats']['image_path'] = image_file
            valid, msg = check_image_valid(image_file, image_folder)
            if not valid:
                result['valid'] = False
                result['errors'].append(msg)
    
    # 统计文本tokens
    total_text = ""
    has_image_token = False
    
    for i, conv in enumerate(sample['conversations']):
        if 'from' not in conv or 'value' not in conv:
            result['valid'] = False
            result['errors'].append(f"对话索引{i}格式错误：缺少from或value字段")
            continue
        
        # 检查是否包含图像标记
        if image_token in conv['value']:
            has_image_token = True
            # 图像标记应该只出现在user输入中（偶数索引）
            if i % 2 != 0:
                result['valid'] = False
                result['errors'].append(f"图像标记{image_token}出现在assistant回复中（索引{i}）")
        
        total_text += conv['value'] + " "
    
    # 验证图像标记与实际图片的一致性
    if has_image_token and not result['stats']['has_image']:
        result['valid'] = False
        result['errors'].append(f"文本中包含{image_token}但没有图片")
    
    if result['stats']['has_image'] and not has_image_token:
        result['valid'] = False
        result['errors'].append(f"有图片但文本中没有{image_token}标记")
    
    # 估算token数量
    if processor and hasattr(processor, 'tokenizer'):
        result['stats']['text_tokens'] = estimate_tokens(total_text, processor.tokenizer)
    else:
        result['stats']['text_tokens'] = len(total_text) // 3  # 粗略估算
    
    # 估算总长度（文本tokens + 图像tokens）
    # 注意：图像tokens数量取决于模型和图像分辨率，这里使用估算值
    if result['stats']['has_image']:
        # Qwen2-VL等模型的图像token数量通常在256-1024之间，取决于分辨率
        result['stats']['estimated_image_tokens'] = 576  # 常见的默认值
        result['stats']['total_tokens'] = result['stats']['text_tokens'] + result['stats']['estimated_image_tokens']
    else:
        result['stats']['total_tokens'] = result['stats']['text_tokens']
    
    return result

def main():
    parser = argparse.ArgumentParser(description='验证多模态数据集')
    parser.add_argument('--input_json', type=str, help='输入JSON文件路径',
                        default='data_path/clean_up/formatted_single_turn_dialogue_20K_valid.json')
    parser.add_argument('--valid_json', type=str, help='有效样本输出路径',
                        default='data_path/clean_up/formatted_single_turn_dialogue_20K_valid_0.json' )
    parser.add_argument('--invalid_json', type=str, 
                        default='data_path/clean_up/formatted_single_turn_dialogue_20K_invalid_1.json', 
                        help='无效样本输出路径')
    parser.add_argument('--stats_json', type=str, 
                        default='data_path/clean_up/formatted_single_turn_dialogue_20K_Statistic.json', 
                        help='统计信息输出路径')
    parser.add_argument('--image_folder', type=str, default=None, help='图片文件夹路径')
    parser.add_argument('--model_id', type=str, 
                        default=None, 
                        help='模型ID用于加载processor')
    parser.add_argument('--image_token', type=str, default='<image>', help='图像标记')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 尝试加载processor（用于更准确的token计算）
    processor = None
    try:
        print(f"加载processor: {args.model_id}")
        processor = AutoProcessor.from_pretrained(args.model_id)
    except Exception as e:
        print(f"警告：无法加载processor，将使用粗略估算: {e}")
    
    # 验证样本
    valid_samples = []
    invalid_samples = []
    stats = defaultdict(list)
    
    print("开始验证样本...")
    count = 0
    for sample in tqdm(data):
        try:
            result = validate_sample(sample, processor, args.image_folder, args.image_token)
            
            if result['valid']:
                valid_samples.append(sample)
            else:
                invalid_sample = sample.copy()
                invalid_sample['_validation_errors'] = result['errors']
                invalid_samples.append(invalid_sample)
            
            # 收集统计信息
            stats['total_tokens'].append(result['stats']['total_tokens'])
            stats['text_tokens'].append(result['stats']['text_tokens'])
            stats['has_image'].append(result['stats']['has_image'])
            stats['conversation_turns'].append(result['stats']['conversation_turns'])
            
        except Exception as e:
            print(f"处理样本时出错: {sample.get('id', 'unknown')}")
            traceback.print_exc()
            invalid_sample = sample.copy()
            invalid_sample['_validation_errors'] = [f"处理异常: {str(e)}"]
            invalid_samples.append(invalid_sample)
        count += 1
        
        # if count >= 100:
        #     break
    
    # 保存结果
    print(f"\n保存有效样本到: {args.valid_json}")
    with open(args.valid_json, 'w', encoding='utf-8') as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)
    
    print(f"保存无效样本到: {args.invalid_json}")
    with open(args.invalid_json, 'w', encoding='utf-8') as f:
        json.dump(invalid_samples, f, ensure_ascii=False, indent=2)
    
    # 生成统计报告
    total_samples = len(data)
    valid_count = len(valid_samples)
    invalid_count = len(invalid_samples)
    
    statistics = {
        'total_samples': total_samples,
        'valid_samples': valid_count,
        'invalid_samples': invalid_count,
        'valid_rate': f"{valid_count/total_samples*100:.2f}%",
        'samples_with_image': sum(stats['has_image']),
        'samples_without_image': len(stats['has_image']) - sum(stats['has_image']),
        'token_statistics': {
            'avg_total_tokens': sum(stats['total_tokens']) / len(stats['total_tokens']) if stats['total_tokens'] else 0,
            'max_total_tokens': max(stats['total_tokens']) if stats['total_tokens'] else 0,
            'min_total_tokens': min(stats['total_tokens']) if stats['total_tokens'] else 0,
            'avg_text_tokens': sum(stats['text_tokens']) / len(stats['text_tokens']) if stats['text_tokens'] else 0,
        },
        'conversation_statistics': {
            'avg_turns': sum(stats['conversation_turns']) / len(stats['conversation_turns']) if stats['conversation_turns'] else 0,
            'max_turns': max(stats['conversation_turns']) if stats['conversation_turns'] else 0,
            'min_turns': min(stats['conversation_turns']) if stats['conversation_turns'] else 0,
        }
    }
    
    # 添加长度分布统计
    if stats['total_tokens']:
        token_ranges = {
            '0-512': 0,
            '513-1024': 0,
            '1025-2048': 0,
            '2049-4096': 0,
            '4097+': 0
        }
        
        for tokens in stats['total_tokens']:
            if tokens <= 512:
                token_ranges['0-512'] += 1
            elif tokens <= 1024:
                token_ranges['513-1024'] += 1
            elif tokens <= 2048:
                token_ranges['1025-2048'] += 1
            elif tokens <= 4096:
                token_ranges['2049-4096'] += 1
            else:
                token_ranges['4097+'] += 1
        
        statistics['token_distribution'] = token_ranges
    
    print(f"\n保存统计信息到: {args.stats_json}")
    with open(args.stats_json, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    # 打印统计摘要
    print("\n=== 验证统计摘要 ===")
    print(f"总样本数: {total_samples}")
    print(f"有效样本: {valid_count} ({valid_count/total_samples*100:.2f}%)")
    print(f"无效样本: {invalid_count} ({invalid_count/total_samples*100:.2f}%)")
    print(f"\n包含图片的样本: {statistics['samples_with_image']}")
    print(f"纯文本样本: {statistics['samples_without_image']}")
    print(f"\n平均总tokens: {statistics['token_statistics']['avg_total_tokens']:.1f}")
    print(f"最大总tokens: {statistics['token_statistics']['max_total_tokens']}")
    print(f"最小总tokens: {statistics['token_statistics']['min_total_tokens']}")
    print(f"\n平均对话轮数: {statistics['conversation_statistics']['avg_turns']:.1f}")
    
    if 'token_distribution' in statistics:
        print("\nToken长度分布:")
        for range_name, count in statistics['token_distribution'].items():
            print(f"  {range_name}: {count} ({count/total_samples*100:.1f}%)")

if __name__ == "__main__":
    main()