#!/usr/bin/env python3
"""
基于Qwen2.5-VL-32B模型的图像QA数据处理脚本
功能：输入图像、问题和原始答案，生成直接回答和推理回答两个版本
"""

import json
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoProcessor,
    BitsAndBytesConfig
)
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
import os
from PIL import Image
import gc
import logging
import sys


# Context Learning Examples
DIRECT_EXAMPLES = """Here are some examples of direct answer conversions:
Original QA:  
Question: What is the purpose of the diagram?  
Answer: The purpose of the flow diagram is to illustrate the lung cancer screening process implemented by the local government of Yokohama City. It shows the steps involved in the screening, from the primary examination to the follow-up examinations for participants with positive results.  
Converted to direct answer:  
Answer: Lung cancer screening.

Original QA:  
Question: What type of image is this?  
Answer: This is a CT (computed tomography) scan image, which is a type of medical imaging technique that uses X-rays to create detailed cross-sectional images of the body. In this case, the image is a coronal view, meaning it is taken from the front of the body, looking towards the back.  
Converted to direct answer:  
Answer: CT (computed tomography)

Original QA:  
Question: What type of hemorrhage is shown in the image?  
Answer: The image shows a subarachnoid hemorrhage, which is a type of bleeding that occurs in the subarachnoid space, the area between the brain and the thin tissues that cover it.  
Converted to direct answer:  
Answer: Subarachnoid hemorrhage

Original QA:  
Question: Can you see the pacemaker in the image?  
Answer: Yes, the chest X-ray image shows the VVI pacemaker. The image is used to confirm the correct placement of the pacemaker within the patient's chest.  
Converted to direct answer:  
Answer: Yes
"""

REASONING_EXAMPLES = """Here are some examples of reasoning-based answer conversions:
Original QA:  
Question: How about the condition of the lungs on POD 30?  
Answer: On POD 30 (postoperative day 30), the chest X-ray shows that the bilateral pulmonary edema has disappeared, indicating an improvement in the patient's lung condition.  
Converted with reasoning:  
Answer: An improvement in the patient's lung conditionAn improvement in the patient's lung condition, as the chest X-ray shows the bilateral pulmonary edema has disappeared.

Original QA:  
Question: Can you see the pacemaker in the image?  
Answer: Yes, the chest X-ray image shows the VVI pacemaker. The image is used to confirm the correct placement of the pacemaker within the patient's chest.  
Converted with reasoning:  
Answer: Yes, the image confirms the pacemaker is correctly placed.

Original QA:  
Question: What do the black bars represent?  
Answer: The black bars in the figure represent the number of patients with confirmed Ankylosing Spondylitis (AS), which is a chronic inflammatory condition that primarily affects the spine and sacroiliac joints.  
Converted with reasoning:  
Answer: The number of patients with confirmed Ankylosing Spondylitis (AS), a chronic spinal condition.
"""


def setup_logger(log_file=None, level=logging.INFO):
    logger = logging.getLogger('QwenVL_QA')
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# 在main函数或全局加：
logger = None  # 全局logger

class Qwen25VLProcessor:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-32B-Instruct", device: str = "cuda"):
        """Initialize Qwen2.5-VL-32B model with fp16"""
        print(f"Loading model: {model_path}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model with fp16 for better speed
        if "Qwen2.5" in model_path:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            model_path,
                            device_map="auto",
                            trust_remote_code=True,
                            torch_dtype=torch.float16,  # Use fp16 instead of 4bit
                            # attn_implementation="flash_attention_2"  # Use Flash Attention 2 for speed
                             )
        else:               
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Use fp16 instead of 4bit
                # attn_implementation="flash_attention_2"  # Use Flash Attention 2 for speed
                )
            
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        self.device = device
        print("Model loaded successfully with fp16")
    
    def generate_direct_answer(self, image_path: str, question: str, original_answer: str) -> str:
        """Generate direct answer version with image context"""
        messages = [
            {
                "role": "system",
                "content": "You are a professional medical image analysis assistant. Convert long answers to concise, direct answers based on the examples provided. Consider the image content when converting the answer."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": f"""{DIRECT_EXAMPLES}

Now, looking at this medical image and considering the original QA, convert to a direct answer:

Original QA:
Question: {question}
Answer: {original_answer}

Converted to direct answer:
Answer:"""
                    }
                ]
            }
        ]
        
        return self._generate_response(
            messages, 
            max_new_tokens=128,
            temperature=0.1,  # Lower temperature for more consistent direct answers
            top_p=0.9,
            repetition_penalty=1.0
        )
    
    def generate_reasoning_answer(self, image_path: str, question: str, original_answer: str) -> str:
        """Generate reasoning answer version with image context"""
        messages = [
            {
                "role": "system",
                "content": "You are a professional medical image analysis assistant. Based on the provided examples, and taking into account the input question and original answer, convert the original answer into a version that includes either a clear and concise direct answer, or a brief reasoning or status description—one of the two is sufficient. When converting the answer, please consider the image content and only transform information from the original answer without adding any new content."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": f"""{REASONING_EXAMPLES}

Now, looking at this medical image and considering the original QA, convert with reasoning:

Original QA:
Question: {question}
Answer: {original_answer}

Converted with reasoning:
Answer:"""
                    }
                ]
            }
        ]
        
        return self._generate_response(
            messages,
            max_new_tokens=256,
            temperature=0.3,  # Slightly higher for more varied reasoning
            top_p=0.9,
            repetition_penalty=1.0
        )
    
    def _generate_response(self, messages: List[Dict], max_new_tokens: int = 256, 
                          temperature: float = 0.7, top_p: float = 0.9, 
                          repetition_penalty: float = 1.0) -> str:
        """Generate response using Qwen2.5-VL with optimizations"""
        # Process messages to extract images
        images = []
        formatted_messages = []
        
        for msg in messages:
            if isinstance(msg.get("content"), list):
                # Extract images from content
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        if "image" in content_item:
                            # Load image from path
                            try:
                                image = Image.open(content_item["image"])
                                images.append(image)
                            except Exception as e:
                                logger.info(f"Error loading image: {e}")
                                continue
                formatted_messages.append(msg)
            else:
                formatted_messages.append(msg)
        
        # Prepare text for the model
        text = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # Generate with optimizations
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0.0,
                num_beams=1,  # Greedy decoding for speed
                use_cache=True,  # Enable KV cache
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()
    
    def process_qa_data(self, args, prefix_fn, data: List[Dict[str, Any]], batch_size: int = 1) -> tuple:
        """Process QA data with images and generate two versions"""
        direct_data = []
        reasoning_data = []
        
        # Process items one by one (batch processing with images is memory intensive)
        count = 0 
        for item in tqdm(data, desc="Processing QA data with images"):
            image_path = item["image"]
            
            # Check if image exists
            if not os.path.exists(image_path):
                logger.info(f"Warning: Image not found: {image_path}")
                continue
            
            # Extract question and answer
            question = None
            answer = None
            
            for conv in item["conversations"]:
                if conv["from"] == "human":
                    question = conv["value"].replace("<image>\n", "")
                elif conv["from"] == "gpt":
                    answer = conv["value"]
                # import pdb;pdb.set_trace()
                if answer is None:
                    continue
                try:
                    # Generate direct answer
                    direct_answer = self.generate_direct_answer(image_path, question, answer)
                    
                    # Clear GPU cache between generations
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Generate reasoning answer
                    reasoning_answer = self.generate_reasoning_answer(image_path, question, answer)
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Create direct answer item
                    direct_item = {
                        "id": f"{item['id']}_direct",
                        "image": item["image"],
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image>\n[Direct Answer Required] {question}"
                            },
                            {
                                "from": "gpt",
                                "value": direct_answer
                            }
                        ]
                    }
                    direct_data.append(direct_item)
                    
                    # Create reasoning answer item
                    reasoning_item = {
                        "id": f"{item['id']}_reasoning",
                        "image": item["image"],
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image>\n[Reasoning Required] {question}"
                            },
                            {
                                "from": "gpt",
                                "value": reasoning_answer
                            }
                        ]
                    }
                    reasoning_data.append(reasoning_item)
                    
                except Exception as e:
                    logger.info(f"Error processing item {item['id']}: {str(e)}")
                    continue
            count += 1
            if count % 20 == 0:
                # break
                logger.info(f'Gene Sample Count : {count}')
                direct_output = os.path.join(args.output_dir, f"{prefix_fn}_qa_direct_answers.json")
                reasoning_output = os.path.join(args.output_dir, f"{prefix_fn}_qa_reasoning_answers.json")
                
                with open(direct_output, 'w', encoding='utf-8') as f:
                    json.dump(direct_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Direct answers saved to: {direct_output}")
                
                with open(reasoning_output, 'w', encoding='utf-8') as f:
                    json.dump(reasoning_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Reasoning answers saved to: {reasoning_output}")    
                
            
        return direct_data, reasoning_data

def main():
    parser = argparse.ArgumentParser(description="Process QA data with images using Qwen2.5-VL-32B")
    parser.add_argument("--input", type=str, help="Input JSON file path",)
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--model-path", type=str, 
                        help="Qwen2.5-VL model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device type")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum number of items to process (for testing)")
    args = parser.parse_args()

    # 初始化logger
    
    prefix_fn = args.input.split('/')[-1].split('.')[0]
    global logger
    logger = setup_logger(None)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    logger.info(f"Loading data: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    # Limit data if specified
    if args.max_items:
        data = data[:args.max_items]
    
    # Initialize processor
    processor = Qwen25VLProcessor(model_path=args.model_path, device=args.device)
    
    # Process data
    logger.info(f"Processing {len(data)} items with images...")
    direct_data, reasoning_data = processor.process_qa_data(args, prefix_fn, data)
    
    # Save results
    direct_output = os.path.join(args.output_dir, f"{prefix_fn}_qa_direct_answers.json")
    reasoning_output = os.path.join(args.output_dir, f"{prefix_fn}_qa_reasoning_answers.json")
    
    with open(direct_output, 'w', encoding='utf-8') as f:
        json.dump(direct_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Direct answers saved to: {direct_output}")
    
    with open(reasoning_output, 'w', encoding='utf-8') as f:
        json.dump(reasoning_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Reasoning answers saved to: {reasoning_output}")
    
    # Save processing statistics
    stats = {
        "total_items": len(data),
        "processed_items": len(direct_data),
        "failed_items": len(data) - len(direct_data)
    }
    
    stats_output = os.path.join(args.output_dir, "processing_stats.json")
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nProcessing complete!")
    logger.info(f"Total items: {stats['total_items']}")
    logger.info(f"Processed: {stats['processed_items']}")
    logger.info(f"Failed: {stats['failed_items']}")

if __name__ == "__main__":
    main()