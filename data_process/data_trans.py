import json
import os
from typing import Dict, List, Any, Tuple
from collections import defaultdict

def create_type_mapping() -> Dict[str, List[str]]:
    """
    Create a mapping of question patterns to question types.
    """
    type_mapping = {
        # Modality questions
        "modality": ["what modality", "which modality", "what type of scan", "what type of image", 
                    "what imaging", "what kind of scan"],
        
        # Organ/anatomy questions
        "anatomy": ["what organ", "which organ", "what structure", "what body part", 
                   "what anatomical", "what is shown", "what do you see", "identify the organ",
                   "what tissue", "what anatomy"],
        
        # Location questions
        "location": ["where is", "what location", "which side", "what region", 
                    "what area", "in which part", "located", "position"],
        
        # Abnormality/pathology questions
        "abnormality": ["what abnormality", "what is wrong", "what pathology", 
                       "what disease", "what condition", "what diagnosis", "what disorder",
                       "what illness", "what problem"],
        
        # Descriptive questions
        "description": ["describe", "what are the findings", "explain", 
                       "what features", "what characteristics", "tell me about"],
        
        # Yes/no questions (typically close-set)
        "yes_no": ["is there", "are there", "does", "do you see", "can you see",
                  "is this", "are these", "is it", "do you observe"],
        
        # Counting questions
        "counting": ["how many", "count", "number of", "what is the number"],
        
        # Comparison questions
        "comparison": ["compare", "difference between", "which is larger", 
                      "which is smaller", "bigger", "smaller"],
        
        # Treatment/procedure questions
        "procedure": ["what procedure", "what treatment", "what intervention",
                     "what surgery", "what operation"],
        
        # Compression/effect questions
        "effect": ["what is being compressed", "what is affected", "what is the effect",
                  "what is causing", "what impact"],
        
        # Size/measurement questions
        "measurement": ["what size", "how large", "how big", "what dimension", 
                       "measure", "what is the measurement", "how long", "how wide"],
        
        # Color/appearance questions
        "appearance": ["what color", "what appearance", "how does it look", "what shape"],
        
        # Presence/absence questions (typically close-set)
        "presence": ["is present", "are present", "presence of", "absence of"]
    }
    
    return type_mapping

def determine_answer_category(answer_type: str, question: str, answer: str) -> str:
    """
    Determine if the question is close-set or open-ended based on answer type and content.
    """
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Direct answer type mapping
    if answer_type and answer_type.lower() in ["close-ended", "closed"]:
        return "close-set"
    
    # Check for yes/no patterns
    yes_no_answers = ["yes", "no", "true", "false", "present", "absent", "normal", "abnormal"]
    if answer_lower in yes_no_answers:
        return "close-set"
    
    # Check for yes/no question patterns
    yes_no_patterns = ["is there", "are there", "does", "do you see", "can you see", 
                      "is it", "is this", "are these", "is present", "are present"]
    for pattern in yes_no_patterns:
        if pattern in question_lower:
            return "close-set"
    
    # Check for limited choice patterns (e.g., left/right, specific modalities)
    limited_choices = {
        "side": ["left", "right", "bilateral"],
        "modality": ["ct", "mri", "x-ray", "xray", "ultrasound", "pet"],
        "position": ["anterior", "posterior", "lateral", "medial", "superior", "inferior"]
    }
    
    for category, choices in limited_choices.items():
        if answer_lower in choices:
            return "close-set"
    
    # Default to open-ended
    return "open-ended"

def handle_duplicate_ids(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Handle duplicate IDs by adding suffixes.
    """
    id_counts = defaultdict(int)
    processed_entries = []
    
    for entry in entries:
        original_id = entry["id"]
        
        if id_counts[original_id] == 0:
            # First occurrence, use as is
            processed_entries.append(entry)
        else:
            # Duplicate found, add suffix
            entry["id"] = f"{original_id}#&#num{id_counts[original_id]}"
            processed_entries.append(entry)
        
        id_counts[original_id] += 1
    
    return processed_entries

def convert_dataset_format(input_data: Dict[str, Any], 
                         type_mapping: Dict[str, List[str]],
                         image_base_path: str = "/path/to/your/images/") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert a single dataset entry from the old format to the new format.
    Returns both the converted data and type information.
    """
    # Determine question type
    question_type = determine_question_type(input_data["text"], type_mapping)
    
    # Determine answer category (close-set vs open-ended)
    answer_category = determine_answer_category(
        input_data.get("answer_type", ""),
        input_data["text"],
        input_data["gpt4_answer"]
    )
    
    # Create the new format - 修复了这里的问题
    converted_data = {
        "id": input_data["question_id"],
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{input_data['text']}"  # 添加 <image> 标记
            },
            {
                "from": "gpt",
                "value": input_data["gpt4_answer"]
            }
        ],
        "image": os.path.join(image_base_path, input_data["image"])
    }
    
    # Type information for separate JSON
    type_info = {
        "id": input_data["question_id"],
        "question_type": question_type,
        "answer_category": answer_category,
        "original_answer_type": input_data.get("answer_type", ""),
        "domain": input_data.get("domain", {})
    }
    
    return converted_data, type_info

def determine_question_type(question: str, type_mapping: Dict[str, List[str]]) -> str:
    """
    Determine the type of question based on the mapping.
    """
    question_lower = question.lower()
    
    for question_type, patterns in type_mapping.items():
        for pattern in patterns:
            if pattern in question_lower:
                return question_type
    
    # Default type if no pattern matches
    return "general"

def process_dataset_file(input_file: str, output_file: str, 
                        type_mapping_file: str,
                        image_base_path: str = "/path/to/your/images/") -> None:
    """
    Process an entire dataset file and convert all entries.
    Also generates a separate type mapping file.
    """
    # Create type mapping
    type_mapping = create_type_mapping()
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
    
    # Convert all entries
    converted_data = []
    type_info_data = []
    
    for entry in data:
        converted_entry, type_info = convert_dataset_format(entry, type_mapping, image_base_path)
        converted_data.append(converted_entry)
        type_info_data.append(type_info)
    
    # Handle duplicate IDs
    converted_data = handle_duplicate_ids(converted_data)
    type_info_data = handle_duplicate_ids(type_info_data)
    
    # Generate statistics
    stats = {
        "total_entries": len(converted_data),
        "question_types": defaultdict(int),
        "answer_categories": defaultdict(int),
        "type_by_category": defaultdict(lambda: defaultdict(int))
    }
    
    for type_info in type_info_data:
        q_type = type_info["question_type"]
        a_category = type_info["answer_category"]
        
        stats["question_types"][q_type] += 1
        stats["answer_categories"][a_category] += 1
        stats["type_by_category"][a_category][q_type] += 1
    
    # Create type mapping output
    type_mapping_output = {
        "type_definitions": type_mapping,
        "entries": type_info_data,
        "statistics": {
            "total_entries": stats["total_entries"],
            "question_types": dict(stats["question_types"]),
            "answer_categories": dict(stats["answer_categories"]),
            "type_by_category": {k: dict(v) for k, v in stats["type_by_category"].items()}
        }
    }
    
    # Write output files
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    with open(type_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(type_mapping_output, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"Conversion complete! Processed {len(converted_data)} entries.")
    print(f"\nFiles created:")
    print(f"  - Main dataset: {output_file}")
    print(f"  - Type mapping: {type_mapping_file}")
    
    print("\nQuestion type distribution:")
    for q_type, count in sorted(stats["question_types"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {q_type}: {count}")
    
    print("\nAnswer category distribution:")
    for category, count in stats["answer_categories"].items():
        print(f"  {category}: {count}")
    
    print("\nQuestion types by answer category:")
    for category, types in stats["type_by_category"].items():
        print(f"\n  {category}:")
        for q_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {q_type}: {count}")

def main():
    """
    Main function to run the conversion.
    可以根据问题自动区分哪些是 close-set问题 哪些是 open-set问题
    """
    input_json = '/test.json'
    output_json = 'slake_test.json'
    type_mapping_json = 'slake_test_type_mapping.json'
    image_path = 'datasets/slake_parsed/images'
    
    process_dataset_file(input_json, output_json, type_mapping_json, image_path)

if __name__ == "__main__":
    main()