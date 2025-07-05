# 环境设置

base_model=/Qwen/Qwen2-VL-2B-Instruct
loar_model=/output/mult_gpu_train_validDataset_0702/checkpoint-30066


test_json=data_path/test_data/slake_test.json

python test_ppl_propmt.py  \
    --base_model ${base_model} \
    --lora_model ${loar_model} \
    --json_path ${test_json} 
