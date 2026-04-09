# python react_code_log_dec7.py \
# --model_name gpt-5.2-thinking \
# --gpu_id 5


python react_code_log_gen_docs.py \
--model_name gpt-5.2-thinking \
--gpu_id 4 \
--config_path config/config_gpt.yaml


# tmux new -s docsgen
# tmux new -s yjh
# tmux new -s qwen
# tmux new -s claude

# tmux attach -t docsgen
# tmux attach -t yjh
# tmux attach -t qwen
# tmux attach -t claude
