#usage - bash get_scores.sh <path_to_model.pt> #optionally also give path of gt_file second

if [ "$#" -ne 1 ]; then
    python3 diora/scripts/parse.py     --batch_size 100    --data_type nli     --elmo_cache_dir ~/data/elmo     --load_model_path $1 --validation_path $2 --model_flags ~/Downloads/diora-checkpoints/mlp-softmax-shared/flags.json --validation_filter_length 40 --data_type nli --postprocess --validation_filter_length 60 --postprocess --cuda > score_log.txt 2 > /dev/null
else
    python3 diora/scripts/parse.py     --batch_size 1000     --data_type nli     --elmo_cache_dir ~/Content_alignment/diora_snli/data/elmo     --load_model_path $1 --validation_path ~/Content_alignment/diora_snli/data/snli_1.0/snli_1.0_dev.jsonl --model_flags ~/Content_alignment/diora_snli/Downloads/diora-checkpoints/mlp-softmax-shared/flags.json --validation_filter_length 60 --postprocess --cuda > score_log.txt
fi

cd diora/scripts

if [ "$#" -ne 1 ]; then
    python3 eval_brackets.py --test_file $(tail -n 1 ../../score_log.txt) --gt_file $2
else
    python3 eval_brackets.py --test_file $(tail -n 1 ../../score_log.txt)
fi

sleep 5;


cd ../../

echo $(tail -n 1 score_log.txt)
# python3 eval_brackets.py --test_file /home/ritesh/Content_alignment/diora_with_rules/diora/log/0ccfbcf9/parse.jsonl  --gt_file ~/Content_alignment/Tree-Transformer/wsj_train_bracket_.jsonl 

