# echo "$1" | tee -a readesult_top_rules.txt 
cat model_list_rules.txt | while read line 
do
   echo "$line" | tee -a result_top_rules.txt 
    A=$(echo "$line" | cut -f1 -d' ')
    B=$(echo "$line" | cut -f2 -d' ')
    echo $A $B
   python3 diora/scripts/best_rules.py     --batch_size 100    --data_type nli     --elmo_cache_dir ~/Content_alignment/diora_snli/data/elmo     --load_model_path "~/Content_alignment/diora_with_rules/diora/log/""$A/model.step_50000.pt" --validation_path ~/Content_alignment/diora_snli/data/snli_1.0/snli_1.0_dev.jsonl --train_path ~/Content_alignment/diora_snli/data/snli_1.0/snli_1.0_dev.jsonl --model_flags "~/Content_alignment/diora_with_rules/diora/log/""$A/flags.json" --validation_filter_length 40 --data_type nli --rule_based 

   echo "__________" | tee -a result_top_rules.txt 
   python3 print_rules.py --indices "top_rules_""$A.txt" --rule_file "../data_preprocessing/Basic-CYK-Parser/""$B.txt" | tee -a result_top_rules.txt 
   # echo $line
done
