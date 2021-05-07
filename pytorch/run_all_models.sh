echo "$1" | tee -a result_all.txt 
cat model_list.txt | while read line 
do
   echo "$line" | tee -a result_all.txt 
   bash get_scores.sh "~/Content_alignment/diora_with_rules/diora/log/""$line/model_periodic.pt" "$1" > temp_log 2>/dev/null;

   tail -n 40 temp_log | tee -a result_all.txt
   echo "__________" | tee -a result_all.txt 
done

#bash get_scores.sh "~/Content_alignment/diora_with_rules/diora/log/e3e99bee/model_periodic.pt " "~/Content_alignment/data_for_diora/mnli/multinli_1.0/multinli_1.0_dev_all.jsonl" 


