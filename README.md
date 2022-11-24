# MultiABSA
 Prompt Based Multi-Task Framework with Contrastive Pre-training for Unified Aspect Based Sentiment Analysis

 Run the code with the script

 ```
 python main.py --task 15res  --train_dataset_path data/15res/train  --dev_dataset_path data/15res/dev  --test_dataset_path data/15res/test  --model_name_or_path t5-base  --do_train   --do_eval  --train_batch_size 4  --gradient_accumulation_steps 4  --eval_batch_size 16  --learning_rate 3e-4  --num_train_epochs 20   --logger_name batch_4_12__15res_logs_regressor0.2_and_tagger_with_contrast6.txt  --log_message epoch6_4_4_3e4_0.2default --gpu_id 1 --absa_task ae
 ```
 
