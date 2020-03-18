ls -d $PWD/model_data/$1/data_train/*.jpg -F -1 >model_data/$1/data_train.txt
ls -d $PWD/model_data/$1/data_val/*.jpg -F -1  >model_data/$1/data_val.txt