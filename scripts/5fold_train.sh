for i in {0..4}
do
   python ./code/main.py --model spikehunter --kfold 5 --fold_num $i --name 5fold_$i >>output/5fold_$i.log
done