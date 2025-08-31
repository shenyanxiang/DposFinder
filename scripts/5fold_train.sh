# for i in {0..4}; do
#    python ./code/main.py --model deposcope --kfold 5 --mode train --fold_num $i --name 5fold_$i >>output/5fold_$i.log
#    sleep 10
# done
# for i in {0..4}; do
#    python ./code/main.py --model phagedpo --kfold 5 --mode train --fold_num $i --name 5fold_$i >>output/5fold_phagedpo_$i.log
#    sleep 10
# done
for i in {0..4}; do
   python ./code/main.py --model dposfinder --kfold 5 --mode train --hid_dim 64 --batch_size 8 --fold_num $i --name 5fold_$i >>output/5fold_dposfinder64_$i.log 
   sleep 10
done