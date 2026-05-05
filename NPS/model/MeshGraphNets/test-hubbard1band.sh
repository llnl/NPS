#!/usr/bin/env bash
# #BSUB -G cmi amsdnn msgnn
#BSUB -G msgnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# generate connected graphs of order 4
# ./geng  -c 4 2>/dev/null | listg -e |grep Graph -A2 |grep -v '^--'|sed 'N;N;s/\n/ /g;s/,//g' |awk '{system("mkdir H"NR); ng=$6; for (i=1;i<=ng;i++) {print $(2*i+5)+1, $(2*i+6)+1 >>"H"NR"/in_graph";} }'

PYTHON=$HOME/lassen-space/anaconda2021/envs/PT19/bin/python
DAT=$HOME/data/hubbard-1band/?/*/[1-4]_*/
DAT=$HOME/data/hubbard-1band/[1-7]/*/*/

if false; then
$PYTHON MeshGraphNets/main.py --data=$DAT --dataset=hubbard1band --dir=tmp/test-hubbard1band --model=GNN \
  --nfeat_in=3 --nfeat_out=2 --nfeat_out_global=1 --dim=0 --cache=T \
  --n_mpassing=3 --nfeat_latent_node=48 --mlp_activation=mish \
  --mode=train --lr=1e-3 --RNN=0 --print_freq=1000 --valid_freq=5000 --batch=1024
fi

device=0
# grid search of hyper parameters
for nlayer in 11 10; do
for batch in 256; do
        for mul in 128 256; do
        # for loss in mse; do
        for lr in 1e-4 2e-4; do
        # for wd in 1.1e-3; do
            DIR=experiment/hubbard_ntot-nlayer${nlayer}_mul${mul}_batch${batch}_lr${lr}
            mkdir -p $DIR
            echo $DIR $device
            CUDA_VISIBLE_DEVICES=$device $PYTHON MeshGraphNets/main.py --data=$DAT --dataset=hubbard1band-sum_node_y --dir=$DIR --model=GNN \
  --nfeat_in=3 --nfeat_out=1 --nfeat_out_global=1 --dim=0 --cache=T \
  --n_mpassing=$nlayer --nfeat_latent_node=$mul --mlp_activation=mish \
  --mode=train --lr=$lr --RNN=0 --print_freq=2000 --valid_freq=20000 --batch=1024 &>>$DIR/log &
            device=$(((device+1)%4))
done
done
done
done
wait


