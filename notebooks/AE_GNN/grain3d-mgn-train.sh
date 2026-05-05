#!/usr/bin/env bash
source /usr/WS2/tian9/tuolu_venv2/bin/activate
PYTHON=/usr/WS2/tian9/tuolu_venv2/bin/python

device=0
for dimension in 32; do
# for dimension in 96; do
# dimension=160
infer_mode=original
# infer_mode=optimize




DAT=/usr/WS2/tian9/KMC_3D_2_4_bh_slow ###(32,32,32)  ### for figure 2 #--n_out_valid=24
# DAT=/usr/WS2/tian9/KMC_3D_quick_compress ###clone of KMC_3D_predict (96,75,96,96,96) for train and (16,75,96,96,96) for valid


# DAT=/usr/WS2/tian9/KMC_3D_slow_compress ###clone of KMC_3D_stats_long_slow (6, 300, 96, 96, 96, 1) for train and (6, 75, 96, 96, 96, 1) for valid
# DAT=/usr/WS2/tian9/KMC_3D_slow_compress_corr ###clone of KMC_3D_stats_long_slow (6, 225, 96, 96, 96, 1) for train and (6, 75, 96, 96, 96, 1) for valid


for n_in in 1; do
for n_out in 1 ; do
for nhid in 96; do
    for batch in 4; do   ### batch size has influence
    # for nmp in 3; do
    for nmp in 3; do

    for model in NPS_autoencoder; do #NPS_autoencoder
    # for model in NPS; do

        for noise in 1e-3; do
        for lr in 1e-4; do
            for loss in L2 ; do #(L1, L2)

            # for nae in 1 2 3; do
            for nae in 1; do


            for nencdec in 1; do
            ae_stride=`python -c "print(','.join(map(str,[1,2,2,2][:$nae])))"`
            ae_block=`python -c "print(','.join(map(str,[2,2,2,2][:$nae])))"`
            # nfeat_autoencoder=`python -c "print(4**$nae)"` # --nfeat_autoencoder=$nfeat_autoencoder

            DIR=experiment/steps_tune/grain_${model}-batch${batch}_lr${lr}_nin${n_in}_nout${n_out}_noise${noise}_loss${loss}_nmp${nmp}_nhid${nhid}_nae${nae}_nencdec${nencdec}

            mkdir -p $DIR
            echo $DIR $device

            if [ "$model" = "NPS" ]; then
                LOG_FILE=$DIR/log_gnn_${dimension}
            elif [[ "$model" == "NPS_autoencoder" && "$infer_mode" == "original" ]]; then
                LOG_FILE=$DIR/log_org_${dimension}
            else
                LOG_FILE=$DIR/log_lat_${dimension}
            fi

            echo $model $infer_mode
            echo $LOG_FILE


            CUDA_VISIBLE_DEVICES=$device $PYTHON -m NPS.main --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=$dimension --nfeat_in=1 --periodic --pointgroup=1 \
 --dim=3 --periodic \
 --model=NPS.model.MeshGraphNets --gnnmodel=$model --autoencoder=rev2wae --feat_out_method=id --trainer=NPS.model.MeshGraphNets --act=relu --nfeat_hid=$nhid \
 --register_args=NPS.model.MeshGraphNets.register_cond \
 --n_mpassing=$nmp --nstrides_2wae=$ae_stride --nblocks_2wae=$ae_block --nlayer_mlp_encdec=$nencdec \
 --optimizer=adamw \
 --batch=$batch --lr=$lr --n_in=${n_in} --n_out=${n_out} --noise_op="add_normal/0:1/$noise" --nepoch=10 --epoch_size=2000 --n_out_valid=$n_out --infer_mode=$infer_mode --loss=$loss \
 --print_freq=500 --valid_freq=1 --scheduler=plateau --lr_decay_patience=8 --n_traj_out=100 \
 --n_out_predict=100 --clip_step_valid=60 --mode=train --log_mem &>>$LOG_FILE &
            device=$(((device+1)%4))

done
done
done
done
done
done
done
done
done
done
done
done
wait


