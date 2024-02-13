
#OUTPUT_DIR=work_dirs/online_v11
OUTPUT_DIR=/home/user/OnlineRefer_Modified/work_dirs/online_ytvos_r50

# training
python3 main.py --with_box_refine --binary --freeze_text_encoder \
--epochs 6 --lr_drop 3 5 \
--lr=1e-5 \
--lr_backbone=5e-6 \
--num_frames=2 \
--sampler_steps 4 \
--sampler_lengths 2 3 \
--sampler_interval=5 \
--output_dir=${OUTPUT_DIR} \
--online \
--use_checkpoint_for_more_frames \

#-m torch.distributed.launch --nproc_per_node=8 --master_port=29500 --use_env \
# inference
#CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
#python3 inference_ytvos_online.py --with_box_refine --binary --freeze_text_encoder \
#--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT} \
#--ngpu=8 \
#--online \
#
#echo "Working path is: ${OUTPUT_DIR}"