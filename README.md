迁移学习训练命令：
conda activate eitlem_env
cd /workspace/EITLEM-Kinetics/Code
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python iter_train_scripts.py -i 8 -t iterativeTrain -m MACCSKeys -d 0 \
  --batch_size 32 --num_workers 2 --prefetch_factor 2

tensorboard命令：
source ~/miniforge3/etc/profile.d/conda.sh
conda activate eitlem_env
cd /workspace/EITLEM-Kinetics/Code
tensorboard --logdir ../Results --port 6006 --bind_all

tmux命令:
tmux new-session -d -s eitlem_train "source activate eitlem_env && cd /workspace/EITLEM-Kinetics/Code && PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python iter_train_scripts.py -i 8 -t iterativeTrain -m MACCSKeys -d 0 --batch_size 32 --num_workers 2 --prefetch_factor 2"

进入会话：
tmux attach -t eitlem_train

退出但保持运行：
Ctrl+b  然后按 d

结束任务：
tmux kill-session -t eitlem_train