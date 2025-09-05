#!/usr/bin/env bash
set -e

######### 可按需修改 #########
SESSION="mmkcat_train"
ENV_NAME="eitlem_env"                     # 你的conda环境名
# 若你的miniforge/miniconda路径不同，请改这一行：
CONDA_SH="/root/miniforge3/etc/profile.d/conda.sh"

ROOT="/workspace/EITLEM-Kinetics"
LOGDIR="${ROOT}/Resultsmm1/KCAT/MMKcat-KCAT-train/logs"   # 用 Resultsmm1
OUTFILE="${ROOT}/Resultsmm1/train.out"

PORT=6006
CUDA=0
CMD="python iter_train_scripts.py -i 1 -t MMKCAT -m MACCSKeys -d ${CUDA} --mm_root ../Data --log10 False --epochs 100 --batch_size 32"

################################

mkdir -p "${LOGDIR}"
mkdir -p "$(dirname "${OUTFILE}")"

# 如果已有会话，先杀掉
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}"

# 启动一个新的会话（后台）
tmux new-session -d -s "${SESSION}"

# 一个帮助函数：在 pane 里以登录方式执行命令 + 激活环境
run_in_pane () {
  local target="$1"; shift
  local cmd="$*"
  tmux send-keys -t "${target}" "bash -lc 'source ${CONDA_SH} && conda activate ${ENV_NAME} && ${cmd}'" C-m
}

######## Pane 1: 训练 ########
run_in_pane "${SESSION}" "cd ${ROOT}/Code && ${CMD} 2>&1 | tee -a ${OUTFILE}"

######## Pane 2: TensorBoard ########
tmux split-window -h -t "${SESSION}"
run_in_pane "${SESSION}.1" "cd ${LOGDIR} && python -c \"import tensorboard\" >/dev/null 2>&1 || pip install -U tensorboard && tensorboard --logdir . --port ${PORT} --bind_all"

######## Pane 3: 监控（htop缺就用top） ########
tmux split-window -v -t "${SESSION}.0"
run_in_pane "${SESSION}.2" "command -v htop >/dev/null 2>&1 || (apt-get update && apt-get install -y htop || true); (htop || top)"

######## Pane 4: 实时看训练日志 ########
tmux split-window -v -t "${SESSION}.1"
run_in_pane "${SESSION}.3" "touch ${OUTFILE}; tail -f ${OUTFILE}"

tmux select-layout -t "${SESSION}" tiled
tmux attach -t "${SESSION}"
