PY_NAME="${HOME_DIR}/experiments/bench.py"

# Main configurations
CHECKPOINTS="${HOME_DIR}/results/grasp_lidar_ford/r01.pth"
CHECKPOINT_NET_CONFIG="True"
CODEC_CONFIG="${HOME_DIR}/config/codec_config/grasp_ford.yaml"
INPUT="${HOME_DIR}/datasets/ford/ford_02_q1mm ${HOME_DIR}/datasets/ford/ford_03_q1mm"
# INPUT="${HOME_DIR}/datasets/ford/ford_02_q1mm/Ford_02_vox1mm-0100.ply ${HOME_DIR}/datasets/ford/ford_02_q1mm/Ford_02_vox1mm-0101.ply ${HOME_DIR}/datasets/ford/ford_02_q1mm/Ford_02_vox1mm-1599.ply ${HOME_DIR}/datasets/ford/ford_03_q1mm/Ford_03_vox1mm-0200.ply ${HOME_DIR}/datasets/ford/ford_03_q1mm/Ford_03_vox1mm-0201.ply ${HOME_DIR}/datasets/ford/ford_03_q1mm/Ford_03_vox1mm-1699.ply"
COMPUTE_D2="True"
MPEG_REPORT="mpeg_report.csv"
MPEG_REPORT_SEQUENCE="True" # view the input point clouds as sequences
WRITE_PREFIX="grasp_"
PRINT_FREQ="1"
PC_WRITE_FREQ="-1"
TF_SUMMARY="False"
REMOVE_COMPRESSED_FILES="True"
PEAK_VALUE="30000"
BIT_DEPTH="18"
SLICE="0"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
