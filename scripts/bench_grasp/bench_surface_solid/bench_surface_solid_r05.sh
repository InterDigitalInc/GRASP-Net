PY_NAME="${HOME_DIR}/experiments/bench.py"

# Main configurations
CHECKPOINTS="${HOME_DIR}/results/grasp_surface_solid/r05.pth"
CHECKPOINT_NET_CONFIG="True"
CODEC_CONFIG="${HOME_DIR}/config/codec_config/grasp_surface.yaml"
INPUT="${HOME_DIR}/datasets/cat1/A/queen_0200.ply ${HOME_DIR}/datasets/cat1/A/soldier_vox10_0690.ply ${HOME_DIR}/datasets/cat1/A/Facade_00064_vox11.ply ${HOME_DIR}/datasets/cat1/A/dancer_vox11_00000001.ply ${HOME_DIR}/datasets/cat1/A/Thaidancer_viewdep_vox12.ply"
COMPUTE_D2="True"
MPEG_REPORT="mpeg_report.csv"
WRITE_PREFIX="grasp_"
PRINT_FREQ="1"
PC_WRITE_FREQ="-1"
TF_SUMMARY="False"
REMOVE_COMPRESSED_FILES="True"
PEAK_VALUE="1023 1023 2047 2047 4095"
BIT_DEPTH="10 10 11 11 12"
SLICE="0"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
