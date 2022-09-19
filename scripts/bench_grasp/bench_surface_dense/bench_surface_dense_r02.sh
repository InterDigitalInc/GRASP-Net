PY_NAME="${HOME_DIR}/experiments/bench.py"

# Main configurations
CHECKPOINTS="${HOME_DIR}/results/grasp_surface_dense/r02.pth"
CHECKPOINT_NET_CONFIG="True"
CODEC_CONFIG="${HOME_DIR}/config/codec_config/grasp_surface.yaml"
INPUT="${HOME_DIR}/datasets/cat1/A/soldier_viewdep_vox12.ply ${HOME_DIR}/datasets/cat1/A/boxer_viewdep_vox12.ply ${HOME_DIR}/datasets/cat1/A/Facade_00009_vox12.ply ${HOME_DIR}/datasets/cat1/A/House_without_roof_00057_vox12.ply"
COMPUTE_D2="True"
MPEG_REPORT="mpeg_report.csv"
WRITE_PREFIX="grasp_"
PRINT_FREQ="1"
PC_WRITE_FREQ="-1"
TF_SUMMARY="False"
REMOVE_COMPRESSED_FILES="True"
PEAK_VALUE="4095 4095 4095 4095"
BIT_DEPTH="12 12 12 12"
SLICE="0"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
