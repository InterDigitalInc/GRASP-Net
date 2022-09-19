PY_NAME="${HOME_DIR}/experiments/train.py"

# Main configurations
NET_CONFIG="${HOME_DIR}/config/net_config/grasp_dus1_faiss.yaml"
OPTIM_CONFIG="${HOME_DIR}/config/optim_config/optim_cd_sparse.yaml"
TRAIN_DATA_CONFIG="${HOME_DIR}/config/data_config/modelnet_voxel_solid.yaml train_cfg"
VAL_DATA_CONFIG="${HOME_DIR}/config/data_config/modelnet_voxel_solid.yaml val_cfg"

# Method-specific parameters
ALPHA="5" # Distortion trade-off
BETA="128" # Rate trade-off
SCALING_RATIO="0.125" # quantization ratio
POINT_MUL="150" # point multiplication, also the number of neighbors to search
SKIP_MODE="False" # skip mode

# Logging settings
PRINT_FREQ="20"
PC_WRITE_FREQ="-1"
TF_SUMMARY="True"
SAVE_CHECKPOINT_FREQ="1"
SAVE_CHECKPOINT_MAX="10"
VAL_FREQ="-1"
VAL_PRINT_FREQ="20"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
LOG_FILE_ONLY="False"
