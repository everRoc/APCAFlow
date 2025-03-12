# CUDA_VISIBLE_DEVICES=0 python evaluate.py --model models/apca-chairs.pth --dataset chairs
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --model models/apca-things.pth --dataset kitti
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model models/apca-things.pth --dataset sintel