# train
python train.py --weights ./weights/yolov5n.pt --cfg ./models/yolov5n.yaml --data ./data/armor.yaml --epochs 99 --imgsz 512 --batch-size 20 --adam

# train_sparity
python train_sparity.py  --sr 0.0001  --weights ./ --data ./data/armor.yaml  --adam  --epochs 99  --batch-size 20  --imgsz 512

# pruned_model
python prune.py  --percent 0.9  --weights runs/train/exp6/weights/last.pt  --data data/armor.yaml  --cfg models/yolov5n.yaml

# finetune
python finetune_pruned.py  --weights pruned_model.pt  --data data/armor.yaml  --epochs 100  --imgsz 512 --epochs 99 --batch-size 20 --adam

# detect
python detect.py --weights ./runs/train/exp6/weights/last.pt --source ./datasets/armor/images --imgsz 512

