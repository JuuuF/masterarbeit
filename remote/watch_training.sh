#1/bin/sh

history_path="dump/training_history.png"
prediction_path="dump/prediction.png"
prev_hash_history=""
prev_hash_prediction=""
prev_hash_weights=""

while true; do
    # echo "-------------------------------------------------------------"
    # History
    new_hash_history=$(ssh ma_kiel md5sum masterarbeit/$history_path | awk '{print $1}')
    # echo "prev hash: $prev_hash_history - new hash: $new_hash_history"
    if [ "$prev_hash_history" != "$new_hash_history" ]; then
        scp ma_kiel:/home/jfuerstenwerth/masterarbeit/$history_path $history_path
        scp ma_kiel:/home/jfuerstenwerth/masterarbeit/dump/training_history.pkl dump/training_history.pkl
        prev_hash_history="$new_hash_history"
    fi

    # Prediction
    new_hash_prediction=$(ssh ma_kiel md5sum masterarbeit/$prediction_path | awk '{print $1}')
    # echo "prev hash: $prev_hash_prediction - new hash: $new_hash_prediction"
    if [ "$prev_hash_prediction" != "$new_hash_prediction" ]; then
        rsync -avz ma_kiel:/home/jfuerstenwerth/masterarbeit/$prediction_path dump/
        prev_hash_prediction="$new_hash_prediction"
    fi

    # Find latest checkpoint directory
    latest_dir_weights=$(ssh ma_kiel "ls -1t masterarbeit/data/ai/checkpoints/darts | head -n 1")
    new_hash_weights=$(ssh ma_kiel md5sum masterarbeit/data/ai/checkpoints/darts/$latest_dir_weights/latest.weights.h5 | awk '{print $1}')
    # echo "prev hash: $prev_hash_weights - new hash: $new_hash_weights"
    if [ "$prev_hash_weights" != "$new_hash_weights" ]; then
        rsync -avz --progress ma_kiel:masterarbeit/data/ai/checkpoints/darts/$latest_dir_weights/latest.weights.h5 data/ai/darts/
        prev_hash_weights="$new_hash_weights"
    fi
    sleep 2
done
