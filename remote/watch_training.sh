#1/bin/sh

prev_hash=""

while true; do
    new_hash=$(ssh ma_kiel md5sum masterarbeit/dump/training_history.png | awk '{print $1}')

    if [ "$prev_hash" != "$new_hash" ]; then
        scp ma_kiel:/home/jfuerstenwerth/masterarbeit/dump/training_history.png dump/training_history.png
        scp ma_kiel:/home/jfuerstenwerth/masterarbeit/dump/training_history.pkl dump/training_history.pkl

        # Find latest checkpoint
        latest_dir=$(ssh ma_kiel "ls -1t masterarbeit/data/ai/checkpoints/darts | head -n 1")
        rsync -avz --progress ma_kiel:masterarbeit/data/ai/checkpoints/darts/$latest_dir/latest.weights.h5 data/ai/darts/

        prev_hash="$new_hash"
    fi
    sleep 2
done
