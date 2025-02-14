#1/bin/sh

while true; do
    ssh ma_kiel cp /home/jfuerstenwerth/masterarbeit/dump/pred.png /home/jfuerstenwerth/masterarbeit/dump/transfer.png
    scp ma_kiel:/home/jfuerstenwerth/masterarbeit/dump/transfer.png dump/pred.png
    ssh ma_kiel cp /home/jfuerstenwerth/masterarbeit/dump/training_history.png /home/jfuerstenwerth/masterarbeit/dump/transfer.png
    scp ma_kiel:/home/jfuerstenwerth/masterarbeit/dump/transfer.png dump/training_history.png
    sleep 60
done
