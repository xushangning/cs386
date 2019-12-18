cd EDSR/src
echo "generating 4K from 1080p and 720p..."
python main.py --data_test Demo --scale 2 --pre_train download --test_only --save_results --dir_demo ../../images/1080p
mv ../experiment/test/results-Demo/ ../experiment/test/1080p
python main.py --data_test Demo --scale 3 --pre_train download --test_only --save_results --dir_demo ../../images/720p
mv ../experiment/test/results-Demo/ ../experiment/test/720p
