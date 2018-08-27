python SVM_syn.py --job_name="ps" --task_index=0 &
python SVM_syn.py --job_name="worker" --task_index=0 --delay=0.01&
python SVM_syn.py --job_name="worker" --task_index=1 --delay=0.012&
python SVM_syn.py --job_name="worker" --task_index=2 --delay=0.015&
