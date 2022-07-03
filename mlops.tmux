new-session mlops
rename-session mlops

new-window "cd ~/Downloads/ML_Ops_Pipeline/logs && tail -f stderr_model_analysis.log "
rename-window model_analysis
split-window -h
send "cd ~/Downloads/ML_Ops_Pipeline/logs && tail -f stdout_model_analysis.log " C-m

new-window "cd ~/Downloads/ML_Ops_Pipeline/logs && tail -f stderr_model_visualizer_loop.log "
rename-window model_visualizer
split-window -h
send "cd ~/Downloads/ML_Ops_Pipeline/logs && tail -f stdout_model_visualizer_loop.log " C-m

new-window "cd ~/Downloads/ML_Ops_Pipeline/logs && tail -f stderr_model_training.log "
rename-window model_training
split-window -h
send "cd ~/Downloads/ML_Ops_Pipeline/logs && tail -f stdout_model_training.log " C-m
