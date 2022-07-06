#new-session mlops
#rename-session mlops
new-window "cd ~/bosch/MLOps_Pipeline/logs/ && tail -f stderr_model_analysis.log "
rename-window "model_analysis"
split-window -h "cd ~/bosch/MLOps_Pipeline/logs/ && tail -f stdout_model_analysis.log "
new-window "cd ~/bosch/MLOps_Pipeline/logs/ && tail -f stderr_model_visualizer_loop.log "
rename-window "model_visualizer"
split-window -h "cd ~/bosch/MLOps_Pipeline/logs/ && tail -f stdout_model_visualizer_loop.log " C-m
new-window "cd ~/bosch/MLOps_Pipeline/logs/ && tail -f stderr_model_training.log "
rename-window "model_training"
split-window -h "cd ~/bosch/MLOps_Pipeline/logs/ && tail -f stdout_model_training.log " C-m