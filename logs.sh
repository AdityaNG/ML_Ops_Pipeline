session="mlops"

# Check if the session exists, discarding output
# We can check $? for the exit status (zero for success, non-zero for failure)
tmux has-session -t $session 2>/dev/null

if [ $? != 0 ]; then
	# Set up your session
	if tmux info &> /dev/null; then 
		echo "tmux running"
	else
		echo "tmux not running, starting now"
		tmux new
	fi
	
	tmux source-file mlops.tmux
else
	# Attach to created session
	tmux attach-session -t $session
fi

