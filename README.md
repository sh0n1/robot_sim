Sequential Distance Transform 

To run the program use conda to create a virtual environment from environment.yml:
"conda create -f environment.yml"

on Windows,
Use Anaconda Prompt and navigate to the project  directory: e.g "cd ./user/Documents/robot_sim" 
then activate the conda environment you created: 
"conda activate sim_thing"

To benchmark, change the "N" variable to change  grid resolution, then run this line in the conda terminal:
python main.py --playback-speed=2 --perf-runs=10 --perf

optionally, if you don't want to see visualisation, turn off rendering matplotlib with:
python main.py --perf-runs=10 --perf --no-render

rendering gets pretty messy and takes very long so I would keep this off
