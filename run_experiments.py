import yaml
import subprocess
import argparse
import os


def run_script(script_path, parameters):
	"""
	Run a script with the specified parameters.
	"""
	script_dir, script_name = os.path.dirname(script_path), os.path.basename(script_path)
	cmd = ['python', script_name]
	if parameters:
		for key, value in parameters.items():
			if key == 'save_all': # save_all is a flag
				if value:
					cmd.append(f"--{key}")
			else: # other parameters are key-value pairs
				cmd.append(f"--{key}")
				cmd.append(str(value))
	print(f"Running: {' '.join(cmd)}")
	subprocess.run(cmd, check=True, cwd=script_dir)


def run_experiment(experiment, script_types):
	"""
	Run an experiment.
	"""
	print(f"Running experiment: {experiment['name']}")
	for script_type in script_types:
		script_info = experiment.get(script_type)
		if script_info:
			print(f"Running {script_type} for {experiment['name']}")
			run_script(script_info['script'], script_info['parameters'])


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description="Run experiments.")
	
	parser.add_argument('--config', type=str, default="config/experiments.yml", help="Path to the YAML configuration file.")
	parser.add_argument('--experiment', type=str, nargs='*', help="Name(s) of the experiment(s) to run. If not specified, all experiments are run.")
	parser.add_argument('--bayesian', action='store_true', help="Run the Bayesian version of the evaluation script.")

	args = parser.parse_args()

	# Load configuration
	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)

	experiments = config['experiments']
	script_types = ['evaluate', 'evaluate_bayesian']

	if args.experiment:
		experiments_to_run = [exp for exp in experiments if exp['name'] in args.experiment]
		if not experiments_to_run:
			print("No matching experiments found.")
			exit()
		script_types = ['evaluate_bayesian'] if args.bayesian else ['evaluate']
	else:
		experiments_to_run = experiments

	# Run the selected experiments
	for experiment in experiments_to_run:
		run_experiment(experiment, script_types)
