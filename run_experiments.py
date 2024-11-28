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
			cmd.append(f"--{key}")
			cmd.append(str(value))
	print(f"Running: {' '.join(cmd)}")
	subprocess.run(cmd, check=True, cwd=script_dir)


def run_experiment(experiment):
	"""
	Run an experiment.
	"""
	print(f"Running experiment: {experiment['name']}")
	for script_type in ['evaluate', 'evaluate_bayesian']:
		script_info = experiment.get(script_type)
		if script_info:
			print(f"Running {script_type} for {experiment['name']}")
			run_script(script_info['script'], script_info['parameters'])


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description="Run experiments.")
	
	parser.add_argument('--config', type=str, default="config/experiments.yml", help="Path to the YAML configuration file.")
	parser.add_argument('--experiment', type=str, nargs='*', help="Name(s) of the experiment(s) to run. If not specified, all experiments are run.")

	args = parser.parse_args()

    # Load configuration
	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)

	experiments = config['experiments']

	if args.experiment:
		experiments_to_run = [exp for exp in experiments if exp['name'] in args.experiment]
		if not experiments_to_run:
			print("No matching experiments found.")
			exit()
	else:
		experiments_to_run = experiments

	# Run the selected experiments
	for experiment in experiments_to_run:
		run_experiment(experiment)
