import subprocess
import time
import csv
import os
import glob

PURPLE = '\033[0;35m'
NC = '\033[0m'  # No color

def run_script(script_path, language):
    start_time = time.time()
    try:
        # Run the script and capture its output
        result = subprocess.run([language, script_path], capture_output=True, text=True, check=True)
        return result.stdout, time.time() - start_time
    except subprocess.CalledProcessError as e:
        return "Error: {e.stderr}".format(time.time() - start_time)

def generate_neoml_scripts():
    path_benchmark = os.getcwd()
    compiled_programs_dir = os.path.join(path_benchmark, "compiled_programs")

    # Delete and create the compiled_programs directory
    for ext in ['py', 'r']:
        pattern = os.path.join(compiled_programs_dir, f"*.{ext}")
        for file_path in glob.glob(pattern):
            os.remove(file_path)

    # Build NeoML and generate scripts
    os.chdir("../NeoML/")
    #subprocess.run(["npm", "run", "build"], check=True)

    for neoml_script in glob.glob("../Programs_examples/complete_*.neoml"):
        script_name = os.path.basename(neoml_script).replace(".neoml", "")
        generate_command = ["./bin/cli.js", "generate", "-d", compiled_programs_dir, "-l", "Python", neoml_script]
        subprocess.run(generate_command, check=True)
        generate_command = ["./bin/cli.js", "generate", "-d", compiled_programs_dir, "-l", "R", neoml_script]
        subprocess.run(generate_command, check=True)

def main():
    generate_neoml_scripts()

    print(f"{PURPLE}Running programs and generating benchmark{NC}")
    script_directory = "./compiled_programs"
    output_file = "output.csv"
    os.chdir("../benchmark")

    with open(output_file, 'w') as csvfile:
        fieldnames = ['Benchmark', 'Variant', 'Result', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header to the CSV file
        writer.writeheader()

        # Iterate over files in the script directory
        for script_name in os.listdir(script_directory):
            if script_name.endswith(".py") or script_name.endswith(".r"):
                script_path = os.path.join(script_directory, script_name)

                # Run Python script
                if script_name.endswith(".py"):
                    output, execution_time = run_script(script_path, 'python')
                    language = 'Python'
                elif script_name.endswith(".r"):
                    # Run R script
                    output, execution_time = run_script(script_path, 'Rscript')
                    language = 'R'

                # Write the data to the CSV file
                writer.writerow({
                    'Benchmark': script_name,
                    'Variant': language,
                    'Result': output,
                    'Time': execution_time
                })

if __name__ == "__main__":
    main()
