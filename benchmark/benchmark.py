import subprocess
import time
import csv
import os
import glob
import statistics
import numpy as np

PURPLE = '\033[0;35m'
NC = '\033[0m'  # No color

def run_script(script_path, language):
    start_time = time.time()
    try:
        # Run the script and capture its output
        result = subprocess.run([language, script_path], capture_output=True, text=True, check=True)
        return result.stdout, time.time() - start_time
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}".format(time.time() - start_time))

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

def generate_with_fuzzer():
    os.chdir("../fuzzer")
    fuzzer_command = ["./fuzzer_generate-and-compile.sh", "20", "clean", "compile", "norun"]
    subprocess.run(fuzzer_command, check=True)

def main():
    generate_neoml_scripts()
    generate_with_fuzzer()

    print(f"{PURPLE}Running programs and generating benchmark{NC}")
    script_directories = ["./compiled_programs", "../fuzzer/compiled_programs"]
    output_file = "output.csv"
    os.chdir("../benchmark")

    with open(output_file, 'w') as csvfile:
        fieldnames = ['Benchmark', 'Result_Python', 'Time_Python', 'Result_R', 'Time_R']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header to the CSV file
        writer.writeheader()

        exec_times_py = []
        exec_times_r = []

        # Iterate over files in the script directory
        for script_directory in script_directories:
            for script_name in os.listdir(script_directory):
                if script_name.endswith(".py"):
                    script_path_py = os.path.join(script_directory, script_name)
                    script_path_r = os.path.join(script_directory, os.path.splitext(script_name)[0] + '.r')

                    # Run Python script
                    output_py, execution_time_py = run_script(script_path_py, 'python3')
                    print(f"{PURPLE}Runned " + script_path_py)
                    exec_times_py += [execution_time_py]
                    # Run R script
                    output_r, execution_time_r = run_script(script_path_r, 'Rscript')
                    print(f"{PURPLE}Runned " + script_path_r)
                    exec_times_r += [execution_time_r]

                    # Write the data to the CSV file
                    writer.writerow({
                        'Benchmark': os.path.splitext(script_name)[0]+'.neoml',
                        'Result_Python': output_py,
                        'Time_Python': execution_time_py,
                        'Result_R': output_r,
                        'Time_R': execution_time_r
                    })
        
        writer.writerow({
            'Benchmark': '',
            'Result_Python': '',
            'Time_Python': '',
            'Result_R': '',
            'Time_R': ''
        })

        writer.writerow({
            'Benchmark': 'Means',
            'Result_Python': '',
            'Time_Python': statistics.mean(exec_times_py),
            'Result_R': '',
            'Time_R': statistics.mean(exec_times_r)
        })

        writer.writerow({
            'Benchmark': 'Variances',
            'Result_Python': '',
            'Time_Python': np.var(exec_times_py),
            'Result_R': '',
            'Time_R': np.var(exec_times_r)
        })

if __name__ == "__main__":
    main()
