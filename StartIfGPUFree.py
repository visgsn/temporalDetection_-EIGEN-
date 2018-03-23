'''
    USAGE: python StartToTrainIfGPUFree <Script_to_start_on_a_GPU.py>

    (Insert e.g.

        import sys
        <gpus_var> = str(sys.argv[1])   #Adapted to use with script "StartIfGPUFree.py" # GPU to use for execution

    where the desired GPU is selected IN YOUR PROGRAM!)
'''

import logging
import subprocess
import sys
from time import sleep


### Configuration variables ###
# Define all possible GPUs for your process as list, e.g. ["0", "1"]
possible_gpus =     ["0", "1", "2", "3"]    # ***deneb***
#possible_gpus =     ["0"]                   # ***sadr***    (Indeces twisted! "0"-->GPU#1, "1"-->GPU#0)

check_interval =    10                      # Time interval to check if any GPU is free
logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)



def main():
    # Wait until one of the GPUs is free (no process started)
    while True:
        print "----------------------------------------------------------------"
        # Variable declarations
        remaining_gpus = possible_gpus[:]
        start_line_found = False

        # Test for free GPUs
        bashCommand = "nvidia-smi"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        output_saved = output[:]
        output = output.splitlines()
        for lineNr in range(0, len(output)):
            if "===============================================================" in output[lineNr]:
                start_line_found = True
                continue

            elif start_line_found:
                # Remove trailing characters from line with GPU-numbers
                stripped_output = output[lineNr].strip('| ')

                # Identify GPUs with processes
                logging.debug("### Stripped output: " + stripped_output[0])
                if stripped_output[0] == "+" and len(remaining_gpus) != 0:      # GPUs free! --> Execute Process
                    # Starting process on free GPU
                    print "******************************************************************"
                    logging.info("***** Starting process: " + str(sys.argv[1]) + " *****")
                    logging.info("On GPU #" + str(remaining_gpus[0]))
                    logging.debug("GPUs free! --> Execute Process")
                    logging.debug("List of remaining GPUs: " + str(remaining_gpus))
                    print "nvidia-smi output at startup:"
                    print str(output_saved)
                    bashCommand = "python " + str(sys.argv[1]) + " " + str(remaining_gpus[0])
                    process = subprocess.Popen(bashCommand.split(), stdout=None)
                    output_2, error_2 = process.communicate() # Waiting for process to finish!
                    print "******************************************************************"
                    logging.info("DONE")
                    return 0

                elif stripped_output[0] == "+" or len(remaining_gpus) == 0:     # No free GPUs --> Try again later
                    logging.info("No free GPUs --> Try again later")
                    logging.debug("List of remaining GPUs: " + str(remaining_gpus))
                    break

                else:                                                           # Check not finished... proceed!
                    logging.debug("Check not finished... proceed!")
                    logging.debug("List of remaining GPUs: " + str(remaining_gpus))
                    # Checking if certain GPU is busy. If yes --> remove
                    for single_gpu in remaining_gpus:
                        if single_gpu == stripped_output[0]:
                            logging.debug("Removing GPU #" + str(single_gpu))
                            remaining_gpus.remove(single_gpu)
                            break

        # Wait for specified time interval to check again!
        sleep(check_interval)


# MAIN
if __name__ == '__main__':
   main()