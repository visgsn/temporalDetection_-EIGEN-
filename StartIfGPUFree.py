'''
    USAGE: python StartToTrainIfGPUFree <Script_to_start_on_a_GPU.py> <GPU_list_to_use>

    (Insert e.g.

        import sys
        <gpus_var> = str(sys.argv[1])   #Adapted to use with script "StartIfGPUFree.py" # GPU to use for execution

    where the desired GPU is selected IN YOUR PROGRAM!)
'''

import logging
import subprocess
import sys
from time import sleep

possible_gpus = {}



##### Configuration variables ##########################################################################################
# Define all possible GPUs for your process as list, e.g. ["0", "1"], with the corresponding server name
possible_gpus['d']  = ["0", "1", "2", "3"]  # ***deneb***
possible_gpus['z']  = ["1", "3"]            # ***zaurak***
possible_gpus['m']  = ["0", "1"]            # ***minkar***
possible_gpus['s']  = ["0"]                 # ***sadr***    (Indeces twisted! "0"-->GPU#1, "1"-->GPU#0)
possible_gpus['h']  = ["0"]                 # ***at Home***

check_interval      = 5                     # Time interval to check if any GPU is free
interval_to_disp    = 12 * 5                # Display status every interval_to_disp check intervals
logging.basicConfig(format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
########################################################################################################################



# Choose possible GPUs from given list
possible_gpus_extracted = possible_gpus[str(sys.argv[2])]

def main():
    # Init counter
    counter = 0

    # Wait until one of the GPUs is free (no process started)
    while True:
        if counter % interval_to_disp == 0:  # Only print this e.g. once every minute
            print "\n----------------------------------------------------------------"
        # Variable declarations
        remaining_gpus = possible_gpus_extracted[:]
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
                    if counter % interval_to_disp == 0:  # Only print this e.g. once a minute
                        logging.info("No free GPUs --> Try again later")
                    sys.stdout.write('.')
                    sys.stdout.flush()
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
        counter = counter + 1


# MAIN
if __name__ == '__main__':
   main()