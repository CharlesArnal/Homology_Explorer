#!/bin/bash
#OAR -l /nodes=2,walltime=80:00:00
#
# The job is submitted to the default queue
#OAR -q default
#
#OAR -p cluster='dellc6420' or cluster='dellc6420b' or cluster='dellr940'
#

# source /etc/profile.d/modules.sh
# module load gcc/9.2.0
conda activate ML_RAG_2
python3 opti_experiment_2_RL_Rastrigin.py

# tester d'abord en interactif (taper "oarsub... etc.")

# pour depuis ma session sans oar : ~/polymake-install/bin/polymake

