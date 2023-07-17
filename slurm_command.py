import subprocess
from os import path as op

from importlib import reload

print(__doc__)

# wrapper to run python script via qsub. Python3
fname_wrap = op.join('/', 'home', 'fm02', 'Desktop', 'MEG_EOS_scripts',
                     'Python2SLURM.sh')

job_list = [
 #    # Neuromag Maxfilter
 #    {'N':   'SemCat_TimeGen',                  # job name
 #     'Py':  'final_individual_SemCat_TempGen_slurm',  # Python script
 #     'Ss':  list(range(0, 18)),                    # subject indices
 #     'mem': '16G',                   # memory for qsub process
	# },
    # {'N':   'SemCat_TimeGen_SVC',                  # job name
    #  'Py':  'final_individual_SemCat_TempGen_slurm_SVC',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '16G',                   # memory for qsub process
    # }
    # {'N':   'LDSD_TimeGen',                  # job name
    #  'Py':  'final_individual_LDvsSD_TempGen_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '16G',                   # memory for qsub process
    # },
    # {'N':   'semcat_conc',                  # job name
    #  'Py':  'concat_individual_SemCat_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'semcat_conc_timegen',                  # job name
    #  'Py':  'concat_individual_SemCat_TempGen',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'semcat_conc_patterns',                  # job name
    #  'Py':  'final_combined_SemCat_concat',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'semcat_NONconc_patterns',                  # job name
    #  'Py':  'final_combined_SemCat_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    #  {'N':   'semcat_confusion',                  # job name
    #  'Py':  'concat_individual_SemCat_confusion-matrix_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'semcat_conc2',                  # job name
    #  'Py':  'concat_individual_SemCat_concrete_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },

    # {'N':   'new_epochs',                  # job name
    #  'Py':  'create_metadata',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'mysources',                  # job name
    #  'Py':  'source_estimates',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'n_letters',                  # job name
    #  'Py':  'mysources_individual_SemCat_confusion-matrix_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'concVSabs',                  # job name
    #  'Py':  'mysources_concavs_confusion-matrix_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G'},                   # memory for qsub process
    {'N':   'concVSabC01s',                  # job name
     'Py':  'mysources_semcat_balanced_crossLDSD_confusion-matrix_slurm',  # Python script
     'Ss':  list(range(0, 18)),                    # subject indices
     'mem': '4G',                   # memory for qsub process
    },
    # {'N':   'scores_nonconcat',                  # job name
    #  'Py':  'mysources_concabs_balanced_scores_nonconcat',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
    # {'N':   'combined_balanced',                  # job name
    #  'Py':  'mysources_concabs_balanced_combined_slurm',  # Python script
    #  'Ss':  list(range(0, 18)),                    # subject indices
    #  'mem': '4G',                   # memory for qsub process
    # },
	]  # node constraint for MF, just picked one

# directory where python scripts are
dir_py = op.join('/', 'home', 'fm02', 'Decoding_SDLD', 'SourceSpaceDecoding_SDvsLD')

# directory for qsub output
dir_sbatch = op.join('/', 'home', 'fm02', 'Desktop', 'MEG_EOS_scripts',
                     'sbatch_out')

# keep track of qsub Job IDs
Job_IDs = {}

for job in job_list:

    for Ss in job['Ss']:

        Ss = str(Ss)  # turn into string for filenames etc.

        N = Ss + job['N']  # add number to front
        Py = op.join(dir_py, job['Py'])
        mem = job['mem']

        # files for qsub output
        file_out = op.join(dir_sbatch,
                           job['N'] + '_' + '-%s.out' % str(Ss))
        file_err = op.join(dir_sbatch,
                           job['N'] + '_' + '-%s.err' % str(Ss))


        # sbatch command string to be executed
        sbatch_cmd = 'sbatch \
                        -o %s \
                        -e %s \
                        --export=pycmd="%s.py",subj_idx=%s, \
                        --mem=%s -t 1-00:00:00 -J %s %s' \
                        % (file_out, file_err, Py,Ss, mem,
                           N, fname_wrap)

        # format string for display
        print_str = sbatch_cmd.replace(' ' * 25, '  ')
        print('\n%s\n' % print_str)

        # execute qsub command
        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, shell=True)

        # get linux output
        (out, err) = proc.communicate()

        # keep track of Job IDs from sbatch, for dependencies
        Job_IDs[N, Ss] = str(int(out.split()[-1]))
