# Python script to create more pairs based on a file of date pairs and an exisiting directory of igram pairs set up for topsApp processing
# Works by making directorys and copying the reference and secondary files from the relevant folder
# Doesn't need processed igrams, just the xml files from Cunren's S1_pairs.py script
import os 
import glob
import shutil


# infile = 'annual_pairs.txt'
track = 'T13a'
process_dir = '/marmot-nobak/olstephe/InSAR/Makran/{}/process'.format(track)
fname = 'six_month_pairs.txt'
infile = os.path.join(process_dir,fname)

# Expected format: each line has format 'YYYYMMDD-YYYYMMDD'
# Refers to a specific directory in process_dir

f = open(infile,'r')

# Get pairs that have already been made (e.g. using S1 pairs)
existing_dates = glob.glob(os.path.join(process_dir,'*-*'))
print('Adding pairs')
for line in f:
    
    dates = line.rstrip().split('-')
    ref_date = dates[0]
    sec_date = dates[1]
    # Look for the order of the dash 
    ref_date += '-'
    sec_date = '-'+sec_date

    # Find existing directories
    ref_matches = [match for match in existing_dates if ref_date in match]
    sec_matches = [match for match in existing_dates if sec_date in match]
    ref_dir = ref_matches[0] # Get the first one that matches (possible problems if we've had a failed run of this script)
    sec_dir = sec_matches[0]
    
    # Make directory for new pair
    directory = os.path.join(process_dir,line.rstrip())
    if not os.path.exists(directory):
        os.mkdir(directory)
        print('Making: {}'.format(directory))
    else:
        print('{} already exists, skipping'.format(directory))
        continue

    shutil.copy(os.path.join(process_dir,ref_dir,'reference.xml'),directory)
    shutil.copy(os.path.join(process_dir,sec_dir,'secondary.xml'),directory)

print('Done')
