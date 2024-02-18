# .bash_profile


## bash env for specific machines
ARIUS=arius
KAMB=kamb.gps.caltech.edu
HOST=$(hostname)

if [ $HOST == $KAMB ]
then
    echo "Log into: KAMB"
    BASHRC=.bashrc_kamb
    if [ -f $HOME/$BASHRC ]; then . $HOME/$BASHRC; fi
fi


if [ $HOST == $ARIUS ]
then
    echo "Log into: Arius"
    BASHRC=.bashrc_arius
    if [ -f $HOME/$BASHRC ]; then . $HOME/$BASHRC; fi
fi



# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi





# set unicode
export LC_ALL="en_US.UTF-8"
export LANG="en_US.utf-8"

# Prompt name format
#export PS1="\e[1;35m[\u@\h \W]\$ \e[m "
export PS1="\[\033[01;35m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "

# Get the aliases and functions
if [ -f ~/.bashrc ]; then . ~/.bashrc; fi

# echo "Use 'sbashrc' to source .bashrc"

# User specific environment and startup programs
export PATH=$PATH:$HOME/bin



##------------ Define environment variable ISCE_VERSION and create an alias -------------------##
# load isce2 core module with GPU support from Cunren's code
alias load_isce2_cunrenl='export ISCE_VERSION="_cunrenl"; source ~/apps/conda_envs/isce2/config.rc'

# load isce2 core module with GPU support installed by Lijun
alias load_isce2_lijunz='echo "load ISCE-2 compiled with GPU support from Lijun"; source /home/geomod/apps/rhel7/isce2.rc'

# load isce2 core module from a default jpl version (not installed)
alias load_isce2_jpl='echo "load ISCE-2 JPL default"; source ~/apps/conda_envs/insar/config.rc'

alias load_isce2_scons='echo "load ISCE-2 Scons built"; conda activate isce2; source /home/ykliu/.isce/config.rc'

##------------------------------- InSAR processing tools --------------------------------------##
# source InSAR conda env
conda activate insar

# ISCE
load_isce2_cunrenl

# FRinGE
source ~/apps/conda_envs/fringe/config.rc



