# Source global definitions
if [ -f /etc/bashrc ]; then . /etc/bashrc; fi



# set unicode
export LC_ALL="en_US.UTF-8"
export LANG="en_US.utf-8"

# Prompt name format
#export PS1="\e[1;35m[\u@\h \W]\$ \e[m "
export PS1="\[\033[01;35m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "

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





# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=
# ======================================================================


##---------------------- Aliases -----------------------##

alias bashrc='vi ~/$BASHRC'
alias cp='cp -i'
alias d='ls -ltrh'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'
alias grep='grep --color=auto'
alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'
alias mv='mv -i'
alias p='pwd'
alias rm='rm -i'
alias sbashrc='source ~/$BASHRC'
alias sbashpr='source ~/.bash_profile'
alias sinsarrc='source ~/apps/conda_envs/insar/config.rc'
alias sfringerc='source ~/apps/conda_envs/fringe/config.rc'
alias ssh='ssh -X'
alias vi='vim'
alias which='alias | /usr/bin/which --tty-only --read-alias --show-dot --show-tilde'

##----------------------- Jupyter -------------------------##
function jpn(){
    jupyter notebook --no-browser --port=$1
}

function jpl(){
    jupyter lab --no-browser --port=$1
}

function tport(){
    ssh -N -f -L localhost:$2:localhost:$1 $3
}

function kport(){
    port=$(ps aux|grep ssh|grep localhost|grep $1|awk '{print $2}')
    if [ -z "$port" ]
    then
            echo '>> No forwarded port:'$1
            echo '>> Check all SSH here:'
            ps aux|grep ssh
    else
            while true; do
                echo '>> Found process: '
                echo '>>' $(ps aux|grep ssh|grep localhost|grep $1)
                read -p ">> Do you wish to stop this process? [y]/n: " yn
                case $yn in
                 [Yy]* ) kill $port; break;;
                 [Nn]* ) break;;
                 * ) echo "Enter [y] or [n].";;
                esac
            done
    fi
}

##---------------------- Anaconda -----------------------##
export TOOL_DIR=~/apps
export CONDAPATH=${TOOL_DIR}/Anaconda
export PATH=${PATH}:$CONDAPATH/bin
export PATH=${PATH}:$CONDAPATH/condabin
export PYTHON3DIR=${TOOL_DIR}/Anaconda
export PATH=${PATH}:${PYTHON3DIR}/bin



# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ykliu/apps/Anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ykliu/apps/Anaconda/etc/profile.d/conda.sh" ]; then
        . "/home/ykliu/apps/Anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/home/ykliu/apps/Anaconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

