# Setting up kamb & kraken for InSAR

We will use Python and Jupyter Notebooks for computation, Python's matplotlib module for simple plots and Generic Mapping Tools (GMT) for plotting maps.

Here are some tips of setting up the computer (Kamb) ready for the InSAR related works

Things that will be covered here:

- Paths and environments settings on Kamb (Mark's server that we use for heavy computational work)
- Setup to run Jupyter notebooks on Kamb
  - Jupyter notebook provides a Matlab-like interface to run python code and display results and plots.
  - There are several useful notebooks for the tutorials of `ARIA-tools` and `MintPy` available provided by the developers.
- Installation of `ARIA-tools` and `MintPy` 



<br />

# 0. SSH to Kamb

In order get into the Kamb server, you will need to do the ssh every time
Such as `ssh username@kamb.gps.caltech.edu`
And you need to enter the password every time

There is a way to avoid entering password every time

Checkout this link (step 1 to 3):

[https://phoenixnap.com/kb/setup-passwordless-ssh](https://phoenixnap.com/kb/setup-passwordless-ssh)

We still need to type the full IP address  `kamb.gps.caltech.edu` when doing ssh.

To further save some effort, we can create SSH alias names. Do the following on our laptop:

```bash
# First open the terminal on your laptop

# go into .ssh directory (if not exist, make one)
cd ~/.ssh

# create or edit a config file
vi config

## Paste the following in that config file

# KAMB server: Mark Simon's group
Host kamb
    Hostname kamb.gps.caltech.edu
    User my_name
    ServerAliveInterval 720
    ServerAliveCountMax 120
    ForwardX11Timeout = 24h

# EARTH server: Caltech GPS server
Host earth
    Hostname earth.gps.caltech.edu
    User my_name
    ServerAliveInterval 720
    ServerAliveCountMax 120
    ForwardX11Timeout = 24h
```

Now, I can log into Kamb by simply doing `ssh kamb`

Without having to type the full IP address and the password every time.



<br />



# 1. Access to Kraken (create your home directory there)

```bash
# SSH to Kamb server, you will be at your home directory on Kamb
ssh username@kamb.gps.caltech.edu

# Go kraken, the no backup directory for storting large dataset (up to 20TB)
cd /net/kraken/nobak/

# make your own folder here
mkdir my_name

# make a symbolic link of this path to your home directory
ln -s /net/kraken/nobak/my_name ~/kraken-nobak

# Go to kraken, the 60-day backup directory for storing codes, notes, etc (up to 5TB)

# make your own folder here
mkdir my_name

# make a symbolic link of this path to your home directory
ln -s /net/kraken/bak/my_name ~/kraken-bak

## Now everytime you log on kamb, you will see and can `cd` into kraken (either bak or nobak)
## easily without needing to type the full paths to access them.
```



<br />



# 2. Install Miniconda on Kamb

```bash
# Go to kamb (a lazy way here)
ssh kamb

# Create a tools directory (for all the downloaded softwares, source codes) at your home directory on Kamb
# -p command: make this directory if not exist. Then we `cd` into tools folder
mkdir -p ~/tools; cd ~/tools

# download (mini/ana)conda. Miniconda is better, a light version without many useless stuff
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# make the bash script executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# run it to install
./Miniconda3-latest-Linux-x86_64.sh -b -p ~/tools/miniconda3
~/tools/miniconda3/bin/conda init bash

# Close and restart the shell for changes to take effect.
exit                 # logout
ssh kamb				     # login again

# First, set conda-forge as default conda channels 
conda config --add channels conda-forge
conda config --set channel_priority strict

# now you can make your new conda env and install stuff under that env
conda create --name your_new_env
conda activate your_new_env
conda install some_packages_you_want

```

Reference of the above guidance: [Yunjun's GitHub repository](https://github.com/yuankailiu/conda_envs) (Yunjun is the postdoc here. He is the developer of MintPy. so, we will use this site again when installing MintPy)



# 3. Git

We will use `git` to get the source codes of `ARIA-tools` (getting the InSAR unwrapped datasets) and `MintPy` (doing InSAR time-series analysis).

[git](https://git-scm.com/) is an open source [version control system](https://www.atlassian.com/git/tutorials/what-is-version-control). You can find some very basic tutorials online if you are bored. Otherwise, we will only use the very basic stuff of `git` (e.g. how to download source codes of a open source software). The things explained in this document about `git` will be enough for us now.

You might already have `git` installed on Kamb. If not, simply install it in the `base` conda env

```bash
# Go to Kamb
ssh kamb

# You are now at the base env. The prompt will show (base)

# Install git via conda
conda install git

# check git version (you should have version >= 2.30.2)
git version
```



Now we can downlaod the "softwares" of ARIA-tools and MintPy. They are just a bunch of source codes written in Python, C, etc

```bash
# Go to the tools folder on Kamb
cd ~/tools

# use git clone to download these open source softwares reposited at GitHub
git clone https://github.com/aria-tools/ARIA-tools.git
git clone https://github.com/insarlab/MintPy.git
git clone https://github.com/insarlab/PySolid.git
git clone https://github.com/yunjunz/PyAPS.git

# Now you can see these folders (ARIA_tools, MintPy, PySolid, PyAPS) under tools directory
ls
```



Next, we will start to prepare the python environment and install some pre-requisites for running the above softwares.



# 4. Setting up `conda` Python environment

Create an environment with packages needed for `ARIA-tools` and `MintPy`

```bash
# First, the env you want to launch Jupyter (usually the base env) should intalled with nb_conda
conda activate base
conda install --channel conda-forge nb_conda

# Create a new env for the InSAR softwares (ARIA-tools and MintPy)
conda create -n insar    # create the env called "insar"
conda activate insar		 # warm up the "insar" env

# Install some packages:
# - python
# - numpy
# - matplotlib: matlab-like plotting in python
# - jupyter: jupyter notebook
# - jupyterlab: jupyter lab (an advanced, powered version of jupyter notebook)
conda install --channel conda-forge --yes python numpy matplotlib jupyter jupyterlab

# Install the packages required by those softwares (ARIA-tools, MintPy)
conda install --channel conda-forge --yes --file MintPy/docs/conda.txt --file ARIA-tools/requirements.txt

# install dependencies not available from conda
ln -s ${CONDA_PREFIX}/bin/cython ${CONDA_PREFIX}/bin/cython3
$CONDA_PREFIX/bin/pip install git+https://github.com/tylere/pykml.git
$CONDA_PREFIX/bin/pip install scalene      # CPU, GPU and memory profiler
$CONDA_PREFIX/bin/pip install ipynb        # import functions from ipynb files

# compile PySolid
cd ~/tools/PySolid/pysolid
f2py -c -m solid solid.for

```



Set the following environment variables (paths) in your source file (e.g. `~/.bashrc`  on Kamb). So that when you log in Kamb, you can call functions at those paths.

```bash
# Define the root directory
export TOOL_DIR=~/tools
export DATA_DIR=~/data   # data / nobak

if [ -z ${PYTHONPATH+x} ]; then export PYTHONPATH=""; fi

##-------------- MintPy / PyAPS / PySolid -------------##
export MINTPY_HOME=${TOOL_DIR}/MintPy
export PYTHONPATH=${PYTHONPATH}:${MINTPY_HOME}:${TOOL_DIR}/PyAPS:${TOOL_DIR}/PySolid
export PATH=${PATH}:${MINTPY_HOME}/mintpy
export WEATHER_DIR=${DATA_DIR}/aux

##-------------- ARIA-tools ---------------------------##
export ARIATOOLS_HOME=${TOOL_DIR}/ARIA-tools/tools
export PYTHONPATH=${PYTHONPATH}:${ARIATOOLS_HOME}
export PATH=${PATH}:${ARIATOOLS_HOME}/bin
```



Now, exit Kamb, close the terminal. And log in again to see if all the installations work.

```bash
# disconnect kamb, or just close the terminal
exit

# re-login
ssh kamb

# first activate the `insar` conda env
# you must activate it every time you log in to use this env
conda activate insar

# testing these softwares
ariaDownload.py -h       # test ARIA-tools
smallbaselineApp.py -h   # test MintPy
solid_earth_tides.py -h  # test PySolid
tropo_pyaps3.py -h       # test PyAPS

```



Hope the above works!



# 5. How to use Jupyter Notebook on Kamb

The most performant way to do Jupyter on Kamb is to [forward a webpage port](https://linuxize.com/post/how-to-setup-ssh-tunneling/#local-port-forwarding) (independently of X11), such that the browser is local to your computer, but gets the data through the tunnel from the server.

With SSH to KAMB, we can have Jupyter Notebooks open locally (on a laptop browser), while all the computations is done on the remote server. 

This Jupyter instance will only run as long as you have your SSH connection and shell open. If you want to keep Jupyter running even while you're logged out, you can open a [screen instance](https://linuxize.com/post/how-to-use-linux-screen/), and run the Jupyter command in there (that's what I do). Simply detach the screen and it'll stay running in the background. The next time you SSH into the machine, just open that same link as before, and your Jupyter process will be ready where you left it off.



By defining some bash aliases, we can achieve what we want easily:

1. First, open a terminal on your laptop
2. Open the ~/.bashrc by doing `vi ~/.bashrc`
3. Append the below content at the end of  your ~/.bashrc file

```bash
##-----------USINGING JUPYTER NOTEBOOKS FROM REMOTE------------##
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

```

5. Do the same thing on the ~/.bashrc file on your KAMB account



Functions explanations:

- `jpn 5550 `: Run **Jupyter Notebook** in a specified port number #5550 in the background without opening it (execute on the remote machine)
- `jpl 5550`: Run **Jupyter Lab** instead (execute on the remote machine)
- `tport 5550 5551 username@kamb`: Forward the remote port #5550 to a local port #5551 via a tunnel ssh to that user on KAMB (execute on the local machine)
- `kport 5551`: kill the local port #5551 (execute on the local machine)



After these function definitions, we can do the following to control the Jupyter Notebook remotely:

1. Go to KAMB. `ssh username@kamb`
2. Open a "screen instance" and name it "jupyter". `screen -S jupyter`
3. Run Jupyer Notebook in the background. `jpn 5550`
4. You can now close the screen (or even exit Kamb)
5. On your laptop, do `tport 5550 5551` to forward the port
6. On the laptop, open your web browser and enter the url `http://localhost:5551/`
7. Now you can play the Notebook!



# 6. Download products from Earthdata

1. Make sure you have an account for Earthdata Login: https://urs.earthdata.nasa.gov/profile

2. Go to `Applications` > `Authorized Apps`
3. Manually add and authorize the following apps:

![image-20210623143401112](/Users/ykliu/Library/Application Support/typora-user-images/image-20210623143401112.png)



4. In order to [access data over HTTP from a web server with curls and wget](https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget), we need to enter our account name and password every time. We can configure the account credentials in a file for automatic authentication (no need for username and password every time). Please do the following:

   ```bash
   # go to your home dir
   cd ~
   
   # make a .netrc file
   touch .netrc
   
   # type in your account info, save into that .netrc file
   echo "machine urs.earthdata.nasa.gov login uid_goes_here password password_goes_here" > .netrc
   
   # change permission: only you can read and write
   chmod 0600 .netrc
   
   # Create a cookie file
   # This will be used to persist sessions across individual cURL/Wget calls, making it more efficient.
   cd ~
   touch .urs_cookies
   ```

   + Now, `exit` the Kamb server and re-login again. Run the downlaod codes, see if you can download files without entering account info.









# More TBA...

## How to set .bashrc or .bash_profile on Kamb

- Path and format settings on the server
- Some aliases of bash commands run in your terminal

Add below (if they don't exist) to `~/.bashrc`

```
# .bashrc

## Source global definitions
if [ -f /etc/bashrc ]; then . /etc/bashrc; fi

## set unicode
export LC_ALL="en_US.UTF-8"
export LANG="en_US.utf-8"

## Prompt name format
export PS1="\[\033[01;35m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "


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


##----------------- Set path for bin ---------------------##
# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi


##---------------------- Conda -----------------------##
# conda installed under: /home/ykliu/apps/miniconda3
export TOOL_DIR=~/apps
export CONDAPATH=${TOOL_DIR}/miniconda3
export PATH=${PATH}:$CONDAPATH/bin
export PATH=${PATH}:$CONDAPATH/condabin
export PYTHON3DIR=${TOOL_DIR}/miniconda3
export PATH=${PATH}:${PYTHON3DIR}/bin
```



## Installation of ARIA-tools 

already covered above. More details here:
The GitHub page of ARIA-tools: https://github.com/aria-tools/ARIA-tools


## Intallation of MintPy

already covered above. More details here:
The GitHub page of MintPy: https://github.com/insarlab/MintPy

