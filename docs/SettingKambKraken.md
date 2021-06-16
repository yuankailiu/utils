# Setting up kamb & kraken for InSAR



Here are some tips of setting up the computer (Kamb) ready for the InSAR related works

Things that will be covered here:

- Paths and environments settings on Kamb (Mark's server that we use for heavy computational work)
- Setup to run Jupyter notebooks on Kamb
  - Jupyter notebook provides a Matlab-like interface to run python code and display results and plots.
  - There are several useful notebooks for the tutorials of `ARIA-tools` and `MintPy` available provided by the developers.
- Installation of `ARIA-tools` and `MintPy` 



<br />



# SSH to Kamb

In order get into the Kamb server, you will need to do the ssh every time
Such as `ssh username@kamb.gps.caltech.edu`
And you need to enter the password every time

There is a way to avoid entering password every time

Checkout this link (step 1 to 3 can get the work done!):

[https://phoenixnap.com/kb/setup-passwordless-ssh](https://phoenixnap.com/kb/setup-passwordless-ssh)

But, we still need to type the full IP address  `kamb.gps.caltech.edu` when doing ssh.

To further save some effort, we can create SSH alias names. Do the following on our laptop:

```bash
# First open the terminal on your laptop

# go into .ssh directory (if not exist, make one)
cd ~/.ssh

# create a config file if not exist
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

Now, I can log into Kamb by simply doing `ssh username@kamb`

Without having to type the full IP address and the password every time.



<br />



# Access to Kraken

```bash
# SSH to Kamb server, you will be at your home directory on Kamb
ssh username@kamb.gps.caltech.edu

# Go kraken, the no backup directory for storting large dataset (up to 20TB)
cd /net/kraken/nobak/

# make your own folder here, go into it
mkdir my_name; cd my_name

# make a symbolic link of this path to your home directory
ln -s . ~/kraken-nobak

# Go to kraken, the 60-day backup directory for storing codes, notes, etc (up to 5TB)

# make your own folder here, go into it
mkdir my_name; cd my_name

# make a symbolic link of this path to your home directory
ln -s . ~/kraken-bak

## Now everytime you log on kamb, you will see and can `cd` into kraken (either bak or nobak)
## easily without needing to type the full paths to access them.
```



<br />



# More TBA...

## Installing conda on Kamb

```bash
# Go to kamb
ssh usename@kamb

# Create a tools directory at your home directory on Kamb
mkdir -p ~/tools; cd ~/tools

# download, install and setup (mini/ana)conda
# for Linux, use Miniconda3-latest-Linux-x86_64.sh
# for macOS, opt 2: curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-MacOSX-x86_64.sh -b -p ~/tools/miniconda3
~/tools/miniconda3/bin/conda init bash

# Close and restart the shell for changes to take effect.
exit                 # logout
ssh usename@kamb     # login again

# First, set conda-forge as default conda channels 
conda config --add channels conda-forge
conda config --set channel_priority strict

# now you can make your new conda env and install stuff under that env
conda create --name your_new_env
conda activate your_new_env
conda install some_packages_you_want

```

Reference of the above guidance: [Yunjun's GitHub repository](https://github.com/yuankailiu/conda_envs) (Yunjun is the postdoc here. He is the developer of MintPy. so, we will use this site again when installing MintPy)



## How to set .bashrc or .bash_profile on Kamb

- Some aliases of bash commands run in your terminal
- Path settings once ARIA-tools and MintPy are properly installed

## How to use Jupyter Notebook on Kamb

You can run Jupyter notebooks on Kamb while navigate and play around with it on the web browser on your own laptop.

Will add this later...

## Getting Git ready

We will use git to get the source codes of `ARIA-tools` (getting the InSAR unwrapped datasets) and `MintPy` (doing InSAR time-series analysis).

Will add this later...

## Installation of ARIA-tools 

The GitHub page of ARIA-tools: https://github.com/aria-tools/ARIA-tools

## Intallation of MintPy

The GitHub page of MintPy: https://github.com/insarlab/MintPy

