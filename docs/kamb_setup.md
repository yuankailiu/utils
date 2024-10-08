# Intro of what to do

A large portion of Geophysics is about computing stuff. The amount of computation can somtimes (if not always) exceeds what your personal computer can handle within reasonable time. Plus we want to prolong the lifespan of our own laptops. A common practice here in Seismolab is that each research group has its own group cluster (sitting behind the printing room on the 2nd floor of South Mudd) for computation. The operating system is Linux. Students will be granted access to the group cluster(s) to which they belong. Ex. If you work with Mark and Zach, you will have access to both of their clusters. All Seismolab machines are managed by Scott Dungan <sdungan@caltech.edu>, our GPS tech admin.


# Beginners
Here are some things to get started on. Feel free to quickly skim or skip some. The idea is to know some vocabulary.

### Account:

1. Make sure you’ve sent Scott Dugan your Caltech email address. Read his instructions in his reply.
	- In Mark's group, available machines are: `kamb`, `hokkaido`, `kyushu`, `shikoku`, `honshu`
2. [Request a Caltech VPN](https://www.imss.caltech.edu/services/wired-wireless-remote-access/Virtual-Private-Network-VPN/vpn-apps-students) (helpful when you work off-campus)
### Linux:
- Get a terminal app that you feel happy about. Rather than the built-in Mac Terminal, I like [iterm2](https://iterm2.com/).
- Get familiar with the terminal, read any simple online blog/tutorial ([like this](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview), but don't get sucked into it).
- Text editing. How to use `vim`, check [this](https://medium.com/@BetterEverything/using-vim-to-make-and-edit-files-on-linux-is-easy-with-these-steps-5b47e00b9d22) out:
- Connect to servers. Read about [SSH]([https://www.baeldung.com/cs/ssh-intro](https://www.baeldung.com/cs/ssh-intro)).
### Conda:
- Spend a little time read about [conda]([https://docs.conda.io/en/latest/](https://docs.conda.io/en/latest/)) (don't get sucked into it).
### Programming IDEs
Different people have varying preferences when it comes to IDEs. While some people might not use them at all (always do `SSH` and `vim`), others may rely on several for different tasks. Here are a few popular ones for Python users:
- **Jupyter Notebook:** Great for interactive code-writing at early stages, data visualization, and going through tutorials that were written as Notebooks. Even writing your homework. [Learn more about Jupyter Notebook](https://realpython.com/jupyter-notebook-introduction/).
  
- **Visual Studio Code (VS Code):** A versatile and widely-used editor that supports numerous programming languages and extensions. It's known for its rich feature set and customizability.
  

Feel free to explore more (`Spyder`, `pycharm`).

# Setting up your home

Here is some guidance for setting up the computer (environments on Kamb) ready for the InSAR, geodetic-related works. Other than some specialized packages, here are general tools you may be using in the future:
- [Python](https://www.python.org/) (most InSAR tools we use are in Python)
- [Jupyter Notebooks](https://jupyter.org/)
- Python's [matplotlib](https://matplotlib.org/) module for simple plots
- [Generic Mapping Tools (GMT)](https://www.generic-mapping-tools.org/) for plotting maps.

Things that will be covered here:

- Paths and environments settings on Kamb (The server of Mark Simons group where we run computations, named after the glaciologist [Barclay Kamb](https://calteches.library.caltech.edu/4534/1/Obituary.pdf))
- Setup to run Jupyter Notebooks on Kamb
  - Jupyter Notebook provides a Matlab-like interface to run Python code and display results and plots.
  - For interferograms and time-series analysis, there are several useful notebooks for the tutorials of `ARIA-tools` and `MintPy` available provided by the developers.
- Installation of `ARIA-tools` and `MintPy` 

## Know the machines

```bash
########################
#   Computing nodes
########################

# Kamb: kamb@gps.caltech.edu
# disk space : 100 GB (personal space in your home)
# computation: has 48 CPUs + 8 GPUs (Tesla V100-PCIE-16GB)
# usage:       major login server; home directory; install your packages/codes; light storage, run CPU/GPU tasks
# backup:      the home directory is backup daily


# Four CPU nodes:
#    hokkaido@gps.caltech.edu
#    honshu@gps.caltech.edu
#    kyushu@gps.caltech.edu
#    shikoku@gps.caltech.edu
# disk space : N/A (connects to your home/ on Kamb)
# computation: has 32 CPUs + no GPUs
# usage:       can also login there, share the same home dir with Kamb, has the same package/env you setup; run CPU tasks
# backup:      N/A


########################
#   Storage servers
########################

# /net/kraken/nobak        
# You can store 20TB in the 'nobak'; NOT backed up

# /net/kraken/bak
# You can store 5TB in the 'bak'; backed up. You have 60 days of backup. After 60 days, deleted files cannot be restored.

# /marmot-nobak/
# You can store more than 20TB currently; NOT backed up

# How to use these storage servers:
#  1. Go there and create your own directory (mkdir yourusername)
#  2. Make symbolic links at your home directory and easily access these storage disks
```


## 0. An easier way of SSH (optional, but good for laziness)

To access the Kamb server, you'll need to use SSH each time with the command: `ssh username@kamb.gps.caltech.edu`. 
The `username` is your GPS division username, the one you got from Scott. You'll be prompted to enter your password.

#### SSH key authentication
SSH key authentication will help us to waive password entering. Check out this [guidance](https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/) (Step1 to Step3 will do it).
If there is a two-factor authentication for your system, SSH key authentication bypasses the password+2fa code. So setting this up is very convenient.

#### SSH aliases
After setting the above, we don't need to type a password. But we still need to type our username and the full IP address when doing SSH, i.e., `ssh username@kamb.gps.caltech.edu`

To further save your energy, we can create alias names for SSH on your local computer. Do the following on **your own laptop**:

```bash
# First open the terminal on your laptop

# go into .ssh directory (if not exist, make one by mkdir .ssh/)
cd ~/.ssh

# create or edit a config file
touch config

```

Copy and paste the following content in that config file, remember to change `your_user_name` to your username.

```bash
# KAMB server: Mark Simon's group
Host kamb
    Hostname kamb.gps.caltech.edu
    User your_user_name
    ServerAliveInterval 720
    ServerAliveCountMax 120
    ForwardX11Timeout = 24h

# EARTH server: Caltech GPS server
Host earth
    Hostname earth.gps.caltech.edu
    User your_user_name
    ServerAliveInterval 720
    ServerAliveCountMax 120
    ForwardX11Timeout = 24h

# Other servers: you can do the same thing on Hokkaido, Honshu, Kyushu, Shikoku, 
# Host name_of_server
#    Hostname name_of_server.gps.caltech.edu
#    User your_user_name
#    ServerAliveInterval 720
#    ServerAliveCountMax 120
#    ForwardX11Timeout = 24h

```

Now, you don't need to type the username and the full IP address. From your laptop, you can log into Kamb by simply doing:

```bash
ssh kamb
ssh earth
```

If you are using off-campus internet, you might not SSH onto some machines without first connecting to [Caltech VPN](https://www.imss.caltech.edu/services/wired-wireless-remote-access/Virtual-Private-Network-VPN). `kamb` and `earth` works for me without VPN. But other machines require that.

<br />

## 1. Access to Kraken and Marmot

- Kraken is a disk server meant to store data. It is named after the [sea monster in Scandinavian folklore](https://en.wikipedia.org/wiki/Kraken); like a hub connecting to other machines with its' strong and agile tentacles.
- [Marmot](https://natural-history-journal.blogspot.com/2016/09/yellow-bellied-marmot-denizen-of-high.html)
- Two subfolders under Kraken: `bak` (60-day backup for saving just codes) and `nobak` (no backup for saving Gigabyte-sized datasets)

```bash
# SSH to Kamb server, you will be at your home directory on Kamb
ssh username@kamb.gps.caltech.edu

## Go Kraken-nobak, the no backup directory for storing large datasets (up to 20TB)
cd /net/kraken/nobak/

# make your own folder here
mkdir my_name

# make a symbolic link of this path to your home directory
ln -s /net/kraken/nobak/my_name ~/kraken-nobak


## Go to Kraken-bak, the 60-day backup directory for storing codes, notes, etc (up to 5TB)
cd /net/kraken/bak/

# make your own folder here
mkdir my_name

# make a symbolic link of this path to your home directory
ln -s /net/kraken/bak/my_name ~/kraken-bak


## Go to marmot-nobak, do the same thing
cd /marmot-nobak
mkdir my_name
ln -s /marmot-nobak/my_name ~/marmot-nobak

## Now every time you log on to kamb, you will see and can `cd` into your folder on kraken-bak, kraken-nobak, marmot-nobak
## easily without needing to type the full paths to access them.
```

<br />

## 2. Install Conda on Kamb

During your career path here, you will do seismology, GPS, InSAR, forward/inverse modeling, machine learning, etc. You will use many different codes, and packages that require quite different computer prerequisites and environments. [That is why we need conda to manage all these nerdy stuff](https://www.freecodecamp.org/news/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c/). 

```bash
# Go to kamb (a lazy way now)
ssh kamb

# Create a tools directory (where you will store all downloaded software, and source codes) at your home directory on Kamb
# -p command: make this directory if not exist. Then we `cd` into the tools folder
mkdir -p ~/tools; cd ~/tools

# get the installer from web (for macOS, use Miniforge3-MacOSX-x86_64.sh, and optionally use `curl -L -O https://...` syntax to download)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# install it under ~/tools/
bash Miniforge3-Linux-x86_64.sh -b -p ~/tools/miniforge

# initialize it
~/tools/miniforge/bin/mamba init bash

# Close and restart the shell for changes to take effect. 
exit                         # logout
ssh kamb                     # login again; you will see on your command line prompt: (base) username@kamb


# set conda-forge as the default conda channels 
conda config --add channels conda-forge
conda config --set channel_priority strict

# install the following basic utilities under the (main) env
mamba install wget git tree numpy --yes
pip install bypy


# Now you can make your new conda env and install stuff under 
mamba create --name your_new_env       # create a env
mamba activate your_new_env            # activate this env
mamba install some_packages_you_want   # install new stuff
mamba deactivate                       # leave this env
```
Reference of the above guidance: [Yunjun's GitHub repository](https://github.com/yunjunz/conda-envs) ([Yunjun Zhang](https://github.com/yunjunz) was a postdoc here, the developer of `MintPy`, and my InSAR mentor!).

See [Conda Cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
The basic `mamba` and `conda` commands are usually used interchangeably. Calling `mamba` is much faster. But `conda` has the complete utilities.
```bash
# Below are equivalent

# This is faster
mamba install some_packages_you_want

# This is usually slower
conda install some_packages_you_want
```


<br />

## 3. Git

We will use `git` to get the source codes of `ARIA-tools` (getting the InSAR unwrapped datasets) and `MintPy` (doing InSAR time-series analysis).

[git](https://git-scm.com/) is an open source [version control system](https://www.atlassian.com/git/tutorials/what-is-version-control). You can find some very basic tutorials online if you are bored. Otherwise, we will only use the very basic stuff of `git` (e.g. how to download source codes of a open source software). The things explained in this document about `git` will be enough for us now. Feel free to register a GitHub account.

You might already have `git` installed on Kamb. If not, simply install it in the `base` conda env

```bash
# Go to Kamb
ssh kamb

# You are now at the base env. The prompt will show (base)

# Install git via conda
mamba install git

# check git version (you should have version >= 2.30.2)
git version
```


## 4. Install InSAR software

```bash
# From your laptop,SSH to Kamb
ssh kamb

# You can create a new insar env for these software
# create a new conda env called insar
mamba create --name insar --yes
mamba activate insar
```

Starting as an end-user, if you are not helping to develop the codes, just follow **Option1** here to install the `isce2` and `mintpy` stable releases on conda:
https://github.com/yunjunz/conda-envs?tab=readme-ov-file#2-install-isce-2-and-mintpy

This will install all the prerequisites and `isce2` and `mintpy` to your `insar` env.

**IMPORTANT PATH SETTINGS!!!** (for `isce2`)

You will need to do it manually for both `$PATH$` and `$PYTHONPATH$`.

1. If you install via [**Option2**](https://github.com/yunjunz/conda-envs?tab=readme-ov-file#option-2-install-isce2-conda-version-and-mintpy-development-version), then step c there will source the `config.rc` and set the paths.
2. If you install via [**Option1**](https://github.com/yunjunz/conda-envs?tab=readme-ov-file#option-1-install-isce2-conda-version-and-mintpy-conda-version), then you need to add the following manually in your system source file (.bash_profile or .bashrc)
```bash
# in your .bash_profile or .bashrc, add this line if it does not already exist
if [ -z ${PYTHONPATH+x} ]; then export PYTHONPATH=""; fi

# for isce2 env paths (must activate env first to get $ISCE_HOME and $CONDA_PREFIX)
alias load_insar='mamba activate insar; export PATH=${PATH}:${ISCE_HOME}/bin:${ISCE_HOME}/applications; export PYTHONPATH=${CONDA_PREFIX}/packages:${PYTHONPATH}'

```

Now, every time we want to use those software, we will just SSH to kamb, and activate the env.

```bash
# warm up conda environment
load_insar

# equivalent to doing `mamba activate insar` and set all the above paths manually

# Run the following to test the installation:
topsApp.py -h            # test ISCE-2
smallbaselineApp.py -h   # test MintPy


# If you want to go to the base env
mamba deactivate
```

<br />


## 5. How to run Jupyter Notebook on Kamb?

```bash
# activate your env
mamba activate insar

# install jupyter notebook, jupyterlab, ipykernal
mamba install -c conda-forge notebook jupyterlab ipykernel

# install this insar env in the jupyter ecosystem (so you can use this env on jupyter)
python -m ipykernel install --user --name=insar

```

To run a notebook or a jupyterLab session on your laptop:
```bash
# run the notebook, minimalism of jupyterLab
jupyter notebook

# run jupyterLab
jupyter lab
```

But how about run jupyter on a remote server like on `kamb`?
```bash
# go to a remote server
ssh kamb

# run jupyer: almost the same, but you will not show the browser on the remote server; you will forward that to a port
jupyter lab --no-browser --port=1236

# now go to your local terminal (your laptop), tunnel that remote port to a local port
ssh -N -f -L 8080:localhost:1236 <REMOTE_USER>@<REMOTE_HOST>

# both local and remote --port numbers are arbitrary, choose as long as nobody is using it
```

REFERENCE:
- **The basic idea: [online reference here](https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/).**

- **Guide: https://github.com/yunjunz/conda-envs/blob/main/docs/jupyter.md**

WHY WE DO THIS:
Using Jupyter Notebook on Kamb by [forwarding a webpage port](https://linuxize.com/post/how-to-setup-ssh-tunneling/#local-port-forwarding) (independently of X11), such that the browser is local to your computer but gets the data through the tunnel from the server. With SSH to KAMB, we can have Jupyter Notebooks open locally (on a laptop browser), while all the computations is done on the remote server. 

This Jupyter instance will only run as long as you have your SSH connection and shell open. If you want to keep Jupyter running even while you're logged out, you can open a **[Linux screen instance](https://linuxize.com/post/how-to-use-linux-screen/)**, and run the Jupyter command in there (that's what I do). Simply detach the `screen` and it'll stay running in the background. The next time you SSH into the machine, just open that same link as before, and your Jupyter process will be ready where you left it off.


<br />

## 6. Downloading products from Earthdata (Sentinel-1 data)

1. Make sure you have an account for Earthdata Login: https://urs.earthdata.nasa.gov/profile

2. You might have to manually check some agreements. Go to `Applications` > `Authorized Apps`
   After checking, you should have these on the agreements page automatically.

3. In order to [access data over HTTP from a web server with curls and wget](https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget), we need to enter our account name and password every time. We can configure the account credentials in a file for automatic authentication (no need for a username and password every time). Please do the following:

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

+ Now, `exit` the Kamb server and re-login again. Run the download script (you got from the ASF website for your selected datasets), see if you can download files without entering account info.

<br />

## 7. Customize your `.bashrc` on Kamb (optional)

- Syntax and format settings on the server. This can make your terminal display colorful texts and highlights.
- Some aliases of bash commands. This makes command-calling easier.

Add below (if they don't exist) to `~/.bashrc`:

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
alias pp='pwd -P'
alias rm='rm -i'
alias sbashrc='source ~/$BASHRC'
alias sbashpr='source ~/.bash_profile'
alias ssh='ssh -X'
alias vi='vim'
alias which='alias | /usr/bin/which --tty-only --read-alias --show-dot --show-tilde'
```

Add below to your `~/.bash_profile` if they are not there

```bash
if [ -r ~/.bashrc ]; then
   source ~/.bashrc
fi
```

<br />


# Extra practices
1. Practice installing Conda on your own Mac.
2. Create a conda environment, activate it, and write a Python script to print ‘hello world’.



# Papers:

Simons, M., & Rosen, P. A. (2007). Interferometric synthetic aperture radar geodesy. _Geodesy_, _3_, 391-446. [2nd edition 2015](https://simons.caltech.edu/publications/pdfs/Simons_etal_2015.pdf).

#### `ISCE`
[Rosen, P. A., Gurrola, E., Sacco, G. F., & Zebker, H. (2012, April). The InSAR scientific computing environment. In _EUSAR 2012; 9th European conference on synthetic aperture radar_ (pp. 730-733). VDE.](https://ieeexplore.ieee.org/abstract/document/6217174)

Fattahi, H., P. Agram, and M. Simons (2016), A Network-Based Enhanced Spectral Diversity Approach for TOPS Time-Series Analysis, IEEE Transactions on Geoscience and Remote Sensing, 55(2), 777-786, doi:[10.1109/TGRS.2016.2614925](https://ieeexplore.ieee.org/abstract/document/7637021).

#### `ARIA`
Buzzanga, B., Bekaert, D. P. S., Hamlington, B. D., & Sangha, S. S. (2020). Towards Sustained Monitoring of Subsidence at the Coast using InSAR and GPS: An Application in Hampton Roads, Virginia. Geophysical Research Letters, 47, e2020GL090013. https://doi.org/10.1029/2020GL090013

Bekaert, D., Arena, N., Bato, M. G., Buzzanga, B., Govorcin, M., Havazli, E., ... & Zinke, R. (2023, July). The Aria-S1-Gunw: The ARIA Sentinel-1 Geocoded Unwrapped Phase Product for Open Insar Science and Disaster Response. In _IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium_ (pp. 2850-2853). IEEE.**DOI:** [10.1109/IGARSS52108.2023.10282671](https://doi.org/10.1109/IGARSS52108.2023.10282671)

#### `MintPy`
Yunjun, Z., Fattahi, H., and Amelung, F. (2019), Small baseline InSAR time series analysis: Unwrapping error correction and noise reduction, *Computers & Geosciences*, *133*, 104331, doi:[10.1016/j.cageo.2019.104331](https://doi.org/10.1016/j.cageo.2019.104331), [arXiv](https://eartharxiv.org/9sz6m/), [data & figures](https://github.com/geodesymiami/Yunjun_et_al-2019-MintPy).

<br />

# Reference & useful links

The information/guidance provided in this document is not original. It is a condensed version from some of the following sources. Please refer to the corresponding links to learn more about the details.

### Online course
- [UNAVCO InSAR courses](https://www.unavco.org/education/professional-development/short-courses/course-materials/insar/insar.html) 
+ [UNAVCO Short course material, 2020](https://github.com/isce-framework/isce2-docs): InSAR Processing and Time-Series Analysis for Geophysical Applications: InSAR Scientific Computing Environment (ISCE), ARIA Tools, and MintPy

### InSAR software & installation procedures

- [ISCE2 framework, JPL](https://github.com/isce-framework/isce2): SAR, and InSAR processing app.
	- To process a few interferograms, use `topsApp.py` or `alos2App.py`
	- To process a huge stack of interferograms, use the [stack processor](https://github.com/isce-framework/isce2/tree/main/contrib/stack): [topsStack](https://github.com/isce-framework/isce2/tree/main/contrib/stack/topsStack "topsStack"), [stripmapStack](https://github.com/isce-framework/isce2/tree/main/contrib/stack/stripmapStack "stripmapStack"), [alosStack](https://github.com/isce-framework/isce2/tree/main/contrib/stack/alosStack "alosStack")

+ [ARIA, JPL](https://aria.jpl.nasa.gov/products/): overview, product explanations

+ [ARIA-tools]( https://github.com/aria-tools/ARIA-tools): downloading the JPL-processed unwrapped interferograms standard products

+ [ARIA-tools documentations](https://github.com/aria-tools/ARIA-tools-docs): tutorials

+ [MintPy](https://github.com/insarlab/MintPy): InSAR time-series analysis

+ [MintPy documentations](https://mintpy.readthedocs.io/en/latest/) : tutorials

+ [ATBD](https://github.com/nisar-solid/ATBD)

### Data archives

+ [ASF Data Search](https://search.asf.alaska.edu/#/): search Sentinel-1 SLCs, ARIA standard products

+ [Earthdata Login](https://urs.earthdata.nasa.gov/profile)
- [GMTSAR](https://topex.ucsd.edu/gmtsar/demgen/): download DEMs

- [GMRT](https://www.gmrt.org/): Global Multi-Resolution Topography Data Synthesis (DEMs and Bathymetries)

- [UTexas Plates](http://www-udc.ig.utexas.edu/external/plates/data.htm): plate boundaries

- [Natural Earth Data, Roads](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/roads/): roads

- [US cities lon lat](https://www.w3.org/2003/01/geo/test/ustowns/latlong.htm)

### Bash, Linux, etc

- [What is Linux bashrc?](https://www.routerhosting.com/knowledge-base/what-is-linux-bashrc-and-how-to-use-it-full-guide/)
+ [Linux GNU Screen instance](https://linuxize.com/post/how-to-use-linux-screen/)

+ [What is Git and GitHub??](https://blog.devmountain.com/git-vs-github-whats-the-difference/)

+ [Vim cheat sheet](https://vim.rtorr.com/)

### Python, conda, etc

+ [What is conda environments](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)
+ [Conda: Managing your environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Powerful apps for coding

+ [iterm2](https://iterm2.com/): a fancier and more flexible terminal that does amazing things
+ [Visual Studio Code](https://code.visualstudio.com/): a text editor with powerful IDE-like features (makes your code editing feels like Matlab interface)

### Art of presenting

- [Scientific Colour Maps](https://www.fabiocrameri.ch/colourmaps/)
- [Paul Tol's Notes](https://personal.sron.nl/~pault/): on colour schemes and templates
- [Essay - Plotting Data](https://github.com/yuankailiu/utils/blob/main/docs/notes/Santamarina_Essay_Plotting_data%20.pdf) by [Carlos Santamarina](https://www.kaust.edu.sa/en/study/faculty/carlos-santamarina)
