# Usage of various InSAR-related software

## Snaphu

The Zebker's group at Stanford EE/Geophysics developed the [*snaphu*](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) package for unwrapping the interferogram. You could download the .tar.gz file from here: [snaphu-v2.0.4.tar.gz](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu-v2.0.4.tar.gz)

### installation

Place that .tar.gz file under the directory you prefer. I put it under the bin directory of my home on Mac: `/Users/home/bin/` (if you don't have it, just create a `bin/` directory under your home). 

Then untar the file:

```
tar xvf snaphu-v2.0.4.tar.gz 
```

Go into the source code directory

```
cd snaphu-v2.0.4/src 
```

Build the code

```
make
```

Make sure the compiled code is executable

```
chmod 755 ../bin/snaphu
```

Now, we can run the code in this directory by first `cd ../bin`, then 

```
./snaphu
```

In order to run it everywhere on your machine, do

```
export SNAPHU_HOME=/Users/home/bin
```

Or write this line to your ~/.bashrc file as a default every time you initiate the bash

```
echo 'export SNAPHU_HOME=/Users/home/bin' >> ~/.bashrc
```

Try to run *snaphu* and see if it works

```
$SNAPHU_HOME/snaphu
```

You should see a printout of the short-version manual as below image. You can find how to use *snaphu* on a more detailed [manual page](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/snaphu_man1.html) as well.

Basically, we will do:

```
$SNAPHU_HOME/snaphu wrappedfile 1024 -o unwrappedfile
```

Above is assuming line length is 1024 complex samples. The output file from snaphu is in the magnitude values followed by phase values format. So say you have a 1024 length line, you read in 1024 float32 magnitude values and then 1024 unwrapped phase values for each line.

### file I/O format

More tips when preparing the file for *SNAPHU*:

*snaphu* can read the input file as real-and-imaginary interleaved float32. So we can save our complex interferogram this way.

```python
## write the flatten ifg to a binary file called "wrappedfile"

# real and imaginary parts are interleaved by each other
real = np.real(ifg_flat)
real = np.imag(ifg_flat)
resl = np.dstack([real,imag]).reshape(real.shape[0], -1)

# write the resl array as float32 to file
outfile = open('wrappedfile', 'wb')
outfile.write(resl.astype(np.float32))
outfile.close()
```

 Then run Snaphu to unwrap the file:

```
snaphu wrappedfile 1024 -o unwrappedfile
```

When reading the output file from *snaphu*, the file is in the magnitude values followed by phase values format. So say you have a 1024 length line, you read in 1024 float32 magnitude values and then 1024 unwrapped phase values for each line.