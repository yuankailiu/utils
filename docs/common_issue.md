## ERA5 downloading issue

The [legacy CDS](https://cds.climate.copernicus.eu/#!/home) and [legacy ADS](https://ads.atmosphere.copernicus.eu/cdsapp#!/home) will be **decommissioned on September 26, 2024** and will no longer be accessible from this date onwards. The most recent CDS and api service is hosted on [CDS Beta version from copernicus](https://cds-beta.climate.copernicus.eu).

#### How-to use it from MintPy/PyAPS:

- Check this [account setup for ERA5](https://github.com/insarlab/PyAPS?tab=readme-ov-file#2-account-setup-for-era5) to create a bew account on CDS beta.
- Login, under [Datasets > ERA5 hourly data on pressure levels from 1940 to present > Download > Terms of use](https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download), click **Accept** to accespt the license to use Copernicus Products.

- Create the local file`$HOME/.cdsapirc` and add the following two lines

  *make sure url is the beta version api; use only the key, not include UID*

  ```bash
  url: https://cds-beta.climate.copernicus.eu/api
  key: ebfbd500-719b-4e03-aca7-6c880f64bf79 (put your key here)
  ```

#### Test on the latest PyAPS

```
git clone https://github.com/insarlab/PyAPS.git --depth 1
python PyAPS/tests/test_dload.py
```



# PR to github

This [old instance of the CDS](https://cds.climate.copernicus.eu/api-how-to) will be [decommissioned on 26 September 2024](https://confluence.ecmwf.int/display/CKB/Please+read%3A+CDS+and+ADS+migrating+to+new+infrastructure%3A+Common+Data+Store+%28CDS%29+Engine) and will no longer be accessible from this date onwards. Let's migrate the guide and the code to CDS-Beta.

#### NEW URL: https://cds-beta.climate.copernicus.eu/api

#### Commits:

+ update beta version url in README
+ update beter version url in `autoget.py` 
+ for `key` within either `.cdsapirc` or `model.cfg`, only type in your personal access token, no UID is needed and allowed.
+ point to where to find my token.
+ point to where to accept the Terms of Use (since I find it hard to see the button).



#### Examples

My `model.cfg` looks like:

```
# old style
[CDS]
key = 311901:e041172e-0a2a-4e0f-88fa-ea3b6bae7058

# new style
[CDS]
key = ebfbd500-719b-4e03-aca7-6c880f64bf79
```



My `~/.cdsapirc` looks like:

```
# old style
url: https://cds.climate.copernicus.eu/api/v2
key: 311901:e041172e-0a2a-4e0f-88fa-ea3b6bae7058

# new style
url: https://cds-beta.climate.copernicus.eu/api
key: ebfbd500-719b-4e03-aca7-6c880f64bf79
```



#### Test:

I guess you might still use this old style until 9/26 with this reminder pops up:
```
As per our announcements on the Forum, this instance of CDS will be decommissioned on 26 September 2024 and will no longer be accessible from this date onwards.
Please update your cdsapi package to a version >=0.7.2, create an account on CDS-Beta and update your .cdsapirc file. We strongly recommend users to check our Guidelines at https://confluence.ecmwf.int/x/uINmFw
```
And the request will be queued and get stuck.



Update the new url and using the token from the beta website, run the test:

` cd tests/ && python test_dload.py` 

it works:

```
------------------------------------------------
import pyaps3 from /home/ykliu/apps/mambaforge/envs/insar/lib/python3.12/site-packages/pyaps3/__init__.py
------------------------------------------------
test ERA5 data download
NOTE: Account setup is required on the Copernicus Climate Data Store (CDS).
      More detailed info can be found on: https://retostauffer.org/code/Download-ERA5/
      Add your account info to ~/.cdsapirc file.
INFO: You are using the latest ECMWF platform for downloading datasets:  https://cds-beta.climate.copernicus.eu/api
Downloading 1 of 2: /home/ykliu/apps/PyAPS/tests/data/ERA5/ERA5_N30_N40_E120_E140_20200601_14.grb
{'product_type': 'reanalysis', 'format': 'grib', 'variable': ['geopotential', 'temperature', 'specific_humidity'], 'pressure_level': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'], 'year': '2020', 'month': '06', 'day': '01', 'time': '14:00', 'area': '40.00/120.00/30.00/140.00'}
2024-09-16 15:44:43,153 INFO Request ID is c4bdbde8-f73c-4dfa-b92a-58006ab76355
2024-09-16 15:44:43,333 INFO status has been updated to accepted
2024-09-16 15:44:45,026 INFO status has been updated to running
2024-09-16 15:44:47,459 INFO status has been updated to successful
Downloading 2 of 2: /home/ykliu/apps/PyAPS/tests/data/ERA5/ERA5_N30_N40_E120_E140_20200901_14.grb
{'product_type': 'reanalysis', 'format': 'grib', 'variable': ['geopotential', 'temperature', 'specific_humidity'], 'pressure_level': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'], 'year': '2020', 'month': '09', 'day': '01', 'time': '14:00', 'area': '40.00/120.00/30.00/140.00'}
2024-09-16 15:44:51,651 INFO Request ID is e6348440-f2c1-4e5b-9781-dfdf3d477c85
2024-09-16 15:44:51,816 INFO status has been updated to accepted
2024-09-16 15:44:53,493 INFO status has been updated to successful
------------------------------------------------
Downloads OK
------------------------------------------------
```



## Okada4py compilation

Upstream `okada4py` is on Romain's repo: https://github.com/jolivetr/okada4py

Lijun fixed a setup.py issue on his repo: https://github.com/lijun99/okada4py.git

```
mamba activate your_env

# clone and use lijun's setup
git clone git@github.com:lijun99/okada4py.git
git checkout setup

# build
export CC=gcc (use system /usr/bin/gcc)
python setup.py build (build and compile)

# Link in a user module directory (don't do python setup.py install --user)
pip install --no-cache-dir .

# test it
pip show okada4py
cd test/ && python test.py
```

