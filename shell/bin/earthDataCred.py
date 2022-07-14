#!/usr/bin/env python3

from http.cookiejar import CookieJar
from urllib.parse import urlencode
import urllib.request

# The user credentials that will be used to authenticate access to the data
username = "yuankai_liu"
password = "Proverbs2:11"

# The url of the file we wish to retrieve
#url = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11/N38W123.SRTMGL3.hgt.zip"
#url = "http://naciscdn.org/naturalearth/50m/physical/ne_50m_ocean.zip"
#url = "http://naciscdn.org/naturalearth/50m/physical/ne_50m_river.zip"
url = "http://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip"

# Create a password manager to deal with the 401 reponse that is returned from
# Earthdata Login

password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

# Create a cookie jar for storing cookies. This is used to store and return
# the session cookie given to use by the data server (otherwise it will just
# keep sending us back to Earthdata Login to authenticate).  Ideally, we
# should use a file based cookie jar to preserve cookies between runs. This
# will make it much more efficient.

cookie_jar = CookieJar()

# Install all the handlers.

opener = urllib.request.build_opener(
    urllib.request.HTTPBasicAuthHandler(password_manager),
    #urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
    #urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
    urllib.request.HTTPCookieProcessor(cookie_jar))
urllib.request.install_opener(opener)

# Create and submit the request. There are a wide range of exceptions that
# can be thrown here, including HTTPError and URLError. These should be
# caught and handled.

request = urllib.request.Request(url)
response = urllib.request.urlopen(request)

# Print out the result (not a good idea with binary data!)

body = response.read()
#print(body)
