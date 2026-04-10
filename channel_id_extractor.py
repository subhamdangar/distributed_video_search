import requests
import re

url = input("Enter url of the yt_page : ")
response = requests.get(url)

# This regex looks for the UC... pattern inside the page source
channel_id = re.search(r'UC[\w-]{22}', response.text)

if channel_id:
    print(f"The Channel ID is: {channel_id.group()}")
else:
    print("Could not find the ID.")