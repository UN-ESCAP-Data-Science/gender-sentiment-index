"""Usage: python3 get_users.py <country>"""
import csv
import tweepy
import sys

# Country is passed as first argument value
country = sys.argv[1]
print('Trying to get {}'.format(country))
output_file_name = country + "_users.csv"

# Put Twitter API key and tokens here
api_key = ''
api_secret = ''
access_token = ''
access_token_secret = ''

# Create the output CSV file
file = open(output_file_name, 'w')
output_csv = csv.writer(file)
header = (
    'id',
    'screen_name',
    'name',
    'created_at',
    'followers_count',
    'notifications',
    'time_zone',
    'favourites_count',
    'protected',
    'description',
    'statuses_count',
    'listed_count',
    'friends_count',
    'following',
    'location',
    'lang',
    'geo_enabled',
    'verified'
)
output_csv.writerow(header)

# Create the tweepy authorization and api objects to query Twitter API
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit_notify=True, wait_on_rate_limit=True)

# Get all users by a simple query on country
page_count = 1
user_count = 0
user_set = set()
while True:
    try:
        users = api.search_users(q=country, page=page_count)
    except tweepy.error.TweepError as error:
        if error.api_code == 44:
            print('No more pages')
        break
    for user in users:
        if user.screen_name not in user_set:
            line = (
                user.id,
                user.screen_name,
                user.name,
                user.created_at,
                user.followers_count,
                user.notifications,
                user.time_zone,
                user.favourites_count,
                user.protected,
                user.description,
                user.statuses_count,
                user.listed_count,
                user.friends_count,
                user.following,
                user.location,
                user.lang,
                user.geo_enabled,
                user.verified
            )
            output_csv.writerow(line)
            user_set.add(user.screen_name)
            user_count += 1
    page_count += 1
print("\n\n Done: Obtained {} users".format(user_count))
file.close()
