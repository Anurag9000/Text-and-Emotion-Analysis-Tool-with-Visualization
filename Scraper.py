import requests
import json

def get_access_token(client_id, client_secret, user_agent):
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    headers = {"User-Agent": user_agent}
    data = {"grant_type": "client_credentials"}

    response = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch access token: {response.status_code} {response.text}")

    return response.json()["access_token"]

def scrape_reddit_subreddit(subreddit_url, headers, limit):
    # Extract subreddit name from URL
    subreddit_name = subreddit_url.rstrip('/').split('/')[-1]

    # Fetch subreddit posts
    posts_response = requests.get(f"https://oauth.reddit.com/r/{subreddit_name}/new.json?limit={limit}", headers=headers)
    if posts_response.status_code != 200:
        raise Exception(f"Failed to fetch subreddit posts: {posts_response.status_code}")

    posts_data = posts_response.json()
    all_posts = []

    for post in posts_data['data']['children']:
        post_details = {
            "title": post['data'].get('title', ''),
            "content": post['data'].get('selftext', ''),
        }

        # Fetch comments for each post
        post_id = post['data']['id']
        comments_response = requests.get(f"https://oauth.reddit.com/comments/{post_id}.json", headers=headers)
        if comments_response.status_code == 200:
            comments_data = comments_response.json()
            comments = [
                comment['data']['body'] for comment in comments_data[1]['data']['children'] if 'body' in comment['data']
            ]
        else:
            comments = []

        all_posts.append({"post_details": post_details, "comments": comments})

    return all_posts

# Save data to a text file
def save_to_text(subreddit_name, all_posts, file_name="subreddit_data.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(f"Subreddit: {subreddit_name}\n\n")
        for idx, post in enumerate(all_posts):
            file.write(f"Post {idx + 1}:\n")
            file.write(f"Title: {post['post_details']['title']}\n")
            file.write(f"Content: {post['post_details']['content']}\n\n")
            file.write("Comments:\n")
            for comment in post['comments']:
                file.write(f"- {comment}\n")
            file.write("\n\n")

    print(f"Data saved to {file_name}")

if __name__ == "__main__":
    # Take inputs from the user
    subreddit_url = input("Enter the Reddit subreddit URL: ").strip()
    file_name = input("Enter the name of the output file (default: subreddit_data.txt): ").strip()
    if not file_name:
        file_name = "subreddit_data.txt"

    print("To fetch data, enter your Reddit API credentials. You can get these by registering a script on https://www.reddit.com/prefs/apps.")
    client_id = input("Enter your Client ID: ").strip()
    client_secret = input("Enter your Client Secret: ").strip()
    user_agent = input("Enter your User-Agent string: ").strip()

    limit = input("Enter the number of latest posts to fetch: ").strip()
    if not limit.isdigit() or int(limit) <= 0:
        print("Invalid number of posts. Defaulting to 10.")
        limit = 10
    else:
        limit = int(limit)

    try:
        # Get Bearer token
        access_token = get_access_token(client_id, client_secret, user_agent)
        headers = {
            "User-Agent": user_agent,
            "Authorization": f"Bearer {access_token}"
        }

        # Scrape subreddit posts and comments
        subreddit_name = subreddit_url.rstrip('/').split('/')[-1]
        all_posts = scrape_reddit_subreddit(subreddit_url, headers, limit)

        # Save results to a text file
        save_to_text(subreddit_name, all_posts, file_name)
    except Exception as e:
        print(f"An error occurred: {e}")
