import praw
from dotenv import load_dotenv
import os
import datetime
from backend.llm_response import reddit_select_titles
from reddit_pinecone import PineconeInsertion

# Load API credentials from .env file
load_dotenv()
CLIENT_ID = os.getenv("RedditClientId")
CLIENT_SECRET = os.getenv("RedditClientSecret")
USER_AGENT = "RealEstateSentiment/1.0"  # Corrected user agent
print(USER_AGENT)

# Reddit API Initialization
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Define Subreddits
subreddits = ["RealEstate", "realestateinvesting", "bostonhousing", "BostonHousingTips", 
              "SouthofBostonHousing", "Dorchester", "ApartmentsBoston", "Boston_apartments"]
# Extract Posts
posts = []
filtered_count=0
for subreddit_name in subreddits:
    print(f"Fetching posts from: {subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)
    
    # Fetch a list of post titles from the hot posts
    title_list = [post.title for post in subreddit.hot(limit=250)]
    print("Total Titles:", len(title_list))
    
    # Use your LLM or filtering function to select relevant titles
    # reddit_select_titles returns a newline-separated string of titles.
    filtered_titles_str = reddit_select_titles(subreddit_name, title_list)
    filtered_titles = filtered_titles_str.splitlines()  # Convert to list
    print(filtered_titles)
    c=len(filtered_titles)
    print("Filtered Count:", c)
    filtered_count+=c
    
    # Loop over the posts again and process only the posts with a title in the filtered list.
    for post in subreddit.hot(limit=250):
        if post.title in filtered_titles:
            post_data = {
                "post_id": post.id,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "author": str(post.author),
                "subreddit": post.subreddit.display_name,
                "created_utc": datetime.datetime.fromtimestamp(post.created_utc, tz=datetime.timezone.utc),
            }
            pinecone = PineconeInsertion()
            pinecone.insert_embeddings(post_data)
            # Append the filtered post data to the posts list
            posts.append(post_data)

# After collecting all posts in the `posts` list
print("Total Filtered Posts:", len(posts))

# Save the posts list (which contains dictionaries) to a text file
with open("Reddit_Data.txt", 'w') as file:
    for post in posts:
        # Write each dictionary (post) as a string to the file
        file.write(str(post) + "\n")  # Convert dictionary to string and add a newline for separation

print("Data successfully written to Reddit_Data.txt")
