
import pandas as pd
import datetime
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
# nltk.download('punkt')


def get_posts(subreddit):
    from bs4 import BeautifulSoup, SoupStrainer, Comment
    import requests
    import pandas as pd
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
    }

    result = requests.get('https://www.reddit.com' +
                          subreddit, headers=headers)
    soup = BeautifulSoup(result.content, 'html.parser')
    content = soup.findAll(class_="title")

    # need to cut out first two and last 4 and repeating titles
    titles = content[2:-4:2]

    sentences = [sen.text for sen in titles]
    paragraph = ' '.join(w.strip() for w in [sen.text for sen in titles])
    return paragraph, sentences


def analyze_paragraph(paragraph):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(paragraph)
    return ss['pos'], ss['neu'], ss['neg'], ss['compound']


def analyze_sentences(sentences):
    sid = SentimentIntensityAnalyzer()
    pos, neu, neg, com = 0, 0, 0, 0
    for sentence in sentences:
        ss = sid.polarity_scores(sentence)
        pos += ss['pos']
        neu += ss['neu']
        neg += ss['neg']
        com += ss['compound']

    return pos, neu, neg, com


def get_days_teams():
    # get games of the day
    from bs4 import BeautifulSoup, SoupStrainer, Comment
    import requests
    import pandas as pd
    import datetime

    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
    }

    now = datetime.datetime.now()

    url = "https://www.basketball-reference.com/boxscores/?month={0}&day={1}&year={2}".format(
        now.month, now.day-1, now.year)

    result = requests.get(url, headers=headers)
    soup = BeautifulSoup(result.content, 'html.parser')
    content = soup.findAll(class_="teams")

    winners = []
    losers = []
    for teams in content:
        # find winner
        winner = teams.find(class_="winner")
        for td in winner.find('td'):
            winners.append(td.text)
        loser = teams.find(class_="loser")
        for td in loser.find('td'):
            losers.append(td.text)

    return winners, losers


subreddits = {
    'LA Lakers': '/r/lakers',
    'Golden State': '/r/warriors',
    'Chicago': '/r/chicagobulls',
    'Toronto': '/r/torontoraptors',
    'Boston': '/r/bostonceltics',
    'Cleveland': '/r/clevelandcavs',
    'New York': '/r/nyknicks',
    'San Antonio': '/r/nbaspurs',
    'Miami': '/r/heat',
    'Houston': '/r/rockets',
    'Philadelphia': '/r/sixers',
    'Portland': '/r/ripcity',
    'Oklahoma City': '/r/thunder',
    'Minnesota': '/r/timberwolves',
    'Dallas': '/r/mavericks',
    'Atlanta': '/r/atlantahawks',
    'LA Clippers': '/r/laclippers',
    'Detroit': '/r/detroitpistons',
    'Washington': '/r/washingtonwizards',
    'Charlotte': '/r/charlottehornets',
    'Sacramento': '/r/kings',
    'Milwaukee': '/r/mkebucks',
    'Phoenix': '/r/suns',
    'Indiana': '/r/pacers',
    'Orlando': '/r/orlandomagic',
    'Denver': '/r/denvernuggets',
    'Utah': '/r/utahjazz',
    'Brooklyn': '/r/gonets',
    'Memphis': '/r/memphisgrizzlies',
    'New Orleans': '/r/nolapelicans'
}


def compute_all():
    winners, losers = get_days_teams()
    winner_data = compute_winners(winners)
    loser_data = compute_losers(losers)
    data = winner_data + loser_data
    return data


def compute_winners(winners):
    data = []
    for team in winners:
        paragraph, sentences = get_posts(subreddits[team])
        sen_pos, sen_neu, sen_neg, sen_com = analyze_sentences(sentences)
        par_pos, par_neu, par_neg, par_com = analyze_paragraph(paragraph)
        data.append([team, sen_pos, sen_neu, sen_neg, sen_com,
                     par_pos, par_neu, par_neg, par_com, True])
    return data


def compute_losers(losers):
    data = []
    for team in losers:
        paragraph, sentences = get_posts(subreddits[team])
        sen_pos, sen_neu, sen_neg, sen_com = analyze_sentences(sentences)
        par_pos, par_neu, par_neg, par_com = analyze_paragraph(paragraph)
        data.append([team, sen_pos, sen_neu, sen_neg, sen_com,
                     par_pos, par_neu, par_neg, par_com, False])
    return data


# In[13]:


# store results


data = compute_all()
df = pd.DataFrame(data, columns=['team', 'sen_pos', 'sen_neu', 'sen_neg',
                                 'sen_com', 'par_pos', 'par_neu', 'par_neg', 'par_com', 'won'])
df['date'] = pd.to_datetime(datetime.datetime.now())
df.index = df['date']
del df['date']


with open('nba_sentiment.csv', 'a') as f:
    df.to_csv(f, header=False)
#     df.to_csv(f, index=False, mode='a', header=(not os.path.exists(f)))


# In[14]:


df
