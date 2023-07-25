import pandas as pd
import re

url_rx = re.compile(r'http\S+|www\S+|@\w+|#\w+')   # urls
domain_rx = re.compile(r'\b\w+\.\w+/\w+(/\w+)*\b')  # domains
html_rx = re.compile(r'<.*?>')                     # html tags
esc_rx = re.compile(r'\\[ntr]')                    # escape chars

emoji_rx = re.compile(                             # common emojis
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# sources at end of text
ref_rx = re.compile(r'[Ss]ource text.*')
reuters_rx = re.compile(re.escape('(Reuters)'),
                        re.IGNORECASE)              # reuters sign
# reporters' identity
report_rx = re.compile(r'\(Re[^)]*\)')
# AFP sign
afp_src_rx = re.compile(r'\(AFP[^)]*\)')
# AP sign
ap_src_rx = re.compile(r'\(AP[^)]*\)')
# CBS sign
cbs_src_rx = re.compile(r'\(CBS[^)]*\)')
# PRNews sign
prnews_src_rx = re.compile(r'\(PRNews[^)]*\)')
# CNN sign
cnn_src_rx = re.compile(r'\(CNN[^)]*\)')
ap_end_src_rx = re.compile(
    r'\([^/]+/Associated Press\)')                   # AP sign
# domain specific sign
starting_brac_rx = re.compile(r'^\([^\)]*\)\s*')
# news published timestamp on hindu times
htimes_rx = re.compile(r'\w+ Updated: \w+ \d{2}, \d{4},? \d{2}:\d{2} IST')
# news publised timestamp on hindu
hindu_rx = re.compile(r'(?:\w+ )?\d{2} \w+ \d{4} \d{2}:\d{2} IST')
# news published timestamp
date_first_rx = re.compile(r'^\w+ \d{1,2}, \d{4}')
# reporters' identity with timestamp
reporters_rx = re.compile(r"By\s+.*?\d\d{1,2}\s*-")


replacements = [
    'Advertisement',
    'Advertisement Story continues below advertisement',
    'Advertising Read more',
    'Share Pin Pocket WhatsApp Buffer Email Reddit',
    'NEW You can now listen to Fox News articles!',
    'Once you have subscribed we will use the email you provided to send you the newsletter. You can unsubscribe at any time by clicking the unsubscribe link at the bottom of the newsletter email.',
    'CNN —',
    'Comment on this story Comment Gift Article Share',
    'Get all the latest news on coronavirus and more delivered daily to your inbox. Sign up here.',
    'Get breaking news alerts and special reports. The news and stories that matter, delivered weekday mornings.',
    'Here are today’s top news, analysis and opinion. Know all about the latest news and other news updates from Hindustan Times.',
    'Image copyright',
    'Getty Images',
    'Image caption',
    'PA Media',
    'Last updated on .From the section',
    'Listen to this article',
    'News updates from Hindustan Times',
    'Placeholder while article actions load',
    'Share this article Share Comment on this story Comment',
    'Sign Up For Newsletters',
    'This is a carousel. Use Next and Previous buttons to navigate',
    'FILE PHOTO:'
]

right_substring = [
    '/Associated Press',
    '/PRNewswire/',
    '(AP) -',
    '(AP) —',
    '(AP) _',
    '(Reuters) -',
    'Comment Gift Article Share',
    r'{{ /verifyErrors }}'
]

substrings = [
    'IST Source:',
    'am IST',
    'pm IST'
]

left_substring = [
    'First Published:',
    'Photo:'
]

split_chars = ['-', '—']

exclusions = [
    'winning numbers',  # lottery
    'lotter',  # lottery
    r') _',  # lottery
    r'(\d+)\s+shares',  # sharemarket
    'AccuWeather',  # weather
    'top stories making headlines',  # ads
    '(1st:',  # lottery
    'Sorry, we do not have any active offers in your city',  # error
    r'{{ }}',  # error
    'All zodiac signs have',  # zodiac
    'CBS Essentials is created independently'  # ads
]

excluded_domains = [
    'boereport.com',
    'it.eurosport.com',
    'video-it.eurosport.com',
    'www.hulldailymail.co.uk',
    'www.jpost.com',
    'www.mozzartsport.com',
    'www.tickerreport.com',
    'www.wrdw.com'
]

exclusion_pattern = '|'.join(re.escape(p) for p in exclusions)


def clean_text(text):

    # remove whitespace, line breaks.
    text = ' '.join(text.split())

    text = html_rx.sub('', text)
    text = url_rx.sub('', text)
    text = emoji_rx.sub('', text)
    text = esc_rx.sub('', text)
    text = ref_rx.sub('', text)
    text = reuters_rx.sub('', text)
    text = report_rx.sub('', text)
    text = afp_src_rx.sub('', text)
    text = ap_src_rx.sub('', text)
    text = cbs_src_rx.sub('', text)
    text = cnn_src_rx.sub('', text)
    text = ap_end_src_rx.sub('', text)
    text = starting_brac_rx.sub('', text)
    text = domain_rx.sub('', text)
    text = prnews_src_rx.sub('', text)
    text = htimes_rx.sub('', text)
    text = hindu_rx.sub('', text)
    text = date_first_rx.sub('', text)
    text = reporters_rx.sub('', text)

    for pattern in replacements:
        text = text.replace(pattern, '')

    for substring in right_substring:
        idx = text.rfind(substring)

        if idx != -1:
            text = text[idx + len(substring):]

    for substring in substrings:
        idx = text.find(substring)

        if idx != -1:
            text = text[idx + len(substring):]

    for substring in left_substring:
        idx = text.rfind(substring)

        if idx != -1:
            text = text[:idx]

    if text.startswith('By'):
        idx = text.find('-')
        if idx != -1:
            text = text[idx + 1:]

    for split_char in split_chars:
        texts = text.split(split_char)
        if len(texts) > 1 and len(texts[0].split()) < 5:
            text = ''.join(texts[1:])
            break

    return text.strip()


df = pd.read_csv('data.csv')
df.drop_duplicates(subset=['text'], inplace=True, ignore_index=True)

df['text'] = df['text'].apply(clean_text)

df = df[~df['text'].str.contains(exclusion_pattern, case=False, regex=True)]
df = df[~df['domain'].isin(excluded_domains)]

df['text_len'] = df['text'].str.split().str.len()
df = df[df['text_len'] > 30]

df.drop(['text_len'], axis=1).to_csv('processed_data_full.csv', index=False)
