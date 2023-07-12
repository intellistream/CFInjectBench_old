import glob
import json
import csv

csv_rows = []
with open('data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    csv_rows.append(['timestamp', 'domain', 'text'])

    for filename in glob.iglob('news-please/cc_download_articles/**/*.json', recursive=True):
        with open(filename, 'r') as file:
            data = json.load(file)

            pub_date = data['date_publish'].split()[0]  # discard time
            main_text = data['maintext']

            if pub_date and main_text:  # pub_date and main_text shouldn't be None
                csv_rows.append([pub_date, data['source_domain'], main_text])

    writer.writerows(csv_rows)
