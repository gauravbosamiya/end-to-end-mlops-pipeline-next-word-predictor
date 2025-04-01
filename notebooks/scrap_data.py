import requests
from bs4 import BeautifulSoup
import re

lst  = ["https://en.wikipedia.org/wiki/Predictive_text",
        "https://en.wikipedia.org/wiki/Natural_language_processing"
        # "https://en.wikipedia.org/wiki/Mahatma_Gandhi",
        # "https://en.wikipedia.org/wiki/Ratan_Tata",
        # "https://en.wikipedia.org/wiki/P._V._Sindhu",
        # "https://en.wikipedia.org/wiki/India",
        # "https://en.wikipedia.org/wiki/Deep_learning",
        # "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
        # "https://en.wikipedia.org/wiki/Amazon_(company)",
        # "https://en.wikipedia.org/wiki/Gmail",
        ]


def remove_number(text):
    return re.sub(r'\[\d+\]', '', text)

def scrap_text(lst, filename):
    all_content = []
    with open(filename, "w", encoding='utf-8') as f:
        for url in lst:
            res = requests.get(url)
            if res.status_code==200:
                soup = BeautifulSoup(res.text,'html.parser')
            
                for div in soup.find_all('div'):
                    for p in div.find_all("p"):
                        content = p.get_text().strip()
                        cleaned_content = remove_number(content)
                        # print(cleaned_content)
                        all_content.append(cleaned_content)
                        f.write(cleaned_content+"\n")
            else:
                print("Something wrong!!!")
    return all_content


scraped_text = scrap_text(lst, "notebooks/text.txt")
print("Scraping complete.")
