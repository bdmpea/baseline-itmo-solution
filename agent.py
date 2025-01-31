import requests
import json
from bs4 import BeautifulSoup

def get_itmo_news(topics):
    news_list = []
    for topic in topics:
        for i in range(1, 25):
            url = "https://news.itmo.ru/ru/" + topic + "/" + str(i)
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("h4") 
            
            for article in articles:
                title = article.find("a").text.strip()
                link = "https://news.itmo.ru" + article.find("a")["href"]
                if topic in link:
                    print(link)
                    soup1 = BeautifulSoup(requests.get(link).text, "html.parser")
                    text = soup1.find_all("div", class_='content js-mediator-article')[0].text
                    news_list.append({"topic": topic, "title": title, "text": text, "link": link})
        
    return news_list


topics1 = ['science/it', 'science/photonics', 'science/cyberphysics', 'science/new_materials', 'science/life_science']
topics2 = ['education/cooperation', 'education/trend', 'education/students', 'education/official']
topics3 = ['startups_and_business/business_success', 'startups_and_business/innovations', 'startups_and_business/startup','startups_and_business/initiative']
topics4 = ['university_life/ratings', 'university_life/achievements', 'university_life/leisure', 'university_life/adds', 'university_life/sicial_activity']

news = get_itmo_news(topics1 + topics2 + topics3 + topics4)
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in news:
    data.append(item) 

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
