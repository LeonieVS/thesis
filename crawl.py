#Authors: __"Tom De Smedt, Masha Ivenskaya"__
# Each time you run this script (which requires Pattern),
# it collects articles from known sources and their bias,
# and appends to a CSV-file (/data/news1.csv)

from pattern.db  import Datasheet
from pattern.db  import pd
from pattern.web import Newsfeed
from pattern.web import URL
from pattern.web import DOM
from pattern.web import plaintext

sources = {

    #('neutral', 'De Standaard') : 'https://www.standaard.be/rss/section/1f2838d4-99ea-49f0-9102-138784c7ea7c',
    #('neutral', 'De Standaard') : 'https://www.standaard.be/rss/section/e70ccf13-a2f0-42b0-8bd3-e32d424a0aa0',
    #('neutral', 'De Standaard') : 'https://www.standaard.be/rss/section/451c8e1e-f9e4-450e-aa1f-341eab6742cc',
    ('neutral', 'De Tijd') : 'https://www.tijd.be/rss/politiek_belgie.xml',
    ('neutral', 'De Tijd') : 'https://www.tijd.be/rss/politiek_internationaal.xml',
    ('neutral', 'De Tijd') : 'https://www.tijd.be/rss/politiek_europa.xml',
    #('neutral', 'De Morgen') : 'https://www.demorgen.be/nieuws/rss.xml',
    #('neutral', 'De Morgen') : 'https://www.demorgen.be/binnenland/rss.xml',
    #('neutral', 'De Morgen') : 'https://www.demorgen.be/buitenland/rss.xml',
    ('neutral', 'VRT') : 'https://www.vrt.be/vrtnws/nl.rss.headlines.xml',
    ('neutral', 'NOS') : 'http://feeds.nos.nl/nosnieuwsalgemeen',
    ('neutral', 'NOS') : 'http://feeds.nos.nl/nosnieuwsbinnenland',
    ('neutral', 'NOS') : 'http://feeds.nos.nl/nosnieuwsbuitenland',
    ('neutral', 'NOS') : 'http://feeds.nos.nl/nosnieuwspolitiek',
    ('neutral', 'Volkskrant') : 'https://www.volkskrant.nl/nieuws/rss.xml',
    ('neutral', 'Volkskrant') : 'https://www.volkskrant.nl/buitenland/rss.xml',
    ('neutral', 'Belang van Limburg') : 'http://www.hbvl.be/rss/section/D1618839-F921-43CC-AF6A-A2B200A962DC',
    ('neutral', 'Belang van Limburg') : 'http://www.hbvl.be/rss/section/A160C0A6-EFC9-45D8-BF88-86B6F09C92A6',
    ('neutral', 'Belang van Limburg') : 'http://www.hbvl.be/rss/section/FBAF3E6E-21C4-47D3-8A71-902A5E0A7ECB',
    ('neutral', 'Belang van Limburg') : 'http://www.hbvl.be/rss/section/18B4F7EE-C4FD-4520-BC73-52CACBB3931B',
    ('neutral', 'Nieuwsblad') : 'https://feeds.nieuwsblad.be/nieuws/binnenland',
    ('neutral', 'Nieuwsblad') : 'https://feeds.nieuwsblad.be/nieuwsblad/buitenland',
    ('neutral', 'Nieuwsblad') : 'https://feeds.nieuwsblad.be/economie/home',
    ('neutral', 'Knack') : 'http://www.knack.be/nieuws/feed.rss',
    ('neutral', 'Gazet Van Antwerpen') : 'https://www.gva.be/rss/section/5685D99C-D24E-4C8A-9A59-A2AC00E293B1',
    ('neutral', 'Gazet Van Antwerpen') : 'https://www.gva.be/rss/section/472C34EB-D1BC-4E10-BDE7-A2AC00E2D820',
    ('neutral', 'Gazet Van Antwerpen') : 'https://www.gva.be/rss/section/ca750cdf-3d1e-4621-90ef-a3260118e21c',
    ('neutral', 'RTL') : 'http://www.rtlnieuws.nl/service/rss/nederland/index.xml',
    ('neutral', 'RTL') : 'http://www.rtlnieuws.nl/service/rss/nederland/politiek/index.xml',
    ('neutral', 'RTL') : 'http://www.rtlnieuws.nl/service/rss/buitenland/index.xml',
    ('neutral', 'RTL') : 'https://www.gva.be/rss/section/ca750cdf-3d1e-4621-90ef-a3260118e21c',
    #('links', 'Alert Magazine') : 'http://www.alertmagazine.nl/?feed=rss2',
    #('links', 'De Wereld Morgen') : 'http://www.dewereldmorgen.be/rss/actueel',
    #('links', 'Kafka') : 'https://kafka.nl/feed/',
    #('links', 'Sap Rood') : 'https://www.sap-rood.org/feed/',
    #('links', 'Radio Centraal') : 'https://redactie.radiocentraal.be/Home/feed/',
    #('links', 'Trouw') : 'https://www.trouw.nl/home/rss.xml',
    #('links', 'Marxisme.be') : 'https://nl.marxisme.be/marxisme-vandaag/feed/',
    #('links', 'Uitpers') : 'http://www.uitpers.be/feed/',
    #('links', 'Krapuul') : 'http://www.krapuul.nl/feed/',

}

PATH = pd('news.csv')

try:
    csv = Datasheet.load(PATH)
    seen = set(csv.columns[-2]) # use url as id
except:
    csv = Datasheet()
    seen = set()

for (label, name), url in sources.items():
    try:
        f = Newsfeed()
        f = f.search(url, cached=False)
    except:
        continue

    for r in f:

        # 1) Download source & parse the HTML tree:
        try:
            src = URL(r.url).download(cached=True)
            dom = DOM(src)
        except Exception as e:
            continue

        # 2) Find article text w/ CSS selectors:
        for selector in (
      "article[class*='node-article']",            # The Hill
         "span[itemprop='articleBody']",
          "div[itemprop='articleBody']",
          "div[id='rcs-articleContent'] .column1", # Reuters
          "div[class='story-body']",
          "div[class='article-body']",
          "div[class='article-content']",
          "div[class^='tg-article-page']",
          "div[class^='newsArticle']",
          "div[class^='article-']",
          "div[class^='article_']",
          "div[class*='article']",
          "div[id*='storyBody']",                  # Associated Press
          "article",
          ".story"):
            e = dom(selector)
            if e:
                e = e[0]
                break

        # 3) Remove ads, social links, ...
        try:
            e("div[id='rightcolumn']")[0]._p.extract()
            e("div[class='section-blog-right']")[0]._p.extract()
            e("div[class='blog-sidebar-links']")[0]._p.extract()
            e("div[role='complementary']")[0]._p.extract()
        except:
            pass

        # 4) Remove HTML tags:
        try:
            s = plaintext(e)
            s = s.strip()
        except:
            continue

        #if not s:
        #    print r.url
        #    print
        #    continue

        # 5) Save to CSV:
        if r.url not in seen:
            seen.add(r.url)
            csv.append((
                name,
                r.title,
                s,
                label,
            ))
            print(name, r.title)


    csv.save(pd(PATH))

# To read the dataset:
#for name, label, article in Datasheet.load(PATH):
    #print(name, label, article)
#datasheet = Datasheet.load(PATH)
#print(datasheet)
