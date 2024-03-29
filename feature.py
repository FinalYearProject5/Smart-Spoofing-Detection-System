import ipaddress
import re
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
from googlesearch import search
import whois
from datetime import date, datetime
import time
from dateutil.parser import parse as date_parse
from urllib.parse import urlparse


class featureextraction:
     features = []
     def __init__(self,url):
        self.features = []
        self.url = url
        self.domain = ""
        self.whois_response = ""
        self.urlparse = ""
        self.response = ""
        self.soup = ""

        try:
            self.response = requests.get(url)
            self.soup = BeautifulSoup(response.text, 'html.parser')
        except:
            pass

        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except:
            pass

        try:
            self.whois_response = whois.whois(self.domain)
        except:
            pass


        

        self.features.append(self.UsingIp())
        self.features.append(self.getLength())
        self.features.append(self.tinyURL())
        self.features.append(self.haveAtSign())
        self.features.append(self.redirection())
        self.features.append(self.prefixSuffix())
        self.features.append(self.SubDomains())
        self.features.append(self.httpDomain())
        self.features.append(self.RegLen())
        self.features.append(self.Favicon())
        

        self.features.append(self.NonStdPort())
        self.features.append(self.HTTPSDomainURL())
        self.features.append(self.Request())
        self.features.append(self.Anchor())
        self.features.append(self.LinksInScript())
        self.features.append(self.FormHandler())
        self.features.append(self.Info())
        self.features.append(self.AbnormalURL())
        self.features.append(self.Forwarding())
        self.features.append(self.StatusBar())

        self.features.append(self.RightClick())
        self.features.append(self.PopupWindow())
        self.features.append(self.Iframe())
        self.features.append(self.AgeofDomain())
        self.features.append(self.Recording())
        self.features.append(self.Traffic())
        self.features.append(self.PageRank())
        self.features.append(self.Index())
        self.features.append(self.Links())
        self.features.append(self.Stats())

     # 1.UsingIp
     def UsingIp(self):
        try:
            ipaddress.ip_address(self.url)
            return -1
        except:
            return 1

    # 2.longUrl
     def getLength(self):
      url_length = len(self.url)
      if url_length < 54:
        return 1
      elif 54 <= url_length <= 75:
        return 0
      else:
        return -1


    # 3.shortUrl
     def tinyURL(self):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                    'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                    'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                    'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                    'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                    'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                    'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net', self.url)
        if match:
            return -1
        return 1

    # 4.Symbol@
     def haveAtSign(self):
      if '@' in self.url:
        return -1
      else:
        return 1
    
    # 5.Redirecting//
     def redirection(self):
      return -1 if self.url.rfind('//') > 6 else 1

    
    # 6.prefixSuffix
     def prefixSuffix(self):
        try:
            res = re.findall('\-', self.domain)
            if res:
                return -1
            return 1
        except:
            return -1
    
    # 7.SubDomains
     def SubDomains(self):
        count = len(re.findall("\.", self.url))
        if count == 1:
            return 1
        elif count == 2:
            return 0
        return -1

    # 8.HTTPS
     def httpDomain(self):
      try:
        if 'https' in self.urlparse.scheme:
            return 1
        else:
            return -1
      except:
        return -1


    # 9.DomainRegLen
     def RegLen(self):
        try:
            exdate = self.whois_response.exdate
            credate = self.whois_response.credate
            try:
                if(len(exdate)):
                    exdate = exdate[0]
            except:
                pass
            try:
                if(len(credate)):
                    credate = credate[0]
            except:
                pass

            a = (exdate.year-credate.year)*12+ (exdate.month-credate.month)
            if a >=12:
                return 1
            return -1
        except:
            return -1

    # 10. Favicon
     def Favicon(self):
        try:
            for head in self.soup.find_all('head'):
                for head.link in self.soup.find_all('link', href=True):
                    dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
                    if self.url in head.link['href'] or len(dots) == 1 or domain in head.link['href']:
                        return 1
            return -1
        except:
            return -1

    # 11. NonStdPort
     def NonStdPort(self):
        try:
            domain_parts = self.domain.split(":")
            if len(domain_parts)>1:
                return -1
            return 1
        except:
            return -1

    # 12. HTTPSDomainURL
     def HTTPSDomainURL(self):
        try:
            if 'https' in self.domain:
                return -1
            else:
                return 1
        except:
            return -1
    
    # 13. RequestURL
     def Request(self):
        try:
            for img in self.soup.find_all('img', src=True):
                dots = [x.start(0) for x in re.finditer('\.', img['src'])]
                if self.url in img['src'] or self.domain in img['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for audio in self.soup.find_all('audio', src=True):
                dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
                if self.url in audio['src'] or self.domain in audio['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for embed in self.soup.find_all('embed', src=True):
                dots = [x.start(0) for x in re.finditer('\.', embed['src'])]
                if self.url in embed['src'] or self.domain in embed['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for iframe in self.soup.find_all('iframe', src=True):
                dots = [x.start(0) for x in re.finditer('\.', iframe['src'])]
                if self.url in iframe['src'] or self.domain in iframe['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            try:
                percentage = success/float(i) * 100
                if percentage < 22.0:
                    return 1
                elif((percentage >= 22.0) and (percentage < 61.0)):
                    return 0
                else:
                    return -1
            except:
                return 0
        except:
            return -1
    
    # 14. AnchorURL
     def Anchor(self):
        try:
            i,unsafe = 0,0
            for a in self.soup.find_all('a', href=True):
                if "#" in a['href'] or "javascript" in a['href'].lower() or "mailto" in a['href'].lower() or not (url in a['href'] or self.domain in a['href']):
                    unsafe = unsafe + 1
                i = i + 1

            try:
                percentage = unsafe / float(i) * 100
                if percentage < 31.0:
                    return 1
                elif ((percentage >= 31.0) and (percentage < 67.0)):
                    return 0
                else:
                    return -1
            except:
                return -1

        except:
            return -1

    # 15. LinksInScriptTags
     def LinksInScript(self):
        try:
            i,success = 0,0
        
            for link in self.soup.find_all('link', href=True):
                dots = [x.start(0) for x in re.finditer('\.', link['href'])]
                if self.url in link['href'] or self.domain in link['href'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            for script in self.soup.find_all('script', src=True):
                dots = [x.start(0) for x in re.finditer('\.', script['src'])]
                if self.url in script['src'] or self.domain in script['src'] or len(dots) == 1:
                    success = success + 1
                i = i+1

            try:
                percentage = success / float(i) * 100
                if percentage < 17.0:
                    return 1
                elif((percentage >= 17.0) and (percentage < 81.0)):
                    return 0
                else:
                    return -1
            except:
                return 0
        except:
            return -1

    # 16. ServerFormHandler
    
     def FormHandler(self):
      try:
        forms_with_action = self.soup.find_all('form', action=True)

        if len(forms_with_action) == 0:
            # If there are no forms with actions, return 1
            return 1
        else:
            for form in forms_with_action:
                if form['action'] == "" or form['action'] == "about:blank":
                    # If the form action is empty or about:blank, return -1
                    return -1
                elif self.url not in form['action'] and self.domain not in form['action']:
                    # If the form action is neither the URL nor the domain, return 0
                    return 0
                else:
                    # If the form action is either the URL or the domain, return 1
                    return 1
      except:
        # Return -1 if any exception occurs during the process
        return -1


# 17. InfoEmail
     def Info(self):
      try:
        # Using regular expression to find email-related patterns in the HTML content
        email_patterns = re.findall(r"(mail\(\)|mailto:?)", self.soup)

        if email_patterns:
            # If email patterns are found, return -1 indicating the presence of email-related content
            return -1
        else:
            # If no email patterns are found, return 1 indicating no email-related content
            return 1
      except:
        # If any exception occurs during the process, return -1
        return -1


    # 18. AbnormalURL
     def AbnormalURL(self):
      try:
        # Check if the response text matches the whois response
        if self.response.text == self.whois_response:
            # If the response text matches the whois response, return 1
            return 1
        else:
            # If the response text does not match the whois response, return -1
            return -1
      except:
        # If any exception occurs during the process, return -1
        return -1


    # 19. WebsiteForwarding
     def Forwarding(self):
      try:
        # Check the number of redirections in the response history
        num_redirections = len(self.response.history)
        
        if num_redirections <= 1:
            # If there is only one redirection or none, return 1
            return 1
        elif num_redirections <= 4:
            # If there are between 2 and 4 redirections, return 0
            return 0
        else:
            # If there are more than 4 redirections, return -1
            return -1
      except:
        # If any exception occurs, return -1 to indicate an error
        return -1


    # 20. StatusBarCust
     def StatusBar(self):
        try:
            if re.findall("<script>.+onmouseover.+</script>", self.response.text):
                return 1
            else:
                return -1
        except:
             return -1

    # 21. DisableRightClick
     def RightClick(self):
        try:
            if re.findall(r"event.button ?== ?2", self.response.text):
                return 1
            else:
                return -1
        except:
             return -1

    # 22. UsingPopupWindow
     def PopupWindow(self):
        try:
            if re.findall(r"alert\(", self.response.text):
                return 1
            else:
                return -1
        except:
             return -1

    # 23. IframeRedirection
     def Iframe(self):
        try:
            if re.findall(r"[<iframe>|<frameBorder>]", self.response.text):
                return 1
            else:
                return -1
        except:
             return -1

    # 24. AgeofDomain
     def AgeofDomain(self):
        try:
            credate = self.whois_response.credate
            try:
                if(len(credate)):
                    credate = credate[0]
            except:
                pass

            today  = date.today()
            a = (today.year-credate.year)*12+(today.month-credate.month)
            if a >=6:
                return 1
            return -1
        except:
            return -1

    # 25. DNSRecording    
     def Recording(self):
        try:
            credate = self.whois_response.credate
            try:
                if(len(credate)):
                    credate = credate[0]
            except:
                pass

            today  = date.today()
            a = (today.year-credate.year)*12+(today.month-credate.month)
            if a >=6:
                return 1
            return -1
        except:
            return -1

    # 26. WebsiteTraffic   
     def Traffic(self):
        try:
            rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find("REACH")['RANK']
            if (int(rank) < 100000):
                return 1
            return 0
        except :
            return -1

    # 27. PageRank
     def PageRank(self):
        try:
            prank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {"name": self.domain})

            global_rank = int(re.findall(r"Global Rank: ([0-9]+)", prank_checker_response.text)[0])
            if 0 < global_rank < 100000:
                return 1
            else:
             return -1
        except:
            return -1
            

    # 28. GoogleIndex
     def Index(self):
        try:
            site = search(self.url, 5)
            if site:
                return 1
            else:
                return -1
        except:
            return 1

    # 29. LinksPointingToPage
     def Links(self):
        try:
            links = len(re.findall(r"<a href=", self.response.text))
            if links == 0:
                return 1
            elif links <= 2:
                return 0
            else:
                return -1
        except:
            return -1

    # 30. StatsReport
     def Stats(self):
        try:
            url_match = re.search(
        'at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly', url)
            ip_address = socket.gethostbyname(self.domain)
            ip_match = re.search('146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|'
                                '107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|'
                                '118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|'
                                '216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|'
                                '34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|'
                                '216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42', ip_address)
            if url_match:
                return -1
            elif ip_match:
                return -1
            return 1
        except:
            return 1
    



     def featureslist(self):
        return self.features
