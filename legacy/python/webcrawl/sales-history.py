from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)



url_template = "https://www.domain.com.au/property-profile/{}-28-mort-street-braddon-act-2612"

for i in range(1, 66):
    url = url_template.format(i)
    raw_html = simple_get(url)
    if raw_html is None:
        print ("{} is invalid".format(url))
    else:
        html = BeautifulSoup(raw_html, 'html.parser')
        feature_values = []
        features = html.select('span.property-details-strip__feature-type')
        for feature in features:
            feature_value = feature.find_previous_sibling('span')
            feature_values.append(feature_value.text)
            feature_values.append(feature.text)
        print('{}/28 mort street, {}, {}'.format(i, ' '.join(feature_values), url))
        for li in html.select('li.property-timeline-item'):
            months = li.select('div.property-timeline__card-date-month')
            if len(months) > 0:
                month = months[0].text
                year = li.select('div.property-timeline__card-date-year')[0].text
                category = li.select('div.property-timeline__card-category')[0].text
                price = li.select('span.property-timeline__card-heading')[0].text
                print ("{}/{}, {} {}".format(month, year, category, price))
