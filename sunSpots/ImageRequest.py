import warnings

from urllib.request import urlopen
import numpy as np
from cv2 import cv2
from splinter import Browser


class ImageRequest:
    url = 'http://spaceweather.com/'

    def __init__(self):
        self.browser = Browser('firefox')
        self.browser.driver.minimize_window()
        self.browser.driver.set_page_load_timeout(60)
        self.browser.visit(self.url)

    def __del__(self):
        self.browser.quit()

    def get_image(self, day, month, year):
        if month <= 0 or month > 12:
            raise ValueError('Month should be 1 to 12')
        self.browser.select('day', f'{day:02}')
        self.browser.select('month', f'{month:02}')
        self.browser.select('year', year)
        self.browser.find_by_name('view').click()
        link = self.browser.find_by_text('no labels')
        if link:
            link = link['href']
        else:
            warnings.warn('No image without label found. Use labeled image instead')
            link = self.browser.find_link_by_partial_href(f'images{year}/')['href']
        resp = urlopen(link)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

