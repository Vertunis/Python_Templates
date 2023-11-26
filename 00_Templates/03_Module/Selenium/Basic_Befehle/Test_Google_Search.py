from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# Webbrowser starten und zur Google-Startseite navigieren
browser = webdriver.Chrome()
browser.get('https://www.google.com/')

try:
    # Suchfeld finden und den Suchbegriff "Giraffe" eingeben
    search_box = browser.find_element_by_name('q')
    search_box.send_keys('Giraffe')
    search_box.send_keys(Keys.RETURN)
finally:
    # Browser schließen
    browser.quit()