from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# Webbrowser starten und zur URL navigieren
browser = webdriver.Chrome()
browser.get('https://www.google.de/')

# Button finden und darauf klicken
button = browser.find_element_by_xpath("//button[@id='mybutton']")
button.click()

# Formular ausfüllen und abschicken
username = browser.find_element_by_name('username')
password = browser.find_element_by_name('password')
username.send_keys('myusername')
password.send_keys('mypassword')
password.send_keys(Keys.RETURN)

# Browser schließen
browser.quit()