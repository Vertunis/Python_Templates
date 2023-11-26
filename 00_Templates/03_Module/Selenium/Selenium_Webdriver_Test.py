from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
import time

# Pfad zum Chromedriver aus deinem System
chromedriver_path = rf'C:\webdrivers\chromedriver.exe'  # Muss manuell installiert und Pfadvariable muss angelegt werden

# URL zu deiner PHP-Seite
#url = rf'http://localhost/Button.php' # Link zu Testfile in XAMPP
url = rf'http://localhost/Button_Google.php'
#url = rf'http://localhost/Button_farbige_seite.php'
# Starte den Chrome-Browser
service = ChromeService(chromedriver_path)
driver = webdriver.Chrome(service=service)

try:
    # Öffne die Webseite
    driver.get(url)

    # Warte, bis die Seite vollständig geladen ist (kann angepasst werden)
    driver.implicitly_wait(10)


    # Finde den Button nach dem Text "Weiterleiten"
    hinweis_button = driver.find_element(By.XPATH, '//button[text()="Weiterleiten"]')

    time.sleep(5)  # Pausiere für 5 Sekunden (kann angepasst werden)

    # Klicke auf den Button
    hinweis_button.click()

    # Warte kurz, um das Popup zu sehen (kann angepasst werden)
    driver.implicitly_wait(20)
    time.sleep(5)  # Pausiere für 5 Sekunden (kann angepasst werden)
    # Gib eine Nachricht aus und warte auf Benutzereingabe, bevor das Skript beendet wird
    input("Drücke Enter, um das Skript zu beenden.")

finally:
    print("success")
    # Kommentiere oder entferne den driver.quit()-Befehl
    driver.quit()
