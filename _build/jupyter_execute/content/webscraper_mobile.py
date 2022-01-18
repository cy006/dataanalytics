#!/usr/bin/env python
# coding: utf-8

# # 2. Webscraper mobile.de

# Im Rahmen meiner Recherche nach passenden Daten, konnte ich nichts passendes finden. So entschied ich mich dazu, einen Webscraper für die Website "mobile.de" zu entwickeln. Mobile.de hat auf den einzelnen Fahrzeugseiten viele Fahrzeuginformationen, die ich in meinen Recherchen als potenzielle relevante Variablen identifiziert habe. Da der Preis auf dieser Website durch den Insersatersteller gesetzt wird, besteht natürlich die Gefahr, dass realitätsferne Preise möglich sind. Um dieses Risiko zu reduzieren, habe ich nur die Inserate gescrapt die von den Händlern erstellt wurden. Um den Leasingcharakter der gescrapten Fahrzahreuge beizubehalten wurden nur Fahrzeuge gescrapt deren Erstzulassung zwischen 2017 und 2021 liegt.

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait  
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
#import winsound
 
import time
import pandas as pd


price_minimum = "1000"
price_maximum = "200000"


PATH = r"/Users/cenkyagkan/Downloads/chromedriver"
url = r"https://www.mobile.de"
ort_plz = "Deutschland"
num_cars_scrape = 1000


while True:
    cars = []
    driver = webdriver.Chrome(PATH)
    driver.maximize_window()
    driver.get(url)
    time.sleep(15)
    
    try:
        driver.find_element_by_id("gdpr-consent-accept-button").click()
    except:
        pass
    
    driver.find_element_by_id("qsdet").click()
    time.sleep(5)
    
    # gebraucht
    driver.find_element_by_id("usage-USED-ds").click()
    time.sleep(5)
    
    # Marke
    #select Mercedes-Benz
    wait = WebDriverWait(driver, 1)
    element = wait.until(EC.visibility_of_element_located((By.ID, "selectMake1-ds")))
    select = Select(element)
    select.select_by_visible_text("Mercedes-Benz")
    time.sleep(3)
    
    #minPrice
    price_min = driver.find_element_by_id("minPrice")
    price_min.send_keys(price_minimum)
    time.sleep(0.5)
    #maxPrice
    price_max = driver.find_element_by_id("maxPrice")
    price_max.send_keys(price_maximum)
    time.sleep(0.5)
    
    #Erstzulassung
    erstzulassung_min = driver.find_element_by_id("minFirstRegistrationDate")
    erstzulassung_min.send_keys("2017")
    time.sleep(0.5)
    erstzulassung_max = driver.find_element_by_id("maxFirstRegistrationDate")
    erstzulassung_max.send_keys("2021")
    time.sleep(0.5)
    
    
    #Ortsuche
    search_ort_plz = driver.find_element_by_id("ambit-search-location")
    search_ort_plz.clear()
    search_ort_plz.send_keys(ort_plz)
    search_ort_plz.send_keys(Keys.RETURN)
    time.sleep(10)
    
    # Neuste Inserate zuerst anzeigen
    wait0 = WebDriverWait(driver, 5)
    element = wait0.until(EC.visibility_of_element_located((By.ID, "so-sb")))
    select = Select(element)
    select.select_by_visible_text("Neueste Inserate zuerst")
    #select.select_by_visible_text("Kilometer absteigend")
    #select.select_by_visible_text("Älteste Inserate zuerst")
    #select.select_by_visible_text("Standard-Sortierung")
    time.sleep(10)
    
    #buttons = driver.find_elements_by_xpath('.//a[@class="link--muted no--text--decoration result-item"]')
    #buttons = driver.find_elements_by_xpath('.//div[@class="cBox-body cBox-body--resultitem fsboAd rbt-reg rbt-no-top"]')
    try:
        buttons = driver.find_elements_by_xpath('.//div[@class="cBox-body cBox-body--resultitem dealerAd"]')
        button = buttons[2]
    except:
        buttons = driver.find_elements_by_xpath('.//div[@class="cBox-body cBox-body--resultitem fsboAd"]')
        button = buttons[2]
            
    
    ActionChains(driver).move_to_element(button).click(button).perform()
    #button.click()
    time.sleep(8)
        
    for i in range(num_cars_scrape):
        time.sleep(0.2)  
        
        try:
            WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.ID, "z1234")))
            WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.XPATH, './/span[@class="h3 prime-price"]')))
            time.sleep(1)
        except:
            print('exce__________')
            time.sleep(4)
        
        # dcoreSurveyOverlay  -- CloseDCoreOverlay
        #try:
        #    umfrage = driver.find_element_by_id("CloseDCoreOverlay")
        #    umfrage.click()
        #except:
        #    pass
        
        # CARNAME
        try:
            carname = driver.find_element_by_id("ad-title").text
        except:
            carname = -1
            
        # Preis
        try:
            price = driver.find_element_by_class_name('contact-vehicle-info__price').text
        except:
            price = -1
         
        # KILOMETERSTAND
        try:
            milage = driver.find_element_by_id("mileage-v") #  .text
            milage.location_once_scrolled_into_view
            milage = milage.text
        except:
            milage = -1
        
        # HUBRAUM
        try:
            hubraum = driver.find_element_by_id("cubicCapacity-v").text
        except:
            hubraum = -1
        
        # LEISTUNG 
        try:
            power = driver.find_element_by_id("power-v").text
        except:
            power = -1
        
        # KRAFTSTOFF
        try:
            fuel_type = driver.find_element_by_id("fuel-v").text
        except:
            fuel_type = -1
        
        # GETRIEBE
        try:
            transmission = driver.find_element_by_id("transmission-v").text
        except:
            transmission = -1
        
        
        # ERSTZULASSUNG 
        try:
            first_registration = driver.find_element_by_id("firstRegistration-v").text
        except:
            first_registration = -1
            
        # BAUJAHR
        try:
            construction_year = driver.find_element_by_id("constructionYear-v").text
        except:
            construction_year = -1
        
        # SITZPLÄTZE
        try:
            num_seats = driver.find_element_by_id("numSeats-v").text
        except:
            num_seats = -1
        
        # ANZAHL TÜREN 
        try:
            num_doors = driver.find_element_by_id("doorCount-v").text
        except:
            num_doors = -1
        
        # EMMISION CLASS
        try:
            emission_class = driver.find_element_by_id("emissionClass-v").text
        except:
            emission_class = -1
        
        # FAHRZEUGTYPE
        try:
            car_type = driver.find_element_by_id("category-v").text
        except:
            car_type = -1
            
        # ANZAHL DERVORBESITZER 
        try:
            num_owners = driver.find_element_by_id("numberOfPreviousOwners-v").text
        except:
            num_owners = -1
            
        # SCHADEN
        try:
            damage = driver.find_element_by_id("damageCondition-v").text
        except:
            damage = -1
        time.sleep(0.2)
        cars.append({"carname" : carname,
                         "price" : price,
                         "milage" : milage,
                         "hubraum" : hubraum,
                         "power" : power,
                         "fuel_type" : fuel_type,
                         "transmission" : transmission,
                         "construction_year" : construction_year,
                         "first_registration" : first_registration,
                         "num_seats" : num_seats,
                         "num_doors" : num_doors,
                         "emission_class" : emission_class,
                         "car_type" : car_type,
                         "num_owners" : num_owners,
                         "damage" : damage})
        
        
        print(i,'/',num_cars_scrape)
        print(price)
        
        # a class = link--muted nav-next u-hide
        # i class gicon-next-s icon--s icon--grey-60 u-no-margin-right
        #EC.visibility_of_element_located    funkt
        #EC.element_to_be_clickable
        # './/a[@class="link--muted nav-next u-hide"]'   funkt
        # './/i[@class="gicon-next-s icon--s icon--grey-60 u-no-margin-right"]'
        try:
            try:
                wait2 = WebDriverWait(driver, 60)
                next_btn = wait2.until(EC.element_to_be_clickable((By.XPATH, './/a[@class="link--muted nav-next u-hide" and @title="Nächstes Inserat"]')))
                #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                #time.sleep(1.2)
                #next_btn = driver.find_element_by_xpath('.//a[@class="link--muted nav-next u-hide" and @title="Nächstes Inserat"]')
                next_btn.location_once_scrolled_into_view
                time.sleep(10)
                next_btn.send_keys(u'\ue007')
            except:
                driver.refresh()
                wait2 = WebDriverWait(driver, 60)
                next_btn = wait2.until(EC.element_to_be_clickable((By.XPATH, './/a[@class="link--muted nav-next u-hide" and @title="Nächstes Inserat"]')))
                #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                #time.sleep(1.2)
                #next_btn = driver.find_element_by_xpath('.//a[@class="link--muted nav-next u-hide" and @title="Nächstes Inserat"]')
                next_btn.location_once_scrolled_into_view
                time.sleep(10)
                next_btn.send_keys(u'\ue007')
            
        except:
            time.sleep(10)
            try:
                try:
                    body = driver.find_element_by_css_selector('body')
                except:
                    body = driver.find_element_by_id("ad-title")
                for i in range(5):
                    body.send_keys(Keys.PAGE_DOWN)
                time.sleep(10)
                next_btn = driver.find_element_by_xpath('.//a[@class="link--muted nav-next u-hide" and @title="Nächstes Inserat"]')
                next_btn.location_once_scrolled_into_view
                next_btn.send_keys(u'\ue007')
            except:
                driver.refresh()
                time.sleep(5)
                el = driver.find_element_by_id("srp-back-link")
                el.location_once_scrolled_into_view
                el.click()
                time.sleep(8)     
                btn = driver.find_element_by_xpath('.//span[@class="btn btn--orange btn--s next-resultitems-page"]')
                btn.click()
                time.sleep(8)
                try:
                    buttons = driver.find_elements_by_xpath('.//div[@class="cBox-body cBox-body--resultitem dealerAd"]')
                    button = buttons[2]
                except:
                    buttons = driver.find_elements_by_xpath('.//div[@class="cBox-body cBox-body--resultitem fsboAd"]')
                    button = buttons[2]
                ActionChains(driver).move_to_element(button).click(button).perform()
                time.sleep(8)
    
    driver.quit()
    print('Scraping succesfull!')
    data = pd.DataFrame(cars)
    break
        
data.to_csv("07_mobile_data_" + price_minimum + "_" + price_maximum + ".csv", index=False)


# In[ ]:





# In[2]:


data = pd.DataFrame(cars)
data.to_csv("234_mobile_data_" + price_minimum + "_" + price_maximum + ".csv", index=False)


# In[3]:


data


# In[ ]:




