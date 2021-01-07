from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re


PATH = "C:/Users/PC/Documents/Selenium/chromedriver.exe"
#URL = "https://www.investing.com/indices/"
URL = "https://www.investing.com/indices/france-40-components"
START_DATE = ''
END_DATE = ''
MARKET = {'CAC40': 'france-40'}
MARKET_COMPONENTS = {'CAC40': 'france-40-components'}

if __name__ == '__main__':
    driver = webdriver.Chrome(PATH)
    driver.get(URL)

    # TO KEEP
    # chose_europe = driver.find_element_by_id("sml_1241").click()
    # time.sleep(5)
    # send_form = driver.find_element_by_link_text("Search").click()
    # time.sleep(5)
    # cac_40_page = driver.find_element_by_link_text("CAC 40").click()
    # time.sleep(5)
    # cac_40_indexes = driver.find_element_by_xpath('//a[contains(@href,'
    #                                               '"/indices/france-40-components")]').click()

    regex_cac40 = r'pair_(\d*)'
    companies = driver.find_element_by_id("cr1").find_elements_by_tag_name("tr")
    for company in companies[1:]:
        if 'Air Liquide' in company.text:
            company_name = driver.find_element_by_tag_name("a").click()
            time.sleep(5)
            historical_data_page = driver.find_element_by_link_text(
                "Historical Data").click()
            time.sleep(2)
            date_icon = driver.find_element_by_class_name(
                'hasDatepicker').click()
            time.sleep(2)
            start_date = driver.find_element_by_id('startDate')
            start_date.send_keys("01/01/2010")

    # try:
    #     send_form = WebDriverWait(driver, 15).until(
    #         EC.presence_of_element_located((By.LINK_TEXT, "Search"))
    #     )
    #     send_form.click()
    #
    #     cac_40_page = WebDriverWait(driver, 15).until(
    #         EC.presence_of_element_located((By.LINK_TEXT, "CAC 40"))
    #     )
    #     cac_40_page.click()
    #
    # except:
    #     driver.quit()


    # date_icon = driver.find_element_by_class_name('hasDatepicker')
    # date_icon.click()
    # start_date = date_icon.find_element_by_id('startDate')
    # start_date.send_keys("01/01/2010")

    # search = driver.find_element_by_id('curr_table')
    # # search.send_keys("test")
    # # search.send_keys(Keys.RETURN)
    # print(search.text)

    # delete pop ups


    # navigation process
    # enter website
    # click on CAC40 company
    # click on historical data
    # chose from 01/01/2010
    # chose enddate to 31/12/2020
    # retrieve all tr information in tbody
