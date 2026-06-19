import requests
from bs4 import BeautifulSoup
import pandas as pd  

url = "https://realpython.github.io/fake-jobs/"
print("🚚 Fetching all webpage data...")
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")


job_cards = soup.find_all("div", class_="card-content")


titles = []
companies = []
locations = []

print(f"🔍 Parsing through {len(job_cards)} job listings found on the page...")


for card in job_cards:
    title_element = card.find("h2", class_="title")
    company_element = card.find("h3", class_="company")
    location_element = card.find("p", class_="location")
    

    titles.append(title_element.text.strip())
    companies.append(company_element.text.strip())
    locations.append(location_element.text.strip())


job_data = pd.DataFrame({
    "Job Title": titles,
    "Company Name": companies,
    "Location": locations
})


job_data.to_csv("scraped_jobs.csv", index=False)
print("💾 Success! All listings saved to 'scraped_jobs.csv'!")