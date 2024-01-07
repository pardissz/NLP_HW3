import os
import requests
from bs4 import BeautifulSoup
import csv

def get_html(url):
    try:
        response = requests.get(url)
        # response.raise_for_status()
    except:
        print('Error getting HTML content')
        return None
    return response.text

def parse_drug_page(url):
    html = get_html(url)
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('figure', class_='wp-block-table')

    data = {}
    if len(tables) >= 2:
        # Extract data from the first table
        rows = tables[0].find_all('tr')
        if len(rows) > 1:
            cells = rows[1].find_all('td')
            data.update({
                'generic_name': cells[0].get_text(strip=True),
                'brand_name': cells[1].get_text(strip=True),
                'shekl_daroei': cells[2].get_text(strip=True),
                'pharmacologic_category': cells[4].get_text(strip=True),
                'goroh_darmani': cells[5].get_text(strip=True)
            })

        # Extract data from the second table
        rows = tables[1].find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if cells and len(cells) == 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                if key and value:
                    data[key] = value

    return data

def get_drug_urls(main_url, file_path):
    if os.path.exists(file_path):
        print("Reading URLs from file...")
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]
    
    main_html = get_html(main_url)
    main_soup = BeautifulSoup(main_html, 'html.parser')
    groups = main_soup.main.find_all('a', href=True)

    print(groups)

    drug_urls = []
    for i, group in enumerate(groups):
        # vce-button--style-basic--border-square
        if 'vce-button--style-basic' in group.get('class', []):
            print(f'Processing group {i + 1}')
            group_html = get_html(group['href'])
            group_soup = BeautifulSoup(group_html, 'html.parser')
            subgroups = group_soup.main.find_all('a', href=True)

            for i, subgroup in enumerate(subgroups):
                # vce-button--style-basic--border-square
                if 'vce-button--style-basic' in subgroup.get('class', []):
                    print(f'Processing subgroup {i + 1}')
                    subgroup_html = get_html(subgroup['href'])
                    subgroup_soup = BeautifulSoup(subgroup_html, 'html.parser')
                    try:
                        drugs = subgroup_soup.main.find_all('a', href=True)
                    except:
                        print('Error getting drugs: ', subgroup['href'])

                    for drug in drugs:
                        # vce-button--style-outline--size-medium
                        if 'vce-button--style-outline--border-square' in drug.get('class', []):
                            drug_urls.append(drug['href'])
                            print('added', drug['href'])
                elif 'vce-button--style-outline--border-square' in subgroup.get('class', []):
                    drug_urls.append(subgroup['href'])
                    print('added', subgroup['href'])

    print('Writing drug URLs to a text file')
    with open(file_path, 'w') as file:
        for url in drug_urls:
            file.write(url + '\n')

    return drug_urls
# Main crawling function
if __name__ == '__main__':
    url = 'https://exir.co.ir/بانک-اطلاعات-دارويي/'
    file_path = 'exir_drug_urls.txt'
    csv_path = 'exir_drugs.csv'

    drug_urls = get_drug_urls(url, file_path)

    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['generic_name', 'brand_name', 'shekl_daroei', 'pharmacologic_category', 'goroh_darmani', 'mavared_masraf', 'nahve_masraf', 'mavared_mane', 'mavared_ehtiyat_masraf', 'avarez_shayeh', 'masraf_hamelegi', 'moshkel_kabedi', 'test_moredeh_niyaz'])

        for i, url in enumerate(drug_urls):
            print(f'Processing drug {i + 1}')
            data = parse_drug_page(url)
            if data:
                writer.writerow(data.values())
