import requests

def check_term_exists(term: str) -> bool:
    """
    Проверяет, существует ли термин в Wikidata.
    Отправляет запрос к API wbsearchentities и возвращает True, если найден хотя бы один результат.
    """
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'ru',
        'search': term
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Если по запросу найден хотя бы один объект – возвращаем True
        return bool(data.get("search"))
    except requests.RequestException as e:
        print(f"Ошибка при запросе для термина '{term}': {e}")
        return False

def search_wikidata(term, language='ru'):
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': language,
        'search': term
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка: {response.status_code}")
        return None