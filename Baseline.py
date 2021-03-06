from Strategy import Strategy
import requests


class Baseline(Strategy):

    def __init__(self):
        self.name = 'Baseline'
        self.k = None

    # ignore ip address
    def get_recommendations(self, user_index, known_user_projects, k, ip_address):
        self.k = k
        csrf_token = requests.get('https://scistarter.org').text.split('csrfmiddlewaretoken" value="')[1].split('"')[0]
        response = requests.post('https://scistarter.org/ui/request', json={'key': 'collection', 'slug': 'recommended'},
                                 headers={'Referer': 'https://scistarter.org/',
                                          'X-CSRFToken': csrf_token},
                                 cookies={
                                     'csrftoken': csrf_token})
        entities = response.json()['messages'][0]['entities']
        return [ent['id'] for ent in entities][:k]

    # here, also recommend Scistarter projects
    def get_highest_online_project(self):
        return self.get_recommendations(-1, None, self.k, None)[-1]
