import requests

class CloudIntegration:
    def __init__(self, service_url):
        self.service_url = service_url

    def send_data(self, data):
        response = requests.post(self.service_url, json=data)
        return response.status_code == 200
