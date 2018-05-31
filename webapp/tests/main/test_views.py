from flask import url_for


class TestPage(object):
    def test_home_page(self, client):
        """ Home page should respond with a success 200. """
        response = client.get(url_for('main.home'))
        assert response.status_code == 200

    def test_cats_page(self, client):
        """ Terms page should respond with a success 200. """
        response = client.get(url_for('main.cats'))
        assert response.status_code == 200
