from app import app  # Assumes 'app' is defined in app.py

def test_index_route():
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
