import unittest
from app import app, predict_emotions, get_prediction_proba  # Import the Flask app and functions
from unittest.mock import patch

class TestApp(unittest.TestCase):
    
    def setUp(self):
        # Set up the test client for the Flask app
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.pipe_lr.predict')  # Mocking the prediction function
    def test_predict_emotions(self, mock_predict):
        # Arrange
        mock_predict.return_value = ['happy']  # Mocked return value
        test_input = "I am so glad today!"

        # Act
        result = predict_emotions(test_input)

        # Assert
        self.assertEqual(result, 'happy')
        mock_predict.assert_called_once_with([test_input])

    @patch('app.pipe_lr.predict_proba')  # Mocking the probability prediction function
    def test_get_prediction_proba(self, mock_predict_proba):
        # Arrange
        mock_predict_proba.return_value = [[0.1, 0.9, 0.0]]  # Mocked probabilities
        test_input = "I am so glad today!"

        # Act
        result = get_prediction_proba(test_input)

        # Assert
        self.assertEqual(result[0][1], 0.9)  # Check if it returns the expected probability
        mock_predict_proba.assert_called_once_with([test_input])

    def test_index_get(self):
        # Act
        response = self.app.get('/')

        # Assert
        self.assertEqual(response.status_code, 200)  # Check if the status code is 200
        self.assertIn(b'raw_text', response.data)  # Check if raw_text is in the response

    @patch('app.pipe_lr.predict')  # Mocking for the POST request
    @patch('app.pipe_lr.predict_proba')  # Mocking the probability prediction function
    def test_index_post(self, mock_predict_proba, mock_predict):
        # Arrange
        mock_predict.return_value = 'happy'
        mock_predict_proba.return_value = [[0.1, 0.9, 0.0]]  # Mocked probabilities
        
        # Act
        response = self.app.post('/', data={'raw_text': 'I am so glad today!'})

        # Assert
        self.assertEqual(response.status_code, 200)  # Check if the status code is 200
        self.assertIn(b'happy', response.data)  # Check if the predicted emotion is in the response

    def test_predict_endpoint(self):
        # Act
        response = self.app.post('/predict', json={'text': 'I am so glad today!'})

        # Assert
        self.assertEqual(response.status_code, 200)  # Check if the status code is 200
        json_data = response.get_json()
        self.assertIn('predicted_emotion', json_data)  # Check if predicted_emotion is in the response

if __name__ == '__main__':
    unittest.main()
