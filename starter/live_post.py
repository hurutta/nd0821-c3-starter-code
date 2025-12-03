import requests


def test_predict_low_income():

    base_url = "https://nd0821-c3-c11v.onrender.com"
    endpoint = "/predict"
    url = f"{base_url}{endpoint}"

    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    try:
        response = requests.post(url, json=payload)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    test_predict_low_income()