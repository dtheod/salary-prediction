import requests

prediction = requests.post(
    "http://127.0.0.1:5000/predict",
    headers={"content-type": "application/json"},
    data='{"age":"25-34","industry":"education (higher education)","job": "program manager","country": "united states","years_field_experience":"6","education":"14","gender": "Man","senior":"1","principal":"0","staff":"0","assistant":"0","intern":"0"}',
).text

print(prediction)
