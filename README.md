End To End ML Project created a environment conda create -p venv python==3.8

conda activate venv/ Install all necessary libraries pip install -r requirements.txt


To run flask application 

```
python app.py
```


To access your flask application open new tab in and paste the url:
```
https:127.0.0.1:5000/
```

# to Build the Docker Image
docker build -t stonepriceprediction .