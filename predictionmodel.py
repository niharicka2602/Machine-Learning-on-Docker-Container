import joblib

model = joblib.load("salarymodel.pkl")
print(model.predict([[7.5]])) #here inside predict(), we can add value of choice
