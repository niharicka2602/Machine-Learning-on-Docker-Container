import panadas
import joblib
from sklearn.linear_model import LinearRegression

db = pandas.read_csv("SalaryData.csv")
a = db["YearsExperience"].values.reshape(30,1)
b = db["Salary"]

model = LinearRegression()
model.fit(a,b)

joblib.dump(model, "salarymodel.pkl") #dumping model into .pkl file
                     
