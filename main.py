# from typing import Union

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


#=============================================

# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle

# # laod model
# rfr = pickle.load(open('rfr.pkl','rb'))
# x_train = pd.read_csv('X_train.csv')

# def pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp):
#     features = np.array([[Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp]])
#     prediction = rfr.predict(features).reshape(1,-1)
#     return prediction[0]


# # web app
# # Gender Age Height Weight Duration Heart_Rate Body_Temp
# st.title("Calories Burn Prediction")

# Gender = st.selectbox('Gender', x_train['Gender'])
# Age = st.selectbox('Age', x_train['Age'])
# Height = st.selectbox('Height', x_train['Height'])
# Weight = st.selectbox('Weight', x_train['Weight'])
# Duration = st.selectbox('Duration (minutes)', x_train['Duration'])
# Heart_rate = st.selectbox('Heart Rate (bpm)', x_train['Heart_Rate'])
# Body_temp = st.selectbox('Body Temperature', x_train['Body_Temp'])

# result = pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp)

# if st.button('predict'):
#     if result:
#         st.write("You have consumed this calories :",result)


#=============================

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
   allow_origins=["*"],  # Update this with the origin of your React Native app
     #allow_origins=["http://192.168.1.102:8081"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
rfr = pickle.load(open('rfr.pkl', 'rb'))
x_train = pd.read_csv('X_train.csv')


# MongoDB Connection
client = AsyncIOMotorClient("mongodb+srv://sumeraruwen:1234@signup.p1rds2b.mongodb.net/?retryWrites=true&w=majority")
db = client["Signup"]  # Replace "your-database-name" with your actual database name
collection = db["UserData"]  # Replace "your-collection-name" with your actual collection name
collection2 = db["Login"]  # Replace "your-collection-name" with your actual collection name


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class InputData(BaseModel):
    Gender: int
    Age: int
    Height: int
    Weight: int
    Duration: int
    Heart_rate: int
    Body_temp: int

@app.post("/predict_calories")
def predict_calories(data: InputData):
    
    print("Received request for predicting calories.")
    try:
         
        features = np.array([[data.Gender, data.Age, data.Height, data.Weight, data.Duration, data.Heart_rate, data.Body_temp]])
        prediction = rfr.predict(features).reshape(1, -1)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
       

class UserData(BaseModel):
    name: str
    email: str
    phone: str
    password: str

class Login(BaseModel):
    email: str
    password: str    

@app.post("/users/")
async def create_user(user_data: UserData):
    try:
        user_dict = user_data.dict()

        # Hash the password before storing it in the database
        user_dict["password"] = pwd_context.hash(user_dict["password"])

        result = await collection.insert_one(user_dict)
        return {"id": str(result.inserted_id)}
    except Exception as e:
        error_detail = f"Error creating user: {str(e)}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)
        

# Sign-in endpoint
@app.post("/signin/")
async def sign_in(login_data: Login):
    try:
        user = await collection.find_one({"email": login_data.email})

        if user and pwd_context.verify(login_data.password, user["password"]):
            return {"message": "Sign-in successful", "user_id": str(user["_id"])}
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        print(f"Error during sign-in: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")




