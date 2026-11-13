
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union
from pathlib import Path
import pandas as pd
import joblib


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    description="Predict whether a hotel booking will be cancelled",
    version="1.0.0"
)


# --------------------------------------------------
# 2. Load trained ML model
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).parent /
    "hotel_booking_cancellation_model.joblib"
)

try:
    model_package = joblib.load(MODEL_PATH)

    model = model_package["model"]
    threshold = model_package["threshold"]
    model_name = model_package["model_name"]
    feature_names = model_package["feature_names"]

except Exception as e:
    raise RuntimeError(f"Unable to load model: {e}")


# --------------------------------------------------
# 3. Define input schema using Pydantic
# --------------------------------------------------

class BookingInput(BaseModel):

    hotel: str
    lead_time: int
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int

    stays_in_weekend_nights: int
    stays_in_week_nights: int

    adults: int
    children: Optional[float] = None
    babies: int

    meal: str
    country: Optional[str] = None

    market_segment: str
    distribution_channel: str

    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int

    reserved_room_type: str
    booking_changes: int
    deposit_type: str

    agent: Optional[Union[str, int]] = None
    company: Optional[Union[str, int]] = None

    days_in_waiting_list: int
    customer_type: str
    adr: float

    required_car_parking_spaces: int
    total_of_special_requests: int

    class Config:
        extra = "forbid"


# --------------------------------------------------
# 4. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Hotel Booking Cancellation API is running",
        "model": model_name
    }


# --------------------------------------------------
# 5. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_booking(data: BookingInput):

    try:

        # Convert Pydantic input into dictionary
        booking_data = data.dict()

        # Convert dictionary into DataFrame
        input_df = pd.DataFrame(
            [booking_data],
            columns=feature_names
        )

        # Convert agent and company into categorical strings
        for col in ["agent", "company"]:

            if pd.notna(input_df[col].iloc[0]):
                input_df[col] = input_df[col].astype(str)

        # Generate prediction probabilities
        probabilities = model.predict_proba(input_df)[0]

        # Identify probability corresponding to class 1
        class_1_index = list(model.classes_).index(1)

        cancellation_probability = float(
            probabilities[class_1_index]
        )

        # Apply saved classification threshold
        prediction = int(
            cancellation_probability >= threshold
        )

        # Generate readable result
        result = (
            "Canceled"
            if prediction == 1
            else "Not Canceled"
        )

        # Return API response
        return {
            "prediction": prediction,
            "result": result,
            "cancellation_probability": round(
                cancellation_probability * 100, 2
            ),
            "model": model_name,
            "threshold": threshold
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
