# Hotel_Reservations_Project
This project aims to **predict hotel booking cancellations** using machine learning and deep learning techniques. The dataset contains information about hotel reservations, including the number of adults and children, room type, meal plan, lead time, and other relevant features.

The project includes:
- Exploratory Data Analysis (EDA) for understanding and visualizing the data.
- Data preprocessing: categorical encoding and numerical scaling.
- Application of multiple classification models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Decision Tree
  - Extra Trees
  - AdaBoost
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
- Building an Artificial Neural Network (ANN) using Keras.

## Dataset
The dataset contains 36,275 entries and 19 columns:
- `Booking_ID` – Reservation ID
- `no_of_adults`, `no_of_children` – Number of adults and children
- `no_of_weekend_nights`, `no_of_week_nights` – Stay duration
- `type_of_meal_plan` – Type of meal plan
- `required_car_parking_space` – Parking space requirement
- `room_type_reserved` – Reserved room type
- `lead_time` – Days in advance the booking was made
- `arrival_year`, `arrival_month`, `arrival_date` – Arrival date
- `market_segment_type` – Market segment
- `repeated_guest` – Indicator if the guest is a repeat customer
- `no_of_previous_cancellations`, `no_of_previous_bookings_not_canceled` – Previous booking history
- `avg_price_per_room` – Average room price
- `no_of_special_requests` – Number of special requests
- `booking_status` – Reservation status (`Canceled` or `Not_Canceled`)

## Technologies & Libraries
- Python 
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost, LightGBM
- Keras, TensorFlow
- tqdm

