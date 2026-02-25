# Hotel_Reservations_Project

This project analyzes a real hotel booking dataset and builds Machine Learning models capable of predicting whether a booking will be canceled or not. Ten classical classification algorithms are tested and compared, along with an Artificial Neural Network (ANN) built with Keras, with the goal of identifying the best-performing model for this problem.

## Business Problem

The hospitality industry faces a critical issue: a significant percentage of bookings are canceled, often at the last minute. This generates:

- Direct revenue losses — empty rooms cannot be resold quickly
- Operational planning difficulties — staff, housekeeping, and catering are miscalculated
- Uncontrolled overbooking
- Suboptimal pricing — without a clear prediction of real demand, managers cannot dynamically adjust rates

The central question of the project is:

**"Can we identify high-risk bookings early enough so the hotel can take preventive action?"**

## Business Impact

A cancellation prediction model has direct applicability in hotel operations:

- Revenue Management – Adjusting prices based on cancellation probability
- Strategic Overbooking – Accepting extra bookings only for high-risk segments
- Customer Retention – Sending personalized offers to customers with high cancellation risk
- Operations – More accurate planning of human and logistical resources
- Marketing – Targeting loyal customers (repeated guests) with loyalty programs

## Project Overview

- Dataset: Hotel Reservations.csv
- Target: booking_status → Canceled / Not_Canceled
- Problem Type: Supervised binary classification
- Models tested: 10 classical ML algorithms + ANN (Keras)
- Evaluation metrics: MAE, Accuracy, Classification Report, Confusion Matrix

## Key Steps
### 1. Data Preprocessing

The raw data was cleaned and prepared for modeling:

- Removed the Booking_ID column (unique identifier with no predictive value)
- Checked for missing values — the dataset contains no null values
- Encoded categorical variables (type_of_meal_plan, room_type_reserved, market_segment_type) using One-Hot Encoding (get_dummies)
- Label encoding of the target: Not_Canceled → 0, Canceled → 1
- Split: 80% train / 20% test (random_state=42)
- Scaling with MinMaxScaler on all numerical variables to eliminate magnitude differences

### 2. Exploratory Data Analysis (EDA)

Visualizations were generated for each numerical variable in relation to booking_status, including:

- Boxplots (to identify distribution and outliers)
- Histograms (to observe overall distribution)

## Factors Influencing Booking Cancellations

Based on the boxplots generated during EDA, the following factors show visible differences between the Canceled and Not_Canceled groups:

### 1. lead_time — Booking anticipation time
The strongest visible factor in the charts.
The Canceled group has nearly double the median compared to Not_Canceled (~120 vs ~50 days), and the entire IQR is significantly shifted toward higher values.
The earlier the booking is made, the higher the cancellation risk.

### 2. no_of_previous_bookings_not_canceled — History of completed bookings
The Not_Canceled group contains nearly all customers with a rich booking history (outliers up to 60), while the Canceled group almost exclusively has values of 0.
Customers with a strong history of honored bookings are significantly less likely to cancel.

### 3. avg_price_per_room — Average room price
The Canceled group shows a slightly higher median and IQR compared to Not_Canceled.
Higher-priced bookings show a stronger tendency toward cancellation.

### 4. no_of_weekend_nights — Number of weekend nights
The difference is subtle but visible: the Canceled group has a wider IQR (0–2) compared to Not_Canceled (0–1), suggesting that longer weekend stays are slightly associated with higher cancellation rates.

Note: The variables no_of_adults, no_of_children, required_car_parking_space, arrival_month, arrival_date, arrival_year, no_of_week_nights, repeated_guest, no_of_previous_cancellations, and no_of_special_requests do not show visible differences between the two groups in the boxplots.

### 3. Modeling

The following classification models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Decision Tree
- Extra Trees
- AdaBoost
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- ANN (Keras)

All models were trained on the same train/test split, and performance was evaluated comparatively using MAE (Mean Absolute Error) — the lower the MAE, the fewer classification errors the model makes.

## 4. Hyperparameter Tuning

For the Artificial Neural Network (ANN), the following hyperparameters were configured:

- Architecture: Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Batch size: 64
- Max epochs: 100 (with EarlyStopping on val_loss, patience=10)
- Validation split: 20% of training data

The evolution of loss and accuracy for train vs. validation was visualized to detect overfitting.

### 5. Predictions

After training, each model generated predictions on the test set.
A centralized comparison matrix was created containing the MAE of each model to identify the best-performing algorithm.
For the ANN, a detailed Confusion Matrix was also generated, showing the distribution of correct and incorrect classifications: True Positive, True Negative, False Positive and False Negative.

##  Business Insights
High-Risk Customer Profile

Strictly based on the EDA visualizations, the typical profile of a customer who will cancel looks like this:

- Booked far in advance — the clearest signal across all charts, with evident separation between groups
- Is a new customer, with no history of completed bookings (no_of_previous_bookings_not_canceled = 0) — almost all loyal customers (with rich booking history) appear exclusively in the Not_Canceled group
- The booking is for a higher-priced room — the Canceled group has a slightly higher median price
- The stay includes more weekend nights — the Canceled group distribution is slightly shifted toward higher values

## Applicable Improvements

- Lead-time-based guarantee policies: Bookings made more than 90 days in advance represent a real-risk segment — requiring a deposit or applying a non-refundable rate for this segment could reduce losses without affecting last-minute bookings.
- Loyal customers are the most stable asset: Data clearly shows that a history of honored bookings is one of the best predictors of non-cancellation — loyalty programs are not just marketing tools; they directly impact cancellation rates.
- Premium rooms with long lead time: The combination of high price + early booking concentrates the highest risk. These bookings can be monitored separately and treated with reconfirmation policies.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Keras / TensorFlow
- tqdm
