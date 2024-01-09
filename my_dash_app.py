import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


df1 = pd.read_csv("homeprices.csv")

#Removing null values 
df2 = df1.dropna()

#Cleaning data 
df4 = df2.drop("Unnamed: 0", axis=1, errors='ignore')
df4['Price ($)'] = df4['Price ($)'].astype(int)
df4 = df4[df4['Price ($)'] >= 500000]

# split data 80:20 for training 
train_data, test_data = train_test_split(df4, test_size=0.2, random_state=42)

# Select features and target variable for training (Linear Regression)
X_train = train_data[['lat', 'lng']]
y_train = train_data['Price ($)']

# Select features and target variable for testing (Linear Regression)
X_test = test_data[['lat', 'lng']]
y_test = test_data['Price ($)']

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Select features and target variable for training (Decision Tree)
X_train_dt = train_data[['lat', 'lng']]
y_train_dt = train_data['Price ($)']

# Select features and target variable for testing (Decision Tree)
X_test_dt = test_data[['lat', 'lng']]
y_test_dt = test_data['Price ($)']

# Train Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_dt, y_train_dt)

# Select features and target variable for training (Random Forest)
X_train_rf = train_data[['lat', 'lng']]
y_train_rf = train_data['Price ($)']

# Select features and target variable for testing (Random Forest)
X_test_rf = test_data[['lat', 'lng']]
y_test_rf = test_data['Price ($)']

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Create a Dash app
app = dash.Dash(__name__,title="Ontario Housing Price Predictor")
server = app.server

# Define the layout of the app
app.layout = html.Div(style={
    'textAlign': 'center',
    'backgroundImage': 'linear-gradient(to bottom, #333333, #ffffff)',
    'padding': '50px'
    
}, children=[
    html.H1("Ontario Housing Price Predictor", style={'textDecoration': 'underline'}),

    html.Label("Enter Latitude:"),
    dcc.Input(id='input-lat', type='number', value=0),

    html.Label(" Enter Longitude:"),
    dcc.Input(id='input-lon', type='number', value=0),

    html.Button('Predict Prices', id='predict-button'),

    html.Div(id='output-container')
])



# Define callback to update the output based on user input
@app.callback(
    Output('output-container', 'children'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('input-lat', 'value'),
     dash.dependencies.State('input-lon', 'value')]
)
def update_output(n_clicks, input_lat, input_lon):
    # Check if both latitude and longitude are zero
    if input_lat == 0 and input_lon == 0:
        return ''  # Return an empty string if both are zero

    # Create a DataFrame with user input
    user_input = pd.DataFrame({'lat': [input_lat], 'lng': [input_lon]})
    
    # Make predictions using the trained models
    predicted_price_lr = model.predict(user_input)[0]
    predicted_price_dt = dt_model.predict(user_input)[0]
    predicted_price_rf = rf_model.predict(user_input)[0]

    # Format predicted prices with commas for thousands
    formatted_lr = f"${predicted_price_lr:,.2f}"
    formatted_dt = f"${predicted_price_dt:,.2f}"
    formatted_rf = f"${predicted_price_rf:,.2f}"

    # Display formatted predicted prices with enhanced formatting
    output_text = html.Div([
        html.P(f"Linear Regression Predicted Price: {formatted_lr}", style={'fontWeight': 'bold'}),
        html.P(f"Decision Tree Predicted Price: {formatted_dt}", style={'fontWeight': 'bold'}),
        html.P(f"Random Forest Predicted Price: {formatted_rf}", style={'fontWeight': 'bold'}),
    ], style={'margin': '20px'})

    return output_text



# Run the app in the external mode if it's the main script
if __name__ == '__main__':
    app.run_server(debug=True)
