# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
import numpy as np
import json

# Create a dash application
app = dash.Dash(__name__)

# Get the data
df = pd.read_csv('real_estate_data.csv')

# Clean the data and add a more useful characteristic for zip codes
df = df.dropna()
df.index = np.arange(df.shape[0])


df['ZipCode'] = df['ZipCode'].str.slice(-5)
ZipPrice = df.groupby(by='ZipCode').mean()['Price']
temp = pd.Series(ZipPrice[df['ZipCode']])
temp.index = np.arange(temp.shape[0])

df.drop(columns='Unnamed: 0', inplace=True)

df['ZipAvg'] = temp


# Prepare data for training a model
a = df.drop(columns=['Price', 'ZipCode', 'LotSize'])
X = a
a_mean = a.mean()
X = X / a_mean
y = df['Price']

# Train a linear regression model (this model has a 80% accuracy)
model = KNeighborsRegressor(n_neighbors=10).fit(X, y)
print(model.score(X, y))
present_zips = list(df['ZipCode'].unique())

# Get a reasonable part of the dataset, as a lot of the values above this can be considered outliers
df = df[df['Price'] < 5E6]

zip_to_avg = df.groupby('ZipCode').mean()['ZipAvg']

map_data = df.groupby('ZipCode').mean()['ZipAvg']
map_data = map_data.reset_index()

map_data['zip_color'] = map_data['ZipAvg'].apply(np.log2)

max_val = map_data['zip_color'].max()
min_val = map_data['zip_color'].min()

map_data['zip_color'] = map_data['zip_color'] - min_val

map_data['ZipAvg'] = map_data['ZipAvg'].astype('int32')


values = [i for i in range(int(min_val), int(max_val))]
ticks = [2**i for i in values]

print(map_data.head())

# Get the GeoJSON file
with open('LA_County_ZIP_Codes.geojson') as f:
    geo_zips = json.load(f)
    
zip_geo = []

for zip in geo_zips['features']:
    
    zip_code = zip['properties']['ZIPCODE']
    
    if zip_code in present_zips:
        
        geometry = zip['geometry']
        
        zip_geo.append({
            'type': 'Feature',
            'geometry': geometry,
            'id': zip_code
                       })

print(len(zip_geo))
print(len(present_zips))

zip_geo_ok = {'type': 'FeatureCollection', 'features': zip_geo}

# Application layout

                                
map_fig= px.choropleth_mapbox(map_data,
                                    geojson=zip_geo_ok,
                                    locations='ZipCode',
                                    color='zip_color',
                                    range_color=(0, map_data['zip_color'].max()),
                                    mapbox_style="carto-positron",
                                    zoom=8, center = {"lat": 34.0522, "lon": -118.2437},
                                    opacity=0.5,
                                    hover_data={'zip_color': False, 'ZipCode': True, 'ZipAvg': True},
                                    labels={'ZipAvg': 'Average House Price'}
                                    )
map_fig.update_layout( margin={'r':0,'t':0,'l':0,'b':0},
                coloraxis_colorbar={
                     'title':'Price',
                     'tickvals':values,
                     'ticktext':ticks})


app.layout = html.Div(children=[ 
                                html.H1(['House Prices in LA City'], style={'color': '#503D36', 'font-size': 24, 'textAlign': 'center'}),
                                html.Br(),
                                dcc.Graph(figure=map_fig, id='map_plot', style={'width': '80%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '8px 8px 8px grey',
                                 'background-color': '#f9f9f9',
                                 'padding': '10px',
                                 'margin-bottom': '10px'

                                 }),
                                dcc.Graph(figure=px.histogram(df['Price'], x='Price'), id='price_hist', style={'width': '80%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '8px 8px 8px grey',
                                 'background-color': '#f9f9f9',
                                 'padding': '10px',
                                 'margin-bottom': '10px'

                                 }),
                                html.Div(children=[
                                    dcc.Graph(figure=px.scatter(df[df['HouseSize']<2E4], x='HouseSize', y='Price'), id='size_price_plot', style={'width':'50%'}),
                                    dcc.Graph(figure=px.histogram(df[df['HouseSize']<2E4], x='HouseSize'), id='size_hist', style={'width':'50%'})
                                ], style={'width': '80%', 'display': 'flex',
                                 'border-radius': '15px',
                                 'box-shadow': '8px 8px 8px grey',
                                 'background-color': '#f9f9f9',
                                 'padding': '10px',
                                 'margin-bottom': '10px'
                            
                                 }),
                                html.Div([
                                    html.Div([
                                        dcc.Input(
                                            id='my_size',
                                            type='number',
                                            placeholder='House Size',
                                            minLength=0,
                                            maxLength=8,
                                            size='10',
                                            readOnly=False,
                                            required=True,
                                            disabled=False,
                                            style={'width':'80px', 'height':'80px', 'margin':'5px 20px 10px 20px', 'font-size':'large', 'justifyContent':'center', 'textAlign': 'center'}
                                        ),
                                                                                
                                        dcc.Input(
                                            id='my_bed',
                                            type='number',
                                            placeholder='Bedrooms',
                                            minLength=0,
                                            maxLength=8,
                                            size='10',
                                            readOnly=False,
                                            required=True,
                                            disabled=False,
                                            style={'width':'80px', 'height':'80px', 'margin':'5px 20px 10px 20px', 'font-size':'large', 'justifyContent':'center', 'textAlign': 'center'}
                                        ),
                                       dcc.Input(
                                            id='my_bath',
                                            type='number',
                                            placeholder='Bathrooms',
                                            minLength=0,
                                            maxLength=8,
                                            size='10',
                                            readOnly=False,
                                            required=True,
                                            disabled=False,
                                            style={'width':'80px', 'height':'80px', 'margin':'5px 20px 10px 20px', 'font-size':'large', 'justifyContent':'center', 'textAlign': 'center'}
                                        ),
                                      dcc.Input(
                                            id='my_zip',
                                            type='number',
                                            placeholder='Zip Code',
                                            minLength=0,
                                            maxLength=8,
                                            size='10',
                                            readOnly=False,
                                            required=True,
                                            disabled=False,
                                            style={'width':'80px', 'height':'80px', 'margin':'5px 20px 10px 20px', 'font-size':'large', 'justifyContent':'center', 'textAlign': 'center'}
                                        ),
                                    
                                    html.Button(id='submit_button', n_clicks=0, children='Submit',
                                            style={'width':'80px', 'height':'80px', 'margin':'5px 20px 10px 20px', 'background-color':'white'})]),
                                            
                                    html.Div(children=[], id='predicted_price',
                                            style={'width':'560px', 'height':'100px', 'margin':'5px 20px 10px 20px', 'font-size':'xx-large', 'textAlign':'center'}),
                                
                                    ], style={'width': '80%', 'display': 'inline-block',
                                    'border-radius': '15px',
                                    'box-shadow': '8px 8px 8px grey',
                                    'background-color': '#f9f9f9',
                                    'padding': '10px',
                                    'margin-bottom': '10px',
                                    'textAlign':'center'

                                 }),
                                
                                
                                
                            ])

# Callback function definition
@app.callback( [Output(component_id='predicted_price', component_property='children')],
               [Input(component_id='submit_button', component_property='n_clicks')],
               [State(component_id='my_size', component_property='value'),
                State(component_id='my_bed', component_property='value'),
                State(component_id='my_bath', component_property='value'),
                State(component_id='my_zip', component_property='value')],)
                

def fun_callback(n_clicks, size, beds, baths, zip):
    
    predicted_price = dcc.Textarea()
    
    if n_clicks > 0:
        if size != None and beds != None and baths != None and zip != None:
        
            sample = np.array([float(beds), float(baths), float(size), float(zip_to_avg[str(zip)])])
            sample /= a_mean
            sample = np.reshape(np.array(sample), newshape=(1, -1))
            predicted = str(model.predict(sample)[0])
            print(predicted)
            predicted_price = dcc.Textarea(value=predicted, style={'width':'560px', 'height':'100px', 'margin':'5px 20px 10px 20px', 'font-size':'xxx-large', 'textAlign':'center'})
            
        else:
            raise PreventUpdate
            
    return predicted_price,
# Run the app
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050, debug=True)
