
import pandas as pd
import geopandas as gpd
import plotly.express as px
import json


#Read in the data with pandas
DATA = '../data/all_country_temp_data_CLEAN.csv'
df = pd.read_csv(DATA)

#Read in the shapefile with geopandas
SHAPEFILE = '../data/ne_110m_admin_0_countries.shp'
gdf = gpd.read_file(SHAPEFILE)[['ADMIN', 'geometry']]

# groupby country and year
df = df.groupby(['country','year'])[['monthly_anomaly']].mean().reset_index()

# merge geodataframa with dataframe
gdf = gdf.merge(df, left_on='ADMIN', right_on='country')
gdf.drop('ADMIN', axis=1, inplace=True)

# turn it into a geojson 
gdf.to_json()

# generate a GEOJSON string for a single year
gdf_2000 = gdf[gdf['year'] == 2000]
json_2000 = gdf_2000.to_json()

# convert the GeoJson string to an actual python dict 
# (plotly requires the GeoJSON data in python to be an actual dictionary)
# use the built-in json library to convert it
json_2000 = json.loads(json_2000)

# generate an interactive choropleth map of the data for a single year

'''
fig = px.choropleth(
    data_frame = gdf,               # use the year 2000 data
    geojson=json_2000,    
    locations='country',                # name of the dataframe column that contains country names
    color='monthly_anomaly',                    # name of the dataframe column that contains numerical data you want to display
    locationmode='country names',    # leave this as default
    scope='world',                   # change this to world, usa, ... 
    color_continuous_scale="thermal",
    range_color=(0.1, 1.1),
    color_continuous_midpoint=0.5
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
'''
#fig.show()

#fig.write_html("2000_map.html", include_plotlyjs='cdn')



fig2 = px.choropleth_mapbox(
    mapbox_style='open-street-map',
    data_frame = gdf,                  # dataframe that contains all years
    geojson=json_2000,                   # we can still use the JSON data from 2000, assuming the countries are the same over time
    featureidkey='properties.country',      # name of JSON key within the "properties" value that contains country names
    locations='country',                    # name of the dataframe column that contains country names
    color='monthly_anomaly',                       # name of the dataframe column that contains numerical data you want to display
    center={"lat": 51.1657, "lon": 10.4515},
    zoom=None,
    animation_frame= 'year',             # name of dataframe column that you want to make frames of
    animation_group='country',   
    color_continuous_scale="thermal",
    range_color=(-0.5, 1.5),
    color_continuous_midpoint=1
    )

fig2.write_html("all_years_interactive.html", include_plotlyjs='cdn')
#this could take up to minute to generate -- file is very LARGE!