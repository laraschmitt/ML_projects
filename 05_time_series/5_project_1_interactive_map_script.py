import pandas as pd
import geopandas as gpd
import plotly.express as px
import json


df = pd.read_csv('data/berlin_temp_clean/berlin_all_districts_temp.csv', index_col=0)

# remove whitespaces in df
df = df.rename(columns=lambda x: x.strip())

# convert the temperature into unit of 1°C 
df['TG'] = df['TG']/10

df['datetime'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
df['year'] = pd.DatetimeIndex(df['datetime']).year
df['month'] = pd.DatetimeIndex(df['datetime']).month


df['join_col'] = df['Station_name'].str.replace('BERLIN-','').str.capitalize()

df['join_col'] = df['join_col'].str.replace('Tegel', 'Reinickendorf')

df['join_col'] = df['join_col'].str.replace('Ostkreuz', 'Friedrichshain-Kreuzberg')

df['join_col'] = df['join_col'].str.replace('Marzahn', 'Marzahn-Hellersdorf')

df['join_col'] = df['join_col'].str.replace('Zehlendorf', 'Steglitz-Zehlendorf')

df['join_col'] = df['join_col'].str.replace('Rudow', 'Neukölln')

df['join_col'] = df['join_col'].str.replace('Tempelhof', 'Tempelhof-Schöneberg')
df['join_col'] = df['join_col'].str.replace('Treptow', 'Treptow-Köpenick')
df['join_col'] = df['join_col'].str.replace('Ostkreuz', 'Lichtenberg')
df['join_col'] = df['join_col'].str.replace('Invalidenstrasse', 'Pankow')

#Read in the shapefile with geopandas
SHAPEFILE = 'data/shapefile.shp/lor_ortsteile.shp'
gdf = gpd.read_file(SHAPEFILE)[['BEZIRK', 'geometry']]

# groupby Station and year
df = df.groupby(['join_col','year'])[['TG']].mean().reset_index()

#  merge geodataframa with dataframe
gdf = gdf.merge(df, left_on='BEZIRK', right_on='join_col')
gdf.drop('join_col', axis=1, inplace=True)


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


fig = px.choropleth(df, geojson=json_2000, color="TG",
                    locations="join_col", featureidkey="properties.BEZIRK",
                    projection="mercator", color_continuous_scale="thermal",
    range_color=(10, 12)
                   )
#fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

"""
fig = px.choropleth(
    data_frame = gdf,               # use the year 2000 data
    geojson=json_2000,    
    locations='BEZIRK',                # name of the dataframe column that contains country names
    color='TG',                    # name of the dataframe column that contains numerical data you want to display
    locationmode='country names',    # leave this as default
    #scope='europe',                   # change this to world, usa, ... 
    color_continuous_scale="thermal",
    range_color=(0.1, 0.3),
    color_continuous_midpoint=0.5
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
"""
#fig.show()

fig.write_html("berlin_temp_2000_map.html", include_plotlyjs='cdn')

