# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:13:51 2021

@author: emanu
"""
#importing libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import dash_table as dt
import base64
import flask
import glob
import os

#External stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.read_csv('raw_data_all_cleaned.csv')    #loading data         

df['Date'] = pd.to_datetime(df['Date'])    #set date as datetime               
df['Year']=df['Date'].dt.year              #put year into the table
df['Day']=df['Date'].dt.day               #put day into the table

#rename columns
df.rename(columns={'Power [kW]':'Power','Temperature [°C]':'Temperature','Pressure [mbar]':'Pressure','Solar Radiation [W/m2]':'G','Week Day':'WeekDay','Rain Day':'RainDay'},inplace=True)

#useful command for the dropdown menu
available_years = df['Year'].unique()
available_months = df['Month'].unique()
available_day = df['Day'].unique()
available_hour = df['Hours'].unique()

#generate table
# def generate_table(dataframe, max_rows=31):                            #function used to generate table
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#             ]) for i in range(min(len(dataframe), max_rows))
#         ])
#     ])




#TO BE USED FOR ALL THE IMAGES
# image_Cluster1 = 'assets/Cluster1.png' # replace with your own image
# image_Cluster2 = 'assets/Cluster2.png'
# image_Cluster3 = 'assets/Cluster3.png'
# image_Cluster4 = 'assets/Cluster4.png'
# image_3dclust = 'assets/3d_clustering.png'
# image_powerd = 'assets/PowerDensity.png'
# image_Tempd = 'assets/TemperatureDensity.png'
# image_dpattern= 'assets/Daily_Pattern.png'
# image_fregr='assets/F_regression.png'
# image_mireg='assets/m_i_regression.png'
# image_ranfr='assets/Random_forest.png'
# image_Bootstr='assets/Bootstr.png'
# image_DecisionTree='assets/DecisionTree.png'
# image_EGB='assets/EGB.png'
# image_GB='assets/GB.png'
# image_LR='assets/LR.png'
# image_Neural='assets/Neural.png'
# image_RF_regression='assets/RF.png'
# image_SVRlinear='assets/SVRlinear.png'
# image_SVRr='assets/SVRr.png'


# encoded_image_c1 = base64.b64encode(open(image_Cluster1, 'rb').read())
# encoded_image_c2 = base64.b64encode(open(image_Cluster2, 'rb').read())
# encoded_image_c3 = base64.b64encode(open(image_Cluster3, 'rb').read())
# encoded_image_c4 = base64.b64encode(open(image_Cluster4, 'rb').read())
# encoded_image_c3d = base64.b64encode(open(image_3dclust, 'rb').read())
# encoded_image_powerd = base64.b64encode(open(image_powerd, 'rb').read())
# encoded_image_tempd = base64.b64encode(open(image_Tempd, 'rb').read())
# encoded_image_dailyp = base64.b64encode(open(image_dpattern, 'rb').read())
# encoded_image_fregr = base64.b64encode(open(image_fregr, 'rb').read())
# encoded_image_mireg = base64.b64encode(open(image_mireg, 'rb').read())
# encoded_image_ranfr = base64.b64encode(open(image_ranfr, 'rb').read())
# encoded_image_Bootstr= base64.b64encode(open(image_Bootstr, 'rb').read())
# encoded_image_DecisionTree=base64.b64encode(open(image_DecisionTree, 'rb').read())
# encoded_image_EGB=base64.b64encode(open(image_EGB, 'rb').read())
# encoded_image_GB=base64.b64encode(open(image_GB, 'rb').read())
# encoded_image_LR=base64.b64encode(open(image_LR, 'rb').read())
# encoded_image_Neural=base64.b64encode(open(image_Neural, 'rb').read())
# encoded_image_RF_regression=base64.b64encode(open(image_RF_regression, 'rb').read())
# encoded_image_SVRlinear=base64.b64encode(open(image_SVRlinear, 'rb').read())
# encoded_image_image_SVRr=base64.b64encode(open(image_SVRr, 'rb').read())

#C:\Users\emanu\Desktop\Polito Magistrale\Secondo Anno\Secondo Semestre\Energy services (Erasmus)\Project2\assets

image_directory_clustering = '/assets/Clustering/'
list_of_images_clustering = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory_clustering))]
static_image_route_clustering = '/static/'

#static_image_route = '/static/'

image_directory_Feature = '/assets/Feature/'
list_of_images_Feature = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory_Feature))]
static_image_route_Feature = '/static/'

image_directory_Regression = '/assets/Regression/'
list_of_images_Regression = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory_Regression))]
static_image_route_Regression = '/static/'

app.layout = html.Div([
                                      #main division
    html.H1('Civil Building Monitor'),    #it's an HTML
        html.Img(src='assets/IST-1.png'
              ,
            style={
                'height': '10%',
                'width': '10%'
            }
            ),
    html.H6('By Emanuele D Argenzio - ist1100846'),

    dcc.Tabs(id='tabs', value='tab1', children=[                    #Here the value states the starting tab when we open the dashboard
        dcc.Tab(label='Raw data visualization', value='tab1'),
        dcc.Tab(label='Exploratory Data Analysis', value='tab2'),
        dcc.Tab(label='Clustering', value='tab3'),
        dcc.Tab(label='Feature Selection', value='tab4'), #qui dare la possibilità di scegliere diverse features per avere in uscita diversi output
        dcc.Tab(label='Regression model and results', value='tab5'),      #if i want to introduce another tab
    ]), 
        #     style={
        #     'textAlign': 'center',
        #     'color': colors['text']
        # }
    html.Div(id='tabs-content')   
])   

tab1_layout = html.Div([
                        html.H4('In this table, it is possible to navigate inside the cleaned datas and to sort them in ascending and descending order '),
                        html.H5('It is also possible to clean some columns, if you don\' want to see them'),
                        html.H6('To be noticed: cleaned datas means datas obtained after the feature selection'),
    dt.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        #column_selectable="single",
        #row_selectable="multi",
        #row_deletable='false',
        #selected_columns=[],
        #selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 10,
    ),
    html.Div(id='datatable-interactivity-container')
])
      #        html.Div([

      # html.Div(
      # #         ([
      # #     dcc.Dropdown(options=[
      # #         {'label': i, 'value': i} for i in range(5)
      # #     ])
      # # ]),
      # #try to remove this children and put directly the division
      #     children=[
      #     html.Div(children="Year", className="menu-title"),
      #     dcc.Dropdown(
      #         id="year",
      #         options=[
      #             {"label": year, "value": year}
      #             for year in np.sort(available_years)
      #         ],
      #       value="2017",
      #         clearable=False,
      #         className="dropdown",
      #         style={'width': '33%', 'display': 'inline-block'}
      #   ),
    

      # html.Div(children="Month", className="menu-title"),
      #   dcc.Dropdown(
      #         id="month",
      #         options=[
      #             {"label": month, "value": month}
      #             for month in np.sort(available_months)
      #         ],
      #         value="January",
      #         clearable=False,
      #         className="dropdown",
      #         style={'width': '33%', 'display': 'inline-block'}
      #     ),
    

      # html.Div(children="Day", className="menu-title"),
      #     dcc.Dropdown(
      #         id="day",
      #         options=[
      #             {"label": day, "value": day}
      #             for day in np.sort(available_day)
      #         ],
      #         value="1",
      #         clearable=False,
      #         className="dropdown",
      #         style={'width': '33%', 'display': 'inline-block'}
      #     ),
          #className="menu",
      
          #generate_table(df)
          #dt.DataTable('datas'),
# html.Div(html.H2('Table'),
#          dt.DataTable(
#     id='table',
#     columns=[{"name": i, "id": i} for i in df.columns],
#     data=df.to_dict('records')
# )),


# @app.callback(
#       [Output("datas", "table")],
#       [
#           Input("year", "value"),
#           Input("month", "value"),
#           Input("day", "value"),
#       ])
#],),])
         # update dynamic table (y,n,d)
          #return table 
            
#             ],),],)


# html.Div(
#             children=[
#                 html.Div(
#                     children=[generate_table(df)
#                          (id="datas", config={"displayModeBar": False}],
                    
#                     ),
#                 ],
#             )


tab2_layout= html.Div(children=[
                      html.H5('In this section, different graphs show the datas seen in the table of the first section'),
    #children=[
    # dcc.Dropdown(
    #     id='dropdown',
    #     options=[{'label': i, 'value': i} for i in df.columns],
    #     value=df.columns[5]
    # ),
    dcc.Graph(
              id='PowerTemp',
                  figure={
                      'data': [
                          {'x': df['Date'], 'y': df['Power'], 'type': 'line', 'name': 'Power [kW]'},
                          {'x': df['Date'], 'y': df['Temperature'], 'type': 'line', 'name': 'Temperature [°C]'},
                          ],
                      'layout': {
                          'title': 'Power Consumption vs Temperature'
            }
        }
    ),
    dcc.Graph(
              id='PowerG',
                  figure={
                      'data': [
                          {'x': df['Date'], 'y': df['G'], 'type': 'line', 'name': 'Solar Radiation [W/m2]'},
                          {'x': df['Date'], 'y': df['Power'], 'type': 'line', 'name': 'Power [kW]'},
                          
                          ],
                      'layout': {
                          'title': 'Power Consumption vs Irradiance'
            },
        },
    ),
    html.H5('Probability distribution for power'),
    html.H6('To be noticed: those are the cleaned data'),
    dcc.Graph(id="prob"),
    html.P("Distributions"),
    dcc.RadioItems(
        id='dist-marginal',
        options=[{'label': x, 'value': x} 
                 for x in ['box', 'violin', 'rug']],
        value='box'
    ),
    #,
    html.H5('Probability distribution for temperature'),
    html.H6('To be noticed: those are the cleaned data'),
    dcc.Graph(id="prob2"),
    html.P("Distributions"),
    dcc.RadioItems(
        id='dist-marginal2',
        options=[{'label': x, 'value': x} 
                  for x in ['box', 'violin', 'rug']],
        value='box'
    )
# html.Div(
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_powerd.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_tempd.decode()))  
#        ) 
])          

@app.callback(
    Output("prob", "figure"), 
    [Input("dist-marginal", "value")])
def display_graph(marginal):
    fig = px.histogram(
        df['Power'], x="Power",color_discrete_sequence=['red'],
        marginal=marginal)
        #hover_data=df.columns)
    return fig

@app.callback(
    Output("prob2", "figure"), 
[Input("dist-marginal2", "value")])
def display_graph_1(marginal):
    fig2 = px.histogram(
        df['Temperature'], x="Temperature",#color="Temperature",
        marginal=marginal)
        #hover_data=df.columns)
    return fig2

# #inserire boxplot e istrogrammi per la potenza e la temperatura

tab3_layout= html.Div([
                       html.H6('With the dropdown menu, you can select the different clusterings obtained during the data analysis'),
    dcc.Dropdown(
        id='image-dropdown-clustering',
        options=[{'label': i, 'value': i} for i in list_of_images_clustering],
        value=list_of_images_clustering[0]
    ),
    html.Center(html.Img(id='image-clustering'))  
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_c1.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_c2.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_c3.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_c4.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_c3d.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_dailyp.decode()))

 ])
    #inserire le immagini del clustering


tab4_layout= html.Div([
                        html.H6('In this section, the feature selection process is reported. In particular, three different models can be selected. The features reported on the X axes are:'),
                        dcc.Markdown('''  *0   Temperature       
    *1   HR                      
    *2   Pressure         
    *3   Rain Day                
    *4   Holiday                 
    *5   Week Day                
    *6   Hours                   
    *7   Month                     
    *8   Solar Radiation   
    *9   Power-1 

                                  '''),
                        html.H6('As result, it has been decided to use:'),
                        dcc.Markdown('''            
Temperature     
Week Day                
Hours                   
Month                   
Solar Radiation  
Power-1  '''),
  
        dcc.RadioItems(
        id='image-dropdown-feature',
        options=[{'label': i, 'value': i} for i in list_of_images_Feature],
        value=list_of_images_Feature[0]
    ),
    html.Center(html.Img(id='image-feature'))
    
#inserire l'immagine del feature selection con i nomi delle features selezionate
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_fregr.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_mireg.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_ranfr.decode()))    

    ])
                                 
  # 0   Power              
  # 1   Temperature     
  # 2   Week Day                
  # 3   Hours                   
  # 4   Month                   
  # 5   Solar Radiation  
  # 6   Power-1  
                                         
                                         
tab5_layout= html.Div([  #dropdown menu dove , per ogni metodo, ottengo immagine e risultati
                            html.H6('In this section, the regression model used are presented. The selected features are'),
                     dcc.Markdown('''
                                     Temperature
                                     
                                     Week Day
                                     
                                     Hours
                                     
                                     Solar Radiation
                                     
                                     Power-1
                                     '''),
                        html.H6('The results for each model are:'),
                        dcc.Markdown('''
                                     
Linear Regression (LR.png)                        - results: MAE_LR=21.33 MSE_LR=808.51 RMSE_LR=28.43 cvRMSE_LR=0.138

Support Vector Regressor, kernel RBF (SVRr.png)   - results MAE_SVR=11.15 MSE_SVR=255.25 RMSE_SVR=15.97 cvRMSE_SVR=0.077 

Decision tree regressor (DecisionTree.png)        - results: MAE_DT=11.39 MSE_DT=329.86 RMSE_DT=18.16 cvRMSE_DT=0.088

Random Forest (RF.png)                            - results: MAE_RF=8.80 MSE_RF=172.88 RMSE_RF=13.14 cvRMSE_RF=0.0638

Gradient Boosting (GB.png)                        - results: MAE_GB=9.45 MSE_GB=187.57 RMSE_GB=13.69 cvRMSE_GB=0.0665

Extreme Gradient Boosting (EGB.png)               - results: MAE_XGB=8.71 MSE_XGB=170.66 RMSE_XGB=13.06 cvRMSE_XGB=0.0635

Bootstrapping (Bootstr.png)                       - results: MAE_BT=8.97 MSE_BT=196.31 RMSE_BT=14.01 cvRMSE_BT=0.0682

Neural Network (Neural.png)                       - results: MAE_NN=13.58 MSE_NN=380.66 RMSE_NN=19.51 cvRMSE_NN=0.0945

'''),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_Bootstr.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_DecisionTree.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_EGB.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_GB.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_LR.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_Neural.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_RF_regression.decode())),
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_SVRlinear.decode())), 
# html.Img(src='data:image/png;base64,{}'.format(encoded_image_image_SVRr.decode()))

  dcc.Dropdown(
         id='image-dropdown-regression',
         options=[{'label': i, 'value': i} for i in list_of_images_Regression],
         value=list_of_images_Regression[0]
     ),
     html.Center(html.Img(id='image-regression'))    
    ])


@app.callback(
    dash.dependencies.Output('image-clustering', 'src'),
    [dash.dependencies.Input('image-dropdown-clustering', 'value')])
def update_image_src_cl(value):
    return static_image_route_clustering + value
@app.server.route('{}<image_path>.png'.format(static_image_route_clustering))
def serve_image_cl(image_path):
    image_name_clustering = '{}.png'.format(image_path)
    if image_name_clustering not in list_of_images_clustering:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory_clustering, image_name_clustering)

@app.callback(
    dash.dependencies.Output('image-feature', 'src'),
    [dash.dependencies.Input('image-dropdown-feature', 'value')])
def update_image_src_fe(value):
    return static_image_route_Feature + value
@app.server.route('{}<image_path>.png'.format(static_image_route_Feature))
def serve_image_fe(image_path):
    image_name_Feature = '{}.png'.format(image_path)
    if image_name_Feature not in list_of_images_Feature:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory_Feature, image_name_Feature)

@app.callback(
    dash.dependencies.Output('image-regression', 'src'),
    [dash.dependencies.Input('image-dropdown-regression', 'value')])
def update_image_src_re(value):
    return static_image_route_Regression + value
@app.server.route('{}<image_path>.png'.format(static_image_route_Regression))
def serve_image_re(image_path):
    image_name_regression = '{}.png'.format(image_path)
    if image_name_regression not in list_of_images_Regression:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory_Regression, image_name_regression)
# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server



@app.callback(Output('tabs-content', 'children'), #output
              Input('tabs', 'value'))




# def update_rows(selected_value):
#     data = df[df['State'] == selected_value]
#     columns = [{"name": i, "id": i} for i in data.columns]
#     return [dt.DataTable(data=data, columns=columns)]


def render_content(tab):#used to rendering content of the first tab
    if tab == 'tab1':     #create the first tab
        return tab1_layout
    elif tab == 'tab2':
      return tab2_layout
    elif tab == 'tab3':
      return tab3_layout
    elif tab == 'tab4':
      return tab4_layout
    elif tab == 'tab5':
      return tab5_layout


if __name__ == '__main__':
    app.run_server(debug=True)
     
     
     
     
     
     
     



# app = dash.Dash()

# app.layout = html.Div([
#     dcc.Dropdown(
#         id='image-dropdown',
#         options=[{'label': i, 'value': i} for i in list_of_images],
#         value=list_of_images[0]
#     ),
#     html.Img(id='image')
# ])

# @app.callback(
#     dash.dependencies.Output('image', 'src'),
#     [dash.dependencies.Input('image-dropdown', 'value')])
# def update_image_src(value):
#     return static_image_route + value

# # Add a static image route that serves images from desktop
# # Be *very* careful here - you don't want to serve arbitrary files
# # from your computer or server
# @app.server.route('{}<image_path>.png'.format(static_image_route))
# def serve_image(image_path):
#     image_name = '{}.png'.format(image_path)
#     if image_name not in list_of_images:
#         raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
#     return flask.send_from_directory(image_directory, image_name)
