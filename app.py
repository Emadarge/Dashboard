
import dash
import dash_core_components as dcc
import dash_html_components as html
import django_heroku
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import dash_table as dt
import base64
import flask
import glob
import os
from sklearn.cluster import KMeans

app=app.server
django_heroku.settings(locals())
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']  
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)   

df = pd.read_csv('raw_data_all_cleaned.csv')            

raw_data_Weather=pd.read_csv('IST_meteo_data_2017_2018_2019.csv')
raw_data_holiday=pd.read_csv('holiday_17_18_19.csv')
raw_data_Civ_2017=pd.read_csv('IST_Civil_Pav_2017_Ene_Cons.csv')
raw_data_Civ_2018=pd.read_csv('IST_Civil_Pav_2018_Ene_Cons.csv')

raw_data_Civ_all=pd.merge(raw_data_Civ_2017,raw_data_Civ_2018,how='outer') 

raw_data_Civ_all['Date'] = pd.to_datetime(raw_data_Civ_all['Date_start'],format='%d-%m-%Y %H:%M')
raw_data_Weather['Date'] = pd.to_datetime(raw_data_Weather['yyyy-mm-dd hh:mm:ss'],format='%Y-%m-%d %H:%M:%S') 
raw_data_holiday['Date'] = pd.to_datetime(raw_data_holiday['Date'],format='%d.%m.%Y')

del raw_data_Civ_all['Date_start']
del raw_data_Weather['yyyy-mm-dd hh:mm:ss']

raw_data_Civ_all.rename(columns={'Power_kW':'Power [kW]'},inplace=True) #'Date_start':'Date',
raw_data_Weather.rename(columns={'temp_C':'Temperature [°C]','windSpeed_m/s':'Wind Speed [m/s]','windGust_m/s':'Wind Gust [m/s]','pres_mbar':'Pressure [mbar]','solarRad_W/m2':'Solar Radiation [W/m2]','rain_mm/h':'Rain [mm/h]','rain_day':'Rain Day'},inplace=True)

raw_data_Civ_all=raw_data_Civ_all.set_index('Date')
raw_data_Weather=raw_data_Weather.set_index('Date')
raw_data_holiday=raw_data_holiday.set_index('Date')

raw_data_holiday=raw_data_holiday.resample('1H').ffill(23)                
raw_data_holiday=raw_data_holiday.fillna(0)                              

raw_data_Weather=raw_data_Weather.resample('H').mean()           

raw_data_all=pd.merge(raw_data_Civ_all,raw_data_Weather,on='Date')  

raw_data_all=raw_data_all.dropna()
raw_data_all=pd.merge(raw_data_all,raw_data_holiday,how='left',on='Date')

raw_data_all=raw_data_all.reset_index()                      
raw_data_all['Week Day']=raw_data_all['Date'].dt.weekday     
raw_data_all['Hours']=raw_data_all['Date'].dt.hour           
raw_data_all['Month']=raw_data_all['Date'].dt.month          
raw_data_all=raw_data_all.set_index(['Date'])                

raw_data_all=raw_data_all.drop(columns='Wind Speed [m/s]')     
raw_data_all=raw_data_all.drop(columns='Wind Gust [m/s]')
raw_data_all=raw_data_all.drop(columns='Rain [mm/h]')
raw_data_all=np.around(raw_data_all, decimals=1)               

raw_data_all = raw_data_all[raw_data_all['Power [kW]'] >raw_data_all['Power [kW]'].quantile(0.25) ]  


Solar_Rad = raw_data_all.pop('Solar Radiation [W/m2]')

model = KMeans(n_clusters=3).fit(raw_data_all)               
pred = model.labels_

raw_data_all['Cluster']=pred

df.rename(columns={'Power [kW]':'Power','Temperature [°C]':'Temperature','Pressure [mbar]':'Pressure','Solar Radiation [W/m2]':'G','Week Day':'WeekDay','Rain Day':'RainDay'},inplace=True)

image_dpattern= 'assets/Clustering/Daily Pattern.png'
encoded_image_dailyp = base64.b64encode(open(image_dpattern, 'rb').read())


image_directory_Feature = '/Users/emanu/Desktop/Polito Magistrale/Secondo Anno/Secondo Semestre/Energy services (Erasmus)/Project2/assets/Feature/'
list_of_images_Feature = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory_Feature))]
static_image_route_Feature = '/static2/'


image_directory_Regression = '/Users/emanu/Desktop/Polito Magistrale/Secondo Anno/Secondo Semestre/Energy services (Erasmus)/Project2/assets/Regression/'
list_of_images_Regression = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory_Regression))]
static_image_route_Regression = '/static3/'

app.layout = html.Div([                     
                                      
    html.H1('Civil Building Monitor'),      
        html.Img(src='assets/IST-1.png'     
              ,
            style={
                'height': '10%',
                'width': '10%'
            }
            ),
    html.H6('By Emanuele D\'Argenzio - ist1100846'),        #my name

    dcc.Tabs(id='tabs', value='tab1', children=[                    
        dcc.Tab(label='Raw data visualization', value='tab1'),
        dcc.Tab(label='Exploratory Data Analysis', value='tab2'),
        dcc.Tab(label='Clustering', value='tab3'),
        dcc.Tab(label='Feature Selection', value='tab4'), 
        dcc.Tab(label='Regression model and results', value='tab5'),      
    ]), 
    html.Div(id='tabs-content')    
])   

tab1_layout = html.Div([
                        html.H5('In this table, it is possible to navigate inside the cleaned datas and to sort them in ascending and descending order. '),
                        html.H6('It is also possible to clean some columns, if you don\'t want to see them.'),
                        html.H6('To be noticed: cleaned datas means datas obtained after the feature selection'),
    dt.DataTable(                         
        id='datatable-interactivity',     
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=True,
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current= 0,
        page_size= 10,
    ),
    html.Div(id='datatable-interactivity-container')
])

tab2_layout= html.Div([
                      html.H5('In this section, different graphs show the datas seen in the table of the first section. Please select the one you want to display:'),
            dcc.Dropdown( 
        id='dropdown_exp',
        options=[
            {'label': 'Power consumption Vs Temperature', 'value': 11},
            {'label': 'Power consumption vs Solar Radiation', 'value': 22},
            {'label': 'Power probability distribution', 'value': 33},
            {'label': 'Temperature probability distribution', 'value': 44},
        ], 
        value=11
        ),
        html.Div(id='Exploratory_data_analysis'),
    ])

tab3_layout= html.Div([
                html.H5('Please, select the clustering model you want to display:'),
            dcc.Dropdown( 
        id='dropdown_clustering',
        options=[
            {'label': 'Power Vs Temperature', 'value': 1},
            {'label': 'Power vs Hours', 'value': 2},
            {'label': 'Power vs Week Day', 'value': 3},
            {'label': 'Hours vs Power', 'value': 4},
            {'label': '3D Clustering', 'value': 5},
            {'label': 'Daily pattern', 'value': 6}
        ], 
        value=1
        ),
        html.Div(id='Clustering_id'),
    ])

tab4_layout= html.Div([
                        html.H5('In this section, the feature selection process is reported. In particular, three different models can be selected. The features reported on the X axes are:'),
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
  html.H6('Please, select the model you want to display:'),
        dcc.RadioItems(
        id='image-dropdown-feature',
        options=[{'label': i, 'value': i} for i in list_of_images_Feature],
        value=list_of_images_Feature[0],
        labelStyle={'display': 'inline-block'}
    ),
    html.Center(html.Img(id='image-feature')
                )

    ])

tab5_layout= html.Div([
                            html.H5('In this section, the regression model used are presented. The selected features are'),
                     dcc.Markdown('''
                                     Temperature
                                     
                                     Week Day
                                     
                                     Hours
                                     
                                     Solar Radiation
                                     
                                     Power-1
                                     '''),
                                     
html.H5('Please, select the model you want to display:'),
  dcc.Dropdown(
         id='image-dropdown-regression',
         options=[{'label': i, 'value': i} for i in list_of_images_Regression],
         value=list_of_images_Regression[0]
     ),
     html.Center(html.Img(id='image-regression')),   
     
     html.H5('The numerical results for each model are:'),
                        
    dcc.RadioItems(
        id='checkboxes',
        options=[
            {'label': 'Bootstrapping', 'value': 111},
            {'label': 'Decision Tree', 'value': 222},
            {'label': 'Extreme gradient boosting', 'value': 333},
            {'label': 'Gradient boosting', 'value': 444},
            {'label': 'Linear regression', 'value': 555},
            {'label': 'Neural network', 'value': 666},
            {'label': 'Random forest (regression)', 'value': 777},
            {'label': 'SVR (kernel - rbf)', 'value': 888},
        ],
        value=111,
        labelStyle={'display': 'inline-block'}
    ),
    html.H6(id='Regression-result-text')
 
    ])


        
@app.callback(Output('Clustering_id', 'children'), 
              Input('dropdown_clustering', 'value'))
def render_figure_png(cluster_diff):
    
    if cluster_diff == 1:
        return html.Div([dcc.Graph(
        figure=px.scatter(raw_data_all,x='Power [kW]', y='Temperature [°C]',                                     
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Power vs Temperature')
        ),])
    elif cluster_diff == 2:
        return html.Div([        dcc.Graph(id='prova2',
        figure=px.scatter(raw_data_all,x='Power [kW]', y='Hours',                                               
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Power vs Hours')
        ),])
    elif cluster_diff == 3:
        return html.Div([dcc.Graph(id='prova3',
        figure=px.scatter(raw_data_all,x='Power [kW]', y='Week Day',                                              
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Power vs Week day')
        ),])
    elif cluster_diff == 4:
        return html.Div([dcc.Graph(id='prova4',
        figure=px.scatter(raw_data_all,x='Hours',y='Power [kW]',                                                 
                        color='Cluster',color_continuous_scale='jet',title='Clustering: Hours vs Power')
         ),])
    elif cluster_diff == 5:
        return html.Div([        dcc.Graph(id='prova5',
        figure = px.scatter_3d(raw_data_all,x='Hours',y='Week Day',z='Power [kW]',                               
                        color='Cluster',color_continuous_scale='jet',height=1000,title='3D Clustering')
        ),])
    elif cluster_diff == 6:
        return html.Div([html.H6('In below, the daily pattern, with the corresponding three clusters, are reported:'),
        html.Center(html.Img(src='data:image/png;base64,{}'.format(encoded_image_dailyp.decode()))),
        ])
    

@app.callback(Output('Exploratory_data_analysis', 'children'), 
              Input('dropdown_exp', 'value'))
def render_figure_exp(exp_graphs):
    
    if exp_graphs == 11:
        return html.Div([dcc.Graph(
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
    ),])
    elif exp_graphs == 22:
        return html.Div([dcc.Graph(              
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
    ),])
    elif exp_graphs == 33:
        return html.Div([dcc.Graph(id="prob"),         
    html.P("Distributions"),
    dcc.RadioItems(                
        id='dist-marginal',
        options=[{'label': x, 'value': x} 
                 for x in ['box', 'violin', 'rug']],
        value='box'
    ),])
    elif exp_graphs == 44:
        return html.Div([ dcc.Graph(id="prob2"),      
    html.P("Distributions"),
    dcc.RadioItems(                  
        id='dist-marginal2',
        options=[{'label': x, 'value': x} 
                  for x in ['box', 'violin', 'rug']],
        value='box'
    ),])

@app.callback(
    Output("prob", "figure"), 
    [Input("dist-marginal", "value")])
def display_graph(marginal):
    fig = px.histogram(
        df['Power'], x="Power",color_discrete_sequence=['red'],
        marginal=marginal,title= 'Probability distribution for the power')
    return fig


@app.callback(
    Output("prob2", "figure"), 
[Input("dist-marginal2", "value")])
def display_graph_1(marginal):
    fig2 = px.histogram(
        df['Temperature'], x="Temperature",title= 'Probability distribution for the temperature',
        marginal=marginal)
    return fig2


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
 
@app.callback(
    Output('Regression-result-text', 'children'),
    [Input('checkboxes', 'value')])
def update_reg_res(reg_res):
    if   reg_res==111:
        return html.Div([html.H6('Bootstrapping (Bootstrapping.png)                       - results: MAE_BT=8.97 MSE_BT=196.31 RMSE_BT=14.01 cvRMSE_BT=0.0682')])
    elif reg_res==222:
        return html.Div([html.H6('Decision tree regressor (Decision Tree.png)        - results: MAE_DT=11.39 MSE_DT=329.86 RMSE_DT=18.16 cvRMSE_DT=0.088')])
    elif reg_res==333:
        return html.Div([html.H6('Extreme Gradient Boosting (Extreme Gradient Boosting.png)               - results: MAE_XGB=8.71 MSE_XGB=170.66 RMSE_XGB=13.06 cvRMSE_XGB=0.0635')])
    elif reg_res==444:
        return html.Div([html.H6('Gradient Boosting (Gradient Boosting.png)                        - results: MAE_GB=9.45 MSE_GB=187.57 RMSE_GB=13.69 cvRMSE_GB=0.0665')])
    elif reg_res==555:
        return html.Div([html.H6('Linear Regression (Linear Regression.png)                        - results: MAE_LR=21.33 MSE_LR=808.51 RMSE_LR=28.43 cvRMSE_LR=0.138')])
    elif reg_res==666:
        return html.Div([html.H6('Neural Network (Neural Network.png)                       - results: MAE_NN=13.58 MSE_NN=380.66 RMSE_NN=19.51 cvRMSE_NN=0.0945')])
    elif reg_res==777:
        return html.Div([html.H6('Random Forest (Random Forest.png)                            - results: MAE_RF=8.80 MSE_RF=172.88 RMSE_RF=13.14 cvRMSE_RF=0.0638')])
    elif reg_res==888:
        return html.Div([html.H6('Support Vector Regressor, kernel RBF (SVR - RBF kernel.png)   - results: MAE_SVR=11.15 MSE_SVR=255.25 RMSE_SVR=15.97 cvRMSE_SVR=0.077 ')])


@app.callback(Output('tabs-content', 'children'), #output
              Input('tabs', 'value'))


def render_content(tab):              
    if tab == 'tab1':                
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
     
