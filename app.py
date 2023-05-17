from dash import dcc, html, dash_table, Dash
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import json
import numpy as np
import io
import base64
import sys


from glob import glob
import re
import os

def get_images_from_id(pid, data_dir=r'./data/CDD-CESM/Low energy images of CDD-CESM/', prefix='P'):
    image_glob = prefix +  "_".join(re.findall(r'(\d+)([L,R]+)', pid)[0]) + "*.jpg"
    image_glob = os.path.join(data_dir, image_glob)
    return glob(image_glob)

#Extract
all_results = pd.read_csv('output/all_results.csv', index_col=1)
ensemble_predictions = pd.read_csv('output/ensemble_predictions.csv')
annotation = pd.read_csv('data/annotation/annotation.csv', index_col=-1)[['symptoms']]

#Transform
all_results = all_results.merge(annotation, left_index=True, right_index=True, how='left')
all_results['images'] = [get_images_from_id(pid) for pid in all_results.index]

#Load
DATA_DF = all_results.copy()


external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    dbc.icons.BOOTSTRAP,
]

try:
    assert __IPYTHON__
    app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
except:
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    
server = app.server



def app_runner():
    app.layout = html.Div([
        html.H5('Disease prediction using Multimodal data (NLP and Vision)'),
        html.Div([
            html.Div([
                dcc.Dropdown(DATA_DF.index, DATA_DF.index[0:3],
                             multi=True,
                             id='patient-selector'),                
                html.Div([], id="view"),
            ],
            className='row')
            ])])
    return app




def get_alert(label , ptype):
    icon, text = get_prediction_type(ptype)
    if label == "Normal":
        return html.Div(
            [
                text,
                html.Hr(),
                dbc.Alert([
            
                html.I(
                       className="bi bi-check-circle-fill me-2",
                      ),
                html.Br(),

                label
            ],
            color="success",
            className="align-items-center",style={'padding': '5px'})
            ],
        )
    if label == "Malignant":
        return html.Div(
            [
                text,
                html.Hr(),
                dbc.Alert(
            [
                html.I(className="bi bi-x-octagon-fill me-2"),
                label,
            ],
            color="danger",
            className="align-items-center", style={'padding': '5px'})
            ],
        )
    if label == "Benign":
        return html.Div(
            [
                text,
                html.Hr(),
                dbc.Alert(
                    [
                        html.I(className="bi bi-exclamation-triangle-fill me-2"),
                        label,
                    ],
            color="warning",
            className="align-items-center", style={'padding': '5px'})
            ], 
        )

def get_prediction_type(ptype):
    if ptype == 'vision_predictions':
        return (html.I(className="bi bi-eye test"), 'Vision Results')
    if ptype == 'nlp_predictions':
            return (html.I(className="bi bi-file-earmark-text"), 'NLP Results')
    if ptype == 'ensemble_predictions':
        return (html.I(className="bi bi-plus-circle"), 'Ensemble Results')

def get_preds(pid):
    ptypes = ['ensemble_predictions']
    preds = DATA_DF.loc[pid][ptypes]
    
    card_content = [
        dbc.CardHeader([html.I(className="bi bi-clipboard-data", style={'padding': '5px'}), "Results"]),
        dbc.CardBody(
            [get_alert(pred, ptype) for ptype, pred in zip(ptypes, preds)]

        )]
    
    return dbc.Card(dbc.CardBody(card_content))
    
def get_card(pid):
    text = DATA_DF.loc[pid]['symptoms']
    
    card_content = [
        dbc.CardHeader([html.I(className="bi bi-file-earmark-person", style={'padding': '5px'}), "Study ID " + pid]),
        dbc.CardBody([html.P(text)])
    ]
    
    return dbc.Card(dbc.CardBody(card_content))

def get_imgs(pid):
    imgs = DATA_DF.loc[pid]['images']
    card_content = [
        dbc.CardHeader([html.I(className="bi bi-images", style={'padding': '5px'}), "Mamogram"]),
        dbc.CardBody(
            [html.Img(src=b64_image(img), style={'height':'50%', 'width':'50%', 'padding': '5px'}) for img in imgs]
        )]
    return dbc.Card(dbc.CardBody(card_content))

def get_row(pid):
    return dbc.Row(
        [
            dbc.Col(html.Div(get_card(pid))),
            dbc.Col(get_imgs(pid)),
            dbc.Col(get_preds(pid)),
        ], style={'border': "1px solid grey"}
    )

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')


@app.callback(
    Output('view', 'children'),
    Input('patient-selector', 'value')
)
def update_row(input_values):
    return [get_row(pid) for pid in input_values[::-1]]


def show_results(host="127.0.0.1", port=5555):
    app = app_runner()
    app.run_server(mode="inline", host=host,  port=port)