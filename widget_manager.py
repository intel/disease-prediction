from ipywidgets import GridspecLayout, Layout, Button, Image, Box, HBox, VBox, Textarea
from IPython.display import display
import ipywidgets as widgets
from glob import glob
import pandas as pd
import re
import os

class Data_Extractor:
    def __init__(self, all_results:pd.core.frame.DataFrame) -> None:
        self.all_results = all_results


    def get_patient_file(self, patient_id:str) -> pd.core.series.Series:
        if patient_id in self.all_results.index:
            return self.all_results.loc[patient_id]
        else:
            return None


    def get_symptoms(self, patient_id:str, card:pd.core.series.Series) -> Button and Textarea:
        title = Button(
                    layout=Layout(height="50%", width="100%"),
                    value=False,
                    description='Study ID: {}'.format(patient_id),
                    disabled=True,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    icon='user-circle' # (FontAwesome names without the `fa-` prefix)
                )


        text_area = Textarea(
                            layout=Layout(height="98%", width="100%"),
                            value=card['symptoms'],
                            placeholder='Type something',
                            description='Symptoms:',
                            disabled=True
                    )
        
        return title, text_area
    

    def get_images(self, card:pd.core.series.Series) -> Button and VBox:
        study = Button(
            layout=Layout(height="50%", width="100%"),
            value=False,
            description='Mamogram',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            icon='image' # (FontAwesome names without the `fa-` prefix)
        )

        items = [Image(layout=Layout(height="100%", width="100%"), value=open(image, "rb").read(), format='jpg') for image in card['images']]
        box_layout = Layout(overflow='scroll hidden',
                            border='3px solid black',
                            width='100%',
                            height='',
                            flex_flow='row',
                            display='flex')
        carousel = Box(children=items, layout=box_layout)
        images_box = VBox([carousel])

        return study, images_box


    def get_results(self, card:pd.core.series.Series):
        results = Button(
            layout=Layout(height="50%", width="100%"),
            value=False,
            description='Results ',
            disabled=True,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            icon='file-text-o' # (FontAwesome names without the `fa-` prefix)
        )
        
        items_layout = Layout(width='100%')

        box_layout = Layout(display='flex',
                            flex_flow='column',
                            align_items='stretch',
                            width='100%')

        if card['ensemble_predictions'] == 'Malignant':
            icon = 'times-circle'
            color = 'danger'
        elif card['ensemble_predictions'] == 'Benign':
            icon = 'warning'
            color = 'warning'
        else:
            icon = 'check-circle'
            color = 'success'


        items = [Button(description="Ensemble Results:", object_position='left', layout=items_layout, button_style='', disabled=True),
                Button(description=card['ensemble_predictions'], object_position='left', layout=items_layout, button_style=color, icon=icon, disabled=True)]
        results_box = Box(children=items, layout=box_layout)

        return results, results_box


    def get_report_card(self, patient_id:str):
        card = self.get_patient_file(patient_id)
        print(card)

        grid = GridspecLayout(4, 3, height='300px', layout=Layout( border='solid', width='100%', height="100%"))
        
        title, text_area = self.get_symptoms(patient_id, card) 
        
        study, images_box = self.get_images(card)

        results, results_box = self.get_results(card)
        
        grid[0, 0] = title
        grid[1:, 0] = text_area
        grid[0, 1] = study
        grid[1:, 1] = images_box
        grid[0, 2] = results
        grid[1, 2] = results_box
        #print(text_area.keys)
        return grid


    def get_id_set(self, id_list) -> Box:

        items = [self.get_report_card(id) for id in id_list]
        box_layout = Layout(overflow='auto',
                            border='3px solid black',
                            width='auto',
                            height='300px',
                            flex_flow='column',
                            display='block')
        region = VBox(children=items, layout=box_layout)
        id_set = Box([region])

        return id_set


#-------------------------------------------------------------------------------

class Manager:
    def __init__(self, all_results:pd.core.frame.DataFrame):
        self.extractor = Data_Extractor(all_results)
        self.patient_ids = all_results.index
        self.patient_ids_selector = all_results.index
        self.patient_ids_selected = pd.Index([])
        self.selector = None
        self.selected = None
        self.grid = None


    def update_selection(self):
        self.selector.index = 0
        self.selected.index = 0
        self.selector.disabled = True
        self.selected.disabled = True

        id_selector_list = self.patient_ids_selector.to_list()
        id_selector_list.insert(0,"Select ID:")
        self.selector.options = id_selector_list

        id_selected_list = self.patient_ids_selected.to_list()
        id_selected_list.insert(0,"Select to Delete:")
        self.selected.options = id_selected_list

        id_set = self.extractor.get_id_set(id_selected_list[1:])
        self.grid[1:,:] = id_set

        self.selector.disabled = False
        self.selected.disabled = False


    def on_selected_change(self, change):
        #print(change)
        id_selected_pos=change['new']
        if id_selected_pos != 0:
            id_selected_pos-=1

            self.patient_ids_selector = self.patient_ids_selector.append( \
                pd.Index([self.patient_ids_selected[id_selected_pos]]))
            self.patient_ids_selected = self.patient_ids_selected.delete(id_selected_pos)
            self.update_selection()


    def on_selector_change(self, change):
        #print(change)
        id_selector_pos=change['new']
        if id_selector_pos != 0:
            id_selector_pos-=1

            self.patient_ids_selected = self.patient_ids_selected.append( \
                pd.Index([self.patient_ids_selector[id_selector_pos]]))
            self.patient_ids_selector = self.patient_ids_selector.delete(id_selector_pos)
            self.update_selection()


    def display(self):
        id_select_list = self.patient_ids_selector.to_list()
        id_select_list.insert(0,"Select ID:")
        self.selector = widgets.Select(
            options= id_select_list,
            value = "Select ID:",
            rows=5,
            description='Patient ID:',
            disabled=False
        )
        id_select_list.pop()

        self.selected = widgets.Select(
            options= ["Select to Delete:"],
            value = "Select to Delete:",
            rows=5,
            description='Selected:',
            disabled=False
        )

        self.selector.observe(self.on_selector_change, 'index')
        self.selected.observe(self.on_selected_change, 'index')
        self.grid = GridspecLayout(5, 3, height='500px')
        self.grid[0, 0] = self.selector
        self.grid[0, 1] = self.selected

        display(self.grid)

#-------------------------------------------------------------------------------

def get_images_from_id(pid, data_dir=r'./data/CDD-CESM/Low energy images of CDD-CESM/', prefix='P'):
    image_glob = prefix +  "_".join(re.findall(r'(\d+)([L,R]+)', pid)[0]) + "*.jpg"
    image_glob = os.path.join(data_dir, image_glob)
    return glob(image_glob)

#Extract
all_results = pd.read_csv('output/all_results.csv', index_col=1)
ensemble_predictions = pd.read_csv('output/ensemble_predictions.csv')
annotations = pd.read_csv('data/annotation/annotation.csv', index_col=-1)[['symptoms']]

#Transform
all_results = all_results.merge(annotations, left_index=True, right_index=True, how='left')
all_results['images'] = [get_images_from_id(pid) for pid in all_results.index]
#print(all_results)

data_extractor = Data_Extractor(all_results)
#display(data_extractor.get_report_card('156R'))
#display(data_extractor.get_id_set(['156R', '165R']))#, '162R', '154R', '9L', '14L', '123L', '128R', '283R', '121L', '108L', '107R']))
widget_manager = Manager(all_results)
widget_manager.display()

