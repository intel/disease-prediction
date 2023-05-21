from ipywidgets import GridspecLayout
from IPython.display import display
import ipywidgets as widgets
from glob import glob
import pandas as pd
import re
import os

class Manager:
    def __init__(self, patient_id:pd.core.indexes.base.Index) -> None:
        self.patient_id = patient_id
        self.patient_id_selector = patient_id
        self.patient_id_selected = pd.Index([])
        self.selector = None
        self.selected = None


    def update_selection(self):
        self.selector.index = 0
        self.selected.index = 0

        id_selector_list = self.patient_id_selector.to_list()
        id_selector_list.insert(0,"Select ID:")
        self.selector.options = id_selector_list

        id_selected_list = self.patient_id_selected.to_list()
        id_selected_list.insert(0,"Select to Delete:")
        self.selected.options = id_selected_list


    def on_selected_change(self, change):
        print(change)
        id_selected_pos=change['new']
        if id_selected_pos != 0:
            id_selected_pos-=1

            self.patient_id_selector = self.patient_id_selector.append( \
                pd.Index([self.patient_id_selected[id_selected_pos]]))
            self.patient_id_selected = self.patient_id_selected.delete(id_selected_pos)
            self.update_selection()


    def on_selector_change(self, change):
        print(change)
        id_selector_pos=change['new']
        if id_selector_pos != 0:
            id_selector_pos-=1

            self.patient_id_selected = self.patient_id_selected.append( \
                pd.Index([self.patient_id_selector[id_selector_pos]]))
            self.patient_id_selector = self.patient_id_selector.delete(id_selector_pos)
            self.update_selection()


    def display(self):
        id_select_list = self.patient_id_selector.to_list()
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
        grid = GridspecLayout(4, 3, height='300px')
        grid[3, 1] = self.selector
        grid[3, 2] = self.selected

        display(grid)


class Data_Extractor:
    def __init__(self, all_results:pd) -> None:
        self.all_results = all_results

    def get_patient_file(self, patient_id:str):
        if patient_id in self.all_results.index:
            return self.all_results.loc[patient_id]
        else:
            return None

    def get_report_card(self, patient_id:str):
        card = self.get_patient_file(patient_id)
        print(card)

        grid = GridspecLayout(4, 3, height='300px')
        
        text_area = widgets.Textarea(
                            layout=widgets.Layout(height="100%", width="auto"),
                            value=card['symptoms'],
                            placeholder='Type something',
                            description='Symptoms:',
                            disabled=True
                        )
        grid[:3, 1:] = text_area
        print(text_area.keys)
        return grid


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
display(data_extractor.get_report_card('107R'))
#widget_manager = Manager(all_results.index)
#widget_manager.display()

