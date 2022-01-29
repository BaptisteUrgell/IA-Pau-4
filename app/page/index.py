import streamlit as st
import pandas as pd
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


def app():
    st.title("Add dataset, image or ")        
            
    files = st.file_uploader("Upload a Dataset", type=["jpeg","csv","json"], key="files", accept_multiple_files=True)
    
    pie_chart = st.empty()
    
    
    def json_file(file):
        coco = COCO(annotation_file = 'instances_train.json')
        coco_categories = coco.dataset['categories'][1:]
        nb_anns_per_cat = {cat['name']: len(coco.getAnnIds(catIds=[cat['id']])) for cat in coco_categories}

        labels = []
        sizes = []

        for key in sorted(nb_anns_per_cat, key=nb_anns_per_cat.get, reverse=True):
            labels.append(key)
            sizes.append(nb_anns_per_cat[key])
        
        
        # Plot
        fig, ax = plt.subplots()
        explode = (0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5)
        colors = ['#191970','#001CF0','#0038E2','#0055D4','#0071C6','#008DB8','#00AAAA','#00C69C','#00E28E','#00FF80',]
        ax.pie(sizes, labels=labels, startangle=90, explode=explode, colors=colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig.set_alpha(1)
        pie_chart = st.pyplot(fig)
    
    
    for file in files:
        if file.type == "text/csv":
            df = pd.read_csv(file)
            st.dataframe(df)
            """
            le code sur le dataframe à faire
            """
        elif file.type == "image/jpeg":
            st.write(file.type)
            st.image(file)
        elif file.type == "application/json":
            json_file(file)
