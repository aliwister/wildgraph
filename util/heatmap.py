import h3
from shapely.geometry import Polygon, Point
import numpy as np
import pandas as pd
import random

def softmax(x): 
    np.exp(x)/sum(np.exp(x))
class Heatmap():
    def __init__(self, file):
        self.heatmap_resolution = 7
        self.generate_heatmap(file)
    
    def generate_heatmap(self, file):
        #data = joblib.load('heatmap.0.5.joblib')
        #dict = {row[0]: row[1] for row in data}
        columns = ['idx', 'label', 'location.lat', 'location.long']
        df = pd.read_csv(file, usecols=columns)
        heatmap_resolution = 6
        df['heatmap'] = df.apply(lambda row: h3.latlng_to_cell(row['location.lat'], row['location.long'], heatmap_resolution), axis=1)
        heatmap_counts = df['heatmap'].value_counts().reset_index()
        heatmap_counts.columns = ['heatmap', 'occurrences']
        self.dict = {row[0]: row[1] for row in heatmap_counts.values}

    def sample(self, h3_region_id):

        cells = np.array(list(h3.cell_to_children(h3_region_id, self.heatmap_resolution))) 
        values = [self.dict.get(key, 0) for key in cells]
        selected_cell = np.random.choice(cells, p=softmax(values))      

        # Get the polygon vertices of the H3 region
        hexagon_vertices = h3.cell_to_boundary(str(selected_cell), geo_json=False)
        
        # Create a Shapely Polygon from the vertices
        hexagon_polygon = Polygon(hexagon_vertices)
        
        # Generate a random point inside the polygon
        min_x, min_y, max_x, max_y = hexagon_polygon.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if hexagon_polygon.contains(random_point):
                break
        
        return [random_point.x, random_point.y]