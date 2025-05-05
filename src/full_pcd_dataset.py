import torch
import numpy as np
import open3d as o3d
import random 
import json
from copy import deepcopy

class FullPCDDataCreator:
    def __init__(self):
        # arrays to store data
        self.pcd_array = None
        self.position_array = None
        self.class_array = None
        # generation parameters
        self.n_points = 1000
        self.z_min = 0.5
        self.z_max = 3
        self.indent = 0.25

    def set_parameters(self, n_points, z_min, z_max, indent):
        self.n_points = n_points
        self.z_min = z_min
        self.z_max = z_max
        self.indent = indent

    def generate_random_transformation(self):
        z_min = self.z_min
        z_max = self.z_max
        indent = self.indent
        # generate a random rotation using quaternion sampling
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

        # Quaternion to rotation matrix
        # R =  np.array([
        #     [1 - 2*(q3**2 + q4**2),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        #     [    2*(q2*q3 + q1*q4), 1 - 2*(q2**2 + q4**2),     2*(q3*q4 - q1*q2)],
        #     [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2)]
        # ])
        
        # random translation in z direction
        z = z_min + random.random()*(z_max-z_min)
        
        r = (z*np.tan(np.pi/3) - indent) * random.random
        angle = random.random()*2*np.pi
        x = r*np.cos(angle)
        y = r*np.sin(angle)

        T = np.array([
            [1 - 2*(q3**2 + q4**2),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3), x],
            [    2*(q2*q3 + q1*q4), 1 - 2*(q2**2 + q4**2),     2*(q3*q4 - q1*q2), y],
            [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2), z],
            [0,0,0,1]
        ])
        return T
    
    def generate_pcd(self, model:o3d.geometry.TriangleMesh, transformation):
        model = deepcopy(model)
        model.transform(transformation)
        return model.sample_points_poisson_disk(self.n_points)

    def generate_data(self, model_paths, labels, n_samples = 10000):
        models = [o3d.io.read_triangle_mesh(path) for path in model_paths]
        model_idxs = list(range(len(models)))
        label_dict = {i:labels[i] for i in range(len(models))}
        position_list = []
        pcd_list = []
        idx_list = []
        for i in range(n_samples):
            transformation = self.generate_random_transformation()
            position_list.append(transformation)
            idx = random.choice(model_idxs)
            idx_list.append(idx)
            model = models[idx]
            pcd = self.generate_pcd(model, transformation)
            pcd_list.append(pcd)
            
        self.pcd_array = np.array(pcd_list)
        self.position_array = np.array(position_list)
        self.class_array = np.array(idx_list)
        np.savez(f"full_pcd_{n_samples}_samples.npz", pcds=self.pcd_array, transformations = self.position_array,classes = self.class_array)
        with open("label_dict.json", "w") as f:
            json.dump(label_dict,f)