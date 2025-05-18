"""Classes to generate point clouds and build dataset for training and testing."""

import torch
import time
import numpy as np
import open3d as o3d
import random
import json
from copy import deepcopy
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class FullPCDDataCreator:
    """Class to generate point clouds and store them in a dataset.


    Attributes:
        pcd_array (np.ndarray): Array to store point clouds.
        transformation_array (np.ndarray): Array to store transformations in the form of 4x4 matrices.
        quaternion_array (np.ndarray): Array to store quaternions.
        position_array (np.ndarray): Array to store positions.
        class_array (np.ndarray): Array to store class labels.
        d6_rotations_array (np.ndarray): Array to store 6D representations of rotations.
        n_points (int): Number of points to sample from each mesh.
        z_min (float): Minimum z-coordinate for the center of the point cloud.
        z_max (float): Maximum z-coordinate for the center of the point cloud.
        indent (float): Indentation value for the point cloud generation."""

    def __init__(self):
        # arrays to store data
        self.pcd_array = None
        self.transformation_array = None
        self.quaternion_array = None
        self.position_array = None
        self.class_array = None
        self.d6_rotations_array = None
        # generation parameters
        self.n_points = 1000
        self.z_min = 0.5
        self.z_max = 3
        self.indent = 0.25

    def set_parameters(self, n_points, z_min, z_max, indent):
        """Set parameters for point cloud generation.

        Args:
            n_points (int): Number of points to sample from each mesh.
            z_min (float): Minimum z-coordinate for the center of the point cloud.
            z_max (float): Maximum z-coordinate for the center of the point cloud.
            indent (float): Indentation value for the point cloud generation that affects the position of the center of the point cloud in x/y plane.
        """
        self.n_points = n_points
        self.z_min = z_min
        self.z_max = z_max
        self.indent = indent

    def generate_random_transformation_quaternion(self):
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

        r = (z*np.tan(np.pi/4) - indent) * random.random()
        angle = random.random()*2*np.pi
        x = r*np.cos(angle)
        y = r*np.sin(angle)

        T = np.array([
            [1 - 2*(q3**2 + q4**2),     2*(q2*q3 - q1*q4),
             2*(q2*q4 + q1*q3), x],
            [2*(q2*q3 + q1*q4), 1 - 2*(q2**2 + q4**2),     2*(q3*q4 - q1*q2), y],
            [2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2**2 + q3**2), z],
            [0, 0, 0, 1]
        ])
        return T, np.array([q1, q2, q3, q4]), np.array([x, y, z])

    def generate_random_transformation_6d(self, sample_size=1):
        z_min = self.z_min
        z_max = self.z_max
        indent = self.indent
        # shape (sample_size, 3, 3)
        random_rotations = R.random(sample_size).as_matrix()
        rot_6d = random_rotations[:, :, :2].reshape(sample_size, 6)

        z = z_min + np.random.rand(sample_size)*(z_max-z_min)
        r = (z*np.tan(np.pi/4) - indent) * np.random.rand(sample_size)
        angle = np.random.rand(sample_size)*2*np.pi
        x = r*np.cos(angle)
        y = r*np.sin(angle)
        translations = np.column_stack((x, y, z))  # shape (sample_size, 3)
        transformations = np.zeros((sample_size, 4, 4))
        transformations[:, :3, :3] = random_rotations
        transformations[:, 3, :3] = translations
        transformations[:, 3, 3] = 1
        return transformations, rot_6d, translations

    def generate_pcd(self, model: o3d.geometry.TriangleMesh, transformation) -> o3d.geometry.PointCloud:
        """Generate a point cloud from a mesh after the transformation.

        Args:
            model (o3d.geometry.TriangleMesh): mesh
            transformation (np.ndarray): transformation 4x4 matrix

        Returns:
            o3d.geometry.PointCloud: sampled point cloud
        """
        model = deepcopy(model)
        model.transform(transformation)
        return model.sample_points_poisson_disk(self.n_points)

    def generate_data_6d(self, model_paths, labels, n_samples=10000):
        """Generate point clouds and save in numpy format.

        Args:
            model_paths (list): list of paths to the mesh files
            labels (list): list of string labels for each mesh 
            n_samples (int): number of samples to generate
        """
        # load models as triangle meshes
        models = [o3d.io.read_triangle_mesh(path) for path in model_paths]
        model_idxs = list(range(len(models)))
        # labels should be alligned with the models
        if len(labels) != len(models):
            raise ValueError("Labels should be the same length as models.")
        label_dict = {i: labels[i] for i in range(len(models))}
        # generate random transformations
        start = time.perf_counter()
        self.transformation_array, self.d6_rotations_array, self.position_array = self.generate_random_transformation_6d(
            n_samples)
        end = time.perf_counter()
        print(f"Subfunction took {end - start:.4f} seconds")
        # transformation_list = []
        # quaternion_list = []
        # position_list = []
        pcd_list = []
        idx_list = []
        for i in range(n_samples):
            if (i+1) % 1000 == 0:
                print(f"Generating sample {i+1}/{n_samples}")
            # transformation, quaternion, position = self.generate_random_transformation()
            # position_list.append(position)
            # quaternion_list.append(quaternion)
            # transformation_list.append(transformation)
            # select a random model
            idx = random.choice(model_idxs)
            idx_list.append(idx)
            model = models[idx]
            # generate a point cloud from the model with the sampled transformation
            pcd = self.generate_pcd(model, self.transformation_array[i])
            # transform the point cloud to numpy array and append to the list
            pcd_list.append(np.asarray(pcd.points))

        self.pcd_array = np.array(pcd_list)
        # self.transformation_array = np.array(transformation_list)
        # self.quaternion_array = np.array(quaternion_list)
        # self.position_array = np.array(position_list)
        self.class_array = np.array(idx_list)
        np.savez(f"full_pcd_{n_samples}_samples.npz", pcds=self.pcd_array,
                 transformations=self.transformation_array, rotations_6d=self.d6_rotations_array, positions=self.position_array, classes=self.class_array)
        with open("label_dict.json", "w") as f:
            json.dump(label_dict, f)


class FullPCDDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.point_clouds = data['pcds'] 
        self.labels = data['classes']
        self.positions = data['positions']
        self.transformations = data['transformations']
        # self.quaternions = data['quaternions']
        self.rotations_6d = data['rotations_6d']
        self.transform = transform

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx]
        rotation_6d = self.rotations_6d[idx]
        position = self.positions[idx]

        # Convert to tensors
        point_cloud = torch.from_numpy(point_cloud).float()
        label = torch.tensor(label, dtype=torch.long)
        position = torch.from_numpy(position).float()
        rotation_6d = torch.from_numpy(rotation_6d).float()
        if self.transform:
            point_cloud = self.transform(point_cloud)

        return point_cloud, (label, rotation_6d, position)
