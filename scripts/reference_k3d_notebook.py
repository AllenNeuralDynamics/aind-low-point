# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np

from ipywidgets import interact, interactive, widgets, fixed

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.cm as cm
import matplotlib
import k3d
implant_cmap = matplotlib.cm.get_cmap('rainbow')

import pandas as pd
from aind_mri_utils import rotations as rot
from aind_mri_utils.file_io.slicer_files import markup_json_to_numpy, markup_json_to_dict,load_segmentation_points,find_seg_nrrd_header_segment_info,create_slicer_fcsv,read_slicer_fcsv
from aind_mri_utils.file_io.obj_files import get_vertices_and_faces
from aind_mri_utils.file_io.simpleitk import save_sitk_transform,load_sitk_transform
from aind_mri_utils.plots import make_3d_ax_look_normal,plot_tri_mesh,set_axes_equal,get_prop_cycle
from aind_mri_utils import coordinate_systems as cs
from aind_mri_utils.measurement import angle

from aind_mri_utils.optimization import get_headframe_hole_lines
from aind_mri_utils.optimization import optimize_transform_labeled_lines
from aind_mri_utils.rotations import prepare_data_for_homogeneous_transform as append_ones_column

from aind_mri_utils.meshes import mask_to_trimesh
from aind_mri_utils.sitk_volume import resample3D
from aind_mri_utils.plots import rgb_to_hex_string,rgb_to_int,hex_string_to_int

from aind_mri_utils.chemical_shift import compute_chemical_shift,chemical_shift_transform


import pywavefront
from pywavefront import visualization
from pathlib import Path
import os

from scipy.optimize import fmin
from scipy.spatial.transform import Rotation
from rotations import fit_params


import json
# %matplotlib ipympl

colors = get_prop_cycle()

def create_single_colormap(colorname,N = 256,saturation = 0,start_color = "white",is_transparent = True,is_reverse = False):
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    cmap = ListedColormap([start_color,colorname])
    start_color = np.array(cmap(0))
    if is_transparent:
        start_color[-1] = 0
    if not is_reverse:
        cmap = ListedColormap(
            np.vstack(
                (np.linspace(start_color,cmap(1),N),
                np.tile(cmap(1),(int(saturation*N),1)))
            )
        )
    else:
        cmap = ListedColormap(
            np.vstack(
                (np.tile(cmap(1),(int(saturation*N),1)),
                np.linspace(cmap(1),start_color,N),)
            )
        )
    return cmap

def define_transform(source_landmarks, target_landmarks):

    """
    Defines a non-linear warp between a set of source and target landmarks

    Parameters
    ==========
    source_landmarks - np.ndarray (N x 3)
    target_landmarks - np.ndarray (N x 3)

    Returns
    =======
    transform - vtkThinPlateSplineTransform

    """

    transform = vtk.vtkThinPlateSplineTransform()
    source_points = vtk.vtkPoints()
    target_points = vtk.vtkPoints()

    for i in range(source_landmarks.shape[0]):
        source_points.InsertNextPoint(source_landmarks[i,:])

    for i in range(target_landmarks.shape[0]):
        target_points.InsertNextPoint(target_landmarks[i,:])

    transform.SetBasisToR() # for 3D transform
    transform.SetSourceLandmarks(source_points)
    transform.SetTargetLandmarks(target_points)
    transform.Update()

    return transform

import trimesh


def transform_matrix_from_angles_and_target(AP,ML,Target,degrees = True):
    #T = trimesh.transformations.euler_matrix(np.deg2rad(AP),np.deg2rad(ML),0)
    R = Rotation.from_euler('XYZ',np.array([np.deg2rad(AP),np.deg2rad(ML),0])).as_matrix()
    T = np.zeros([4,4])
    T[:3,:3] = R
    T[0:3,3] = Target
    return T

def apply_transform_to_mesh(mesh,T):
    mesh.vertices = trimesh.transform_points(mesh.vertices,T)
    return mesh


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.

    see https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def load_newscale_model(filename, move_down  = 0):
    mesh = trimesh.load_mesh(filename)
    mesh = as_mesh(mesh)
    mesh.vertices = mesh.vertices[:,[0,2,1]]
    mesh.vertices[:,2] = mesh.vertices[:,2]-move_down
    trimesh.repair.broken_faces(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    return mesh

def move_and_copy_newscale_model(mesh,move_down = 0):
    this_mesh = mesh.copy()
    this_mesh.vertices[:,2] = this_mesh.vertices[:,2]-move_down
    return this_mesh

def load_sitk_transform(filename):
    B = sitk.ReadTransform(filename)
    A = B.GetInverse()
    matrix = np.array(A.GetParameters())[:9].reshape((3, 3)).T
    offset = np.array(A.GetParameters()[-3:])
    trans = np.vstack([matrix, offset])
    return trans

def load_sitk_transform_inverse(filename):
    A = sitk.ReadTransform(filename)
    #A = B.GetInverse()
    matrix = np.array(A.GetParameters())[:9].reshape((3, 3)).T
    offset = np.array(A.GetParameters()[-3:])
    trans = np.vstack([matrix, offset])
    return trans

def get_colormap_colors(N, colormap='viridis'):
    """
    Generate N colors evenly spaced across the given colormap.
    
    Parameters:
        N (int): Number of colors needed.
        colormap (str): Name of the matplotlib colormap to use.
    
    Returns:
        list: A list of N RGB tuples.
    """
    cmap = cm.get_cmap(colormap, N)  # Get the colormap with N discrete colors
    colors = [cmap(i) for i in range(N)]  # Extract the colors
    return colors


def points_in_mask(
    mask_path: str,
    points_physical: np.ndarray,
    coord_system: str = "LPS",
) -> np.ndarray:
    """
    Determine which physical‑space points lie inside a binary mask image.

    Parameters
    ----------
    mask_path : str
        Path to a **binary** mask image (.nrrd, .nii, …) readable by SimpleITK.
    points_physical : (N, 3) ndarray[float]
        Physical‑space coordinates of the points.
    coord_system : {"LPS", "RAS"}, default "LPS"
        Coordinate convention of *points_physical*.
        * SimpleITK uses **LPS** internally, so if your points are RAS we
          convert them to LPS (x ← −x, y ← −y).

    Returns
    -------
    inside : (N,) ndarray[bool]
        True for points that fall inside the mask (mask > 0).

    Notes
    -----
    • Vectorised: no Python‑level loops.  
    • Reads the mask once and accesses it with a zero‑copy array view.  
    """
    if coord_system.upper() not in {"LPS", "RAS"}:
        raise ValueError("coord_system must be 'LPS' or 'RAS'")

    pts = np.asarray(points_physical, dtype=float)

    # Convert RAS → LPS if needed (flip X and Y axes)
    if coord_system.upper() == "RAS":
        pts = pts.copy()
        pts[:, 0:2] *= -1.0

    # -------- load mask & prepare transform --------------------------
    mask_img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayViewFromImage(mask_img)          # (z, y, x)

    spacing   = np.asarray(mask_img.GetSpacing())            # (sx, sy, sz)
    origin    = np.asarray(mask_img.GetOrigin())             # (ox, oy, oz)
    direction = np.asarray(mask_img.GetDirection()).reshape(3, 3)

    # Physical → continuous index coordinates
    world_to_cont = (direction / spacing).T                  # (3,3)
    idx_float = (world_to_cont @ (pts.T - origin[:, None])).T

    # Nearest‑voxel integer indices
    idx_int = np.floor(idx_float + 0.5).astype(int)

    # ---- bounds check (vectorised) ---------------------------------
    size_x, size_y, size_z = mask_img.GetSize()
    valid = (
        (idx_int[:, 0] >= 0) & (idx_int[:, 0] < size_x) &
        (idx_int[:, 1] >= 0) & (idx_int[:, 1] < size_y) &
        (idx_int[:, 2] >= 0) & (idx_int[:, 2] < size_z)
    )

    # SimpleITK array order is (z, y, x)
    kji = idx_int[valid][:, ::-1]       # swap xyz → zyx
    inside = np.zeros(len(pts), dtype=bool)
    inside[valid] = mask_arr[kji[:, 0], kji[:, 1], kji[:, 2]] > 0

    return inside


import numpy as np
from scipy.spatial import cKDTree

def local_density_fixed_radius(points, r):
    """
    For each point i, density[i] = # neighbours within radius r 
                                  / volume_of_sphere(r)
    Returns an array of shape (N,).
    """
    tree = cKDTree(points)

    # query_ball_point with return_length=True → one integer per point
    counts = tree.query_ball_point(points, r, return_length=True)
    counts -= 1                      # drop the point itself
    volume = (4.0/3.0) * np.pi * r**3
    return counts / volume

def generate_sitk_trimesh_from_ccf(
    structure_acronym,
    annotation,
    structuredf,
    include_descendants=True,
    dtype=sitk.sitkUInt8,
):
    """
    Generate a trimesh mesh for a specified Allen CCF structure from a SimpleITK
    annotation image, and return its Allen structure color.

    Parameters
    ----------
    structure_acronym : str
        Acronym of the target structure, e.g. "CA1", "VISp", "PL".

    annotation : sitk.Image
        SimpleITK annotation image. Voxel values should be Allen structure IDs.

    structuredf : pandas.DataFrame
        Allen structure table. Must contain columns:
        - 'id'
        - 'acronym'
        - 'structure_id_path'
        - 'color_hex_triplet'

    include_descendants : bool, default True
        If True, include all child structures whose structure_id_path contains
        the target structure ID.

    dtype : SimpleITK PixelIDValueEnum, default sitk.sitkUInt8
        Output mask pixel type.

    Returns
    -------
    mesh : trimesh.Trimesh
        Mesh of the requested structure.

    mask_img : sitk.Image
        Binary SimpleITK mask used to generate the mesh.

    color_hex_triplet : str
        Allen structure color hex triplet, e.g. "7ED04B".
    """

    # -----------------------------
    # Find requested structure row
    # -----------------------------
    matches = structuredf.loc[structuredf["acronym"] == structure_acronym]

    if len(matches) == 0:
        raise ValueError(f"Could not find acronym '{structure_acronym}' in structuredf.")

    structure_row = matches.iloc[0]
    this_id = int(structure_row["id"])

    if "color_hex_triplet" not in structuredf.columns:
        raise ValueError("structuredf must contain a 'color_hex_triplet' column.")

    color_hex_triplet = structure_row["color_hex_triplet"]

    # Optional: normalize to a string without '#'
    if color_hex_triplet is not None:
        color_hex_triplet = str(color_hex_triplet).lstrip("#")

    # -----------------------------
    # Get IDs to include
    # -----------------------------
    if include_descendants:
        these_ids = []
        target_token = str(this_id)

        for _, row in structuredf.iterrows():
            path = str(row["structure_id_path"])
            path_ids = [p for p in path.split("/") if p != ""]

            if target_token in path_ids:
                these_ids.append(int(row["id"]))

    else:
        these_ids = [this_id]

    if len(these_ids) == 0:
        raise ValueError(
            f"No IDs found for acronym '{structure_acronym}' "
            f"with include_descendants={include_descendants}."
        )

    # -----------------------------
    # Build binary mask
    # -----------------------------
    ann_arr = sitk.GetArrayFromImage(annotation)
    mask_arr = np.isin(ann_arr, these_ids).astype(np.uint8)

    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.CopyInformation(annotation)
    mask_img = sitk.Cast(mask_img, dtype)

    # -----------------------------
    # Convert mask to trimesh
    # -----------------------------
    mesh = mask_to_trimesh(mask_img)

    return mesh, mask_img, color_hex_triplet


# %%
# File Paths
mouse = "836656"
current_user = "YB"
#target_structures = ["PL"]#, "CLA", "MD", "CA1", "VM", "BLA", "RSP"]

WHOAMI = "Yoni"

if WHOAMI == "Galen":
    base_path = Path("/mnt/aind1-vast/scratch")
elif WHOAMI == "Yoni":
    base_path = Path(r"Y:/")
else:
    raise ValueError("Who are you again?")

# File Paths
# Image and image annotations.
annotations_path = base_path / "ephys/persist/data/MRI/processed/{}".format(mouse)
image_path = annotations_path / "{}_100.nii.gz".format(mouse)
labels_path = annotations_path / "{}_HeadframeHoles.seg.nrrd".format(mouse)
brain_mask_path = annotations_path / "{}_auto_skull_strip.nrrd".format(mouse)
image_transform_file = annotations_path / "{}_com_plane.h5".format(mouse)
structure_mask_path = annotations_path / "Masks"
# structure_files = {
#     structure: structure_mask_path / f"{mouse}-{structure}-Mask.nrrd"
#     for structure in target_structures
# }

# Implant annotation 
# Note that this can be different than the image annotation, 
# this is in the event that an instion is planned with data from multiple scans (see 750107 for example).
implant_annoation_path = annotations_path
#headframe_transform_file = implant_annoation_path / "com_plane.h5".format(mouse)
implant_file = implant_annoation_path / "{}_ImplantHoles.seg.nrrd".format(mouse)
implant_mesh_file = implant_annoation_path / "{}_ImplantHoles.obj".format(mouse)
implant_fit_transform_file = implant_annoation_path/"{}_implant_fit.h5".format(mouse) 



# OBJ files
model_path = base_path / "ephys/persist/data/MRI/HeadframeModels"
modified_probe_mesh_file = model_path / "modified_probe_holder.obj"
dovetail_tweezer_file = model_path / "dovetailtweezer_oneShank_centered_corrected.obj"
dovetail_tweezer_4shank_file = model_path / "dovetailwtweezer_fourShank_centeredOnShank0.obj"
quadbase_file_0 = model_path / "QB_Centering"/"Quadbase_customHolder_centeredOnShank0.obj"
quadbase_dovetail_file_0 = model_path / "QB_Centering"/"Quadbase_dovetailHolder_centeredOnShank_0.obj"
quadbase_file_3 = model_path / "QB_Centering"/"Quadbase_customHolder_centeredOnShank3.obj"
quadbase_dovetail_file_3 = model_path / "QB_Centering"/"Quadbase_dovetailHolder_centeredOnShank_3.obj"

pipette_file = model_path / 'injection_pipette.obj'
newscale_model_file = model_path / "Centered_Newscale_2pt0.obj"
headframe_file = model_path / "TenRunHeadframe.obj"
holes_file = model_path / "OneOff_HolesOnly.obj"
cone_file = model_path/"TacoForBehavior"/"0160-200-72_X06.obj"
well_file = model_path/"WHC_Well"/"0274-400-07_X02.obj"
implant_model_file = model_path/"0283-300-04.obj"



# Save file paths
transform_save_file = annotations_path / "{}_test.h5".format(mouse)


# Magic numbers
resolution = 100

# %%
image_transform_file

# %%
# Load the image transform
image_trans = load_sitk_transform(str(image_transform_file)) 


# %%
# Handle inconsistant labeling
label_vol = sitk.ReadImage(str(labels_path))
odict = {k: label_vol.GetMetaData(k) for k in label_vol.GetMetaDataKeys()}
insert_underscores = "_" in list(find_seg_nrrd_header_segment_info(odict).keys())[0]

# Load the points on the headframe lines.
pts1, pts2, order = get_headframe_hole_lines(
    insert_underscores=insert_underscores, coordinate_system="LPS"
)

# %%
# Handle inconsistant labeling
label_vol = sitk.ReadImage(str(labels_path))
odict = {k: label_vol.GetMetaData(k) for k in label_vol.GetMetaDataKeys()}
insert_underscores = "_" in list(find_seg_nrrd_header_segment_info(odict).keys())[0]

# Load the points on the headframe lines.
pts1, pts2, order = get_headframe_hole_lines(
    insert_underscores=insert_underscores, coordinate_system="LPS"
)

# order.remove('anterior_vertical')

image = sitk.ReadImage(str(image_path))
positions, labels, weights = load_segmentation_points(
    str(labels_path), order=order, image=image
)
positions = append_ones_column(positions)

# Load the headframe
headframe, headframe_faces = get_vertices_and_faces(headframe_file)
headframe_lps = cs.convert_coordinate_system(
    headframe, "ASR", "LPS"
)  # Preserves shape!
headframe_mesh = trimesh.Trimesh(headframe_lps,headframe_faces[0])

# # Load the headframe
# cone, cone_faces = get_vertices_and_faces(cone_file)
# cone_lps = cs.convert_coordinate_system(
#     cone, "ASR", "LPS"
# )  # Preserves shape!

well, well_faces = get_vertices_and_faces(well_file)
well_lps = cs.convert_coordinate_system(
    well, "ASR", "LPS"
)  # Preserves shape!
well_mesh = trimesh.Trimesh(well_lps,well_faces[0])


# Load just the headframe holes
holes, holes_faces = get_vertices_and_faces(holes_file)
holes_faces = holes_faces[-1]
holes_lps = cs.convert_coordinate_system(holes, "ASR", "LPS")

# Load the implant
implant, implant_faces = get_vertices_and_faces(implant_model_file)
implant_lps = cs.convert_coordinate_system(
    implant, "ASR", "LPS"
)  # Preserves shape!


# Load the brain mask
mask = sitk.ReadImage(str(brain_mask_path))
brain_mask = mask_to_trimesh(mask)

# %%
# Get the trimesh objects for each hole.
# These are made using blender from the cad file
hole_folder = Path(r"Y:\ephys\persist\data\MRI\HeadframeModels\HoleOBJs")

hole_files = [
    x for x in os.listdir(hole_folder) if ".obj" in x and "Hole" in x
]
hole_dict = {}
for ii, flname in enumerate(hole_files):
    hole_num = int(flname.split("Hole")[-1].split(".")[0])
    hole_dict[hole_num] = trimesh.load(os.path.join(hole_folder, flname))
    hole_dict[hole_num].vertices = cs.convert_coordinate_system(
        hole_dict[hole_num].vertices, "ASR", "LPS"
    )  # Preserves shape!

model_implant_targets = {}
for ii, hole_id in enumerate(hole_dict.keys()):
    if hole_id < 0:
        continue
    model_implant_targets[hole_id] = hole_dict[hole_id].centroid

# %%
# If implant has holes that are segmented.
implant_vol = sitk.ReadImage(str(implant_file))
odict = {k: implant_vol.GetMetaData(k) for k in implant_vol.GetMetaDataKeys()}
label_dict = find_seg_nrrd_header_segment_info(odict)

implant_names = []
implant_targets = []
implant_pts = []

for ii, key in enumerate(label_dict.keys()):
    filt = sitk.EqualImageFilter()
    is_label = filt.Execute(implant_vol, label_dict[key])
    idxx = np.where(sitk.GetArrayViewFromImage(is_label))
    idx = np.vstack((idxx[2], idxx[1], idxx[0])).T
    implant_pos = np.vstack(
        [
            implant_vol.TransformIndexToPhysicalPoint(idx[ii, :].tolist())
            for ii in range(idx.shape[0])
        ]
    )
    implant_pts.append(implant_pos)
    implant_targets.append(np.mean(implant_pos, axis=0))
    this_key = key.split("-")[-1].split("_")[-1]
    implant_names.append(int(this_key))
implant_targets = np.vstack(implant_targets)

# %%
# Read the checmical shift from the image (assumes UW scanner as default)
chem_shift_trans = np.vstack(chemical_shift_transform(compute_chemical_shift(image)))
chem_shift_trans

# %%
# Fit implant 
implant_fit_trans = load_sitk_transform(implant_fit_transform_file)
transformed_implant = {}
for ii,key in enumerate(model_implant_targets.keys()):
    transformed_implant[key] = np.dot(append_ones_column(model_implant_targets[key]),implant_fit_trans)
    transformed_implant[key] = np.dot(append_ones_column(transformed_implant[key]),image_trans)


# %%
# # Read the Histology data

rabies_cell_pts =annotations_path/f"{mouse}_rabies_pts_from_698928_LPS.csv"
rabies_targets_pt = annotations_path/f"{mouse}_max_density_by_target_from_698928_LPS.csv"


rabies_cells = pd.read_csv(rabies_cell_pts)
rabies_cells = np.vstack([rabies_cells.x.values,rabies_cells.y.values,rabies_cells.z.values]).T
rabies_cells = rabies_cells[points_in_mask(brain_mask_path,rabies_cells),:]

rabies_targets = pd.read_csv(rabies_targets_pt)
rabies_target_names = rabies_targets.structure
rabies_targets = np.vstack([rabies_targets.L.values,rabies_targets.P.values,rabies_targets.S.values]).T


# And finally,plot the rabies data
chem_shift_rabies_cells = np.dot(append_ones_column(rabies_cells),chem_shift_trans)
trans_rabies_cells = np.dot(append_ones_column(chem_shift_rabies_cells),image_trans)
density_funciton = local_density_fixed_radius(trans_rabies_cells,.1)

chem_shift_rabies_targets = np.dot(append_ones_column(rabies_targets),chem_shift_trans)
trans_rabies_targets = np.dot(append_ones_column(chem_shift_rabies_targets),image_trans)
trans_rabies_targets



# %%
def arc_angles_to_hit_two_points(target_pt, extra_pt, ap_offset=14, degrees=True):
    """
    Compute the arc angles needed for a probe trajectory that intersects 2 points.

    Note that order matters on the points;
    currently "target" is the intended deep point and "extra" is a point at/above the surface.

    This should probably have some coordinate system awareness, and this documentation should be expanded to show logic.

    # Returns AP, ML angle
    """
    this_vector = (extra_pt - target_pt) / np.linalg.norm(extra_pt - target_pt)
    phi = np.arcsin(this_vector[0])
    theta = np.arcsin(-this_vector[1] / np.cos(phi))
    return np.rad2deg(theta) + ap_offset, -np.rad2deg(phi)

# Find the angle from each hole to each target.
target_locs = dict(zip(rabies_target_names,[trans_rabies_targets[ii,:] for ii in range(len(trans_rabies_targets))]))
hole_locs = transformed_implant

ap_angle = np.zeros([len(target_locs),len(hole_locs)])
ml_angle = np.zeros([len(target_locs),len(hole_locs)])
insertion_works = np.zeros([len(target_locs),len(hole_locs)])
TARGET = []
HOLE = []
RIG_AP = []
ML = []
AP = []


for ii,target in enumerate(target_locs.keys()):
    for jj, hole in enumerate(hole_locs.keys()):
        ap_angle[ii,jj],ml_angle[ii,jj] =arc_angles_to_hit_two_points(target_locs[target],hole_locs[hole])
        if np.abs(ml_angle[ii,jj])<40 and np.abs(ap_angle[ii,jj])<40:
            TARGET.append(target)
            HOLE.append(hole)
            RIG_AP.append(ap_angle[ii,jj])
            AP.append(ap_angle[ii,jj]-14)
            ML.append(ml_angle[ii,jj])
            insertion_works[ii,jj] =True
            
df = pd.DataFrame(
{
    "target": TARGET,
    "hole": HOLE,
    "rig_ap": RIG_AP,
    "ml": ML,
    "ap": AP,
})        

# %%
used_target_structure = ['PL','RSP','MD','VM','CLA','CA1','BLA']
structure_color_mapping = dict(zip(used_target_structure,[rgb_to_int(int(x[0]*255),int(x[1]*255),int(x[2]*255)) for x in get_colormap_colors(len(used_target_structure),colormap='rainbow')]))
structure_color_mapping

# %%
# Setup k3d plotting.
plot = k3d.plot()
plot.display()
plot.grid_visible= False

brain_color = hex_string_to_int('#EFC3CA')
headframe_color = hex_string_to_int('#E2EAF4')
well_color = hex_string_to_int('#E7DDFF')
implant_color = rgb_to_int(255,0,255)
target_color = rgb_to_int(255,255,0)

# Plot the brain mask
this_target_brain = brain_mask.copy()
this_target_brain.vertices = np.dot(
    append_ones_column(brain_mask.vertices), chem_shift_trans
)
this_target_brain.vertices = np.dot(
    append_ones_column(this_target_brain.vertices), image_trans
)
plot +=k3d.mesh(this_target_brain.vertices.astype(float),
                this_target_brain.faces,
                name = 'Brain',
                color = brain_color,
               opacity = .1)



#plot+=k3d.points(positions=trans_rabies_cells,point_size=.05,opacities=density_funciton**2,colors=density_funciton)

# Load implant mesh
implant_mesh = trimesh.Trimesh(vertices=implant_lps, faces=implant_faces[0])
implant_mesh.vertices = np.dot(append_ones_column(implant_mesh.vertices),implant_fit_trans)
implant_mesh.vertices = np.dot(append_ones_column(implant_mesh.vertices),image_trans)
plot +=k3d.mesh(implant_mesh.vertices.astype(float),
        implant_mesh.faces,
        name = 'Implant',
        color = implant_color)

for ii,target in enumerate(target_locs.keys()):
    if target not in used_target_structure:
        continue
    plot+=k3d.points(positions=target_locs[target],point_size=.5,opacitie = .8,colors = [structure_color_mapping[target]])
    for jj, hole in enumerate(hole_locs.keys()):
        if insertion_works[ii,jj]:
            plot+=k3d.line([target_locs[target],hole_locs[hole]],colors = [structure_color_mapping[target],structure_color_mapping[target]])

# %%

# %%
# implant_holes_annotation_file = r'Y:/ephys/persist/data/MRI/processed/836657/836657_ImplantHoles.seg.nrrd'
# mask = sitk.ReadImage(str(implant_holes_annotation_file))
# implant_annotation = mask_to_trimesh(mask)

# %%
## Special plot just for hole testing
# Setup k3d plotting.
plot = k3d.plot()
plot.display()
plot.grid_visible= False

cal_file = os.path.join(r'Y:\ephys\persist\data\probe_calibrations\CSVCalibrations',f'{df.Calibration[0]}')
cal,_,_ = fit_rotation_params_from_manual_calibration(cal_file,find_scaling = False)
R,T,S = cal[df['Probe #'][0]]
hole_locs_LPS = transform_probe_to_bregma(df[['X','Y','Z']].values/1000,R,T)
hole_locs_RAS = hole_locs_LPS.copy()
hole_locs_RAS[:,:2] = -hole_locs_RAS[:,:2]

brain_color = hex_string_to_int('#EFC3CA')
headframe_color = hex_string_to_int('#E2EAF4')
well_color = hex_string_to_int('#E7DDFF')
implant_color = rgb_to_int(255,0,255)
target_color = rgb_to_int(255,255,0)

# Plot the brain mask
this_target_brain = brain_mask.copy()
this_target_brain.vertices = np.dot(
    append_ones_column(brain_mask.vertices), chem_shift_trans
)
this_target_brain.vertices = np.dot(
    append_ones_column(this_target_brain.vertices), image_trans
)
plot +=k3d.mesh(this_target_brain.vertices.astype(float),
                this_target_brain.faces,
                name = 'Brain',
                color = brain_color,
               opacity = .1)

# Load implant mesh
implant_mesh = trimesh.Trimesh(vertices=implant_lps, faces=implant_faces[0])
implant_mesh.vertices = np.dot(append_ones_column(implant_mesh.vertices),implant_fit_trans)
implant_mesh.vertices = np.dot(append_ones_column(implant_mesh.vertices),image_trans)
plot +=k3d.mesh(implant_mesh.vertices.astype(float),
        implant_mesh.faces,
        name = 'Implant',
        color = implant_color)

# loat the implant annotations mesh



plot +=k3d.points(hole_locs_RAS,point_size = .1)
this_vertices = np.dot(append_ones_column(implant_annotation.vertices),image_trans)

plot +=k3d.mesh(this_vertices,
        implant_annotation.faces,
        name = 'Annotations',
        color = well_color)

for ii,target in enumerate(target_locs.keys()):
    if target not in used_target_structure:
        continue
    plot+=k3d.points(positions=target_locs[target],point_size=.5,opacitie = .8,colors = [structure_color_mapping[target]])
    for jj, hole in enumerate(hole_locs.keys()):
        if insertion_works[ii,jj]:
            plot+=k3d.line([target_locs[target],hole_locs[hole]],colors = [structure_color_mapping[target],structure_color_mapping[target]])

# %%
from aind_mri_utils.reticle_calibrations import find_probe_angle
find_probe_angle(R)

# %%

# %%
# Now lets check every pair of insertions to see if they are cross compatible.
ap_wiggle = 2
ml_wiggle = 1 
ap_min = 15
ml_min = 15
rotation_inc = 45

pair_works = np.zeros([len(df),len(df)])
for ii,row_i in df.iterrows():
    for jj,row_j in df.iterrows():
        # Matrix is symetrical, skip second triangle
        if ii == jj or ii < jj:
            continue
        # Don't need two probes to one target
        elif row_i.target == row_j.target:
           continue
        elif row_i.hole==0 and row_j.hole==0:
            continue
        # Can't have two probes in one hole
        elif row_i.hole == row_j.hole:
            continue
        # Structure is on our list of targets
        elif row_i.target not in used_target_structure or row_j.target not in used_target_structure:
            continue
        # If probes need to be on the same arc but overlap in ML
        elif (np.abs(row_i.ap - row_j.ap) < ap_wiggle) and (
            np.abs(row_i.ml - row_j.ml) < ml_min
        ):
            continue
        # If probes need to be on AP arcs that are not compatible.
        elif (np.abs(row_i.ap - row_j.ap) < ap_min) and (
            np.abs(row_i.ap - row_j.ap) > ap_wiggle
        ):
            
            continue
        else:
            pair_works[ii,jj] = True
            pair_works[jj,ii] = True

# %%
import networkx as nx
manditory_insertion_PL =1 # Choose the PL hole we want.
manditory_insertion_MD =21 # Choose the PL hole we want.

needed_probes = 6 # How many probes we need to fit together

G = nx.from_numpy_array(pair_works)
all_cliques = nx.enumerate_all_cliques(G)
#cliques = [c for c in all_cliques if len(c)>=needed_probes and (manditory_insertion_PL in c) and (manditory_insertion_MD in c) and not(53 in c)]
cliques = []
for c in all_cliques:
    if len(c)>=needed_probes:
        continue
    elif (manditory_insertion_PL not in c):
        continue
    elif (manditory_insertion_MD not in c):
        continue
    elif 5 in df.loc[c].hole.values:
        continue
    else:
        cliques.append(c)

print(f"Found {len(cliques)} compatible sets")

# %%

# %%
# Probe models for plotting.

probe_model_dict = {'2.0':load_newscale_model(modified_probe_mesh_file),
              '2.1':load_newscale_model(dovetail_tweezer_file),
              'quadbase0':load_newscale_model(quadbase_file_0),
             'quadbase_dovetail0':load_newscale_model(quadbase_dovetail_file_0),
             'quadbase3':load_newscale_model(quadbase_file_3),
             'quadbase_dovetail3':load_newscale_model(quadbase_dovetail_file_3),
             '2.4':load_newscale_model(dovetail_tweezer_4shank_file),
             'pipette':load_newscale_model(pipette_file)}

# %%
# #### Initialization!!!!

# arc_dict = {0:31,
#             1:6,
#             2:-10,
#             3:-28,
#            4:31-15}

# probe_info = {}
# probe_info['MD'] = {'arc':0,
#                    'ML':-10,
#                    'spin':135-180,
#                    'x_offset':-.15,
#                    'y_offset':.15,
#                    'probe_type': 'quadbase',
#                    'hole':3,
#                    'distance_past_target':1.5,
#                    'target_LPS':target_locs['MD']
#                    }

# probe_info['BLA'] = {'arc':1,
#                    'ML':27,
#                    'spin':50,
#                    'x_offset':.15,
#                    'y_offset':0,
#                    'probe_type': '2.1',
#                    'hole':4,
#                    'distance_past_target':1,
#                    'target_LPS':target_locs['BLA']
#                 }

# probe_info['PL'] = {'arc':1,
#                    'ML':-30,
#                    'spin':123,
#                    'x_offset':.05,# L 
#                    'y_offset':-.95,#P
#                    'probe_type': 'quadbase',
#                    'hole':1,
#                    'distance_past_target':.5,
#                    'target_LPS':target_locs['PL']
#                    }



# probe_info['VM'] = {'arc':2,
#                    'ML':-20,
#                    'spin':0,
#                    'x_offset':0,
#                    'y_offset':.2,
#                    'probe_type': '2.1',
#                    'hole':6,
#                    'distance_past_target':.5,
#                    'target_LPS':target_locs['VM']
#                    }

# probe_info['RSP'] = {'arc':2,
#                    'ML':22,
#                    'spin':-100,
#                    'x_offset':0,
#                    'y_offset':.2,
#                    'probe_type': '2.1',
#                    'hole':5,
#                    'distance_past_target':2,
#                    'target_LPS':hole_locs[5]
#                    }

# probe_info['CA1'] = {'arc':2,
#                    'ML':6,
#                    'spin':-180,
#                    'x_offset':-.6,
#                    'y_offset':-.2,
#                    'probe_type': '2.1',
#                    'hole':10,
#                    'distance_past_target':1.2,
#                    'target_LPS':target_locs['CA1']
#                    }

# probe_info['CLA'] = {'arc':3,
#                    'ML':-27,
#                    'spin':-180-15,
#                    'x_offset':.3,
#                    'y_offset':.3,
#                    'probe_type': '2.1',
#                    'hole':12,
#                    'distance_past_target':6,
#                    'target_LPS':hole_locs[12]
#                    }





# %%
## Option to load an existing 

def load_plan(csv_file):
    df = pd.read_csv(csv_file)
    #df = df[:-1]
    probe_info = {}
    arc_opts = {}
    for _, row in df.iterrows():
        s = row['structure']
        probe_info[s] ={}
        # assume probe_info already has this key
        if pd.isna(row['probe_type']):
            break
        probe_info[s]['probe_type']          = row['probe_type']
        probe_info[s]['arc']                 = int(row['ap_arc_id'])
        probe_info[s]['ML']                  = row['ml_angle']
        probe_info[s]['spin']                = row['spin']
        probe_info[s]['hole']                = row['hole']
        probe_info[s]['distance_past_target']= row['distance_past_target']

        # reconstruct target + offsets
        if 'ideal_pt_L' in row.keys(): # LPS
            probe_info[s]['target_LPS'] = [
                -row['ideal_pt_L'], -row['ideal_pt_P'], row['ideal_pt_S']
            ]
            probe_info[s]['x_offset']   = (-row['target_pt_L']) - (-row['ideal_pt_L'])
            probe_info[s]['y_offset']   = (-row['target_pt_P']) - (-row['ideal_pt_P'])
        else:
            probe_info[s]['target_LPS'] = [
                -row['ideal_pt_R'], -row['ideal_pt_A'], row['ideal_pt_S']
            ]
            probe_info[s]['x_offset']   = (-row['target_pt_R']) - (-row['ideal_pt_R'])
            probe_info[s]['y_offset']   = (-row['target_pt_A']) - (-row['ideal_pt_A'])
        arc_opts[row['ap_arc_id']] = row['ap_rig_angle']
        
    # refresh widgets & redraw scene
    
    # with log_out: print(f"Loaded plan from {csv_file}")
    return probe_info,arc_opts
    
csv_file = r"Y:\ephys\persist\data\MRI\processed\836656\836656_YB_GuiInsertionPlan_2026-05-05T12-21-04.csv"
probe_info,arc_dict = load_plan(csv_file)
print(probe_info)

# Load colors for the indicated targets
used_target_structure = list(probe_info.keys())
structure_color_mapping = dict(zip(used_target_structure,[rgb_to_int(int(x[0]*255),int(x[1]*255),int(x[2]*255)) for x in get_colormap_colors(len(used_target_structure),colormap='rainbow')]))


# %%
#probe_info['MD'] = probe_info['PL'].copy()
probe_info['BLA']['probe_type'] = 'quadbase_dovetail0'


# %%
annotations = sitk.ReadImage(r"Y:\ephys\persist\data\MRI\processed\836656\ccfv3\ccf_annotation_in_subject.nii.gz")
structure_df = pd.read_csv(r"C:\Users\yoni.browning\Downloads\adult_mouse_ccf_structures.csv")
structure_list = list(probe_info.keys())
structure_mesh_dict = {}
#structure_color_dict = {}
for structure in structure_list:
    mesh,_,_ = generate_sitk_trimesh_from_ccf(structure,annotations,structure_df)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    mesh.vertices = np.dot(append_ones_column(mesh.vertices), chem_shift_trans)
    mesh.vertices = np.dot(append_ones_column(mesh.vertices), image_trans)
    structure_mesh_dict[structure] = mesh


# %%

# %%


# ---------------- GUI + Plot + Collision Checks (fixed “+=”) --------------
# Assumes all helper functions, meshes, and dicts are already defined.
import ipywidgets as widgets
from IPython.display import display, clear_output
import k3d, trimesh, numpy as np, SimpleITK as sitk
from datetime import datetime

RED = rgb_to_int(255, 0, 0)


# A dict of any candidate targets you want to pick from
#    key →  [R, A, S]  (same units as target_LPS)
potential_targets = {}
for key in target_locs.keys():
    potential_targets[key]= target_locs[key]
for key in hole_locs.keys():
    potential_targets[str(key)]= hole_locs[key]

# Keep a mutable “current angle” for each arc (start = design value)
arc_angle = {arc_id: ang for arc_id, ang in arc_dict.items()}

# --------------------------------------------------------------------------
# 1.  Pre-compute static structure meshes (only once)
# --------------------------------------------------------------------------
# structure_mesh_dict = {}
# for struct in probe_info:
#     if struct not in structure_mesh_dict:
#         mask = sitk.ReadImage(structure_mask_path / f"{mouse}-{struct}-Mask.nrrd")
#         m = mask_to_trimesh(mask > 0)
#         m.vertices = np.dot(append_ones_column(m.vertices), chem_shift_trans)
#         m.vertices = np.dot(append_ones_column(m.vertices), image_trans)
#         trimesh.repair.fix_normals(m)
#         trimesh.repair.fix_inversion(m)
#         structure_mesh_dict[struct] = m

# --------------------------------------------------------------------------
# 2.  Build the K3D scene (static objects first)
# --------------------------------------------------------------------------
plot = k3d.plot(grid_visible=False)


this_target_brain = brain_mask.copy()
this_target_brain.vertices = np.dot(
    append_ones_column(brain_mask.vertices), chem_shift_trans
)
this_target_brain.vertices = np.dot(
    append_ones_column(this_target_brain.vertices), image_trans
)
brain_h = k3d.mesh(
    this_target_brain.vertices.astype(float), brain_mask.faces,
    name='Brain', color=rgb_to_int(239, 195, 202), opacity=0.1
)
plot += brain_h

headframe_h = k3d.mesh(
    headframe_mesh.vertices.astype(float), headframe_mesh.faces,
    name='Headframe', color=rgb_to_int(226, 234, 244)
)
plot += headframe_h

# cone_h = k3d.mesh(
#     cone_mesh.vertices.astype(float), cone_mesh.faces,
#     name='Cone', color=rgb_to_int(255, 200, 200)
# )
# plot += cone_h

#plot += k3d.points(hole_locs_RAS)

well_h = k3d.mesh(
    well_mesh.vertices.astype(float), well_mesh.faces,
    name='Well', color=rgb_to_int(231, 221, 255),shader = 'mesh'
)
plot += well_h

# Load implant mesh
implant_mesh = trimesh.Trimesh(vertices=implant_lps, faces=implant_faces[0])
implant_mesh.vertices = np.dot(append_ones_column(implant_mesh.vertices),implant_fit_trans)
implant_mesh.vertices = np.dot(append_ones_column(implant_mesh.vertices),image_trans)

implant_h = k3d.mesh(
    implant_mesh.vertices.astype(float), implant_mesh.faces,
    name='Implant', color=rgb_to_int(255, 0, 255)
)
plot += implant_h

# structure masks (semi-transparent)
structure_handle_dict = {}
for s, m in structure_mesh_dict.items():
    mh = k3d.mesh(
        m.vertices.astype(float), m.faces,
        name=s, color=structure_color_mapping[s], opacity=0.4
    )
    plot += mh
    structure_handle_dict[s] = mh

# --------------------------------------------------------------------------
# 3.  Create probe and target-sphere handles
# --------------------------------------------------------------------------
probe_handles   = {}
sphere_handles  = {}

for s, p in probe_info.items():
    # initial probe
    probe_mesh = move_and_copy_newscale_model(
        probe_model_dict[p['probe_type']], move_down=p['distance_past_target']
    )
    TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(-p['spin']))
    T1 = transform_matrix_from_angles_and_target(
        arc_dict[p['arc']]-14, -p['ML'],
        [p['target_LPS'][0] + p['x_offset'],
         p['target_LPS'][1] + p['y_offset'],
         p['target_LPS'][2]]
    )
    apply_transform_to_mesh(probe_mesh, TA)
    apply_transform_to_mesh(probe_mesh, T1)

    ph = k3d.mesh(
        probe_mesh.vertices.astype(float),
        probe_mesh.faces.astype(np.uint32),
        name=f"Probe_{s}",
        color=structure_color_mapping[s],
        side="double",
        shader = 'mesh',
    )
    plot += ph
    probe_handles[s] = ph

    # target sphere
    sphere = trimesh.creation.uv_sphere(radius=0.25)
    sphere.apply_translation([
        p['target_LPS'][0],
        p['target_LPS'][1],
        p['target_LPS'][2]
    ])
    sh = k3d.mesh(sphere.vertices.astype(float), sphere.faces.astype(np.uint32),
                  name=f"Target_{s}", color=structure_color_mapping[s])
    plot += sh
    sphere_handles[s] = sh

# Add the anatomy data.
plot+=k3d.points(trans_rabies_cells,point_size=.01,opacities=.2,colors=density_funciton)

plot.display()
# --------------------------------------------------------------------------
# 4.  Widgets
# --------------------------------------------------------------------------
# Widgets (fixed syntax)
structure_dd = widgets.Dropdown(options=list(probe_info.keys()),
                                description='Structure:')

arc_dd = widgets.Dropdown(options=[(f"{k}: {v}°", k) for k, v in arc_dict.items()],
                          description='Arc:')

ml_slider = widgets.FloatSlider(value=0, min=-50, max=50, step=0.25,
                                description='ML (mm):', continuous_update=False)

spin_slider = widgets.IntSlider(value=0, min=-180, max=180, step=1,
                                description='Spin (°):', continuous_update=False)

xoff_slider = widgets.FloatSlider(value=0, min=-1.5, max=1.5, step=0.01,
                                  description='X off (mm):', continuous_update=False)

yoff_slider = widgets.FloatSlider(value=0, min=-1.5, max=1, step=0.01,
                                  description='Y off (mm):', continuous_update=False)

depth_slider = widgets.FloatSlider(value=0, min=-5, max=10, step=0.1,
                                   description='Depth (mm):', continuous_update=False)

update_btn = widgets.Button(description='Update', button_style='success')
save_btn = widgets.Button(description='💾 Save plan → CSV', button_style='info')
snap_btn = widgets.Button(description='Snap View', button_style='success')


log_out    = widgets.Output(layout={'border': '1px solid lightgray'})




# --- Arc ----------------------------------------------------------
target_dd = widgets.Dropdown(description='Target:', layout={'width': '250px'})



arc_sliders = {}
for arc_id, ang in arc_angle.items():
    arc_sliders[arc_id] = widgets.FloatSlider(
        value=ang, min=-60, max=60, step=0.5,
        description=f"Arc {arc_id} angle (°)",
        continuous_update=False
    )
# tidy container (accordion so it can be collapsed)
arc_accordion = widgets.Accordion(
    children=[widgets.VBox(list(arc_sliders.values()))]
)
arc_accordion.set_title(0, 'Arc Angles (absolute)')

def _load_into_widgets(change):
    p = probe_info[change['new']]
    arc_dd.value       = p['arc']
    ml_slider.value    = p['ML']
    spin_slider.value  = p['spin']
    xoff_slider.value  = p['x_offset']
    yoff_slider.value  = p['y_offset']
    depth_slider.value = p['distance_past_target']

    # populate target dropdown: keys that start with structure name (or all)
    opts = {k: k for k in potential_targets}
    target_dd.options = opts
    # pre-select whichever target matches this probe
    current = next((k for k, v in potential_targets.items()
                    if np.allclose(v, p['target_LPS'])), None)
    target_dd.value = current

structure_dd.observe(_load_into_widgets, names='value')
_load_into_widgets({'new': structure_dd.value})

# --------------------------------------------------------------------------
# 5.  Update & collision check
# --------------------------------------------------------------------------
def update_scene(*_):
    key = structure_dd.value
    
    probe_info[key].update({
        'arc' : arc_dd.value,
        'ML'  : ml_slider.value,
        'spin': spin_slider.value,
        'x_offset': xoff_slider.value,
        'y_offset': yoff_slider.value,
        'distance_past_target': depth_slider.value,
    })
    # NEW: switch target if user picked one
    if target_dd.value is not None:
        probe_info[key]['target_LPS'] = potential_targets[target_dd.value]

    for aid, sld in arc_sliders.items():
        arc_angle[aid] = sld.value

    arc_dd.options = [(f"{aid}: {arc_angle[aid]:.1f}°", aid) for aid in arc_angle]

    implantCM = trimesh.collision.CollisionManager()
    probeCM   = trimesh.collision.CollisionManager()
    #coneCM    = trimesh.collision.CollisionManager()
    wellCM    = trimesh.collision.CollisionManager()
    implantCM.add_object('implant', implant_mesh)
    #coneCM.add_object('cone', cone_mesh)
    wellCM.add_object('well', well_mesh)

    with log_out:
        clear_output()
        for struct, params in probe_info.items():
            # regenerate probe
            mesh = move_and_copy_newscale_model(
                probe_model_dict[params['probe_type']],
                move_down=params['distance_past_target']
            )
            TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(-params['spin']))
            T1 = transform_matrix_from_angles_and_target(
                arc_angle[params['arc']] - 14,          # <-- was arc_dict[…]-14
                -params['ML'],
                [params['target_LPS'][0] + params['x_offset'],
                 params['target_LPS'][1] + params['y_offset'],
                 params['target_LPS'][2]]
            )
            apply_transform_to_mesh(mesh, TA)
            apply_transform_to_mesh(mesh, T1)

            # update vertices only
            probe_handles[struct].vertices = mesh.vertices.astype(float)

            # target sphere
            tgt = trimesh.creation.uv_sphere(radius=0.25)
            tgt.apply_translation([params['target_LPS'][0],
                                   params['target_LPS'][1],
                                   params['target_LPS'][2]])
            sphere_handles[struct].vertices = tgt.vertices.astype(float)

            # collision managers
            probeCM.add_object(struct, mesh)
            implantCM.add_object(struct, mesh)
            #coneCM.add_object(struct, mesh)
            wellCM.add_object(struct, mesh)

            structureCM = trimesh.collision.CollisionManager()
            structureCM.add_object("probe", mesh)
            structureCM.add_object("structure", structure_mesh_dict[struct])

            pointCM = trimesh.collision.CollisionManager()
            pointCM.add_object("probe", mesh)
            pointCM.add_object("point", tgt)

            hit_struct = structureCM.in_collision_internal()
            hit_point  = pointCM.in_collision_internal()
            print(f"{struct:>6}: "
                  f"{'Probe→struct HIT' if hit_struct else 'miss'}, "
                  f"{'Probe↔point HIT'  if hit_point  else 'miss'}")

        # global reports
        if probeCM.in_collision_internal():
            _, names = probeCM.in_collision_internal(return_names=True)
            print("\nPROBE–PROBE collisions:", list(names))
        else:
            print("\nNo probe–probe collisions.")

        def _flag(cm, label, handle, ok_color):
            if cm.in_collision_internal():
                print(f"Probes collide with {label}!")
                handle.color = RED
            else:
                print(f"Probes clear {label}.")
                handle.color = ok_color

        _flag(implantCM, "implant", implant_h, rgb_to_int(255, 0, 255))
        #_flag(coneCM,    "cone",    cone_h,    rgb_to_int(255, 200, 200))
        _flag(wellCM,    "well",    well_h,    rgb_to_int(231, 221, 255))

update_btn.on_click(update_scene)
update_scene()


def _save_plan(_):
    csv_path = annotations_path / f'{mouse}_{current_user}_GuiInsertionPlan_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.csv'
    html_path = annotations_path / f'{mouse}_{current_user}_GuiInsertionPlan_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.html'
    df_dict = {              # build exactly the same dict you posted
        'structure':[], 'probe_type':[], 'ap_arc_id':[], 'ap_angle':[],
        'ap_rig_angle':[], 'ml_angle':[], 'spin':[],
        'target_pt_R':[], 'target_pt_A':[], 'target_pt_S':[],
        'ideal_pt_R':[],  'ideal_pt_A':[],  'ideal_pt_S':[],
        'hole':[], 'distance_past_target':[]
    }
    for struct, p in probe_info.items():
        df_dict['structure'].append(struct)
        df_dict['probe_type'].append(p['probe_type'])
        df_dict['ap_arc_id'].append(p['arc'])
        df_dict['ap_angle'].append(arc_angle[p['arc']] - 14)
        df_dict['ap_rig_angle'].append(arc_angle[p['arc']])
        df_dict['ml_angle'].append(p['ML'])
        df_dict['spin'].append(p['spin'])
        df_dict['target_pt_R'].append(-(p['target_LPS'][0] + p['x_offset']))
        df_dict['target_pt_A'].append(-(p['target_LPS'][1] + p['y_offset']))
        df_dict['target_pt_S'].append(p['target_LPS'][2])
        df_dict['ideal_pt_R'].append(-p['target_LPS'][0])
        df_dict['ideal_pt_A'].append(-p['target_LPS'][1])
        df_dict['ideal_pt_S'].append(p['target_LPS'][2])
        df_dict['hole'].append(p['hole'])
        df_dict['distance_past_target'].append(p['distance_past_target'])

    pd.DataFrame(df_dict).to_csv(csv_path, index=False)
    with log_out:
        print(f"Saved current plan → {csv_path}")
    with open(html_path, 'w') as f:
        f.write(plot.get_snapshot())

save_btn.on_click(_save_plan)

def _on_snap_button_click(_):
    plot.camera=[15.128249175465996, 1.8669338948268699, -5.361631178296464, -23.59602346805174, 3.299589364138823, 6.923953645591561, 0.28672420267420967, -0.0827001419484639, 0.9544369639334744]

snap_btn.on_click(_on_snap_button_click)
# --------------------------------------------------------------------------
# 6.  Display GUI
# --------------------------------------------------------------------------

buttons = widgets.HBox([update_btn,save_btn,snap_btn,])
controls = widgets.VBox([
    structure_dd, 
    target_dd,      
    arc_dd,
    ml_slider, spin_slider,
    xoff_slider, yoff_slider, depth_slider,
    buttons,
    log_out,
    arc_accordion                   
])
display(controls)
_on_snap_button_click(_)

# %%
structure_mesh_dict

# %%
_save_plan(_)


# %%
### FIND THE MIDDLE OF THE PL PROBE IN PL- this is, effectivly, the ideal injection location

def edge_face_intersections(mesh_source: trimesh.Trimesh,
                            mesh_target: trimesh.Trimesh,
                            tol: float = 1e-8):
    """
    Find all intersection points where edges of mesh_source intersect faces of mesh_target.

    Returns
    -------
    points : (M,3) float
        The XYZ locations of each intersection.
    edge_indices : (M,) int
        For each intersection, the index (into mesh_source.edges_unique) of the edge that hit.
    face_indices : (M,) int
        For each intersection, the index (into mesh_target.faces) of the triangle that was hit.
    """
    # 1. Get unique edges (pairs of vertex indices)
    edges = mesh_source.edges_unique
    verts = mesh_source.vertices

    # 2. Build ray origins, directions, and segment lengths
    start_pts = verts[edges[:, 0]]
    end_pts   = verts[edges[:, 1]]
    vecs      = end_pts - start_pts
    lengths  = np.linalg.norm(vecs, axis=1)
    nonzero  = lengths > tol

    origins    = start_pts[nonzero]
    directions = vecs[nonzero] / lengths[nonzero][:, None]
    max_dist   = lengths[nonzero]

    # 3. Ray-mesh intersection: returns all hits (including those beyond segment)
    #    If you have PyEmbree installed this will be very fast; otherwise it will
    #    fall back to the slower Python implementation.
    locations, ray_idx, tri_idx = mesh_target.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True)

    # 4. Filter to keep only those hits within the segment length
    #    Compute along-ray distance t for each hit:
    vec_hit = locations - origins[ray_idx]
    t_hit   = np.einsum('ij,ij->i', vec_hit, directions[ray_idx])
    valid   = (t_hit >= -tol) & (t_hit <= max_dist[ray_idx] + tol)

    locations   = locations[valid]
    ray_idx     = ray_idx[valid]
    tri_idx     = tri_idx[valid]

    # 5. Map back to the original edge indices
    edge_idx = np.nonzero(nonzero)[0][ray_idx]

    return locations, edge_idx, tri_idx
locations,_,_ = edge_face_intersections(this_probe_mesh,this_target_mesh)


PL_mesh = structure_mesh_dict['PL']

p = probe_info['PL']
probe_mesh = move_and_copy_newscale_model(
    probe_model_dict[p['probe_type']], move_down=p['distance_past_target']
)
TA = trimesh.transformations.euler_matrix(0, 0, np.deg2rad(-p['spin']))
T1 = transform_matrix_from_angles_and_target(
    arc_dict[p['arc']]-14, -p['ML'],
    [p['target_LPS'][0] + p['x_offset'],
     p['target_LPS'][1] + p['y_offset'],
     p['target_LPS'][2]]
)
apply_transform_to_mesh(probe_mesh, TA)
apply_transform_to_mesh(probe_mesh, T1)

loc,_,_ = edge_face_intersections(probe_mesh,PL_mesh)
print('Injection_location (LPS):' + str(np.mean(loc,0)))


# %%
plot.camera=[15.128249175465996, 1.8669338948268699, -5.361631178296464, -23.59602346805174, 3.299589364138823, 6.923953645591561, 0.28672420267420967, -0.0827001419484639, 0.9544369639334744]

# %%
hole_locs[2]


# %%
# Find the injection angle for the specified point.
def arc_angles_to_hit_two_points(target_pt, extra_pt, ap_offset=14, degrees=True):
    """
    Compute the arc angles needed for a probe trajectory that intersects 2 points.

    Note that order matters on the points;
    currently "target" is the intended deep point and "extra" is a point at/above the surface.

    This should probably have some coordinate system awareness, and this documentation should be expanded to show logic.

    # Returns AP, ML angle
    """
    this_vector = (extra_pt - target_pt) / np.linalg.norm(extra_pt - target_pt)
    phi = np.arcsin(this_vector[0])
    theta = np.arcsin(-this_vector[1] / np.cos(phi))
    return np.rad2deg(theta) + ap_offset, -np.rad2deg(phi)
    
injection_location = np.mean(loc,0)
arc_angles_to_hit_two_points(injection_location,hole_locs[2])

# %%
import ipywidgets as widgets, json
fu = widgets.FileUpload(accept='.csv')
def dbg(change):
    print("event fired! type:", type(change['new']), "length:", len(change['new']))
fu.observe(dbg, names='value')
display(fu)

# %%
from datetime import datetime


# %%
