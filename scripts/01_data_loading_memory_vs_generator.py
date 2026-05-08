#!/usr/bin/env python3
# Exported from notebooks/01_data_loading_memory_vs_generator.ipynb
# This script is a linearized version of the notebook for code review and portfolio browsing.
# Some cells may still require a notebook/runtime environment, downloaded data, or trained model artifacts.

import asyncio


# %% [markdown] cell 1
# <div style="text-align: center;">
#   <a href="https://cognitiveclass.ai/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/image/IDSN-logo.png" width="400">
#   </a>
# </div>
#

# %% [markdown] cell 2
# <h1 align=left><font size = 5>Compare Memory-Based versus Generator-Based Data Loading</font></h1>

# %% [markdown] cell 3
# <h5>Estimated time: 30 minutes</h5>

# %% [markdown] cell 4
# <h2>Learning objective</h2>
# After completing this lab, you'll be able to:
#
# - Download and manage satellite image datasets using Keras and Python libraries
# - Compare memory-based and generator-based data loading strategies in terms of performance, memory efficiency, and implementation complexity
# - Build a simple, memory-efficient image pipeline by loading and visualizing geospatial image data sequentially
# - Assess the trade-offs between loading entire image datasets into memory versus accessing image paths on demand

# %% [markdown] cell 5
# ## Introduction

# %% [markdown] cell 6
# Geospatial data analysis is a highly competitive and growing sector. It is used for land cover mapping, building roads, detection, and temporal land usage monitoring, among other applications. In this lab, you will learn to efficiently build a classifier by training a model from a curated dataset. For problem formulation, you will use images to denote agricultural land vs. non-agricultural land.
#
# Additionally, you will learn to download a dataset from the cloud. The notebook will illustrate two common ways to work with image datasets in Python: Loading file paths and opening images one by one, which is sequential loading and holding all images in memory at once, which is bulk loading. You will learn how each approach affects code simplicity, memory use, and I/O performance.
#
# **Note:** Throughout this lab, you’ll answer strategically placed questions designed to reinforce your learning and complete the quiz.

# %% [markdown] cell 7
# ## Table of contents
#
# <font size = 3>    
#
# 1. [Import libraries and packages](#Import-libraries-and-packages)
# 2. [Download data](#Download-data)
# 3. [Load images](#Load-images)
#
# </font>
# </div>

# %% [markdown] cell 8
# ### Install the required libraries
#
# All the required libraries are __not__ pre-installed in the Skills Network Labs environment. __You need to run the following cell__ to install them, and this might take a few minutes.

# %% [code] cell 9
# Notebook-only command: %%capture captured_output
# Notebook-only command: !pip install numpy==1.26
# Notebook-only command: !pip install matplotlib==3.9.2
# Notebook-only command: !pip install skillsnetwork

# %% [code] cell 10
output_text = captured_output.stdout
lines = output_text.splitlines()
output_last_10_lines = '\n'.join(lines[-10:])
if "error" in output_last_10_lines.lower():
    print("Library installation failed!")
    print("--- Error Details ---")
    print(output_last_10_lines)
else:
    print("Library installation was successful, let's proceed ahead")

# %% [markdown] cell 11
# ## Import libraries and packages

# %% [markdown] cell 12
# Let's begin by importing the libraries and packages that you will need to complete the rest of this lab.

# %% [code] cell 13
import os
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork

from PIL import Image

# %% [markdown] cell 14
# ## Download data

# %% [markdown] cell 15
# The data is on a cloud server. You can retrieve and unzip it easily using the **skillsnetwork.prepare** command. So, let's run the following line of code to download the data. 

# %% [code] cell 16
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5vTzHBmQUaRNJQe5szCyKw/images-dataSAT.tar"

extraction_path = "."
asyncio.run(skillsnetwork.prepare(url = url, path = extraction_path, overwrite = True))

# %% [markdown] cell 17
# Now, you will be able to see the **images_dataSAT** folder in the left pane. It has two folders  **class_0_non_agri** and **class_1_agri**.
# The folder structure looks as follows:
#
# ```python
# images_dataSAT/
# ├── class_0_non_agri/
# └── class_1_agri/
#
# ```
#
# <table>
#     <tr>
#         <td style="text-align:center;"><b>Primary folder</b></td>
#         <td style="text-align:center;"><b>Subfolders</b></td>
#     </tr>
#     <tr>
#         <td><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/owykkC4Pr2zxLtU6vskQ5A/DL0321EN-M1L1-file-tree-Screenshot-1.png" style="width:300px; border:0px solid black;"></td>
#         <td><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/okqnadJpvAeedGUXXYBIFg/DL0321EN-M1L1-file-tree-Screenshot-2.png" style="width:350px; border:0px solid black;"></td>
#     </tr>
# </table>
#
#
# **class_0_non_agri** is the non-agricultural land class, as defined earlier, and it represents images with non-cultivable land. 
#
# **class_1_agri**, on the other hand, is the agricultural land class, and it represents the images with cultivable land.
#
#
#
# <table>
#     <tr>
#         <td style="text-align:center;"><b>class_0_non_agri</b></td>
#         <td style="text-align:center;"><b>class_1_agri</b></td>
#     </tr>
#     <tr>
#         <td><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1jSl6X5tUkVro8I_av8lEQ/DL0321EN-M1L1-file-tree-screenshot-3.png" style="width:300px; border:1px solid black;"></td>
#         <td><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/9f7sT5DBeFE_6Mp2OV3JKQ/DL0321EN-M1L1-file-tree-screenshot-4.png" style="width:300px; border:1px solid black;"></td>
#     </tr>
# </table>

# %% [markdown] cell 18
# **Note:** There are a few thousand images in each folder, so don't double-click on the folders. This may consume all of your memory, and you may end up with a **50*** error.

# %% [markdown] cell 19
# ## Load images

# %% [markdown] cell 20
# Next, you will use the standard approach of loading all images in memory and demonstrate how this approach is not efficient when it comes to building deep learning models for classifying images.

# %% [code] cell 21
# Define directories
extract_dir = "."

base_dir = os.path.join(extract_dir, 'images_dataSAT')
dir_non_agri = os.path.join(base_dir, 'class_0_non_agri')
dir_agri = os.path.join(base_dir, 'class_1_agri')

# %% [markdown] cell 22
# Let's start by reading in the non_agri images. First, you will use **os.scandir** to build an iterator to iterate through the *./images_dataSAT/class_0_non_agri* directory, which contains all the images with non-agriculture land.
# Display the first 5 entries in the non_agri list 

# %% [code] cell 23
non_agri = os.scandir(dir_non_agri)
# print first 5 file paths
for f_path in range(5):
    print(next(non_agri))

# %% [markdown] cell 24
# Then, you will grab the first file in the directory.

# %% [code] cell 25
file_name = next(non_agri)
file_name

# %% [markdown] cell 26
# Since the directory can contain elements that are not files, you only need to read the element if it is a file.

# %% [code] cell 27
os.path.isfile(file_name)

# %% [markdown] cell 28
# Get the image name.

# %% [code] cell 29
image_name = str(file_name).split("'")[1]
image_name

# %% [markdown] cell 30
# Read within the image data.

# %% [code] cell 31
image_data = plt.imread(os.path.join(dir_non_agri, image_name))
image_data

# %% [markdown] cell 32
# ### **Question 1**: What are the dimensions of a single image according to **image_data**? 

# %% [code] cell 33
## You can use this cell to type the code to answer the question.

print(image_data.shape)

# %% [markdown] cell 34
# Double-click **here** for the solution.
# <!--
# print(image_data.shape)
# -->

# %% [markdown] cell 35
# Let's take a look at the image.

# %% [code] cell 36
plt.imshow(image_data)

# %% [markdown] cell 37
# Now that you are familiar with the process of reading image data, let's loop through all the images in the *./images_dataSAT/class_0_non-agri* directory, read them all, and save them in the list **non_agri_images**. You will also need to note how long it takes to read all the images.

# %% [code] cell 38
# Notebook-only command: %%time

non_agri_images = []
for file_name in non_agri:
    if os.path.isfile(file_name):
        image_name = str(file_name).split("'")[1]
        image_data = plt.imread(os.path.join(dir_non_agri, image_name))
        non_agri_images.append(image_data)
    
non_agri_images = np.array(non_agri_images)

# %% [markdown] cell 39
# Loading images into memory is not the right approach when working with images, as it takes a long time or can quickly exhaust memory and other resources. Therefore, let's repeat the previous process but save the paths to the images in a variable instead of loading and saving the individual images.

# %% [markdown] cell 40
# So, instead of using **os.scandir**, let's use **os.listdir**.

# %% [code] cell 41
non_agri_images = os.listdir(dir_non_agri)
# print first 5 file paths
non_agri_images[:5]

# %% [markdown] cell 42
# Notice how the images are not sorted, so let's call the <code>sort</code> method to sort the images.

# %% [code] cell 43
non_agri_images.sort()

# print first 5 file paths
non_agri_images[:5]

# %% [markdown] cell 44
# Before you can show an image, you need to open it. You can do this by using the **Image** module in the **PIL** library. To open the first image, run the following:

# %% [code] cell 45
image_data = Image.open(os.path.join(dir_non_agri, non_agri_images[0]))

# %% [markdown] cell 46
# To view the image, run:

# %% [code] cell 47
image_data

# %% [markdown] cell 48
# or use the <code>plt.imshow()</code> method as follows:

# %% [code] cell 49
plt.imshow(image_data)

# %% [markdown] cell 50
# Let's loop through all the images in the <code>'./images_dataSAT/class_0_non_agri/'</code> directory and add their paths.

# %% [code] cell 51
non_agri_images_paths = [os.path.join(dir_non_agri, image) for image in non_agri_images]
#print first five paths
non_agri_images_paths[:5]

# %% [markdown] cell 52
# Let's check how many images of non-agricultural land exist in the dataset.

# %% [code] cell 53
len(non_agri_images_paths)

# %% [markdown] cell 54
# ### Question 2: Display the first four images in `'./images_dataSAT/class_0_non_agri/'` directory.

# %% [code] cell 55
## You can use this cell to type the code to answer the question.
n_images = 4
for image_path in non_agri_images[:n_images]:
    print(image_path)
    image_data = Image.open(os.path.join(dir_non_agri, image_path))
    plt.imshow(image_data)
    plt.show()

# %% [markdown] cell 56
# Double-click **here** for the solution.
# <!--
# n_images = 4
# for image_path in non_agri_images[:n_images]:
#     print(image_path)
#     image_data = Image.open(os.path.join(dir_non_agri, image_path))
#     plt.imshow(image_data)
#     plt.show()
# -->

# %% [markdown] cell 57
# ### **Task 1**: 
# Save the paths to all the images in the `dir_agri` directory in a list called **agri_images_paths**. Make sure you sort the paths at the end.

# %% [code] cell 58
## Type your answer here

agri_images_paths = []
for image in os.listdir(dir_agri):
    agri_images_paths.append(os.path.join(dir_agri, image))

agri_images_paths.sort()

# %% [markdown] cell 59
# Double-click **here** for the solution.
# <!--
# agri_images_paths = []
# for image in os.listdir(dir_agri):
#     agri_images_paths.append(os.path.join(dir_agri, image))
#
# agri_images_paths.sort()
# -->

# %% [markdown] cell 60
# ### Question 3: How many images of agricultural land exist in the <code>'./images_dataSAT/class_1_agri/'</code> directory?

# %% [code] cell 61
## You can use this cell to type the code to answer the question.

print(len(agri_images_paths))

# %% [markdown] cell 62
# Double-click **here** for the solution.
# <!--
# print(len(agri_images_paths))
# -->

# %% [markdown] cell 63
# ### Question 4: Display the first four images of the agricultural land.

# %% [code] cell 64
## You can use this cell to type the code to answer the question.

n_images = 4
for image_path in agri_images_paths[:n_images]:
    print(image_path)
    image_data = Image.open(image_path)
    plt.imshow(image_data)
    plt.show()

# %% [markdown] cell 65
# Double-click **here** for the solution.
# <!--
# n_images = 4
# for image_path in agri_images_paths[:n_images]:
#     print(image_path)
#     image_data = Image.open(image_path)
#     plt.imshow(image_data)
#     plt.show()
# -->

# %% [markdown] cell 66
# ## Save and download the notebook for **final project** submission and evaluation
#
# You will need to save and download the completed notebook for final project submission and evaluation. 
# <br>For saving and downloading the completed ntoebook, please follow the steps given below:</br>
#
# <font size = 4>  
#
# 1) **Complete** all the tasks and questions given in the notebook.
#
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/nv4jHlPU5_R1q7ZJrZ69eg/DL0321EN-M1L1-Save-IPYNB-Screenshot-1.png" style="width:600px; border:0px solid black;">
#
# 2) **Save** the notebook.</style>
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/9-WPWD4mW1d-RV5Il5otTg/DL0321EN-M1L1-Save-IPYNB-Screenshot-2.png" style="width:600px; border:0px solid black;">
#
# 3) Identify and right click on the **correct notebook file** in the left pane.</style>
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/RUSRPw7NT6Sof94B7-9naQ/DL0321EN-M1L1-Save-IPYNB-Screenshot-3.png" style="width:600px; border:0px solid black;">
#
# 4) Click on **Download**.</style>
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/HHry4GT-vhLEcRi1T_LHGg/DL0321EN-M1L1-Save-IPYNB-Screenshot-4.png" style="width:600px; border:0px solid black;">
#
# 5) Download and **Save** the Jupyter notebook file on your computer **for final submission**.</style>
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hhsJbxc6R-T8_pXQGjMjvg/DL0321EN-M1L1-Save-IPYNB-Screenshot-5.png" style="width:600px; border:0px solid black;">
#   </font>

# %% [markdown] cell 67
# ### Congratulations!
# You have successfully completed this lab. 

# %% [markdown] cell 68
# ### Summary
#
# This notebook shows a memory-efficient and simple sequential loading pattern by sorting and indexing filenames and then opening images one by one. To compare, you can try a bulk-loading approach by reading all images into a single list of arrays to observe the trade-off between faster repeated access and higher memory usage.

# %% [markdown] cell 69
# ### Thank you for completing this lab!
#
# This notebook is part of an IBM course on **Coursera** called *AI Capstone Project with Deep Learning*.

# %% [markdown] cell 70
# <h2>Author</h2>
#
# [Aman Aggarwal](https://www.linkedin.com/in/aggarwal-aman)
#
# Aman Aggarwal is a PhD working at the intersection of neuroscience, AI, and drug discovery. He specializes in quantitative microscopy and image processing.

# %% [markdown] cell 71
# <!--
# ## Change Log
#
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2025-06-25  | 1.0  | Aman  |  Created the lab |
# | 2025-06-30  | 2.0  | Gagandeeep |  ID review |
#
# -->

# %% [markdown] cell 72
# © Copyright IBM Corporation. All rights reserved.
