#!/usr/bin/env python
# coding: utf-8

# In[8]:

# In[1]:


import pandas as pd
import os
import re
from tqdm import tqdm
from pytube import YouTube

df = pd.read_excel("./final_Aus_Open_2021to2023_youtube (1).xlsx")
df2 = pd.read_excel("./final_French_Open_2021to2023_youtube.xlsx")


# # Basics

# In[2]:


def show_folder():
    for i in df["name"].unique():
        print(i)
show_folder()


# In[3]:


df


# In[ ]:


df2


# # Create Folder

# In[4]:


def title_preprocessing(string):
    match = re.search(r"\|([\w\s]+)$", string)

    if match:
        result = match.group(1).strip()
        print(result)
    else:
        print("No match found.")
    return result


# In[5]:


df["title"] = df["title"].apply(title_preprocessing)
df


# In[7]:


def create_folder():
    for i in sorted(df["title"].unique()):
        print(i)
        folder_name = 'games/'+i
        os.mkdir(folder_name)
    print("Folder created successfully!")
    print(len(df["title"].unique()))
create_folder()


# In[8]:


df_sorted = df.sort_values("title").reset_index(drop=True)
df_sorted


# # Download

# In[17]:


# In[9]:


from tqdm import tqdm

def on_progress_callback(stream, chunk, bytes_remaining):
    total_size = stream.filesize # total volume
    bytes_downloaded = total_size - bytes_remaining
    progress = bytes_downloaded / total_size * 100
    tqdm_bar.update(progress - tqdm_bar.n)

def tube_download(link, output_dir, file_name):
    # Create a YouTube object
    video = YouTube(link)

    # Get the highest resolution video stream
    stream = video.streams.get_highest_resolution()

    # Create a progress bar for the download
    global tqdm_bar
    tqdm_bar = tqdm(total=stream.filesize, unit="bytes", unit_scale=True)

    # Register the progress callback function
    video.register_on_progress_callback(on_progress_callback)

    # Download the video to the specified directory with the specified file name
    stream.download(output_path=output_dir, filename=file_name + '.' + stream.subtype)

    # Close the progress bar when the download is complete
    tqdm_bar.close()


# In[ ]:


for i in range():
    tube_download(df_sorted["url"][i], "games/"+df_sorted["title"][i], df_sorted["name"][i])
