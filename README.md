<p align="middle">
  <img height="40mm" src="https://dataloop.ai/wp-content/uploads/2020/03/logo.svg">
</p>

## Pipeline library application for Google Vision Functions

---

<p align="left">
  
  <img src="assets/pipeline_node.png">

![versions](https://img.shields.io/pypi/pyversions/dtlpy.svg)

</p>

---

## Description

Application for the Dataloop custom Pipeline nodes feature.

The Application allows access to more pipeline nodes that can be used in the pipeline.

The application gives a variety of nodes created from OpenCV Functions, where every function is separate node. List of the nodes:

- [clahe](clahe/clahe.py)
- [crop](crop/crop.py)
- [face bluring](face_blur/face_blur.py)

## Installations

- Clone the repository

- Publishing the app:

`dlp app publish --project-name "<PROJECT_NAME>"`

- To install for a project:

`dlp app install --dpk-id "<DPK ID>" --project-name "<PROJECT_NAME>"`
