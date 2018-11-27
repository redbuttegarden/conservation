# Conservation
###### Repo for [Red Butte Garden's](https://www.redbuttegarden.org/) Conservation Department

Projects
---
### Rana Video Processing

The `rana_dl` folder contains code used to help process video of rare plant populations. Input videos in 
this project made use of [Rana technology](https://www.tumblingdice.co.uk/rana/) to condense a continuously
monitoring video stream into just the parts that contain pollinators. 

The goal of this project is handle the post-processing of Rana videos so as to ascertain the species
composition and frequency of pollinators visiting the plant population in the videos. This process can
roughly be broken down into two steps:

1. Pollinator Extraction
    11. Using a moving average of previous frames, the static areas of each frame are subtracted, leaving
    just the areas that have changed. We assume that this process captures pollinators since they tend
    to move around a lot.
    
2. Pollinator Classification
    22. The above pollinator extraction method is prone to false-positives. To mitigate this issue, we
    pass all possible pollinator detections through a deep learning algorithm that we've trained on
    thousands of images of known pollinators.
    
For more information and a guide for how to work with this code, please see the [README contained in the 
`rana_dl` folder](rana_dl/README.md).

