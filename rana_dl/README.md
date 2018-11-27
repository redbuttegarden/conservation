# Rana Post-Processing Using Deep Learning

The goal of this project is handle the post-processing of Rana videos so as to ascertain the species
composition and frequency of pollinators visiting the plant population in the videos.

## Usage

```
ls -la /dev | grep nvidia
```

```
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    --device /dev/nvidia0:/dev/nvidia0 \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /media/myhostuser/Rana_vids/Videos:/code/media/videos \
    -v /media/myhostuser/Rana_vids/possible_pollinator_photos:/code/media/pollinator_photos \
    rana
```