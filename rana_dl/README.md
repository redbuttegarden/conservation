# Rana Post-Processing Using Deep Learning

The goal of this project is handle the post-processing of Rana videos so as to ascertain the species composition and 
frequency of pollinators visiting the plant population in the videos.

## Usage

To create new models, you'll need to have a [CUDA compatible GPU](https://developer.nvidia.com/cuda-gpus).

For your convenience, a docker container containing everything needed to run all the code has already been created for 
you. You can run it using either Docker or Singularity:

### With Docker

Running the container with docker will look something like this:

```bash
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/nvidia0:/dev/nvidia0 \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
    -v /media/myhostuser/Rana/Videos:/code/media/videos \
    -v /media/myhostuser/Rana/possible_pollinator_photos:/code/media/pollinator_photos \
    -u docker
    auslaner/rana_process:chpc
```

This pulls and runs the container located on Docker Hub at auslaner/rana_process:chpc.

Let's break down the above command a bit. The `-it` combination of flags allows us to open an interactive shell into the 
container. Note that our command doesn't specify a program to run so it defaults to whatever `CMD` was defined in the 
[Dockerfile](./Dockerfile) which in this case is `/bin/bash`.

The optional `--rm` flag instructs our container to automatically clean itself up after exiting. Note that this means 
all changes made inside the container will be lost. To preserve data, volume mounts are used, as explained below.

The `-e` flag sets the `DISPLAY` environment variable to match that of our host. In conjunction with the first `-v` 
flag, we are granted X11 forwarding from container to host which allows the graphical elements of the programs running 
in our container to be visible on our host machine. This flag can be omitted when your aim is merely to interact with an 
existing dataset (e.g. build a model) and you don't need to see any part of the GUI.

The `--device` flags are used to ensure your GPU is available to the container. You may need to adjust these flags based
on your own configuration. You can check the correct paths to use using the following command:

```bash
ls -la /dev | grep nvidia
```

The two last `-v` flags are used to mount data volumes from the host onto the container, allowing data to persist
beyond the lifecycle of the container. Use these flags to specify where on the host machine your videos and photos are
located, following the notation `-v HOST_PATH:CONTAINER_PATH`. The host path will vary based on where you have your data
saved, but the container path must use the paths as shown in the above example.

The `-u docker` flag is used to instruct the container to run as the `docker` user and is necessary for successful X11 
forwarding. It's also ideal from a security standpoint, so when in doubt, you should use the `-u docker` flag. If any
additional configuration is needed inside the container, it will be necessary to omit this flag so you are able to run
the container as the `root` user. When doing so, it would probably make sense to also omit the `--rm` flag so the
container is not removed after exiting and any changes you have made are saved.

To access a container you've made changes to and then exited (one you ran without the `--rm` flag), first find the name 
of the stopped container using the following command:

```bash
docker ps -a
```

Then use the `docker attach` command to open a shell into it:

```
docker attach CONTAINER_NAME
```

More information about running containers can be found in the `docker run` 
[documentation](https://docs.docker.com/engine/reference/run/).

### With Singularity

Running the container with singularity will look something like this:

```bash
singularity run \
    --nv \
    -B ./data/Videos:/code/media/videos \
    -B ./data/photos:/code/media/pollinator_photos \
    docker://auslaner/rana_process:chpc
```

The `--nv` flag allows us to pass our Nvidia GPU thru to the container.

The `-B` flags function similarly to the `-v` flags of Docker, allowing us to mount folders from our filesystem into the
containers.

The above `singularity run` command creates an ephemeral container that will be lost when exited. If you would like the 
container to persist, pull it from the docker repository first:

```bash
singularity pull --name rana.img docker://auslaner/rana_process:chpc
```

Then, as above, run the pulled image by name:

```bash
singularity run \
    --nv \
    -B ./data/Videos:/code/media/videos \
    -B ./data/photos:/code/media/pollinator_photos \
    rana.img
```
