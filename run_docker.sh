sudo docker build -t pytorch -f docker/pytorch-no-cuda/Dockerfile . && sudo docker run -it -p 6006:6006 -v ~/git-repos/mlinseconds-general-cpu:/mlinseconds-general-cpu -w /mlinseconds-general-cpu --name pytorch --rm pytorch
sudo docker run -it -p 6006:6006 -v ~/git-repos/mlinseconds-general-cpu:/mlinseconds-general-cpu -w /mlinseconds-general-cpu --name pytorch --rm pytorch
