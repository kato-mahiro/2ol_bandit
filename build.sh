docker build . -t neat
docker run --rm -it -v $(pwd):/experiment neat /bin/bash