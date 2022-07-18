docker build .  -t 2ol_bandit
docker run --rm -it -v $(pwd):/experiment 2ol_bandit /bin/bash && cd /experiment
