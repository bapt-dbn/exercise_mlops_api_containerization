from invoke import task

IMAGE_NAME = "course-mlops"
CONTAINER_NAME = "course-mlops-container"
PORT = "8000"


@task
def train(c, config="config/config.yaml"):
    print(f"[+] Using configuration file: {config}")
    c.run(f"python -m src -c {config}", pty=True)


@task
def run(c):
    print(f"[+] Building Docker image: {IMAGE_NAME}")
    c.run(f"docker build -t {IMAGE_NAME}:latest .")

    c.run(f"docker rm -f {CONTAINER_NAME}", warn=True, hide=True)

    c.run(
        f"docker run -p {PORT}:8000 -v ./mlruns:/app/mlruns --name {CONTAINER_NAME} {IMAGE_NAME}:latest",
        pty=True,
    )
