import os
from invoke import task

IMAGE_NAME = "course-mlops-api"
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


@task
def compose_up(c, build=False):
    c.run(f"docker compose up -d {'--build' if build else ''}")


@task
def deploy(c):
    status = c.run("minikube status", warn=True, hide=True)
    if status.failed or "Running" not in status.stdout:
        print("[+] Starting Minikube...")
        c.run("minikube start")
    else:
        print("[+] Minikube is already running.")

    print("[+] Pointing Docker to Minikube environment...")
    result = c.run("minikube docker-env", hide=True)
    env = os.environ.copy()
    for line in result.stdout.splitlines():
        if line.startswith("export"):
            key_value = line.replace("export ", "").split("=")
            if len(key_value) == 2:
                key, value = key_value
                env[key] = value.strip('"')

    print(f"[+] Building API image ({IMAGE_NAME}:latest) inside Minikube...")
    c.run(f"docker build -t {IMAGE_NAME}:latest .", env=env)

    print("[+] Deploying with Helm...")

    c.run("helm upgrade --install course-mlops charts/course-mlops --values charts/course-mlops/values.yaml")

    print("\n[+] Deployment complete!")
    print("    Access API via: minikube service course-mlops-api")
    print("    Access MLflow via: minikube service course-mlops-mlflow")

