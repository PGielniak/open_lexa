import base64
import getpass
import json
import subprocess

username = input("GitHub username: ").strip()
token = getpass.getpass("GitHub PAT (classic): ").strip()

if not username or not token:
    raise SystemExit("Brak użytkownika lub tokenu — sekret nie został zmieniony.")

auth = base64.b64encode(f"{username}:{token}".encode()).decode()
docker_config = {"auths": {"ghcr.io": {"auth": auth}}}

secret = {
    "apiVersion": "v1",
    "kind": "Secret",
    "metadata": {
        "name": "tekton-kaniko-push",
        "namespace": "tekton-builds",
    },
    "type": "kubernetes.io/dockerconfigjson",
    "data": {
        ".dockerconfigjson": base64.b64encode(
            json.dumps(docker_config).encode()
        ).decode()
    },
}

subprocess.run(
    ["kubectl", "--context", "default", "apply", "-f", "-"],
    input=json.dumps(secret),
    text=True,
    check=True,
)
