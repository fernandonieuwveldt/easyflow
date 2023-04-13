import os
import subprocess


class ModelServer:
    """
    A class to handle the deployment and management of TensorFlow Keras models using TensorFlow Serving and Docker.
    """

    def __init__(self, model_name, model_path, rest_api_port=8501):
        """
        Initialize the ModelServer instance.

        Args:
            model_name (str): The name of the model to be served.
            model_path (str): The path to the directory containing the saved model.
            rest_api_port (int, optional): The port to expose the REST API on the host system. Defaults to 8501.
        """
        self.model_name = model_name
        self.model_path = model_path
        self.rest_api_port = rest_api_port
        self.docker_image = 'tensorflow/serving'
        self.container_name = f"{model_name}_serving"

    def pull_docker_image(self):
        """
        Check if the TensorFlow Serving Docker image is available locally; if not, pull it from the Docker registry.
        """
        print(f"Checking if {self.docker_image} is available locally...")
        result = subprocess.run(f"docker images -q {self.docker_image}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        if not result.stdout.strip():
            print(f"Pulling {self.docker_image} from Docker registry...")
            subprocess.run(f"docker pull {self.docker_image}", shell=True)
        else:
            print(f"{self.docker_image} is available locally.")

    def start(self):
        """
        Start the TensorFlow Serving container to serve the model.
        """
        self.pull_docker_image()
        print(f"Starting {self.container_name} container...")
        command = f"docker run -d -p {self.rest_api_port}:8501 \
            --name {self.container_name} \
            --mount type=bind,source={os.path.abspath(self.model_path)},target=/models/{self.model_name} \
            -e MODEL_NAME={self.model_name} -t {self.docker_image}"
        subprocess.run(command, shell=True)

    def stop(self):
        """
        Stop the TensorFlow Serving container.
        """
        print(f"Stopping {self.container_name} container...")
        command = f"docker stop {self.container_name}"
        subprocess.run(command, shell=True)
        command = f"docker rm {self.container_name}"
        subprocess.run(command, shell=True)

    def restart(self):
        """
        Restart the TensorFlow Serving container.
        """
        self.stop()
        self.start()
