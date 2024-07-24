import os


class ConfigManager:
    def __init__(self):
        self.config = {
            "rabbitmq_host": os.environ.get("RABBITMQ_HOST", "localhost"),
            "rabbitmq_port": int(os.environ.get("RABBITMQ_PORT", 5672)),
            "consul_host": os.environ.get("CONSUL_HOST", "localhost"),
            "consul_port": int(os.environ.get("CONSUL_PORT", 8500)),
            "api_host": os.environ.get("API_HOST", "0.0.0.0"),
            "api_port": int(os.environ.get("API_PORT", 8000)),
        }

    def get(self, key, default=None):
        return self.config.get(key, default)
