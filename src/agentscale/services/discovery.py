import json

import consul

from agentscale.core.config import ConfigManager


class ConsulClient:
    def __init__(self, host=None, port=None):
        config = ConfigManager()
        self.consul = consul.Consul(
            host=host or config.get("consul_host"),
            port=port or config.get("consul_port"),
        )

    def register_service(
        self,
        service_name: str,
        service_id: str,
        address: str,
        port: int,
        capabilities: list,
    ):
        self.consul.agent.service.register(
            service_name,
            service_id=service_id,
            address=address,
            port=port,
            meta={"capabilities": json.dumps(capabilities)},
        )

    def get_all_services(self) -> list:
        index, services = self.consul.catalog.services()
        return [service for service in services if service != "consul"]

    def get_service_details(self, service_name: str) -> dict:
        index, service = self.consul.health.service(service_name, passing=True)
        if service:
            return {
                "id": service[0]["Service"]["ID"],
                "address": service[0]["Service"]["Address"],
                "port": service[0]["Service"]["Port"],
                "capabilities": json.loads(
                    service[0]["Service"]["Meta"]["capabilities"]
                ),
            }
        return None
