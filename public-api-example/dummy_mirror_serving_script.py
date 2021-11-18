import re
import os
import kfserving
from typing import Dict

class DummyMirrorService(kfserving.KFModel):
    
    def __init__(self, service_name: str):
        super(DummyMirrorService, self).__init__(service_name)

        self.service_name = service_name
        self.ready = False

    def load(self) -> None:
        pass

    def predict(self, request: Dict) -> Dict:
        response = request
        
        response["echo"] = "Hello from {}".format(self.service_name)
        
        return response

if __name__ == "__main__":
    service_name = "kf-serving-default"

    host_name = os.environ.get('HOSTNAME')

    if host_name is not None:
        x = re.compile(r'(kfserving-\d+)').search(host_name)

        if x is not None:
            service_name = x[0]

    dummy_mirror_service = DummyMirrorService(service_name)

    dummy_mirror_service.load()

    kfserving.KFServer(workers=1).start([dummy_mirror_service])
