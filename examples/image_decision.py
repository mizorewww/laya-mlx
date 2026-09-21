"""Run with: python examples/image_decision.py [IMAGE]"""

import json
import sys

import laya_mlx as laya

if len(sys.argv) > 1:
    image = sys.argv[1]
else:
    from huggingface_hub import hf_hub_download

    image = hf_hub_download(
        "VinayHajare/Fruits-30", "FruitImageDataset/apples/0.jpg", repo_type="dataset"
    )

questions = {
    "kind": {
        "type": "choice",
        "instructions": "What type of object is this?",
        "criteria": ["toy", "fruit", "car"],
    },
    "edible": {"type": "noul", "instructions": "Is the subject of the image edible?"},
}

state = laya.image_state(image, questions)
print(state)

agent = laya.load("aac6fef/laya-mlx")
print(json.dumps(agent.predict(state, questions)["answers"], indent=2))
