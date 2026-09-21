import numpy as np
import pytest

from laya_mlx.vision import CLIP, describe_image, image_state, question_labels, sanitize_weights


class FakeScorer:
    """Ranks labels by their position in a fixed preference list."""

    def __init__(self, preference):
        self.preference = list(preference)

    def embed_image(self, image):
        return image

    def rank(self, embedding, labels, prompt=None):
        labels = list(labels)
        order = sorted(labels, key=lambda label: self.preference.index(label))
        weights = np.exp(-np.arange(len(order), dtype=np.float64))
        return list(zip(order, (weights / weights.sum()).tolist()))


def test_question_labels_selects_rankable_options():
    assert question_labels({"type": "choice", "criteria": ["toy", "fruit"]}) == ["toy", "fruit"]
    assert question_labels({"type": "choice", "criteria": {"toy": "a plaything"}}) == ["toy"]
    assert question_labels({"type": "score", "criteria": ["dark", "bright"]}) == ["dark", "bright"]
    assert question_labels({"type": "noul"}) is None
    assert question_labels({"type": "choice", "criteria": []}) is None
    assert question_labels("not a question") is None


def test_describe_image_ranks_scene_and_each_choice_question():
    scorer = FakeScorer(["fruit", "food", "toy", "car"])
    text = describe_image(
        "image",
        {
            "kind": {
                "type": "choice",
                "instructions": "What type of object is this?",
                "criteria": ["toy", "fruit", "car"],
            },
            "edible": {"type": "noul", "instructions": "Is it edible?"},
        },
        scorer=scorer,
        labels=["toy", "car", "fruit", "food"],
        top_k=2,
    )
    assert "The image shows: fruit (64%), food (24%)." in text
    assert "Visual match for 'What type of object is this?': fruit (67%)" in text
    assert "Is it edible?" not in text


def test_image_state_puts_evidence_before_the_caller_state():
    scorer = FakeScorer(["fruit"])
    evidence = describe_image("image", scorer=scorer, labels=["fruit"])
    assert image_state("image", scorer=scorer, labels=["fruit"]) == evidence
    assert image_state("image", None, "ticket text", scorer=scorer, labels=["fruit"]).startswith(
        evidence
    )
    combined = image_state("image", None, {"id": 7}, scorer=scorer, labels=["fruit"])
    assert combined == {"image": evidence, "state": {"id": 7}}


def test_agent_predict_routes_images_through_clip(tiny_checkpoint, questions, monkeypatch):
    from laya_mlx import agent as agent_module
    from laya_mlx import vision

    seen = {}

    def fake_describe(image, qs=None, **kwargs):
        seen["image"], seen["questions"] = image, qs
        return "hello hello"

    monkeypatch.setattr(vision, "describe_image", fake_describe)
    result = agent_module.Agent(tiny_checkpoint).predict(None, questions, image="apple.jpg")
    assert seen == {"image": "apple.jpg", "questions": questions}
    assert set(result["answers"]) == set(questions)


def test_sanitize_weights_drops_buffers_and_transposes_patch_conv():
    import mlx.core as mx

    out = sanitize_weights(
        {
            "vision_model.embeddings.position_ids": mx.zeros((1, 5)),
            "vision_model.embeddings.patch_embedding.weight": mx.zeros((8, 3, 2, 2)),
        }
    )
    assert list(out) == ["vision_model.embeddings.patch_embedding.weight"]
    assert out["vision_model.embeddings.patch_embedding.weight"].shape == (8, 2, 2, 3)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def test_clip_matches_transformers():
    import mlx.core as mx

    torch.manual_seed(0)
    config = transformers.CLIPConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 64,
            "max_position_embeddings": 16,
            "eos_token_id": 2,
        },
        vision_config={
            "hidden_size": 24,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 8,
        },
        projection_dim=16,
    )
    reference = transformers.CLIPModel(config).eval()
    weights = sanitize_weights(
        {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
    )
    model = CLIP(config.to_dict())
    model.load_weights(list(weights.items()), strict=True)
    model.eval()

    pixels = np.random.default_rng(0).normal(size=(2, 16, 16, 3)).astype(np.float32)
    ids = np.array([[49, 3, 7, 63, 0, 0], [49, 11, 63, 0, 0, 0]], dtype=np.int32)

    image = np.asarray(model.visual_projection(model.vision_model(mx.array(pixels))))
    text = np.asarray(model.text_projection(model.text_model(mx.array(ids))))
    with torch.no_grad():
        vision = reference.vision_model(torch.from_numpy(pixels.transpose(0, 3, 1, 2)))
        want_image = reference.visual_projection(vision.pooler_output).numpy()
        want_text = reference.text_projection(
            reference.text_model(input_ids=torch.from_numpy(ids.astype(np.int64))).pooler_output
        ).numpy()

    assert np.allclose(image, want_image, atol=2e-4)
    assert np.allclose(text, want_text, atol=2e-4)
