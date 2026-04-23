"""Protocol shape test: anything with the required class attrs and method is an adapter."""
from collections.abc import Iterator

from calibration.datasets.base import Conversation, DatasetAdapter, Message


def test_conversation_defaults_tools_to_none():
    conv = Conversation(messages=[{"role": "user", "content": "hi"}])
    assert conv.tools is None


def test_conversation_accepts_tools_list():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    conv = Conversation(messages=[{"role": "user", "content": "hi"}], tools=tools)
    assert conv.tools == tools


def test_message_typed_dict_allows_all_roles():
    roles: list[Message] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "t"},
    ]
    assert len(roles) == 4


def test_stub_class_satisfies_protocol_at_runtime():
    class Stub:
        name = "stub"
        dataset_id = "fake/fake"
        default_split = "train"

        def iter_conversations(self, *, split: str) -> Iterator[Conversation]:
            yield Conversation(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(Stub(), DatasetAdapter)
