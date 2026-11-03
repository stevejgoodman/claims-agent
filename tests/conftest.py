import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage  # noqa: F401

from agent.data import load_all, split_ab


class FakeToolModel(GenericFakeChatModel):
    """A chat model that replays the given messages and accepts bind_tools."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def scripted_model():
    def make(*messages: AIMessage) -> FakeToolModel:
        return FakeToolModel(messages=iter(messages))

    return make


@pytest.fixture(scope="session")
def dataset():
    return load_all()


@pytest.fixture(scope="session")
def sets(dataset):
    return split_ab(dataset)
