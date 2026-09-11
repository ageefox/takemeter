from takemeter.data import LABELS, load_data
from takemeter.splits import make_thread_held_out_splits


def test_splits_are_disjoint_and_complete():
    frame = load_data("takemeter_dataset.csv")
    splits = make_thread_held_out_splits(frame)

    ids = [set(part["id"]) for part in splits.as_dict().values()]
    sources = [set(part["source_url"]) for part in splits.as_dict().values()]

    assert sum(map(len, ids)) == len(frame)
    assert set.union(*ids) == set(frame["id"])
    assert all(ids[i].isdisjoint(ids[j]) for i in range(3) for j in range(i + 1, 3))
    assert all(
        sources[i].isdisjoint(sources[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )


def test_each_split_contains_every_label():
    splits = make_thread_held_out_splits(load_data("takemeter_dataset.csv"))

    for frame in splits.as_dict().values():
        assert set(frame["label"]) == set(LABELS)


def test_split_assignment_is_stable_when_input_order_changes():
    frame = load_data("takemeter_dataset.csv")
    expected = make_thread_held_out_splits(frame)
    shuffled = frame.sample(frac=1, random_state=99).reset_index(drop=True)
    actual = make_thread_held_out_splits(shuffled)

    for name in expected.as_dict():
        assert list(expected.as_dict()[name]["id"]) == list(actual.as_dict()[name]["id"])


def test_current_split_sizes_and_thread_counts_are_recorded():
    splits = make_thread_held_out_splits(load_data("takemeter_dataset.csv"))

    assert {name: len(frame) for name, frame in splits.as_dict().items()} == {
        "train": 140,
        "validation": 35,
        "test": 37,
    }
    assert {
        name: frame["source_url"].nunique()
        for name, frame in splits.as_dict().items()
    } == {"train": 19, "validation": 5, "test": 4}
