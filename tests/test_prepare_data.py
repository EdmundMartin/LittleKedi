import numpy as np

from prepare_data import clean_wikipedia

PROSE = "Aarau is a town on the Swiss plateau, in the valley of the Aare. " * 10


def test_clean_wikipedia_strips_leftovers_and_trailing_sections():
    text = (f"Aarau (, ) is a town in Aargau ( ; ; ), west of Zürich,  south of Basel.\n\n\n\nHistory\n\n {PROSE}\n\n"
            "See also\n Lenzburg\n\nReferences\n\nExternal links\n\n Cities in Switzerland\nMunicipalities of Aargau")
    out = clean_wikipedia("Aarau", text)
    assert out.startswith("Aarau is a town in Aargau, west of Zürich, south of Basel.\n\nHistory\n\nAarau is")
    assert "See also" not in out and "Lenzburg" not in out and "Cities in Switzerland" not in out
    assert "\n\n\n" not in out and "  " not in out


def test_clean_wikipedia_drops_flattened_tables_but_keeps_headings():
    table = "\n".join(["Results", "", "Season", "Gala", "Kelso", "", "R1 = Round 1", "QF = Quarter-finals"])
    out = clean_wikipedia("Club", f"History\n\n{PROSE}\n\n{table}\n\nGrounds\n\n{PROSE}")
    assert out == f"History\n\n{PROSE.strip()}\n\nGrounds\n\n{PROSE.strip()}"


def test_clean_wikipedia_drops_lists_disambiguation_and_stubs():
    assert clean_wikipedia("Aba", "Aba may refer to:\n\nAnimals\n Aba roundleaf bat\n" + PROSE) is None
    assert clean_wikipedia("Mercury (disambiguation)", PROSE) is None
    assert clean_wikipedia("List of Swiss cantons", PROSE) is None
    assert clean_wikipedia("Stub", "A short article.\n\nReferences\n" + PROSE) is None
    assert clean_wikipedia("Aarau", PROSE) == PROSE.strip()


def test_reddit_comment_filters():
    from prepare_data import reddit_comment
    long = "I think the main reason is that " * 10
    assert reddit_comment(long, "12", "someone") == long.strip()
    assert reddit_comment(long, "2", "someone") is None          # not upvoted enough
    assert reddit_comment(long, "50", "AutoModerator") is None    # bot
    assert reddit_comment("[deleted]", "50", "someone") is None
    assert reddit_comment("Too short.", "50", "someone") is None
    assert reddit_comment(long, "n/a", "someone") is None
    assert reddit_comment("Fish &amp; chips &gt; " + long, "9", "x").startswith("Fish & chips > ")


def test_interleave_round_robins_and_drops_finished_streams():
    from prepare_data import interleave
    a, b, c = iter(["a1", "a2", "a3"]), iter(["b1"]), iter(["c1", "c2"])
    assert list(interleave([a, b, c], batch=1)) == ["a1", "b1", "c1", "a2", "c2", "a3"]
    assert list(interleave([iter(range(5)), iter("xy")], batch=2)) == [0, 1, "x", "y", 2, 3, 4]


def test_train_writer_parts_read_back_as_one_array(tmp_path):
    from littlekedi.data import load_train_tokens
    from prepare_data import TrainWriter

    tokens = np.arange(25, dtype=np.uint16)
    with TrainWriter(tmp_path, part_tokens=10) as f:
        for chunk in (tokens[:7], tokens[7:19], tokens[19:]):
            f.write(chunk)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train-00000.bin", "train-00001.bin", "train-00002.bin"]
    data = load_train_tokens(tmp_path)
    assert len(data) == 25
    assert data[8:13].tolist() == list(range(8, 13))  # straddles a part boundary
    assert data[3:5].tolist() == [3, 4]
    assert data[18:25].tolist() == list(range(18, 25))
    head = data[:12]
    assert len(head) == 12 and head[9:12].tolist() == [9, 10, 11]


def test_train_writer_single_file(tmp_path):
    from littlekedi.data import load_train_tokens
    from prepare_data import TrainWriter

    with TrainWriter(tmp_path, part_tokens=None) as f:
        f.write(np.arange(5, dtype=np.uint16))
    assert [p.name for p in tmp_path.iterdir()] == ["train.bin"]
    assert load_train_tokens(tmp_path)[:5].tolist() == list(range(5))


def test_train_writer_resume_drops_what_came_after_the_saved_position(tmp_path):
    from littlekedi.data import load_train_tokens
    from prepare_data import TrainWriter

    tokens = np.arange(40, dtype=np.uint16)
    with TrainWriter(tmp_path, part_tokens=10) as f:
        f.write(tokens[:13])
        saved = f.position()
        f.write(np.full(15, 999, dtype=np.uint16))  # lost in the crash: runs on into a later part
    with TrainWriter(tmp_path, part_tokens=10, resume_at=saved) as f:
        f.write(tokens[13:])
    data = load_train_tokens(tmp_path)
    assert len(data) == 40 and data[0:40].tolist() == tokens.tolist()


def test_train_writer_resume_single_file(tmp_path):
    from littlekedi.data import load_train_tokens
    from prepare_data import TrainWriter

    with TrainWriter(tmp_path, part_tokens=None) as f:
        f.write(np.arange(5, dtype=np.uint16))
        saved = f.position()
        f.write(np.full(3, 999, dtype=np.uint16))
    with TrainWriter(tmp_path, part_tokens=None, resume_at=saved) as f:
        f.write(np.arange(5, 8, dtype=np.uint16))
    assert load_train_tokens(tmp_path)[:8].tolist() == list(range(8))
