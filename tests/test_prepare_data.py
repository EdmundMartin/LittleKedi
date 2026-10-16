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
